import os
import gc

import numpy as np
from numpy import genfromtxt

# import numpy
# import jax
# from jax.config import config
# import jax.numpy as np
# config.update('jax_enable_x64', True)
# import jax.dlpack as dlpack

import pandas as pd

import tqdm
import time
import json

import cv2
# from cv2 import cuda # TODO: Compile CV2 with cuda

from skimage import feature

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.io import read_image
from torch.profiler import profile, record_function, ProfilerActivity

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg') # Solves memory leak
# matplotlib.use('GTKAgg')

# import seaborn
# import seaborn_image as isns

from helper_functions import log
from helper_functions import convert_to_tensor
import helper_functions
import data_helpers
import image_functions
import torch_models
# from canny import CannyFilter

# https://github.com/jacobgil/pytorch-grad-cam
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# from numba import jit

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
# from collections import Counter
# import linecache
# import tracemalloc

# def display_top(snapshot, key_type='lineno', limit=3):
#   snapshot = snapshot.filter_traces((
#     tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#     tracemalloc.Filter(False, "<unknown>"),
#   ))
#   top_stats = snapshot.statistics(key_type)

#   print("Top %s lines" % limit)
#   for index, stat in enumerate(top_stats[:limit], 1):
#     frame = stat.traceback[0]
#     # replace "/path/to/module/file.py" with "module/file.py"
#     filename = os.sep.join(frame.filename.split(os.sep)[-2:])
#     print("#%s: %s:%s: %.1f KiB"
#         % (index, filename, frame.lineno, stat.size / 1024))
#     line = linecache.getline(frame.filename, frame.lineno).strip()
#     if line:
#       print('    %s' % line)

#   other = top_stats[limit:]
#   if other:
#     size = sum(stat.size for stat in other)
#     print("%s other: %.1f KiB" % (len(other), size / 1024))
#   total = sum(stat.size for stat in top_stats)
#   print("Total allocated size: %.1f KiB" % (total / 1024))

def get_cuda_objects():
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj) and obj.is_cuda:
        tensor = obj
        # print(f'{type(obj).__name__} {obj.size()} {obj.device} {obj.data_ptr()}')
        print(f'{type(tensor).__name__} {tensor.size()} {tensor.device} {tensor.data_ptr()} {tensor.element_size() * tensor.nelement() / (1024 ** 2)} MB, Variable name: {tensor.name}')
        # print(f'Variable name: {tensor.name}')
    except:
      pass

class CustomGradCAM(BaseCAM):
  def __init__(self, model, target_layers, use_cuda=False, compute_device=None, reshape_transform=None):
    super(CustomGradCAM, self).__init__(model, target_layers, use_cuda, compute_device, reshape_transform)

  def get_cam_weights(self, input_tensor, target_layer, target_category, activations, grads):
    return torch.mean(grads, axis=(2, 3))

def run_kl_grad_cam(config, model):
  log('Compute Grad-CAM for KL-Divergence', config)
  dev = helper_functions.fetch_device(config)
  pair_paths, meta = data_helpers.load_segmented_data(config)

  raw_frames = None
  segmentation_maps = None
  edge_maps = None

  if config['preload_segmentation_maps'] and config["cook_data"]:
    segmentation_maps, total_segmentation_summary = preload_segmentation_maps(pair_paths, config)
    log('Loaded Segmentation Maps', config)

    raw_frames = preload_input_frames(pair_paths, segmentation_maps, config)
    log('Loaded input frames', config)

    edge_maps = preload_edge_maps(pair_paths, config)
    log('Loaded Edge Maps', config)
    log('Loaded all data into memory.', config)

  # first and last model
  grad_type = 'Base' #'Custom' # 'Base'
  batch_size, config = compute_grad_cam_batch_size(config)

  # compute_grad_cam_results(model, config, in_line=True, epoch_count=epoch, grad_type='Base')
  # compute_grad_cam_results(model, config, dev, pair_paths, meta, raw_frames, segmentation_maps, edge_maps, total_segmentation_summary, in_line=True, epoch_count=epoch, grad_type='Custom')

  remainder = len(pair_paths) % batch_size
  total_count = len(pair_paths) - remainder

  # First Model
  model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=0)
  first_model_gradcam_plots = run_model_gradcam_with_kl(config, dev, model, batch_size, total_count, pair_paths, raw_frames, segmentation_maps, grad_type)
  first_model_gradcam_plots = first_model_gradcam_plots.cpu().detach()

  # Clean-up
  helper_functions.clear_gpu(torch, config, pause=False)

  # Last Model
  model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=config["epochs"])
  last_model_gradcam_plots = run_model_gradcam_with_kl(config, dev, model, batch_size, total_count, pair_paths, raw_frames, segmentation_maps, grad_type)
  last_model_gradcam_plots = last_model_gradcam_plots.cpu().detach()

  kl_divergence = F.kl_div(first_model_gradcam_plots, last_model_gradcam_plots, reduction='batchmean')
  helper_functions.log(f"KL_Divergence ({grad_type} type): {kl_divergence}", config)

  helper_functions.clear_gpu(torch, config, pause=False)

def run_model_gradcam_with_kl(config, dev, model, batch_size, total_count, pair_paths, raw_frames, segmentation_maps, grad_type):
  transposed_dims = list(np.array(config["input_size"]).T)
  heatmaps = torch.zeros((len(pair_paths), *transposed_dims))

  cam_function = fetch_cam_function(config, model, grad_type)
  pbar = tqdm.tqdm(total=total_count)

  dtype = helper_functions.compute_gradcam_dtype(config)

  # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
  # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
  # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
  for i in range(0, total_count, batch_size):
    # i = batch_size
    segmentation_frames = segmentation_maps[i:i+batch_size]
    frames = load_input_image(config, dtype, dev, i, batch_size, total_count, pair_paths, raw_frames, segmentation_frames)
    grad_cam_plots = compute_heat_map(config, model, cam_function, frames, batch_size) # Faster compute

    for j in range(len(grad_cam_plots)):
      heatmaps[i+j] = grad_cam_plots[j]

    pbar.update(batch_size)
    del frames

  # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
  # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
  # prof.export_chrome_trace("trace3.json")

  return heatmaps


def run_grad_cam(config, model=None, model_count=100):
  # Grad-CAM
  log('==========================================================================================', config)
  log('Run Grad-CAM ...', config)
  # https://medium.com/the-owl/gradcam-in-pytorch-7b700caa79e5
  # https://github.com/jacobgil/pytorch-grad-cam ... pip install grad-cam
  # https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

  if model == None and not config["compute_grad_cam_results"]:
    compute_grad_cam_for_random_models(config, model_count)

  if config["compute_grad_cam_results"]:
    compute_grad_cam_for_model_epochs(model, config) # Each saved epoch model

  if config["save_grad_cam"]:
    log('GradCAM Example ...', config)

    # TODO: GradCAM Other Methods
    xt[0] = test_ds[50][0].clone().detach()
    input_tensor = xt # Note: input_tensor can be a batch tensor with several images!

    create_gradcam(model, input_tensor, config, name='single_test')

    # TODO: Segmentation map on video?? (Need to generate segments for Udacity)
    generate_grad_cam_video(model, config)

  # log(f'Test time: {time.time() - start_test_time} secs.', config)

def compute_gradcam_epochs(config):
  if ("epoch_start" in config) and ("epoch_end" in config):
    range_of_epochs = [*range(config["epoch_start"], config["epoch_end"])]
    return np.unique(np.sort(np.array(range_of_epochs)))
  elif config["grad_cam_initial_epochs_to_save"] > 0:
    range_of_epochs = [*range(config["grad_cam_initial_epochs_to_save"])]
    return np.unique(np.sort(np.array(range_of_epochs + config["grad_cam_epochs"])))
  else:
    return config['grad_cam_epochs']

def compute_grad_cam_for_random_models(config, model_count):
  log(f'Compute Random Grad-CAM for number: {model_count}', config)

  dev = helper_functions.fetch_device(config)
  pair_paths, meta = data_helpers.load_segmented_data(config)

  # Cant set both at the same time
  if "grad_cam_coarse" in config:
    if config["grad_cam_coarse"]:
      pair_paths = filter_coarse_only(pair_paths, config)

  if "grad_cam_fine" in config:
    if config["grad_cam_fine"]:
      pair_paths = filter_fine_only(pair_paths, config)

  raw_frames = None
  segmentation_maps = None
  edge_maps = None

  if config['preload_segmentation_maps'] and config["cook_data"]:
    segmentation_maps, total_segmentation_summary = preload_segmentation_maps(pair_paths, config)
    log('Loaded Segmentation Maps', config)
    raw_frames = preload_input_frames(pair_paths, segmentation_maps, config)
    log('Loaded input frames', config)

    edge_maps = preload_edge_maps(pair_paths, config)
    log('Loaded Edge Maps', config)
    log('Loaded all data into memory.', config)

  for i in tqdm.tqdm(range(model_count)):
    # Compile model
    model = torch_models.compile_model(config)
    dtype = helper_functions.compute_model_dtype(config) # TODO: Other dtype rather?
    model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

    compute_grad_cam_results(model, config, dev, pair_paths, meta, raw_frames, segmentation_maps, edge_maps, total_segmentation_summary, in_line=True, epoch_count=i, grad_type='Base') # 'Custom'

    helper_functions.clear_gpu(torch, config, pause=False)

def filter_coarse_only(pair_paths, config):
  log("Filter Coarse Only", config)
  filtered_pair_paths = []

  for pair in tqdm.tqdm(pair_paths):
    if "Coarse" in pair[1]:
      filtered_pair_paths.append(pair)

  log(f"Count After: {len(filtered_pair_paths)}", config)
  return filtered_pair_paths

def filter_fine_only(pair_paths, config):
  log("Filter Fine Only", config)
  filtered_pair_paths = []

  for pair in tqdm.tqdm(pair_paths):
    if "Fine" in pair[1]:
      filtered_pair_paths.append(pair)

  log(f"Count After: {len(filtered_pair_paths)}", config)
  return filtered_pair_paths

def compute_grad_cam_for_model_epochs(model, config):
  if config["grad_cam_epochs"] == None or not config["compute_grad_cam_results"]:
    return None

  log('Compute Grad-CAM for epochs', config)
  dev = helper_functions.fetch_device(config)
  pair_paths, meta = data_helpers.load_segmented_data(config)

  raw_frames = None
  segmentation_maps = None
  edge_maps = None

  helper_functions.clear_gpu(torch, config, pause=False)

  log(f"Pair Paths Count: {len(pair_paths)}", config)

  # Cant set both at the same time
  if "grad_cam_coarse" in config:
    if config["grad_cam_coarse"]:
      pair_paths = filter_coarse_only(pair_paths, config)

  if "grad_cam_fine" in config:
    if config["grad_cam_fine"]:
      pair_paths = filter_fine_only(pair_paths, config)

  if config['preload_segmentation_maps'] and config["cook_data"]:
    segmentation_maps, total_segmentation_summary = preload_segmentation_maps(pair_paths, config)
    log('Loaded Segmentation Maps', config)

    raw_frames = preload_input_frames(pair_paths, segmentation_maps, config)
    log('Loaded input frames', config)

    edge_maps = preload_edge_maps(pair_paths, config)
    log('Loaded Edge Maps', config)
    log('Loaded all data into memory.', config)

  epochs = compute_gradcam_epochs(config)

  # if "best_model_epoch" in config:
  #   if not (config["best_model_epoch"] in epochs):
  # if "best_val_autonomy_epoch" in config:
    # epochs.append(config["best_val_autonomy_epoch"])

  epochs.append(config["best_model_epoch"])
  epochs.append(config["best_val_autonomy_epoch"])

  grad_cam_only_best_epoch = False
  if "grad_cam_only_best_epoch" in config:
    grad_cam_only_best_epoch = True

    if config["grad_cam_only_best_epoch"]:
      if "best_val_autonomy_epoch" in config:
        epochs = [config["best_val_autonomy_epoch"]]
      else:
        epochs = [config["best_model_epoch"]]

  if "best_model_epoch" in config:
    grad_cam_only_best_epoch = True

  best_val_autonomy_epoch = False
  if "best_val_autonomy_epoch" in config:
    best_val_autonomy_epoch = True

  if "grad_cam_only_first_epoch" in config:
    if config["grad_cam_only_first_epoch"]:
      epochs = [1]

  for epoch in tqdm.tqdm(epochs):
    model = None

    if (epoch <= config["best_model_epoch"]) or (epoch <= config["best_val_autonomy_epoch"]):
      # if best_val_autonomy_epoch:
      if epoch == config["best_val_autonomy_epoch"]:
        model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=epoch, append_path="best_val_autonomy_model")

      # if grad_cam_only_best_epoch:
      if epoch == config["best_model_epoch"]:
        model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=epoch, append_path="best_model")

      if model == None: # Normal Epochs
        model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=epoch)

      dev = helper_functions.fetch_device(config)
      dtype = helper_functions.compute_gradcam_dtype(config)
      model.to(dev, non_blocking=config["non_blocking"], dtype=dtype)

      # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
      compute_grad_cam_results(model, config, dev, pair_paths, meta, raw_frames, segmentation_maps, edge_maps, total_segmentation_summary, in_line=True, epoch_count=epoch, grad_type='Base')
      # compute_grad_cam_results(model, config, dev, pair_paths, meta, raw_frames, segmentation_maps, edge_maps, total_segmentation_summary, in_line=True, epoch_count=epoch, grad_type='Custom')

      # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))
      # prof.export_chrome_trace(f"epoch_{epoch}_trace.json")

      helper_functions.clear_gpu(torch, config, pause=False)

  # TODO: Fix
  # log('Compute Grad-CAM for final model', config)
  # compute_grad_cam_results(model, config, config["epochs"], grad_type='Base') # Final check
  # compute_grad_cam_results(model, config, config["epochs"], grad_type='Custom') # Final check
  # gradcam_functions.compute_grad_cam_results(model, config, config["epochs"], grad_type='FullGrad') # Final check

def generate_binary_map_for_perturbations(config, segmentation_map):
  binary_map = segmentation_map.copy()

  # Assume CityScapes segmentation
  labels_to_mask = config["gradcam_masking_labels"]
  for label in labels_to_mask:
    value_to_modify = fetch_key_value(label)

    # TODO: Other strategies
    # if config["perturbation_strategy"] == 'zero'
    binary_map[binary_map == value_to_modify] = 0

  binary_map[binary_map != 0] = 1

  return binary_map

def load_segmentation_frame(config, idex, pair_paths):
  if config["cook_data"]:
    # segmentation_frame = image_functions.open_image(pair_paths[idex][1]).T
    return read_image(pair_paths[idex][1]).permute(0, 2, 1)
  else:
    return image_functions.load_and_process_image(pair_paths[idex][1], config, is_segmentation=True, interpolation=cv2.INTER_AREA).T

def load_input_image(config, dtype, dev, i, batch_size, total_count, pair_paths, raw_frames, segmentation_frame):
  if raw_frames != None:
    x_frames = raw_frames[i:i+batch_size]
  # elif config["cook_data"]:
  #   # x_frame = np.transpose(image_functions.open_image(pair_paths[idex][0]))
  #   # x_frame = image_functions.open_image(pair_paths[idex][0]).T
  #   x_frame = read_image(pair_paths[idex][0]).permute(0, 2, 1)
  # else:
  #   x_frame = image_functions.load_and_process_image(pair_paths[idex][0], config, is_segmentation=False, interpolation=cv2.INTER_AREA).T

  # Apply Segmentation
  if "add_perturbations" in config and config["add_perturbations"] and raw_frames == None:
    for j in range(x_frames.shape[0]):
      binary_map = generate_binary_map_for_perturbations(config, segmentation_frame)
      x_frames[j] = x_frames[j] * binary_map

  dtype = helper_functions.compute_gradcam_dtype(config)
  frames = x_frames.to(dev, non_blocking=config["non_blocking"], dtype=dtype)

  # return torch.unsqueeze(frames, dim=0).float()
  # return frames
  # TODO: Possible optimisation here
  return frames#.float()

def load_image_batch(pair_paths_batch, config, dimensions):
  frames = torch.empty(*dimensions)

  for i in range(len(pair_paths_batch)):
    img_path, seg_path, edge_path = pair_paths_batch[i]

    frames[i] = read_image(img_path).permute(0, 2, 1)

    if config["add_perturbations"]: # TODO: Add bounding box here and below
      binary_map = generate_binary_map_for_perturbations(config, segmentation_maps[i])
      frames[i] = frames[i] * binary_map

  return frames.float()

def process_counts(total_threshold_maps, total_edge_maps, total_segmentation_maps, idex, grad_type, epoch_count, config, heatmap, edge_plot, frame, segmentation_map, pair_paths, total_segmentation_summary, total_number_of_maps):
  # count_start_time = time.time()
  # heatmap = image_functions.vertical_flip(image_functions.rotate_90_clockwise(grad_cam_plot))
  # heatmap = grad_cam_plot
  # TODO: make note of Nearest neighbour here

  # heatmap_time = time.time()

  # Sample - comment out when not needed
  # image, heatmap, visualisation = compute_gradcam_map(config, cam_function, frame, use_rgb=True)
  # save_sample_gradcam(pair_paths, image, heatmap, segmentation_map, idex)
  if "compute_attention" in config:
    if config["compute_attention"]:
      heatmap = heatmap[0]

  total_pixels = heatmap.shape[0] * heatmap.shape[1]

  threshold_maps, edge_maps, segmentation_map_for_usage = compute_segmentation_counts(segmentation_map, heatmap, edge_plot, total_pixels, config)

  total_threshold_maps[idex] = threshold_maps
  total_edge_maps[idex] = edge_maps

  if segmentation_map_for_usage is not None:
    total_segmentation_maps[idex] = segmentation_map_for_usage

  # totals_time = time.time()

  # Sanity Check
  grad_cam_sanity_indexes = [100, 250, 500, 1500, 5000, 5500, 12400, 15000, 24799] # Some defaults here
  # if "grad_cam_sanity_indexes" in config:
  #   grad_cam_sanity_indexes = config["grad_cam_sanity_indexes"]

  if idex in grad_cam_sanity_indexes:
    sanity_check_segmentation_results(grad_type, epoch_count, idex, config, pair_paths, heatmap, edge_plot, frame, segmentation_map, total_segmentation_summary, threshold_maps, total_number_of_maps)

  # sanity_check_time = time.time()

  # Uncomment for debug
  # loop total time: ±0.03178763389587402 secs.
  # log(f'\n\n', config)
  # log(f'heatmap_time: {heatmap_time - count_start_time} secs.', config)
  # log(f'segmentation_time: {segmentation_time - heatmap_time} secs.', config)
  # log(f'threshold_time: {threshold_time - segmentation_time} secs.', config)
  # log(f'edge_time: {edge_time - threshold_time} secs.', config)
  # log(f'save_frame_counts_time: {save_frame_counts_time - edge_time} secs.', config)
  # log(f'totals_time: {totals_time - save_frame_counts_time} secs.', config)
  # log(f'sanity_check_time: {sanity_check_time - totals_time} secs.', config)
  # log(f'\n\n', config)
  # return[1,2,3]

  return [total_threshold_maps, total_edge_maps, total_segmentation_maps]

def sanity_check_segmentation_results(grad_type, epoch_count, idex, config, pair_paths, heatmap, edge_plot, frame, segmentation_map, total_segmentation_summary, threshold_counts, total_number_of_maps):
  log(f'Grad-CAM results sanity check for epoch: {epoch_count}, idex: {idex} ...', config)

  log(f'Path for image: {pair_paths[idex][0]}', config)
  log(f'Path for segmentation map: {pair_paths[idex][1]}', config)

  # if frame != None:
  #   x_frame = frame
  # if config["cook_data"]:
  #   x_frame = image_functions.open_image(pair_paths[idex][0])
  # else:
  #   x_frame = image_functions.load_and_process_image(pair_paths[idex][0], config, is_segmentation=False, interpolation=cv2.INTER_AREA)
  x_frame = frame.permute(2, 1, 0).int().cpu().numpy()
  # x_frame = image_functions.rotate_180(x_frame)

  pixel_total = np.multiply(*heatmap.shape)

  # TODO: Fix this to take in algo
  compute_using_threshold = True
  if "grad_cam_algo" in config:
    if config["grad_cam_algo"] == "absolute":
      compute_using_threshold = False

  total_threshold_counts, _total_edge_counts = compute_final_counts(threshold_counts, [])

  segmentation_summary_total = sum(total_segmentation_summary.values())

  manual_counts = get_manual_counts_for(segmentation_map).values()
  segmentation_summary_manual = sum(manual_counts)

  log(f'Check segmentation_summary: auto: {segmentation_summary_total}, manual: {segmentation_summary_manual}, pixel total: {pixel_total}', config)

  if torch.is_tensor(heatmap):
    heatmap = heatmap.float().cpu().detach().numpy()

  heatmap = image_functions.vertical_flip(image_functions.rotate_90_anti_clockwise(heatmap))
  heatmap = image_functions.rotate_180(heatmap)

  blank_dim = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1]))

  segmentation_map = np.dstack((segmentation_map, blank_dim))
  segmentation_map = np.dstack((segmentation_map, blank_dim))
  segmentation_map = image_functions.vertical_flip(image_functions.rotate_90_anti_clockwise(np.array(segmentation_map)))
  segmentation_map = convert_segmentation_to_colour(segmentation_map.astype('int'))
  segmentation_map = image_functions.rotate_180(segmentation_map)

  edge_plot = image_functions.vertical_flip(image_functions.rotate_90_anti_clockwise(np.array(edge_plot)))
  edge_plot = image_functions.rotate_180(edge_plot)

  log('Save plots ...', config)

  plot_filepath = f'{compute_grad_cam_path(config, grad_type=grad_type)}/epoch_{epoch_count}_sanity_{idex}.png'
  plot, fig = compute_grad_cam_plt(x_frame, heatmap, edge_plot, segmentation_map)
  plot.savefig(plot_filepath)
  plt.close('all')

  # Crop Images here
  cropped_plot_filepath = f'{compute_grad_cam_path(config, grad_type=grad_type)}/epoch_{epoch_count}_sanity_{idex}_cropped.png'
  opend_plot = image_functions.open_image(plot_filepath)
  cropped_image = image_functions.crop_image_with_roi(opend_plot, roi = [100, 300, 380, 2910])
  image_functions.save_image(cropped_image, cropped_plot_filepath)

  plot_filepath = f'{compute_grad_cam_path(config, grad_type=grad_type)}/epoch_{epoch_count}_sanity_{idex}_3_graphs.png'
  plot, fig = compute_grad_cam_plt(x_frame, heatmap, edge_plot, segmentation_map, ncols=3, figsize=(32, 3))
  plot.savefig(plot_filepath)
  plt.close('all')

  # Crop Images here
  cropped_plot_filepath = f'{compute_grad_cam_path(config, grad_type=grad_type)}/epoch_{epoch_count}_sanity_{idex}_3_graphs_cropped.png'
  opend_plot = image_functions.open_image(plot_filepath)
  cropped_image = image_functions.crop_image_with_roi(opend_plot, roi = [30, 260, 380, 2900])
  image_functions.save_image(cropped_image, cropped_plot_filepath)

  # _float_counts_manual, threshold_counts_manual = get_manual_diff_for(segmentation_map, heatmap, config)
  # threshold_counts_manual = np.fromiter(_float_counts_manual.values(), dtype=int).sum()
  threshold_counts_total = np.fromiter(total_threshold_counts.values(), dtype=int).sum()

  log(f'Check threshold_counts: auto: {threshold_counts_total}, pixel total: {pixel_total}', config)

  if (pixel_total != segmentation_summary_manual):
    raise Exception("The counts for manual segmentation summaries dont match!")

  if (segmentation_summary_total != (pixel_total * len(pair_paths))):
    raise Exception("Segmentation summary did not count all pixels!")

  if (threshold_counts_total != pixel_total):
    raise Exception("Segmentation summary did not count all pixels!")

def preload_input_frames(pair_paths, segmentation_maps, config):
  transposed_dims = list(np.array(config["input_size"]).T)
  frames = torch.zeros((len(pair_paths), *transposed_dims))

  for i in range(len(pair_paths)):
    frame = read_image(pair_paths[i][0]).permute(0, 2, 1)

    if config["add_perturbations"]: # TODO: Add bounding box here and below
      binary_map = generate_binary_map_for_perturbations(config, segmentation_maps[i])
      frame = frame * binary_map

    frames[i] = frame

  return frames

def preload_segmentation_maps(pair_paths, config):
  total_segmentation_summary = init_empty_count_hash()

  transposed_dims = list(np.array(config["input_size"]))
  transposed_dims[0] = 1 # Flatten the index array

  maps = np.zeros((len(pair_paths), *transposed_dims))

  # Helper maps
  # colour_map = fetch_colour_map()

  for i in tqdm.tqdm(range(len(pair_paths))):
    segmentation_map = image_functions.open_image(pair_paths[i][1])
    maps[i] = segmentation_map[:, :, 0].T # Already in index form

  maps = maps.astype(int)

  label_map_keys_list = fetch_keys()
  segmentation_counts = np.bincount(maps.flatten())

  for i in range(len(segmentation_counts)):
    total_segmentation_summary[label_map_keys_list[i]] += segmentation_counts[i]

  return maps, total_segmentation_summary

def preload_edge_maps(pair_paths, config):
  dims = [config["input_size"][1], config["input_size"][2]]
  maps = np.zeros((len(pair_paths), *dims))

  for i in range(len(pair_paths)):
    # TODO Custom path
    edge_path = pair_paths[i][2]
    edge_map = image_functions.open_image(edge_path)
    maps[i] = edge_map[:,:,0].T

  return maps

# TODO: Make separate tool which takes in a config and can run these tests
# TODO: vision models segmentation maps
# TODO: Sanity checks
def compute_grad_cam_batch_size(config):
  # batch_size = 1
  batch_size = 8 # Default

  if "grad_cam_batch_size" in config:
    batch_size = config["grad_cam_batch_size"]

  batch_size = torch_models.external_test_model_batch_size(config)

  config["train_batch_size"] = batch_size
  config["gradcam_batch_size"] = batch_size
  helper_functions.log(f"Computed Grad-CAM batch size: {batch_size}", config)

  return [batch_size, config]

def compute_grad_cam_results(model, config, dev, pair_paths, meta, raw_frames, segmentation_maps, edge_maps, total_segmentation_summary, in_line=False, epoch_count=0, grad_type='Base', edge_map=True, in_test=False):
  log(f'Compute GradCAM Results for {config["grad_cam_result_dataset"]} ...', config)
  start_time = time.time()

  log(f'Generate GradCam frames ({grad_type}) ...', config)
  visualisation_filepaths = []

  batch_size, config = compute_grad_cam_batch_size(config)
  cam_function = fetch_cam_function(config, model, grad_type)

  heatmaps = None
  dtype = helper_functions.compute_gradcam_dtype(config)

  remainder = len(pair_paths) % batch_size
  total_count = len(pair_paths) - remainder

  if total_count < 1:
    total_count = len(pair_paths)
    batch_size = 1

  pbar = tqdm.tqdm(total=total_count)

  total_counts = torch.zeros([total_count, raw_frames[0].shape[1], raw_frames[0].shape[2]], dtype=torch.float16)
  total_edge_counts = torch.zeros([total_count, raw_frames[0].shape[1], raw_frames[0].shape[2]], dtype=torch.float16)
  total_segmentation_maps = torch.zeros([total_count, raw_frames[0].shape[1], raw_frames[0].shape[2]], dtype=torch.float16)

  # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
  # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
  # for i in range(0, 200, batch_size):
  for i in range(0, total_count, batch_size):
    total_counts, total_edge_counts, total_segmentation_maps = inner_batch_cam_loop(config, model, dtype, dev, i, batch_size, pbar, segmentation_maps, total_count, pair_paths, raw_frames, cam_function, edge_maps, total_counts, total_edge_counts, total_segmentation_maps, total_segmentation_summary, epoch_count=epoch_count, grad_type=grad_type)

  # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

  log('Generate GradCam Results ...', config)
  compute_using_threshold = True
  if "grad_cam_algo" in config:
    if config["grad_cam_algo"] == "absolute":
      compute_using_threshold = False

  if compute_using_threshold:
    total_counts, total_edge_counts = compute_final_counts(total_counts, total_edge_counts)
  else:
    log("Compute final absolute counts ...", config)
    total_edge_counts = compute_final_edge_counts(total_edge_counts)
    total_counts = compute_final_absolute_counts(total_counts, total_segmentation_maps, dev, dtype)

  total_threshold_percentage = compute_segmentation_percentages(total_segmentation_summary, total_counts)

  # Do not save files if testing
  if in_line and not in_test:
    save_in_line_totals(config, grad_type, epoch_count, total_segmentation_summary, total_counts, total_edge_counts, total_threshold_percentage)
  elif not in_test:
    save_final_totals(config, grad_type, total_segmentation_summary, total_counts, total_edge_counts, total_threshold_percentage)

  if config["keep_gradcam_frames"] == False and config["grad_cam_in_memory"] == False:
    log('Deleting generated frames...', config)
    for frame_path in visualisation_filepaths:
      os.remove(frame_path)

  log(f'Grad-CAM Result time: {time.time() - start_time} secs.', config)
  log('Complete!', config)

  return total_threshold_percentage

def inner_batch_cam_loop(config, model, dtype, dev, i, batch_size, pbar, segmentation_maps, total_count, pair_paths, raw_frames, cam_function, edge_maps, total_counts, total_edge_counts, total_segmentation_maps, total_segmentation_summary, epoch_count=0, grad_type='Base'):
  segmentation_frames = segmentation_maps[i:i+batch_size]

  frames = load_input_image(config, dtype, dev, i, batch_size, total_count, pair_paths, raw_frames, segmentation_frames)
  grad_cam_plots = compute_heat_map(config, model, cam_function, frames, batch_size) # Faster compute
  edge_plots = edge_maps[i:i+batch_size]

  total_number_of_maps = batch_size

  for j in range(batch_size):
    total_counts, total_edge_counts, total_segmentation_maps = process_counts(total_counts, total_edge_counts, total_segmentation_maps, i+j, grad_type, epoch_count, config, grad_cam_plots[j], edge_plots[j], frames[j], segmentation_frames[j][0], pair_paths, total_segmentation_summary, total_number_of_maps)

  pbar.update(batch_size)

  # Clean-up
  del segmentation_frames
  del frames
  del grad_cam_plots
  del edge_plots
  helper_functions.clear_gpu(torch, config)

  return [total_counts, total_edge_counts, total_segmentation_maps]

def save_frame_counts(config, grad_type, i, x_path, segmentation_summary, threshold_counts, threshold_percentage, edge_counts, edge_percentage):
  grad_cam_path = compute_grad_cam_path(config, grad_type=grad_type)
  frame_count_path = f'{grad_cam_path}/frame_summaries/'
  helper_functions.detect_or_create_folder(frame_count_path, print_error=False)

  file_path = f'{frame_count_path}/segmentation_summary.csv'
  csv_data = compute_csv_line(i, segmentation_summary, x_path=x_path)
  save_csv_header(file_path, i, segmentation_summary, x_path=x_path)
  helper_functions.save_csv(file_path, csv_data)

  file_path = f'{frame_count_path}/threshold_counts.csv'
  csv_data = compute_csv_line(i, threshold_counts, x_path=x_path)
  save_csv_header(file_path, i, threshold_counts, x_path=x_path)
  helper_functions.save_csv(file_path, csv_data)

  file_path = f'{frame_count_path}/threshold_percentage.csv'
  csv_data = compute_csv_line(i, threshold_percentage, x_path=x_path)
  save_csv_header(file_path, i, threshold_percentage, x_path=x_path)
  helper_functions.save_csv(file_path, csv_data)

  file_path = f'{frame_count_path}/edge_counts.csv'
  csv_data = compute_csv_line(i, edge_counts, x_path=x_path)
  save_csv_header(file_path, i, edge_counts, x_path=x_path)
  helper_functions.save_csv(file_path, csv_data)

  file_path = f'{frame_count_path}/edge_percentage.csv'
  csv_data = compute_csv_line(i, edge_percentage, x_path=x_path)
  save_csv_header(file_path, i, edge_percentage, x_path=x_path)
  helper_functions.save_csv(file_path, csv_data)

def compute_csv_header(i, summary, x_path=None):
  if x_path:
    csv_header = 'i,x_path'
  else:
    csv_header = 'i'

  keys = list(summary.keys())
  for i in range(len(keys)):
    csv_header = f'{csv_header},{keys[i]}'

  return csv_header

def save_csv_header(file_path, i, summary, x_path=None):
  try:
    csv_file = Path(file_path)
    check = csv_file.is_file()
  except OSError as e:
    check = False

  if not check:
    header = compute_csv_header(i, summary, x_path=None)

    with open(file_path, 'a') as f:
      f.write(f'{header}\n')

  return True

def compute_csv_line(i, summary, x_path=None):
  if x_path:
    csv_line = f'{i},{x_path}'
  else:
    csv_line = f'{i}'

  keys = list(summary.keys())
  for i in range(len(keys)):
    csv_line = f'{csv_line},{summary[keys[i]]}'

  return csv_line

def save_in_line_totals(config, grad_type, epoch_count, total_segmentation_summary, total_threshold_counts, total_edge_counts, total_threshold_percentage):
  log("Save in line totals ...", config)
  grad_cam_path = compute_grad_cam_path(config, grad_type=grad_type)

  file_path = f'{grad_cam_path}/total_segmentation_summary.csv'
  csv_data = compute_csv_line(epoch_count, total_segmentation_summary, x_path=None)
  save_csv_header(file_path, epoch_count, total_segmentation_summary, x_path=None)
  helper_functions.save_csv(file_path, csv_data)

  file_path = f'{grad_cam_path}/total_threshold_counts.csv'
  csv_data = compute_csv_line(epoch_count, total_threshold_counts, x_path=None)
  save_csv_header(file_path, epoch_count, total_threshold_counts, x_path=None)
  helper_functions.save_csv(file_path, csv_data)

  file_path = f'{grad_cam_path}/total_edge_counts.csv'
  csv_data = compute_csv_line(epoch_count, total_edge_counts, x_path=None)
  save_csv_header(file_path, epoch_count, total_edge_counts, x_path=None)
  helper_functions.save_csv(file_path, csv_data)

  file_path = f'{grad_cam_path}/total_threshold_percentage.csv'
  csv_data = compute_csv_line(epoch_count, total_threshold_percentage, x_path=None)
  save_csv_header(file_path, epoch_count, total_threshold_percentage, x_path=None)
  helper_functions.save_csv(file_path, csv_data)

def save_final_totals(config, grad_type, total_segmentation_summary, total_threshold_counts, total_edge_counts, total_threshold_percentage):
  log("Save totals ...", config)
  grad_cam_path = compute_grad_cam_path(config, grad_type=grad_type)

  file_path = f'{grad_cam_path}/total_segmentation_summary.json'
  save_json(file_path, total_segmentation_summary)

  file_path = f'{grad_cam_path}/total_threshold_counts.json'
  save_json(file_path, total_threshold_counts)

  file_path = f'{grad_cam_path}/total_edge_counts.json'
  save_json(file_path, total_edge_counts)

  file_path = f'{grad_cam_path}/total_threshold_percentage.json'
  save_json(file_path, total_threshold_percentage)

def save_json(file_path, data):
  active_file = open(file_path, 'w', encoding='utf-8')
  json.dump(data, active_file)
  active_file.write("\n")
  active_file.close()

def save_sample_gradcam(pair_paths, image, heatmap, segmentation_map, i):
  image_functions.save_image(image_functions.open_image(pair_paths[i][0]), '/home/trex22/orig.png')
  image_functions.save_image(image_functions.undo_rescale_pixels(image), '/home/trex22/image.png')
  image_functions.save_image(image_functions.undo_rescale_pixels(heatmap), '/home/trex22/plot.png')
  image_functions.save_image(convert_segmentation_to_colour(segmentation_map.astype('int')), '/home/trex22/seg.png')

  seg = image_functions.open_image(pair_paths[i][1])
  image_functions.save_image(convert_segmentation_to_colour(seg.astype('int')), '/home/trex22/orig_seg.png')

def labels_to_ignore():
  # "ai_unlabeled"
  return ["ego vehicle", "rectification border", "out of roi", "static", "polegroup", "license plate"]

def fetch_label_map():
  # How to fetch:
  # map = helper_functions.open_json('../segmentation_info/cityscapes_labels.json')

  # Hardcoded:
  # 'ai_unlabeled': 0
  json_map = {
    'unlabeled': 0,
    'ego vehicle': 1,
    'rectification border': 2,
    'out of roi': 3,
    'static': 4,
    'dynamic': 5,
    'ground': 6,
    'road': 7,
    'sidewalk': 8,
    'parking': 9,
    'rail track': 10,
    'building': 11,
    'wall': 12,
    'fence': 13,
    'guard rail': 14,
    'bridge': 15,
    'tunnel': 16,
    'pole': 17,
    'polegroup': 18,
    'traffic light': 19,
    'traffic sign': 20,
    'vegetation': 21,
    'terrain': 22,
    'sky': 23,
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'caravan': 29,
    'trailer': 30,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33,
    'license plate': -1
  }

  # for label in labels_to_ignore():
  #   del json_map[label]

  return json_map

def fetch_full_group_map():
  json_map = {
    'flat': ["road", "sidewalk", "parking", "rail track"],
    'human': ["person", "rider"],
    'vehicle': ["car", "truck", "bus", "motorcycle", "bicycle", "caravan", "trailer"],
    'construction': ["building", "wall", "fence", "guard rail", "bridge", "tunnel"],
    'object': ["pole", "polegroup", "traffic sign", "traffic light"],
    'nature': ["vegetation", "terrain"],
    'sky': ["sky"],
    'void': ["ground", "dynamic", "static"]
  }

  return json_map

def fetch_example_group_map():
  label_map = fetch_label_map()

  road = label_map['road']
  person = label_map['person']
  car = label_map['car']
  building = label_map['building']
  traffic_sign = label_map['traffic sign']
  vegetation = label_map['vegetation']
  sky = label_map['sky']
  ground = label_map['ground']

  json_map = {'flat': road, 'human': person, 'vehicle': car, 'construction': building, 'object': traffic_sign, 'nature': vegetation, 'sky': sky, 'void': ground}
  return json_map

def fetch_colour_map():
  json_map = fetch_label_map()

  # colour_map = helper_functions.open_json('../segmentation_info/cityscapes_colours.json')
  # 'ai_unlabeled': [0, 0, 0],
  colours = {
    'unlabeled': [0, 0, 0],
    'ego vehicle': [0, 0, 0],
    'rectification border': [0, 0, 0],
    'out of roi': [0, 0, 0],
    'static': [0, 0, 0],
    'dynamic': [111, 74, 0],
    'ground': [81, 0, 81],
    'road': [128, 64, 128],
    'sidewalk': [244, 35, 232],
    'parking': [250, 170, 160],
    'rail track': [230, 150, 140],
    'building': [70, 70, 70],
    'wall': [102, 102, 156],
    'fence': [190, 153, 153],
    'guard rail': [180, 165, 180],
    'bridge': [150, 100, 100],
    'tunnel': [150, 120, 90],
    'pole': [153, 153, 153],
    'polegroup': [153, 153, 153],
    'traffic light': [250, 170, 30],
    'traffic sign': [220, 220, 0],
    'vegetation': [107, 142, 35],
    'terrain': [152, 251, 152],
    'sky': [70, 130, 180],
    'person': [220, 20, 60],
    'rider': [255, 0, 0],
    'car': [0, 0, 142],
    'truck': [0, 0, 70],
    'bus': [0, 60, 100],
    'caravan': [0, 0, 90],
    'trailer': [0, 0, 110],
    'train': [0, 80, 100],
    'motorcycle': [0, 0, 230],
    'bicycle': [119, 11, 32],
    'license plate': [0, 0, 142]
  }

  # for label in labels_to_ignore():
  #   del json_map[label]

  return colours

def fetch_keys():
  # How to fetch:
  # label_map = fetch_label_map()
  # keys = list(label_map.keys())

  # Hardcoded:
  # 'ai_unlabeled',
  keys = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']
  # keys = ['unlabeled', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole','traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
  return keys

def fetch_key_value(label):
  value = None
  keys = fetch_keys()

  for i in range(len(keys)):
    if keys[i].lower() == label.lower():
      value = i

  return value

def init_empty_count_hash():
  # How to do this:
  # count_hash = { "ai_unlabeled": 0 }
  # keys = fetch_keys()

  # for i in range(len(keys)):
  #   count_hash[keys[i]] = 0

  # hard-code this:
  # 'ai_unlabeled': 0
  count_hash = {'unlabeled': 0, 'ego vehicle': 0, 'rectification border': 0, 'out of roi': 0, 'static': 0, 'dynamic': 0, 'ground': 0, 'road': 0, 'sidewalk': 0, 'parking': 0, 'rail track': 0, 'building': 0, 'wall': 0, 'fence': 0, 'guard rail': 0, 'bridge': 0, 'tunnel': 0, 'pole': 0, 'polegroup': 0, 'traffic light': 0, 'traffic sign': 0, 'vegetation': 0, 'terrain': 0, 'sky': 0, 'person': 0, 'rider': 0, 'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0, 'license plate': 0}
  return count_hash

def add_totals(frame_count, total_count, total_pixels):
  # exp_start_time_22 = time.time()
  total_sum = 0.0

  # loop_time: 0.030210018157958984 secs.
  for key in frame_count:
    total_count[key] += frame_count[key]
    total_sum += frame_count[key] # Count missing activated pixels

  # loop_time = time.time()

  if frame_count["ai_unlabeled"] > 0:
    total_count["ai_unlabeled"] += frame_count["ai_unlabeled"]
  else:
    total_count["ai_unlabeled"] += total_pixels - total_sum

  # if_time = time.time()

  # For Debugging
  # print(f'\n\n')
  # print(f'loop_time: {loop_time - exp_start_time_22} secs.')
  # print(f'if_time: {if_time - loop_time} secs.')
  return total_count

def sum_mappings(counts, skip_ai_unlabeled=True):
  total_count = 0
  for key in counts:
    total_count += counts[key]

  if not skip_ai_unlabeled:
    total_count += counts["ai_unlabeled"]

  return total_count

# Not used anymore:
def get_manual_diff_for(segmentation_map, heatmap, config):
  label_map = fetch_label_map()
  threshold_plot = np.array(cv2.threshold(heatmap, config["grad_cam_threshold"], 1, cv2.THRESH_BINARY_INV)[1]) # TODO: Make the threshold configurable

  threshold_counts = init_empty_count_hash()
  float_counts = init_empty_count_hash()

  for i in range(segmentation_map.shape[0]):
    for j in range(segmentation_map.shape[1]):
      label = GetKey(label_map, segmentation_map[i][j])

      if threshold_plot[i][j] > 0:
        threshold_counts[label] += 1

      float_counts[label] += heatmap[i][j]

  return [float_counts, threshold_counts]

def get_manual_counts_for(map):
  label_map = fetch_label_map()
  counts = init_empty_count_hash()

  for i in range(map.shape[0]):
    for j in range(map.shape[1]):
      label = GetKey(label_map, map[i][j])
      counts[label] += 1

  return counts

def get_counts_for(map):
  label_map = fetch_label_map()
  # count_hash = init_empty_count_hash()

  count_hash = { "ai_unlabeled": 0 }
  keys = fetch_keys()

  for key in keys:
    count = np.count_nonzero(map == label_map[key])

    if count > 0:
      count_hash[key] = count

  return count_hash

def get_float_counts_for(map, heatmap):
  label_map = fetch_label_map()
  counts = init_empty_count_hash()
  keys = fetch_keys()

  for i in range(len(keys)):
    truth_map = (map == label_map[keys[i]]).astype(int)
    counts[keys[i]] = (truth_map * heatmap).sum()

  return counts

# Computing Thresholds:
# OpenCV Threshold
# threshold_plot = np.array(cv2.threshold(grad_cam_plot, config["grad_cam_threshold"], 1, cv2.THRESH_BINARY)[1])

# NumPy Threshold
# exp_start_time = time.time()

# Old Way - Attempt 1
# threshold_plot = grad_cam_plot.copy()
# threshold_plot[threshold_plot <= config["grad_cam_threshold"]] = -2 # -1 is license plates
# threshold_plot[threshold_plot >= config["grad_cam_threshold"]] = 1

# Attempt 2
# if isinstance(grad_cam_plot, torch.Tensor):
#   threshold_plot = torch.where(grad_cam_plot > config["grad_cam_threshold"], 1, 0) # TODO: Perhaps -2?
# else:
#   threshold_plot = torch.tensor(np.where(grad_cam_plot > config["grad_cam_threshold"], 1, 0)).to(dev)

# Attempt 3
# threshold_plot = (grad_cam_plot > config["grad_cam_threshold"]) * 1
# colors, counts = np.unique(segmentation_map.reshape(-1, 1), return_counts = True, axis = 0)

# Attempt 4
# colors, counts
# segmentation_counts = np.unique(segmentation_map.reshape(-1, 1), return_counts = True, axis = 0)
# threshold_counts = np.unique((segmentation_map * threshold_plot).reshape(-1, 1), return_counts = True, axis = 0)
# edge_counts = np.unique((segmentation_map * edge_plot).reshape(-1, 1), return_counts = True, axis = 0)

# Attempt 5
# segmentation_counts = pd.unique(segmentation_map.reshape(-1, 1))
# threshold_counts = pd.unique((segmentation_map * threshold_plot).reshape(-1, 1))
# edge_counts = pd.unique((segmentation_map * edge_plot).reshape(-1, 1))

# Attempt 6
# segmentation_counts = np.bincount(segmentation_map.reshape(-1))
# threshold_counts = np.bincount((segmentation_map * threshold_plot).reshape(-1))
# edge_counts = np.bincount((segmentation_map * edge_plot).reshape(-1))

# Attempt 7
# threshold_counts = torch.tensor(segmentation_map).to(dev) * threshold_plot.to(dev)
# edge_counts = torch.tensor(segmentation_map * edge_plot)

def compute_threshold_maps_using_thresholds(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config, dev):
  dtype = helper_functions.compute_gradcam_dtype(config)

  if isinstance(grad_cam_plot, torch.Tensor):
    threshold_plot = torch.where(grad_cam_plot > config["grad_cam_threshold"], 1, 0) # TODO: Perhaps -2?
  else:
    threshold_plot = torch.tensor(np.where(grad_cam_plot > config["grad_cam_threshold"], 1, 0)).to(dev)

  threshold_maps = convert_to_tensor(segmentation_map, dev, dtype) * convert_to_tensor(threshold_plot, dev, dtype)

  edge_threshold_map = np.where(edge_plot == 255, 1, 0)
  edge_maps = convert_to_tensor(segmentation_map, dev, dtype) * convert_to_tensor(edge_threshold_map, dev, dtype)

  return [threshold_maps.to(dev, dtype=dtype), edge_maps, None]

def compute_absolute_sum_maps(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config, dev):
  dtype = helper_functions.compute_gradcam_dtype(config)

  grad_cam_plot = convert_to_tensor(grad_cam_plot, dev, dtype)
  # threshold_maps = torch.tensor(segmentation_map).to(dev, dtype=dtype) * grad_cam_plot.to(dev, dtype=dtype)

  segmentation_map = convert_to_tensor(segmentation_map, dev, dtype)

  edge_threshold_map = np.where(edge_plot == 255, 1, 0)
  edge_maps = convert_to_tensor(segmentation_map, dev, dtype) * convert_to_tensor(edge_threshold_map, dev, dtype)

  return [grad_cam_plot, edge_maps, segmentation_map]

def compute_segmentation_counts(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config):
  if not (segmentation_map.shape == grad_cam_plot.shape):
    raise Exception(f"The shapes of the segmentation_map and grad_cam_plot don't match! segmentation_map.shape: {segmentation_map.shape}, grad_cam_plot.shape: {grad_cam_plot.shape}")

  dev = helper_functions.fetch_device(config, verbose=False)

  compute_using_threshold = True
  if "grad_cam_algo" in config:
    if config["grad_cam_algo"] == "absolute":
      compute_using_threshold = False

  if compute_using_threshold:
    return compute_threshold_maps_using_thresholds(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config, dev)
  else:
    return compute_absolute_sum_maps(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config, dev)

def flatten_segmentation(segmentation_map):
  return extract_single_value(segmentation_map).flatten()

def extract_single_value(segmentation_map):
  if np.array(segmentation_map.shape).shape[0] < 3:
    return segmentation_map

  if segmentation_map[0, 0, 0].size() == torch.Size([]):
    return segmentation_map

  return segmentation_map[:, :, 0]  # Should be just the segmentation map with labels not colours!

def compute_final_edge_counts(total_edge_counts):
  if total_edge_counts != []:
    total_edge_counts = total_edge_counts.flatten()
    total_edge_counts = torch.bincount(total_edge_counts.to(dtype=torch.int8))

  label_map_keys_list = fetch_keys()

  total_threshold_counts_dict = init_empty_count_hash()
  total_edge_counts_dict = init_empty_count_hash()

  if total_edge_counts != []:
    for i in range(len(total_edge_counts)):
      total_edge_counts_dict[label_map_keys_list[i]] += total_edge_counts[i].item()

  return total_edge_counts_dict

def compute_absolute_counts_for(grad_cam_plot, segmentation_map, total_threshold_counts_dict, label_map_keys_list, dev, dtype):
  flattened_segmentation_map = convert_to_tensor(flatten_segmentation(segmentation_map), dev, torch.int8)
  flattened_grad_cam_plot = convert_to_tensor(grad_cam_plot.flatten(), dev, dtype)

  # Check Dimensions here!
  if flattened_grad_cam_plot.shape != flattened_segmentation_map.shape:
    raise "GradCAM and segmentation dimensions dont't match!"

  # unique_labels = torch.unique(flattened_segmentation_map)

  # Disable the check for slightly better performance
  # if flattened_segmentation_map.size() != flattened_grad_cam_plot.size():
  #   raise "Maps don't equal in size!"

  # Attempt 1
  # absolute_map = (flattened_segmentation_map * flattened_grad_cam_plot).cpu()
  # for i in range(len(flattened_segmentation_map)):
  #   label_name = label_map_keys_list[flattened_segmentation_map[i]]
  #   total_threshold_counts_dict[label_name] += absolute_map[i]

  # Attempt 2
  # for label in tqdm.tqdm(unique_labels):
  #   sub_segmentation_map = flattened_segmentation_map.clone()
  #   sub_segmentation_map[sub_segmentation_map != label] = 0

  #   label_name = label_map_keys_list[label.item()]
  #   total_threshold_counts_dict[label_name] += (sub_segmentation_map * flattened_grad_cam_plot).sum().item()

  # Attempt 3
  label_counts = torch.bincount(flattened_segmentation_map, weights=flattened_grad_cam_plot)
  for label, count in enumerate(label_counts):
    label_name = label_map_keys_list[label]
    total_threshold_counts_dict[label_name] += count.item()

  return total_threshold_counts_dict

def compute_final_absolute_counts(total_threshold_counts, total_segmentation_maps, dev, dtype):
  label_map_keys_list = fetch_keys()
  total_threshold_counts_dict = init_empty_count_hash()

  total_threshold_counts_dict = compute_absolute_counts_for(total_threshold_counts, total_segmentation_maps, total_threshold_counts_dict, label_map_keys_list, dev, dtype)

  return total_threshold_counts_dict

def compute_final_counts(total_threshold_counts, total_edge_counts):
  total_threshold_counts = total_threshold_counts.flatten()
  total_threshold_counts = torch.bincount(total_threshold_counts.to(dtype=torch.int8))

  if total_edge_counts != []:
    total_edge_counts = total_edge_counts.flatten()
    total_edge_counts = torch.bincount(total_edge_counts.to(dtype=torch.int8))

  label_map_keys_list = fetch_keys()

  # convert to dict
  # total_segmentation_summary_dict = init_empty_count_hash()
  total_threshold_counts_dict = init_empty_count_hash()
  total_edge_counts_dict = init_empty_count_hash()

  for i in range(len(total_threshold_counts)):
    total_threshold_counts_dict[label_map_keys_list[i]] += total_threshold_counts[i].item()

  if total_edge_counts != []:
    for i in range(len(total_edge_counts)):
      total_edge_counts_dict[label_map_keys_list[i]] += total_edge_counts[i].item()

  return total_threshold_counts_dict, total_edge_counts_dict

def compute_segmentation_percentages(segmentation_summary, threshold_counts):
  threshold_percentage = init_empty_count_hash()

  for key in segmentation_summary:
    if segmentation_summary[key] > 0:
      if key in threshold_counts:
        threshold_percentage[key] = (threshold_counts[key] / segmentation_summary[key]) * 100

  return threshold_percentage

def convert_segmentation_to_colour(segmentation_map):
  # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
  label_map = fetch_label_map()
  colour_map = fetch_colour_map()

  width, height = segmentation_map.shape[0:2]

  # binary_seg[binary_seg == label_value] = 255
  # for key, value in label_map.items():
  #   segmentation_map = set_colour(segmentation_map, value, key, colour_map)

  for w in range(width):
    for h in range(height):
      label = GetKey(label_map, segmentation_map[w][h][0])
      segmentation_map[w][h] = colour_map[label][::-1] # Reversed for BGR

      # if segmentation_map[w][h][0] == 26:
      #   segmentation_map[w][h] = colour_map['car']

  return segmentation_map

def set_colour(segmentation_map, value, key, colour_map):
  try:
    # segmentation_map[segmentation_map == [value, 0, 0]] = colour_map[key]
    segmentation_map[segmentation_map == [value, value, value]] = colour_map[key]
    return segmentation_map
  except:
    return segmentation_map

def GetKey(dict, val):
  selected_key = "unlabeled"

  for key, value in dict.items():
    if val == value:
      selected_key = key

  return selected_key

def clone(map):
  if isinstance(map, torch.Tensor):
    return map.clone()
  else:
    return map.copy()

def to_numpy(arr):
  if isinstance(arr, torch.Tensor):
    return arr.numpy()
  else:
    return arr

def to_torch(arr):
  if isinstance(arr, torch.Tensor):
    return arr
  else:
    return torch.tensor(arr)

def generate_binary_map_for(label_value, map):
  try:
    if isinstance(label_value, str):
      label_value = fetch_label_map()[label_value]
  except KeyError:
    label_value = fetch_example_group_map()[label_value]

  # fetch_full_group_map()
  binary_seg = extract_single_value(clone(to_torch(map)))
  binary_seg[binary_seg != label_value] = 0
  binary_seg[binary_seg == label_value] = 255

  return binary_seg

def generate_combined_binary_map(group_label, map):
  labels = fetch_full_group_map()[group_label]

  results = []
  for label in labels:
    results.append(to_numpy(generate_binary_map_for(label, np.array(map))))

  return torch.tensor(np.sum(np.array(results), axis=0))

def generate_grad_cam_video(model, config):
  log(f'GradCAM {config["grad_cam_dataset"]} ...', config)

  dev = helper_functions.fetch_device(config)
  x_frames = data_helpers.load_video_sequence(config, config["grad_cam_dataset"])

  visualisation_filepaths = []
  plot_filepaths = []
  grad_cam_plots = []

  # TODO: Make GradCam work in batches (performance optimisation)
  pbar = tqdm.tqdm(total=len(x_frames))
  for i in range(len(x_frames)):
    # TODO: dlpack conversion
    frame = torch.tensor(x_frames[i]).float().to(dev, non_blocking=config["non_blocking"])
    name = f'{i}'

    visualisation_filepaths.append(compute_visualisation_filepath(config, name))
    plot_filepaths.append(compute_plot_filepath(config, name))

    # TODO: Add in name besides counter
    # save_gradcam = not config["grad_cam_in_memory"]
    plot = create_gradcam(model, frame, config, name=name, save=True)

    # TODO: Get in-memory working
    # if config["grad_cam_in_memory"] == True:
    #   grad_cam_plots.append(plot)

    pbar.update(1)

  log('Generate Video outputs ...', config) # TODO: Save filepaths
  visualisation_video_path = f'{compute_grad_cam_path(config)}/{config["grad_cam_dataset"]}_visualisation.mp4'
  image_functions.save_video(visualisation_filepaths, visualisation_video_path, config["input_size"][1], config["input_size"][2], fps=20)

  plot_video_path = f'{compute_grad_cam_path(config)}/{config["grad_cam_dataset"]}_plot.mp4'
  # image_functions.save_video(plot_filepaths, plot_video_path, 1200, 400, fps=20)
  # image_functions.save_video(plot_filepaths, plot_video_path, 1518, 710, fps=20) # TODO: Make width and height configurable
  image_functions.save_video(plot_filepaths, plot_video_path, 1118, 310, fps=20)

  if config["keep_gradcam_frames"] == False: #and config["grad_cam_in_memory"] == False:
    log('Deleting generated frames...', config)
    for frame_path in visualisation_filepaths:
      os.remove(frame_path)

  log('Complete!', config)

# TODO: model.eval()

def create_gradcam(model, input_tensor, config, save=True, name='single_test', grad_type='Base'):
  cam_function = fetch_cam_function(config, model, grad_type)
  grad_cam_output = compute_gradcam_map(config, model, cam_function, input_tensor, use_rgb=True) # TODO: Configure RGB
  edge_plot = compute_edge_map(config, input_tensor)

  if save:
    save_gradcam(config, grad_cam_output, edge_plot, name)

  return grad_cam_output

def fetch_cam_function(config, model, grad_type='Base'):
  if "compute_attention" in config:
    if config["compute_attention"]:
      return None

  use_cuda = False
  dev = None

  if config["summary_device_name"] == "cuda":
    use_cuda = True
    dev = helper_functions.fetch_device(config, verbose=False)
  else:
    dev = torch.device("cpu") # Safe Fallback

  dtype = helper_functions.compute_gradcam_dtype(config)
  model.to(dev, dtype=dtype)

  model.eval()

  # TODO: Set batch size
  if grad_type == 'Base':
    try: # If my fork of the library
      base_cam = GradCAM(model=model, target_layers=model.target_layers, use_cuda=use_cuda, compute_device=dev)
    except:
      # base_cam = GradCAM(model=model, target_layers=model.target_layers, use_cuda=use_cuda)
      base_cam = GradCAM(model=model, target_layers=model.target_layers)

    return base_cam
  elif grad_type == 'Custom':
    custom_stack = []

    for i in range(len(model.cnn_stack) - 1):
      layer = model.cnn_stack[i]
      # if not isinstance(layer, nn.Flatten) or not isinstance(layer, nn.Linear):
      if isinstance(layer, nn.Conv2d):
        custom_stack.append(layer)

    try: # If my fork of the library
      return GradCAM(model=model, target_layers=custom_stack, use_cuda=use_cuda, compute_device=dev)
    except:
      # return GradCAM(model=model, target_layers=custom_stack, use_cuda=use_cuda)
      return GradCAM(model=model, target_layers=custom_stack)
  elif grad_type == 'FullGrad':
    try: # If my fork of the library
      return FullGrad(model=model, target_layers=model.target_layers, use_cuda=use_cuda, compute_device=dev)
    except:
      # return FullGrad(model=model, target_layers=model.target_layers, use_cuda=use_cuda)
      return FullGrad(model=model, target_layers=model.target_layers)

def compute_heat_map(config, model, cam_function, input_tensor, batch_size):
  if "compute_attention" in config:
    if config["compute_attention"]:
      return model.attention(input_tensor)

  # Construct the CAM object once, and then re-use it on many images:
  targets = None # [ClassifierOutputTarget(None)] # None # TODO: Config See docs
  cam_function.batch_size = batch_size

  # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
  # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:

  # Attempt to make it GradCAM compatible
  # dtype = torch.float32
  # model.to(dtype=dtype)
  # input_tensor.to(dtype=dtype)

  heatmap = cam_function(input_tensor=input_tensor, targets=targets) # TODO: Configure smoothing

  # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
  # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
  # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))
  # prof.export_chrome_trace("gpu_single_trace.json")

  return heatmap

def compute_edge_map(config, input_tensor, batch=False):
  # TODO: Possibly? Sobel Filter
  # Assume batch_size of 1 (for now)

  # CV2 Implementation
  frame = input_tensor[0]
  # frame = np.array(frame.T.type(torch.int8).cpu())
  frame = frame.permute(2, 1, 0).type(torch.int8).cpu().numpy()

  image_functions.compute_canny_edges(frame, config["canny_threshold1"], config["canny_threshold2"])
  return canny

def compute_gradcam_map(config, model, cam_function, input_tensor, use_rgb=True):
  heatmap = compute_heat_map(config, model, cam_function, input_tensor) # TODO: Batch Size
  edge_plot = compute_edge_map(config, input_tensor) # TODO: Batch Size

  # Make sure that cuda converts to cpu otherwise does not
  image = image_functions.rescale_pixels(np.array(input_tensor[0].cpu().permute(1, 2, 0)))
  visualisation = show_cam_on_image(image, heatmap, use_rgb=use_rgb)

  # TODO: GradCam sanity check
  # TODO: KL-DIV Comparison
  image = image_functions.rotate_90_anti_clockwise(image)
  heatmap = image_functions.rotate_90_anti_clockwise(heatmap)
  visualisation = image_functions.rotate_90_anti_clockwise(visualisation)
  edge_plot = image_functions.rotate_90_anti_clockwise(edge_plot)

  grad_cam_output = [
    image,
    heatmap,
    visualisation
  ]

  if config["draw_grad_cam"]:
    plot, _fig = compute_grad_cam_plt(image, heatmap, edge_plot, visualisation)
    plot.show()

  return grad_cam_output

def compute_grad_cam_plt(image, heatmap, edge_plot, segmentation_map, title='GradCam', nrows=1, ncols=4, figsize=(32, 4)):
  # Generate the plot
  fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

  for i, ax in enumerate(axs.flatten()):
    ax.axis('off')
    plt.sca(ax)

    if i == 0:
      plt.imshow(image, cmap=plt.cm.jet)
      plt.title('Image')
    if i == 1:
      h = plt.imshow(heatmap, cmap=plt.cm.jet)
      plt.title('GradCAM Heat Map')
      # plt.colorbar(h)
    if i == 2:
      plt.imshow(segmentation_map, cmap=plt.cm.jet)
      plt.title("Dataset Segmentation Map")
    if i == 3 and ncols == 4:
      plt.title('Canny Edges Plot')
      plt.imshow(edge_plot, cmap='Greys', interpolation='nearest')

  #plt.tight_layout()
  # plt.suptitle(title) # Remove the title

  return [plt, fig]

def compute_grad_cam_seaborn_plt(image, heatmap, visualisation, title='GradCam', nrows=1, ncols=3, figsize=(12, 4)): # (12, 4), #(24, 8)
  # isns.imgplot
  # plot.savefig(plot_filepath)
  # plt.savefig('/data/tmp/fig.png')

  # g = isns.ImageGrid(heatmap)
  # g = isns.ImageGrid([image, heatmap, visualisation])

  heatmap_8bit = (heatmap/256).astype(np.int8)
  heatmap_3d = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
  vis = np.concatenate((image, heatmap_3d, visualisation), axis=1)

  g = isns.ImageGrid(vis)

# TODO: Improve the heatmap
# TODO: Headings

def compute_cv2_plt(image, heatmap, visualisation, title='GradCam', bordersize=100, spacing=25, value=[255, 255, 255]):
  image_unscaled = image_functions.undo_rescale_pixels(image)
  image_unscaled = image_functions.add_image_border(image_unscaled, bordersize=spacing, value=value)

  # Generate heatmap
  heatmap_8bit = image_functions.undo_rescale_pixels(heatmap).astype(np.int8)
  heatmap_3d = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
  heatmap_3d = image_functions.add_image_border(heatmap_3d, bordersize=spacing, value=value)

  visualisation = image_functions.add_image_border(visualisation, bordersize=spacing, value=value)

  vis = np.concatenate((image_unscaled, heatmap_3d, visualisation), axis=1)
  vis_border = image_functions.add_image_border(vis, bordersize=bordersize, value=value)

  # image_functions.save_image(image_functions.undo_rescale_pixels(heatmap), "/data/tmp/1_heat.png")
  # image_functions.save_image(vis_border, "/data/tmp/1_img.png")

  return vis_border

def compute_grad_cam_path(config, grad_type=None):
  full_save_path = helper_functions.compute_base_save_path(config)

  grad_cam_ident = 'grad_cam'
  if "grad_cam_ident" in config:
    grad_cam_ident = config["grad_cam_ident"]

  dataset_ident = config['grad_cam_result_dataset']

  if "grad_cam_drop_percentage" in config:
    if config["grad_cam_drop_percentage"]:
      dataset_ident = f'{dataset_ident}_turning_only'

  if config["add_perturbations"]:
    dataset_ident = f'{dataset_ident}_perturbation_{config["perturbation_strategy"]}'

  if grad_type:
    grad_cam_path = f"{full_save_path}/{grad_cam_ident}/{dataset_ident}/{grad_type}/"
  else:
    grad_cam_path = f"{full_save_path}/{grad_cam_ident}/{dataset_ident}/"

  helper_functions.detect_or_create_folder(grad_cam_path, print_error=False)

  return grad_cam_path

def compute_image_path(config, name):
  grad_cam_path = compute_grad_cam_path(config)

  image_path = f'{grad_cam_path}/image/'
  helper_functions.detect_or_create_folder(image_path, print_error=False)
  image_filepath = f'{image_path}/{name}.png'

  return image_filepath

def compute_heatmap_path(config, name):
  grad_cam_path = compute_grad_cam_path(config)

  heatmap_path = f'{grad_cam_path}/heatmap/'
  helper_functions.detect_or_create_folder(heatmap_path, print_error=False)
  heatmap_filepath = f'{heatmap_path}/{name}.png'

  return heatmap_filepath

def compute_edge_plot_path(config, name):
  grad_cam_path = compute_grad_cam_path(config)

  edge_plot_path = f'{grad_cam_path}/edge_plot/'
  helper_functions.detect_or_create_folder(edge_plot_path, print_error=False)
  edge_plot_filepath = f'{edge_plot_path}/{name}.png'

  return edge_plot_filepath

def compute_visualisation_filepath(config, name):
  grad_cam_path = compute_grad_cam_path(config)

  visualisation_path = f'{grad_cam_path}/visualisation'
  helper_functions.detect_or_create_folder(visualisation_path, print_error=False)
  visualisation_filepath = f'{visualisation_path}/{name}.png'

  return visualisation_filepath

def compute_plot_filepath(config, name):
  grad_cam_path = compute_grad_cam_path(config)

  plot_path = f'{grad_cam_path}/plot/'
  helper_functions.detect_or_create_folder(plot_path, print_error=False)
  plot_filepath = f'{plot_path}/{name}.png'

  return plot_filepath

# TODO: Save to wandb

def save_gradcam(config, grad_cam_output, edge_plot, name):
  image, heatmap, visualisation = grad_cam_output

  image_filepath = compute_image_path(config, name)
  heatmap_filepath = compute_heatmap_path(config, name)
  edge_plot_filepath = compute_edge_plot_path(config, name)
  visualisation_filepath = compute_visualisation_filepath(config, name)
  plot_filepath = compute_plot_filepath(config, name)

  if config["save_grad_cam"]:
    image_functions.save_image(image_functions.undo_rescale_pixels(image), image_filepath)
    image_functions.save_image(image_functions.undo_rescale_pixels(heatmap), heatmap_filepath)
    image_functions.save_image(np.array(visualisation), visualisation_filepath)

  if config["save_grad_cam_plot"] and config["grad_cam_plot"] == "matplotlib":
    plot, fig = compute_grad_cam_plt(image, heatmap, edge_plot, visualisation)
    plot.savefig(plot_filepath)
    # plt.close(fig)
    plt.close('all')

    # TODO: Crop image here
  elif config["save_grad_cam_plot"] and config["grad_cam_plot"] == "seaborn":
    compute_grad_cam_seaborn_plt(image, heatmap, visualisation)
  elif config["save_grad_cam_plot"]: # cv2
    vis = compute_cv2_plt(image, heatmap, visualisation)
    image_functions.save_image(vis, plot_filepath)

  # return [image_filepath, heatmap_filepath, visualisation_filepath, plot_filepath]
