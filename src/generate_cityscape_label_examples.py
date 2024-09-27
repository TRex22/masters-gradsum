import time
from datetime import datetime
import tqdm
import math
import os
import json
from subprocess import Popen

# import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as np
# config.update('jax_enable_x64', True)

import numpy as np
import pandas as pd

# See: https://pytorch.org/tutorials/beginner/nn_tutorial.html?highlight=cnn
import torch
# import torchmetrics

start_time = time.time()

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torchinfo import summary

# TODO: LR Warmup
# https://pypi.org/project/pytorch-warmup/
# https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch

# import torcheck
# TODO: https://github.com/pengyan510/torcheck
# https://towardsdatascience.com/testing-your-pytorch-models-with-torcheck-cb689ecbc08c

import sys
sys.path.insert(1, './pyTorch_phoenix/')
sys.path.insert(1, './data_Functions/')
sys.path.insert(1, './carla/')
sys.path.insert(1, './carla/benchmark_tool/')

# Import training modules
import helper_functions
from helper_functions import log
environment = helper_functions.check_environment()

import datasets
import data_functions
import image_functions
import cooking_functions
import data_helpers
import gradcam_functions

import torch_models
import torch_optimisers
from torch_dataset import CustomImageDataset
import torch_trainer

config = {
  "purge_cuda_memory": False,
  "model_save_path": "/mnt/excelsior/trained_models",
  "base_data_path":  "/data/data",
  "base_cook_data_path": "/data/data/cooking",
  "roi": {
    "cookbook": [76, 135, 0, 255],
    "cityscapes": [420, 950, 150, 1942],
    "cityscapesvideo": [420, 950, 150, 1942],
    "carlaimitation": None,
    "udacity": [160, 470, 0, 640],
    "fromgames": [300, 720, 150, 1764]
  },
  "dataset_name": "udacity",
  "grad_cam_result_dataset": "cityscapes",
  "log_to_file": False,
  "device_name": "cuda:0",
  "input_size": [3, 256, 60],
  "win_drive": "G",
  "cook_data": True,
  "cook_only": False,
  "convert_to_greyscale": False,
  "add_perturbations": False,
}

config["grad_cam_coarse"] = False
config["grad_cam_fine"] = True

helper_functions.clear_gpu(torch, config)
dev = helper_functions.fetch_device(config)

raw_frames = None
segmentation_maps = None

log("Load Data", config)
dev = helper_functions.fetch_device(config)

pair_paths, meta = data_helpers.load_segmented_data(config)

if config["grad_cam_coarse"]:
  pair_paths = gradcam_functions.filter_coarse_only(pair_paths, config)
elif config["grad_cam_fine"]:
  pair_paths = gradcam_functions.filter_fine_only(pair_paths, config)

segmentation_maps, total_segmentation_summary = gradcam_functions.preload_segmentation_maps(pair_paths, config)
log('Loaded Segmentation Maps', config)

raw_frames = gradcam_functions.preload_input_frames(pair_paths, segmentation_maps, config)
log('Loaded input frames', config)

# Groups
log("Compute Group Examples ...", config)
save_base_path = "/home/trex22/Downloads/cityscapes_group_examples/"
helper_functions.detect_or_create_folder(save_base_path)

groups = gradcam_functions.fetch_full_group_map()

for group in tqdm.tqdm(groups):
  log(f"Group: {group}", config)

  selected_segmentation_map = segmentation_maps[0]
  selected_binary_map = selected_segmentation_map
  selected_raw_image = raw_frames[0]
  selected_index = 0
  total_sum = 0

  for i in tqdm.tqdm(range(len(segmentation_maps))):
    raw_image = raw_frames[i]
    segmentation_map = segmentation_maps[i]

    binary_map = gradcam_functions.generate_combined_binary_map(group, np.array(segmentation_map))
    binary_sum = binary_map.sum()

    if total_sum < binary_sum:
      total_sum = binary_sum
      selected_binary_map = binary_map
      # selected_segmentation_map = segmentation_map
      # selected_raw_image = raw_image
      selected_index = i

  selected_segmentation_map = segmentation_maps[selected_index]
  selected_raw_image = raw_frames[selected_index]

  segmentation_map = selected_segmentation_map.T
  blank_dim = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1]))

  segmentation_map = np.dstack((segmentation_map, blank_dim))
  segmentation_map = np.dstack((segmentation_map, blank_dim))
  colour_map = gradcam_functions.convert_segmentation_to_colour(segmentation_map)

  binary_map = selected_binary_map.numpy().T
  binary_map = np.dstack((binary_map, blank_dim))
  binary_map = np.dstack((binary_map, blank_dim))

  binary_map[:,:,1] = binary_map[:,:,0]
  binary_map[:,:,2] = binary_map[:,:,0]

  example_image = np.concatenate((selected_raw_image.T, segmentation_map, binary_map), axis=0)
  image_functions.save_image(example_image, f"{save_base_path}/{group}.png")

  log(f"Path: {pair_paths[selected_index][0]}", config)

log("Compute Label Examples ...", config)
save_base_path = "/home/trex22/Downloads/cityscapes_label_examples/"
helper_functions.detect_or_create_folder(save_base_path)

labels = gradcam_functions.fetch_label_map()

for label in tqdm.tqdm(labels):
  log(f"Label: {label}", config)

  if label == 'license plate':
    next

  selected_segmentation_map = segmentation_maps[0]
  selected_binary_map = torch.zeros((segmentation_maps[0].shape))
  selected_raw_image = raw_frames[0]
  selected_index = 0
  total_sum = 0

  for i in tqdm.tqdm(range(len(segmentation_maps))):
    raw_image = raw_frames[i]
    segmentation_map = segmentation_maps[i]

    binary_map = gradcam_functions.generate_binary_map_for(label, np.array(segmentation_map))
    binary_sum = binary_map.sum()

    if total_sum < binary_sum:
      total_sum = binary_sum
      selected_binary_map = binary_map
      # selected_segmentation_map = segmentation_map
      # selected_raw_image = raw_image
      selected_index = i

  selected_segmentation_map = segmentation_maps[selected_index]
  selected_raw_image = raw_frames[selected_index]

  segmentation_map = selected_segmentation_map.T
  blank_dim = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1]))

  segmentation_map = np.dstack((segmentation_map, blank_dim))
  segmentation_map = np.dstack((segmentation_map, blank_dim))
  colour_map = gradcam_functions.convert_segmentation_to_colour(segmentation_map)

  binary_map = selected_binary_map.numpy().T
  binary_map = np.dstack((binary_map, blank_dim))
  binary_map = np.dstack((binary_map, blank_dim))

  binary_map[:,:,1] = binary_map[:,:,0]
  binary_map[:,:,2] = binary_map[:,:,0]

  example_image = np.concatenate((selected_raw_image.T, segmentation_map, binary_map), axis=0)
  image_functions.save_image(example_image, f"{save_base_path}/{label}.png")

  log(f"Path: {pair_paths[selected_index][0]}", config)

log(f"Done!", config)


