import time
from datetime import datetime
import tqdm
import math
import os
import json
from subprocess import Popen

import numpy as np
import pandas as pd

# See: https:#pytorch.org/tutorials/beginner/nn_tutorial.html?highlight=cnn
import torch
# import torchmetrics

start_time = time.time()

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torchinfo import summary

# TODO: LR Warmup
# https:#pypi.org/project/pytorch-warmup/
# https:#stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch

# import torcheck
# TODO: https:#github.com/pengyan510/torcheck
# https:#towardsdatascience.com/testing-your-pytorch-models-with-torcheck-cb689ecbc08c

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
import torch_trainer
import torch_tester

from torch_dataset import CustomImageDataset

def run_gradcam_tests(config):
  log('==========================================', config)
  log(f'=== GradCAM Tests                      ===', config)
  log('==========================================', config)
  dev = helper_functions.fetch_device(config, verbose=False) # Should be "cpu"
  config["bfloat16"] = False
  dtype = helper_functions.compute_gradcam_dtype(config)
  # dtype = torch.float16

  expected_batch_size = 400 #1 # Hardcoded for now
  actual_batch_size, returned_config = gradcam_functions.compute_grad_cam_batch_size(config)
  assert expected_batch_size == actual_batch_size, f"Batch Size is invalid! Expected: {expected_batch_size}, actual: {actual_batch_size}"

  log('==========================================', config)
  log(f'=== GradCAM Absolute Sum Maps          ===', config)
  log('==========================================', config)
  segmentation_map = torch.tensor(np.array([[15, 15, 15], [3, 3, 3]]))
  grad_cam_plot = torch.tensor(np.array([[0.00, 0.26, 0.7], [0.0996, 0.14, 0.8]]))
  edge_plot = np.array([[0, 255, 0], [255, 255, 0]]) # TODO: fix the scaling
  total_pixels = np.prod(segmentation_map.shape)

  expected_threshold_maps = torch.tensor(np.array([[0.0000, 3.9004, 10.5000], [0.2988, 0.4199, 2.3984]])).to(dev, dtype=dtype)
  expected_edge_maps = np.array([[0, 15,  0], [3,  3,  0]])

  gradcam_maps, edge_maps, segmentation_map_for_usage = gradcam_functions.compute_absolute_sum_maps(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config, dev)

  assert torch.equal(segmentation_map, segmentation_map_for_usage), "Segmentation Maps don't match!"
  assert torch.equal(grad_cam_plot.to(dtype=dtype), gradcam_maps), "expected_threshold_maps don't match!"
  assert np.array_equal(expected_edge_maps, edge_maps), "Edge maps don't match!"

  log('==========================================', config)
  log(f'=== GradCAM Threshold Maps             ===', config)
  log('==========================================', config)
  # GradCAM plot is torch tensor
  grad_cam_plot = torch.tensor(np.array([[0.00, 0.26, 0.7], [0.0996, 0.14, 0.8]]))

  expected_threshold_maps = torch.tensor(np.array([[0.0, 0.0, 15.0], [0.0, 0.0, 3.0]])).to(dev, dtype=dtype)
  # expected_edge_maps = # Same as above
  expected_segmentation_map_for_usage = None

  threshold_maps, edge_maps, segmentation_map_for_usage = gradcam_functions.compute_threshold_maps_using_thresholds(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config, dev)

  assert np.array_equal(expected_segmentation_map_for_usage, segmentation_map_for_usage), "Segmentation Maps don't match!"
  assert torch.equal(expected_threshold_maps, threshold_maps), "expected_threshold_maps don't match!"
  assert np.array_equal(expected_edge_maps, edge_maps), "Edge maps don't match!"

  # GradCAM plot is numpy array
  grad_cam_plot = np.array([[0.00, 0.26, 0.7], [0.0996, 0.14, 0.8]])

  expected_threshold_maps = torch.tensor(np.array([[0.0, 0.0, 15.0], [0.0, 0.0, 3.0]])).to(dev, dtype=dtype)
  # expected_edge_maps = # Same as above
  expected_segmentation_map_for_usage = None

  threshold_maps, edge_maps, segmentation_map_for_usage = gradcam_functions.compute_threshold_maps_using_thresholds(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config, dev)

  assert np.array_equal(expected_segmentation_map_for_usage, segmentation_map_for_usage), "Segmentation Maps don't match!"
  assert torch.equal(expected_threshold_maps, threshold_maps), "expected_threshold_maps don't match!"
  assert np.array_equal(expected_edge_maps, edge_maps), "Edge maps don't match!"

  log('==========================================', config)
  log(f'=== GradCAM Segmentation Counts        ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "threshold"
  threshold_maps, edge_maps, segmentation_map_for_usage = gradcam_functions.compute_segmentation_counts(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config)

  assert np.array_equal(expected_segmentation_map_for_usage, segmentation_map_for_usage), "Segmentation Maps don't match!"
  assert torch.equal(expected_threshold_maps, threshold_maps), "expected_threshold_maps don't match!"
  assert np.array_equal(expected_edge_maps, edge_maps), "Edge maps don't match!"

  config["grad_cam_algo"] = "absolute"
  grad_cam_plot = torch.tensor(np.array([[0.00, 0.26, 0.7], [0.0996, 0.14, 0.8]]))
  gradcam_maps, edge_maps, segmentation_map_for_usage = gradcam_functions.compute_segmentation_counts(segmentation_map, grad_cam_plot, edge_plot, total_pixels, config)

  assert torch.equal(segmentation_map, segmentation_map_for_usage), "Segmentation Maps don't match!"
  assert torch.equal(grad_cam_plot.to(dtype=dtype), gradcam_maps), "expected_threshold_maps don't match!"
  assert np.array_equal(expected_edge_maps, edge_maps), "Edge maps don't match!"

  log('==========================================', config)
  log(f'=== GradCAM preload_segmentation_maps  ===', config)
  log('==========================================', config)
  pair_paths = [['../example_data/1_leftImg8bit.png', '../example_data/1_gtFine_labelIds.png', '../example_data/1_edge_gtFine_labelIds.png'], ['../example_data/2_leftImg8bit.png', '../example_data/2_gtFine_labelIds.png', '../example_data/2_edge_gtFine_labelIds.png']]

  expected_map_1 = image_functions.open_image('../example_data/1_gtFine_labelIds.png')
  expected_map_2 = image_functions.open_image('../example_data/2_gtFine_labelIds.png')

  maps, total_segmentation_summary = gradcam_functions.preload_segmentation_maps(pair_paths, config)
  expected_total_count = np.prod(maps[0][0].shape) * 2

  assert np.array_equal(expected_map_1.T[0, :, :], maps[0][0]), "First Segmentation Map Does Not Equal!"
  assert np.array_equal(expected_map_2.T[0, :, :], maps[1][0]), "Second Segmentation Map Does Not Equal!"
  assert expected_total_count == sum(total_segmentation_summary.values()), "The total count is invalid!"

  log('==========================================', config)
  log(f'=== GradCAM Process Counts (Threshold) ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "threshold"
  idex = 1
  grad_type = 'Base'
  epoch_count = 1
  heatmap = grad_cam_plot

  frame = torch.tensor(np.array([[255, 0, 255], [150, 150, 255]]))
  pair_paths = [['../example_data/1_leftImg8bit.png', '../example_data/1_gtFine_labelIds.png', '../example_data/1_edge_gtFine_labelIds.png'], ['../example_data/2_leftImg8bit.png', '../example_data/2_gtFine_labelIds.png', '../example_data/2_edge_gtFine_labelIds.png']]
  segmentation_maps, total_segmentation_summary = gradcam_functions.preload_segmentation_maps(pair_paths, config)

  total_threshold_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_edge_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_segmentation_maps = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)

  expected_zeros = torch.tensor(np.array([[0, 0, 0], [0, 0, 0]]))
  expected_threshold_map = expected_threshold_maps
  expected_edge_map = torch.tensor(expected_edge_maps)

  total_threshold_maps, total_edge_maps, total_segmentation_maps = gradcam_functions.process_counts(total_threshold_counts, total_edge_counts, total_segmentation_maps, idex, grad_type, epoch_count, config, heatmap, edge_plot, frame, segmentation_map, pair_paths, total_segmentation_summary, 2)
  assert torch.equal(expected_zeros, total_threshold_maps[0]), "Zero total threshold maps dont equal!"
  assert torch.equal(expected_zeros, total_edge_maps[0]), "Zero total edge maps dont equal!"
  assert torch.equal(expected_zeros, total_segmentation_maps[0]), "Zero total segmentation maps dont equal!"

  assert torch.equal(expected_threshold_map, total_threshold_maps[1]), "Total threshold maps dont equal!"
  assert torch.equal(expected_edge_map, total_edge_maps[1]), "Total edge maps dont equal!"
  assert torch.equal(expected_zeros, total_segmentation_maps[1]), "Zero total segmentation maps dont equal!"

  log('==========================================', config)
  log(f'=== GradCAM Process Counts (Absolute)  ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "absolute"

  total_threshold_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_edge_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_segmentation_maps = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)

  total_threshold_maps, total_edge_maps, total_segmentation_maps = gradcam_functions.process_counts(total_threshold_counts, total_edge_counts, total_segmentation_maps, idex, grad_type, epoch_count, config, heatmap, edge_plot, frame, segmentation_map, pair_paths, total_segmentation_summary, 2)

  assert torch.equal(expected_zeros, total_threshold_maps[0]), "Zero total threshold maps dont equal!"
  assert torch.equal(expected_zeros, total_edge_maps[0]), "Zero total edge maps dont equal!"
  assert torch.equal(expected_zeros, total_segmentation_maps[0]), "Zero total segmentation maps dont equal!"

  expected_threshold_map = torch.tensor(np.array([[0.0000, 3.9004, 10.5000], [0.2988, 0.4199, 2.3984]])).to(dev, dtype=dtype)
  expected_edge_map = torch.tensor(np.array([[0, 15, 0], [3, 3, 0]]))
  expected_segmentation_map = torch.tensor(np.array([[15, 15, 15], [3, 3, 3]]))

  grad_cam_plot = helper_functions.convert_to_tensor(grad_cam_plot, dev, dtype)
  assert torch.equal(grad_cam_plot, total_threshold_maps[1]), "Total threshold maps dont equal!"
  assert torch.equal(expected_edge_map, total_edge_maps[1]), "Total edge maps dont equal!"
  assert torch.equal(expected_segmentation_map, total_segmentation_maps[1]), "Zero total segmentation maps dont equal!"

  log('==========================================', config)
  log(f'=== GradCAM Final Counts (Threshold)   ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "threshold"

  total_threshold_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_edge_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_segmentation_maps = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)

  expected_final_total_threshold_counts = {'unlabeled': 10, 'ego vehicle': 0, 'rectification border': 0, 'out of roi': 1, 'static': 0, 'dynamic': 0, 'ground': 0, 'road': 0, 'sidewalk': 0, 'parking': 0, 'rail track': 0, 'building': 0, 'wall': 0, 'fence': 0, 'guard rail': 0, 'bridge': 1, 'tunnel': 0, 'pole': 0, 'polegroup': 0, 'traffic light': 0, 'traffic sign': 0, 'vegetation': 0, 'terrain': 0, 'sky': 0, 'person': 0, 'rider': 0, 'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0, 'license plate': 0}
  expected_final_total_edge_counts = {'unlabeled': 9, 'ego vehicle': 0, 'rectification border': 0, 'out of roi': 2, 'static': 0, 'dynamic': 0, 'ground': 0, 'road': 0, 'sidewalk': 0, 'parking': 0, 'rail track': 0, 'building': 0, 'wall': 0, 'fence': 0, 'guard rail': 0, 'bridge': 1, 'tunnel': 0, 'pole': 0, 'polegroup': 0, 'traffic light': 0, 'traffic sign': 0, 'vegetation': 0, 'terrain': 0, 'sky': 0, 'person': 0, 'rider': 0, 'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0, 'license plate': 0}

  total_threshold_counts, total_edge_counts, total_segmentation_maps = gradcam_functions.process_counts(total_threshold_counts, total_edge_counts, total_segmentation_maps, idex, grad_type, epoch_count, config, heatmap, edge_plot, frame, segmentation_map, pair_paths, total_segmentation_summary, 2)
  final_total_threshold_counts, final_total_edge_counts = gradcam_functions.compute_final_counts(total_threshold_counts, total_edge_counts)

  assert expected_final_total_threshold_counts == final_total_threshold_counts, "Final total counts don't match!"
  assert expected_final_total_edge_counts == final_total_edge_counts, "Final total edge counts dont match!"

  log('==========================================', config)
  log(f'=== GradCAM Final Counts (Absolute)    ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "absolute"

  total_threshold_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_edge_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_segmentation_maps = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)

  expected_final_total_edge_counts = {'unlabeled': 9, 'ego vehicle': 0, 'rectification border': 0, 'out of roi': 2, 'static': 0, 'dynamic': 0, 'ground': 0, 'road': 0, 'sidewalk': 0, 'parking': 0, 'rail track': 0, 'building': 0, 'wall': 0, 'fence': 0, 'guard rail': 0, 'bridge': 1, 'tunnel': 0, 'pole': 0, 'polegroup': 0, 'traffic light': 0, 'traffic sign': 0, 'vegetation': 0, 'terrain': 0, 'sky': 0, 'person': 0, 'rider': 0, 'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0, 'license plate': 0}
  expected_final_total_absolute_counts = {'unlabeled': 0.0, 'ego vehicle': 0.0, 'rectification border': 0.0, 'out of roi': 1.0396000146865845, 'static': 0.0, 'dynamic': 0.0, 'ground': 0.0, 'road': 0.0, 'sidewalk': 0.0, 'parking': 0.0, 'rail track': 0.0, 'building': 0.0, 'wall': 0.0, 'fence': 0.0, 'guard rail': 0.0, 'bridge': 0.9599999785423279, 'tunnel': 0, 'pole': 0, 'polegroup': 0, 'traffic light': 0, 'traffic sign': 0, 'vegetation': 0, 'terrain': 0, 'sky': 0, 'person': 0, 'rider': 0, 'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0, 'license plate': 0}

  total_threshold_counts, total_edge_counts, total_segmentation_maps = gradcam_functions.process_counts(total_threshold_counts, total_edge_counts, total_segmentation_maps, idex, grad_type, epoch_count, config, heatmap, edge_plot, frame, segmentation_map, pair_paths, total_segmentation_summary, 2)
  final_total_edge_counts = gradcam_functions.compute_final_edge_counts(total_edge_counts)
  final_total_absolute_counts = gradcam_functions.compute_final_absolute_counts(total_threshold_counts, total_segmentation_maps, dev, dtype)

  assert expected_final_total_edge_counts == final_total_edge_counts, "Final total edge counts dont match!"
  assert expected_final_total_absolute_counts == final_total_absolute_counts, "Final total absolute counts don't match!"

  log('==========================================', config)
  log(f'=== GradCAM Percentages (Threshold)    ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "threshold"

  total_threshold_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_edge_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_segmentation_maps = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)

  total_threshold_counts, total_edge_counts, total_segmentation_maps = gradcam_functions.process_counts(total_threshold_counts, total_edge_counts, total_segmentation_maps, idex, grad_type, epoch_count, config, heatmap, edge_plot, frame, segmentation_map, pair_paths, total_segmentation_summary, 2)
  total_threshold_counts, final_total_edge_counts = gradcam_functions.compute_final_counts(total_threshold_counts, total_edge_counts)

  expected_total_segmentation_summary = {'unlabeled': 0, 'ego vehicle': 294, 'rectification border': 0, 'out of roi': 4, 'static': 127, 'dynamic': 2, 'ground': 6, 'road': 20854, 'sidewalk': 1815, 'parking': 523, 'rail track': 2, 'building': 8, 'wall': 5, 'fence': 368, 'guard rail': 362, 'bridge': 5, 'tunnel': 12, 'pole': 108, 'polegroup': 8, 'traffic light': 5, 'traffic sign': 15, 'vegetation': 2970, 'terrain': 2805, 'sky': 0, 'person': 18, 'rider': 19, 'car': 216, 'truck': 9, 'bus': 109, 'caravan': 37, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 14, 'license plate': 0}

  segmentation_maps, total_segmentation_summary = gradcam_functions.preload_segmentation_maps(pair_paths, config)

  expected_sum_of_segmentation_summary_count = np.prod(segmentation_maps.shape) #30720
  actual_summary_count = np.array(list(total_segmentation_summary.values())).sum()

  assert expected_sum_of_segmentation_summary_count == actual_summary_count, "Summary totals don't match!"
  assert expected_total_segmentation_summary == total_segmentation_summary, "Total segmentation summary does not match!"
  assert expected_sum_of_segmentation_summary_count ==  sum(expected_total_segmentation_summary.values()), "Summary totals 2 don't match!"

  expected_threshold_percentage = {'unlabeled': 0, 'ego vehicle': 0.0, 'rectification border': 0, 'out of roi': 25.0, 'static': 0.0, 'dynamic': 0.0, 'ground': 0.0, 'road': 0.0, 'sidewalk': 0.0, 'parking': 0.0, 'rail track': 0.0, 'building': 0.0, 'wall': 0.0, 'fence': 0.0, 'guard rail': 0.0, 'bridge': 20.0, 'tunnel': 0.0, 'pole': 0.0, 'polegroup': 0.0, 'traffic light': 0.0, 'traffic sign': 0.0, 'vegetation': 0.0, 'terrain': 0.0, 'sky': 0, 'person': 0.0, 'rider': 0.0, 'car': 0.0, 'truck': 0.0, 'bus': 0.0, 'caravan': 0.0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0.0, 'license plate': 0}
  total_threshold_percentage = gradcam_functions.compute_segmentation_percentages(total_segmentation_summary, total_threshold_counts)
  assert expected_threshold_percentage == total_threshold_percentage, "total_threshold_percentage is wrong!"

  # Check edge percentages
  expected_threshold_percentage = {'unlabeled': 0, 'ego vehicle': 0.0, 'rectification border': 0, 'out of roi': 50.0, 'static': 0.0, 'dynamic': 0.0, 'ground': 0.0, 'road': 0.0, 'sidewalk': 0.0, 'parking': 0.0, 'rail track': 0.0, 'building': 0.0, 'wall': 0.0, 'fence': 0.0, 'guard rail': 0.0, 'bridge': 20.0, 'tunnel': 0.0, 'pole': 0.0, 'polegroup': 0.0, 'traffic light': 0.0, 'traffic sign': 0.0, 'vegetation': 0.0, 'terrain': 0.0, 'sky': 0, 'person': 0.0, 'rider': 0.0, 'car': 0.0, 'truck': 0.0, 'bus': 0.0, 'caravan': 0.0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0.0, 'license plate': 0}
  total_threshold_percentage = gradcam_functions.compute_segmentation_percentages(total_segmentation_summary, final_total_edge_counts)
  assert expected_threshold_percentage == total_threshold_percentage, "total_threshold_percentage for edges is wrong!"

  log('==========================================', config)
  log(f'=== GradCAM Percentages (Absolute)     ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "absolute"

  total_threshold_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_edge_counts = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)
  total_segmentation_maps = torch.zeros([2, frame.shape[0], frame.shape[1]], dtype=dtype)

  total_threshold_counts, total_edge_counts, total_segmentation_maps = gradcam_functions.process_counts(total_threshold_counts, total_edge_counts, total_segmentation_maps, idex, grad_type, epoch_count, config, heatmap, edge_plot, frame, segmentation_map, pair_paths, total_segmentation_summary, 2)
  total_threshold_counts = gradcam_functions.compute_final_absolute_counts(total_threshold_counts, total_segmentation_maps, dev, dtype)
  final_total_edge_counts = gradcam_functions.compute_final_edge_counts(total_edge_counts)

  expected_total_segmentation_summary = {'unlabeled': 0, 'ego vehicle': 294, 'rectification border': 0, 'out of roi': 4, 'static': 127, 'dynamic': 2, 'ground': 6, 'road': 20854, 'sidewalk': 1815, 'parking': 523, 'rail track': 2, 'building': 8, 'wall': 5, 'fence': 368, 'guard rail': 362, 'bridge': 5, 'tunnel': 12, 'pole': 108, 'polegroup': 8, 'traffic light': 5, 'traffic sign': 15, 'vegetation': 2970, 'terrain': 2805, 'sky': 0, 'person': 18, 'rider': 19, 'car': 216, 'truck': 9, 'bus': 109, 'caravan': 37, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 14, 'license plate': 0}

  segmentation_maps, total_segmentation_summary = gradcam_functions.preload_segmentation_maps(pair_paths, config)

  expected_sum_of_segmentation_summary_count = np.prod(segmentation_maps.shape) #30720
  actual_summary_count = np.array(list(total_segmentation_summary.values())).sum()

  assert expected_sum_of_segmentation_summary_count == actual_summary_count, "Summary totals don't match!"
  assert expected_total_segmentation_summary == total_segmentation_summary, "Total segmentation summary does not match!"
  assert expected_sum_of_segmentation_summary_count ==  sum(expected_total_segmentation_summary.values()), "Summary totals 2 don't match!"

  expected_threshold_percentage = {'unlabeled': 0, 'ego vehicle': 0.0, 'rectification border': 0, 'out of roi': 25.990000367164612, 'static': 0.0, 'dynamic': 0.0, 'ground': 0.0, 'road': 0.0, 'sidewalk': 0.0, 'parking': 0.0, 'rail track': 0.0, 'building': 0.0, 'wall': 0.0, 'fence': 0.0, 'guard rail': 0.0, 'bridge': 19.199999570846558, 'tunnel': 0.0, 'pole': 0.0, 'polegroup': 0.0, 'traffic light': 0.0, 'traffic sign': 0.0, 'vegetation': 0.0, 'terrain': 0.0, 'sky': 0, 'person': 0.0, 'rider': 0.0, 'car': 0.0, 'truck': 0.0, 'bus': 0.0, 'caravan': 0.0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0.0, 'license plate': 0}
  total_threshold_percentage = gradcam_functions.compute_segmentation_percentages(total_segmentation_summary, total_threshold_counts)
  assert expected_threshold_percentage == total_threshold_percentage, "total_threshold_percentage is wrong!"

  # Check edge percentages
  expected_threshold_percentage = {'unlabeled': 0, 'ego vehicle': 0.0, 'rectification border': 0, 'out of roi': 50.0, 'static': 0.0, 'dynamic': 0.0, 'ground': 0.0, 'road': 0.0, 'sidewalk': 0.0, 'parking': 0.0, 'rail track': 0.0, 'building': 0.0, 'wall': 0.0, 'fence': 0.0, 'guard rail': 0.0, 'bridge': 20.0, 'tunnel': 0.0, 'pole': 0.0, 'polegroup': 0.0, 'traffic light': 0.0, 'traffic sign': 0.0, 'vegetation': 0.0, 'terrain': 0.0, 'sky': 0, 'person': 0.0, 'rider': 0.0, 'car': 0.0, 'truck': 0.0, 'bus': 0.0, 'caravan': 0.0, 'trailer': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0.0, 'license plate': 0}
  total_threshold_percentage = gradcam_functions.compute_segmentation_percentages(total_segmentation_summary, final_total_edge_counts)
  assert expected_threshold_percentage == total_threshold_percentage, "total_threshold_percentage for edges is wrong!"

  # log('==========================================', config)
  # log(f'=== Compute GradCAM Result (Threshold) ===', config)
  # log('==========================================', config)
  # config["grad_cam_algo"] = "threshold"
  # compute_grad_cam_results(model, config, dev, pair_paths, meta, raw_frames, segmentation_maps, edge_maps, total_segmentation_summary, in_line=True, epoch_count=i, grad_type='Base') # 'Custom'

  log('==========================================', config)
  log(f'=== Compute GradCAM Result (Absolute)  ===', config)
  log('==========================================', config)
  config["grad_cam_algo"] = "absolute"
  config["model_name"] = "End to End"
  config["add_perturbations"] = False
  config["grad_cam_drop_percentage"] = 0.0 #0.5

  model = torch_models.compile_model(config)
  # optimizer, _loss_func = torch_optimisers.fetch_loss_opt_func(config, model)

  checkpoint = torch.load("./example_meta/end_to_end_example_model.pth", map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['model'], strict=False)

  dtype = helper_functions.compute_model_dtype(config)

  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])
  model.eval()

  print(summary(model))

  label_map_keys_list = gradcam_functions.fetch_keys()

  # Inner batch run:
  dev = helper_functions.fetch_device(config)
  dtype = helper_functions.compute_gradcam_dtype(config)
  batch_size = 1
  total_count = 1
  pbar = tqdm.tqdm(total=total_count)
  grad_type = "Base" #"FullGrad" #"Base" # "FullGrad" # "Custom"
  cam_function = gradcam_functions.fetch_cam_function(config, model, grad_type)

  dataset_string = data_helpers.compute_dataset_string(config)
  dataset, width, height, dim = datasets.extract_options_from(dataset_string)
  dim_shift = (int(height), int(width), int(dim)) # (60, 256, 3)
  roi = [420, 840, 150, 1942]

  # Load test files
  unprocessed_frame = image_functions.open_image("./example_meta/cityscapes/raw_image1.png")
  edge_map = image_functions.open_image("./example_meta/cityscapes/edge_map1.png")
  segmentation_map = image_functions.open_image("./example_meta/cityscapes/segmentation_1.png")

  raw_frame = cooking_functions.process_image(unprocessed_frame, dim_shift=dim_shift, roi=roi, greyscale=False)
  raw_frame = raw_frame # torch.from_numpy(raw_frame).permute(2, 0, 1)
  raw_frames = torch.from_numpy(np.array([raw_frame])).permute(0, 3, 1, 2).to(dev, non_blocking=config["non_blocking"], dtype=dtype) #raw_frame.unsqueeze(0).to(dev, non_blocking=config["non_blocking"], dtype=dtype)

  total_counts = torch.zeros([total_count, raw_frames[0].shape[1], raw_frames[0].shape[2]], dtype=torch.float32)
  total_edge_counts = torch.zeros([total_count, raw_frames[0].shape[1], raw_frames[0].shape[2]], dtype=torch.float32)
  total_segmentation_maps = torch.zeros([total_count, raw_frames[0].shape[1], raw_frames[0].shape[2]], dtype=torch.float32)

  segmentation_maps = torch.from_numpy(np.array([segmentation_map[:, :, 0]])).unsqueeze(0)
  total_segmentation_summary = gradcam_functions.init_empty_count_hash()
  segmentation_counts = np.bincount(segmentation_maps.flatten())

  edge_maps = torch.from_numpy(np.array(edge_map[:, :, 0])).unsqueeze(0)

  for i in range(len(segmentation_counts)):
    total_segmentation_summary[label_map_keys_list[i]] += segmentation_counts[i]

  # segmentation_maps, total_segmentation_summary = gradcam_functions.preload_segmentation_maps(pair_paths, config)

  i = 0
  total_counts, total_edge_counts, total_segmentation_maps = gradcam_functions.inner_batch_cam_loop(config, model, dtype, dev, i, batch_size, pbar, segmentation_maps, total_count, pair_paths, raw_frames, cam_function, edge_maps, total_counts, total_edge_counts, total_segmentation_maps, total_segmentation_summary, grad_type=grad_type)

  assert float(total_counts.sum()) != 0.0, "There are no counts!"

  # TODO: Train? Possibly load a trained model
  # gradcam_functions.compute_grad_cam_results(model, config, dev, pair_paths, meta, raw_frames, segmentation_maps, edge_maps, total_segmentation_summary, in_line=True, epoch_count=i, grad_type='Base') # 'Custom'
  # in_test=True

  print("Done!")


available_datasets = ['cityscapes', 'udacity', 'cookbook', 'carlaimitation']
config = {
  "use_wandb": False,
  "wandb_watch_freq": 1000,
  "wandb_watch_log": "all", # "gradients", "parameters", "all", or None
  "wandb_watch_log_graph": True,
  "wandb_project_name": "random-test-4", # TestModel1-udacity-dmczc
  "wandb_name_prefix": "TEST", #"linear" #"carlaimitation 0.02 sample",
  "Notes": "Large Scale Epoch Test",
  "device_name": "cpu", #"cuda:0" #"cuda:1" #cpu
  "summary_device_name": "cpu",
  "non_blocking": True,
  "pin_memory": False,
  "cuda_spawn": True,
  "purge_cuda_memory": True, # Turn off on shared resources
  "compute_attention": False, # used for vision transformers
  "compute_mean_attention": False,
  "model_name": "End to End", #"deit_tiny_model", #"ViT-H_14", # "ViT-B_32", # "ViT-L_16", # "ViT-L_32", # "ViT-B_16", # "ViT-H_14", # "End to End No Dropout", #"End to End", #"Autonomous Cookbook", #"Net SVF", #"Net HVF", #"TestModel1", #"TestModel2",
  "track_attention_weights": False, # Set to True before running cam analysis
  "average_attn_weights": False,
  "compute_attn_mean": False,
  "model_name": "End to End", #"End to End", #"Autonomous Cookbook", #"Net SVF", #"Net HVF", #"TestModel1", #"TestModel2", #"TestModel3",
  "dataset_name": "cityscapes",
  "datasets": ["cookbook", "udacity", "cityscapes"], # "fromgames", "carlaimitation"
  "randomise_weights": "xavier_uniform", # "uniform", "normal", "ones", "zeros", "eye", "dirac", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_uniform", "orthogonal", "sparse"
  "roi": {
    "cookbook": [76, 135, 0, 255],
    "cityscapes": [420, 840, 150, 1942],
    "cityscapesvideo": [420, 840, 150, 1942],
    "carlaimitation": None,
    "udacity": [160, 470, 0, 640],
    "fromgames": [300, 720, 150, 1764]
  }, #roi y -> x
  "convert_to_greyscale": False,
  "cook_only": False,
  "compile_only": False, # Only compile the model
  "cook_data": True,
  "compute_other_test_loss": True,
  "benchmark": False,
  "load_model": False,
  "grad_cam_threshold": 0.5, # 0.2, # 0.01,
  "draw_grad_cam": False,
  "save_grad_cam": False,
  "save_grad_cam_plot": False, # efficiency issues here
  "grad_cam_plot": "cv2", #"seaborn", #"matplotlib"
  "generate_grad_cam_video": False,
  "grad_cam_dataset": "cityscapes", #"udacity", #"cityscapesvideo", # TODO: Others # This is used for the video TODO: Rename
  "compute_grad_cam_results": True,
  "keep_gradcam_frames": True,
  "grad_cam_in_memory": True,
  "grad_cam_drop_percentage": 0.0, #1.0, # Separate data process
  "grad_cam_result_dataset": "cityscapes", #"cityscapes", #"fromgames", # TODO: Turn into an array
  "grad_cam_epochs": [0, 1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], # can be None
  "preload_segmentation_maps": True,
  "log_to_file": False,
  # "run_name": "Net SVF-udacity-zdrvx",
  # "run_name": "TestModel1-udacity",
  "warmup_data": False,
  "warmup_cycles": 2,
  "input_size": [3, 256, 60], #[3, 512, 120], #[3, 256, 60], #(3, 512, 120), #(3, 64, 30), #(3, 128, 30), #(3, 255, 59), #(3, 64, 40), # (3, 256, 59), #(3, 256, 144), #(3, 256, 59) #(3, 48, 64) #(3, 100, 50) #(3, 256, 144) #(1, 28, 28) # used for the summary and pre-process,
  "lr": 0.001, # 0.0001 # 0.1 # 0.02 # learning rate,
  "beta_1": 0.9,
  "beta_2": 0.999,
  "epsilon": 1e-03, #1e-08, cant be too small for float16
  "weight_decay": 1e-03, #1e-03, #0,
  "amsgrad": True, # Default False
  "initialisation_number": 1, #10,
  "initial_epochs": 1, #5,
  "epochs": 1, #250, #1000, #250, #150, #250, #500, #10, #5, #124, #250 #15 # how many epochs to train for,
  "epoch_torcheck": 200, # 3,
  "torcheck": {
    "module_name": "sanity_check",
    "changing": None, # True
    "output_range": None, # (-1.0, 1.0),
    "check_nan": True,
    "check_inf": True
  },
  "sanity_check": True,
  "normalise_output": False,
  "sigmoid": True,
  "momentum": 0.8,
  "number_of_outputs": 1,
  "output_key": "Steering",
  "output_tanh": False,
  "model_save_path": "/mnt/excelsior/trained_models",
  "base_data_path":  "/data/data",
  "base_cook_data_path": "/data/data/cooking",
  "data_drop_precision": 2,
  "accuracy_precision": 2,
  "zero_drop_percentage": 0.6, #0.8, #0.72, # 0.95
  "calculate_zero_drop_percentage_even": True,
  "drop_invalid_steering_angles": False,
  "sample_percentage": 1,
  "train_val_test_split": [0.7, 0.2, 0.1],
  "split_by_temporal_lines": True,
  "combine_val_test": False,
  "horizontal_flip_data": True,
  "win_drive": "G",
  "loss_name": "mse_loss_func",
  "opt_name": "Adam",
  "mixed_precision": False,
  "bfloat16": False,
  "cache_enabled": True,
  "clip_grad_norm": False,
  "grad_max_norm": 0.1,
  "dataset": {
    "available_24": {
      "Net SVF": {
        "train": { "batch_size": 200, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 200, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net HVF": {
        "train": { "batch_size": 400, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 400, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 400, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "End to End": {
        "train": { "batch_size": 1000, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 1000, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 1000, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Autonomous Cookbook": {
        "train": { "batch_size": 1000, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 1000, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 1000, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel1": {
        "train": { "batch_size": 1000, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 1000, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 1000, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel2": {
        "train": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 25, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": None }
      }
    },
    "available_12": {
      "Net SVF": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net HVF": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "End to End": {
        "train": { "batch_size": 500, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 500, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 500, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Autonomous Cookbook": {
        "train": { "batch_size": 200, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel1": {
        "train": { "batch_size": 200, "shuffle": True, "num_workers": 3, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel2": {
        "train": { "batch_size": 200, "shuffle": True, "num_workers": 3, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      }
    },
    "available_6": {
      "Net SVF": {
        "train": { "batch_size": 15, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 15, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 15, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net HVF": {
        "train": { "batch_size": 20, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 20, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 20, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "End to End": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Autonomous Cookbook": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel1": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 300, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel2": {
        "train": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 25, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": None }
      }
    },
    "available_4": {
      "Net SVF": {
        "train": { "batch_size": 10, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 10, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 10, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net HVF": {
        "train": { "batch_size": 20, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 20, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 20, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "End to End": {
        "train": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 25, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 25, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Autonomous Cookbook": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel1": {
        "train": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 25, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 25, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel2": {
        "train": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 25, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 25, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": None }
      }
    }
  },
  "carla_benchmark_v1": {
    "params": {
      "port": 2000,
      "suite": "FullTown01-v3", #'town1', # 'ALL_SUITES'
      "big_cam": "store_True",
      "seed": 2021,
      "autopilot": False,
      "show": False,
      "resume": "store_True",
      "max_run": 1 #3,
    },
    "agent_args": {
      "camera_args": {
        "fixed_offset": 4.0,
        "fov": 90,
        "h": 160,
        "w": 384,
        "world_y": 1.4
      },
    "pid": {
      "1" : { "Kp": 0.5, "Ki": 0.20, "Kd":0.0 },
      "2" : { "Kp": 0.7, "Ki": 0.10, "Kd":0.0 },
      "3" : { "Kp": 1.0, "Ki": 0.10, "Kd":0.0 },
      "4" : { "Kp": 1.0, "Ki": 0.50, "Kd":0.0 }
    },
      "steer_points": { "1": 4, "2": 3, "3": 2, "4": 2 }
    }
  }
}

helper_functions.clear_gpu(torch, config)

if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass
  run_gradcam_tests(config)
