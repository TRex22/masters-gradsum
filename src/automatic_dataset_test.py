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

def print_meta_summary(config, raw_meta, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train):
  log(f'\nRaw-Meta Number of normal_meta data points: { normal_meta.shape[0] }', config)
  log(f'Raw-Meta Number of swerve_meta data points: { swerve_meta.shape[0] }', config)

  swerve_ratio = swerve_meta.shape[0] / normal_meta.shape[0]
  log(f'swerve_ratio: {swerve_ratio}', config)

  log(f'\n\nTotal Raw-Meta: {len(raw_meta)}', config)
  log(f'Total Meta: {len(meta)}', config)
  log(f'Total sum of normal and swerve: {len(normal_meta) + len(swerve_meta)} \n\n', config)

  log(f"Drop percentage: {config['zero_drop_percentage']}", config)
  log(f'Expected Drop Result Count: {expected_number_of_dropped_data_points(config, normal_meta, swerve_meta)}', config)
  log(f'Dropped meta size: {len(dropped_meta)}\n\n', config)

  log(f'Split: { config["train_val_test_split"] } (not used for Cityscapes)', config)

  train_normal_meta = train[train['Is Swerve'] == False]
  train_swerve_meta = train[train['Is Swerve'] == True]
  train_swerve_ratio = train_swerve_meta.shape[0] / train_normal_meta.shape[0]
  log(f'Number of train data points (before drop): { train.shape[0] }, straight: {train_normal_meta.shape[0]}, swerve: {train_swerve_meta.shape[0]}, ratio: {train_swerve_ratio}', config)

  dropped_train_normal_meta = dropped_train[dropped_train['Is Swerve'] == False]
  dropped_train_swerve_meta = dropped_train[dropped_train['Is Swerve'] == True]
  dropped_train_swerve_ratio = dropped_train_swerve_meta.shape[0] / dropped_train_normal_meta.shape[0]
  log(f'Number of dropped_train data points (after drop): { dropped_train.shape[0] }, straight: {dropped_train_normal_meta.shape[0]}, swerve: {dropped_train_swerve_meta.shape[0]}, ratio: {dropped_train_swerve_ratio}', config)

  val_normal_meta = val[val['Is Swerve'] == False]
  val_swerve_meta = val[val['Is Swerve'] == True]
  val_swerve_ratio = val_swerve_meta.shape[0] / val_normal_meta.shape[0]
  log(f'Number of val data points (after drop): { val.shape[0] }, straight: {val_normal_meta.shape[0]}, swerve: {val_swerve_meta.shape[0]}, ratio: {val_swerve_ratio}', config)

  test_normal_meta = test[test['Is Swerve'] == False]
  test_swerve_meta = test[test['Is Swerve'] == True]
  test_swerve_ratio = test_swerve_meta.shape[0] / test_normal_meta.shape[0]
  log(f'Number of test data points (after drop): { test.shape[0] }, straight: {test_normal_meta.shape[0]}, swerve: {test_swerve_meta.shape[0]}, ratio: {test_swerve_ratio}', config)

def expected_number_of_dropped_data_points(config, normal_meta, swerve_meta):
  percentage_to_keep = (1.0 - config["zero_drop_percentage"])
  zero_angle_values_to_keep = normal_meta.sample(frac=percentage_to_keep)

  return math.ceil(len(zero_angle_values_to_keep)) + len(swerve_meta)

def assert_dropped_meta(dropped_meta, expected_data_points, message):
  meta_length = len(dropped_meta)
  difference = abs(meta_length - expected_data_points)

  if difference > 1.0: # We ignore a difference of 1 as sometimes there is an extra image and sometimes not
    assert len(dropped_meta) == expected_data_points, message # Will fail

def double_check_meta_counts(config, raw_meta, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train):
  assert len(raw_meta) == len(meta), "Length of raw_meta and meta do not match. Data may have been lost."
  assert len(raw_meta) == (len(normal_meta) + len(swerve_meta)), "Length of raw_meta and normal + swerve do not match. Data may have been lost."
  assert len(raw_meta) == (train.shape[0] + val.shape[0] + test.shape[0]), "Length of raw_meta and split metas do not match. Data may have been lost."

  # Drop percentage is correct
  expected_data_points = expected_number_of_dropped_data_points(config, normal_meta, swerve_meta)
  assert_dropped_meta(dropped_meta, expected_data_points, "Length of dropped meta does not equal the expected length.")

  # Check that all data is used for cityscapes
  if config["dataset_name"] == "cityscapes":
    expected_train_size = np.array(meta[meta['Split Folder'] == 'train']["Cooked Edge Path"]).shape[0]
    expected_val_size = np.array(meta[meta['Split Folder'] == 'val']["Cooked Edge Path"]).shape[0]
    expected_test_size = np.array(meta[meta['Split Folder'] == 'test']["Cooked Edge Path"]).shape[0]

    dropped_train_normal_meta = dropped_train[dropped_train['Is Swerve'] == False]
    dropped_train_swerve_meta = dropped_train[dropped_train['Is Swerve'] == True]
    expected_dropped_train_size = expected_number_of_dropped_data_points(config, train[train['Is Swerve'] == False], train[train['Is Swerve'] == True])

    assert len(train) == expected_train_size, "Cityscapes Train size is invalid!"
    assert len(val) == expected_val_size, "Cityscapes Train size is invalid!"
    assert len(test) == expected_test_size, "Cityscapes Train size is invalid!"

    assert_dropped_meta(dropped_train, expected_dropped_train_size, "Cityscapes Dropped Train size is invalid!")
  else: # TODO
    # Expected split counts
    expected_train_data_size = config["train_val_test_split"][0] * len(raw_meta)
    assert expected_train_data_size == train.shape[0], "Number of train data points is incorrect!"

    expected_val_data_size = config["train_val_test_split"][1] * len(raw_meta)
    assert expected_val_data_size == val.shape[0], "Number of val data points is incorrect!"

    expected_test_data_size = config["train_val_test_split"][2] * len(raw_meta)
    assert expected_test_data_size == test.shape[0], "Number of test data points is incorrect!"

def count_swerve_accuracy(raw_meta_cityscapes, raw_meta_udacity, raw_meta_cookbook, accuracy):
  log('==========================================', config)
  log(f'Check accuracy of: {accuracy}', config)

  log('==========================================', config)
  log('Cityscapes:', config)
  raw_meta_cityscapes['Is Swerve'] = raw_meta_cityscapes.apply(
    lambda r: data_functions.is_swerve(r, accuracy=accuracy), axis=1
  )
  normal_meta_cityscapes = raw_meta_cityscapes[raw_meta_cityscapes['Is Swerve'] == False]
  swerve_meta_cityscapes = raw_meta_cityscapes[raw_meta_cityscapes['Is Swerve'] == True]
  log(f'normal_meta_cityscapes: {len(normal_meta_cityscapes)}', config)
  log(f'swerve_meta_cityscapes: {len(swerve_meta_cityscapes)}', config)

  log('==========================================', config)
  log('Udacity:', config)
  raw_meta_udacity['Is Swerve'] = raw_meta_udacity.apply(
    lambda r: data_functions.is_swerve(r, accuracy=accuracy), axis=1
  )
  normal_meta_udacity = raw_meta_udacity[raw_meta_udacity['Is Swerve'] == False]
  swerve_meta_udacity = raw_meta_udacity[raw_meta_udacity['Is Swerve'] == True]
  log(f'normal_meta_udacity: {len(normal_meta_udacity)}', config)
  log(f'swerve_meta_udacity: {len(swerve_meta_udacity)}', config)

  log('==========================================', config)
  log('Cookbook', config)
  raw_meta_cookbook['Is Swerve'] = raw_meta_cookbook.apply(
    lambda r: data_functions.is_swerve(r, accuracy=accuracy), axis=1
  )
  normal_meta_cookbook = raw_meta_cookbook[raw_meta_cookbook['Is Swerve'] == False]
  swerve_meta_cookbook = raw_meta_cookbook[raw_meta_cookbook['Is Swerve'] == True]
  log(f'normal_meta_cookbook: {len(normal_meta_cookbook)}', config)
  log(f'swerve_meta_cookbook: {len(swerve_meta_cookbook)}', config)
  log('==========================================', config)


def run_data_test(config):
  log('==========================================', config)
  log(f'=== Data Tests ===', config)
  log('==========================================', config)

  ignore_value = 'Is Swerve'

  log('Load Cityscapes meta ...', config)
  config["dataset_name"] = "cityscapes"
  raw_meta_cityscapes = helper_functions.open_dataframe("./example_meta/cityscapes/raw_meta.csv")
  meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

  dropped_meta = data_functions.drop_dataframe_if_zero_angle(meta, percent=config["zero_drop_percentage"], precision=config["data_drop_precision"], ignore_value=ignore_value)

  normal_meta = raw_meta_cityscapes[raw_meta_cityscapes['Is Swerve'] == False].sample(frac=1)
  swerve_meta = raw_meta_cityscapes[raw_meta_cityscapes['Is Swerve'] == True].sample(frac=1)

  train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)
  dropped_train = data_functions.drop_dataframe_if_zero_angle(train, percent=config["zero_drop_percentage"], precision=config["data_drop_precision"], ignore_value=ignore_value)

  # Print meta details
  print_meta_summary(config, raw_meta_cityscapes, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train)
  double_check_meta_counts(config, raw_meta_cityscapes, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train)

  # Test data doubling via reflection

  log('\n\nLoad Udacity meta ...', config)
  config["dataset_name"] = "udacity"
  raw_meta_udacity = helper_functions.open_dataframe("./example_meta/udacity/raw_meta.csv")
  meta = helper_functions.open_dataframe("./example_meta/udacity/meta.csv")

  dropped_meta = data_functions.drop_dataframe_if_zero_angle(meta, percent=config["zero_drop_percentage"], precision=config["data_drop_precision"], ignore_value=ignore_value)

  normal_meta = raw_meta_udacity[raw_meta_udacity['Is Swerve'] == False].sample(frac=1)
  swerve_meta = raw_meta_udacity[raw_meta_udacity['Is Swerve'] == True].sample(frac=1)

  train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)
  dropped_train = data_functions.drop_dataframe_if_zero_angle(train, percent=config["zero_drop_percentage"], precision=config["data_drop_precision"], ignore_value=ignore_value)

  # Print meta details
  print_meta_summary(config, raw_meta_udacity, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train)
  double_check_meta_counts(config, raw_meta_udacity, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train)

  log('\n\nLoad cookbook meta ...', config)
  config["dataset_name"] = "cookbook"
  raw_meta_cookbook = helper_functions.open_dataframe("./example_meta/cookbook/raw_meta.csv")
  meta = helper_functions.open_dataframe("./example_meta/cookbook/meta.csv")

  dropped_meta = data_functions.drop_dataframe_if_zero_angle(meta, percent=config["zero_drop_percentage"], precision=config["data_drop_precision"], ignore_value=ignore_value)

  normal_meta = raw_meta_cookbook[raw_meta_cookbook['Is Swerve'] == False].sample(frac=1)
  swerve_meta = raw_meta_cookbook[raw_meta_cookbook['Is Swerve'] == True].sample(frac=1)

  train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)
  dropped_train = data_functions.drop_dataframe_if_zero_angle(train, percent=config["zero_drop_percentage"], precision=config["data_drop_precision"], ignore_value=ignore_value)

  # Print meta details
  print_meta_summary(config, raw_meta_cookbook, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train)
  # double_check_meta_counts(config, raw_meta_cookbook, meta, dropped_meta, normal_meta, swerve_meta, train, val, test, dropped_train)

  log('\n\nDataset Stats', config)
  count_swerve_accuracy(raw_meta_cityscapes, raw_meta_udacity, raw_meta_cookbook, 1)
  count_swerve_accuracy(raw_meta_cityscapes, raw_meta_udacity, raw_meta_cookbook, 2)
  count_swerve_accuracy(raw_meta_cityscapes, raw_meta_udacity, raw_meta_cookbook, 3)


available_datasets = ['cityscapes', 'udacity', 'cookbook', 'carlaimitation']
config = {
  "use_wandb": False,
  "wandb_watch_freq": 1000,
  "wandb_watch_log": "all", # "gradients", "parameters", "all", or None
  "wandb_watch_log_graph": True,
  "wandb_project_name": "random-test-4", # TestModel1-udacity-dmczc
  "wandb_name_prefix": "Mixed Precision", #"linear" #"carlaimitation 0.02 sample",
  "Notes": "Large Scale Epoch Test",
  "device_name": "cuda:0", #"cuda:0" #"cuda:1" #cpu
  "summary_device_name": "cuda",
  "non_blocking": True,
  "pin_memory": False,
  "cuda_spawn": True,
  "purge_cuda_memory": True, # Turn off on shared resources
  "model_name": "TestModel1", #"End to End", #"Autonomous Cookbook", #"Net SVF", #"Net HVF", #"TestModel1", #"TestModel2", #"TestModel3",
  #"dataset_name": "cityscapes", #'carlaimitation', #'cookbook', #'cookbook', #'udacity', #'cityscapes', #'fromgames', #'cityscapes_pytorch'
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
  "draw_grad_cam": False,
  "save_grad_cam": False,
  "save_grad_cam_plot": False, # efficiency issues here
  "grad_cam_plot": "cv2", #"seaborn", #"matplotlib"
  "generate_grad_cam_video": False,
  "grad_cam_dataset": "udacity", #"cityscapesvideo", # TODO: Others # This is used for the video TODO: Rename
  "compute_grad_cam_results": True,
  "keep_gradcam_frames": True,
  "grad_cam_in_memory": True,
  "grad_cam_drop_percentage": 1.0, # Separate data process
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
  "momentum": 0.8,
  "number_of_outputs": 1,
  "output_key": "Steering",
  "output_tanh": True,
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
  "mixed_precision": True,
  "bfloat16": True,
  "cache_enabled": True,
  "clip_grad_norm": True,
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
  run_data_test(config)
