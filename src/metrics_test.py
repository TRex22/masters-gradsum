import time
from datetime import datetime
import tqdm
import math
import os
import json
import random
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

def run_metrics_test(config):
  log('==========================================', config)
  log(f'=== Metrics Tests ===', config)
  log('==========================================', config)
  dev = torch.device("cpu")
  dtype = torch.float32

  mse_loss_func = torch.nn.MSELoss()

  expected_mse = 0.3913875

  expected_steering_count = 3.0
  expected_steering_mse = 0.4493666666666667

  expected_straight_count = 3.0
  expected_straight_mse = 0.3334083333333333

  # Expected error margin
  expected_autonomy_sum_of_interventions = 3.0
  expected_autonomy = 50

  log('=== Pair Array Tests ===', config)
  # Pair batch of [[], []]
  expected_batch = torch.tensor(np.array([1.0, 0.0, 0.5, 0.048, 0.00, -0.23]))

  # [tensor(True), tensor(False), tensor(True), tensor(False), tensor(False), tensor(True)]
  is_swerve_array = []
  for i in range(len(expected_batch)):
    is_swerve_array.append(data_functions.is_swerve_angle(expected_batch[i]))

  yb = torch.tensor(np.array([0.0, 1.0, 0.5, 0.033, -0.00, 0.36]))
  pairs = [expected_batch, yb]
  batch_count = (6)

  expected_computed_mse = mse_loss_func(yb, expected_batch)
  assert expected_computed_mse == expected_mse, "expected_computed_mse does not match expected_mse"

  losses = [expected_computed_mse]

  # Autonomy metric
  # Weighted MSE
  log_outputs = {}
  sequence_name = "train"

  log_outputs = torch_trainer.compute_loss_functions(log_outputs, pairs, losses, batch_count, "train", dev, dtype, should_convert_to_tensor=False)
  autonomy = data_functions.compute_final_autonomy(log_outputs["autonomy_sum_of_interventions_train"], len(yb))

  assert expected_mse == log_outputs["selected_train_loss"], f'selected train loss does not Match! Expected: {expected_mse} Actual: {log_outputs["selected_train_loss"]}'
  assert expected_mse == log_outputs["train_mse_loss_func"], f'train mse loss func does not Match! Expected: {expected_mse} Actual: {log_outputs["train_mse_loss_func"]}'
  # assert expected_mse == log_outputs["mse_weighted_train"], f'mse_weighted_train does not Match! Expected: {expected_mse} Actual: {log_outputs["mse_weighted_train"]}'

  assert expected_steering_count == log_outputs["steering_count_train"], f'steering_count_train does not Match! Expected: {expected_steering_count} Actual: {log_outputs["steering_count_train"]}'
  assert expected_straight_count == log_outputs["straight_count_train"], f'straight_count_train does not Match! Expected: {expected_straight_count} Actual: {log_outputs["straight_count_train"]}'

  assert expected_steering_mse == log_outputs["mse_steering_train"], f'mse_steering_train does not Match! Expected: {expected_steering_mse} Actual: {log_outputs["mse_steering_train"]}'
  assert expected_straight_mse == log_outputs["mse_straight_train"], f'mse_straight_train does not Match! Expected: {expected_straight_mse} Actual: {log_outputs["mse_straight_train"]}'

  assert expected_autonomy_sum_of_interventions == log_outputs["autonomy_sum_of_interventions_train"], f'Autonomy interventions do not Match! Expected: {expected_autonomy_sum_of_interventions} Actual: {log_outputs["autonomy_sum_of_interventions_train"]}'
  assert expected_autonomy == autonomy, f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {autonomy}'
  assert expected_autonomy == log_outputs["train_autonomy"], f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {log_outputs["train_autonomy"]}'

  log('=== Batches of pairs tests ===', config)
  # Batches of pairs [[[], []], [[], []]]
  pairs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.048, 0.033], [0.00, -0.00], [-0.23, 0.36]]
  pairs = torch.tensor(np.array(pairs))

  # Autonomy metric
  log_outputs = {}
  sequence_name = "train"
  log_outputs = torch_trainer.compute_loss_functions(log_outputs, pairs, losses, batch_count, "train", dev, dtype, should_convert_to_tensor=False)
  autonomy = data_functions.compute_final_autonomy(log_outputs["autonomy_sum_of_interventions_train"], len(yb))

  assert expected_mse == log_outputs["selected_train_loss"], f'selected train loss does not Match! Expected: {expected_mse} Actual: {log_outputs["selected_train_loss"]}'
  assert expected_mse == log_outputs["train_mse_loss_func"], f'train mse loss func does not Match! Expected: {expected_mse} Actual: {log_outputs["train_mse_loss_func"]}'
  # assert expected_mse == log_outputs["mse_weighted_train"], f'mse_weighted_train does not Match! Expected: {expected_mse} Actual: {log_outputs["mse_weighted_train"]}'

  assert expected_steering_count == log_outputs["steering_count_train"], f'steering_count_train does not Match! Expected: {expected_steering_count} Actual: {log_outputs["steering_count_train"]}'
  assert expected_straight_count == log_outputs["straight_count_train"], f'straight_count_train does not Match! Expected: {expected_straight_count} Actual: {log_outputs["straight_count_train"]}'

  assert expected_steering_mse == log_outputs["mse_steering_train"], f'mse_steering_train does not Match! Expected: {expected_steering_mse} Actual: {log_outputs["mse_steering_train"]}'
  assert expected_straight_mse == log_outputs["mse_straight_train"], f'mse_straight_train does not Match! Expected: {expected_straight_mse} Actual: {log_outputs["mse_straight_train"]}'

  assert expected_autonomy_sum_of_interventions == log_outputs["autonomy_sum_of_interventions_train"], f'Autonomy interventions do not Match! Expected: {expected_autonomy_sum_of_interventions} Actual: {log_outputs["autonomy_sum_of_interventions_train"]}'
  assert expected_autonomy == autonomy, f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {autonomy}'
  assert expected_autonomy == log_outputs["train_autonomy"], f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {log_outputs["train_autonomy"]}'

  log('=== Nested Batches of pairs tests ===', config)
  expected_mse = 0.3913875
  expected_mse_more_decimals = 0.39138755202293396

  expected_steering_mse = 0.4493666739344598
  expected_straight_mse = 0.3334083333392938

  # Batches of pairs [[[], []], [[], []]]
  expected_batch = torch.tensor(np.array([1.0, 0.0, 0.5, 0.048, 0.00, -0.23]))
  yb = torch.tensor(np.array([0.0, 1.0, 0.5, 0.033, -0.00, 0.36]))
  pairs = [[expected_batch, yb]]

  # Autonomy metric
  log_outputs = {}
  sequence_name = "train"
  log_outputs = torch_trainer.compute_loss_functions(log_outputs, pairs, losses, batch_count, "train", dev, dtype, should_convert_to_tensor=False)
  autonomy = data_functions.compute_final_autonomy(log_outputs["autonomy_sum_of_interventions_train"], len(yb))

  assert expected_mse == log_outputs["selected_train_loss"], f'selected train loss does not Match! Expected: {expected_mse} Actual: {log_outputs["selected_train_loss"]}'
  assert expected_mse_more_decimals == log_outputs["train_mse_loss_func"], f'train mse loss func does not Match! Expected: {expected_mse} Actual: {log_outputs["train_mse_loss_func"]}'
  # assert expected_mse == log_outputs["mse_weighted_train"], f'mse_weighted_train does not Match! Expected: {expected_mse} Actual: {log_outputs["mse_weighted_train"]}'

  assert expected_steering_count == log_outputs["steering_count_train"], f'steering_count_train does not Match! Expected: {expected_steering_count} Actual: {log_outputs["steering_count_train"]}'
  assert expected_straight_count == log_outputs["straight_count_train"], f'straight_count_train does not Match! Expected: {expected_straight_count} Actual: {log_outputs["straight_count_train"]}'

  assert expected_steering_mse == log_outputs["mse_steering_train"], f'mse_steering_train does not Match! Expected: {expected_steering_mse} Actual: {log_outputs["mse_steering_train"]}'
  assert expected_straight_mse == log_outputs["mse_straight_train"], f'mse_straight_train does not Match! Expected: {expected_straight_mse} Actual: {log_outputs["mse_straight_train"]}'

  assert expected_autonomy_sum_of_interventions == log_outputs["autonomy_sum_of_interventions_train"], f'Autonomy interventions do not Match! Expected: {expected_autonomy_sum_of_interventions} Actual: {log_outputs["autonomy_sum_of_interventions_train"]}'
  assert expected_autonomy == autonomy, f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {autonomy}'
  assert expected_autonomy == log_outputs["train_autonomy"], f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {log_outputs["train_autonomy"]}'

  log('=== Quad Data Nested Batches of pairs tests ===', config)
  batch_count = (6, 6, 6, 6)
  expected_steering_count = 12.0
  expected_straight_count = 12.0

  expected_mse = 0.3913875
  expected_mse_more_decimals = 0.3913874924182892

  expected_steering_mse = 0.4493666739344598
  expected_straight_mse = 0.3334083333392938

  expected_autonomy_sum_of_interventions = 12.0
  expected_autonomy = 50.0

  # Batches of pairs [[[], []], [[], []]]
  expected_batch = torch.tensor(np.array([1.0, 0.0, 0.5, 0.048, 0.00, -0.23]))
  yb = torch.tensor(np.array([0.0, 1.0, 0.5, 0.033, -0.00, 0.36]))
  pairs = [[expected_batch, yb], [expected_batch, yb], [expected_batch, yb], [expected_batch, yb]]

  # Autonomy metric
  log_outputs = {}
  sequence_name = "train"
  log_outputs = torch_trainer.compute_loss_functions(log_outputs, pairs, losses, batch_count, "train", dev, dtype, should_convert_to_tensor=False)
  autonomy = data_functions.compute_final_autonomy(log_outputs["autonomy_sum_of_interventions_train"], sum(batch_count))

  assert expected_mse == log_outputs["selected_train_loss"], f'selected train loss does not Match! Expected: {expected_mse} Actual: {log_outputs["selected_train_loss"]}'
  assert expected_mse_more_decimals == log_outputs["train_mse_loss_func"], f'train mse loss func does not Match! Expected: {expected_mse} Actual: {log_outputs["train_mse_loss_func"]}'
  # assert expected_mse == log_outputs["mse_weighted_train"], f'mse_weighted_train does not Match! Expected: {expected_mse} Actual: {log_outputs["mse_weighted_train"]}'

  assert expected_steering_count == log_outputs["steering_count_train"], f'steering_count_train does not Match! Expected: {expected_steering_count} Actual: {log_outputs["steering_count_train"]}'
  assert expected_straight_count == log_outputs["straight_count_train"], f'straight_count_train does not Match! Expected: {expected_straight_count} Actual: {log_outputs["straight_count_train"]}'

  assert expected_steering_mse == log_outputs["mse_steering_train"], f'mse_steering_train does not Match! Expected: {expected_steering_mse} Actual: {log_outputs["mse_steering_train"]}'
  assert expected_straight_mse == log_outputs["mse_straight_train"], f'mse_straight_train does not Match! Expected: {expected_straight_mse} Actual: {log_outputs["mse_straight_train"]}'

  assert expected_autonomy_sum_of_interventions == log_outputs["autonomy_sum_of_interventions_train"], f'Autonomy interventions do not Match! Expected: {expected_autonomy_sum_of_interventions} Actual: {log_outputs["autonomy_sum_of_interventions_train"]}'
  assert expected_autonomy == autonomy, f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {autonomy}'
  assert expected_autonomy == log_outputs["train_autonomy"], f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {log_outputs["train_autonomy"]}'

  log('=== Autonomy Vs MSE Metrics test ===', config)
  # Perfect Predictions
  expected_batch = torch.tensor(np.array([0.0, 1.0, 0.5, 0.25, -1.0, -0.25, 0.15, 0.10, -0.10, -0.15]))
  yb = torch.tensor(np.array([0.0, 1.0, 0.5, 0.25, -1.0, -0.25, 0.15, 0.10, -0.10, -0.15]))

  sum_of_interventions = data_functions.compute_autonomy_interventions(expected_batch, yb)
  autonomy = data_functions.compute_final_autonomy(sum_of_interventions, len(expected_batch))

  mse_loss = torch_optimisers.mse_loss_func(expected_batch, yb)

  log(f'Perfect Predictions: autonomy {autonomy}%, mse_loss: {mse_loss}', config)

  # Massively bad Predictions
  expected_batch = torch.tensor(np.array([0.0, 1.0, 0.5, 0.25, -1.0, -0.25, 0.15, 0.10, -0.10, -0.15]))
  yb = torch.tensor(np.array([1.0, 0.0, -0.5, -0.25, 1.0, 0.25, -0.15, -0.10, 0.10, 0.15]))

  sum_of_interventions = data_functions.compute_autonomy_interventions(expected_batch, yb)
  autonomy = data_functions.compute_final_autonomy(sum_of_interventions, len(expected_batch))

  mse_loss = torch_optimisers.mse_loss_func(expected_batch, yb)

  log(f'Very Bad Predictions: autonomy {autonomy}%, mse_loss: {mse_loss}', config)

  # Half Predictions
  expected_batch = torch.tensor(np.array([0.0, 1.0, 0.5, 0.25, -1.0, -0.25, 0.15, 0.10, -0.10, -0.15]))
  yb = torch.tensor(np.array([0.5, 0.5, 1.0, 0.5, -0.5, -0.5, 0.3, 0.20, -0.20, -0.3]))

  sum_of_interventions = data_functions.compute_autonomy_interventions(expected_batch, yb)
  autonomy = data_functions.compute_final_autonomy(sum_of_interventions, len(expected_batch))

  mse_loss = torch_optimisers.mse_loss_func(expected_batch, yb)

  log(f'Half Bad Predictions: autonomy {autonomy}%, mse_loss: {mse_loss}', config)

  # Error less than epsilon
  expected_batch = torch.tensor(np.array([0.0, 1.0, 0.5, 0.25, -1.0, -0.25, 0.15, 0.10, -0.10, -0.15]))
  yb = torch.tensor(np.array([0.1, 0.98, 0.49, 0.24, -0.9, -0.26, 0.14, 0.11, -0.09, -0.16]))

  sum_of_interventions = data_functions.compute_autonomy_interventions(expected_batch, yb)
  autonomy = data_functions.compute_final_autonomy(sum_of_interventions, len(expected_batch))

  mse_loss = torch_optimisers.mse_loss_func(expected_batch, yb)
  log(f'Error lower than Epsilon Predictions: autonomy {autonomy}%, mse_loss: {mse_loss}', config)

  # Lots of small error
  duplicate_amount = 10000
  noise_scale = 0.05 #1.0

  expected_batch = torch.tensor(np.array([0.0, 1.0, 0.5, 0.25, -1.0, -0.25, 0.15, 0.10, -0.10, -0.15] * duplicate_amount))
  yb = torch.tensor(np.array([0.1, 0.9, 0.4, 0.2, -0.9, -0.2, 0.1, 0.1, -0.00, -0.1] * duplicate_amount))

  noise = torch.randn(yb.size()) * noise_scale
  yb = yb + noise

  sum_of_interventions = data_functions.compute_autonomy_interventions(expected_batch, yb)
  autonomy = data_functions.compute_final_autonomy(sum_of_interventions, len(expected_batch))

  mse_loss = torch_optimisers.mse_loss_func(expected_batch, yb)
  log(f'Lots of Error lower than Epsilon Predictions: autonomy {autonomy}%, mse_loss: {mse_loss}', config)
################################################################################
  # Deprecated:
  # log('=== 2X Nested Batches of pairs tests ===', config)
  # expected_mse = 0.3913875
  # expected_mse_more_decimals = 0.39138755202293396

  # expected_steering_mse = 0.4493666739344598
  # expected_straight_mse = 0.3334083333392938

  # # Batches of pairs [[[], []], [[], []]]
  # expected_batch_1 = torch.tensor(np.array([1.0, 0.0, 0.5]))
  # yb_1 = torch.tensor(np.array([0.0, 1.0, 0.5]))

  # expected_batch_2 = torch.tensor(np.array([0.048, 0.00, -0.23]))
  # yb_2 = torch.tensor(np.array([0.033, -0.00, 0.36]))

  # pairs = [[expected_batch_1, yb_1], [expected_batch_2, yb_2]]

  # # Autonomy metric
  # log_outputs = {}
  # sequence_name = "train"

  # log_outputs = torch_trainer.compute_loss_functions(log_outputs, pairs, losses, batch_count, "train", dev, dtype, should_convert_to_tensor=False)
  # autonomy = data_functions.compute_final_autonomy(log_outputs["autonomy_sum_of_interventions_train"], len(yb))

  # assert expected_mse == log_outputs["selected_train_loss"], f'selected train loss does not Match! Expected: {expected_mse} Actual: {log_outputs["selected_train_loss"]}'
  # assert expected_mse_more_decimals == log_outputs["train_mse_loss_func"], f'train mse loss func does not Match! Expected: {expected_mse} Actual: {log_outputs["train_mse_loss_func"]}'
  # # assert expected_mse == log_outputs["mse_weighted_train"], f'mse_weighted_train does not Match! Expected: {expected_mse} Actual: {log_outputs["mse_weighted_train"]}'

  # assert expected_steering_count == log_outputs["steering_count_train"], f'steering_count_train does not Match! Expected: {expected_steering_count} Actual: {log_outputs["steering_count_train"]}'
  # assert expected_straight_count == log_outputs["straight_count_train"], f'straight_count_train does not Match! Expected: {expected_straight_count} Actual: {log_outputs["straight_count_train"]}'

  # assert expected_steering_mse == log_outputs["mse_steering_train"], f'mse_steering_train does not Match! Expected: {expected_steering_mse} Actual: {log_outputs["mse_steering_train"]}'
  # assert expected_straight_mse == log_outputs["mse_straight_train"], f'mse_straight_train does not Match! Expected: {expected_straight_mse} Actual: {log_outputs["mse_straight_train"]}'

  # assert expected_autonomy_sum_of_interventions == log_outputs["autonomy_sum_of_interventions_train"], f'Autonomy interventions do not Match! Expected: {expected_autonomy_sum_of_interventions} Actual: {log_outputs["autonomy_sum_of_interventions_train"]}'
  # assert expected_autonomy == autonomy, f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {autonomy}'
  # assert expected_autonomy == log_outputs["train_autonomy"], f'Autonomy Does not Match! Expected: {expected_autonomy} Actual: {log_outputs["train_autonomy"]}'
################################################################################

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
  run_metrics_test(config)
