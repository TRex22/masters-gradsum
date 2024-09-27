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
  "base_data_path":  "~/example_data/",
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

cityscapes_labels = {
  "unlabeled"           :  0,
  "ego vehicle"         :  1,
  "rectification border":  2,
  "out of roi"          :  3,
  "static"              :  4,
  "dynamic"             :  5,
  "ground"              :  6,
  "road"                :  7,
  "sidewalk"            :  8,
  "parking"             :  9,
  "rail track"          : 10,
  "building"            : 11,
  "wall"                : 12,
  "fence"               : 13,
  "guard rail"          : 14,
  "bridge"              : 15,
  "tunnel"              : 16,
  "pole"                : 17,
  "polegroup"           : 18,
  "traffic light"       : 19,
  "traffic sign"        : 20,
  "vegetation"          : 21,
  "terrain"             : 22,
  "sky"                 : 23,
  "person"              : 24,
  "rider"               : 25,
  "car"                 : 26,
  "truck"               : 27,
  "bus"                 : 28,
  "caravan"             : 29,
  "trailer"             : 30,
  "train"               : 31,
  "motorcycle"          : 32,
  "bicycle"             : 33,
  "license plate"       : -1
}

cityscapes_colours = {
  "unlabeled"           : [  0,  0,  0],
  "ego vehicle"         : [  0,  0,  0],
  "rectification border": [  0,  0,  0],
  "out of roi"          : [  0,  0,  0],
  "static"              : [  0,  0,  0],
  # "ai_unlabeled"        : [  0,  0,  0],
  "dynamic"             : [111, 74,  0],
  "ground"              : [ 81,  0, 81],
  "road"                : [128, 64,128],
  "sidewalk"            : [244, 35,232],
  "parking"             : [250,170,160],
  "rail track"          : [230,150,140],
  "building"            : [ 70, 70, 70],
  "wall"                : [102,102,156],
  "fence"               : [190,153,153],
  "guard rail"          : [180,165,180],
  "bridge"              : [150,100,100],
  "tunnel"              : [150,120, 90],
  "pole"                : [153,153,153],
  "polegroup"           : [153,153,153],
  "traffic light"       : [250,170, 30],
  "traffic sign"        : [220,220,  0],
  "vegetation"          : [107,142, 35],
  "terrain"             : [152,251,152],
  "sky"                 : [ 70,130,180],
  "person"              : [220, 20, 60],
  "rider"               : [255,  0,  0],
  "car"                 : [  0,  0,142],
  "truck"               : [  0,  0, 70],
  "bus"                 : [  0, 60,100],
  "caravan"             : [  0,  0, 90],
  "trailer"             : [  0,  0,110],
  "train"               : [  0, 80,100],
  "motorcycle"          : [  0,  0,230],
  "bicycle"             : [119, 11, 32],
  "license plate"       : [  0,  0,142]
}

car_classify_counts_train = {
  1.0: 2810,
  0.0:  165
}

car_classify_counts_val = {
  1.0: 477,
  0.0:  23
}

# TODO: Fix these counts!
vehicle_classify_counts_train = {
  1.0: 2892,
  0.0:   83
}

vehicle_classify_counts_val = {
  1.0: 489,
  0.0:  11
}

################################################################################

def binary_label_test(config):
  log('==========================================', config)
  log(f'=== Segmentation Label Test            ===', config)
  log('==========================================', config)

  label_map = gradcam_functions.fetch_label_map()
  colour_map = gradcam_functions.fetch_colour_map()

  label_keys = gradcam_functions.fetch_keys()

  ignore_value = 'Is Swerve'

  config["dataset_name"] = "cityscapes"
  config["output_key"] = "Classify"
  config["label"] = "car"

  raw_meta_cityscapes = helper_functions.open_dataframe("./example_meta/cityscapes/raw_meta.csv")
  meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

  zero_labels = data_helpers.set_binary_label(config, meta, placeholder=True)
  assert 0.0 == meta[config["output_key"]].sum(), "Place-holder map not working! Should all be 0.0"

  config["label"] = "car"
  meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

  # train_extra
  meta = data_helpers.filter_fine_only(meta, config)
  train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)

  seg_labels = data_helpers.set_binary_label(config, train, placeholder=False)
  assert car_classify_counts_train[1.0] == seg_labels['Classify'].value_counts()[1.0]
  assert car_classify_counts_train[0.0] == seg_labels['Classify'].value_counts()[0.0]

  seg_labels = data_helpers.set_binary_label(config, val, placeholder=False)
  assert car_classify_counts_val[1.0] == seg_labels['Classify'].value_counts()[1.0]
  assert car_classify_counts_val[0.0] == seg_labels['Classify'].value_counts()[0.0]

  config["label"] = "vehicle" # Test Group
  meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

  seg_labels = data_helpers.set_binary_label(config, train, placeholder=False)
  assert vehicle_classify_counts_train[1.0] == seg_labels['Classify'].value_counts()[1.0]
  assert vehicle_classify_counts_train[0.0] == seg_labels['Classify'].value_counts()[0.0]

  seg_labels = data_helpers.set_binary_label(config, val, placeholder=False)
  assert vehicle_classify_counts_val[1.0] == seg_labels['Classify'].value_counts()[1.0]
  assert vehicle_classify_counts_val[0.0] == seg_labels['Classify'].value_counts()[0.0]

  log('==========================================', config)
  log(f'=== Segmentation Analysis of Groups    ===', config)
  log('==========================================', config)

  group_names = ["void", "sky", "nature", "object", "construction", "vehicle", "human", "flat"]
  group_map = gradcam_functions.fetch_full_group_map()

  log('=== Train                              ===', config)
  for group_name in group_names:
    config["label"] = group_name
    meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

    meta = data_helpers.filter_fine_only(meta, config)
    train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)

    seg_labels = data_helpers.set_binary_label(config, train, placeholder=False)

    yes_counts = 0
    if 1.0 in seg_labels['Classify'].value_counts():
      yes_counts = seg_labels['Classify'].value_counts()[1.0]

    no_counts = 0
    if 0.0 in seg_labels['Classify'].value_counts():
      no_counts = seg_labels['Classify'].value_counts()[0.0]

    total = yes_counts + no_counts
    print(f"{group_name}, \t yes: {yes_counts}, \t no: {no_counts}, \t total: {total}")

  log('\n=== Val                               ===', config)
  for group_name in group_names:
    config["label"] = group_name
    meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

    meta = data_helpers.filter_fine_only(meta, config)
    train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)

    seg_labels = data_helpers.set_binary_label(config, val, placeholder=False)

    yes_counts = 0
    if 1.0 in seg_labels['Classify'].value_counts():
      yes_counts = seg_labels['Classify'].value_counts()[1.0]

    no_counts = 0
    if 0.0 in seg_labels['Classify'].value_counts():
      no_counts = seg_labels['Classify'].value_counts()[0.0]

    total = yes_counts + no_counts
    print(f"{group_name}, \t yes: {yes_counts}, \t no: {no_counts}, \t total: {total}")

  log('==========================================', config)
  log(f'=== Segmentation Analysis of Labels    ===', config)

  log('==========================================', config)
  log('------------------------------------------', config)

  for group_name in group_names:
    print(f"{group_name}:")
    log('=== Train                              ===', config)
    for label in group_map[group_name]:
      config["label"] = label
      meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

      meta = data_helpers.filter_fine_only(meta, config)
      train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)

      seg_labels = data_helpers.set_binary_label(config, train, placeholder=False)

      yes_counts = 0
      if 1.0 in seg_labels['Classify'].value_counts():
        yes_counts = seg_labels['Classify'].value_counts()[1.0]

      no_counts = 0
      if 0.0 in seg_labels['Classify'].value_counts():
        no_counts = seg_labels['Classify'].value_counts()[0.0]

      total = yes_counts + no_counts
      print(f"{label}, \t yes: {yes_counts}, \t no: {no_counts}, \t total: {total}")

    log('\n=== Val                               ===', config)
    for label in group_map[group_name]:
      config["label"] = label
      meta = helper_functions.open_dataframe("./example_meta/cityscapes/meta.csv")

      meta = data_helpers.filter_fine_only(meta, config)
      train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)

      seg_labels = data_helpers.set_binary_label(config, val, placeholder=False)

      yes_counts = 0
      if 1.0 in seg_labels['Classify'].value_counts():
        yes_counts = seg_labels['Classify'].value_counts()[1.0]

      no_counts = 0
      if 0.0 in seg_labels['Classify'].value_counts():
        no_counts = seg_labels['Classify'].value_counts()[0.0]

      total = yes_counts + no_counts
      print(f"{label}, \t yes: {yes_counts}, \t no: {no_counts}, \t total: {total}")

    log('------------------------------------------', config)

  log('==========================================', config)
  log('Done!', config)

helper_functions.clear_gpu(torch, config)

if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass
  binary_label_test(config)
