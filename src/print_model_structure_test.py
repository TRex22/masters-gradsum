import time
from datetime import datetime
import tqdm
import math
import os
import json
from subprocess import Popen

import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as jnp
# config.update('jax_enable_x64', True)

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
import data_helpers
import gradcam_functions

import torch_models
import torch_optimisers
import torch_trainer
import torch_tester

from torch_dataset import CustomImageDataset

def run_compile_test(config):
  log('==========================================', config)
  log(f'=== Dataset: {config["dataset_name"]} ===', config)
  log('==========================================', config)

  log(f'split_by_temporal_lines: {config["split_by_temporal_lines"]}', config)
  wandb = None
  run_name = f'{config["model_name"]}-{config["dataset_name"]}-{helper_functions.randomword(5)}'
  config["run_name"] = run_name

  log(f'Initiate time: {datetime.now()}', config)

  dev = helper_functions.fetch_device(config)
  log(f'Use wandb: {config["use_wandb"]}.', config)
  log(f'Run name: {config["run_name"]}', config)

  # Summary of experiment to run
  memory_key = helper_functions.config_memory_key(config)
  config["train_batch_size"] = config["dataset"][memory_key][config["model_name"]]["train"]["batch_size"]
  config["valid_batch_size"] = config["dataset"][memory_key][config["model_name"]]["valid"]["batch_size"]
  config["test_batch_size"] = config["dataset"][memory_key][config["model_name"]]["test"]["batch_size"]

  train_workers = config["dataset"][memory_key][config["model_name"]]["train"]["num_workers"]
  log(f'Train Batch Size: {config["train_batch_size"]} for {train_workers} workers.', config)
  log(f'Base Data Path: {config["base_data_path"]}', config)
  log(f'Mixed Precision: {config["mixed_precision"]}', config)
  log(f'bfloat16: {config["bfloat16"]}', config)

  dataset_string = data_helpers.compute_dataset_string(config)
  config["dataset_string"] = dataset_string

  print("Compile Models ...")

  config["model_name"] = "Autonomous Cookbook"
  print(f"{config['model_name']}...")
  model = torch_models.compile_model(config)
  model_stats = summary(model, input_size=(8, config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  print(model_stats)
  print("\n\n")

  config["model_name"] = "TestModel1"
  print(f"{config['model_name']}...")
  model = torch_models.compile_model(config)
  model_stats = summary(model, input_size=(8, config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  print(model_stats)
  print("\n\n")

  config["model_name"] = "TestModel2"
  print(f"{config['model_name']}...")
  model = torch_models.compile_model(config)
  model_stats = summary(model, input_size=(8, config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  print(model_stats)
  print("\n\n")

  config["model_name"] = "End to End"
  print(f"{config['model_name']}...")
  model = torch_models.compile_model(config)
  model_stats = summary(model, input_size=(8, config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  print(model_stats)
  print("\n\n")

  config["model_name"] = "Net SVF"
  print(f"{config['model_name']}...")
  model = torch_models.compile_model(config)
  model_stats = summary(model, input_size=(8, config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  print(model_stats)
  print("\n\n")

  config["model_name"] = "Net HVF"
  print(f"{config['model_name']}...")
  model = torch_models.compile_model(config)
  model_stats = summary(model, input_size=(8, config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  print(model_stats)
  print("\n\n")

  # config["input_size"] = [3, 112, 112]
  # config["model_name"] = "Net HVF"
  # print(f"{config['model_name']} (With Square input)...")
  # model = torch_models.compile_model(config)
  # model_stats = summary(model, input_size=(8, config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  # print(model_stats)
  # print("\n\n")

  # Vision models
  # config["input_size"] = [3, 224, 224]
  # config["model_name"] = "ViT-H_14"
  # print(f"{config['model_name']}...")
  # model = torch_models.compile_model(config)

  # config["model_name"] = "ViT-L_32"
  # print(f"{config['model_name']}...")
  # model = torch_models.compile_model(config)

  # config["model_name"] = "ViT-L_16"
  # print(f"{config['model_name']}...")
  # model = torch_models.compile_model(config)

  # config["model_name"] = "ViT-B_32"
  # print(f"{config['model_name']}...")
  # model = torch_models.compile_model(config)

  # config["model_name"] = "ViT-B_16"
  # print(f"{config['model_name']}...")
  # model = torch_models.compile_model(config)

  # config["model_name"] = "deit_tiny_model"
  # print(f"{config['model_name']}...")
  # model = torch_models.compile_model(config)
  # log('\n\n\n', config)

available_datasets = ['cityscapes', 'udacity', 'cookbook', 'carlaimitation']
config = {
  "compile_model": False,
  "use_wandb": False,
  "wandb_watch_freq": 1000,
  "wandb_watch_log": "all", # "gradients", "parameters", "all", or None
  "wandb_watch_log_graph": True,
  "wandb_project_name": "random-test-4", # TestModel1-udacity-dmczc
  "wandb_name_prefix": "Mixed Precision", #"linear" #"carlaimitation 0.02 sample",
  "Notes": "Large Scale Epoch Test",
  "device_name": "cpu", #"cuda:0" #"cuda:1" #cpu
  "summary_device_name": "cpu",
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
  "output_tanh": False,
  "model_save_path": "/mnt/excelsior/trained_models",
  "base_data_path":  "/data/data",
  "base_cook_data_path": "/data/data/cooking",
  "data_drop_precision": 2,
  "accuracy_precision": 2,
  "zero_drop_percentage": 0.95,
  "drop_invalid_steering_angles": False,
  "sample_percentage": 1,
  "train_val_test_split": [0.7, 0.2, 0.1],
  "split_by_temporal_lines": True,
  "combine_val_test": True,
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
      "deit_tiny_model": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 1, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-H_14": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
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
        "train": { "batch_size": 500, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
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
        "train": { "batch_size": 1000, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 1000, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 1000, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": None }
      }
    },
    "available_12": {
      "deit_tiny_model": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 1, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-H_14": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net SVF": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net HVF": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "End to End": {
        "train": { "batch_size": 500, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 500, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 500, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Autonomous Cookbook": {
        "train": { "batch_size": 200, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 200, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel1": {
        "train": { "batch_size": 500, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 500, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 500, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel2": {
        "train": { "batch_size": 500, "shuffle": True, "num_workers": 3, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 500, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 500, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      }
    },
    "available_10": {
      "deit_tiny_model": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 1, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-H_14": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net SVF": {
        "train": { "batch_size": 75, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 75, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 75, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Net HVF": {
        "train": { "batch_size": 125, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 125, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 125, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "End to End": {
        "train": { "batch_size": 200, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 400, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 400, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "Autonomous Cookbook": {
        "train": { "batch_size": 200, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 200, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 200, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel1": {
        "train": { "batch_size": 300, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 300, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 300, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel2": {
        "train": { "batch_size": 300, "shuffle": True, "num_workers": 3, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 300, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 300, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      }
    },
    "available_6": {
      "deit_tiny_model": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 1, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-H_14": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
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
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 300, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": None }
      }
    },
    "available_4": {
      "deit_tiny_model": {
        "train": { "batch_size": 100, "shuffle": True, "num_workers": 1, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 100, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 100, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-H_14": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-L_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_32": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "ViT-B_16": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": False, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
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
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None }
      },
      "TestModel2": {
        "train": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": 2 },
        "valid": { "batch_size": 50, "shuffle": False, "num_workers": 0, "drop_last": True, "persistent_workers": False, "prefetch_factor": None },
        "test": { "batch_size": 50, "shuffle": True, "num_workers": 2, "drop_last": True, "persistent_workers": True, "prefetch_factor": None }
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
  for dataset_name in available_datasets:
    config["dataset_name"] = dataset_name
    run_compile_test(config)
