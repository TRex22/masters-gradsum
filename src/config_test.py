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
  torch_models.compile_model(config)

  config["model_name"] = "TestModel1"
  print(f"{config['model_name']}...")
  torch_models.compile_model(config)

  config["model_name"] = "TestModel2"
  print(f"{config['model_name']}...")
  torch_models.compile_model(config)

  config["model_name"] = "End to End"
  print(f"{config['model_name']}...")
  torch_models.compile_model(config)

  config["model_name"] = "Net SVF"
  print(f"{config['model_name']}...")
  torch_models.compile_model(config)

  config["model_name"] = "Net HVF"
  print(f"{config['model_name']}...")
  torch_models.compile_model(config)

  log('\n\n\n', config)

available_datasets = ['cityscapes', 'udacity', 'cookbook', 'carlaimitation']

config_path = "../experiments/defaults/dev.config"
print(f'Config path: {config_path}')

config = helper_functions.open_config(config_path)
config["run_name"] = "test run"
config["in_test"] = True

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
