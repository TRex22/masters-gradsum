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
torcheck = {} # For parameters
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
import data_helpers
import gradcam_functions

import torch_models
import torch_optimisers
import torch_trainer
import torch_tester

from torch_dataset import CustomImageDataset

config_path = sys.argv[1]
print(f'Config path: {config_path}') # Cannot log before config loaded!

config = helper_functions.open_config(config_path)

number_of_runs = 100

# TODO: https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass
  config["run_name"] = "RandomModelTest"
  config["train_op"] = False

  log(f'Initiate time: {datetime.now()}', config)
  dev = helper_functions.fetch_device(config)
  available_memory = helper_functions.available_free_memory(config)

  log(f'Use wandb: {config["use_wandb"]}.', config)
  log(f'Run name: {config["run_name"]}', config)

  # Summary of experiment to run
  memory_key = helper_functions.config_memory_key(config)
  log(f'Selected Memory Key: {memory_key}', config)
  config["train_batch_size"] = config["dataset"][memory_key][config["model_name"]]["train"]["batch_size"]
  config["valid_batch_size"] = config["dataset"][memory_key][config["model_name"]]["valid"]["batch_size"]
  config["test_batch_size"] = config["dataset"][memory_key][config["model_name"]]["test"]["batch_size"]

  train_workers = config["dataset"][memory_key][config["model_name"]]["train"]["num_workers"]
  log(f'Train Batch Size: {config["train_batch_size"]} for {train_workers} workers.', config)
  log(f'Base Data Path: {config["base_data_path"]}', config)
  log(f'Mixed Precision: {config["mixed_precision"]}', config)
  log(f'bfloat16: {config["bfloat16"]}', config)
  log(f'output_tanh: {config["output_tanh"]}', config)

  dataset_string = data_helpers.compute_dataset_string(config)
  config["dataset_string"] = dataset_string

  ################################################################################
  # Data Processing

  start_data_time = time.time()
  train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds = data_helpers.data_processing(config, dev)
  log(f'Data setup time: {time.time() - start_data_time} secs.', config)

  models = ["Net SVF", "Net HVF", "End to End", "Autonomous Cookbook", "TestModel1", "TestModel2"]
  for model_name in models:
    config["model_name"] = model_name

    model_losses_header = 'cookbook,udacity,cityscapes'
    model_autonomy_header = 'cookbook,udacity,cityscapes'

    model_losses_csv = f'{model_losses_header}\n'
    model_autonomy_csv = f'{model_autonomy_header}\n'

    for i in range(number_of_runs):
      model = torch_models.compile_model(config)

      _opt, loss_func = torch_optimisers.fetch_loss_opt_func(config, model)
      _test_outputs, dataset_losses, dataset_autonomy = torch_tester.run_model_test(model, loss_func, config, dev)

      # model_losses.append(dataset_losses)
      # model_autonomy.append(dataset_autonomy)

      model_losses_csv = f'{model_losses_csv}\n{dataset_losses["cookbook"]},{dataset_losses["udacity"]},{dataset_losses["cityscapes"]}'
      model_autonomy_csv = f'{model_autonomy_csv}\n{dataset_autonomy["cookbook"]},{dataset_autonomy["udacity"]},{dataset_autonomy["cityscapes"]}'

      del model
      helper_functions.clear_gpu(torch, config, pause=False)

    model_losses_file_path = f'{helper_functions.compute_model_save_path(config)}_{model_name}_losses.csv'
    helper_functions.save_csv(model_losses_file_path, model_losses_csv)
    model_autonomy_file_path = f'{helper_functions.compute_model_save_path(config)}_{model_name}_autonomy.csv'
    helper_functions.save_csv(model_autonomy_file_path, model_autonomy_csv)

  helper_functions.clear_gpu(torch, config, pause=False)
