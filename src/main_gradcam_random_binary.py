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

model_save_path = sys.argv[1]
print(f'Model Save path: {model_save_path}') # Cannot log before config loaded!

base_data_path = sys.argv[2]
base_cook_data_path = sys.argv[3]
model_name = sys.argv[4]
dataset_name = sys.argv[5]
label = sys.argv[6]

model_count = 100

config_path = '../experiments/defaults/classification.config'
config = helper_functions.open_config(config_path)
config["running_gradcam"] = True

config["mixed_precision"] = False # Hack to try and fix this
config["bfloat16"] = False

helper_functions.clear_gpu(torch, config)

# TODO: https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass

  wandb = None
  config["non_blocking"] = False
  config["model_name"] = model_name
  config["dataset_name"] = dataset_name

  config["model_save_path"] = model_save_path
  config["base_data_path"] = base_data_path
  config["base_cook_data_path"] = base_cook_data_path

  memory_key = helper_functions.config_memory_key(config)
  config["test_batch_size"] = config["dataset"][memory_key][config["model_name"]]["test"]["batch_size"]

  config['randomise_weights'] == 'xavier_uniform'

  config['label'] = label

  run_name = f'[RandomModel]-{label}-{config["model_name"]}-{config["dataset_name"]}-{time.time()}'
  config["run_name"] = run_name
  config["original_run_name"] = config["run_name"]

  log('################################################################################', config)
  log('Execute Random Model Grad-CAM Results ...', config)

  # Can only log from this point onwards because the save path can now be computed
  log(f'Initiate time: {datetime.now()}', config)

  dev = helper_functions.fetch_device(config)
  log(f'Run name: {config["run_name"]}', config)

  # Summary of experiment to run
  memory_key = helper_functions.config_memory_key(config)
  log(f'Available Memory Key: {memory_key}', config)
  config["train_batch_size"] = config["grad_cam_batch_size"]
  train_workers = config["dataset"][memory_key][config["model_name"]]["train"]["num_workers"]

  log(f'Train Batch Size: {config["train_batch_size"]} for {train_workers} workers.', config)
  log(f'Base Data Path: {config["base_data_path"]}', config)
  log(f'Mixed Precision: {config["mixed_precision"]}', config)
  log(f'bfloat16: {config["bfloat16"]}', config)

  log(f'GradCAM Batch Size: {config["grad_cam_batch_size"]}', config)

  dataset_string = data_helpers.compute_dataset_string(config)
  config["dataset_string"] = dataset_string
  ################################################################################

  # Compile model
  model = torch_models.compile_model(config)
  dtype = helper_functions.compute_gradcam_dtype(config) # TODO: Other dtype rather?
  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  model_stats = summary(model, input_size=(config["train_batch_size"], config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  log(model_stats, config)

  # Estimate total memory usage
  estimated_total_size_of_model = float(f'{model_stats}'.split("\n")[-2].split(" ")[-1])

  if train_workers > 0:
    estimated_total_memory_usage = estimated_total_size_of_model * train_workers
  else:
    estimated_total_memory_usage = estimated_total_size_of_model

  log(f"Estimated total memory usage: {estimated_total_memory_usage} MB", config)

  config["mixed_precision"] = False # Hack to try and fix this
  config["bfloat16"] = False
################################################################################
  # Main Gradcam execution
  #
  # TODO:
  # - Reduce the number of labels we care about
  # - Increase data size again
  # - SPlit out loop
  # - Time each section
  # - Add in bounding boxes
  # - Batch Data
  # - batch GradCAM
  #

  # all data no perturbations
  log(f'All data and no perturbations', config)
  config["add_perturbations"] = False
  config["compute_grad_cam_results"] = False
  config["grad_cam_drop_percentage"] = 0.0 #0.5
  config["grad_cam_algo"] = "absolute"

  config["grad_cam_coarse"] = False
  config["grad_cam_fine"] = True
  config["grad_cam_ident"] = f"fine_grad/"

  train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds = data_helpers.data_processing(config, dev)
  gradcam_functions.run_grad_cam(config, model_count=model_count)

  # all data perturbations
  # config["add_perturbations"] = True
  # config["grad_cam_drop_percentage"] = 0.0 #0.5
  # gradcam_functions.run_grad_cam(config, model_count=model_count)

  # # turning only no perturbations
  # config["add_perturbations"] = False
  # config["grad_cam_drop_percentage"] = 1.0
  # gradcam_functions.run_grad_cam(config, model_count=model_count)

  # # turning only perturbations
  # config["add_perturbations"] = True
  # config["grad_cam_drop_percentage"] = 1.0
  # gradcam_functions.run_grad_cam(config, model_count=model_count)

################################################################################
  # Inside main loop:
  end_time = time.time()
  total_time = end_time - start_time

  config["total_time"] = total_time
  log(f'\n\nTotal Time: {config["total_time"]} secs.', config)

################################################################################
################################################################################
