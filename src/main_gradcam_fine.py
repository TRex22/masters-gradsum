import time
from datetime import datetime
import tqdm
import math
import os
import json
from subprocess import Popen

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

def reset_model():
  model = torch_models.compile_model(config)
  dtype = helper_functions.compute_model_dtype(config)
  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])
  model.eval()

  return model
################################################################################

config_path = sys.argv[1]
print(f'Config path: {config_path}') # Cannot log before config loaded!

if len(sys.argv) > 2:
  model_save_path = sys.argv[2]
  base_data_path = sys.argv[3]
  base_cook_data_path = sys.argv[4]

if len(sys.argv) > 5:
  compute_device = sys.argv[5]

if len(sys.argv) > 7:
  epoch_start = int(sys.argv[6])
  epoch_end = int(sys.argv[7])

experiment_types = '0,1,2,3'
if len(sys.argv) > 8:
  experiment_types = sys.argv[8].split(',')

config = helper_functions.open_config(config_path)
config["running_gradcam"] = True
config["grad_cam_algo"] = "absolute"
config["non_blocking"] = False
config["device_name"] = "cuda:0" # hardcode for now
config["data_parallel_device_ids"] = [1, 0]

# Hack so I dont have to modify configs
if not helper_functions.detect_folder(config["base_data_path"]):
  config["model_save_path"] = "/data/trained_models"
  config["base_data_path"] = "/data/data"
  config["base_cook_data_path"] = "/data/data/cooking"

helper_functions.clear_gpu(torch, config)
helper_functions.trigger_torch_optimisations(torch, config)

# TODO: https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass

  if len(sys.argv) > 2:
    config["model_save_path"] = model_save_path
    config["base_data_path"] = base_data_path
    config["base_cook_data_path"] = base_cook_data_path

  if len(sys.argv) > 5:
    config["device_name"] = compute_device

  if len (sys.argv) > 7:
    config['epoch_start'] = epoch_start
    config['epoch_end'] = epoch_end

  config['experiment_types'] = experiment_types

  wandb = None
  run_name = config["run_name"]
  config["original_run_name"] = config["run_name"]

  log('################################################################################', config)
  log('Execute Grad-CAM Results ...', config)

  # Can only log from this point onwards because the save path can now be computed
  log(f'Initiate time: {datetime.now()}', config)

  dev = helper_functions.fetch_device(config)
  log(f'Run name: {config["run_name"]}', config)

  # Summary of experiment to run
  config["grad_cam_batch_size"], config = gradcam_functions.compute_grad_cam_batch_size(config)
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

  # Load model
  model = reset_model()

  # if config["device_name"]
  model_stats = summary(model, input_size=(config["grad_cam_batch_size"], config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  log(model_stats, config)
  del model

  # Estimate total memory usage
  estimated_total_size_of_model = float(f'{model_stats}'.split("\n")[-2].split(" ")[-1])

  if train_workers > 0:
    estimated_total_memory_usage = estimated_total_size_of_model * train_workers
  else:
    estimated_total_memory_usage = estimated_total_size_of_model

  log(f"Estimated total memory usage: {estimated_total_memory_usage} MB", config)

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
  config["data_drop_precision"] = 2
  config["compute_grad_cam_results"] = True

  config["grad_cam_coarse"] = False
  config["grad_cam_fine"] = True

  config["grad_cam_ident"] = f"fine_grad/"

  # if '0' in config['experiment_types']:
  exp_start_time = time.time()

  # all data no perturbations
  config["add_perturbations"] = False
  config["grad_cam_drop_percentage"] = 0.0 #0.5
  gradcam_functions.run_grad_cam(config)

  exp_end_time = time.time()
  log(f'\nDefault GradCAM Time: {exp_end_time - exp_start_time} secs.', config)

  # Disabled for now
  # if '1' in config['experiment_types']:
  #   exp_start_time = time.time()

  #   # all data perturbations
  #   config["add_perturbations"] = True
  #   config["grad_cam_drop_percentage"] = 0.0 #0.5
  #   gradcam_functions.run_grad_cam(config)

  #   exp_end_time = time.time()
  #   log(f'\nTurning Only GradCAM Time: {exp_end_time - exp_start_time} secs.', config)

  # if '2' in config['experiment_types']:
  #   exp_start_time = time.time()

  #   # turning only no perturbations
  #   config["add_perturbations"] = False
  #   config["grad_cam_drop_percentage"] = 1.0
  #   gradcam_functions.run_grad_cam(config)

  #   exp_end_time = time.time()
  #   log(f'\nMain GradCAM + Drop GradCAM Time: {exp_end_time - exp_start_time} secs.', config)

  # if '3' in config['experiment_types']:
  #   exp_start_time = time.time()

  #   # turning only perturbations
  #   config["add_perturbations"] = True
  #   config["grad_cam_drop_percentage"] = 1.0
  #   gradcam_functions.run_grad_cam(config)

  #   exp_end_time = time.time()
  #   log(f'\nTurning Only + Drop GradCAM Time: {exp_end_time - exp_start_time} secs.', config)

################################################################################
  # Inside main loop:
  end_time = time.time()
  total_time = end_time - start_time

  config["total_time"] = total_time
  log(f'\n\nTotal Time: {config["total_time"]} secs.', config)

################################################################################
################################################################################
