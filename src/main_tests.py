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

# Hard-code better GPU
config["device_name"] = "cuda:0"
config["data_parallel_device_ids"] = [1, 0]

helper_functions.clear_gpu(torch, config)
helper_functions.trigger_torch_optimisations(torch, config)

# TODO: https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass
  ################################################################################
  # Test Model
  config["warmup_data"] = False # Slows down test only sets
  config["compute_other_test_loss"] = True

  # batch_size = torch_models.external_test_model_batch_size(config)

  # config["dataset"]["available_24"][config["model_name"]]["test"]["batch_size"] = config["test_batch_size"]
  # config["dataset"]["available_24"][config["model_name"]]["test"]["persistent_workers"] = True
  # config["dataset"]["available_24"][config["model_name"]]["test"]["num_workers"] = 1
  # config["dataset"]["available_24"][config["model_name"]]["test"]["prefetch_factor"] = 1
  # config["dataset"]["available_24"][config["model_name"]]["test"]["drop_last"] = False

  # config["dataset"]["available_12"][config["model_name"]]["test"]["batch_size"] = config["test_batch_size"]
  # config["dataset"]["available_12"][config["model_name"]]["test"]["persistent_workers"] = True
  # config["dataset"]["available_12"][config["model_name"]]["test"]["num_workers"] = 1
  # config["dataset"]["available_12"][config["model_name"]]["test"]["prefetch_factor"] = 1
  # config["dataset"]["available_12"][config["model_name"]]["test"]["drop_last"] = False

  torch_tester.run_model_tests(config)
  helper_functions.clear_gpu(torch, config)

  torch_tester.run_model_tests(config, best_autonomy=True)
  helper_functions.clear_gpu(torch, config)
  ################################################################################
