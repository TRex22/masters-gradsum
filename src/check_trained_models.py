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
helper_functions.trigger_torch_optimisations(torch, config)

# TODO: https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass
  ################################################################################
  # Check Models
  log(f'=                 Model 1                 =', config)
  log('=================================================================', config)
  best_epoch_model, _opt_epoch_model, _scaler_epoch_model = torch_models.open_model(config, model_eval=True, epoch=config["best_val_autonomy_epoch"], append_path="best_val_autonomy_model")
  summary(best_epoch_model)

  log(f'=                 Model 2                 =', config)
  log('=================================================================', config)
  best_autonomy_model, _opt_autonomy_model, _scaler_autonomy_model = torch_models.open_model(config, model_eval=True, epoch=config["best_model_epoch"], append_path="best_model")
  summary(best_autonomy_model)

  [parameters_match, states_match] = torch_models.compare(best_epoch_model, best_autonomy_model, config, verbose=True)

  if parameters_match or states_match:
    log(f'\033[93mparameters_match: {parameters_match}, states_match: {states_match}', config)
  else:
    log(f'\033[94mparameters_match: {parameters_match}, states_match: {states_match}', config)

  print(f"\033[39m\033[49mDone!")
  ################################################################################
