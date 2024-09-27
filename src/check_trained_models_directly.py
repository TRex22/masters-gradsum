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

config = {
  "model_name": "TestModel1",
  "randomise_weights": 'xavier_uniform',
  "input_size": [3, 256, 60],
  "normalise_output": False,
  "output_tanh": False,
  "number_of_outputs": 1,
  "sigmoid": True,
  "log_to_file": False
}

def compare_models(model1, model2, str, config):
  print(f"\n\n{str}")
  [parameters_match, states_match] = torch_models.compare(model1, model2, config, verbose=True)

  if parameters_match or states_match:
    log(f'\033[93mparameters_match: {parameters_match}, states_match: {states_match}', config)
  else:
    log(f'\033[94mparameters_match: {parameters_match}, states_match: {states_match}', config)

  print(f"\033[39m\033[49mDone!")

def open_model(model_path):
  model = torch_models.compile_model(config)
  dev = torch.device("cpu")

  checkpoint = torch.load(model_path)
  model.load_state_dict(checkpoint['model'], strict=False)

  return model

base_path = "/data/trained_models/tmp/"
# model_names = [
#   "0.pth",
#   "1.pth",
#   "5.pth",
#   "10.pth",
#   "15.pth",
#   "best_model_15.pth",
#   "best_val_autonomy_model_19.pth",
#   "checkpoint.pth",
#   "modeltm.pth"
# ]

compare_models(open_model(f'{base_path}0.pth'), open_model(f'{base_path}1.pth'), "0 - 1", config)
compare_models(open_model(f'{base_path}1.pth'), open_model(f'{base_path}5.pth'), "1 - 5", config)
compare_models(open_model(f'{base_path}5.pth'), open_model(f'{base_path}10.pth'), "5 - 10", config)
compare_models(open_model(f'{base_path}10.pth'), open_model(f'{base_path}15.pth'), "10 - 15", config)

compare_models(open_model(f'{base_path}15.pth'), open_model(f'{base_path}best_model_15.pth'), "15 - best_model_15", config)
compare_models(open_model(f'{base_path}15.pth'), open_model(f'{base_path}best_val_autonomy_model_19.pth'), "15 - best_val_autonomy_model_19", config)

compare_models(open_model(f'{base_path}checkpoint.pth'), open_model(f'{base_path}modeltm.pth'), "checkpoint - modeltm", config)
compare_models(open_model(f'{base_path}checkpoint.pth'), open_model(f'{base_path}best_val_autonomy_model_19.pth'), "checkpoint - best_val_autonomy_model_19", config)

compare_models(open_model(f'{base_path}best_val_autonomy_model_19.pth'), open_model(f'{base_path}best_model_15.pth'), "best_val_autonomy_model_19 - best_model_15", config)



