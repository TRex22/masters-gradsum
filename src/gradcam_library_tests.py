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

def run_gradcam_tests(config):
  log('==========================================', config)
  log(f'=== GradCAM Basic                      ===', config)
  log('==========================================', config)
  # https://github.com/jacobgil/pytorch-grad-cam

  from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
  from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
  from pytorch_grad_cam.utils.image import show_cam_on_image
  from torchvision.models import resnet50

  utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

  # https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
  uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
  ]

  input_tensor = batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
  )

  model = resnet50(pretrained=True)
  target_layers = [model.layer4[-1]]

  # unprocessed_frame = image_functions.open_image("./example_meta/cityscapes/raw_image1.png")
  # input_tensor = torch.from_numpy(unprocessed_frame).permute(2, 0, 1).unsqueeze(0) # Create an input tensor image for your model..

  # Note: input_tensor can be a batch tensor with several images!
  # Construct the CAM object once, and then re-use it on many images:
  cam = GradCAM(model=model, target_layers=target_layers)

  # You can also use it within a with statement, to make sure it is freed,
  # In case you need to re-create it inside an outer loop:
  # with GradCAM(model=model, target_layers=target_layers) as cam:
  #   ...

  # We have to specify the target we want to generate
  # the Class Activation Maps for.
  # If targets is None, the highest scoring category
  # will be used for every image in the batch.
  # Here we use ClassifierOutputTarget, but you can define your own custom targets
  # That are, for example, combinations of categories, or specific outputs in a non standard model.

  targets = [ClassifierOutputTarget(281)]

  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
  assert float(grayscale_cam.sum()) != 0.0, "There are no heat-map values!"
  assert not math.isnan(float(grayscale_cam.sum())), "There are no heat-map values! NaN values instead!"

  # In this example grayscale_cam has only one image in the batch:
  # grayscale_cam = grayscale_cam[0, :]
  # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

  # You can also get the model outputs without having to re-inference
  model_outputs = cam.outputs
  assert float(model_outputs.sum()) != 0.0, "There are no model output values!"
  assert not math.isnan(float(model_outputs.sum())), "There are no model output values! NaN values present!"

  print("Done!")


available_datasets = ['cityscapes', 'udacity', 'cookbook', 'carlaimitation']
config = {
  "use_wandb": False,
  "wandb_watch_freq": 1000,
  "wandb_watch_log": "all", # "gradients", "parameters", "all", or None
  "wandb_watch_log_graph": True,
  "wandb_project_name": "random-test-4", # TestModel1-udacity-dmczc
  "wandb_name_prefix": "TEST", #"linear" #"carlaimitation 0.02 sample",
  "Notes": "Large Scale Epoch Test",
  "device_name": "cpu", #"cuda:0" #"cuda:1" #cpu
  "summary_device_name": "cpu",
  "non_blocking": True,
  "pin_memory": False,
  "cuda_spawn": True,
  "purge_cuda_memory": True, # Turn off on shared resources
  "compute_attention": False, # used for vision transformers
  "compute_mean_attention": False,
  "model_name": "End to End", #"deit_tiny_model", #"ViT-H_14", # "ViT-B_32", # "ViT-L_16", # "ViT-L_32", # "ViT-B_16", # "ViT-H_14", # "End to End No Dropout", #"End to End", #"Autonomous Cookbook", #"Net SVF", #"Net HVF", #"TestModel1", #"TestModel2",
  "track_attention_weights": False, # Set to True before running cam analysis
  "average_attn_weights": False,
  "compute_attn_mean": False,
  "model_name": "End to End", #"End to End", #"Autonomous Cookbook", #"Net SVF", #"Net HVF", #"TestModel1", #"TestModel2", #"TestModel3",
  "dataset_name": "cityscapes",
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
  "grad_cam_threshold": 0.5, # 0.2, # 0.01,
  "draw_grad_cam": False,
  "save_grad_cam": False,
  "save_grad_cam_plot": False, # efficiency issues here
  "grad_cam_plot": "cv2", #"seaborn", #"matplotlib"
  "generate_grad_cam_video": False,
  "grad_cam_dataset": "cityscapes", #"udacity", #"cityscapesvideo", # TODO: Others # This is used for the video TODO: Rename
  "compute_grad_cam_results": True,
  "keep_gradcam_frames": True,
  "grad_cam_in_memory": True,
  "grad_cam_drop_percentage": 0.0, #1.0, # Separate data process
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
  "sigmoid": True,
  "momentum": 0.8,
  "number_of_outputs": 1,
  "output_key": "Steering",
  "output_tanh": False,
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
  "mixed_precision": False,
  "bfloat16": False,
  "cache_enabled": True,
  "clip_grad_norm": False,
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
  run_gradcam_tests(config)
