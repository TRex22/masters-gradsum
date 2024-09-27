import time
from datetime import datetime
import tqdm
import math
import os
import json
import copy
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

# import benchmark_agent

# TODO: 2024 Optimisations to Read up on:
# - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# - https://pytorch.org/blog/flexattention/
# - https://huggingface.co/docs/transformers/perf_train_gpu_many

# 2024: Parallel with pyTorch
# - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# - https://pytorch.org/blog/flexattention/
# - https://huggingface.co/docs/transformers/perf_train_gpu_many
# - https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism
# - https://pytorch.org/docs/stable/distributed.tensor.parallel.html
# - https://pytorch.org/tutorials/intermediate/TP_tutorial.html
# - https://huggingface.co/spaces/hf-accelerate/model-memory-usage

# TODO: READ https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/5
# Figure out if the model structures are correct

# Hyper-params
# "batch_size": 256, #16, #32, #64, #128, # 256, #512 #256 #64,

# TODO: network viz
# https://newbedev.com/how-do-i-visualize-a-net-in-pytorch

# TODO: https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html

# TODO: Set rnd seeds
# torch.cuda.manual_seed and .manual_seed_all
# https://pytorch.org/docs/stable/cuda.html

# Pytorch opt dataloader: https://discuss.pytorch.org/t/how-to-prefetch-data-when-processing-with-gpu/548/18
# https://discuss.pytorch.org/t/dataloader-relationship-between-num-workers-prefetch-factor-and-type-of-dataset/117735
# https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/

# Batch Size
# https://github.com/jwyang/faster-rcnn.pytorch/issues/305

# TODO: Test section and bar charts: https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk
# Mixed Precision: https://arxiv.org/abs/1710.03740

# https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_VISIBLE_DEVICES=1 python main.py ../experiments/corrections_batch_run/NetHVF.config

config_path = sys.argv[1]
print(f'Config path: {config_path}') # Cannot log before config loaded!

config = helper_functions.open_config(config_path)
config["train_op"] = True

helper_functions.trigger_torch_optimisations(torch, config)
# Dont use this! https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html
# if helper_functions.using_distributed_parallelise(config):
#   torch.cuda.set_device(helper_functions.get_cuda_device_id(config))

helper_functions.clear_gpu(torch, config, pause=False)

# TODO: https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
if __name__ == '__main__':
  if config["cuda_spawn"]:
    try:
      torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
      pass

  # TODO: Reset RND
  if (config["use_wandb"]):
    import wandb
    # wandb.init(project="cookbook-phoenix")
    wandb_run = wandb.init(project=config["wandb_project_name"], reinit=True)
    wandb_config = wandb.config
    wandb_config.learning_rate = config["lr"]

    run_name = f'[{config["wandb_name_prefix"]}]{config["model_name"]}-{config["dataset_name"]}-{wandb.run.name}'
    config["run_name"] = run_name
    wandb.run.name = config["run_name"]
  else:
    wandb = None
    run_name = f'{config["model_name"]}-{config["dataset_name"]}-{helper_functions.randomword(5)}'

  # You have to specify the run_name in the config
  if config["load_model"]:
    config["new_run_name"] = f'{config["run_name"]}-loaded-{helper_functions.randomword(5)}'
    config["original_run_name"] = config["run_name"]
  else:
    config["run_name"] = run_name

  helper_functions.set_default_gpu_id(config)

  if helper_functions.using_parallelise(config):
    log(f'Expandable Segments Set: False', config)
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')
  else:
    log(f'Expandable Segments Set: True', config)
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')

  # Save Seeds
  # TODO: Other seeds
  # config["torch_manual_seed"] = torch.manual_seed

  # Can only log from this point onwards because the save path can now be computed
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
  # Compile model
  if config["load_model"]:
    model, opt, scaler = torch_models.open_model(config, model_eval=False)
    _opt, loss_func = torch_optimisers.fetch_loss_opt_func(config, model)
  else:
    log(f'Initialise first model ...', config)
    model = torch_models.compile_model(config)

    opt, loss_func = torch_optimisers.fetch_loss_opt_func(config, model)

    # Sanity Checks
    # torcheck.register(opt)
    # torcheck.add_module(
    #   model,
    #   module_name=config["torcheck"]["module_name"],
    #   changing=config["torcheck"]["changing"],
    #   output_range=config["torcheck"]["output_range"],
    #   check_nan=config["torcheck"]["check_nan"],
    #   check_inf=config["torcheck"]["check_inf"],
    # )

    # See later model finding
    # torcheck.disable()
    # torcheck.enable()

    # torcheck.verbose_on()
    # torcheck.verbose_off()

  model = model.to(device=dev, dtype=torch.float32, non_blocking=config["non_blocking"])

  if helper_functions.using_parallelise(config):
    config["use_data_parallel"] = False

  model_stats = summary(model, input_size=(config["train_batch_size"], config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  log(model_stats, config)

  if helper_functions.using_parallelise(config):
    config["use_data_parallel"] = True

  dtype = helper_functions.compute_model_dtype(config)
  model = helper_functions.parallelise_model(model, config)

  model = model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  # Estimate total memory usage
  estimated_total_size_of_model = float(f'{model_stats}'.split("\n")[-2].split(" ")[-1])

  if train_workers > 0:
    estimated_total_memory_usage = estimated_total_size_of_model * train_workers
  else:
    estimated_total_memory_usage = estimated_total_size_of_model

  log(f"Estimated total memory usage: {estimated_total_memory_usage} MB", config)
  log('=========================================================================================================', config)
  if config["compile_only"] == True:
    raise Exception("End Early!")

  ################################################################################
  # Data Processing
  start_data_time = time.time()
  train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds = data_helpers.data_processing(config, dev)
  log(f'Data setup time: {time.time() - start_data_time} secs.', config)

  ################################################################################
  # Find best model to fully train
  log('=========================================================================================================', config)
  log(f'Find best of {config["initialisation_number"]} random models for {config["initial_epochs"]} initial epochs ...', config)
  if config["initialisation_number"] > 1:
    model_metrics = []

    # torcheck.disable()
    # torcheck.enable()

    append_path = '/init_models////model'
    model_path = helper_functions.compute_model_save_path(config, epoch=0, append_path=append_path)
    helper_functions.detect_or_create_folder(model_path.split('////')[0])
    log(f'Initial model save path: {model_path}', config)

    top_model_idex = -1
    top_model_val_loss = 1000000000 # Make it super huge
    top_model_path = ''
    top_model_init_path = ''
    found_nan = False

    model = torch_models.compile_model(config)
    dtype = helper_functions.compute_model_dtype(config)
    model = helper_functions.parallelise_model(model, config)

    model = model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

    for i in tqdm.tqdm(range(config["initialisation_number"])):
      helper_functions.clear_gpu(torch, config, pause=False)

      model = torch_models.randomise_weights(model, config)
      opt, loss_func = torch_optimisers.fetch_loss_opt_func(config, model)
      scaler = torch.cuda.amp.GradScaler(enabled=config["mixed_precision"])

      # Save initial model before training
      torch_models.save_model(i, model, opt, scaler, config, all_batch_metrics={}, append_path=f'{append_path}_init')

      all_batch_metrics, opt, scaler, found_nan = torch_trainer.fit(config["initial_epochs"], model, loss_func, opt, train_dl, valid_dl, config, wandb, torcheck, skip_wandb=True, save_model=False, early_stop=True)

      selected_train_metric = "selected_train_loss" #'mse_weighted_train'
      if "selected_train_metric" in config:
        selected_train_metric = config["selected_train_metric"]

      train_loss = all_batch_metrics[-1][selected_train_metric]

      selected_val_metric = "selected_val_loss" #'mse_weighted_val'
      if "selected_val_metric" in config:
        selected_val_metric = config["selected_val_metric"]

      val_loss = all_batch_metrics[-1][selected_val_metric]

      log(f'Epoch {i}, {selected_train_metric}: {train_loss}, {selected_val_metric}: {val_loss}', config)
      if val_loss != None and not np.isnan(val_loss) and not np.isnan(train_loss) and top_model_val_loss > val_loss:
        log(f'Epoch {i}, Is the new best epoch!', config)
        top_model_val_loss = val_loss
        top_model_idex = i

        selected_model_path = helper_functions.compute_model_save_path(config, epoch=i, append_path=append_path)
        top_model_init_path = f"{helper_functions.compute_model_save_path(config, epoch=i, append_path=f'{append_path}_init')}.pth"
        top_model_path = f'{selected_model_path}.pth'

      # Save afterwards to save epochs and outliers
      torch_models.save_model(i, model, opt, scaler, config, all_batch_metrics=all_batch_metrics, append_path=append_path)

    # TODO: Remeber to add on the initial_epoch counts
    # config['epochs'] -= config["initial_epochs"] # TODO: Rather improve this in a better way
    log(f'Selected model {top_model_idex} as the best model to proceed.', config)
    log(f'Load the pre-trained model: {config["load_pretrained_model"]}', config)
    # TODO: Add in timing here

    if top_model_idex == -1:
      raise "No valid model was found!"

    # Load model
    helper_functions.copy_init_model(config, top_model_init_path)

    if config["load_pretrained_model"]:
      model, opt, scaler = torch_models.open_model(config, direct_path=top_model_path, model=model, model_eval=False)
    else:
      model, opt, scaler = torch_models.open_model(config, direct_path=top_model_init_path, model=model, model_eval=False)

    _opt, loss_func = torch_optimisers.fetch_loss_opt_func(config, model)

  pretrain_time = time.time()
  log(f'\n\nPre-train Time: {pretrain_time - start_time} secs.', config)
  ################################################################################
  # Train model
  log('=========================================================================================================', config)
  log('Train model fully ...', config)
  if (config["use_wandb"] and config["wandb_watch_freq"] > 0):
    wandb_run.watch(model, log=config["wandb_watch_log"], log_freq=config["wandb_watch_freq"], log_graph=config["wandb_watch_log_graph"])

  all_batch_metrics = None
  found_nan = False

  if not config["load_model"]:
    start_train_time = time.time()
    all_batch_metrics, opt, scaler, found_nan = torch_trainer.fit(config["epochs"], model, loss_func, opt, train_dl, valid_dl, config, wandb, torcheck)
    log(f'Train time: {time.time() - start_train_time} secs.', config)

  torch_models.save_final_model(model, opt, scaler, config, all_batch_metrics=all_batch_metrics)
  # memory_key = helper_functions.config_memory_key(config)
  if config["dataset"][memory_key][config["model_name"]]["train"]["persistent_workers"]:
    train_dl._iterator._shutdown_workers()

  if found_nan:
    raise "The train or validation loss is NaN!"

  helper_functions.clear_gpu(torch, config, pause=False)

  train_time = time.time()
  log(f'\n\nTrain Time: {train_time - pretrain_time} secs.', config)
  ################################################################################
  # https://docs.wandb.ai/library/init
  if (config["use_wandb"]):
    wandb_run.finish()
################################################################################
  # Data Sanity Check
  if config['sanity_check']:
    log('\n*** Data Sanity Check ***', config)
    log('\ndata_tracking (training data access):', config)

    # memory_key = helper_functions.config_memory_key(config)
    if config["dataset"][memory_key][config["model_name"]]["train"]["num_workers"] == 0:
      log(pd.DataFrame(train_ds.data_tracking).value_counts('Use Count'), config)
    else:
      log('More than one training worker! Skipping test.', config)

    log('\ndata_tracking (validation data access):', config)
    log(pd.DataFrame(valid_ds.data_tracking).value_counts('Use Count'), config)

  helper_functions.clear_gpu(torch, config, pause=False)

  # Test Model
  torch_tester.run_model_tests(config)
  helper_functions.clear_gpu(torch, config, pause=False)

  torch_tester.run_model_tests(config, best_autonomy=True)
  log(f'\n\nTest Time: {time.time() - train_time} secs.', config)
################################################################################
  # TODO: Data sanity check
  # TODO: Test accuracy / how many correct vs invalid
  # TODO: Jupyter notebook
  # TODO: Show single input image

  # TODO: Checkpoints
  # TODO: RND seeds
  # torch.manual_seed(your_seed)

  # TODO: Save data Splits
  # TODO: Load data splits

################################################################################
# Uncomment when needed
  # benchmark
  # TODO: something with benchmark time
  # TODO: Double check output observations
  # TODO: Benchmark with higher res
  # TODO: Benchmark defaults

  # if config['benchmark']:
    # log('=========================================================================================================', config)
    # log('Run benchmark ...', config)
    # start_benchmark_time = time.time()

    # simulator_command = "docker run --privileged --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla /bin/bash ./CarlaUE4.sh Town01 -benchmark -fps=10 -RenderOffScreen"
    # simulator_process = Popen(simulator_command, shell=True)
    # time.sleep(10)

    # try:
      # benchmark_time = benchmark_agent.run_benchmark_v1(config)
    # except Exception as e:
      # log("Benchmark V1 failed!", config)
      # simulator_process.kill()

      # raise e

    # simulator_process.kill()
    # log(f'Benchmark time: {time.time() - start_benchmark_time} secs.', config)
################################################################################
  # Inside main loop:
  end_time = time.time()
  total_time = end_time - start_time

  config["total_time"] = total_time
  log(f'\n\nTotal Time: {config["total_time"]} secs.', config)

  helper_functions.clear_gpu(torch, config, pause=False)
################################################################################
