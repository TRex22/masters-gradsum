import time
import tqdm
import numpy as np
import pandas as pd

import torch
from torchinfo import summary

import helper_functions
from helper_functions import log
import data_functions

import torch_trainer
import data_helpers

import torch_models
import torch_optimisers

def run_model_test(model, loss_func, config, dev):
  # Test DataSets
  test_outputs = {}

  dataset_losses = {}
  dataset_sizes = {}
  dataset_data_tracking = {}
  dataset_autonomy = {}

  for other_dataset_name in tqdm.tqdm(config["datasets"]):
    helper_functions.clear_gpu(torch, config, pause=False)
    start_other_test_time = time.time()

    log(f"Dataset: {other_dataset_name}", config)

    # config_batch_size = config["dataset"][memory_key][config["model_name"]]["test"]["batch_size"]
    persistent_workers = False
    num_workers = 0
    prefetch_factor = None
    batch_size = 64 #32 # 8 # override to make other datasets happy

    other_dataset, data_loader = data_helpers.load_dataset(config, dev, other_dataset_name, batch_size=batch_size, persistent_workers=persistent_workers, num_workers=num_workers, prefetch_factor=prefetch_factor)
    other_loss, test_outputs = run_dataset_test(model, loss_func, config, test_outputs, f'{other_dataset_name}_test', data_loader)

    dataset_losses[other_dataset_name] = other_loss
    dataset_sizes[other_dataset_name] = len(data_loader)
    dataset_data_tracking[other_dataset_name] = other_dataset.data_tracking
    dataset_autonomy[other_dataset_name] = test_outputs["test_autonomy"]

    log(f'{other_dataset_name} Test time: {time.time() - start_other_test_time} secs.', config)
    # test_loss_header = f'{test_loss_header},{other_dataset_name}'
    # test_loss_line = f'{test_loss_line},{other_loss}'

    # del other_dataset
    # del data_loader

  return [test_outputs, dataset_losses, dataset_autonomy]

def compute_autonomy(loss, ds):
  return (1 - ((loss * 6) / len(ds))) * 100

def run_model_tests(config, best_autonomy=False):
  if best_autonomy:
    log("Test best autonomy model!", config)
  else:
    log("Test best val loss model!", config)

  config["train_op"] = False

  best_model_path = ""
  model, _opt, _scaler = [None, None, None]

  if best_autonomy:
    best_model_path = f'{helper_functions.compute_model_save_path(config, epoch=config["best_val_autonomy_epoch"], append_path="best_val_autonomy_model")}.pth'
    model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=config["best_val_autonomy_epoch"], direct_path=best_model_path)
  else:
    best_model_path = f'{helper_functions.compute_model_save_path(config, epoch=config["best_model_epoch"], append_path="best_model")}.pth'
    model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=config["best_model_epoch"], direct_path=best_model_path)

  _opt, loss_func = torch_optimisers.fetch_loss_opt_func(config, model)

  model_stats = summary(model, input_size=(config["train_batch_size"], config["input_size"][0], config["input_size"][1], config["input_size"][2]), device=config["summary_device_name"], verbose=0)
  log(model_stats, config)

  log('==========================================================================================', config)
  log("\nTest Model", config)
  start_test_time = time.time()
  test_outputs = {}

  dev = helper_functions.fetch_device(config)

  _train_dl, _valid_dl, test_dl, _train_ds, _valid_ds, test_ds = data_helpers.data_processing(config, dev)
  data_frame_save_path = helper_functions.compute_base_save_path(config)

  if config["output_key"] == "Classify":
    test_data_path = f"{data_frame_save_path}/test.csv"
    test_ds, test_dl = data_helpers.open_dataset_and_loader(config, dev, test_data_path, "test")

    # Single tests
    model.eval()
    xt = torch.zeros([1, config["input_size"][0], config["input_size"][1], config["input_size"][2]], dtype=torch.float32)
    xt[0] = test_ds[0][0].clone().detach()
    yt = float(test_ds[0][1])
    ytp = float(model(xt.to(dev)))
    log(f'Test single: expected: {yt}, predicted: {ytp}', config)

    last_sample_index = len(test_ds) - 1
    xt[0] = test_ds[last_sample_index][0].clone().detach()
    yt = float(test_ds[last_sample_index][1])
    ytp = float(model(xt.to(dev)))
    log(f'Test single [{last_sample_index}]: expected: {yt}, predicted: {ytp}', config)

    # Test DataSets
    test_loss, test_outputs = run_dataset_test(model, loss_func, config, test_outputs, f'{config["dataset_name"]}_test', test_dl, best_autonomy=best_autonomy)

    test_loss_save_path = f'{helper_functions.compute_model_save_path(config)}_test_loss_best_loss_model.csv'
    if best_autonomy:
      test_loss_save_path = f'{helper_functions.compute_model_save_path(config)}_test_loss_best_autonomy_model.csv'

    test_loss_header = f'{config["dataset_name"]}'
    test_loss_line = f'{test_loss}'

    test_autonomy = test_outputs["test_autonomy"]

    if config['compute_other_test_loss']:
      other_datasets = [i for i in config["datasets"] if i != config['dataset_name']]

      dataset_losses = {}
      dataset_sizes = {}
      dataset_data_tracking = {}
      dataset_autonomy = {}

      for other_dataset_name in tqdm.tqdm(other_datasets):
        helper_functions.clear_gpu(torch, config, pause=False)
        start_other_test_time = time.time()

        if best_autonomy:
          model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=config["best_val_autonomy_epoch"], direct_path=best_model_path)
        else:
          model, _opt, _scaler = torch_models.open_model(config, model_eval=True, epoch=config["best_model_epoch"], direct_path=best_model_path)

        _opt, loss_func = torch_optimisers.fetch_loss_opt_func(config, model)

        # config_batch_size = config["dataset"][memory_key][config["model_name"]]["test"]["batch_size"]
        batch_size = config["dataset"]["available_12"][config["model_name"]]["test"]["batch_size"]
        persistent_workers = config["dataset"]["available_12"][config["model_name"]]["test"]["persistent_workers"]
        num_workers = config["dataset"]["available_12"][config["model_name"]]["test"]["num_workers"]
        prefetch_factor = config["dataset"]["available_12"][config["model_name"]]["test"]["prefetch_factor"]

        other_dataset, data_loader = data_helpers.load_dataset(config, dev, other_dataset_name, batch_size=batch_size, persistent_workers=persistent_workers, num_workers=num_workers, prefetch_factor=prefetch_factor)
        other_loss, test_outputs = run_dataset_test(model, loss_func, config, test_outputs, f'{other_dataset_name}_test', data_loader, best_autonomy=best_autonomy)

        dataset_losses[other_dataset_name] = other_loss
        dataset_sizes[other_dataset_name] = len(data_loader)
        dataset_data_tracking[other_dataset_name] = other_dataset.data_tracking
        dataset_autonomy[other_dataset_name] = test_outputs["test_autonomy"]

        log(f'{other_dataset_name} Test time: {time.time() - start_other_test_time} secs.', config)
        test_loss_header = f'{test_loss_header},{other_dataset_name}'
        test_loss_line = f'{test_loss_line},{other_loss}'

        # del other_dataset
        # del data_loader

      helper_functions.save_csv(test_loss_save_path, test_loss_header)
      helper_functions.save_csv(test_loss_save_path, test_loss_line)

    # Need to move wandb.finish below if want to use:
    # if (config["use_wandb"]):
    #   wandb.log(test_outputs)

  helper_functions.clear_gpu(torch, config, pause=False)

  ################################################################################
  if config['compute_other_test_loss']:
    log('==========================================================================================', config)
    log('Compute synthetic autonomy ...', config)

    original_test_outputs = test_outputs
    test_loss, test_outputs = run_dataset_test(model, loss_func, config, original_test_outputs, f'{config["dataset_name"]}_test', test_dl, best_autonomy=best_autonomy)
    test_autonomy = compute_autonomy(test_loss, test_ds)

    log(f'Synthetic test autonomy: {test_autonomy}%', config)

    autonomy_save_path = f'{helper_functions.compute_model_save_path(config)}_autonomy_best_loss_model.csv'
    if best_autonomy:
      autonomy_save_path = f'{helper_functions.compute_model_save_path(config)}_autonomy_best_autonomy_model.csv'

    autonomy_header = f'{config["dataset_name"]}'
    autonomy_line = f'{test_autonomy}'

    dataset_autonomy = {}
    dataset_data_tracking = {}

    main_dataset_name = config["dataset_name"]

    other_datasets = [i for i in config["datasets"] if i != config['dataset_name']]
    for other_dataset_name in tqdm.tqdm(other_datasets):
      config["dataset_name"] = other_dataset_name

      dataset_string = data_helpers.compute_dataset_string(config)
      config["dataset_string"] = dataset_string

      _train_dl, _valid_dl, other_test_dl, _train_ds, _valid_ds, other_test_ds = data_helpers.data_processing(config, dev)
      other_test_loss, test_outputs = run_dataset_test(model, loss_func, config, original_test_outputs, f'{other_dataset_name}_test', other_test_dl, best_autonomy=best_autonomy)

      other_autonomy = compute_autonomy(other_test_loss, other_test_ds)
      dataset_autonomy[other_dataset_name] = other_autonomy # test_outputs[f"{other_dataset_name}_test"]

      log(f'{other_dataset_name} count: {len(other_test_ds)}', config)
      log(f'{other_dataset_name} synthetic test autonomy: {dataset_autonomy[other_dataset_name]}%', config)
      autonomy_header = f'{autonomy_header},{other_dataset_name}'
      autonomy_line = f'{autonomy_line},{dataset_autonomy[other_dataset_name]}'

      dataset_data_tracking[other_dataset_name] = other_test_ds.data_tracking

      # Data Tracking
      log(f'\ndata_tracking ({other_dataset_name} data access):', config)
      log(pd.DataFrame(dataset_data_tracking[other_dataset_name]).value_counts('Use Count'), config)

    helper_functions.save_csv(autonomy_save_path, autonomy_header)
    helper_functions.save_csv(autonomy_save_path, autonomy_line)

    log(f'Test time: {time.time() - start_test_time} secs.', config)
    log(f'\ndata_tracking (test data access - {main_dataset_name}):', config)
    log(pd.DataFrame(test_ds.data_tracking).value_counts('Use Count'), config)

  helper_functions.clear_gpu(torch, config, pause=False)

def run_dataset_test(model, loss_func, config, test_outputs, test_name, data_loader, best_autonomy=False):
  log(f'Run test: {test_name}', config)

  dev = helper_functions.fetch_device(config)
  dtype = helper_functions.compute_model_dtype(config)
  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])
  model.eval()

  scaler = torch.cuda.amp.GradScaler(enabled=config["mixed_precision"])
  prediction_pairs = []
  losses = []

  with torch.no_grad():
    losses, batch_count, prediction_pairs, opts, scalers = zip(
      *[torch_trainer.loss_batch(model, scaler, config, loss_func, xb, yb) for xb, yb in tqdm.tqdm(data_loader)]
    )

  test_outputs = torch_trainer.compute_loss_functions(test_outputs, prediction_pairs, losses, batch_count, "test", dev, dtype, should_convert_to_tensor=True)

  selected_test_metric = "selected_test_loss"
  if "selected_test_metric" in config:
    selected_test_metric = config["selected_test_metric"]

  test_loss = test_outputs[selected_test_metric]

  if best_autonomy:
    log(f'{test_name} Test Loss (Best autonomy model): {test_loss}', config)
  else:
    log(f'{test_name} Test Loss (Best val loss model): {test_loss}', config)

  return [test_loss, test_outputs]
