import numpy as np
import tqdm
import copy

import torch
from torch import nn
from torch import autocast

import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../pyTorch_phoenix/')

import helper_functions
import data_functions

from helper_functions import log
from helper_functions import convert_to_tensor

import torch_models
import torch_optimisers

# from lightning.fabric import Fabric

def loss_batch(model, scaler, config, loss_func, xb, yb, opt=None, swerve_loss_only=False):
  # Original loss without AMP:
  # log(len(prediction), config)
  # log(len(yb), config)
  # if opt is not None:
  #   opt.zero_grad()
  #   loss.backward()
  #   opt.step()

  model = helper_functions.parallelise_model(model, config)

  # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
  if opt is not None:
    # https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
    # TODO: If unscaling my need to move to the end
    # opt.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
    for param in model.parameters(): # Optimisation to save n operations
      param.grad = None

    with autocast(config["summary_device_name"], enabled=config["mixed_precision"], cache_enabled=config["cache_enabled"]):
      prediction = model(xb).flatten()
      loss = loss_func(prediction, yb) # , reduction='mean'

    dev = helper_functions.fetch_device(config, verbose=False)
    dtype = helper_functions.compute_model_dtype(config)

    if dtype == torch.bfloat16:
      model.to(device=dev, dtype=torch.float16, non_blocking=config["non_blocking"])

    scaler.scale(loss).backward()

    if config["clip_grad_norm"]:
      # Unscales the gradients of optimizer's assigned params in-place
      scaler.unscale_(opt)

      # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
      # You may use the same value for max_norm here as you would without gradient scaling.
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_max_norm"])
      # torch.nn.utils.clip_grad_norm_(model.parameters())

    scaler.step(opt)
    scaler.update()

    model = helper_functions.parallelise_model(model, config)

    if dtype == torch.bfloat16:
      model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  else:
    with autocast(config["summary_device_name"], enabled=config["mixed_precision"], cache_enabled=config["cache_enabled"]):
      prediction = model(xb).flatten()
      loss = loss_func(prediction, yb)

  prediction_pairs = [prediction, yb]
  return loss.item(), len(xb), prediction_pairs, opt, scaler

def transpose(pairs):
  try:
    return pairs.T
  except:
    outer_batch_size = len(pairs)
    inner_batch_size = len(pairs[0][0])
    new_pairs = torch.zeros(2, outer_batch_size, inner_batch_size)

    for i in range(outer_batch_size):
      transpose_pairs = pairs[i][:]
      new_pairs[0][i] = transpose_pairs[0]
      new_pairs[1][i] = transpose_pairs[1]

    new_pairs = new_pairs.view(2, outer_batch_size*inner_batch_size)
    return new_pairs

def compute_total_count(batch_count):
  try:
    return sum(batch_count)
  except:
    return batch_count

def compute_loss_functions(log_outputs, pairs, losses, batch_count, step_name, dev, dtype, should_convert_to_tensor=True):
  steering_count = 0
  straight_count = 0

  mse_steering = 0.0
  mse_straight = 0.0
  weighted_mse = 0.0

  autonomy_sum_of_interventions = 0.0

  # selected_loss = np.sum(losses)
  selected_loss = np.sum(np.multiply(losses, batch_count)) / np.sum(batch_count)

  log_outputs[f"selected_{step_name}_loss"] = selected_loss

  if data_functions.pair_structured(pairs):
    pair_size = len(pairs[0])

    expected_batch = pairs[0]
    yb = pairs[1]
  else:
    transpose_pairs = transpose(pairs)
    pair_size = len(transpose_pairs[0])

    expected_batch = transpose_pairs[0]
    yb = transpose_pairs[1]

  if should_convert_to_tensor:
    expected_batch = convert_to_tensor(expected_batch, dev, dtype)
    yb = convert_to_tensor(yb, dev, dtype)

  log_outputs[f"{step_name}_mse_loss_func"] = torch_optimisers.mse_loss_func(expected_batch, yb)

  total_count = compute_total_count(batch_count)

  autonomy_sum_of_interventions = data_functions.compute_autonomy_interventions(expected_batch, yb)
  autonomy_sum_of_interventions = array_item(autonomy_sum_of_interventions * 1.0) # / sum(batch_count)
  # autonomy_sum_of_interventions = np.sum(np.multiply(autonomy_sum_of_interventions, batch_count)) / np.sum(batch_count)
  log_outputs[f'autonomy_sum_of_interventions_{step_name}'] = autonomy_sum_of_interventions

  autonomy = data_functions.compute_final_autonomy(autonomy_sum_of_interventions, total_count)
  log_outputs[f"{step_name}_autonomy"] = autonomy

  batch_steering_count, batch_straight_count, batch_mse_steering, batch_mse_straight, weighted_mse = data_functions.compute_weighted_mse(expected_batch, yb)

  steering_count = batch_steering_count
  straight_count = batch_straight_count

  # steering_count = steering_count / batch_count
  # straight_count = straight_count / batch_count

  mse_steering = batch_mse_steering
  mse_straight = batch_mse_straight

  log_outputs[f'steering_count_{step_name}'] = steering_count # np.sum(np.multiply(steering_count, batch_count)) / np.sum(batch_count)
  log_outputs[f'straight_count_{step_name}'] = straight_count # np.sum(np.multiply(straight_count, batch_count)) / np.sum(batch_count)

  log_outputs[f'mse_steering_{step_name}'] = array_item(detach_value(mse_steering) * 1.0)
  log_outputs[f'mse_straight_{step_name}'] = array_item(detach_value(mse_straight) * 1.0)

  # Confirm its correct
  assert pair_size == (straight_count + steering_count)
  log_outputs[f'mse_weighted_{step_name}'] = array_item(detach_value(weighted_mse) * 1.0)

  return log_outputs

def array_item(value):
  if isinstance(value, torch.Tensor):
    return value.item()
  return value

def detach_value(value):
  if isinstance(value, torch.Tensor):
    return value.item()

  return value

def epoch_to_save(epoch, config):
  is_specified_in_config = (epoch in config["grad_cam_epochs"])
  is_initial_epoch = (epoch) < config["grad_cam_initial_epochs_to_save"]

  return is_specified_in_config or is_initial_epoch

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, config, wandb, torcheck, skip_wandb=False, append_path="", save_model=True, early_stop=False, epoch_start=None):
  log("\nStart Training ...", config)

  found_nan = False

  opt_name = config["opt_name"]
  loss_name = config["loss_name"]

  all_batch_metrics = []

  pbar = tqdm.tqdm(total=epochs)
  scaler = torch.cuda.amp.GradScaler(enabled=config["mixed_precision"])

  best_config = config
  best_epoch = 0
  best_model = copy.deepcopy(model.cpu())
  best_opt = opt
  best_scaler = scaler
  best_val_loss = 1.0 # We dont care for the best models until the loss drops below 0.5

  best_val_autonomy_config = config
  best_val_autonomy_epoch = 0
  best_val_autonomy_model = copy.deepcopy(model.cpu())
  best_autonomy_opt = opt
  best_val_autonomy_scaler = scaler
  best_val_autonomy = 0.0

  dev = helper_functions.fetch_device(config)
  dtype = helper_functions.compute_model_dtype(config)
  model = helper_functions.parallelise_model(model, config)

  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  last_epoch = 0

  epoch_range = range(epochs)
  if epoch_start:
    epoch_range = range(epoch_start, epoch_start + epochs)

  for epoch in epoch_range:
    # if epoch == config["epoch_torcheck"]:
      # torcheck.verbose_on()
      # torcheck.enable()

    log_outputs = {}

    # https://torchmetrics.readthedocs.io/en/latest/
    # accuracy_fn = torchmetrics.Accuracy()

    # prev_model_params = model.parameters() # .state_dict()

    model = helper_functions.parallelise_model(model, config)
    model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])
    model = model.train()

    losses, batch_count, prediction_pairs, opts, scalers = zip(
      *[loss_batch(model, scaler, config, loss_func, xb, yb, opt) for xb, yb in train_dl]
    )

    opt = opts[-1]
    scaler = scalers[-1]

    log_outputs = compute_loss_functions(log_outputs, prediction_pairs, losses, batch_count, "train", dev, dtype, should_convert_to_tensor=True)

    # train_autonomy = data_functions.compute_final_autonomy(log_outputs["autonomy_sum_of_interventions_train"], len(prediction_pairs))
    # log_outputs["train_autonomy"] = train_autonomy

    curr_lr = opt.param_groups[0]["lr"]
    # current_model_params = model.parameters()

    # Validation Step
    model = model.eval()
    with torch.no_grad():
      losses, batch_count, prediction_pairs, _opts, _scalers = zip(
        *[loss_batch(model, scaler, config, loss_func, xb, yb) for xb, yb in valid_dl]
      )

    log_outputs = compute_loss_functions(log_outputs, prediction_pairs, losses, batch_count, "val", dev, dtype, should_convert_to_tensor=True)

    # val_autonomy = data_functions.compute_final_autonomy(log_outputs["autonomy_sum_of_interventions_val"], len(prediction_pairs))
    # log_outputs["val_autonomy"] = val_autonomy

    # https://docs.wandb.ai/guides/track/log
    # images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
    # wandb.log({"examples": images}
    # wandb.log({"gradients": wandb.Histogram(grads)})

    # tensormetrics
    # accuracy_epoch = accuracy_fn.compute()

    for key in log_outputs.keys():
      if isinstance(log_outputs[key], torch.Tensor):
        log_outputs[key] = log_outputs[key].item()

    if (config["use_wandb"]) and skip_wandb == False:
      # log_outputs set above

      # log_outputs = {
      #   # "gradients": wandb.Histogram(current_model_params)
      # }

      wandb.log(log_outputs)

    all_batch_metrics.append(log_outputs)

    if not config["grad_cam_epochs"] == None and config["compute_grad_cam_results"]:
      if epoch_to_save(epoch, config) and save_model:
        torch_models.save_model(epoch, model, opt, scaler, config, all_batch_metrics=all_batch_metrics, append_path=append_path)

    # TODO: Model difference metric
    mse_weighted_train = log_outputs['mse_weighted_train']
    mse_weighted_val = log_outputs["mse_weighted_val"]

    selected_train_metric = "selected_train_loss"
    if "selected_train_metric" in config:
      selected_train_metric = config["selected_train_metric"]

    selected_val_metric = "selected_val_loss"
    if "selected_val_metric" in config:
      selected_val_metric = config["selected_val_metric"]

    train_loss = log_outputs[selected_train_metric]
    val_loss = log_outputs[selected_val_metric]

    log(f'epoch: {epoch} {selected_train_metric}: {train_loss} {selected_val_metric}: {val_loss} lr: {curr_lr}', config)
    pbar.update(1)

    if train_loss == None or np.isnan(train_loss):
      found_nan = True
      log("Train loss is NaN!", config)
      break;

    if val_loss == None or np.isnan(val_loss):
      found_nan = True
      log("Val loss is NaN!", config)
      break;

    # if val_loss == None or np.isnan(mse_weighted_val):
    #   found_nan = True
    #   log("mse_weighted_val loss is NaN!", config)
    #   break;

    # Save Checkpoint
    if config["checkpoint"] and not early_stop: # If early_stop then its pre-training and we want to skip each checkpoint
      helper_functions.remove_checkpoint(config, append_path=append_path)
      torch_models.save_model(epoch, model, opt, scaler, config, all_batch_metrics=all_batch_metrics, append_path=append_path, checkpoint=True)

    if not early_stop and float(val_loss) < float(best_val_loss):
      log(f"Set new best model epoch: {epoch}, loss: {val_loss}", config)

      best_config = config
      best_epoch = epoch
      best_model = copy.deepcopy(model.cpu())
      best_opt = opt
      best_scaler = scaler
      best_val_loss = float(val_loss)

    val_autonomy = log_outputs["val_autonomy"]
    if not early_stop and float(val_autonomy) > float(best_val_autonomy):
      log(f"Set best val autonomy model epoch: {epoch}, autonomy: {float(val_autonomy)}", config)

      best_val_autonomy_config = config
      best_val_autonomy_epoch = epoch
      best_val_autonomy_model = copy.deepcopy(model.cpu())
      best_autonomy_opt = opt
      best_val_autonomy_scaler = scaler
      best_val_autonomy = float(val_autonomy)

    if not early_stop: # I.e. init models
      config["best_model_epoch"] = best_epoch
      config["best_val_autonomy_epoch"] = best_val_autonomy_epoch

    # Save last epoch
    last_epoch = epoch
    if early_stop or epoch == (epochs - 1):
      torch_models.save_model(epoch, model, opt, scaler, config, all_batch_metrics=all_batch_metrics, append_path=append_path, checkpoint=True)

    minimum_epoch_check = 20 # Makes sense to skip the first 20

    # Quick stop code:
    if epoch != 0 and best_epoch != 0 and epoch > minimum_epoch_check and "early_stop_val_loss" in config:
      distance_val_loss = epoch - best_epoch
      if config["early_stop_val_loss"] <= distance_val_loss:
        log(f"early_stop_val_loss reached! epoch: {epoch}, distance: {distance_val_loss}.", config)
        break;

    if epoch != 0 and best_epoch != 0 and epoch > minimum_epoch_check and "early_stop_autonomy" in config:
      distance_autonomy = epoch - best_val_autonomy_epoch
      if config["early_stop_autonomy"] <= distance_val_loss:
        log(f"early_stop_autonomy reached! epoch: {epoch}, distance_autonomy: {distance_autonomy}.", config)
        break;

  if not early_stop: # I.e. init models
    log(f"Saving best model epoch {best_epoch} ...", config)
    # torch_models.save_model(epoch, best_model, opt, scaler, config, all_batch_metrics=all_batch_metrics, append_path=append_path, checkpoint=True)
    torch_models.save_model(best_epoch, best_model, best_opt, best_scaler, best_config, all_batch_metrics=all_batch_metrics, append_path="best_model")
    # torch_models.save_model(best_epoch, best_model, best_opt, best_scaler, best_config, all_batch_metrics=all_batch_metrics, append_path=append_path)

    log(f"Saving best autonomy model epoch {best_val_autonomy_epoch} ...", config)
    torch_models.save_model(best_val_autonomy_epoch, best_val_autonomy_model, best_autonomy_opt, best_val_autonomy_scaler, best_val_autonomy_config, all_batch_metrics=all_batch_metrics, append_path="best_val_autonomy_model")

    [parameters_match, states_match] = torch_models.compare(best_model, best_val_autonomy_model, config, verbose=True)

    if parameters_match or states_match:
      log(f'\033[93mparameters_match: {parameters_match}, states_match: {states_match}', config)
    else:
      log(f'\033[94mparameters_match: {parameters_match}, states_match: {states_match}', config)

    print(f"\033[39m\033[49mDone!")

  log("Training complete.", config)

  if found_nan:
    log("Warning train loss or validation loss is NaN!", config)

  # del train_dl
  # del valid_dl
  # helper_functions.clear_gpu(torch, config)

  return [all_batch_metrics, opt, scaler, found_nan]
