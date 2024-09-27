import numpy as np

import pandas as pd
pd.options.mode.copy_on_write = True

import sys
sys.path.insert(1, './data_Functions/')

import cv2
import tqdm

import torch
import torchvision
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms as T

import datasets
import data_functions
import image_functions
import cooking_functions
import helper_functions
from helper_functions import log
from torch_dataset import CustomImageDataset

import gradcam_functions

# from numba import jit

def compute_dataset_string(config, dataset_name=None):
  if dataset_name:
    return f'{dataset_name}_{config["input_size"][1]}_{config["input_size"][2]}_{config["input_size"][0]}'

  return f'{config["dataset_name"]}_{config["input_size"][1]}_{config["input_size"][2]}_{config["input_size"][0]}'

# TODO: Make more generic - given a folder
# TODO: dataset_name
def load_segmented_data(config):
  log('Load Segmentation Maps and Input Images ...', config)
  dataset_string = compute_dataset_string(config, dataset_name=config["grad_cam_result_dataset"]) # Special segmentation sequence
  log(f'GradCAM result DataSet: {dataset_string}', config)

  base_data_path = datasets.fetch_original_base_path_from(dataset_string, win_drive=config["win_drive"], base_path_linux=config['base_data_path'])

  if config["cook_data"]:
    # roi = config["roi"][config["dataset_name"]]
    roi = config["roi"][config["grad_cam_result_dataset"]] # "cityscapes"
    cooking_functions.cook_dataset(dataset_string, detect_path=False, roi=roi, base_path_linux=config['base_data_path'], greyscale=config["convert_to_greyscale"], config=config)
    base_cook_data_path = datasets.fetch_base_save_path_from(dataset_string, win_drive=config["win_drive"], base_path_linux=config['base_data_path'])
  else:
    base_cook_data_path = base_data_path

  if config["cook_data"] and config["cook_only"]:
    raise Exception("End Early!")

  # Re-use code but manually do the drop here
  if config["dataset_name"] == "cityscapes":
    log("Open Cityscapes val set ...", config)
    # test_ds, test_dl = data_helpers.open_dataset_and_loader(config, dev, test_data_path, "test")
    data_frame_save_path = helper_functions.compute_base_save_path(config)
    meta_path = f"{data_frame_save_path}/val.csv"
    meta = helper_functions.open_dataframe(meta_path)
    meta = data_functions.drop_test_data(meta).reset_index()

    cooking_functions.cook_segmentation_data(config, meta, dataset_string, base_path_linux=config['base_data_path'])
    cooking_functions.cook_edge_data(config, meta, dataset_string, base_path_linux=config['base_data_path'])
  else:
    raw_meta = datasets.fetch_construct_from(config["grad_cam_result_dataset"], base_data_path, base_cook_data_path)

    # Remove test set - since its dummy data
    raw_meta = data_functions.drop_test_data(raw_meta).reset_index()

    if config["cook_data"]:
      meta = data_functions.drop_missing_cooked_data(raw_meta)
      cooking_functions.cook_segmentation_data(config, meta, dataset_string, base_path_linux=config['base_data_path'])
      cooking_functions.cook_edge_data(config, meta, dataset_string, base_path_linux=config['base_data_path'])
    else:
      meta = data_functions.drop_missing_data(raw_meta)

  # meta = data_functions.drop_invalid_steering_data(meta, config)

  # meta = data_functions.drop_missing_segmentation_data(raw_meta)
  # TODO: Move to dataset functions
  # if not "presentation_mode" in config:
  #   if config['grad_cam_result_dataset'] == 'cityscapes':
  #     meta = data_functions.drop_test_data(raw_meta).reset_index()

  #   log(f'Number of data points after drop: { meta.shape[0] }', config)

  if "grad_cam_drop_percentage" in config:
    if config["grad_cam_drop_percentage"]:
      meta = data_functions.drop_dataframe_if_zero_angle(meta, percent=config["grad_cam_drop_percentage"], precision=config["data_drop_precision"], ignore_value='').reset_index()
      log(f'Number of data points after drop (grad_cam_drop_percentage): { meta.shape[0] }', config)

  paired_images = []

  for i in range(len(meta)):
    img_path = meta.iloc[i]['Cooked Path'] # "Full Path"
    seg_path = meta.iloc[i]['Cooked Segmentation Path']
    edge_path = meta.iloc[i]['Cooked Edge Path']
    paired_images.append([img_path, seg_path, edge_path])

  log('Segmentation maps loaded.', config)
  return paired_images, meta

# TODO: Make more generic - given a folder
# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def load_video_sequence(config, dataset_name):
  log('Load Video Sequence ...', config)
  log(f'Cook Data: {config["cook_data"]}', config)

  dataset_string = compute_dataset_string(config, dataset_name=dataset_name) # Special video sequence
  base_data_path = datasets.fetch_original_base_path_from(dataset_string, win_drive=config["win_drive"], base_path_linux=config['base_data_path'])

  if config["cook_data"]:
    cooking_functions.cook_dataset(dataset_string, detect_path=False, roi=config["roi"][config["dataset_name"]], base_path_linux=config['base_data_path'], greyscale=config["convert_to_greyscale"], config=config)
    base_cook_data_path = datasets.fetch_base_save_path_from(dataset_string, win_drive=config["win_drive"], base_path_linux=config['base_data_path'])
  else:
    base_cook_data_path = base_data_path

  if config["cook_data"] and config["cook_only"]:
    raise Exception("End Early!")

  raw_meta = datasets.fetch_construct_from(dataset_string, base_data_path, base_cook_data_path)

  if config["cook_data"]:
    meta = data_functions.drop_missing_cooked_data(raw_meta)
  else:
    meta = data_functions.drop_missing_data(raw_meta)

  log(f'Number of data points after drop: { meta.shape[0] }', config)
  # meta = data_functions.drop_invalid_steering_data(meta, config)

  # TODO: Non-Cooked video data
  x_frames = []
  for img_path in meta['Cooked Path']:
    # x_frames.append(read_image(img_path).permute(0, 2, 1).float())

    image = image_functions.open_image(img_path)
    image = np.transpose(image_functions.open_image(img_path)) # , 2, 0, 1

    x_frames.append(np.array([image]))

  log('Video Sequence loaded.', config)
  return x_frames # x_dl

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def load_dataset(config, dev, other_dataset_name, batch_size=None, persistent_workers=None, num_workers=None, prefetch_factor=2):
  other_dataset_string = compute_dataset_string(config, dataset_name=other_dataset_name)
  base_data_path = datasets.fetch_original_base_path_from(other_dataset_string, win_drive=config["win_drive"], base_path_linux=config['base_data_path'])

  if config["cook_data"]:
    cooking_functions.cook_dataset(other_dataset_string, detect_path=False, roi=config["roi"][other_dataset_name], base_path_linux=config['base_data_path'], greyscale=config["convert_to_greyscale"], config=config)
    base_cook_data_path = datasets.fetch_base_save_path_from(other_dataset_string, win_drive=config["win_drive"], base_path_linux=config['base_data_path'])
  else:
    base_cook_data_path = base_data_path

  raw_meta = datasets.fetch_construct_from(other_dataset_string, base_data_path, base_cook_data_path)

  # Distribution of Swerve
  normal_meta = raw_meta[raw_meta['Is Swerve'] == False].sample(frac=1)
  swerve_meta = raw_meta[raw_meta['Is Swerve'] == True].sample(frac=1)

  log(f'\nNumber of normal_meta data points: { normal_meta.shape[0] }', config)
  log(f'Number of swerve_meta data points: { swerve_meta.shape[0] }', config)

  if config["cook_data"]:
    meta = data_functions.drop_missing_cooked_data(raw_meta)
  else:
    meta = data_functions.drop_missing_data(raw_meta)

  log(f'{other_dataset_name}: number of data points after drop: { meta.shape[0] }', config)
  # meta = data_functions.drop_invalid_steering_data(meta, config)

  # Distribution of Swerve
  normal_meta = raw_meta[raw_meta['Is Swerve'] == False].sample(frac=1)
  swerve_meta = raw_meta[raw_meta['Is Swerve'] == True].sample(frac=1)

  log(f'\nNumber of normal_meta data points (after drop): { normal_meta.shape[0] }', config)
  log(f'Number of swerve_meta data points (after drop): { swerve_meta.shape[0] }', config)

  memory_key = helper_functions.config_memory_key(config)

  config_batch_size = config["dataset"][memory_key][config["model_name"]]["test"]["batch_size"]
  if batch_size is not None:
    config_batch_size = batch_size
  log(f'Batch Size: {config_batch_size}', config)

  config_shuffle = config["dataset"][memory_key][config["model_name"]]["test"]["shuffle"]
  log(f'Shuffle: {config_shuffle}', config)

  config_num_workers = config["dataset"][memory_key][config["model_name"]]["test"]["num_workers"]
  if num_workers is not None:
    config_num_workers = num_workers
  log(f'num_workers: {config_num_workers}', config)

  config_drop_last = config["dataset"][memory_key][config["model_name"]]["test"]["drop_last"]
  log(f'drop_last: {config_drop_last}', config)

  config_persistent_workers = config["dataset"][memory_key][config["model_name"]]["test"]["persistent_workers"]
  if persistent_workers is not None:
    config_persistent_workers = persistent_workers
  log(f'persistent_workers: {config_persistent_workers}', config)

  config_prefetch_factor = config["dataset"][memory_key][config["model_name"]]["test"]["prefetch_factor"]
  if prefetch_factor is None:
    config_prefetch_factor = prefetch_factor
  elif prefetch_factor != 2: # Default
    config_prefetch_factor = prefetch_factor
  log(f'prefetch_factor: {config_prefetch_factor}', config)

  if config["output_key"] == "Classify":
    meta = set_binary_label(config, meta)

  y_vals = torch.tensor(np.array(meta[config['output_key']])).float()
  data_set = CustomImageDataset(meta['Cooked Path'], y_vals, config, dev, other_dataset_name)
  data_loader = DataLoader(data_set, batch_size=config_batch_size, shuffle=config_shuffle, num_workers=config_num_workers, drop_last=config_drop_last, persistent_workers=config_persistent_workers, prefetch_factor=config_prefetch_factor, pin_memory=config["pin_memory"])

  return [data_set, data_loader]

def data_processing_for_gradcam(config, dev):
  log(f'Cook Data: {config["cook_data"]}', config)
  base_data_path = datasets.fetch_original_base_path_from(config["dataset_string"], win_drive=config["win_drive"], base_path_linux=config['base_data_path'])

  if config["cook_data"]:
    # cooking_functions.cook_images(dataset_string, raw_meta, detect_path=False)
    cooking_functions.cook_dataset(config["dataset_string"], detect_path=False, roi=config["roi"][config["dataset_name"]], base_path_linux=config['base_data_path'], greyscale=config["convert_to_greyscale"], config=config)
    base_cook_data_path = datasets.fetch_base_save_path_from(config["dataset_string"], win_drive=config["win_drive"], base_path_linux=config['base_data_path'])
  else:
    base_cook_data_path = base_data_path

  # TODO: Automate
  # Re-use code but manually do the drop here
  raw_meta = datasets.fetch_construct_from(config["dataset_string"], base_data_path, base_cook_data_path)
  if config["cook_data"]:
    meta = data_functions.drop_missing_cooked_data(raw_meta)
  else:
    meta = data_functions.drop_missing_data(raw_meta)

  y = torch.tensor(np.array(raw_meta[config['output_key']])).float()

  if config['cook_data']:
    ds = CustomImageDataset(raw_meta['Cooked Path'], y, config, dev, "train")
  else:
    ds = CustomImageDataset(raw_meta['Full Path'], y, config, dev, "train")

  memory_key = helper_functions.config_memory_key(config)
  dl = DataLoader(ds, batch_size=config["dataset"][memory_key][config["model_name"]]["train"]["batch_size"], shuffle=False, num_workers=config["dataset"][memory_key][config["model_name"]]["train"]["num_workers"], drop_last=False, persistent_workers=config["dataset"][memory_key][config["model_name"]]["train"]["persistent_workers"], prefetch_factor=config["dataset"][memory_key][config["model_name"]]["train"]["prefetch_factor"], pin_memory=config["pin_memory"])

  log(f'Number of Data Points: {len(ds)}', config)
  return [dl, ds]

def data_processing_resume(config, dev):
  save_path = helper_functions.compute_base_save_path(config, original_path=True)

  train_path = f'{save_path}/train.csv'
  train_dl, train_ds = open_dataset_and_loader(config, dev, train_path, "train")

  valid_path = f'{save_path}/val.csv'
  valid_dl, valid_ds = open_dataset_and_loader(config, dev, valid_path, "valid")

  test_path = f'{save_path}/test.csv'
  test_dl, test_ds = open_dataset_and_loader(config, dev, test_path, "test")

  return [train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds]

# TODO: Check original shape
# TODO: Sanity checks on sizes
def data_processing(config, dev):
  log(f'Cook Data: {config["cook_data"]}', config)
  base_data_path = datasets.fetch_original_base_path_from(config["dataset_string"], win_drive=config["win_drive"], base_path_linux=config['base_data_path'])

  if config["cook_data"]:
    # cooking_functions.cook_images(dataset_string, raw_meta, detect_path=False)
    cooking_functions.cook_dataset(config["dataset_string"], detect_path=False, roi=config["roi"][config["dataset_name"]], base_path_linux=config['base_data_path'], greyscale=config["convert_to_greyscale"], config=config)
    base_cook_data_path = datasets.fetch_base_save_path_from(config["dataset_string"], win_drive=config["win_drive"], base_path_linux=config['base_data_path'])
  else:
    base_cook_data_path = base_data_path

  # TODO: Automate
  # Re-use code but manually do the drop here
  raw_meta = datasets.fetch_construct_from(config["dataset_string"], base_data_path, base_cook_data_path)
  # raw_meta = datasets.construct_carla_imitation_meta_dataset_from_file(base_data_path, filetype='.png', cooked_path='')

  # Drop where steering angle is 0.0 - % of the time
  ignore_value = 'Is Swerve'

  if config["cook_data"]:
    meta = data_functions.drop_missing_cooked_data(raw_meta)
  else:
    meta = data_functions.drop_missing_data(raw_meta)

  if not config["split_by_temporal_lines"] and config["dataset_name"] == "cityscapes":
    meta = meta.sample(frac=config["sample_percentage"]).reset_index(drop=True) # Randomise
    # meta = data_functions.drop_dataframe_if_zero_angle(meta, percent=config["zero_drop_percentage"], precision=config["data_drop_precision"], ignore_value=ignore_value)
    # log(f'\nNumber of data points (after drop): { meta.shape[0] }', config)

  # meta = data_functions.drop_invalid_steering_data(meta, config)
  # Distribution of Swerve
  normal_meta = raw_meta[raw_meta['Is Swerve'] == False].sample(frac=1)
  swerve_meta = raw_meta[raw_meta['Is Swerve'] == True].sample(frac=1)

  log(f'\nRaw-Meta Number of normal_meta data points: { normal_meta.shape[0] }', config)
  log(f'Raw-Meta Number of swerve_meta data points: { swerve_meta.shape[0] }', config)
  swerve_ratio = swerve_meta.shape[0] / normal_meta.shape[0]
  log(f'swerve_ratio: {swerve_ratio}', config)

  # Filter Coarse and Fine Samples
  if config["output_key"] == "Classify":
    if "grad_cam_coarse" in config:
      if config["grad_cam_coarse"]:
        meta = filter_coarse_only(meta, config)

    elif "grad_cam_fine" in config:
      if config["grad_cam_fine"]:
        meta = filter_fine_only(meta, config)

  selected_label = 'Is Swerve'
  train, val, test = data_functions.split_meta_data(meta, config["train_val_test_split"], config)
  # Setup Classification if enabled
  if config["output_key"] == "Classify":
    # Skip test set since its only place-holder data
    # Generate label in meta based on config
    # Assume that there are segmentation maps
    train = set_binary_label(config, train)
    val = set_binary_label(config, val)
    test = set_binary_label(config, test, placeholder=True)

    # Output key is used since we only look at one binary label
    # And code already configures models to use the output_key
    selected_label = config["output_key"]

  # TODO: Make drop configurable
  if "calculate_zero_drop_percentage_even" in config and config["calculate_zero_drop_percentage_even"]:
    log(f"calculate_zero_drop_percentage_even: {config['calculate_zero_drop_percentage_even']}", config)

    train_drop_percentage = calculate_drop_percentage(train, label=selected_label)
    val_drop_percentage = calculate_drop_percentage(val, label=selected_label)
    test_drop_percentage = calculate_drop_percentage(test, label=selected_label)
  else:
    train_drop_percentage = config["zero_drop_percentage"]
    val_drop_percentage = 0.0
    test_drop_percentage = 0.0

  # Guard clauses in-case the difference is so large
  if train_drop_percentage >= 1.0:
    train_drop_percentage = 0.0

  if val_drop_percentage >= 1.0:
    val_drop_percentage = 0.0

  if test_drop_percentage >= 1.0:
    test_drop_percentage = 0.0

  log(f"train_drop_percentage: {train_drop_percentage}", config)
  log(f"val_drop_percentage: {val_drop_percentage}", config)
  log(f"test_drop_percentage: {test_drop_percentage}", config)

  train = data_functions.drop_dataframe_if_zero_angle(train, percent=train_drop_percentage, precision=config["data_drop_precision"], ignore_value=ignore_value, selected_label=selected_label)

  val = val.sample(frac=config["sample_percentage"]).reset_index(drop=True) # Randomise
  val = data_functions.drop_dataframe_if_zero_angle(val, percent=val_drop_percentage, precision=config["data_drop_precision"], ignore_value=ignore_value, selected_label=selected_label)

  test = test.sample(frac=config["sample_percentage"]).reset_index(drop=True) # Randomise
  test = data_functions.drop_dataframe_if_zero_angle(test, percent=test_drop_percentage, precision=config["data_drop_precision"], ignore_value=ignore_value, selected_label=selected_label)

  log(f'\nNumber of train data points (after drop): { train.shape[0] }', config)
  log(f'Number of val data points: { val.shape[0] }', config)
  log(f'Number of test data points: { test.shape[0] }', config)

  # Save train val split
  if "save_datasets" in config and config["save_datasets"]:
    log("Save val train split", config)
    data_frame_save_path = helper_functions.compute_base_save_path(config)

    helper_functions.save_dataframe(f"{data_frame_save_path}/raw_meta.csv", raw_meta)
    helper_functions.save_dataframe(f"{data_frame_save_path}/meta.csv", meta)
    helper_functions.save_dataframe(f"{data_frame_save_path}/train.csv", train)
    helper_functions.save_dataframe(f"{data_frame_save_path}/val.csv", val)
    helper_functions.save_dataframe(f"{data_frame_save_path}/test.csv", test)

  # Setup the Tensors
  # https://jax.readthedocs.io/en/latest/_autosummary/jax.dlpack.to_dlpack.html
  # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
  # https://github.com/pytorch/pytorch/issues/32868
  # https://pytorch.org/docs/stable/dlpack.html
  # https://web.archive.org/web/20200522225333/https://docs-cupy.chainer.org/en/stable/reference/interoperability.html
  #
  # y_train_dlpack = dlpack.to_dlpack(np.array(train[config['output_key']]), take_ownership=True) # 'Speed (kmph)'
  # y_train = torch.tensor(torch.utils.dlpack.from_dlpack(y_train_dlpack)).float()
  y_train = torch.tensor(np.array(train[config['output_key']])).float() # 'Speed (kmph)'
  normal_meta = train[train['Is Swerve'] == False].sample(frac=1)
  swerve_meta = train[train['Is Swerve'] == True].sample(frac=1)

  log(f'\nTrain Number of normal_meta data points: { normal_meta.shape[0] }', config)
  log(f'Train Number of swerve_meta data points: { swerve_meta.shape[0] }', config)
  swerve_ratio = swerve_meta.shape[0] / normal_meta.shape[0]
  log(f'swerve_ratio: {swerve_ratio}', config)

  # x_valid = torch.tensor(val.index.tolist(), pin_memory=config["pin_memory"])
  # y_valid_dlpack = dlpack.to_dlpack(np.array(val[config['output_key']]), take_ownership=True)
  # y_valid = torch.tensor(torch.utils.dlpack.from_dlpack(y_valid_dlpack)).float()
  y_valid = torch.tensor(np.array(val[config['output_key']])).float()
  normal_meta = val[val['Is Swerve'] == False].sample(frac=1)
  swerve_meta = val[val['Is Swerve'] == True].sample(frac=1)

  log(f'\nVal Number of normal_meta data points: { normal_meta.shape[0] }', config)
  log(f'Val Number of swerve_meta data points: { swerve_meta.shape[0] }', config)

  if config["output_key"] != "Classify":
    swerve_ratio = swerve_meta.shape[0] / normal_meta.shape[0]
    log(f'swerve_ratio: {swerve_ratio}', config)

  # x_test  = torch.tensor(test.index.tolist(), pin_memory=config["pin_memory"])
  # y_test_dlpack = dlpack.to_dlpack(np.array(test[config['output_key']]), take_ownership=True)
  # y_test = torch.tensor(torch.utils.dlpack.from_dlpack(y_test_dlpack)).float()
  y_test = torch.tensor(np.array(test[config['output_key']])).float()
  normal_meta = test[test['Is Swerve'] == False].sample(frac=1)
  swerve_meta = test[test['Is Swerve'] == True].sample(frac=1)

  log(f'\nTest Number of normal_meta data points: { normal_meta.shape[0] }', config)
  log(f'Test Number of swerve_meta data points: { swerve_meta.shape[0] }', config)

  if config["output_key"] != "Classify":
    swerve_ratio = swerve_meta.shape[0] / normal_meta.shape[0]
    log(f'swerve_ratio: {swerve_ratio}', config)

  # TODO: Resolve segmentation for other DataSets
  if config['cook_data']:
    train_ds = CustomImageDataset(train['Cooked Path'], y_train, config, dev, "train")
    valid_ds = CustomImageDataset(val['Cooked Path'], y_valid, config, dev, "valid")
    test_ds = CustomImageDataset(test['Cooked Path'], y_test, config, dev, "test")
  else:
    train_ds = CustomImageDataset(train['Full Path'], y_train, config, dev, "train")
    valid_ds = CustomImageDataset(val['Full Path'], y_valid, config, dev, "valid")
    test_ds = CustomImageDataset(test['Full Path'], y_test, config, dev, "test")

  memory_key = helper_functions.config_memory_key(config)

  # Make sure config is logical
  if config["dataset"][memory_key][config["model_name"]]["train"]["persistent_workers"] == False and config["dataset"][memory_key][config["model_name"]]["train"]["prefetch_factor"] != None:
    config["dataset"][memory_key][config["model_name"]]["train"]["prefetch_factor"] = None

  train_dl = DataLoader(train_ds, batch_size=config["dataset"][memory_key][config["model_name"]]["train"]["batch_size"], shuffle=config["dataset"][memory_key][config["model_name"]]["train"]["shuffle"], num_workers=config["dataset"][memory_key][config["model_name"]]["train"]["num_workers"], drop_last=config["dataset"][memory_key][config["model_name"]]["train"]["drop_last"], persistent_workers=config["dataset"][memory_key][config["model_name"]]["train"]["persistent_workers"], prefetch_factor=config["dataset"][memory_key][config["model_name"]]["train"]["prefetch_factor"], pin_memory=config["pin_memory"])
  valid_dl = DataLoader(valid_ds, batch_size=config["dataset"][memory_key][config["model_name"]]["valid"]["batch_size"], shuffle=config["dataset"][memory_key][config["model_name"]]["valid"]["shuffle"], num_workers=config["dataset"][memory_key][config["model_name"]]["valid"]["num_workers"], drop_last=config["dataset"][memory_key][config["model_name"]]["valid"]["drop_last"], persistent_workers=config["dataset"][memory_key][config["model_name"]]["valid"]["persistent_workers"], prefetch_factor=config["dataset"][memory_key][config["model_name"]]["valid"]["prefetch_factor"], pin_memory=config["pin_memory"])

  test_batch_size = config["dataset"][memory_key][config["model_name"]]["test"]["batch_size"]
  if len(test_ds) < test_batch_size:
    test_batch_size = len(test_ds)

  test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=config["dataset"][memory_key][config["model_name"]]["test"]["shuffle"], num_workers=config["dataset"][memory_key][config["model_name"]]["test"]["num_workers"], drop_last=config["dataset"][memory_key][config["model_name"]]["test"]["drop_last"], persistent_workers=config["dataset"][memory_key][config["model_name"]]["test"]["persistent_workers"], prefetch_factor=config["dataset"][memory_key][config["model_name"]]["test"]["prefetch_factor"], pin_memory=config["pin_memory"])

  if config["cook_data"] and config["cook_only"]:
    raise Exception("End Early!")

  if config["warmup_data"]:
    train_warmup_check = 0
    valid_warmup_check = 0

    log("\nWarming up data ...", config)
    log('Training Data...', config)
    for cycle in range(config["warmup_cycles"]):
      for i, data in enumerate(train_dl):
        train_warmup_check = train_warmup_check + 1

    log('Validation Data...', config)
    for cycle in range(config["warmup_cycles"]):
      for i, data in enumerate(valid_dl):
        valid_warmup_check = valid_warmup_check + 1

    memory_key = helper_functions.config_memory_key(config)
    train_batch_size = config["dataset"][memory_key][config["model_name"]]["train"]["batch_size"]
    valid_batch_size = config["dataset"][memory_key][config["model_name"]]["valid"]["batch_size"]

    train_total = (train_warmup_check * train_batch_size) / config["warmup_cycles"]
    valid_total = (valid_warmup_check * valid_batch_size) / config["warmup_cycles"]

    log(f'Training data sanity check: batches {train_warmup_check}, total {train_total}', config)
    log(f'Validation data sanity check: batches {valid_warmup_check}, total {valid_total}', config)

    log("Toasty.", config)

    if config["output_key"] == "Classify":
      # Disable test since it only has placeholder data
      test_dl = None
      test_ds = None

  return [train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds]

def set_binary_label(config, df, placeholder=False):
  label = config["label"]

  # Quick and dirty group names
  group_names = ["void", "sky", "nature", "object", "construction", "vehicle", "human", "flat"]
  group_map = gradcam_functions.fetch_full_group_map()

  if placeholder:
    df[config["output_key"]] = df.apply(lambda r: 0.0, axis=1)
  else:
    if label in group_names:
      # Iterate through entire group of labels
      for single_label in group_map[label]:
        df[config["output_key"]] = df.apply(
          lambda r: fetch_binary_label_for(r, single_label, config), axis=1
        )
    else:
      df[config["output_key"]] = df.apply(
        lambda r: fetch_binary_label_for(r, label, config), axis=1
      )

  return df

# if "Coarse" in pair[1]:
def filter_coarse_only(df, config):
  # return df[df['Segmentation Path'].str.contains('Coarse')].reset_index(drop=True, inplace=True)
  # return pd.concat([df[df['Segmentation Path'].str.contains('Coarse')]], ignore_index=True).sample(frac=1.0)

  idex = df[df['Segmentation Path'].str.contains('Fine')].index.values

  start_idex = 0
  end_idex = idex.size

  idex = np.array(idex)[start_idex:end_idex]

  df.drop(idex, inplace=True)
  return df

def filter_fine_only(df, config):
  # return df[df['Segmentation Path'].str.contains('Fine')].reset_index(drop=True, inplace=True)
  # return pd.concat([df[df['Segmentation Path'].str.contains('Fine')]], ignore_index=True).sample(frac=1.0)

  idex = df[df['Segmentation Path'].str.contains('Coarse')].index.values

  start_idex = 0
  end_idex = idex.size

  idex = np.array(idex)[start_idex:end_idex]

  df.drop(idex, inplace=True)
  return df

def fetch_binary_label_for(single_meta, label, config):
  if config["output_key"] in single_meta and single_meta[config["output_key"]] == 1.0:
    return 1.0

  segmentation_path = single_meta['Cooked Segmentation Path']
  map = image_functions.open_image(segmentation_path)

  binary_map = gradcam_functions.generate_binary_map_for(label, map)

  if np.isin(255, binary_map):
    return 1.0

  return 0.0

def calculate_drop_percentage(meta, label='Is Swerve'):
  normal_meta = meta[meta[label] == False].sample(frac=1)
  swerve_meta = meta[meta[label] == True].sample(frac=1)

  normal_count = len(normal_meta)
  swerve_count = len(swerve_meta)

  amount_to_drop = abs(normal_count - swerve_count)

  return (amount_to_drop * 1.0) / normal_count

def open_dataset_and_loader(config, dev, path, name): # name is "test", "train" etc...
  log(f"Opening meta: {path} ...", config)
  df = helper_functions.open_dataframe(path)
  y_df = torch.tensor(np.array(df[config['output_key']])).float()

  if config['cook_data']:
    ds = CustomImageDataset(df['Cooked Path'], y_df, config, dev, name)
  else:
    ds = CustomImageDataset(df['Full Path'], y_df, config, dev, name)

  memory_key = helper_functions.config_memory_key(config)

  test_batch_size = config["dataset"][memory_key][config["model_name"]][name]["batch_size"]
  dl = DataLoader(ds, batch_size=test_batch_size, shuffle=config["dataset"][memory_key][config["model_name"]][name]["shuffle"], num_workers=config["dataset"][memory_key][config["model_name"]][name]["num_workers"], drop_last=config["dataset"][memory_key][config["model_name"]][name]["drop_last"], persistent_workers=config["dataset"][memory_key][config["model_name"]][name]["persistent_workers"], prefetch_factor=config["dataset"][memory_key][config["model_name"]][name]["prefetch_factor"], pin_memory=config["pin_memory"])

  return [ds, dl]
