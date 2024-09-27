import os
import re
from os import listdir
from os.path import isfile, join

# import re
import math
import torch
import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as jnp
# config.update('jax_enable_x64', True)

import pandas as pd
from pathlib import Path

from helper_functions import log
import image_functions
import torch_optimisers

def check_and_create_dir(dirpath, remove_filename=False):
  if remove_filename:
    dirpath = re.sub(dirpath.split('/')[-1], '', dirpath)

  if not (os.path.exists(dirpath)):
    os.makedirs(dirpath, exist_ok=True)
    return False
  else:
    return True

def touch_file(filepath):
  with open(filepath, 'a'):
    os.utime(filepath, None)

  return True

def save_meta(path, meta):
  meta.to_csv(path)
  return meta

def open_meta(path):
  return pd.read_csv(path, sep=',')

# previous_state = list(meta.iloc[i-1][['Steering', 'Throttle', 'Brake', 'Speed (kmph)']])
def compute_steering_angle(meta, label, interpolate_result=True):
  raw_values = meta[label]
  # print(raw_values)
  number_raw_values = raw_values.shape
  # print(number_raw_values)
  list_of_results = np.array(range(1, number_raw_values - 1, 1))

  if (interpolate_result):
    for i in range(1, number_raw_values - 1, 1):
      numerator = (raw_values[i] + raw_values[i-1] + raw_values[i+1])
      count = 3.0
      list_of_results[i] = list(numerator / count)
  else:
    list_of_results = raw_values

  return list_of_results

def is_swerve(single_meta, accuracy=1):
  return is_swerve_angle(single_meta["Steering"], accuracy=accuracy)

def is_swerve_angle(value, accuracy=1):
  # accuracy = 1 # 3 and 4 were tested with cityscapes

  # return round(value, accuracy) != 0
  # return torch.round(value, decimals=accuracy) != 0

  if isinstance(value, torch.Tensor):
    return value.round(decimals=accuracy) != 0.0
  else:
    return round(value, accuracy) != 0.0

def swerve_count(expected_batch):
  count = 0

  for i in range(len(expected_batch)):
    if is_swerve_angle(expected_batch[i]):
      count += 1

  return count

# TODO: Possibly optimise
def swerve_batch(expected_batch, yb):
  # swerve_count = swerve_count(expected_batch)

  output_expected_batch = []
  output_yb_batch = []

  for i in range(len(expected_batch)):
    if is_swerve_angle(expected_batch[i]):
      output_expected_batch.append(expected_batch[i].item())
      output_yb_batch.append(yb[i].item())

  return [torch.tensor(np.array(output_expected_batch)), torch.tensor(np.array(output_yb_batch))]

def pair_structured(pairs):
  return len(pairs) == 2 and len(pairs[0]) == len(pairs[1])

# TODO: Possibly optimise
def straight_batch(expected_batch, yb):
  # swerve_count = swerve_count(expected_batch)
  # straight_count = len(expected_batch) - swerve_count

  output_expected_batch = []
  output_yb_batch = []

  for i in range(len(expected_batch)):
    if not is_swerve_angle(expected_batch[i]):
      output_expected_batch.append(expected_batch[i].item())
      output_yb_batch.append(yb[i].item())

  return [torch.tensor(np.array(output_expected_batch)), torch.tensor(np.array(output_yb_batch))]

def compute_weighted_mse(expected_batch, yb):
  # Assume they equal in length
  mse_steering = 0.0
  mse_straight = 0.0
  weighted_mse = 0.0

  steering_count = 0
  straight_count = 0

  if len(expected_batch) == 1:
    if is_swerve_angle(expected_batch):
      mse_steering = torch_optimisers.mse_loss_func(yb, expected_batch)
      steering_count = 1

      return [steering_count, straight_count, mse_steering, mse_straight, weighted_mse]
    else:
      mse_straight = torch_optimisers.mse_loss_func(yb, expected_batch)
      straight_count = 1

      return [steering_count, straight_count, mse_steering, mse_straight, weighted_mse]

  expected_swerve_batch, yb_swerve_batch = swerve_batch(expected_batch, yb)
  mse_steering = torch_optimisers.mse_loss_func(yb_swerve_batch, expected_swerve_batch)
  steering_count = len(expected_swerve_batch)

  expected_straight_batch, yb_straight_batch = straight_batch(expected_batch, yb)
  mse_straight = torch_optimisers.mse_loss_func(yb_straight_batch, expected_straight_batch)
  straight_count = len(expected_straight_batch)

  weighted_mse += mse_steering
  weighted_mse += mse_straight

  return [steering_count, straight_count, mse_steering, mse_straight, weighted_mse]

def compute_autonomy_interventions(expected_batch, yb):
  # Sum of all things
  # abs(predicted - expected)

  epsilon = 0.1
  abs_difference = torch.abs(expected_batch - yb)

  number_of_interventions = 0

  if expected_batch.shape == torch.Size([]):
    if abs_difference > epsilon:
      return 1
    else:
      return 0

  for i in range(len(abs_difference)):
    # If the difference is larger than the error margin
    if abs_difference[i] > epsilon:
      number_of_interventions += 1

  return number_of_interventions

def compute_final_autonomy(sum_of_interventions, data_point_count):
  # intervention_time = 6
  intervention_time = 1 # Time when the AI was not in control is any frame above epsilon
  return (1 - (sum_of_interventions * intervention_time)/data_point_count) * 100

def check_file_presence(single_meta, idex='Full Path'):
  try:
    # print(single_meta[idex])
    image = Path(single_meta[idex])
    return image.is_file()
  except OSError as e:
    print(e)
    return False

def drop_missing_data(df):
  idex = df[df['File Presence'] == False].index.values

  start_idex = 0
  end_idex = idex.size

  idex = np.array(idex)[start_idex:end_idex]

  df.drop(idex, inplace=True) # TODO: Fix this

  return df

def drop_invalid_steering_data(df, config):
  if config["drop_invalid_steering_angles"]:
    idex = df[df['Steering'] > 1.0].index.values

    start_idex = 0
    end_idex = idex.size

    idex = np.array(idex)[start_idex:end_idex]

    df.drop(idex, inplace=True) # TODO: Fix this

    idex = df[df['Steering'] < -1.0].index.values

    start_idex = 0
    end_idex = idex.size

    idex = np.array(idex)[start_idex:end_idex]

    df.drop(idex, inplace=True) # TODO: Fix this

    log(f"Size of DataFrame after dropping invalid steering angles: {df.shape[0]}", config)
  return df

def drop_missing_segmentation_data(df):
  idex = df[df['Segmentation Presence'] == False].index.values

  start_idex = 0
  end_idex = idex.size

  idex = np.array(idex)[start_idex:end_idex]

  df.drop(idex, inplace=True) # TODO: Fix this

  return df

def drop_test_data(df):
  if 'category' not in df.index:
    return df

  # Have to drop test as there are no segmentation maps
  idex = df[df['category'] == 'test'].index.values

  start_idex = 0
  end_idex = idex.size

  idex = np.array(idex)[start_idex:end_idex]

  df.drop(idex, inplace=True)

  return df

def drop_missing_cooked_data(df):
  idex = df[df['Cooked File Presence'] == False].index.values

  start_idex = 0
  end_idex = idex.size

  idex = np.array(idex)[start_idex:end_idex]

  df.drop(idex , inplace=True)

  return df

def drop_existing_cooked_data(df):
  idex = df[df['Cooked File Presence'] == True].index.values

  start_idex = 0
  end_idex = idex.size

  idex = np.array(idex)[start_idex:end_idex]

  df.drop(idex , inplace=True)

  return df

def drop_dataframe_if_zero_angle(df, percent=0.9, precision=2, ignore_value='', selected_label='Is Swerve'):
  # truncated_values = trunc(df['Steering'], decs=precision)

  percentage_to_keep = (1.0 - percent)
  zero_angle_values_to_keep = df[df[selected_label] == False].sample(frac=percentage_to_keep)

  steering_angle_df = df[df[selected_label] == True]

  return pd.concat([steering_angle_df, zero_angle_values_to_keep], ignore_index=True).sample(frac=1.0)

# https://stackoverflow.com/questions/42021972/truncating-decimal-digits-numpy-array-of-floats
def trunc(values, decs=0):
  return pd.DataFrame(np.trunc(np.array(values)*10**decs)/(10**decs))

def split_meta_data(meta, train_val_test_split, config):
  size = len(meta)

  # train = train.sample(frac=config["sample_percentage"]).reset_index(drop=True) # Randomise

  # TODO: Handle other DataSets
  # TODO: Allow for train_extra to be dropped
  # TODO: "horizontal_flip_data": true,

  if config["split_by_temporal_lines"] and config["dataset_name"] == "cityscapes":
    train = meta[meta['Split Folder'] == 'train']

    original_val = meta[meta['Split Folder'] == 'val']
    original_test = meta[meta['Split Folder'] == 'test']

    if config["combine_val_test"]:
      combined_df = original_val.combine_first(original_test)
      combined_df = combined_df.sample(frac=1.0).reset_index(drop=False)

      test_split = train_val_test_split[2] / train_val_test_split[1]
      val_split = 1.0 - test_split

      val_idex = int(val_split * len(combined_df))
      test_idex = val_idex + int(test_split * len(combined_df))

      val = combined_df.iloc[0:val_idex]
      test = combined_df.iloc[val_idex:test_idex]
    else:
      val = original_val
      test = original_test
  elif config["split_by_temporal_lines"] and config["dataset_name"] == "udacity" and config["combine_val_test"]:
    train = meta[meta['Split Folder'] == 'north']

    combined_df = meta[meta['Split Folder'] == 'south']
    combined_df = combined_df.sample(frac=1.0).reset_index(drop=False)

    test_split = train_val_test_split[2] / train_val_test_split[1]
    val_split = 1.0 - test_split

    val_idex = int(val_split * len(combined_df))
    test_idex = val_idex + int(test_split * len(combined_df))

    val = combined_df.iloc[0:val_idex]
    test = combined_df.iloc[val_idex:test_idex]
  else:
    train_p, val_p, test_p = train_val_test_split

    meta = meta.sample(frac=1.0).reset_index(drop=False)

    train_idex = int(train_p * size)
    val_idex = train_idex + int(val_p * size)
    test_idex = val_idex + int(test_p * size)

    train = meta.iloc[0:train_idex]
    val = meta.iloc[train_idex:val_idex]
    test = meta.iloc[val_idex:]

  # TODO: Drop only train data here - see: data_helpers.py:258

  return [train, val, test]

def process_outputs(meta, outputs, labels=['Steering', 'Speed (kmph)', 'Speed']):
  # ['Steering', 'Throttle', 'Brake', 'Speed (kmph)']
  # print(f'meta: {meta}')
  # print(f'meta[\'Steering\']: {meta[labels[0]]}')

  if (outputs == 1):
    final_data_y = meta[labels[0]]
  elif (outputs == 2):
    final_data_y = meta[labels]

  return final_data_y

def list_files(path):
  return [f for f in listdir(path) if isfile(join(path, f))]
