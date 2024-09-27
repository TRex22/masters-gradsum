import os
import math
import re

from pathlib import Path

import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as jnp
# config.update('jax_enable_x64', True)

import pandas as pd

import image_functions
import data_functions

# DataSet opening methods
def udacity_generator(meta, outputs, batch_size, options={ "shape": (48, 64, 3) }):
  # processed_data = process_udacity_data(raw_dataset, options)
  # yield [read_images(meta), data_functions.process_outputs(meta, outputs)]
  single_image = image_functions.read_image(meta.iloc[0], options)
  im_shape = single_image.shape
  dim = np.array([len(meta), im_shape[0], im_shape[1], im_shape[2]])

  for i in range(meta.shape[0]):
    x = image_functions.read_image(meta.iloc[i])
    y = np.array([data_functions.process_outputs(meta.iloc[i], outputs)])

    yield x,y

def udacity_meta(base_path, train_val_test_split, outputs, options={}, filetype='.jpg', data_folders=['']):
  raw_meta = construct_udacity_meta_dataset_from_file(base_path, filetype=filetype, data_folders=data_folders)

  # Drop any missing images
  meta = data_functions.drop_missing_data(raw_meta)

  # TODO:
  zero_drop_percentage = 0.9 # TODO: Make Configurable

  if 'zero_drop_percentage' in options:
    zero_drop_percentage = options.zero_drop_percentage

  # Drop where steering angle is 0.0  90% of the time
  meta = data_functions.drop_dataframe_if_zero_angle(meta, percent=zero_drop_percentage)

  print(f'Number of data points after drop: { meta.shape[0] }')

  return data_functions.split_meta_data(meta, train_val_test_split)

def udacity(base_path, train_val_test_split, outputs):
  train_meta, val_meta, test_meta = udacity_meta(
    base_path,
    train_val_test_split,
    outputs,
    { zero_drop_percentage: 0.9 }
  )

  y = [
    data_functions.process_outputs(train_meta, outputs),
    data_functions.process_outputs(val_meta, outputs),
    data_functions.process_outputs(test_meta, outputs)
  ]

  # print('Read images')

  x = [
    image_functions.read_images(train_meta),
    image_functions.read_images(val_meta),
    image_functions.read_images(test_meta)
  ]

  return [x,y]

# Using the interpolated results for now
def construct_udacity_meta_dataset_from_file(base_path, filetype='.jpg', data_folders=[''], cooked_path=''):
  full_path_raw_folders = [os.path.join(base_path, f) for f in data_folders]

  dataframes = []
  for folder in full_path_raw_folders:
    current_dataframe = pd.read_csv(
      os.path.join(folder, 'interpolated.csv'), sep=',' # TODO: Check steering.csv
      # os.path.join(folder, 'steering.csv'), sep=','
    )

    current_dataframe['Folder'] = folder
    current_dataframe['Cooked Folder'] = re.sub(base_path, cooked_path, folder)

    dataframes.append(current_dataframe)

  meta_dataset = pd.concat(dataframes, axis=0)
  print(f'Number of data points: {meta_dataset.shape[0]}')

  meta_dataset = meta_dataset.rename(columns={'speed': 'Speed'})
  meta_dataset = meta_dataset.rename(columns={'angle': 'Steering'})

  meta_dataset['Is North'] = meta_dataset.apply(
    lambda r: 'north' in r['Folder'], axis=1
  )

  meta_dataset['Cooked Path'] = meta_dataset.apply(
    lambda r: construct_cooked_file_path(r, frame='center', filetype=filetype), axis=1
  )

  # Add-in filepath
  meta_dataset['Full Path'] = meta_dataset.apply(
    lambda r: construct_file_path(r, frame='center', filetype=filetype), axis=1
  )

  # Check if file is present
  meta_dataset['Cooked File Presence'] = meta_dataset.apply(
    lambda r: data_functions.check_file_presence(r, idex='Cooked Path'), axis=1
  )

  meta_dataset['File Presence'] = meta_dataset.apply(
    lambda r: data_functions.check_file_presence(r), axis=1
  )

  meta_dataset['Is Swerve'] = meta_dataset.apply(
    lambda r: data_functions.is_swerve(r), axis=1
  )

  meta_dataset['Split Folder'] = meta_dataset.apply(
    lambda r: temporal_split(r), axis=1
  )

  return meta_dataset

def compute_udacity_cooked_relative_path(base_path, meta_dataset, i, detect_path=False):
  is_north = meta_dataset.iloc[i]['Is North']

  folder = 'south'
  if is_north:
    folder = 'north'

  frame = 'center' # TODO: Left and right
  timestamp = meta_dataset.iloc[i]['timestamp']
  filetype = '.jpg' # TODO: Make configurable

  folder_path = f'{base_path}/{folder}/{frame}'
  # print(f'Folder Path: {folder_paths}')

  data_functions.check_and_create_dir(folder_path)

  return f'{folder_path}/{timestamp}{filetype}'

def construct_cooked_file_path(single_meta, frame='center', filetype='.jpg'):
  file = single_meta[1]
  folder = single_meta['Cooked Folder']
  frame = 'center' # TODO: Left and right

  filepath = f'{folder}/{frame}/{file}{filetype}'
  filepath = re.sub('.jpg', filetype, filepath)

  # print(filepath)

  return filepath

def construct_file_path(single_meta, frame='center', filetype='.jpg'):
  file = single_meta[1]
  folder = single_meta['Folder']
  frame = 'center' # TODO: Left and right

  filepath = f'{folder}/{frame}/{file}{filetype}'
  # print(filepath)

  return filepath

def temporal_split(single_meta):
  filepath = single_meta["Full Path"]

  if filepath.__contains__("north"):
    return "north"
  elif filepath.__contains__("south"):
    return "south"
  else:
    return "south" # Safe default?
