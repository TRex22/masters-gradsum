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

import helper_functions
environment = helper_functions.check_environment()

import image_functions
import data_functions

# base_paths:
# leftImg8bit
# leftImg8bit_blurred
# vehicle
# gtFine
# gtCoarse
# camera
# gtBboxCityPersons

# TODO: Test float32 vs float16

# DataSet opening methods
def cityscapes_generator(meta, outputs, batch_size, options={ "shape": (50, 100, 3) }):
  # processed_data = process_cityscapes_data(raw_dataset, options)
  # yield [read_images(meta), data_functions.process_outputs(meta, outputs)]
  single_image = read_image(meta.iloc[0], options)
  im_shape = single_image.shape
  dim = np.array([len(meta), im_shape[0], im_shape[1], im_shape[2]])

  for i in range(meta.shape[0]):
    x = read_image(meta.iloc[i])
    y = np.array([data_functions.process_outputs(meta.iloc[i], outputs)])

    yield x,y

def cityscapes_meta(base_path, train_val_test_split, outputs, options={}, filetype='.png', data_folders=['']):
  raw_meta = construct_cityscapes_meta_dataset_from_file(base_path, filetype=filetype)

  # Drop any missing images
  meta = data_functions.drop_missing_data(raw_meta)

  # TODO:
  zero_drop_percentage = 0.9 # TODO Make configurable

  if 'zero_drop_percentage' in options:
    zero_drop_percentage = options.zero_drop_percentage

  # Drop where steering angle is 0.0  90% of the time
  meta = data_functions.drop_dataframe_if_zero_angle(meta, percent=zero_drop_percentage)

  print(f'Number of data points after drop: { meta.shape[0] }')

  return data_functions.split_meta_data(meta, train_val_test_split)

def cityscapes(base_path, train_val_test_split, outputs):
  train_meta, val_meta, test_meta = cityscapes_meta(
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
    read_images(train_meta),
    read_images(val_meta),
    read_images(test_meta)
  ]

  return [x,y]

# Using the interpolated results for now
# TODO: Create Data-classes
def construct_cityscapes_meta_dataset_from_file(base_path, filetype='.png', data_folders=[''], cooked_path=''):
  full_path_raw_folders = [os.path.join(base_path, f) for f in data_folders]

  dataframes = []
  for folder in full_path_raw_folders:
    current_dataframe = pd.read_csv(
      os.path.join(folder, 'telemetry.csv'), sep=','
    )

    # if (environment == 'windows'):
    #   # current_dataframe['json_file_path'] = current_dataframe.apply(
    #   #   lambda r: data_functions.change_to_windows_paths(r, 'json_file_path')
    #   # )

    #   current_dataframe['imagePath'] = current_dataframe['imagePath'].replace('/', re.escape('\\'), regex=True)

    # import pdb; pdb.set_trace()
    current_dataframe['Folder'] = folder
    # print(re.sub(base_path, cooked_path, folder))
    current_dataframe['Cooked Folder'] = re.sub(re.escape(base_path), re.escape(cooked_path), re.escape(folder))

    # TODO: meta['Cooked Segmentation Path']

    dataframes.append(current_dataframe)

  meta_dataset = pd.concat(dataframes, axis=0)
  # print(meta_dataset['Cooked Folder'][0])
  print(f'Number of data points: {meta_dataset.shape[0]}')

  # Rename yawRate to Steering
  # TODO: Speed
  # meta_dataset = meta_dataset.rename(columns={'yawRate': 'Steering'})

  meta_dataset['Is North'] = meta_dataset.apply(
    lambda r: 'north' in r['Folder'], axis=1
  )

  meta_dataset['Is Swerve'] = meta_dataset.apply(
    lambda r: data_functions.is_swerve(r), axis=1
  )

  meta_dataset['Cooked Path'] = meta_dataset.apply(
    lambda r: construct_cooked_file_path(r, filetype=filetype), axis=1
  )

  # Add-in filepath
  meta_dataset['Full Path'] = meta_dataset.apply(
    lambda r: construct_file_path(r, filetype=filetype), axis=1
  )

  meta_dataset['Full Segmentation Path'] = meta_dataset.apply(
    lambda r: construct_segmentation_file_path(r, filetype=filetype), axis=1
  )

  # Check if file is present
  meta_dataset['Cooked File Presence'] = meta_dataset.apply(
    lambda r: data_functions.check_file_presence(r, idex='Cooked Path'), axis=1
  )

  meta_dataset['File Presence'] = meta_dataset.apply(
    lambda r: data_functions.check_file_presence(r), axis=1
  )

  meta_dataset['Segmentation Presence'] = meta_dataset.apply(
    lambda r: data_functions.check_file_presence(r, idex='Full Segmentation Path'), axis=1
  )

  meta_dataset['Cooked Segmentation Path'] = meta_dataset.apply(
    lambda r: construct_cooked_segmentation_file_path(r), axis=1
  )

  meta_dataset['Cooked Edge Path'] = meta_dataset.apply(
    lambda r: construct_cooked_edge_file_path(r), axis=1
  )

  meta_dataset['Split Folder'] = meta_dataset.apply(
    lambda r: temporal_split(r), axis=1
  )

  return meta_dataset

def compute_cityscapes_cooked_relative_path(base_path, meta_dataset, i, detect_path=False):
  is_north = meta_dataset.iloc[i]['Is North']

  if (detect_path):
    folder = 'south'
    if is_north:
      folder = 'north'
  else:
    folder = ''

  if (environment == 'windows'):
    # TODO: Dataset config
    image_path = meta_dataset['imagePath'][i]
    category = meta_dataset['category'][i].lower()
    city = meta_dataset['city'][i].lower()

    folder_path = f'{base_path}\\{category}\\{city}' #.lower()
    data_functions.check_and_create_dir(folder_path)
  else:
    # TODO: Dataset config
    image_path = meta_dataset['imagePath'][i]
    category = meta_dataset['category'][i].lower()
    city = meta_dataset['city'][i].lower()

    folder_path = f'{base_path}/{category}/{city}' #.lower()
    data_functions.check_and_create_dir(folder_path)

  return f'{base_path}{image_path}' #.lower()

def read_images(meta_dataset):
  # Read all the images in order
  # Get image dim / force image resolutions

  single_image = read_image(meta_dataset.iloc[0])
  im_shape = single_image.shape
  dim = np.array([len(meta_dataset), im_shape[0], im_shape[1], im_shape[2]])

  dataset = np.empty(dim)
  for i in range(meta_dataset.shape[0]):
    dataset[i] = read_image(meta_dataset.iloc[i])

  return dataset

# TODO: Add options
# (72, 128, 3) # (66, 200, 3) # (144, 256, 3) # (59, 256, 3)
# TODO % rescaling
# 2048x1024
# 20 x 10
# 200 x 100
# TODO: Blurred check
# TODO: multi input?
# TODO: Segmentation fine/coarse
def read_image(single_meta, options={ "shape": (50, 100, 3) }):
  # dim_shift=(50, 100, 3)
  # roi = [76,135,0,255]
  shape = options["shape"]

  # filepath = construct_file_path(single_meta, frame='left')
  filepath = single_meta["Full Path"]

  # TODO: extract image processing
  # TODO: add in translations
  image = image_functions.open_image(filepath)
  image = image_functions.remove_alpha_channel(image)

  # If using small_data_raw comment out
  # image = image_functions.crop_image_with_roi(image, roi)
  # image = image_functions.normalise_image(image)
  # image = image_functions.rescale_resolution(image, shape[0], shape[1])

  # optional
  # image = image_functions.contrast_stretch(image)
  # image = image_functions.hist_norm(image)
  # image.set_shape(dim_shift)

  image = image_functions.rescale_pixels(image)
  image = image_functions.conditional_resize(image, shape)
  return image

def temporal_split(single_meta):
  filepath = single_meta["Full Path"]

  if filepath.__contains__("train"):
    return "train"
  elif filepath.__contains__("train_extra"):
    return "train"
  elif filepath.__contains__("test"):
    return "test"
  elif filepath.__contains__("val"):
    return "val"
  else:
    return "val" # Safe default?

def construct_cooked_segmentation_file_path(single_meta, filetype='.png'):
  file = single_meta["Segmentation Path"]
  folder = re.sub('leftImg8bit/', '', single_meta['Cooked Folder'])

  filepath = f'{folder}{file}'
  filepath = re.sub('.png', filetype, filepath)
  # print(filepath)

  return filepath

def construct_cooked_edge_file_path(single_meta, filetype='.png'):
  file = single_meta["Segmentation Path"]
  folder = re.sub('leftImg8bit/', '', single_meta['Cooked Folder'])

  data_functions.check_and_create_dir(f'{folder}/edge/')

  filepath = f'{folder}/edge/{file}'
  filepath = re.sub('.png', filetype, filepath)
  # print(filepath)

  return filepath

def construct_cooked_file_path(single_meta, filetype='.png'):
  file = single_meta["imagePath"]
  folder = single_meta['Cooked Folder']

  filepath = f'{folder}{file}'
  # print(filepath)

  return filepath

def construct_file_path(single_meta, filetype='.png'):
  file = single_meta["imagePath"]
  folder = single_meta['Folder']

  filepath = f'{folder}{file}'
  filepath = re.sub('.png', filetype, filepath)
  # print(filepath)

  return filepath

def construct_segmentation_file_path(single_meta, filetype='.png'):
  file = single_meta["Segmentation Path"]
  folder = re.sub('leftImg8bit/', '', single_meta['Folder'])

  filepath = f'{folder}{file}'
  filepath = re.sub('.png', filetype, filepath)
  # print(filepath)

  return filepath

# TODO
def merge_l_c_r_images(base_path):
  return {}
