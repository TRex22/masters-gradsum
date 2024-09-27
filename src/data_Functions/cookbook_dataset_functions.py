import os

import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as jnp
# config.update('jax_enable_x64', True)

import pandas as pd
import re

from pathlib import Path

import image_functions
import data_functions

# Some of the code used here comes from the Microsoft AutonomousDrivingCookbook
# https://github.com/microsoft/AutonomousDrivingCookbook
# train_val_test_split = [0.7, 0.2, 0.1]

# DataSet opening methods
def cookbook_generator(meta, outputs, batch_size, options={ "shape": (59, 256, 3) }):
  for i in range(meta.shape[0]):
    x = read_image(meta.iloc[i], options)
    y = np.array([data_functions.process_outputs(meta.iloc[i], outputs)])

    yield x,y

def cookbook_meta(base_path, train_val_test_split, outputs, options={}):
  raw_meta = construct_cookbook_meta_dataset_from_file(base_path)

  zero_drop_percentage = 0.9

  if 'zero_drop_percentage' in options:
    zero_drop_percentage = options.zero_drop_percentage

  # Drop where steering angle is 0.0  90% of the time
  meta = data_functions.drop_dataframe_if_zero_angle(raw_meta, percent=zero_drop_percentage)

  print(f'Number of data points after drop: { meta.shape[0] }')

  # Split the dataset into its two components
  normal_meta = meta[meta['Is Swerve'] == False].sample(frac=1)
  swerve_meta = meta[meta['Is Swerve'] == True].sample(frac=1)

  norm_split = data_functions.split_meta_data(normal_meta, train_val_test_split)
  swerve_split = data_functions.split_meta_data(swerve_meta, train_val_test_split)

  norm_train_meta, norm_val_meta, norm_test_meta = norm_split
  swerve_train_meta, swerve_val_meta, swerve_test_meta = swerve_split

  # Merge dataset back together
  train_meta = pd.concat([norm_train_meta, swerve_train_meta])
  val_meta = pd.concat([norm_val_meta, swerve_val_meta])
  test_meta = pd.concat([norm_test_meta, swerve_test_meta])

  return [train_meta, val_meta, test_meta]

def cookbook(base_path, train_val_test_split, outputs):
  train_meta, val_meta, test_meta = cookbook_meta(
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

def construct_cookbook_meta_dataset_from_file(base_path, filetype='.png', cooked_path=''):
  data_folders = [
    'normal_1',
    'normal_2',
    'normal_3',
    'normal_4',
    'normal_5',
    'normal_6',
    'swerve_1',
    'swerve_2',
    'swerve_3'
  ]

  dataframes = []
  for folder in data_folders:
    full_path_raw_folder = os.path.join(base_path, folder)

    current_dataframe = pd.read_csv(
      os.path.join(full_path_raw_folder, 'airsim_rec.txt'), sep='\t'
    )

    # current_dataframe['Use Count'] = 0
    current_dataframe['Section'] = folder
    current_dataframe['Folder'] = full_path_raw_folder

    if (cooked_path != ''):
      current_dataframe['Cooked Folder'] = re.sub(base_path, cooked_path, full_path_raw_folder)
    else:
      current_dataframe['Cooked Folder'] = ''

    dataframes.append(current_dataframe)

  meta_dataset = pd.concat(dataframes, axis=0)
  print(f'Number of data points: {meta_dataset.shape[0]}')

  meta_dataset['Cooked Path'] = meta_dataset.apply(
    lambda r: construct_cooked_file_path(r, filetype=filetype), axis=1
  )

  # Add-in filepath
  meta_dataset['Full Path'] = meta_dataset.apply(
    lambda r: construct_file_path(r), axis=1
  )

  # Check if file is present
  meta_dataset['Cooked File Presence'] = meta_dataset.apply(
    lambda r: data_functions.check_file_presence(r, idex='Cooked Path'), axis=1
  )

  meta_dataset['File Presence'] = meta_dataset.apply(
    lambda r: data_functions.check_file_presence(r), axis=1
  )

  # Split the data into Swerve and Not Swerve
  meta_dataset['Is Swerve'] = meta_dataset.apply(
    lambda r: 'swerve' in r['Folder'], axis=1
  )

  return meta_dataset

def compute_cookbook_cooked_relative_path(base_path, meta_dataset, i, detect_path=False):
  file = meta_dataset.iloc[i]['ImageName']
  folder = meta_dataset.iloc[i]['Section']

  folder_path = f'{base_path}{folder}/images'.lower()
  data_functions.check_and_create_dir(folder_path)

  # print(meta_dataset)
  # print(f'{folder_path}/{file}')

  return f'{folder_path}/{file}'

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
# (72, 128, 3) # (66, 200, 3) # (144, 256, 3)
def read_image(single_meta, options={ "shape": (59, 256, 3) }):
  # dim_shift=(59, 256, 3)
  shape = options["shape"]

  # print(f'file: {file}')
  # print(f'folder: {folder}')

  # roi = [76,135,0,255]

  filepath = single_meta["Full Path"]
  # filepath = f'{folder}/images/{file}'

  # TODO: extract image processing
  # TODO: add in translations
  image = image_functions.open_image(filepath)
  image = image_functions.remove_alpha_channel(image)

  # If using small_data_raw comment out
  # image = image_functions.crop_image_with_roi(image, roi)
  # image = image_functions.normalise_image(image)
  # image = image_functions.rescale_resolution(image, dim_shift[0], dim_shift[1])

  # optional
  # image = image_functions.contrast_stretch(image)
  # image = image_functions.hist_norm(image)
  # image.set_shape(dim_shift)

  image = image_functions.rescale_pixels(image)
  image = image_functions.conditional_resize(image, shape)
  return image

def construct_cooked_file_path(single_meta, filetype='.png'):
  file = single_meta['ImageName']
  folder = single_meta['Cooked Folder']

  file = re.sub('.png', filetype, file)
  filepath = f'{folder}/images/{file}'

  # print(filepath)
  # raise "f"

  return filepath

def construct_file_path(single_meta):
  file = single_meta['ImageName']
  folder = single_meta['Folder']

  filepath = f'{folder}/images/{file}'
  # print(filepath)

  return filepath
