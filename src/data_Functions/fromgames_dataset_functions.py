# https://download.visinf.tu-darmstadt.de/data/fromgames/
# @InProceedings{Richter_2016_ECCV,
#   author = {Stephan R. Richter and Vibhav Vineet and Stefan Roth and Vladlen Koltun},
#   title = {Playing for Data: {G}round Truth from Computer Games},
#   booktitle = {European Conference on Computer Vision (ECCV)},
#   year = {2016},
#   editor = {Bastian Leibe and Jiri Matas and Nicu Sebe and Max Welling},
#   series = {LNCS},
#   volume = {9906},
#   publisher = {Springer International Publishing},
#   pages = {102--118}
# }

import os
import math
import re

import helper_functions
environment = helper_functions.check_environment()

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
def fromgames_generator(meta, outputs, batch_size, options={ "shape": (48, 64, 3) }):
  # processed_data = process_fromgames_data(raw_dataset, options)
  # yield [read_images(meta), data_functions.process_outputs(meta, outputs)]
  single_image = read_image(meta.iloc[0], options)
  im_shape = single_image.shape
  dim = np.array([len(meta), im_shape[0], im_shape[1], im_shape[2]])

  for i in range(meta.shape[0]):
    x = read_image(meta.iloc[i])
    y = np.array([data_functions.process_outputs(meta.iloc[i], outputs)])

    yield x,y

def fromgames(base_path, train_val_test_split, outputs):
  train_meta, val_meta, test_meta = fromgames_meta(
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
def construct_fromgames_meta_dataset_from_file(base_path, filetype='.png', data_folders=[''], cooked_path=''):
  images_path = f'{base_path}/images/'
  labels_path = f'{base_path}/labels/'

  images_arr = np.array(os.listdir(images_path))
  images_arr.sort

  labels_arr = np.array(os.listdir(labels_path))
  labels_arr.sort

  data = { 'imagePath': pd.Series(images_arr), 'Segmentation Path': pd.Series(labels_arr) }
  meta_dataset = pd.DataFrame(data)
  print(f'Number of data points: {meta_dataset.shape[0]}')

  meta_dataset['Cooked Path'] = meta_dataset.apply(
    lambda r: construct_cooked_file_path(r, cooked_path, filetype=filetype), axis=1
  )

  # Add-in filepath
  meta_dataset['Full Path'] = meta_dataset.apply(
    lambda r: construct_file_path(r, base_path, filetype=filetype), axis=1
  )

  meta_dataset['Full Segmentation Path'] = meta_dataset.apply(
    lambda r: construct_segmentation_file_path(r, base_path, filetype=filetype), axis=1
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
    lambda r: construct_cooked_segmentation_file_path(r, cooked_path), axis=1
  )

  meta_dataset['Cooked Edge Path'] = meta_dataset.apply(
    lambda r: construct_cooked_edge_file_path(r, cooked_path), axis=1
  )

  return meta_dataset

def compute_fromgames_cooked_relative_path(base_path, meta_dataset, i, detect_path=False):
  image_path = meta_dataset.iloc[i]['imagePath']

  if (environment == 'windows'):
    folder_path = f'{base_path}\\images\\' #.lower()
    data_functions.check_and_create_dir(folder_path)
  else:
    folder_path = f'{base_path}/images/' #.lower()
    data_functions.check_and_create_dir(folder_path)

  return f'{folder_path}{image_path}' #.lower()

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
def read_image(single_meta, options={ "shape": (48, 64, 3) }):
  # dim_shift=(48, 64, 3)
  # roi = [76,135,0,255]
  shape = options["shape"]

  # filepath = construct_file_path(single_meta, frame='center')
  filepath = single_meta[-2] # ['Full Path']

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

def construct_cooked_segmentation_file_path(single_meta, cooked_path, filetype='.png'):
  file = single_meta["Segmentation Path"]

  filepath = f'{cooked_path}/labels/{file}'
  filepath = re.sub('.png', filetype, filepath)

  # print(filepath)
  return filepath

def construct_cooked_edge_file_path(single_meta, cooked_path, filetype='.png'):
  file = single_meta["Segmentation Path"]

  filepath = f'{cooked_path}/edge/{file}'
  filepath = re.sub('.png', filetype, filepath)

  # print(filepath)
  return filepath

def construct_cooked_file_path(single_meta, base_cooked_path, filetype='.png'):
  imagePath = single_meta['imagePath']

  filepath = f'{base_cooked_path}/images/{imagePath}'
  filepath = re.sub('.png', filetype, filepath)

  # print(filepath)
  return filepath

def construct_file_path(single_meta, base_path, filetype='.png'):
  file = single_meta['imagePath']
  filepath = f'{base_path}/images/{file}'

  # print(filepath)
  return filepath

def construct_segmentation_file_path(single_meta, cooked_path, filetype='.png'):
  file = single_meta["Segmentation Path"]

  filepath = f'{cooked_path}/labels/{file}'
  filepath = re.sub('.png', filetype, filepath)

  # print(filepath)
  return filepath
