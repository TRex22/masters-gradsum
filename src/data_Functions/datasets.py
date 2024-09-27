import re

import helper_functions
environment = helper_functions.check_environment()

from cookbook_dataset_functions import cookbook_meta, construct_cookbook_meta_dataset_from_file, compute_cookbook_cooked_relative_path
from udacity_dataset_functions import udacity_meta, construct_udacity_meta_dataset_from_file, compute_udacity_cooked_relative_path
from cityscapes_dataset_functions import cityscapes_meta, construct_cityscapes_meta_dataset_from_file, compute_cityscapes_cooked_relative_path
from cityscapesvideo_dataset_functions import cityscapesvideo_meta, construct_cityscapesvideo_meta_dataset_from_file, compute_cityscapesvideo_cooked_relative_path
from carla_imitation_dataset_functions import carla_imitation_meta, construct_carla_imitation_meta_dataset_from_file, compute_carla_imitation_cooked_relative_path
from fromgames_dataset_functions import construct_fromgames_meta_dataset_from_file, compute_fromgames_cooked_relative_path

# TODO: filetypes and data_folder
def fetch_meta_from(dataset_string, base_path, train_val_test_split, outputs, options):
  dataset, width, height, dim = extract_options_from(dataset_string)

  if (dataset.lower() == 'cookbook'):
    return cookbook_meta(base_path, train_val_test_split, outputs, options)
  elif (dataset.lower() == 'udacity'):
    return udacity_meta(base_path, train_val_test_split, outputs, options) # filetype='.jpg', data_folders=['']
  elif (dataset.lower() == 'cityscapes'):
    return cityscapes_meta(base_path, train_val_test_split, outputs, options) # filetype='.png', data_folders=['']
  elif (dataset.lower() == 'cityscapesvideo'):
    return cityscapesvideo_meta(base_path, train_val_test_split, outputs, options) # filetype='.png', data_folders=['']
  elif (dataset.lower() == 'carlaimitation'):
    return carla_imitation_meta(base_path, train_val_test_split, outputs, options) # filetype='.png', data_folders=['']
  # elif (dataset.lower() == 'fromgames'):
  #   return fromgames_meta(base_path, train_val_test_split, outputs, options) # filetype='.png', data_folders=['']
  else:
    raise Exception("Invalid dataset input")

# TODO: filetypes and data_folder
def fetch_construct_from(dataset_string, base_path, cooked_path):
  # dataset, width, height, dim = extract_options_from(dataset_string)
  dataset = dataset_string.split('_')[0]

  if (dataset.lower() == 'cookbook'):
    return construct_cookbook_meta_dataset_from_file(base_path, cooked_path=cooked_path)
  elif (dataset.lower() == 'udacity'):
    return construct_udacity_meta_dataset_from_file(base_path, filetype='.jpg', data_folders=['north', 'south'], cooked_path=cooked_path)
  elif (dataset.lower() == 'cityscapes'):
    return construct_cityscapes_meta_dataset_from_file(base_path, data_folders=[''], cooked_path=cooked_path) # filetype='.png', data_folders=['']
  elif (dataset.lower() == 'cityscapesvideo'):
    return construct_cityscapesvideo_meta_dataset_from_file(base_path, data_folders=[''], cooked_path=cooked_path) # filetype='.png', data_folders=['']
  elif (dataset.lower() == 'carlaimitation'):
    return construct_carla_imitation_meta_dataset_from_file(base_path, data_folders=[''], cooked_path=cooked_path) # filetype='.png', data_folders=['']
  elif (dataset.lower() == 'fromgames'):
     return construct_fromgames_meta_dataset_from_file(base_path, data_folders=[''], cooked_path=cooked_path) # filetype='.png', data_folders=['']
  else:
    raise Exception("Invalid dataset input")

# TODO: Base filetype
def fetch_original_base_path_from(dataset_string, win_drive='L', base_path_linux='/data/data'):
  dataset = dataset_string.split('_')[0]

  if (environment == 'windows'):
    if (dataset.lower() == 'cookbook'):
      return f'{win_drive}:\\data\\cookbook\\data_raw\\'
    elif (dataset.lower() == 'udacity'):
      return f'{win_drive}:\\data\\udacity\\CH3\\data_raw\\'
    elif (dataset.lower() == 'cityscapes'):
      return f'{win_drive}:\\data\\cityscapes\\data_raw\\leftImg8bit\\'
    elif (dataset.lower() == 'cityscapesvideo'):
      return f'{win_drive}:\\data\\cityscapesvideo\\data_raw\\leftImg8bit\\'
    elif (dataset.lower() == 'cityscapes_segmentation'):
      return f'{win_drive}:\\data\\cityscapesvideo\\data_raw\\'
    elif (dataset.lower() == 'carlaimitation'):
      return f'{win_drive}:\\data\\carla\\imitation\\data_raw\\'
    elif (dataset.lower() == 'fromgames'):
      return f'{win_drive}:\\data\\fromgames\\'
    else:
      raise Exception("Invalid dataset input")
  else:
    if (dataset.lower() == 'cookbook'):
      return f'{base_path_linux}/cookbook/data_raw/'
    elif (dataset.lower() == 'udacity'):
      return f'{base_path_linux}/udacity/CH3/data_raw/'
    elif (dataset.lower() == 'cityscapes'):
      return f'{base_path_linux}/cityscapes/data_raw/leftImg8bit'
    elif (dataset.lower() == 'cityscapesvideo'):
      return f'{base_path_linux}/cityscapes/data_raw/leftImg8bit'
    elif (dataset.lower() == 'cityscapes_segmentation'):
      return f'{base_path_linux}/cityscapes/data_raw/'
    elif (dataset.lower() == 'carlaimitation'):
      return f'{base_path_linux}/carla/imitation/data_raw/'
    elif (dataset.lower() == 'fromgames'):
      return f'{base_path_linux}/fromgames/'
    else:
      raise Exception("Invalid dataset input")

  raise Exception("Invalid environment .... run away now!")

def fetch_base_save_path_from(dataset_string, win_drive='L', base_path_linux='/data/data', greyscale=False, segmentation=False):
  dataset, width, height, dim = extract_options_from(dataset_string)

  if greyscale:
    greyscale_id = "_greyscale"
  else:
    greyscale_id = ""

  if segmentation:
    segmentation_id = "_segmentation"
  else:
    segmentation_id = ""

  if (environment == 'windows'):
    if (dataset.lower() == 'cookbook'):
      return f'{win_drive}:\\data\\cookbook\\processed\\{dataset_string}{greyscale_id}{segmentation_id}\\'
    elif (dataset.lower() == 'udacity'):
      return f'{win_drive}:\\data\\udacity\\CH3\\processed\\center\\{dataset_string}{greyscale_id}{segmentation_id}\\'
    elif (dataset.lower() == 'cityscapes'):
      return f'{win_drive}:\\data\\cityscapes\\processed\\leftImg8bit\\{dataset_string}{greyscale_id}{segmentation_id}\\'
    elif (dataset.lower() == 'cityscapesvideo'):
      return f'{win_drive}:\\data\\cityscapes\\processed\\leftImg8bit\\{dataset_string}{greyscale_id}{segmentation_id}\\'
    elif (dataset.lower() == 'carlaimitation'):
      return f'{win_drive}:\\data\\carla\\imitation\\processed\\{dataset_string}{greyscale_id}{segmentation_id}\\'
    elif (dataset.lower() == 'fromgames'):
      return f'{win_drive}:\\data\\fromgames\\processed\\{dataset_string}{greyscale_id}{segmentation_id}\\'
    else:
      raise Exception("Invalid dataset input")
  else:
    if (dataset.lower() == 'cookbook'):
      return f'{base_path_linux}/cookbook/processed/{dataset_string}{greyscale_id}{segmentation_id}/'
    elif (dataset.lower() == 'udacity'):
      return f'{base_path_linux}/udacity/CH3/processed/center/{dataset_string}{greyscale_id}{segmentation_id}/'
    elif (dataset.lower() == 'cityscapes'):
      return f'{base_path_linux}/cityscapes/processed/leftImg8bit/{dataset_string}{greyscale_id}{segmentation_id}/'
    elif (dataset.lower() == 'cityscapesvideo'):
      base_save_path = f'{base_path_linux}/cityscapes/processed/leftImg8bit/{dataset_string}{greyscale_id}{segmentation_id}/'
      base_save_path = re.sub('cityscapesvideo', 'cityscapes', base_save_path)
      return base_save_path
    elif (dataset.lower() == 'carlaimitation'):
      return f'{base_path_linux}/carla/imitation/processed/{dataset_string}{greyscale_id}{segmentation_id}/'
    elif (dataset.lower() == 'fromgames'):
      return f'{base_path_linux}/fromgames/processed/{dataset_string}{greyscale_id}{segmentation_id}/'
    else:
      raise Exception("Invalid dataset input")

  raise Exception("Invalid environment .... run away now!")

def fetch_cooked_relative_path_from(dataset_string, base_path, meta_dataset, i, detect_path=False):
  dataset, width, height, dim = extract_options_from(dataset_string)

  if (dataset.lower() == 'cookbook'):
    return compute_cookbook_cooked_relative_path(base_path, meta_dataset, i, detect_path)
  elif (dataset.lower() == 'udacity'):
    return compute_udacity_cooked_relative_path(base_path, meta_dataset, i, detect_path)
  elif (dataset.lower() == 'cityscapes'):
    return compute_cityscapes_cooked_relative_path(base_path, meta_dataset, i, detect_path)
  elif (dataset.lower() == 'cityscapesvideo'):
    return compute_cityscapesvideo_cooked_relative_path(base_path, meta_dataset, i, detect_path)
  elif (dataset.lower() == 'carlaimitation'):
    return compute_carla_imitation_cooked_relative_path(base_path, meta_dataset, i, detect_path)
  elif (dataset.lower() == 'fromgames'):
    return compute_fromgames_cooked_relative_path(base_path, meta_dataset, i, detect_path)
  else:
    raise Exception("Invalid dataset input")

def extract_options_from(dataset_string):
  dataset, width_val, height_val, dim = dataset_string.split('_')

  # return { dataset: dataset, width: int(width_val), height: int(height_val), dim: int(dim) }
  return [dataset, int(width_val), int(height_val), int(dim)]

def dataset_stats(dataset_string, train_val_test_split=[0.7, 0.2, 0.1]):
  base_path = fetch_original_base_path_from(dataset_string)
  cooked_path = ''
  meta = fetch_construct_from(dataset_string, base_path, cooked_path)

  meta_size = meta.shape[0]

  options = {
    zero_drop_percentage: config["zero_drop_percentage"]
  }

  train_meta, val_meta, test_meta = fetch_meta_from(dataset_string, base_path, train_val_test_split, 1, options = options)

  train_size = train_meta.shape[0]
  val_size = val_meta.shape[0]
  test_size = test_meta.shape[0]

  drop_size = train_size + val_size + test_size

  return [meta_size, train_size, val_size, test_size, drop_size]
