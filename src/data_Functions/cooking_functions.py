import os
import time
import numpy as np
from numpy import genfromtxt

import re
import cv2

import threading

from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import image_functions
import data_functions

from thread_writer import ThreadWriter
from thread_cooker import ThreadCooker
from thread_segmentation_cooker import ThreadSegmentationCooker
from thread_edge_cooker import ThreadEdgeCooker

from datasets import fetch_meta_from, fetch_construct_from, fetch_original_base_path_from, fetch_base_save_path_from, fetch_cooked_relative_path_from, extract_options_from

# For data processing
number_of_threads = 100 # 100 # 24

segmentation_image_map = genfromtxt('./data_Functions/fromgames_to_cityscapes_colour_map.csv', delimiter=',')
segmentation_image_map = (np.fliplr(segmentation_image_map* 255).round())

def sequential_cook_segmentation_data(i, config, meta_dataset, dataset_string, base_path_linux='/data/data'):
  path = meta_dataset.iloc[i]["Segmentation Path"]
  original_path = meta_dataset.iloc[i]["Full Segmentation Path"]

  cooked_base_path = fetch_base_save_path_from(dataset_string, base_path_linux=base_path_linux)
  data_functions.check_and_create_dir(cooked_base_path)

  segmentation_image_map = None

  # TODO: Move to dataset functions
  if config['grad_cam_result_dataset'] == 'cityscapes':
    data_functions.check_and_create_dir(f"{cooked_base_path}/gtFine")
    data_functions.check_and_create_dir(f"{cooked_base_path}/gtCoarse")
  elif config['grad_cam_result_dataset'] == 'fromgames':
    data_functions.check_and_create_dir(f"{cooked_base_path}/labels")

  seg_path = meta_dataset.iloc[i]['Cooked Segmentation Path']
  data_functions.check_and_create_dir(seg_path, remove_filename=True)

  file_check = Path(seg_path).is_file() # and Path(original_path).is_file() # Removed second part for Kraken

  if not file_check:
    original_segmentation = image_functions.open_image(original_path)
    cooked_segmentation = image_functions.crop_image_with_roi(original_segmentation, roi=config["roi"][config["grad_cam_result_dataset"]])

    # http://web.archive.org/web/20230115025109/https://chadrick-kwag.net/cv2-resize-interpolation-methods/
    interpolation = cv2.INTER_NEAREST #cv2.INTER_AREA # cv2.INTER_NEAREST
    cooked_segmentation = image_functions.conditional_resize(cooked_segmentation, (config["input_size"][2], config["input_size"][1], config["input_size"][0]), interpolation=interpolation)

    # Part of fromgames is applying a map to the labels to make them Cityscapes
    # https://download.visinf.tu-darmstadt.de/data/from_games/
    if config['grad_cam_result_dataset'] == 'fromgames':
      cooked_segmentation = convert_colour_to_segmentation(cooked_segmentation)

    # cooked_segmentation = image_functions.load_and_process_image(original_path, self.config, is_segmentation=True, interpolation=cv2.INTER_NEAREST)
    image_functions.save_image(cooked_segmentation, seg_path)

# For fromgames
def find_index_of(value):
  for i in range(segmentation_image_map.shape[0]):
    if np.array_equal(segmentation_image_map[i], value):
      return i

  return 0

# For fromgames
def convert_colour_to_segmentation(segmentation_map):
  width, height = segmentation_map.shape[0:2]
  pixel_values = segmentation_map.reshape(width * height, 3)

  unique_pixels = np.unique(pixel_values, axis=0, return_counts=False)

  for pixel_value in unique_pixels:
    index_pixel = find_index_of(pixel_value)
    new_pixel_value = np.array([index_pixel, index_pixel, index_pixel]).astype(np.int8)

    matching_rows = (segmentation_map == pixel_value).all(axis=2)
    segmentation_map[matching_rows] = new_pixel_value

  return segmentation_map

def cook_segmentation_data(config, meta_dataset, dataset_string, base_path_linux='/data/data'):
  print('Cooking segmentation data ...')
  total_count = meta_dataset["Segmentation Path"].shape[0]

  # Sequential
  # pbar = tqdm(total=total_count)
  # for i in range(total_count):
  #   sequential_cook_segmentation_data(i, config, meta_dataset, dataset_string, base_path_linux=base_path_linux)
  #   pbar.update(1)

  # pbar.close()

  # Threaded:
  thread_batch_size = int(meta_dataset.shape[0] / number_of_threads) + 1

  start_time = time.time()
  last_thread = None

  for i in range(0, meta_dataset.shape[0], thread_batch_size):
    batch = range(i, i+thread_batch_size)

    last_thread = ThreadSegmentationCooker(batch, config, meta_dataset, dataset_string, base_path_linux)
    last_thread.start()

  if last_thread:
    while(not last_thread.finished):
      last_thread.finished

  end_time = time.time()
  total_time = end_time - start_time
  print(f"Took: {total_time} secs.")

  print("Completed Segmentation Maps!\n")

def sequential_cook_edge_data(i, config, meta_dataset, dataset_string, base_path_linux='/data/data'):
  path = meta_dataset.iloc[i]['Cooked Edge Path']
  original_path = meta_dataset.iloc[i]['Cooked Path']

  data_functions.check_and_create_dir(path, remove_filename=True)
  file_check = Path(path).is_file()

  if not file_check:
    frame = image_functions.open_image(original_path)
    cooked_edge = image_functions.compute_canny_edges(frame, config["canny_threshold1"], config["canny_threshold2"])
    image_functions.save_image(cooked_edge, path)

def cook_edge_data(config, meta_dataset, dataset_string, base_path_linux='/data/data'):
  print('Cooking edge data ...')
  total_count = meta_dataset['Cooked Edge Path'].shape[0]

  # Sequential
  # pbar = tqdm(total=total_count)
  # for i in range(total_count):
  #   sequential_cook_edge_data(i, config, meta_dataset, dataset_string, base_path_linux='/data/data')
  #   pbar.update(1)

  # Threaded:
  thread_batch_size = int(meta_dataset.shape[0] / number_of_threads) + 1

  start_time = time.time()
  last_thread = None

  for i in range(0, meta_dataset.shape[0], thread_batch_size):
    batch = range(i, i+thread_batch_size)
    last_thread = ThreadEdgeCooker(batch, config, meta_dataset, dataset_string, base_path_linux)
    last_thread.start()

  if last_thread:
    while(not last_thread.finished):
      last_thread.finished

  end_time = time.time()
  total_time = end_time - start_time
  print(f"Took: {total_time} secs.")

  print("Completed Edge Maps!\n")

def cook_dataset(dataset_string, detect_path=False, roi=None, base_path_linux='/data/data', greyscale=False, config={}):
  print_output = True
  if "presentation_mode" in config:
    print_output = False

  meta_dataset = construct_dataset(dataset_string, base_path_linux=base_path_linux, greyscale=greyscale, print_output=print_output)
  cook_images(dataset_string, meta_dataset, detect_path, roi=roi, base_path_linux=base_path_linux, greyscale=greyscale)

def construct_dataset(dataset_string, base_path_linux='/data/data', greyscale=False, print_output=True):
  print(f'Begin to Cook: {dataset_string}')

  cooked_path = fetch_base_save_path_from(dataset_string, base_path_linux=base_path_linux, greyscale=greyscale)

  base_path = fetch_original_base_path_from(dataset_string, base_path_linux=base_path_linux)
  raw_meta = fetch_construct_from(dataset_string, base_path, cooked_path)

  meta = data_functions.drop_missing_data(raw_meta)
  meta = data_functions.drop_existing_cooked_data(meta) #.reset_index()

  print(f'Number of data points after drop: { meta.shape[0] }')

  return meta

def fetch_image_batch(batch, meta_dataset, dim_shift, roi, greyscale):
  images = []

  for i in batch:
    if i < meta_dataset.shape[0]:
      image = read_and_cook_image(meta_dataset.iloc[i], dim_shift=dim_shift, roi=roi, greyscale=greyscale)
      images.append(image)

  return images

def process_image_batch(image_batch, meta_dataset, dataset_string, base_save_path, detect_path):
  for i in range(len(image_batch)):
    image = image_batch[i]
    save_path = fetch_cooked_relative_path_from(dataset_string, base_save_path, meta_dataset, i, detect_path)
    image_functions.save_image(image, save_path)

def cook_image_task(batch, meta_dataset, dim_shift, roi, greyscale, dataset_string, base_save_path, detect_path):
  # Sequential
  # for i in batch:
  #   if i < meta_dataset.shape[0]:
  #     image = read_and_cook_image(meta_dataset.iloc[i], dim_shift=dim_shift, roi=roi, greyscale=greyscale)

  #     if image.shape[0] != 0:
  #       save_path = fetch_cooked_relative_path_from(dataset_string, base_save_path, meta_dataset, i, detect_path)
  #       image_functions.save_image(image, save_path)

  # Run batch in multiple threads
  thread_cooker = ThreadCooker(batch, meta_dataset, dim_shift, roi, greyscale, dataset_string, base_save_path, detect_path)
  thread_cooker.start()

  return thread_cooker

def cook_images(dataset_string, meta_dataset, detect_path=False, roi=None, base_path_linux='/data/data', greyscale=False):
  print('Cooking image data ...')

  base_save_path = fetch_base_save_path_from(dataset_string, base_path_linux=base_path_linux, greyscale=greyscale)
  print(f'Base save path: {base_save_path}')

  data_functions.check_and_create_dir(base_save_path)

  dataset, width, height, dim = extract_options_from(dataset_string)
  dim_shift = (int(height), int(width), int(dim))

  # Hack for Carla
  if dataset == 'carlaimitation':
    data_functions.check_and_create_dir(f'{base_save_path}/SeqTrain')
    data_functions.check_and_create_dir(f'{base_save_path}/SeqVal')

  # number_of_threads = 1000 # 100
  thread_batch_size = int(meta_dataset.shape[0] / number_of_threads) + 1
  # threads = []

  # Sequential
  # for i in tqdm(range(0, meta_dataset.shape[0], thread_batch_size)):
  #   batch = range(i, i+thread_batch_size)
    # cook_image_task(batch, meta_dataset, dim_shift, roi, greyscale, dataset_string, base_save_path, detect_path)
    # OR
    # image_batch = fetch_image_batch(batch, meta_dataset, dim_shift, roi, greyscale)
    # process_image_batch(image_batch, meta_dataset, dataset_string, base_save_path, detect_path)

  start_time = time.time()
  last_thread = None

  for i in range(0, meta_dataset.shape[0], thread_batch_size):
    batch = range(i, i+thread_batch_size)

    last_thread = cook_image_task(batch, meta_dataset, dim_shift, roi, greyscale, dataset_string, base_save_path, detect_path)
    # last_thread.start()

  if last_thread:
    while(not last_thread.finished):
      last_thread.finished

  end_time = time.time()
  total_time = end_time - start_time
  print(f"Took: {total_time} secs.")

  # Threading Notes:
  # https://stackoverflow.com/questions/38856172/simple-multithread-for-loop-in-python
  # https://docs.python.org/3/library/threading.html
  # stop = i + thread_batch_size if i + thread_batch_size <= number_of_threads else number_of_threads

  # thread = threading.Thread(target = cook_image_task, args = (batch, meta_dataset, dim_shift, roi, greyscale, dataset_string, base_save_path, detect_path))
  # thread.start()
  # threads.append(thread)
  # Need to .join() too

  # print("\n\nDone.\n\n")
  print("Completed Source Image Cooking!\n")

# TODO: Get 'Full Path' working with all the meta classes
# TODO: Extract out a single read image method
def read_and_cook_image(single_meta, dim_shift=(50, 100, 3), roi=None, greyscale=False):
  # filepath = construct_file_path(single_meta, frame='center')
  filepath = re.sub('//', '/', single_meta['Full Path'])
  # print(filepath)

  image = image_functions.open_image(filepath)
  image = process_image(image, dim_shift=dim_shift, roi=roi, greyscale=greyscale)

  return image

def process_image(image, dim_shift=(50, 100, 3), roi=None, greyscale=False):
  # image = image_functions.crop_image_with_roi(image, roi)

  # image_functions.draw_image(image)
  # roi = (488, 908, 160, 1792)
  # image_functions.draw_image(image_functions.crop_image_with_roi(image, roi=roi))

  try:
    image.shape
  except:
    return np.array([])

  image = image_functions.remove_alpha_channel(image)

  if roi:
    image = image_functions.crop_image_with_roi(image, roi=roi)

  # If using small_data_raw comment out
  image = image_functions.normalise_image(image)
  image = image_functions.rescale_resolution(image, dim_shift[0], dim_shift[1])

  if greyscale:
    image = image_functions.greyscale(image)

  # image = image_functions.rescale_pixels(image)

  # optional
  # image = image_functions.contrast_stretch(image)
  # image = image_functions.hist_norm(image)
  # image.set_shape(dim_shift)

  return image
