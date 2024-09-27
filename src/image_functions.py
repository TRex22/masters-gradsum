import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as jnp
# config.update('jax_enable_x64', True)

import matplotlib.pyplot as plt
import os
import sys
import math
import cv2
import tqdm

import torch

# from numba import jit

# import helper_functions

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def remove_alpha_channel(image_array):
  if len(image_array.shape) == 3 and image_array.shape[2] == 4:
    if (np.all(image_array[:, :, 3] == image_array[0, 0, 3])):
      image_array = image_array[:,:,0:3]
  if len(image_array.shape) != 3 or image_array.shape[2] != 3:
    print('Error: Image is not RGB.')
    sys.exit()

  return image_array

def red_channel(image_array):
  return image_array[0, 0, :]

def render_roi_border(image, width, colour = (255,0,0), points = [], title = ''):
  sample_image_roi = image.copy()

  fillcolor=(colour)
  draw = ImageDraw.Draw(sample_image_roi)

  for i in range(0, len(points), 1):
      draw.line([points[i], points[(i+1)%len(points)]], colour, width)
  del draw

  plt.title(title)
  plt.imshow(sample_image_roi)
  plt.show()

def add_image_border(image, bordersize=10, value=[255, 255, 255]):
  row, col = image.shape[:2]
  bottom = image[row-2:row, 0:col]
  # mean = cv2.mean(bottom)[0]

  border = cv2.copyMakeBorder(
    image,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=value
  )

  return border

def add_heading(image, text, font_size=1.1, font_color=[0, 0, 0], font_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
  row, col = image.shape[:2]
  text_length = len(text)

  x = 0
  y = 0 #(col - text_length) / 2 # You need to have a col greater than text length

  return add_text(image, text, x, y, font_size=font_size, font_color=font_color, font_thickness=font_thickness, font=font)

def add_text(image, text, x, y, font_size=1.1, font_color=[0, 0, 0], font_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
  img_text = cv2.putText(
    image,
    text,
    (x,y),
    font,
    font_size,
    font_color,
    font_thickness,
    cv2.LINE_AA
  )

  return img_text

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def crop_image_with_roi(image_array, roi = []):
  return image_array[roi[0]:roi[1], roi[2]:roi[3]]

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def rescale_pixels(image, scale=1./255.):
  return image * scale

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def undo_rescale_pixels(image, scale=1./255.):
  return image * (1/scale)

# CV2 Resize
# INTER_NEAREST
# INTER_LINEAR
# INTER_AREA
# INTER_CUBIC
# INTER_LANCZOS4
#
def conditional_resize(image, shape, interpolation=cv2.INTER_AREA):
  if (image.shape == shape):
    return image
  else:
    return rescale_resolution(image, shape[0], shape[1], interpolation)

def rescale_resolution(image, height, width, interpolation=cv2.INTER_AREA):
  dim = (width, height)
  return cv2.resize(image, dim, interpolation)

# def contrast_stretch(image, dynamic_range=255):
#   plow = 0 # TODO
#   phigh = 255

#   pmin = plow * 255
#   pmax = phigh *255

#   return (image-pmin)*(dynamic_range/(pmax-pmin))

def contrast_stretch(image):
  """Performs a contrast stretch on the input image"""
  # Get the minimum and maximum pixel values from the image
  min_pixel = np.min(image)
  max_pixel = np.max(image)

  # Calculate the range of pixel values
  range_pixel = max_pixel - min_pixel

  # Stretch the contrast of the image by rescaling the pixel values
  image = np.uint8((255 * (image - min_pixel) / range_pixel))

  return image

def convert_to_lab_colour_space(image_array):
  return cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)

def convert_to_lab_components(image_array):
  lab = convert_to_lab_colour_space(image_array)
  return cv2.split(lab)

def convert_lab_components_to_RGB(l, a, b):
  lab_img = cv2.merge((l,a,b))
  return cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

def hist_norm(image):
  return cv2.equalizeHist(image)

def brightness_shift(image, percent):
  brightness = math.ceil(percent * 255)
  return (image + brightness)

def triple_histogram_normalisation(image_array):
  l, a, b = convert_to_lab_components(image_array)
  lhist = hist_norm(l)
  ahist = hist_norm(a)
  bhist = hist_norm(b)

  return convert_lab_components_to_RGB(lhist, ahist, bhist)

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def normalise_image(image):
  image_array = np.asarray(image)

  l, a, b = convert_to_lab_components(image_array)
  lhist = hist_norm(l)
  return convert_lab_components_to_RGB(lhist, a, b)

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def apply_filter(image, kernel, ddepth = -1):
  return cv2.filter2D(image, ddepth, kernel)

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def horizontal_flip(image):
  return cv2.flip( image, 0 )

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def vertical_flip(image):
  return cv2.flip( image, 1 )

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def flip_on_both_axes(image):
  return cv2.flip( image, -1 )

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def rotate_90_clockwise(image):
  return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def rotate_90_anti_clockwise(image):
  return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def rotate_180(image):
  return cv2.rotate(image, cv2.ROTATE_180)

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def greyscale(image):
  return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def combine_two_channels(channel_1, channel_2):
  return np.concatenate((channel_1, channel_2), axis=0) # vertically

def dual_channel_l_and_greyscale(image):
  l, a, b = convert_to_lab_components(image_array)
  grey = greyscale(image)

  return combine_two_channels(l, grey)

def dual_channel_greyscale_and_red(image):
  red = red_channel(image_array)
  grey = greyscale(image)

  return combine_two_channels(red, grey)

def dual_channel_l_and_red(image):
  l, a, b = convert_to_lab_components(image_array)
  red = red_channel(image_array)

  return combine_two_channels(l, red)

def draw_image(image, title='Sample Image'):
  plt.title(title)
  plt.imshow(image)
  plt.show()

def compute_canny_edges(frame, threshold1, threshold2):
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # # if config["summary_device_name"] == "cuda":
  # #   gpu_frame = cv2.cuda_GpuMat()
  # #   gpu_frame.upload(frame)
  # #   detector = cv2.cuda.createCannyEdgeDetector(low_thresh=config["canny_threshold1"], high_thresg=config["canny_threshold2"])
  # #   distImg = detector.detect(gpu_frame)
  # #   canny = distImg.download()
  # # else:
  # #   canny = cv2.Canny(frame, threshold1=config["canny_threshold1"], threshold2=config["canny_threshold2"])

  canny = cv2.Canny(frame, threshold1=threshold1, threshold2=threshold2)
  # canny = canny / 255 # Make it a mask

  # -------
  # PyTorch Implementation
  # filter = CannyFilter(use_cuda=True) # TODO: Make configurable
  # _blurred, _grad_x, _grad_y, _grad_magnitude, _grad_orientation, thin_edges = filter.forward(input_tensor[0], low_threshold=config["canny_threshold1"], high_threshold=config["canny_threshold2"], hysteresis=False)

  # -------
  # sklearn:
  # canny = feature.canny(frame.cpu(), sigma=3)

  # canny = canny / 255 # Make it a mask
  return canny

def open_image(path, greyscale=False):
  if greyscale:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

  return cv2.imread(path)

def save_image(image, path):
  return cv2.imwrite(path, image)

# https://stackoverflow.com/questions/52498777/apply-matplotlib-or-custom-colormap-to-opencv-image#52626636
def apply_map_to_image(image, map):
  return cv2.applyColorMap(image, map)

# https://stackoverflow.com/questions/52498777/apply-matplotlib-or-custom-colormap-to-opencv-image#52626636
def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):
  assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
  if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

  # Initialize the matplotlib color map
  sm = plt.cm.ScalarMappable(cmap=cmap)

  # Obtain linear color range
  color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
  color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
  color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

  # Apply colormap for each channel individually
  channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
  return np.dstack(channels)

def save_video(image_paths, path, width, height, fps=20):
  # https://stackoverflow.com/questions/43048725/python-creating-video-from-images-using-opencv#43048783
  # choose codec according to format needed
  print(f'Generating: {path}...')
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter(path, fourcc, fps, (width, height))

  pbar = tqdm.tqdm(total=len(image_paths))

  for image_path in image_paths:
     img = cv2.imread(image_path)
     video.write(img)

     pbar.update(1)

  cv2.destroyAllWindows()
  video.release()

  print('Done!')

# pyTorch Stuff
# np.transpose(np.array(image), (2, 0, 1))
def load_and_process_image(filepath, config, is_segmentation=False, interpolation=cv2.INTER_AREA):
  image = open_image(filepath)

  if not is_segmentation:
    image = remove_alpha_channel(image)

  # TODO: input_size pass-through
  # TODO: fix input size
  # contrast_stretch(image, dynamic_range=255)
  # brightness_shift(image, percent)
  # normalise_image(image)

  if not is_segmentation and config["roi"][config["dataset_name"]]:
    image = crop_image_with_roi(image, roi=config["roi"][config["dataset_name"]])
  elif config["roi"][config["grad_cam_result_dataset"]]:
    # image = rotate_180(image)
    image = crop_image_with_roi(image, roi=config["roi"][config["grad_cam_result_dataset"]])
    # image = vertical_flip(image) # Dont need

  # TODO: Make configurable
  # TODO: Brightness Brighten_Range: 0.4
  if not is_segmentation: # c, w, h
    image = conditional_resize(image, (config["input_size"][1], config["input_size"][2], config["input_size"][0]), interpolation=interpolation)
    # image = rescale_pixels(image) # TODO Test
  elif is_segmentation: # h, w, c
    image = conditional_resize(image, (config["input_size"][2], config["input_size"][1], config["input_size"][0]), interpolation=interpolation)

  if config["convert_to_greyscale"] and not is_segmentation:
    image = greyscale(image)

  return image

# TODO: Add options
# (72, 128, 3) # (66, 200, 3) # (144, 256, 3) # (59, 256, 3)
# TODO % rescaling
# @jit(nopython=False, forceobj=True)
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

# @jit(nopython=False, forceobj=True)
def read_images(meta_dataset):
  # Read all the images in order
  # Get image dim / force image resolutions

  single_image = image_functions.read_image(meta_dataset.iloc[0])
  im_shape = single_image.shape
  dim = np.array([len(meta_dataset), im_shape[0], im_shape[1], im_shape[2]])

  dataset = np.empty(dim)
  for i in range(meta_dataset.shape[0]):
    dataset[i] = image_functions.read_image(meta_dataset.iloc[i])

  return dataset

# def read_image_batch(x):
#   # x.shape[0] # is batch_size
#   dtype = helper_functions.compute_dtype(config)
#   images = torch.zeros([config["batch_size"], config["input_size"][0], config["input_size"][1], config["input_size"][2]], dtype=dtype) # TODO: float16

#   for i in range(config["batch_size"]):
#     idex = int(x[i]) # TODO: Double check this - sanity check by marking each as read
#     images[i] = torch.tensor(read_image_custom(meta['Full Path'].iloc[idex])).float() # TODO: pass through meta

#     if config['sanity_check']:
#       # meta['Use Count'].iloc[idex] = meta['Use Count'].iloc[idex] + 1
#       meta.iloc[idex, meta.columns.get_loc('Use Count')] = meta.iloc[idex, meta.columns.get_loc('Use Count')] + 1

#       data_tracking['X'].iloc[i] = idex
#       data_tracking['Use Count'].iloc[i] = data_tracking['Use Count'].iloc[i] + 1

#   return images
