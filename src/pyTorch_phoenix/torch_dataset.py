import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as jnp
# config.update('jax_enable_x64', True)

import torch
from torchvision.io import read_image

import sys
sys.path.insert(1, '../')

import helper_functions

# https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec # VaporWave
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
# TODO: Handle perturbations
class CustomImageDataset:
  def __init__(self, paths, y, config, dev, name):
    self.y = y
    self.paths = paths
    self.config = config
    self.dev = dev
    self.name = name

    self.data_tracking = { 'Use Count': np.zeros(len(y)) }

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    img_path = self.paths.iloc[idx]

    if self.config["cook_data"]:
      image = read_image(img_path).permute(0, 2, 1).float() # .permute(0, 2, 1)
    else:
      image = torch.tensor(image_functions.load_and_process_image(img_path, self.config)).permute(2, 0, 1).float()

    running_gradcam = False
    if "running_gradcam" in self.config:
      if self.config["running_gradcam"]:
        running_gradcam = True

    if running_gradcam:
      dtype = helper_functions.compute_gradcam_dtype(self.config)
    else:
      dtype = helper_functions.compute_dtype(self.config)

    y_out = self.y[idx]

    if self.config['sanity_check']:
      self.data_tracking['Use Count'][idx] = self.data_tracking['Use Count'][idx] + 1

    return image.to(device=self.dev, dtype=dtype, non_blocking=self.config["non_blocking"]), y_out.to(device=self.dev, dtype=dtype, non_blocking=self.config["non_blocking"])
