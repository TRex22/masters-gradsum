# from torchvision import models
# model = models.resnet152(pretrained=True).eval()

################################################################################
# deeplabv3_resnet101
#
# 1. use the Object detection mappings from torch
# 2.
################################################################################

# https://rwightman.github.io/pytorch-image-models/models/resnet/
import time
import cv2

import numpy as np
# import jax
# from jax.config import config
# import jax.numpy as jnp
# config.update('jax_enable_x64', True)

from PIL import Image
from urllib.request import urlopen
from tqdm import tqdm

import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torchvision.transforms as transforms
from torchvision import models

from torchinfo import summary

import sys
sys.path.insert(1, './data_Functions/')

# Import training modules
import helper_functions
from helper_functions import log
environment = helper_functions.check_environment()

import datasets
import data_helpers
import data_functions
import image_functions

# from numba import jit

# Mapping trained on COCO Dataset https://www.tensorflow.org/datasets/catalog/coco
# This one has limited labels
# label_colors = np.array([
#   (0, 0, 0),      # 0 background
#   (128, 0, 0),    # 1 aeroplane
#   (0, 128, 0),    # 2 bicycle
#   (128, 128, 0),  # 3 bird
#   (0, 0, 128),    # 4 boat
#   (128, 0, 128),  # 5 bottle
#   (0, 128, 128),  # 6 bus
#   (128, 128, 128),# 7 car
#   (64, 0, 0),     # 8 cat
#   (192, 0, 0),    # 9 chair
#   (64, 128, 0),   # 10 cow
#   (192, 128, 0),  # 11 diningtable
#   (64, 0, 128),   # 12 dog
#   (192, 0, 128),  # 13 horse
#   (64, 128, 128), # 14 motorbike
#   (192, 128, 128),# 15 person
#   (0, 64, 0),     # 16 pottedplant
#   (128, 64, 0),   # 17 sheep
#   (0, 192, 0),    # 18 sofa
#   (128, 192, 0),  # 19 train
#   (0, 64, 128)    # 20 tvmonitor
# ])

# Trained on Cityscapes and Berkeley DeepDrive DataSet Labels
# https://github.com/fregu856/deeplabv3#pretrained-model
# https://github.com/fregu856/deeplabv3#evaluation
# https://github.com/fregu856/deeplabv3/pull/17/files
# label_colors = np.array([
#   [128, 64,128],#road 0
#   [244, 35,232],#sidewalk 1
#   [ 70, 70, 70],#building 2
#   [190,153,153],#wall  3
#   [153,153,153],#fence 4
#   [250,170, 30],#pole  5
#   [220,220,  0],#traffic light 6
#   [0,0, 255],#traffic sign 7
#   [152,251,152],#vegetation 8
#   [ 70,130,180],#terrain 9
#   [220, 20, 60],#sky 10
#   [255,  0,  0],#person 11
#   [  0,  0,142],#rider 12
#   [  0,  0, 70],#car 13
#   [  0, 60,100],#truck 14
#   [  0, 80,100],#bus 15
#   [  0,  0,230],#train 16
#   [119, 11, 32],#motorcycle 17
#   [81,  0, 81],#bicycle 18
#   [0,0,0]#background 19
# ])

# COCO_INSTANCE_CATEGORY_NAMES = [
#   '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#   'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#   'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#   'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#   'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#   'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#   'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

# Trained on COCO for Object detection
# https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
label_colors = np.array([
  (0, 0, 0),       # __background__
  (128, 0, 0),     # person
  (0, 128, 0),     # bicycle
  (128, 128, 0),   # car
  (0, 0, 128),     # motorcycle
  (128, 0, 128),   # airplane
  (0, 128, 128),   # bus
  (128, 128, 128), # train
  (64, 0, 0),      # truck
  (192, 0, 0),     # boat
  (64, 128, 0),    # traffic light
  (192, 128, 0),   # fire hydrant
  (64, 0, 128),    # N/A
  (192, 0, 128),   # stop sign
  (64, 128, 128),  # parking meter
  (192, 128, 128), # bench
  (0, 64, 0),      # bird
  (128, 64, 0),    # cat
  (0, 192, 0),     # dog
  (128, 192, 0),   # horse
  (0, 64, 128),    # sheep
])

def fetch_fasterrcnn_categories():
  return np.array([
    '__background__',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'N/A',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'co',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'N/A',
    'backpack',
    'umbrella',
    'N/A',
    'N/',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'N/A',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bow',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'N/A',
    'dining table',
    'N/A',
    'N/A',
    'toilet',
    'N/A',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'N/A',
    'boo',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
  ])

def transform_img(path, dev):
  img = Image.open(path).convert('RGB')
  # config = resolve_data_config({}, model=model)

  # transform = create_transform(**config)
  # print(np.array(img).shape) # (1213, 1546, 3)
  # tensor = transform(img).unsqueeze(0) # transform and add batch dimension

  preprocess = transforms.Compose([
    # transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(img)
  img.close()

  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
  input_batch.to(dev)

  return input_batch

def segment_image(tensor, dev, model, segementation_size):
  tensor = torch.tensor(tensor).float().to(dev)

  with torch.no_grad():
    out = model(tensor)

  highest_probabilities = out[0].argmax(0).cpu().numpy()
  # print(highest_probabilities.shape)

  return highest_probabilities

def fetch_categories():
  # Get imagenet class mappings
  url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
  f = urlopen(url).readlines()
  categories = [s.strip() for s in f]

  return categories

def print_top5(probabilities, categories):
  # Print top categories per image
  top5_prob, top5_catid = torch.topk(probabilities, 5)
  for i in range(top5_prob.size(0)):
    print(f'{top5_catid[i]}: {categories[top5_catid[i]], top5_prob[i].item()}')

  # prints class names and probabilities like:
  # [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]

def print_top_categories(categories):
  for i in range(len(categories)):
    print(categories[i])

def save_csv(file_path, row, overwrite=False):
  if overwrite:
    with open(file_path, 'w') as f:
      f.write(f'{row}\n')
  else:
    with open(file_path, 'a') as f:
      f.write(f'{row}\n')

  return True

# https://debuggercafe.com/semantic-segmentation-using-pytorch-fcn-resnet/
def draw_segmentation_map(outputs):
  label_map = [
    (0, 0, 0),  # background
    (128, 0, 0), # aeroplane
    (0, 128, 0), # bicycle
    (128, 128, 0), # bird
    (0, 0, 128), # boat
    (128, 0, 128), # bottle
    (0, 128, 128), # bus
    (128, 128, 128), # car
    (64, 0, 0), # cat
    (192, 0, 0), # chair
    (64, 128, 0), # cow
    (192, 128, 0), # dining table
    (64, 0, 128), # dog
    (192, 0, 128), # horse
    (64, 128, 128), # motorbike
    (192, 128, 128), # person
    (0, 64, 0), # potted plant
    (128, 64, 0), # sheep
    (0, 192, 0), # sofa
    (128, 192, 0), # train
    (0, 64, 128) # tv/monitor
  ]
  labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
  red_map = np.zeros_like(labels).astype(np.uint8)
  green_map = np.zeros_like(labels).astype(np.uint8)
  blue_map = np.zeros_like(labels).astype(np.uint8)

  for label_num in range(0, len(label_map)):
    index = labels == label_num
    red_map[index] = np.array(label_map)[label_num, 0]
    green_map[index] = np.array(label_map)[label_num, 1]
    blue_map[index] = np.array(label_map)[label_num, 2]

  segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
  return segmented_image

def image_overlay(image, segmented_image):
  alpha = 0.6 # how much transparency to apply
  beta = 1 - alpha # alpha + beta should equal 1
  gamma = 0 # scalar added to each sum
  image = np.array(image)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
  cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)

  return image

################################################################################
# Models:
# https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
# https://github.com/fregu856/deeplabv3#pretrained-model
# https://onedrive.live.com/?authkey=%21AF9rKCBVlJ3Qzo8&id=93774C670BD4F835%21933&cid=93774C670BD4F835
# https://github.com/microsoft/multiview-human-pose-estimation-pytorch/blob/master/INSTALL.md

# COCO Labels
# model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Cityscapes Labels
# model = models.resnet50(pretrained=config["pretrained"])
# model = models.segmentation.deeplabv3_resnet101(pretrained=config["pretrained"])
# model.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet152-b121ed2d.pth"))
# model.load_state_dict(torch.load(config["resnet_model_weight_path"]))
# model = nn.Sequential(*list(model.children())[:-3]) # taken from: https://github.com/fregu856/deeplabv3/blob/master/model/resnet.py

# model = models.resnet152(pretrained=True)
# model = timm.create_model(config['segmentation_model'], pretrained=True)

def deeplabv3(config, dev):
  model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)
  dtype = helper_functions.compute_model_dtype(config) # TODO: Look at bfloat16 for using the models
  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  model_stats = summary(model, device=config["summary_device_name"], verbose=0)
  log(model_stats, config)

  return model.eval()

def resnet152_model(config, dev):
  model = models.resnet152(pretrained=True)
  model = nn.Sequential(*list(model.children())[:-3]) # taken from: https://github.com/fregu856/deeplabv3/blob/master/model/resnet.py

  dtype = helper_functions.compute_model_dtype(config) # TODO: Look at bfloat16 for using the models
  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  model_stats = summary(model, device=config["summary_device_name"], verbose=0)
  log(model_stats, config)

  return model.eval()

def fasterrcnn(config, dev):
  model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  dtype = helper_functions.compute_model_dtype(config) # TODO: Look at bfloat16 for using the models
  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  model_stats = summary(model, device=config["summary_device_name"], verbose=0)
  log(model_stats, config)

  return model.eval()

def load_data(config):
  dataset_name = config["segmentation_dataset"]
  dataset_string = data_helpers.compute_dataset_string(config, dataset_name=dataset_name) # Special video sequence
  base_data_path = datasets.fetch_original_base_path_from(dataset_string, win_drive=config["win_drive"], base_path_linux=config['base_data_path'])

  raw_meta = datasets.fetch_construct_from(dataset_string, base_data_path, base_data_path)
  meta = data_functions.drop_missing_data(raw_meta).reset_index()
  log(f'Number of data points after drop: { meta.shape[0] }', config)

  return meta

################################################################################
# main
start_time = time.time()

config = {
  "device_name": "cuda:0",
  "summary_device_name": "cuda",
  "non_blocking": True,
  "log_to_file": False,
  "segmentation_model": 'resnet152',
  "segmentation_dataset": "udacity",
  "segmentation_save_path": "/data/data/udacity/CH3/data_raw/segmentation/",
  "win_drive": "L",
  "base_data_path": "/data/data",
  "cook_data": False,
  "input_size": [3, 256, 60],
  "print_top5": False,
  "print_top_categories": True,
  # "pretrained": False,
  # "resnet_model_weight_path": ""
  # "resnet_model_weight_path": ""
}

dev = helper_functions.fetch_device(config)
meta = load_data(config)

deeplabv3_model = deeplabv3(config, dev)
# resnet_model = resnet152_model(config, dev)
resnet_categories = fetch_categories()

# fasterrcnn_model = fasterrcnn(config, dev)
# fasterrcnn_categories = fetch_fasterrcnn_categories()

save_base_path = f'{config["segmentation_save_path"]}/{config["segmentation_model"]}'
log(f'Save to: {save_base_path}', config)
data_functions.check_and_create_dir(save_base_path)

deeplabv3_save_path = f'{save_base_path}/deeplabv3/'
data_functions.check_and_create_dir(deeplabv3_save_path)

# resnet_save_path = f'{save_base_path}/resnet152/'
# data_functions.check_and_create_dir(resnet_save_path)

# fasterrcnn_save_path = f'{save_base_path}/fasterrcnn/'
# data_functions.check_and_create_dir(fasterrcnn_save_path)

telemetry_filepath = f'{save_base_path}/telemetry.csv'
save_csv(telemetry_filepath, 'imagePath', overwrite=True)

top_categories = []

for i in tqdm(range(meta.shape[0])):
  img = transform_img(meta["Full Path"][i], dev)

  # resnet_segmentation = segment_image(img, dev, resnet_model, img.shape)
  # fasterrcnn_segmentation = segment_image(img, dev, fasterrcnn_model, img.shape)
  deeplabv3_segmentation = segment_image(img, dev, deeplabv3_model, img.shape)
  save_csv(telemetry_filepath, f'{meta["filename"]}', overwrite=False)

  # Save numpy map
  numpy_filename = f'{meta["filename"][i].split(".")[0].split("/")[-1]}' # .npy
  numpy_filepath = f'{deeplabv3_save_path}/{numpy_filename}'
  np.save(numpy_filepath, deeplabv3_segmentation)

  # Resize maps
  # w,h
  # dim = [segementation_size[2], segementation_size[3]]
  # return cv2.resize(probabilities, dim, cv2.INTER_NEAREST)

  # found_categories = np.unique(resnet_segmentation.flatten())
  # for i in range(found_categories.size):
  #   print(resnet_categories[found_categories[i]])
  # resnet_categories[found_categories[0]]

  # scaled_tensor = torch.nn.functional.upsample_nearest(highest_probabilities, size=dim) # mode='nearest'
  # Save PNG


# if config["print_top_categories"]: print_top_categories(list(set(top_categories)))

log(f'Data setup time: {time.time() - start_time} secs.', config)

################################################################################

# https://rwightman.github.io/pytorch-image-models/models/resnet/
# @article{DBLP:journals/corr/HeZRS15,
#   author    = {Kaiming He and
#                Xiangyu Zhang and
#                Shaoqing Ren and
#                Jian Sun},
#   title     = {Deep Residual Learning for Image Recognition},
#   journal   = {CoRR},
#   volume    = {abs/1512.03385},
#   year      = {2015},
#   url       = {http://arxiv.org/abs/1512.03385},
#   archivePrefix = {arXiv},
#   eprint    = {1512.03385},
#   timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
#   biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }
