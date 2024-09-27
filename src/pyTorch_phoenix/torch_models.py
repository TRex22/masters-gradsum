import math
import json
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import sys
sys.path.insert(1, '../')

import torchvision
# from torchvision.models.vision_transformer import vit_h_14, vit_l_32, vit_l_16, vit_b_32, vit_b_16
from vision_transformer import vit_h_14_new, vit_l_32_new, vit_l_16_new, vit_b_32_new, vit_b_16_new

import helper_functions
import image_functions

from helper_functions import log

import torch_optimisers

# https://neptune.ai/blog/moving-from-tensorflow-to-pytorch
# input size as 784 (28×28 is the image size of the MNIST data)

# https://stackoverflow.com/questions/64780641/whats-the-equivalent-of-tf-keras-input-in-pytorch
# https://stackoverflow.com/questions/65708548/tensorflow-vs-pytorch-convolution-confusion

# https://machinelearningknowledge.ai/pytorch-conv2d-explained-with-examples/
# .view(len(mnist_test), 1, 28, 28)

# https://github.com/guotaowang/STVS/blob/master/resnext_101_32x4d_.py
# nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2048, 1000)),  # Linear,

# Good simple explanation
# https://ibelieveai.github.io/cnnlayers-pytorch/#flattening

# Models
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html?highlight=conv2d

# Autonomous Cookbook example
# https://analyticsindiamag.com/wp-content/uploads/2018/02/mad-04.jpg

# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
# https://visualstudiomagazine.com/Articles/2021/02/11/pytorch-define.aspx?Page=2

# https://discuss.pytorch.org/t/how-to-loading-and-using-a-trained-model/85678/2

# TODO VIZ: https://newbedev.com/how-do-i-visualize-a-net-in-pytorch

# https://towardsdatascience.com/deep-learning-for-self-driving-cars-7f198ef4cfa2

################################################################################
#                                  Methods                                     #
################################################################################
def external_test_model_batch_size(config):
  model_batch_size_for_11_gb = {
    "deit_tiny_model": 100,
    "ViT-H_14": 25,
    "ViT-L_32": 100,
    "ViT-L_16": 100,
    "ViT-B_32": 100,
    "ViT-B_16": 200,
    "Net SVF": 45,
    "Net HVF": 80,
    "End to End": 400,
    "End to End No Dropout": 400,
    "Autonomous Cookbook": 800,
    "TestModel1": 1000,
    "TestModel2": 1000
  }

  batch_size = model_batch_size_for_11_gb[config["model_name"]]

  return batch_size

# Model Comparator
# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/9
def compare(model1, model2, config, verbose=True):
  if verbose:
    log("Compare parameters ...", config)

  parameters_match = True
  for p1, p2 in zip(model1.parameters(), model2.parameters()):
    if p1.data.ne(p2.data).sum() > 0:
      if verbose:
        log(f"{config['model_name']} models have mismatching parameters.", config)

      parameters_match = False

  states_match = str(model1.state_dict()) == str(model2.state_dict())

  if states_match == False and verbose:
    log(f"{config['model_name']} models have mismatching state_dicts.", config)

  return [parameters_match, states_match]

def save_final_model(model, opt, scaler, config, all_batch_metrics=None):
  # https://pytorch.org/tutorials/beginner/saving_loading_models.html

  log("Save model...", config)
  # Save init model
  # Save Trained Model
  # Save
  # TODO Save logs
  # TODO AMP mixed-precision saves
  full_save_path = helper_functions.compute_model_save_path(config)

  # TODO: re remove special characters
  # TODO: Save other fns and checkpoints

  model_full_save_path = f'{full_save_path}.pth'

  checkpoint = {
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "scaler": scaler.state_dict(),
    # "lr_scheduler": lr_scheduler.state_dict(), # TODO: setup lr_scheduler
    "epoch": -1,
    "config": config,
  }

  torch.save(checkpoint, model_full_save_path)

  config_full_save_path = f'{full_save_path}_config.json'
  output_config_file = open(config_full_save_path, 'w', encoding='utf-8')
  json.dump(config, output_config_file)
  output_config_file.write("\n")
  output_config_file.close()

  if all_batch_metrics:
    metrics_full_save_path = f'{full_save_path}_metrics.json'
    output_metrics_file = open(metrics_full_save_path, 'w', encoding='utf-8')
    json.dump(all_batch_metrics, output_metrics_file)
    output_metrics_file.close()

    metrics_save_path = f'{helper_functions.compute_model_save_path(config)}_metrics.csv'
    metrics_header = f'selected_train_loss,selected_val_loss'
    helper_functions.save_csv(metrics_save_path, metrics_header)

    for epoch_metrics in all_batch_metrics:
      line = f'{epoch_metrics["selected_train_loss"]},{epoch_metrics["selected_val_loss"]}'
      helper_functions.save_csv(metrics_save_path, line)

  log("Saved.", config)

def save_model(epoch, model, optimizer, scaler, config, all_batch_metrics={}, append_path="", checkpoint=False):
  # https://pytorch.org/tutorials/beginner/saving_loading_models.html
  if checkpoint:
    log(f"Save checkpoint, epoch: {epoch} ...", config)
  else:
    log(f"Save model, epoch: {epoch} ...", config)

  full_save_path = helper_functions.compute_model_save_path(config, epoch=epoch, checkpoint=checkpoint, append_path=append_path)

  checkpoint_dict = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(), # TODO: Handle Validation
    "scaler": scaler.state_dict(),
    # "lr_scheduler": lr_scheduler.state_dict(), # TODO: setup lr_scheduler
    "epoch": epoch,
    "config": config,
    "all_batch_metrics": all_batch_metrics
  }

  torch.save(checkpoint_dict, f'{full_save_path}.pth')

  log("Saved.", config)

def open_model(config, model_eval=False, epoch=None, append_path="", direct_path="", model=None):
  # if model is None:
  model = compile_model(config)

  optimizer, _loss_func = torch_optimisers.fetch_loss_opt_func(config, model)
  scaler = torch.cuda.amp.GradScaler(enabled=config["mixed_precision"])

  dev = helper_functions.fetch_device(config)

  full_save_path = helper_functions.compute_model_save_path(config, original_path=True, epoch=epoch, append_path=append_path)
  model_path = f'{full_save_path}.pth'

  if direct_path != "":
    model_path = direct_path

  log(f"Opening model: {model_path}", config)
  checkpoint = torch.load(model_path)

  model.load_state_dict(checkpoint['model'], strict=False)
  dtype = helper_functions.compute_model_dtype(config)

  model = helper_functions.parallelise_model(model, config)
  model.to(device=dev, dtype=dtype, non_blocking=config["non_blocking"])

  if not model_eval:
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])

  if model_eval == True:
    model.eval()

  return [model, optimizer, scaler]

# TODO: Add in rnd weights
# https://www.askpython.com/python-modules/initialize-model-weights-pytorch
# https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
# Uniform distribution with small variance
def compile_model(config):
  log(f'\n{config["model_name"]}', config)

  if config["model_name"] == "Autonomous Cookbook":
    model = CookbookModel(config)
  elif config["model_name"] == "TestModel1":
    model = TestModel1(config)
  elif config["model_name"] == "TestModel2":
    model = TestModel2(config)
  elif config["model_name"] == "End to End":
    model = EndToEndModel(config)
  elif config["model_name"] == "End to End No Dropout":
    model = EndToEndModelNoDropout(config)
  elif config["model_name"] == "Net SVF":
    model = NetSVFModel(config)
  elif config["model_name"] == "Net SVF2":
    model = NetSVF2Model(config)
  elif config["model_name"] == "Net HVF":
    model = NetHVFModel(config)
  elif is_a_vit_model(config):
    model = ViT_Model(config)
  elif config["model_name"] == "deit_tiny_model":
    model = deit_tiny_model(config)
  else:
    raise Exception("Not Implemented, invalid model!")

  rng_model = randomise_weights(model, config)

  if ('compile_model' in config) and config['compile_model']:
    return torch.compile(rng_model)
  else:
    return rng_model

def is_a_vit_model(config):
  # https://pytorch.org/vision/main/models/vision_transformer.html
  if config["model_name"] == "ViT-H_14":
    return True
  elif config["model_name"] == "ViT-L_32":
    return True
  elif config["model_name"] == "ViT-L_16":
    return True
  elif config["model_name"] == "ViT-B_32":
    return True
  elif config["model_name"] == "ViT-B_16":
    return True

  return False

# https://www.askpython.com/python-modules/initialize-model-weights-pytorch
# https://pytorch.org/docs/stable/nn.init.html
# https://medium.com/ai%C2%B3-theory-practice-business/initializing-the-weights-in-nn-b5baa2ed5f2f
#
def randomise_weights(model, config):
  log(f'Randomise weights: {config["randomise_weights"]}', config)

  if config['randomise_weights'] == 'uniform':
    model.apply(uniform_init)

  elif config['randomise_weights'] == 'normal':
    model.apply(normal_init)

  elif config['randomise_weights'] == 'ones':
    model.apply(ones_init)

  elif config['randomise_weights'] == 'zeros':
    model.apply(zeros_init)

  elif config['randomise_weights'] == 'eye':
    model.apply(eye_init)

  elif config['randomise_weights'] == 'dirac':
    model.apply(dirac_init)

  elif config['randomise_weights'] == 'xavier_uniform':
    model.apply(xavier_uniform_init)

  elif config['randomise_weights'] == 'xavier_normal':
    model.apply(xavier_normal_init)

  elif config['randomise_weights'] == 'kaiming_uniform':
    model.apply(kaiming_uniform_init)

  elif config['randomise_weights'] == 'orthogonal':
    model.apply(orthogonal_init)

  elif config['randomise_weights'] == 'sparse':
    model.apply(sparse_init)

  return model

def uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.uniform_(layer.weight, -1/math.sqrt(5), 1/math.sqrt(5)) # a, b

def normal_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.normal_(layer.weight, 0, 1/math.sqrt(5)) # mean, std^2

def ones_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.ones_(layer.weight)

def zeros_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.zeros_(layer.weight)

def eye_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.eye_(layer.weight)

def dirac_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.dirac_(layer.weight)

def xavier_uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    gain = nn.init.calculate_gain('relu')

    # See: https://discuss.pytorch.org/t/why-my-initial-loss-is-bigger-than-the-expected/29329/3
    if layer.bias is not None:
      nn.init.zeros_(layer.bias)

    nn.init.xavier_uniform_(layer.weight, gain=gain)

def xavier_normal_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    # See: https://discuss.pytorch.org/t/why-my-initial-loss-is-bigger-than-the-expected/29329/3
    if layer.bias is not None:
      nn.init.zeros_(layer.bias)

    nn.init.xavier_normal_(layer.weight)

def kaiming_uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

def kaiming_uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

def orthogonal_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.orthogonal_(layer.weight)

def sparse_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    nn.init.sparse_(layer.weight, sparsity=0.1)

def compute_dynamic_layer(config):
  if config["input_size"] == (3, 128, 30):
    return 2048

  if config["input_size"] == (3, 64, 30):
    return 1024

  if config["input_size"] == (3, 512, 120):
    return 30720

  if config["input_size"] == (3, 256, 60) and not config["model_name"] == "Autonomous Cookbook":
    return 8192

  if config["input_size"] == [3, 256, 60] and not config["model_name"] == "Autonomous Cookbook":
    return 8192

  if config["input_size"] == (3, 256, 192) and not config["model_name"] == "Autonomous Cookbook":
    return 49152

  # ViTs
  if config["input_size"] == (3, 112, 112):
    return 9216

  if config["input_size"] == (3, 256, 192):
    return 24576

  return 7168 # (3, 255, 59) - original

def compute_dynamic_layer_end_to_end(config):
  if config["input_size"] == (3, 128, 30):
    return 120

  if config["input_size"] == (3, 64, 30):
    return 40

  if config["input_size"] == (3, 512, 120):
    return 2500

  if config["input_size"] == (3, 256, 60):
    return 480

  if config["input_size"] == (3, 256, 192):
    return 2160

  return 480 # (3, 255, 59) - original

def compute_dynamic_layer_test_model(config):
  if config["input_size"] == (3, 256, 60):
    return 122880 #15 #4096

  if config["input_size"] == (3, 256, 192):
    return 393216

  if config["input_size"] == (3, 512, 120):
    return 491520

  return 122880 #15 #4096 # (3, 255, 59) - original

def compute_output_layers(config, stack, output_layer_node_size=10):
  if config["output_tanh"]:
    stack.append(nn.Tanh())

  stack.append(nn.Linear(output_layer_node_size, config["number_of_outputs"]))

  if "sigmoid" in config:
    if config["sigmoid"]:
      stack.append(nn.Sigmoid())

  return stack

################################################################################
#                                  Models                                      #
################################################################################
class TestModel1(nn.Module):
  def __init__(self, config):
    super(TestModel1, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.target_layers = [self.target_layer]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      self.target_layer,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      # nn.Dropout(p=0.2),
      nn.Linear(compute_dynamic_layer_test_model(config), 10),
      # nn.Dropout(p=0.2)
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=10)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

class TestModel2(nn.Module):
  def __init__(self, config):
    super(TestModel2, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.target_layers = [self.target_layer]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 32, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      self.target_layer,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      # nn.Dropout(p=0.2),
      nn.Linear(compute_dynamic_layer_test_model(config), 10),
      # nn.Dropout(p=0.2)
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=10)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

class CookbookModel(nn.Module):
  def __init__(self, config):
    super(CookbookModel, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
    self.target_layers = [self.target_layer]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 16, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),#(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer,
      nn.ReLU(),#(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      # nn.Dropout(p=0.2),
      nn.Linear(compute_dynamic_layer(config), 64),
      # nn.Dropout(p=0.2),
      nn.Linear(64, 10),
      # nn.Dropout(p=0.2)
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=10)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

# https://stackoverflow.com/questions/50615396/validation-loss-increasing
class EndToEndModel(nn.Module):
  def __init__(self, config):
    super(EndToEndModel, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer1 = nn.Conv2d(98, 47, kernel_size=5, stride=1, padding=1)
    self.target_layer2 = nn.Conv2d(47, 22, kernel_size=5, stride=1, padding=1)
    self.target_layer3 = nn.Conv2d(22, 20, kernel_size=3, stride=1, padding=1)

    self.target_layers = [self.target_layer1, self.target_layer2, self.target_layer3]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 98, kernel_size=3, stride=1, padding=1),
      self.target_layer1,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer2,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer3,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((5, 5)),
      nn.Flatten(),
      nn.Dropout(p=0.5), # 0.2 ... Found in other variants of this model
      # nn.Dropout(p=0.2), # Added to solve increasing validation error
      nn.Linear(compute_dynamic_layer_end_to_end(config), 18),
      # nn.Dropout(p=0.2), # Added to solve increasing validation error
      nn.Linear(18, 10),
      # nn.Dropout(p=0.2) # Added to solve increasing validation error
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=10)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

class EndToEndModelNoDropout(nn.Module):
  def __init__(self, config):
    super(EndToEndModelNoDropout, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer1 = nn.Conv2d(98, 47, kernel_size=5, stride=1, padding=1)
    self.target_layer2 = nn.Conv2d(47, 22, kernel_size=5, stride=1, padding=1)
    self.target_layer3 = nn.Conv2d(22, 20, kernel_size=3, stride=1, padding=1)

    self.target_layers = [self.target_layer1, self.target_layer2, self.target_layer3]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 98, kernel_size=3, stride=1, padding=1),
      self.target_layer1,
      # nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer2,
      # nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer3,
      # nn.ReLU(inplace=True),
      nn.MaxPool2d((5, 5)),
      nn.Flatten(),
      # nn.Dropout(p=0.5), # 0.2 ... Found in other variants of this model
      # nn.Dropout(p=0.2), # Added to solve increasing validation error
      nn.Linear(compute_dynamic_layer_end_to_end(config), 18),
      # nn.Dropout(p=0.2), # Added to solve increasing validation error
      nn.Linear(18, 10),
      # nn.Dropout(p=0.2) # Added to solve increasing validation error
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=10)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

class NetSVFModel(nn.Module):
  def __init__(self, config):
    super(NetSVFModel, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer0 = nn.Conv2d(638, 318, kernel_size=3, stride=1, padding=1)
    self.target_layer1 = nn.Conv2d(318, 316, kernel_size=3, stride=1, padding=1)
    self.target_layer2 = nn.Conv2d(316, 157, kernel_size=3, stride=1, padding=1)
    self.target_layer3 = nn.Conv2d(157, 155, kernel_size=3, stride=1, padding=1)
    self.target_layer4 = nn.Conv2d(155, 77, kernel_size=3, stride=1, padding=1)
    self.target_layer5 = nn.Conv2d(77, 75, kernel_size=3, stride=1, padding=1)
    self.target_layer6 = nn.Conv2d(75, 37, kernel_size=3, stride=1, padding=1)
    self.target_layer7 = nn.Conv2d(37, 35, kernel_size=3, stride=1, padding=1)
    self.target_layer8 = nn.Conv2d(35, 17, kernel_size=3, stride=1, padding=1)
    self.target_layer9 = nn.Conv2d(17, 1024, kernel_size=3, stride=1, padding=1)

    self.target_layers = [
      self.target_layer0,
      self.target_layer1,
      self.target_layer2,
      self.target_layer3,
      self.target_layer4,
      self.target_layer5,
      self.target_layer6,
      self.target_layer7,
      self.target_layer8,
      self.target_layer9
    ]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 638, kernel_size=3, stride=1, padding=1),
      self.target_layer0,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer1,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer2,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer3,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer4,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer5,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer6,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer7,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer8,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer9,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      nn.Linear(compute_dynamic_layer(config), 1024),
      nn.Linear(1024, 512)
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=512)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

class NetSVF2Model(nn.Module):
  def __init__(self, config):
    super(NetSVF2Model, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer0 = nn.Conv2d(638, 318, kernel_size=3, stride=1, padding=1)
    self.target_layer1 = nn.Conv2d(318, 316, kernel_size=3, stride=1, padding=1)
    self.target_layer2 = nn.Conv2d(316, 157, kernel_size=3, stride=1, padding=1)
    self.target_layer3 = nn.Conv2d(157, 155, kernel_size=3, stride=1, padding=1)
    self.target_layer4 = nn.Conv2d(155, 77, kernel_size=3, stride=1, padding=1)
    self.target_layer5 = nn.Conv2d(77, 75, kernel_size=3, stride=1, padding=1)
    self.target_layer6 = nn.Conv2d(75, 37, kernel_size=3, stride=1, padding=1)
    self.target_layer7 = nn.Conv2d(37, 35, kernel_size=3, stride=1, padding=1)
    self.target_layer8 = nn.Conv2d(35, 17, kernel_size=3, stride=1, padding=1)
    self.target_layer9 = nn.Conv2d(17, 1024, kernel_size=3, stride=1, padding=1)

    self.target_layers = [
      self.target_layer0,
      self.target_layer1,
      self.target_layer2,
      self.target_layer3,
      self.target_layer4,
      self.target_layer5,
      self.target_layer6,
      self.target_layer7,
      self.target_layer8,
      self.target_layer9
    ]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 638, kernel_size=3, stride=1, padding=1),
      self.target_layer0,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer1,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer2,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer3,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer4,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer5,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer6,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer7,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer8,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer9,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      nn.Linear(compute_dynamic_layer(config), 1024),
      # nn.Linear(1024, 512)
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=1024)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

class NetHVFModel(nn.Module):
  def __init__(self, config):
    super(NetHVFModel, self).__init__()

    self.config = config

    # Grad-CAM interface
    self.target_layer0 = nn.Conv2d(349, 173, kernel_size=3, stride=1, padding=1)
    self.target_layer1 = nn.Conv2d(173, 171, kernel_size=3, stride=1, padding=1)
    self.target_layer2 = nn.Conv2d(171, 85, kernel_size=3, stride=1, padding=1)
    self.target_layer3 = nn.Conv2d(85, 83, kernel_size=3, stride=1, padding=1)
    self.target_layer4 = nn.Conv2d(83, 41, kernel_size=3, stride=1, padding=1)
    self.target_layer5 = nn.Conv2d(41, 39, kernel_size=3, stride=1, padding=1)
    self.target_layer6 = nn.Conv2d(39, 19, kernel_size=3, stride=1, padding=1)
    self.target_layer7 = nn.Conv2d(19, 17, kernel_size=3, stride=1, padding=1)
    self.target_layer8 = nn.Conv2d(17, 8, kernel_size=3, stride=1, padding=1)
    self.target_layer9 = nn.Conv2d(8, 1024, kernel_size=3, stride=1, padding=1)

    self.target_layers = [
      self.target_layer0,
      self.target_layer1,
      self.target_layer2,
      self.target_layer3,
      self.target_layer4,
      self.target_layer5,
      self.target_layer6,
      self.target_layer7,
      self.target_layer8,
      self.target_layer9
    ]

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(config["input_size"][0], 349, kernel_size=3, stride=1, padding=1),
      self.target_layer0,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer1,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer2,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer3,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer4,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer5,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer6,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer7,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      self.target_layer8,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((1, 1)),
      self.target_layer9,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      nn.Linear(compute_dynamic_layer(config), 1024),
      nn.Linear(1024, 512)
    )

    self.cnn_stack = compute_output_layers(config, self.cnn_stack, output_layer_node_size=512)

  def forward(self, x):
    logits = self.cnn_stack(x)

    if self.config["normalise_output"]:
      logits = F.normalize(logits, dim = 0)

    return logits

class ViT_Model(torch.nn.Module):
  def __init__(self, config):
    super(ViT_Model, self).__init__()
    self.config = config

    self.image_size =  config["input_size"][1] # Assume square size
    self.num_classes = config["number_of_outputs"]

    if "track_attention_weights" in config:
      self.track_attention_weights = config["track_attention_weights"]
    else:
      self.track_attention_weights = False

    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    if "average_attn_weights" in config:
      self.average_attn_weights = config["average_attn_weights"]
    else:
      self.average_attn_weights = False

    if "compute_attn_mean" in config: # mean attention per layer across heads
      self.compute_attn_mean = config["compute_attn_mean"]
    else:
      self.compute_attn_mean = True # Force mean otherwise

    # Naming is a little confusing. Mean here is across layers
    if "compute_mean_attention" in config:
      self.compute_mean_attention = config["compute_mean_attention"]
    else:
      self.compute_mean_attention = False # Off by default

    if "mean_attention_index" in config:
      self.mean_attention_index = config["mean_attention_index"]
    else:
      self.mean_attention_index = -1 # Last layer by default

    # https://pytorch.org/vision/main/models/vision_transformer.html
    if config["model_name"] == "ViT-H_14":
      # image_size=, patch_size=, pretrained=False
      self.model = vit_h_14_new(average_attn_weights=self.average_attn_weights, track_attention_weights=self.track_attention_weights, compute_attn_mean=self.compute_attn_mean, num_classes=self.num_classes, image_size=self.image_size, weights=None) #pretrained=False
    elif config["model_name"] == "ViT-L_32":
      self.model = vit_l_32_new(average_attn_weights=self.average_attn_weights, track_attention_weights=self.track_attention_weights, compute_attn_mean=self.compute_attn_mean, num_classes=self.num_classes, image_size=self.image_size, weights=None) #pretrained=False
    elif config["model_name"] == "ViT-L_16":
      self.model = vit_l_16_new(average_attn_weights=self.average_attn_weights, track_attention_weights=self.track_attention_weights, compute_attn_mean=self.compute_attn_mean, num_classes=self.num_classes, image_size=self.image_size, weights=None) #pretrained=False
    elif config["model_name"] == "ViT-B_32":
      self.model = vit_b_32_new(average_attn_weights=self.average_attn_weights, track_attention_weights=self.track_attention_weights, compute_attn_mean=self.compute_attn_mean, num_classes=self.num_classes, image_size=self.image_size, weights=None) #pretrained=False
    elif config["model_name"] == "ViT-B_16":
      self.model = vit_b_16_new(average_attn_weights=self.average_attn_weights, track_attention_weights=self.track_attention_weights, compute_attn_mean=self.compute_attn_mean, num_classes=self.num_classes, image_size=self.image_size, weights=None) #pretrained=False

    # Add custom classifier head
    # self.classifier = torch.nn.Linear(in_features=self.num_classes, out_features=1)
    # self.flatten = nn.Flatten()
    self.original_head = self.model.heads.head
    self.new_head = torch.nn.Linear(in_features=self.model.hidden_dim, out_features=self.num_classes)
    self.model.heads.head = self.new_head

    # Move inner model to device
    self.dev = helper_functions.fetch_device(config)
    self.dtype = helper_functions.compute_model_dtype(config)
    self.data_dtype = helper_functions.compute_gradcam_dtype(config)
    self.model = self.model.to(self.dev, dtype=self.dtype)

    # Possibly use the attention layer
    self.cnn_layer = self.model.conv_proj
    self.target_layers = [self.cnn_layer]

  def clear_attn_weights(self):
    for layer in self.model.encoder.layers:
      layer.clear_attn_weights()

  def __del__(self):
    self.clear_attn_weights()

    del self.target_layers
    del self.cnn_layer
    del self.original_head
    del self.new_head
    # del self.encoder
    del self.model.heads.head
    del self.model

    helper_functions.clear_gpu(torch, self.config, pause=False)

  def attention(self, img, shift_single_result=False):
    if self.track_attention_weights == False:
      raise("track_attention_weights has to be set to True!")
    # Set model to eval
    self.model = self.model.eval()
    # self.model = self.model.train()

    # Make sure input is on the correct device
    input_data = img.to(self.dev, dtype=self.data_dtype)

    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    with torch.no_grad():
      outputs = self(input_data)
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=15))

    # TODO: Pull out all layers and combine them
    # For now take the last layer

    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
    # Select last layer and last head for now
    attention = None

    if self.compute_attn_mean and not self.compute_mean_attention:
      attention = self.model.encoder.layers[-1].mean_attn_weights
    elif not self.compute_mean_attention:
      attention = self.model.encoder.layers[-1].attn_weights[:, -1]
    elif self.compute_attn_mean and self.compute_mean_attention:
      mean_attn_weights = [layer.mean_attn_weights for layer in self.model.encoder.layers]
      mean_attn_weights = torch.stack(mean_attn_weights)
      attention = torch.mean(mean_attn_weights, dim=0)
    elif self.compute_mean_attention:
      mean_attn_weights = [layer.attn_weights[:, self.mean_attention_index] for layer in self.model.encoder.layers]
      mean_attn_weights = torch.stack(mean_attn_weights)
      attention = torch.mean(mean_attn_weights, dim=0)

    # Select Last layer and first head
    # attention = self.model.encoder.layers[-1].attn_weights[:, 0]

    # Select First Layer and last head
    # attention = self.model.encoder.layers[0].attn_weights[:, -1]

    # Select First Layer and First Head
    # attention = self.model.encoder.layers[0].attn_weights[:, 0]

    # Extract attention weights from each encoder block
    # attentions = [layer.self_attention.attn for layer in self.model.encoder.layers]

    # Average attention weights across heads
    # attn = torch.stack([attn.mean(dim=1) for attn in attentions]).mean(dim=0)

    # self.model.encoder.layers[-1].attn_weights
    # self.model.encoder.layers[-1].self_attention(input_data)
    # x, self.attn_weights = self.model.encoder.layers[-1].self_attention(x, x, x, need_weights=self.track_attention_weights)
    # self.model._process_input(input_data)

    # Attention is num_heads x batch_size x seq_length x seq_length
    # batch_size, w, h = attention.shape
    # img_shape = [1, img.shape[2], img.shape[3]] # Will be 1 dimension heatmap

    # attention_map = image_functions.conditional_resize(attention.detach(), img_shape, interpolation=cv2.INTER_NEAREST)
    # attention.unsqueeze(0)
    attention_map = F.interpolate(
      attention.unsqueeze(1),
      size=(img.shape[2], img.shape[3]), # Patches are square
      mode="bilinear",
      align_corners=False
    )

    # Visualize each head
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
    # fig, axs = plt.subplots(ncols=batch_size, figsize=(w, h))
    # for i in range(batch_size):
    #     axs[i].matshow(attention_map)
    # plt.imshow(attention_map, cmap=plt.cm.jet)
    # plt.show()

    if shift_single_result == True and attention_map.shape[0] == 1:
      attention_map = attention_map[0][0]

    # Clean-up
    # self.clear_attn_weights()

    # del input_data
    # del outputs
    # del attention

    # helper_functions.clear_cuda_objects()
    helper_functions.clear_gpu(torch, self.config, pause=False)

    return attention_map

  def forward(self, x):
    # Make sure input is on the correct device
    # x = self.model(x.to(self.dev, dtype=self.dtype))
    # x = self.model.forward(x.to(self.dev, dtype=self.dtype))

    # x = self.flatten(x)
    # x = x.mean(dim=1)

    # x = self.classifier(x)

    # return self.model(x)
    # return x

    return self.model.forward(x)

class deit_tiny_model(torch.nn.Module):
  def __init__(self, config):
    super(deit_tiny_model, self).__init__()

    self.config = config
    self.image_size =  config["input_size"][1] # Assume square size
    self.num_classes = config["number_of_outputs"]

    if "track_attention_weights" in config:
      self.track_attention_weights = config["track_attention_weights"]
    else:
      self.track_attention_weights = False

    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    if "average_attn_weights" in config:
      self.average_attn_weights = config["average_attn_weights"]
    else:
      self.average_attn_weights = False

    if "compute_attn_mean" in config: # mean attention per layer across heads
      self.compute_attn_mean = config["compute_attn_mean"]
    else:
      self.compute_attn_mean = True # Force mean otherwise

    # Naming is a little confusing. Mean here is across layers
    if "compute_mean_attention" in config:
      self.compute_mean_attention = config["compute_mean_attention"]
    else:
      self.compute_mean_attention = False # Off by default

    if "mean_attention_index" in config:
      self.mean_attention_index = config["mean_attention_index"]
    else:
      self.mean_attention_index = -1 # Last layer by default

    if "pretrained_weights" in config:
      self.pretrained_weights = config["pretrained_weights"]
    else:
      self.pretrained_weights = False

    self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=self.pretrained_weights)

    self.cnn_layer = self.model.patch_embed.proj
    self.target_layers = [self.cnn_layer]

  def __del__(self):
    del self.target_layers
    del self.cnn_layer

    del self.model.head
    del self.model

    helper_functions.clear_gpu(torch, self.config, pause=False)

  def attention(self, img, shift_single_result=False):
    if self.track_attention_weights == False:
      raise("track_attention_weights has to be set to True!")

    # Set model to eval
    self.model = self.model.eval()
    # self.model = self.model.train()

    # Make sure input is on the correct device
    input_data = img.to(self.dev, dtype=self.data_dtype)

    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    with torch.no_grad():
      outputs = self(input_data)
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=15))

    # TODO: Pull out all layers and combine them
    # For now take the last layer

    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
    # Select last layer and last head for now
    attention = None

    # if self.compute_attn_mean and not self.compute_mean_attention:
    #   attention = self.model.blocks[-1].mean_attn_weights
    # elif not self.compute_mean_attention:
    #   attention = self.model.blocks[-1].attn_weights[:, -1]
    # elif self.compute_attn_mean and self.compute_mean_attention:
    #   mean_attn_weights = [layer.mean_attn_weights for layer in self.model.encoder.layers]
    #   mean_attn_weights = torch.stack(mean_attn_weights)
    #   attention = torch.mean(mean_attn_weights, dim=0)
    # elif self.compute_mean_attention:
    #   mean_attn_weights = [layer.attn_weights[:, self.mean_attention_index] for layer in self.model.encoder.layers]
    #   mean_attn_weights = torch.stack(mean_attn_weights)
    #   attention = torch.mean(mean_attn_weights, dim=0)

    # Manually compute the attention
    last_block = self.model.blocks[-1]

    attention_map = F.interpolate(
      attention.unsqueeze(1),
      size=(img.shape[2], img.shape[3]), # Patches are square
      mode="bilinear",
      align_corners=False
    )

    if shift_single_result == True and attention_map.shape[0] == 1:
      attention_map = attention_map[0][0]

    helper_functions.clear_gpu(torch, self.config, pause=False)

    return attention_map

  def forward(self, x):
    return self.model.forward(x)
