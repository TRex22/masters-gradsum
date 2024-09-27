from pathlib import Path
import platform
import torch
import json
import sys
import gc
import os
import pandas as pd
import time
import shutil
import re
from pynvml import *

from torch import nn
import torch.distributed as dist

sys.path.insert(1, './pyTorch_phoenix/')
import image_functions

# import random as rn
# rn.seed(12345)

import random, string
print(f"PyTorch Version: {torch.__version__}")

def randomword(length):
  letters = string.ascii_lowercase
  return ''.join(random.choice(letters) for i in range(length))

def execute_function(function, params, percent_to_execute=0.5):
  if (random.randrange(0, 100) > percent):
    function(*params)

  return True

def config_memory_key(config):
  free_memory = available_free_memory(config, verbose=False)
  free_memory_in_gbs = free_memory / 1000000000.00

  # 4, 6, 12, 24 # TODO: 48
  if free_memory_in_gbs < 6.0:
    return "available_4"
  elif free_memory_in_gbs < 10.0:
    return "available_6"
  elif free_memory_in_gbs < 11.0:
    return "available_12"
  elif free_memory_in_gbs < 12.0:
    return "available_12"
  elif free_memory_in_gbs < 23.5:
    return "available_24"

  return "available_4" # Safe fallback

def using_parallelise(config):
  return "use_data_parallel" in config and "data_parallel_device_ids" in config and config["use_data_parallel"]

def using_distributed_parallelise(config):
  return "use_distributed_data_parallel" in config and config["use_distributed_data_parallel"] and using_parallelise(config)

def trigger_torch_optimisations(torch_import, config):
  ################################################################################
  # Optimisations
  # https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  # os.environ["CUDA_VISIBLE_DEVICES"]=<YOUR_GPU_NUMBER_HERE>

  if "use_torch_optimisations" in config and config["use_torch_optimisations"]:
    torch_import.backends.cudnn.benchmark = True # Initial training steps will be slower
    torch_import.autograd.set_detect_anomaly(False)
    torch_import.autograd.profiler.profile(False)
    torch_import.autograd.profiler.emit_nvtx(False)

    torch_import.set_float32_matmul_precision('high')

    # https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
    # Requires Ampere or better GPU
    torch_import.backends.cuda.matmul.allow_tf32 = True
    torch_import.backends.cudnn.allow_tf32 = True

    # Full floating point 32-bit
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    return True
  elif "use_torch_optimisations" not in config:
    torch_import.backends.cudnn.benchmark = True # Initial training steps will be slower
    torch_import.autograd.set_detect_anomaly(False)
    torch_import.autograd.profiler.profile(False)
    torch_import.autograd.profiler.emit_nvtx(False)

    torch_import.set_float32_matmul_precision('high')

    return True

  # https://numba.readthedocs.io/en/stable/user/5minguide.html
  # from numba import jit
  # # @jit(nopython=False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
  ################################################################################

  return False

def get_cuda_device_id(config):
  device_parts = config["device_name"].split(":")

  if len(device_parts) < 2:
    return 0 # Default device id

  return int(device_parts[-1])

def set_default_gpu_id(config):
  if torch.cuda.device_count() <= 1:
    torch.cuda.set_device(0)
    # torch.cuda.current_device()
    # torch.cuda.get_device_name(0)

  device_id = get_cuda_device_id(config)
  log(f"SET GPU Device ID: {device_id}", config)

  torch.cuda.set_device(device_id)

# https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
def available_free_memory(config, verbose=True):
  try:
    if torch.cuda.is_available():
      nvmlInit()
      info = None
      default_device_id = get_cuda_device_id(config)

      if using_parallelise(config):
        for id in config["data_parallel_device_ids"]:
          h = nvmlDeviceGetHandleByIndex(id)
          id_info = nvmlDeviceGetMemoryInfo(h)

          if info is not None and id_info.free <= info.free:
            info = id_info

      if info is None:
        h = nvmlDeviceGetHandleByIndex(default_device_id)
        info = nvmlDeviceGetMemoryInfo(h)

      else:
        h = nvmlDeviceGetHandleByIndex(default_device_id)
        info = nvmlDeviceGetMemoryInfo(h)

      if verbose:
        log(f"GPU Device ID: {default_device_id}", config)
        log(f"Total GPU Memory: {info.total}", config)
        log(f"Free GPU Memory: {info.free}", config)
        log(f"Used GPU Memory: {info.used}", config)

      return info.free
  except:
    return 8000000000 # Default is 8 GB

  return 8000000000 # Default is 8 GB

def check_environment():
  return platform.system().lower()

def compute_model_dtype(config):
  if config["mixed_precision"] and config["bfloat16"]:
    return torch.float32 # autocast will automatically select the correct type
  elif config["mixed_precision"]:
    return torch.float32 #torch.float32 # float16 # Un-scaling fp16 fails

  return torch.float32 # fallback

# Used for data conversion
def compute_dtype(config):
  if config["mixed_precision"] and config["bfloat16"]:
    return torch.bfloat16
  elif config["mixed_precision"]:
    return torch.float16 # torch.float16 # Un-scaling fp16 fails

  return torch.float32 # fallback

def compute_gradcam_dtype(config):
  if config["mixed_precision"]:
    return torch.float16

  return torch.float32 # fallback

def detect_or_create_folder(folder_path, print_error=False):
  Path(folder_path).mkdir(parents=True, exist_ok=True)

def detect_folder(folder_path):
  return os.path.exists(folder_path)

def convert_to_tensor(value, dev, dtype):
  if isinstance(value, torch.Tensor):
    if value.device.type == dev and value.dtype == dtype:
      return value
    else:
      return value.to(dev, dtype=dtype)
  else:
    return torch.tensor(value).to(dev, dtype=dtype)

def clear_cuda_objects():
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj) and obj.is_cuda:
        del obj
    except:
      pass

def clear_gpu(torch_instance, config, pause=False):
  if torch.cuda.is_available():
    # if config["purge_cuda_memory"] == True and torch_instance.cuda.is_available():
    # If you need to purge memory
    gc.collect() # Force the Training data to be unloaded. Loading data takes ~10 secs

    if "train_op" in config:
      if config["train_op"]:
        clear_cuda_objects()

    if config["purge_cuda_memory"]:
      # time.sleep(15) # 30
      torch.cuda.empty_cache() # Will nuke the model in-memory
      torch.cuda.synchronize() # Force the unload before the next step

    with torch.no_grad():
      torch.cuda.empty_cache()

    gc.collect()

    if pause:
      time.sleep(15)

# Select device
def fetch_device(config, verbose=True):
  if verbose:
    log(f'Cuda available? {torch.cuda.is_available()}', config)

  # device_name = "cuda" if torch.cuda.is_available() else "cpu"
  device_name = config["device_name"]

  # device_name = "cpu"
  # device_name = "cuda"
  # device_name = "cuda:0" # TODO: Config
  # device_name = "cuda:1"

  dev = torch.device(device_name)

  if verbose:
    log(f'{device_name} selected.', config)

  return dev

def parallelise_model(model, config):
  # if using_parallelise(config) and config["mixed_precision"] and config["bfloat16"]:
  #   raise Exception("Cannot Run Parallel Code and Mixed Precision Brain Float. Not Yet Implemented!")

  if isinstance(model, nn.DataParallel):
    return model

  if isinstance(model, nn.parallel.DistributedDataParallel):
    return model

  # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
  # https://stackoverflow.com/questions/66498045/how-to-solve-dist-init-process-group-from-hanging-or-deadlocks
  if using_distributed_parallelise(config):
    dist.init_process_group("nccl", world_size=2) # TODO: Make configurable

    # return nn.parallel.DistributedDataParallel(model, device_ids=len(config["data_parallel_device_ids"])) # , output_device=i
    return nn.parallel.DistributedDataParallel(model)

  if using_parallelise(config):
    return nn.DataParallel(model, device_ids=config["data_parallel_device_ids"])

  return model

def run_name(config, original_path):
  if original_path or not config["load_model"]:
    return config["run_name"]

  return config["new_run_name"]

def compute_base_save_path(config, original_path=False):
  save_path_folder = f'{config["model_save_path"]}/{config["wandb_project_name"]}/'

  if not "in_test" in config:
    detect_or_create_folder(save_path_folder, print_error=False)

  full_save_path = f'{save_path_folder}/{run_name(config, original_path)}/'

  if not "in_test" in config:
    detect_or_create_folder(full_save_path, print_error=False)

  return full_save_path

def compute_model_save_path(config, original_path=False, epoch=None, checkpoint=False, append_path=""):
  base_save_path = compute_base_save_path(config, original_path=original_path)
  full_save_path = f'{base_save_path}/{run_name(config, original_path)}'

  if checkpoint and append_path != '':
    return f'{base_save_path}{append_path}checkpoint'

  if checkpoint:
    return f'{base_save_path}checkpoint'

  if append_path != '':
    return f'{base_save_path}{append_path}_{epoch}'

  if not epoch == None:
    return f'{full_save_path}_{epoch}'

  return full_save_path

def remove_checkpoint(config, append_path=""):
  path = compute_model_save_path(config, checkpoint=True, append_path=append_path)
  filePath = f'{path}.pth'

  if os.path.exists(filePath):
    os.remove(filePath)

def copy_init_model(config, top_model_init_path):
  source = top_model_init_path
  base_path = compute_base_save_path(config)
  destination = f'{compute_model_save_path(config, epoch=0)}.pth'

  shutil.copy(source, destination)

def open_config(config_path):
  # f = open(config_path)
  # config = json.load(f)
  # f.close()

  # Remove comments first
  raw_json = ""
  with open(config_path) as f:
    for line in f:
      line = line.partition('//')[0]
      line = line.rstrip()
      raw_json += f"{line}\n"

  config = json.loads(raw_json)
  config["input_size"] = (config["input_size"][0], config["input_size"][1], config["input_size"][2])

  return config

def open_json(path):
  # f = open(path)
  # config = json.load(f)
  # f.close()

  # Remove comments first
  raw_json = ""
  with open(path) as f:
    for line in f:
      line = line.partition('//')[0]
      line = line.rstrip()
      raw_json += f"{line}\n"

  config = json.loads(raw_json)

  return config

# You can use the built-in python logger but
# Id like to keep existing functionality
def log(text, config):
  if "presentation_mode" in config:
    return False

  print(text)

  if not "in_test" in config and config["log_to_file"]:
    base_path = compute_base_save_path(config)

    if "running_gradcam" in config:
      log_folder_path = f'{base_path}/grad_cam'
    else:
      log_folder_path = f'{base_path}'

    detect_or_create_folder(log_folder_path, print_error=False)
    log_path = f'{log_folder_path}/console_output.txt'

    with open(log_path, 'a') as f:
      f.write(f'{text}\n')

def save_csv(file_path, csv_data):
  with open(file_path, 'a') as f:
    f.write(f'{csv_data}\n')

def save_dataframe(file_path, df):
  df.to_csv(file_path, sep=',', header=True, index=True)

def open_dataframe(file_path):
  # full_file_path = os.path.normpath(os.path.abspath(file_path))
  # full_file_path = re.sub(' ', "\\ ", full_file_path)
  # full_file_path = re.sub(' ', '\\ ', Path(file_path).resolve())
  full_file_path = Path(file_path).resolve()
  return pd.read_csv(full_file_path)
