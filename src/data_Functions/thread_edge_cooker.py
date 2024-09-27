# https://stackoverflow.com/questions/30135091/write-thread-safe-to-file-in-python

from threading import Thread
from pathlib import Path
import cv2

# from thread_writer import ThreadWriter
import image_functions
import cooking_functions
import data_functions

class ThreadEdgeCooker:
  def __init__(self, batch, config, meta_dataset, dataset_string, base_path_linux):
    self.batch = batch
    self.config = config
    self.meta_dataset = meta_dataset
    self.dataset_string = dataset_string
    self.base_path_linux = base_path_linux

    self.finished = False
    self.thread = Thread(name = "ThreadEdgeCooker", target=self.internal_converter)

  def start(self):
    self.thread.start()

  # def finished(self):
  #   return self.finished

  def internal_converter(self):
    while not self.finished:
      for i in self.batch:
        if i < self.meta_dataset.shape[0]:
          cooking_functions.sequential_cook_edge_data(i, self.config, self.meta_dataset, self.dataset_string, base_path_linux=self.base_path_linux)

      self.finished = True
      return True # Trigger thread to close
