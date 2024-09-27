# https://stackoverflow.com/questions/30135091/write-thread-safe-to-file-in-python

from threading import Thread
# from thread_writer import ThreadWriter
import image_functions
import cooking_functions

class ThreadCooker:
  def __init__(self, batch, meta_dataset, dim_shift, roi, greyscale, dataset_string, base_save_path, detect_path):
    self.batch = batch
    self.meta_dataset = meta_dataset
    self.dim_shift = dim_shift
    self.roi = roi
    self.greyscale = greyscale
    self.dataset_string = dataset_string
    self.base_save_path = base_save_path
    self.detect_path = detect_path

    self.finished = False
    self.thread = Thread(name = "ThreadCooker", target=self.internal_converter)

  def start(self):
    self.thread.start()

  # def finished(self):
  #   return self.finished

  def internal_converter(self):
    while not self.finished:
      for i in self.batch:
        if i < self.meta_dataset.shape[0]:
          image = cooking_functions.read_and_cook_image(self.meta_dataset.iloc[i], dim_shift=self.dim_shift, roi=self.roi, greyscale=self.greyscale)

          if image.shape[0] != 0:
            save_path = cooking_functions.fetch_cooked_relative_path_from(self.dataset_string, self.base_save_path, self.meta_dataset, i, self.detect_path)
            image_functions.save_image(image, save_path)

      self.finished = True
      return True # Trigger thread to close
