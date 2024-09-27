# https://stackoverflow.com/questions/30135091/write-thread-safe-to-file-in-python

from queue import Queue, Empty
from threading import Thread

class ThreadWriter:
  def __init__(self, *args):
    self.filewriter = open(*args)
    self.queue = Queue()
    self.finished = False
    self.auto_close = False
    Thread(name = "ThreadWriter", target=self.internal_writer).start()
    
  def write(self, data, auto_close=False):
    self.queue.put(data)
    self.auto_close = auto_close
    
  def internal_writer(self):
    while not self.finished:
      try:
        data = self.queue.get(True, 1)
      except Empty:
        continue
      self.filewriter.write(data)
      self.queue.task_done()

      if self.auto_close == True:
        self.queue.join()
        self.finished = True
        self.filewriter.close()
    
  def close(self):
    self.queue.join()
    self.finished = True
    self.filewriter.close()
