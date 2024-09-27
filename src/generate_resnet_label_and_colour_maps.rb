require 'pry'
require 'json'
require 'httparty'

# See: https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_resnet.ipynb#scrollTo=primary-bridges
IMAGENET_CLASSES_PATH="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

raw_classes = HTTParty.get(IMAGENET_CLASSES_PATH).split("\n")

# line no;class
labels = (0..raw_classes.size).map { |i| [raw_classes[i], i] }.to_h
pry
