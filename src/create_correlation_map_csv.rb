#!/bin/ruby

require 'console-style'
require 'csv'
# require 'json'
require 'oj'
require 'pry'
require 'shellwords'
require 'fileutils'

VERSION = 0.11
DATASET = 'cityscapes'.freeze
GRAD_CAM_TYPE = 'Base'.freeze # 'Custom'
DELIMETER = ','.freeze
SIGNAL_HEADINGS = [
  # 'i',
  'ground',
  'road',
  'sidewalk',
  'traffic light',
  'traffic sign',
  'sky',
  'pole',
  'person',
  'car'
].freeze

TEXT_FILE = 'txt'
CSV_FILE = 'csv'
SKIP_FILES_AND_FOLDERS = [
  TEXT_FILE,
  CSV_FILE,
  "model_summaries",
  "signal_results",
  "corrections_combined_models",
  "corrections_reports",
]

KL_DIVERGENCE_MATCH = 'KL_Divergence (Base type): '.freeze
TRAIN_LOSS_MATCH = 'opt: Adam loss_name: mse_loss_func epoch:'.freeze

TEST_UDACITY_LOSS_MATCH = 'udacity_test Test Loss: '.freeze
TEST_COOKBOOK_LOSS_MATCH = 'cookbook_test Test Loss: '.freeze
TEST_CITYSCAPES_LOSS_MATCH = 'cityscapes_test Test Loss: '.freeze

AUTONOMY_UDACITY_MATCH = 'Synthetic test autonomy: '.freeze # Udacity / set DataSet
AUTONOMY_COOKBOOK_MATCH = 'cookbook synthetic test autonomy: '.freeze
AUTONOMY_CITYSCAPES_MATCH = 'cityscapes synthetic test autonomy: '.freeze
AUTONOMY_UDACITY_NEW_MATCH = 'udacity synthetic test autonomy: '.freeze

CONSOLE_OUTPUT_FILE = 'console_output.txt'.freeze

# TODO: Make dynamic
# TODO: Edge percentages

def escape(path)
  Shellwords.shellescape(path).gsub('//', '/').gsub(/ /, '\ ')
end

def save_file(path, value, permission: 'w')
  if path.include?('.')
    folder_path = path.split('/')
    folder_path.pop
    folder_path = "'#{folder_path.join('/')}'"
  else
    folder_path = "'#{path.join('/')}'"
  end

  ConsoleStyle::Functions.execute("mkdir -p #{folder_path}")
  ConsoleStyle::Functions.execute("touch #{path}")
  # FileUtils.mkdir_p folder_path

  File.open(path.gsub('//', '/'), permission) do |file|
    file.write value
  end
rescue => _e
  false
end

def open_file(path)
  File.read(path)
rescue => _e
  false
end

def open_json(path)
  file = open_file(path)
  return unless file

  Oj.load(file) # can do the same with JSON
rescue => _e
  false
end

def open_csv(path, headers: true)
  file = open_file(path)
  return unless file

  CSV.parse(file, headers: headers)
end

def extract_column(name, csv)
  csv[name]
end

def csv_from(raw_csv, selected_headers)
  columns = (["i"] + selected_headers).map do |heading|
    column = [heading]
    column += extract_column(heading, raw_csv)

    column
  end

  convert_to_csv_str(columns)
end

def convert_to_csv_str(columns)
  columns.transpose.map { |col| col.join(DELIMETER) }.join("\n")
end

def extract_line_from_txt(file_path, matching_line)
  file = open_file(file_path)
  return unless file

  file.split("\n").select { |line| line.include? matching_line }.map { |line| line.split(": ") }
end

if ARGV.length != 1
  abort 'You have to pass the folder path.'
end

folder_path = ARGV.shift

ConsoleStyle::Functions.print_heading("Generating correlation map for: #{folder_path}...")
sub_folders = ConsoleStyle::Functions.execute_split_lines("ls #{folder_path}").reject { |names| SKIP_FILES_AND_FOLDERS.any? { |name| names.include?(name) } }

sub_folder_count = ConsoleStyle::Functions.style(sub_folders.size, ConsoleStyle::Functions::GREEN)
puts "Total number of models: #{sub_folder_count}"
puts "\n\n"

ConsoleStyle::Functions.print_heading("Generating sub-folders ...")
complete = ConsoleStyle::Functions.style('Done!', ConsoleStyle::Functions::GREEN)
failure = ConsoleStyle::Functions.style('Fail!', ConsoleStyle::Functions::RED)

keys = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

csv_files = sub_folders.map { |sub_folder|
  print("#{sub_folder}...")
  base_path = "#{folder_path}/#{sub_folder}/"
  config = open_json("#{base_path}/#{sub_folder}_config.json")

  grad_cam_path = "#{base_path}/fine_grad/#{DATASET}/"
  results = open_csv("#{grad_cam_path}/#{GRAD_CAM_TYPE}/total_result_grouped_percentages.csv")

  if results.nil?
    grad_cam_path = "#{base_path}/grad_cam/#{DATASET}/"
    results = open_csv("#{grad_cam_path}/#{GRAD_CAM_TYPE}/total_result_grouped_percentages.csv")
  end

  if results.nil?
    grad_cam_path = "#{base_path}/fromgames/#{DATASET}/"
    results = open_csv("#{grad_cam_path}/#{GRAD_CAM_TYPE}/total_result_grouped_percentages.csv")
  end

  if results.nil?
    grad_cam_path = "#{base_path}/fromgames/fromgames/#{DATASET}/"
    results = open_csv("#{grad_cam_path}/#{GRAD_CAM_TYPE}/total_result_grouped_percentages.csv")
  end

  label = nil
  keys.each do |key|
    if base_path.include?(key)
      label = key
    end
  end

  results_path = "#{grad_cam_path}/cityscapes/Base/total_result_grouped_percentages.csv"

  if results.nil? #|| turning_only_results.nil?
    puts failure
    next
  end

  # TODO: Edge percentage

  puts complete

  [
    sub_folder,
    OpenStruct.new(
      config: config,
      base_path: base_path,
      results: results,
      results_path: results_path,
      label: label
    )
  ]
}.compact.to_h

puts "group,flat,human,vehicle,construction,object,nature,sky,void"
csv_files.each do |pairs|
  _path, results = pairs

  group_labels = results.results.headers - ["i"]
  values = group_labels.map do |label|
    results.results[label].map(&:to_f).sum()/results.results[label].size
  end

  puts "#{results.label},#{values.join(",")}"
end

puts complete
puts "#{complete}"
