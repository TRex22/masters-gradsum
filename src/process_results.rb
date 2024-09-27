#!/bin/ruby

require 'console-style'
require 'csv'
# require 'json'
require 'oj'
require 'pry'
# require 'pry-byebug'
require 'shellwords'
require 'fileutils'

VERSION = 0.13
DATASET = 'cityscapes'.freeze
DATASET_FROMGAMES = 'fromgames'.freeze
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

TRAIN_LOSS_MATCH = 'selected_train_loss'.freeze
BEST_EPOCH_MATCH = 'Saving best model epoch '.freeze
BEST_AUTONOMY_EPOCH_MATCH = 'Saving best autonomy model epoch '.freeze

TEST_UDACITY_LOSS_MATCH = 'udacity_test Test Loss (Best val loss model): '.freeze
TEST_COOKBOOK_LOSS_MATCH = 'cookbook_test Test Loss (Best val loss model): '.freeze
TEST_CITYSCAPES_LOSS_MATCH = 'cityscapes_test Test Loss (Best val loss model): '.freeze

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

ConsoleStyle::Functions.print_heading("Generating Report for: #{folder_path}...")
sub_folders = ConsoleStyle::Functions.execute_split_lines("ls #{folder_path}").reject { |names| SKIP_FILES_AND_FOLDERS.any? { |name| names.include?(name) } }

sub_folder_count = ConsoleStyle::Functions.style(sub_folders.size, ConsoleStyle::Functions::GREEN)
puts "Total number of models: #{sub_folder_count}\n\n"

ConsoleStyle::Functions.print_heading("Generating sub-folders ...")
complete = ConsoleStyle::Functions.style('Done!', ConsoleStyle::Functions::GREEN)
failure = ConsoleStyle::Functions.style('Fail!', ConsoleStyle::Functions::RED)

def generate_paths(base_path, sub_folder, selected_dataset)
  grad_cam_path = "#{base_path}/grad_cam/#{selected_dataset}/"
  grad_cam_fine_path = "#{base_path}/fine_grad/#{selected_dataset}/"
  grad_cam_coarse_path = "#{base_path}/coarse_grad/#{selected_dataset}/"
  grad_cam_fromgames_path = "#{base_path}/fromgames/#{selected_dataset}/"
  grad_cam_fromgames_fromgames_path = "#{base_path}/fromgames/fromgames/#{selected_dataset}/"
  grad_cam_turning_path = "#{base_path}/grad_cam/#{selected_dataset}_turning_only/"

  config = open_json("#{base_path}/#{sub_folder}_config.json")

  base_grad_cam = open_csv("#{grad_cam_path}/#{GRAD_CAM_TYPE}/total_threshold_percentage.csv") ||
    open_csv("#{grad_cam_fine_path}/#{GRAD_CAM_TYPE}/total_threshold_percentage.csv") ||
    open_csv("#{grad_cam_coarse_path}/#{GRAD_CAM_TYPE}/total_threshold_percentage.csv") ||
    open_csv("#{grad_cam_fromgames_path}/#{GRAD_CAM_TYPE}/total_threshold_percentage.csv") ||
    open_csv("#{grad_cam_fromgames_fromgames_path}/#{GRAD_CAM_TYPE}/total_threshold_percentage.csv")

  turning_only_base_grad_cam = open_csv("#{grad_cam_turning_path}/#{GRAD_CAM_TYPE}/total_threshold_percentage.csv")

  [config, base_grad_cam, turning_only_base_grad_cam]
end

csv_files = sub_folders.map { |sub_folder|
  print("#{sub_folder}...")

  base_path = "#{folder_path}/#{sub_folder}/"
  config, base_grad_cam, turning_only_base_grad_cam = generate_paths(base_path, sub_folder, DATASET)

  if config.nil? || base_grad_cam.nil? || turning_only_base_grad_cam.nil?
    config, base_grad_cam, turning_only_base_grad_cam = generate_paths(base_path, sub_folder, DATASET_FROMGAMES)
  end

  if config
    run_name = config["run_name"]
  else
    run_name = sub_folder
  end

  save_path = "#{folder_path}/model_summaries/#{run_name}_"

  if base_grad_cam.nil? #|| turning_only_base_grad_cam.nil?
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
      save_path: save_path,
      base_grad_cam: base_grad_cam,
      turning_only_base_grad_cam: turning_only_base_grad_cam
    )
  ]
}.compact.to_h

################################################################################
# Model CSVs
puts "\n\n"
ConsoleStyle::Functions.print_heading("Generating Model CSVs ...")
sub_folders.each do |sub_folder|
  print("#{sub_folder}...")

  csvs = csv_files[sub_folder]
  if csvs.nil?
    puts failure
    next
  end

  # base_path = csvs[:base_path]
  save_path = csvs[:save_path].gsub(' ', '').gsub('[', '').gsub(']', '_')

  base_grad_cam_str = csv_from(csvs[:base_grad_cam], SIGNAL_HEADINGS)
  save_file("#{save_path}base_grad_cam.csv", base_grad_cam_str, permission: 'w')

  if csvs[:turning_only_base_grad_cam]
    turning_only_base_grad_cam_str = csv_from(csvs[:turning_only_base_grad_cam], SIGNAL_HEADINGS)
    save_file("#{save_path}turning_only_base_grad_cam.csv", turning_only_base_grad_cam_str, permission: 'w')
  end

  puts complete
end

################################################################################
# Signal CSVs
puts "\n\n"
ConsoleStyle::Functions.print_heading("Generating signal CSVs ...")
SIGNAL_HEADINGS.each do |signal|
  print("#{signal}...")

  save_path = "#{folder_path}/signal_results/"
  epoch_column = []

  base_grad_cam_columns = sub_folders.map { |sub_folder|
    csvs = csv_files[sub_folder]
    if csvs.nil?
      puts failure
      next
    end

    # TODO: May want to fix this to rather track all epochs and zero missing values
    new_epoch_column = ['i'] + csvs[:base_grad_cam]['i'] # Not that efficient :shrug:
    if new_epoch_column.size > epoch_column.size
      epoch_column = new_epoch_column
    end

    # epoch_column = epoch_column + csvs[:base_grad_cam]['i']

    [sub_folder] + csvs[:base_grad_cam][signal]
  }.compact

  # epoch_column = ['i'] + epoch_column.uniq.sort_by { |name| name.to_i }
  expected_column_count = base_grad_cam_columns.map(&:size).max

  base_grad_cam_columns = base_grad_cam_columns.map { |model_signals|
    if model_signals.size >= expected_column_count
      model_signals
    else
      number_of_missing_signals = expected_column_count - model_signals.size
      model_signals + (["0"] * number_of_missing_signals)
    end
  }

  base_grad_cam_signal_str = convert_to_csv_str([epoch_column] + base_grad_cam_columns)

  if base_grad_cam_signal_str == ""
    puts failure
    next
  end

  turning_only_columns = sub_folders.map { |sub_folder|
    csvs = csv_files[sub_folder]
    # epoch_column = ['i'] + csvs[:base_grad_cam]['i'] # Not that efficient :shrug:
    next if csvs.nil? || csvs[:turning_only_base_grad_cam].nil?
    [sub_folder] + csvs[:turning_only_base_grad_cam][signal]
  }.compact

  expected_column_count = turning_only_columns.map(&:size).max

  turning_only_columns = turning_only_columns.map { |model_signals|
    if model_signals.size >= expected_column_count
      model_signals
    else
      number_of_missing_signals = expected_column_count - model_signals.size
      model_signals + (["0"] * number_of_missing_signals)
    end
  }

  turning_only_signal_str = convert_to_csv_str([epoch_column] + turning_only_columns)

  save_file("#{save_path}/#{signal.gsub(' ', '_')}_base_grad_cam.csv", base_grad_cam_signal_str, permission: 'w')
  save_file("#{save_path}/#{signal.gsub(' ', '_')}_turning_only_base_grad_cam.csv", turning_only_signal_str, permission: 'w')

  puts complete
end

################################################################################
# KL Divergence
puts "\n\n"
ConsoleStyle::Functions.print_heading("Generating KL-Divergence CSV ...")
kl_values = sub_folders.map { |sub_folder|
  base_path = "#{folder_path}/#{sub_folder}/"
  grad_cam_path = "#{base_path}/grad_cam/"

  file_path = "#{grad_cam_path}/#{CONSOLE_OUTPUT_FILE}"

  kl_pair = extract_line_from_txt(file_path, KL_DIVERGENCE_MATCH)&.first
  next unless kl_pair

  kl_div = kl_pair[1]

  {
    model: sub_folder,
    kl_divergence_between_last_epoch_and_first: kl_div
  }
}.compact

kl_csv_string = "Model name,KL Divergence Between Last Epoch And First\n"
kl_values.each do |kl_value|
  kl_csv_string += "#{kl_value[:model]},#{kl_value[:kl_divergence_between_last_epoch_and_first]}\n"
end

save_path = "#{folder_path}/kl_divergence_between_last_and_first.csv"
save_file("#{save_path}", kl_csv_string, permission: 'w')

puts complete

################################################################################
# Training Results
puts "\n\n"
ConsoleStyle::Functions.print_heading("Generating Training CSV ...")
training_results = sub_folders.map { |sub_folder|
  base_path = "#{folder_path}/#{sub_folder}/"
  file_path = "#{base_path}/#{CONSOLE_OUTPUT_FILE}"

  best_loss_epoch = extract_line_from_txt(file_path, BEST_EPOCH_MATCH)&.flatten&.last[/\d+/]

  # BEST_AUTONOMY_EPOCH_MATCH
  losses = extract_line_from_txt(file_path, TRAIN_LOSS_MATCH)&.find { |line| line[1].include?("#{best_loss_epoch} selected_train_loss") }

  next unless losses

  {
    model: sub_folder,
    train_loss: losses[2].split(' selected_val_loss').first,
    val_loss: losses[3].split(' lr').first,
    lr: losses[4]
  }
}.compact

train_csv_string = "Model Name,Train Loss (MSE),Validation Loss (MSE),Learning Rate\n"
training_results.each do |train_value|
  train_csv_string += "#{train_value[:model]},#{train_value[:train_loss]},#{train_value[:val_loss]},#{train_value[:lr]}\n"
end

save_path = "#{folder_path}/train_results.csv"
save_file("#{save_path}", train_csv_string, permission: 'w')

puts complete

################################################################################
# Test Results
puts "\n\n"
ConsoleStyle::Functions.print_heading("Generating Tests CSV ...")
testing_results = sub_folders.map { |sub_folder|
  base_path = "#{folder_path}/#{sub_folder}/"
  file_path = "#{base_path}/#{CONSOLE_OUTPUT_FILE}"

  udacity_test_loss = extract_line_from_txt(file_path, TEST_UDACITY_LOSS_MATCH)&.last&.last
  cookbook_test_loss = extract_line_from_txt(file_path, TEST_COOKBOOK_LOSS_MATCH)&.last&.last
  cityscapes_test_loss = extract_line_from_txt(file_path, TEST_CITYSCAPES_LOSS_MATCH)&.last&.last

  {
    model: sub_folder,
    udacity_test_loss: udacity_test_loss,
    cookbook_test_loss: cookbook_test_loss,
    cityscapes_test_loss: cityscapes_test_loss
  }
}

test_csv_string = "Model Name,Udacity Test Loss(MSE),Cookbook Test Loss(MSE),Cityscapes Test Loss(MSE)\n"
testing_results.each do |testing_value|
  test_csv_string += "#{testing_value[:model]},#{testing_value[:udacity_test_loss]},#{testing_value[:cookbook_test_loss]},#{testing_value[:cityscapes_test_loss]}\n"
end

save_path = "#{folder_path}/test_results.csv"
save_file("#{save_path}", test_csv_string, permission: 'w')

puts complete

################################################################################
# Synthetic Autonomy Results
puts "\n\n"
ConsoleStyle::Functions.print_heading("Generating Synthetic Autonomy CSV ...")
autonomy_results = sub_folders.map { |sub_folder|
  base_path = "#{folder_path}/#{sub_folder}/"
  file_path = "#{base_path}/#{CONSOLE_OUTPUT_FILE}"

  udacity_autonomy = extract_line_from_txt(file_path, AUTONOMY_UDACITY_MATCH)&.last&.last
  cookbook_autonomy = extract_line_from_txt(file_path, AUTONOMY_COOKBOOK_MATCH)&.last&.last
  cityscapes_autonomy = extract_line_from_txt(file_path, AUTONOMY_CITYSCAPES_MATCH)&.last&.last
  udacity_autonomy_new = extract_line_from_txt(file_path, AUTONOMY_UDACITY_NEW_MATCH)&.last&.last

  {
    model: sub_folder,
    udacity_autonomy: udacity_autonomy,
    cookbook_autonomy: cookbook_autonomy,
    cityscapes_autonomy: cityscapes_autonomy,
    udacity_autonomy_new: udacity_autonomy_new,
  }
}

autonomy_csv_string = "Model Name,Udacity Autonomy %,Cookbook Autonomy %,Cityscapes Autonomy %\n"
autonomy_results.each do |autonomy_value|
  autonomy_csv_string += "#{autonomy_value[:model]},#{autonomy_value[:udacity_autonomy]},#{autonomy_value[:cookbook_autonomy]},#{autonomy_value[:cityscapes_autonomy] || autonomy_value[:udacity_autonomy]}\n"
end

save_path = "#{folder_path}/autonomy_results.csv"
save_file("#{save_path}", autonomy_csv_string, permission: 'w')

puts complete

################################################################################
# DataSet Information
# puts "\n\n"
# ConsoleStyle::Functions.print_heading("DataSet Information ...")

################################################################################
puts "#{complete}"
