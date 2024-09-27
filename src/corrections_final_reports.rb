#!/bin/ruby
require 'console-style'
require 'csv'
# require 'json'
require 'oj'
require 'pry'
require 'shellwords'
require 'fileutils'

VERSION = 0.1

CONSOLE_OUTPUT_FILE = 'console_output.txt'.freeze
GROUPED_PERCENTAGES_FILE = 'total_result_grouped_percentages.csv'.freeze

TEXT_FILE = 'txt'.freeze
CSV_FILE = 'csv'.freeze
SKIP_FILES_AND_FOLDERS = [
  TEXT_FILE,
  CSV_FILE,
  "model_summaries",
  "signal_results",
  "corrections_combined_models",
  "corrections_reports",
].freeze

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

BEST_MODEL_MATCH = "best_model_".freeze
BEST_AUTONOMY_MATCH = "best_val_autonomy_model_".freeze

BEST_EPOCH_HEADINGS = "Model Name,Best Epoch".freeze
BEST_AUTONOMY_HEADINGS = "Model Name,Best Autonomy Epoch".freeze

BOX_AND_WHISKER_EPOCH_HEADINGS = "Model Name,min,q1,median,q3,max".freeze
BOX_AND_WHISKER_GRADCAM_HEADINGS = "label,min,q1,median,q3,max".freeze

MODEL_ORDER = [
  "Net SVF",
  "Net HVF",
  "End to End",
  "Autonomous Cookbook",
  "TestModel2",
  "TestModel1",
]

BEST_AUTONOMY_BOX_AND_WHISKER_FILENAME = "best_autonomy_box_and_whisker.csv"
BEST_AUTONOMY_EPOCHS_FILENAME = "best_autonomy_epochs.csv"
BEST_AUTONOMY_GRADCAM_BOX_AND_WHISKER_FILENAME = "best_autonomy_gradcam_box_and_whisker.csv"
BEST_AUTONOMY_GRADCAM_PER_GROUP_FILENAME = "best_autonomy_gradcam_per_group.csv"
BEST_EPOCH_BOX_AND_WHISKER_FILENAME = "best_epoch_box_and_whisker.csv"
BEST_EPOCH_GRADCAM_BOX_AND_WHISKER_FILENAME = "best_epoch_gradcam_box_and_whisker.csv"
BEST_EPOCH_GRADCAM_PER_GROUP_FILENAME = "best_epoch_gradcam_per_group.csv"
BEST_EPOCHS_FILENAME = "best_epochs.csv"

TRAINING_AVERAGE_GRAPHS = "training_average_graphs.csv"
TRAINING_MIN_GRAPHS = "training_min_graphs.csv"
TRAINING_MAX_GRAPHS = "training_max_graphs.csv"

COMBINED_FOLDER_NAME = "corrections_combined_models"

# Methods
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

def to_csv_str(rows)
  rows.join("\n")
end

def extract_line_from_txt(file_path, matching_line)
  file = open_file(file_path)
  return unless file

  file.split("\n").select { |line| line.include? matching_line }.map { |line| line.split(": ") }
end

def list_files(full_path)
  `ls "#{full_path}"`.split("\n")
end

def calc_q1(values)
  # https://en.wikipedia.org/wiki/Quartile
  sorted = values.sort
  midpoint = values.length / 2 # integer division
  median = calc_median(values)

  lower_half = sorted[0, midpoint]

  if values.length.odd?
    lower_half = lower_half.delete(median)
  end

  calc_median(lower_half)
end

def calc_median(values)
  sorted = values.sort
  midpoint = values.length / 2 # integer division

  if values.length.even?
    sorted[midpoint-1, 2].sum / 2.0
  else
    sorted[midpoint]
  end
end

def calc_q3(values)
  # https://en.wikipedia.org/wiki/Quartile
  sorted = values.sort
  midpoint = values.length / 2 # integer division
  median = calc_median(values)

  upper_half = sorted[midpoint, sorted.size]

  if values.length.odd?
    upper_half = upper_half.delete(median)
  end

  calc_median(upper_half)
end

if ARGV.length != 2
  abort 'You have to pass the folder path and experimental names.'
end

folder_path = ARGV.shift
experimental_runs_str = ARGV.shift

experimental_runs = experimental_runs_str.split(",")

ConsoleStyle::Functions.print_heading("Process Correction Reports 2024: #{folder_path} (v#{VERSION}) ...")
puts "\n\n"

ConsoleStyle::Functions.print_heading("Generating sub-folders ...")
complete = ConsoleStyle::Functions.style('Done!', ConsoleStyle::Functions::GREEN)
failure = ConsoleStyle::Functions.style('Fail!', ConsoleStyle::Functions::RED)

# New Folder
save_folder_path = "#{folder_path}/corrections_reports/"
ConsoleStyle::Functions.print_heading("Create Save folder ...")
ConsoleStyle::Functions.execute("mkdir -p #{save_folder_path}")

# 1. Check data is there
if MODEL_ORDER.size != experimental_runs.size
  puts "Not all models have been run! #{failure}"
  # Dont exit
end

experimental_runs.each do |experimental_run|
  print "Checking #{experimental_run} ..."
  run_outer_path = "#{folder_path}/#{experimental_run}"

  filenames = [
    BEST_AUTONOMY_BOX_AND_WHISKER_FILENAME,
    BEST_AUTONOMY_EPOCHS_FILENAME,
    BEST_AUTONOMY_GRADCAM_BOX_AND_WHISKER_FILENAME,
    BEST_AUTONOMY_GRADCAM_PER_GROUP_FILENAME,
    BEST_EPOCH_BOX_AND_WHISKER_FILENAME,
    BEST_EPOCH_GRADCAM_BOX_AND_WHISKER_FILENAME,
    BEST_EPOCH_GRADCAM_PER_GROUP_FILENAME,
    BEST_EPOCHS_FILENAME,
    TRAINING_AVERAGE_GRAPHS,
    TRAINING_MIN_GRAPHS,
    TRAINING_MAX_GRAPHS,
  ]

  check = true

  filenames.each do |filename|
    filepath = "#{run_outer_path}/#{COMBINED_FOLDER_NAME}/#{filename}"
    file = open_csv(filepath)

    unless file
      puts "#{experimental_run} is missing #{filename}!"
      check = false
    end
  end

  if check
    puts " #{complete}\n"
  else
    puts failure
  end
end

# 2. Copy over reports to a set of folders
experimental_runs.each do |experimental_run|
  print "Copying results from #{experimental_run} ..."
  run_outer_path = "#{folder_path}/#{experimental_run}"

  filenames = [
    BEST_AUTONOMY_BOX_AND_WHISKER_FILENAME,
    BEST_AUTONOMY_EPOCHS_FILENAME,
    BEST_AUTONOMY_GRADCAM_BOX_AND_WHISKER_FILENAME,
    BEST_AUTONOMY_GRADCAM_PER_GROUP_FILENAME,
    BEST_EPOCH_BOX_AND_WHISKER_FILENAME,
    BEST_EPOCH_GRADCAM_BOX_AND_WHISKER_FILENAME,
    BEST_EPOCH_GRADCAM_PER_GROUP_FILENAME,
    BEST_EPOCHS_FILENAME,
    TRAINING_AVERAGE_GRAPHS,
    TRAINING_MIN_GRAPHS,
    TRAINING_MAX_GRAPHS,
  ]

  begin
    sub_folders = ConsoleStyle::Functions.execute_split_lines("ls #{run_outer_path}").reject { |names| SKIP_FILES_AND_FOLDERS.any? { |name| names.include?(name) } }
    sub_folder = sub_folders.first

    config_path = "#{run_outer_path}/#{sub_folder}/#{sub_folder}_config.json"

    config = open_json(config_path)
    model_name = config["model_name"]

    subfolder_save_path = "#{save_folder_path}/#{model_name}"
    ConsoleStyle::Functions.execute("mkdir -p '#{subfolder_save_path}'")

    filenames.each do |filename|
      filepath = "#{run_outer_path}/#{COMBINED_FOLDER_NAME}/#{filename}"
      ConsoleStyle::Functions.execute("cp -f '#{filepath}' '#{subfolder_save_path}'")
    end
  rescue => StandardError
  end

  puts " #{complete}\n"
end

# 3. Generate combined autonomy box and whisker
print "Generate Combined Autonomy Box and Whisker ..."

header = ""
output_csv_lines = []

MODEL_ORDER.each do |model_name|
  experimental_runs.each do |experimental_run|
    run_outer_path = "#{folder_path}/#{experimental_run}"

    begin
      sub_folders = ConsoleStyle::Functions.execute_split_lines("ls #{run_outer_path}").reject { |names| SKIP_FILES_AND_FOLDERS.any? { |name| names.include?(name) } }
      sub_folder = sub_folders.first

      config_path = "#{run_outer_path}/#{sub_folder}/#{sub_folder}_config.json"
      config = open_json(config_path)

      if model_name == config["model_name"]
        filename = BEST_AUTONOMY_BOX_AND_WHISKER_FILENAME
        filepath = "#{run_outer_path}/#{COMBINED_FOLDER_NAME}/#{filename}"
        csv_file = open_csv(filepath)

        header = csv_file.headers().join(DELIMETER)
        output_csv_lines << [csv_file[0].to_s.gsub("\n", "")]
      end
    rescue => StandardError
    end
  end
end

best_epochs_filepath = "#{save_folder_path}/best_autonomy_epochs_box_and_whisker.csv"
best_epochs_str = to_csv_str([header] + output_csv_lines)
save_file(best_epochs_filepath, best_epochs_str, permission: 'w')
puts " #{complete}\n"

# 4. Generate combined epoch box and whisker

print "Generate Combined Best Epoch Box and Whisker ..."

header = ""
output_csv_lines = []

MODEL_ORDER.each do |model_name|
  experimental_runs.each do |experimental_run|
    run_outer_path = "#{folder_path}/#{experimental_run}"

    begin
      sub_folders = ConsoleStyle::Functions.execute_split_lines("ls #{run_outer_path}").reject { |names| SKIP_FILES_AND_FOLDERS.any? { |name| names.include?(name) } }
      sub_folder = sub_folders.first

      config_path = "#{run_outer_path}/#{sub_folder}/#{sub_folder}_config.json"
      config = open_json(config_path)

      if model_name == config["model_name"]
        filename = BEST_EPOCH_BOX_AND_WHISKER_FILENAME
        filepath = "#{run_outer_path}/#{COMBINED_FOLDER_NAME}/#{filename}"
        csv_file = open_csv(filepath)

        header = csv_file.headers().join(DELIMETER)
        output_csv_lines << [csv_file[0].to_s.gsub("\n", "")]
      end
    rescue => StandardError
    end
  end
end

best_epochs_filepath = "#{save_folder_path}/best_epochs_box_and_whisker.csv"
best_epochs_str = to_csv_str([header] + output_csv_lines)
save_file(best_epochs_filepath, best_epochs_str, permission: 'w')
puts " #{complete}\n"

puts complete
