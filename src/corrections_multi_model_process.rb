#!/bin/ruby

require 'console-style'
require 'csv'
# require 'json'
require 'oj'
require 'pry'
require 'shellwords'
require 'fileutils'

VERSION = 0.3

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

BEST_MODEL_MATCH = "best_model_".freeze
BEST_AUTONOMY_MATCH = "best_val_autonomy_model_".freeze

BEST_EPOCH_HEADINGS = "Model Name,Best Epoch".freeze
BEST_AUTONOMY_HEADINGS = "Model Name,Best Autonomy Epoch".freeze

BOX_AND_WHISKER_EPOCH_HEADINGS = "Model Name,min,q1,median,q3,max".freeze
BOX_AND_WHISKER_GRADCAM_HEADINGS = "label,min,q1,median,q3,max".freeze

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
  values = values.compact

  # https://en.wikipedia.org/wiki/Quartile
  sorted = values.sort
  midpoint = values.length / 2 # integer division
  median = calc_median(values)

  lower_half = sorted[0, midpoint]

  if values.length.odd?
    lower_half.delete(median)
  end

  calc_median(lower_half)
end

def calc_median(values)
  values = values.compact
  sorted = values.sort
  midpoint = values.length / 2 # integer division

  if values.length.even?
    sorted[midpoint-1, 2].sum / 2.0
  else
    sorted[midpoint]
  end
end

def calc_q3(values)
  values = values.compact

  # https://en.wikipedia.org/wiki/Quartile
  sorted = values.sort
  midpoint = values.length / 2 # integer division
  median = calc_median(values)

  upper_half = sorted[midpoint, sorted.size]

  if values.length.odd?
    upper_half.delete(median)
  end

  calc_median(upper_half)
end

if ARGV.length != 1
  abort 'You have to pass the folder path.'
end

folder_path = ARGV.shift

ConsoleStyle::Functions.print_heading("Process Multi-Model Training Results 2024: #{folder_path} (v#{VERSION}) ...")
sub_folders = ConsoleStyle::Functions.execute_split_lines("ls #{folder_path}").reject { |names| SKIP_FILES_AND_FOLDERS.any? { |name| names.include?(name) } }

sub_folder_count = ConsoleStyle::Functions.style(sub_folders.size, ConsoleStyle::Functions::GREEN)
puts "Total number of models: #{sub_folder_count}"
puts "\n\n"

ConsoleStyle::Functions.print_heading("Generating sub-folders ...")
complete = ConsoleStyle::Functions.style('Done!', ConsoleStyle::Functions::GREEN)
failure = ConsoleStyle::Functions.style('Fail!', ConsoleStyle::Functions::RED)

# New Folder
save_folder_path = "#{folder_path}/corrections_combined_models/"
ConsoleStyle::Functions.print_heading("Create Save folder ...")
ConsoleStyle::Functions.execute("mkdir -p #{save_folder_path}")

def open_grouped_percentages_file(base_path)
  open_file = open_csv("#{base_path}/fine_grad/#{DATASET}/Base/#{GROUPED_PERCENTAGES_FILE}")

  if open_file.nil?
    open_file = open_csv("#{base_path}/grad_cam/#{DATASET}/Base/#{GROUPED_PERCENTAGES_FILE}")
  end

  if open_file.nil?
    open_file = open_csv("#{base_path}/fromgames/#{DATASET}/Base/#{GROUPED_PERCENTAGES_FILE}")
  end

  if open_file.nil?
    open_file = open_csv("#{base_path}/fromgames/#{DATASET_FROMGAMES}/Base/#{GROUPED_PERCENTAGES_FILE}")
  end

  open_file
end

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

# 0. Check data is okay
ConsoleStyle::Functions.print_heading("0. Check data is okay")
sub_folders.each { |sub_folder|
  puts("#{sub_folder}...")

  base_path = "#{folder_path}/#{sub_folder}/"

  base_path = "#{folder_path}/#{sub_folder}/"
  config, base_grad_cam, turning_only_base_grad_cam = generate_paths(base_path, sub_folder, DATASET)

  if config.nil? || base_grad_cam.nil? || turning_only_base_grad_cam.nil?
    config, base_grad_cam, turning_only_base_grad_cam = generate_paths(base_path, sub_folder, DATASET_FROMGAMES)
  end

  unless config
    puts "Config is missing! #{failure}"
    abort "Ending early!"
  end

  unless base_grad_cam
    puts "base_grad_cam is missing! #{failure}"
    abort "Ending early!"
  end

  # Get Best models
  files_list = list_files(base_path)

  unless files_list.find { |file| file.include?(BEST_MODEL_MATCH) }
    puts "best model is missing! #{failure}"
    abort "Ending early!"
  end

  unless files_list.find { |file| file.include?(BEST_AUTONOMY_MATCH) }
    puts "best autonomy model is missing! #{failure}"
    abort "Ending early!"
  end

  csv_file = open_grouped_percentages_file(base_path)
  unless csv_file
    puts "fine grad grouped percentages is missing! #{failure}"
    abort "Ending early!"
  end

  puts complete
}

# 1. Best Epochs
ConsoleStyle::Functions.print_heading("1. Best Epochs")
best_epochs_filepath = "#{save_folder_path}/best_epochs.csv"
selected_model_name = ""
# best_epoch = ""

csv_lines = sub_folders.map { |sub_folder|
  puts("#{sub_folder}...")

  base_path = "#{folder_path}/#{sub_folder}/"
  config = open_json("#{base_path}/#{sub_folder}_config.json")

  files_list = list_files(base_path)
  model_name = files_list.find { |file| file.include?(BEST_MODEL_MATCH) }

  selected_model_name = config["model_name"]
  best_epoch = model_name[/\d+/]

  _csv_line = "#{config["model_name"]},#{best_epoch}"
}

best_epochs_str = to_csv_str([BEST_EPOCH_HEADINGS] + csv_lines)
save_file(best_epochs_filepath, best_epochs_str, permission: 'w')
puts complete

# 2. Best Autonomous Epochs
ConsoleStyle::Functions.print_heading("2. Best Autonomous Epochs")
best_autonomy_epochs_filepath = "#{save_folder_path}/best_autonomy_epochs.csv"

csv_lines = sub_folders.map { |sub_folder|
  puts("#{sub_folder}...")

  base_path = "#{folder_path}/#{sub_folder}/"
  config = open_json("#{base_path}/#{sub_folder}_config.json")

  selected_model_name = config["model_name"]
  files_list = list_files(base_path)
  model_name = files_list.find { |file| file.include?(BEST_AUTONOMY_MATCH) }

  _csv_line = "#{selected_model_name},#{model_name[/\d+/]}"
}

best_epochs_str = to_csv_str([BEST_AUTONOMY_HEADINGS] + csv_lines)
save_file(best_autonomy_epochs_filepath, best_epochs_str, permission: 'w')
puts complete

# 3. Best epoch gradcam per group
ConsoleStyle::Functions.print_heading("3. Best epoch gradcam per group")
best_epoch_gradcam_per_group_filepath = "#{save_folder_path}/best_epoch_gradcam_per_group.csv"
heading = ""

csv_lines = sub_folders.map { |sub_folder|
  puts("#{sub_folder}...")

  base_path = "#{folder_path}/#{sub_folder}/"

  files_list = list_files(base_path)
  model_name = files_list.find { |file| file.include?(BEST_MODEL_MATCH) }

  best_epoch = model_name[/\d+/]
  csv_file = open_grouped_percentages_file(base_path)

  heading = csv_file.headers().join(DELIMETER)
  _csv_line = csv_file.find { |row| row["i"] == best_epoch }.to_s.gsub("\n", "")
}

grad_cam_str = to_csv_str([heading] + csv_lines)
save_file(best_epoch_gradcam_per_group_filepath, grad_cam_str, permission: 'w')
puts complete

# 4. Best autonomy gradcam per group
ConsoleStyle::Functions.print_heading("4. Best autonomy gradcam per group")
best_autonomy_gradcam_per_group_filepath = "#{save_folder_path}/best_autonomy_gradcam_per_group.csv"
heading = ""

csv_lines = sub_folders.map { |sub_folder|
  puts("#{sub_folder}...")

  base_path = "#{folder_path}/#{sub_folder}/"

  files_list = list_files(base_path)
  model_name = files_list.find { |file| file.include?(BEST_AUTONOMY_MATCH) }

  best_autonomy = model_name[/\d+/]
  csv_file = open_grouped_percentages_file(base_path)

  heading = csv_file.headers().join(DELIMETER)
  _csv_line = csv_file.find { |row| row["i"] == best_autonomy }.to_s.gsub("\n", "")
}

grad_cam_str = to_csv_str([heading] + csv_lines)
save_file(best_autonomy_gradcam_per_group_filepath, grad_cam_str, permission: 'w')
puts complete

# 5. Best Epoch Box and Whisker
ConsoleStyle::Functions.print_heading("5. Best Epoch Box and Whisker")
best_epoch_box_and_whisker_filepath = "#{save_folder_path}/best_epoch_box_and_whisker.csv"
best_epochs_combined_csv = open_csv(best_epochs_filepath)

min = best_epochs_combined_csv["Best Epoch"].map { |r| r.to_f }.min
q1 = calc_q1(best_epochs_combined_csv["Best Epoch"].map { |r| r.to_f })
median = calc_median(best_epochs_combined_csv["Best Epoch"].map { |r| r.to_f })
q3 = calc_q3(best_epochs_combined_csv["Best Epoch"].map { |r| r.to_f })
max = best_epochs_combined_csv["Best Epoch"].map { |r| r.to_f }.max

csv_str = "#{selected_model_name}#{DELIMETER}#{min}#{DELIMETER}#{q1}#{DELIMETER}#{median}#{DELIMETER}#{q3}#{DELIMETER}#{max}"

epoch_box_str = to_csv_str([BOX_AND_WHISKER_EPOCH_HEADINGS] + [csv_str])
save_file(best_epoch_box_and_whisker_filepath, epoch_box_str, permission: 'w')

puts complete

# 6. Best autonomy Epoch Box and Whisker
ConsoleStyle::Functions.print_heading("6. Best autonomy Epoch Box and Whisker")
best_autonomy_box_and_whisker_filepath = "#{save_folder_path}/best_autonomy_box_and_whisker.csv"
best_autonomy_combined_csv = open_csv(best_autonomy_epochs_filepath)

min = best_autonomy_combined_csv["Best Autonomy Epoch"].map { |r| r.to_f }.min
q1 = calc_q1(best_autonomy_combined_csv["Best Autonomy Epoch"].map { |r| r.to_f })
median = calc_median(best_autonomy_combined_csv["Best Autonomy Epoch"].map { |r| r.to_f })
q3 = calc_q3(best_autonomy_combined_csv["Best Autonomy Epoch"].map { |r| r.to_f })
max = best_autonomy_combined_csv["Best Autonomy Epoch"].map { |r| r.to_f }.max

csv_str = "#{selected_model_name}#{DELIMETER}#{min}#{DELIMETER}#{q1}#{DELIMETER}#{median}#{DELIMETER}#{q3}#{DELIMETER}#{max}"

autonomy_box_str = to_csv_str([BOX_AND_WHISKER_EPOCH_HEADINGS] + [csv_str])
save_file(best_autonomy_box_and_whisker_filepath, autonomy_box_str, permission: 'w')

puts complete

# 7. Best epoch gradcam box and whisker per group
ConsoleStyle::Functions.print_heading("7. Best epoch gradcam box and whisker per group")
best_epoch_gradcam_box_and_whisker_filepath = "#{save_folder_path}/best_epoch_gradcam_box_and_whisker.csv"
grad_cam_combined_csv = open_csv(best_epoch_gradcam_per_group_filepath)

labels = grad_cam_combined_csv.headers()[1,grad_cam_combined_csv.headers().size]
csv_lines = labels.map do |label|
  min = grad_cam_combined_csv[label].map { |r| r.to_f }.min
  q1 = calc_q1(grad_cam_combined_csv[label].map { |r| r.to_f })
  median = calc_median(grad_cam_combined_csv[label].map { |r| r.to_f })
  q3 = calc_q3(grad_cam_combined_csv[label].map { |r| r.to_f })
  max = grad_cam_combined_csv[label].map { |r| r.to_f }.max

  "#{label}#{DELIMETER}#{min}#{DELIMETER}#{q1}#{DELIMETER}#{median}#{DELIMETER}#{q3}#{DELIMETER}#{max}"
end

grad_cam_box_str = to_csv_str([BOX_AND_WHISKER_GRADCAM_HEADINGS] + csv_lines)
save_file(best_epoch_gradcam_box_and_whisker_filepath, grad_cam_box_str, permission: 'w')

puts complete

# 8. Best autonomy gradcam box and whisker per group
ConsoleStyle::Functions.print_heading("8. Best autonomy gradcam box and whisker per group")
best_autonomy_gradcam_box_and_whisker_filepath = "#{save_folder_path}/best_autonomy_gradcam_box_and_whisker.csv"
grad_cam_combined_csv = open_csv(best_autonomy_gradcam_per_group_filepath)

labels = grad_cam_combined_csv.headers()[1,grad_cam_combined_csv.headers().size]
csv_lines = labels.map do |label|
  min = grad_cam_combined_csv[label].map { |r| r.to_f }.min
  q1 = calc_q1(grad_cam_combined_csv[label].map { |r| r.to_f })
  median = calc_median(grad_cam_combined_csv[label].map { |r| r.to_f })
  q3 = calc_q3(grad_cam_combined_csv[label].map { |r| r.to_f })
  max = grad_cam_combined_csv[label].map { |r| r.to_f }.max

  "#{label}#{DELIMETER}#{min}#{DELIMETER}#{q1}#{DELIMETER}#{median}#{DELIMETER}#{q3}#{DELIMETER}#{max}"
end

grad_cam_box_str = to_csv_str([BOX_AND_WHISKER_GRADCAM_HEADINGS] + csv_lines)
save_file(best_autonomy_gradcam_box_and_whisker_filepath, grad_cam_box_str, permission: 'w')

# 9. Compute Training Average Graphs
def select_row(csv_file, i_value)
  csv_file.find { |row| row["i"] == i_value }
end

def append_row(group, selected_row, heading)
  return group if selected_row.nil?

  if group.nil?
    group = {}
  end

  heading.each do |group_name|
    if group[group_name].nil?
      group[group_name] = [ selected_row[group_name] ]
    else
      group[group_name] = group[group_name] + [ selected_row[group_name] ]
    end
  end

  # group + selected_row
  group
end

def avg(arr)
  arr.map(&:to_f).sum / Float(arr.size)
end

ConsoleStyle::Functions.print_heading("9. Compute Training Average Graphs")
heading = ""

# 0 1 5 10 b1 b2
raw_selected_results = {}

csv_lines = sub_folders.map { |sub_folder|
  puts("#{sub_folder}...")

  base_path = "#{folder_path}/#{sub_folder}/"
  csv_file = open_grouped_percentages_file(base_path)

  files_list = list_files(base_path)

  model_name = files_list.find { |file| file.include?(BEST_MODEL_MATCH) }
  best_epoch = model_name[/\d+/]

  model_name = files_list.find { |file| file.include?(BEST_AUTONOMY_MATCH) }
  best_autonomy = model_name[/\d+/]

  heading = (csv_file.headers - ['i']).uniq

  raw_selected_results["0"] = append_row(raw_selected_results["0"], select_row(csv_file, "0"), heading)
  raw_selected_results["1"] = append_row(raw_selected_results["1"], select_row(csv_file, "1"), heading)
  raw_selected_results["5"] = append_row(raw_selected_results["5"], select_row(csv_file, "5"), heading)
  raw_selected_results["10"] = append_row(raw_selected_results["10"], select_row(csv_file, "10"), heading)
  raw_selected_results["b1"] = append_row(raw_selected_results["b1"], select_row(csv_file, "#{best_epoch}"), heading)
  raw_selected_results["b2"] = append_row(raw_selected_results["b2"], select_row(csv_file, "#{best_autonomy}"), heading)
}

# csv_file["i"].size == csv_file["i"].uniq.size

heading = ["i"] + heading # Add it back

# Average
average_selected_results = {}

raw_selected_results.keys.each do |epoch_row|
  results = {}

  raw_selected_results[epoch_row].keys.each do |group_name|
    results[group_name] = avg(raw_selected_results[epoch_row][group_name])
  end

  average_selected_results[epoch_row] = results
end

output_str = "#{heading.join(',')}"

average_selected_results.keys.each do |epoch|
  output_str = "#{output_str}\n#{epoch},#{average_selected_results[epoch].values.join(',')}"
end

training_save_filepath = "#{save_folder_path}/training_average_graphs.csv"
save_file(training_save_filepath, output_str, permission: 'w')

# Min
min_selected_results = {}

raw_selected_results.keys.each do |epoch_row|
  results = {}

  raw_selected_results[epoch_row].keys.each do |group_name|
    results[group_name] = raw_selected_results[epoch_row][group_name].min
  end

  min_selected_results[epoch_row] = results
end

output_str = "#{heading.join(',')}"

min_selected_results.keys.each do |epoch|
  output_str = "#{output_str}\n#{epoch},#{min_selected_results[epoch].values.join(',')}"
end

training_save_filepath = "#{save_folder_path}/training_min_graphs.csv"
save_file(training_save_filepath, output_str, permission: 'w')

# Max
max_selected_results = {}

raw_selected_results.keys.each do |epoch_row|
  results = {}

  raw_selected_results[epoch_row].keys.each do |group_name|
    results[group_name] = raw_selected_results[epoch_row][group_name].max
  end

  max_selected_results[epoch_row] = results
end

output_str = "#{heading.join(',')}"

max_selected_results.keys.each do |epoch|
  output_str = "#{output_str}\n#{epoch},#{max_selected_results[epoch].values.join(',')}"
end

training_save_filepath = "#{save_folder_path}/training_max_graphs.csv"
save_file(training_save_filepath, output_str, permission: 'w')

puts complete
