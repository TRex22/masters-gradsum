#!/bin/ruby

require 'console-style'
require 'csv'
# require 'json'
require 'oj'
require 'pry'
require 'shellwords'
require 'fileutils'
require 'find'

cityscapes_labels = {
  "unlabeled"            =>  0,
  "ego vehicle"          =>  1,
  "rectification border" =>  2,
  "out of roi"           =>  3,
  "static"               =>  4,
  "dynamic"              =>  5,
  "ground"               =>  6,
  "road"                 =>  7,
  "sidewalk"             =>  8,
  "parking"              =>  9,
  "rail track"           => 10,
  "building"             => 11,
  "wall"                 => 12,
  "fence"                => 13,
  "guard rail"           => 14,
  "bridge"               => 15,
  "tunnel"               => 16,
  "pole"                 => 17,
  "polegroup"            => 18,
  "traffic light"        => 19,
  "traffic sign"         => 20,
  "vegetation"           => 21,
  "terrain"              => 22,
  "sky"                  => 23,
  "person"               => 24,
  "rider"                => 25,
  "car"                  => 26,
  "truck"                => 27,
  "bus"                  => 28,
  "caravan"              => 29,
  "trailer"              => 30,
  "train"                => 31,
  "motorcycle"           => 32,
  "bicycle"              => 33,
  "license plate"        => -1
}

cityscapes_colours = {
  "unlabeled"            => [  0,  0,  0],
  "ego vehicle"          => [  0,  0,  0],
  "rectification border" => [  0,  0,  0],
  "out of roi"           => [  0,  0,  0],
  "static"               => [  0,  0,  0],
  "dynamic"              => [111, 74,  0],
  "ground"               => [ 81,  0, 81],
  "road"                 => [128, 64,128],
  "sidewalk"             => [244, 35,232],
  "parking"              => [250,170,160],
  "rail track"           => [230,150,140],
  "building"             => [ 70, 70, 70],
  "wall"                 => [102,102,156],
  "fence"                => [190,153,153],
  "guard rail"           => [180,165,180],
  "bridge"               => [150,100,100],
  "tunnel"               => [150,120, 90],
  "pole"                 => [153,153,153],
  "polegroup"            => [153,153,153],
  "traffic light"        => [250,170, 30],
  "traffic sign"         => [220,220,  0],
  "vegetation"           => [107,142, 35],
  "terrain"              => [152,251,152],
  "sky"                  => [ 70,130,180],
  "person"               => [220, 20, 60],
  "rider"                => [255,  0,  0],
  "car"                  => [  0,  0,142],
  "truck"                => [  0,  0, 70],
  "bus"                  => [  0, 60,100],
  "caravan"              => [  0,  0, 90],
  "trailer"              => [  0,  0,110],
  "train"                => [  0, 80,100],
  "motorcycle"           => [  0,  0,230],
  "bicycle"              => [119, 11, 32],
  "license plate"        => [  0,  0,142]
}

cityscapes_groups = {
  "flat" => ["road", "sidewalk", "parking", "rail track"],
  "human" => ["person", "rider"],
  "vehicle" => ["car", "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle" ],
  "construction" => ["building", "wall", "fence", "guard rail", "bridge", "tunnel"],
  "object" => ["pole", "polegroup", "traffic sign", "traffic light"],
  "nature" => ["vegetation", "terrain"],
  "sky" => ["sky"],
  "void" => ["ground", "dynamic", "static"], # , "ego vehicle", "rectification border", "out of roi"
  # "unlabeled" => ["unlabeled"] # Added by me
}

def list_paths_to_file(base_path, filename)
  files = []

  # Dir.chdir(base_path) do
  #   Dir.glob("**/#{filename}") do |file|
  #     files << file
  #   end
  # end

  # Find.find(base_path) do |path|
  #   if File.basename(path).downcase.include?(filename)
  #     files << path
  #   end
  # end

  Find.find(base_path) do |path|
    if File.basename(path) == filename
      files << path
    end
  end

  files
end

def open_csv_file(filepath)
  puts "Open: #{filepath}"

  # csv_file = File.open(filepath)
  CSV.read(filepath, headers: true)
end

def generate_grouped_percentages(result_count_csv, total_count_csv, save_path, cityscapes_groups)
  result_csv = CSV.open(save_path, "wb")
  headers = ["i"] + cityscapes_groups.keys
  result_csv << headers

  result_count_csv.each_with_index do |result_row, idex|
    total_count_row = total_count_csv[idex]
    grouped_row_percentages = [result_row["i"]]

    headers.each do |group_name|
      next if group_name == "i"

      result_group_sum = 0.0
      total_group_sum = 0.0

      cityscapes_groups[group_name].each do |label_name|
        result_group_sum += result_row[label_name].to_f
        total_group_sum += total_count_row[label_name].to_f
      end

      grouped_row_percentages << (result_group_sum/total_group_sum) * 100
    end

    result_csv << grouped_row_percentages
  end

  result_csv.close
end

def generate_grouped_counts(count_csv, save_path, cityscapes_groups)
  result_csv = CSV.open(save_path, "wb")
  headers = ["i"] + cityscapes_groups.keys
  result_csv << headers

  count_csv.each_with_index do |result_row, idex|
    grouped_row_counts = [result_row["i"]]

    headers.each do |group_name|
      next if group_name == "i"

      result_group_sum = 0.0

      cityscapes_groups[group_name].each do |label_name|
        result_group_sum += result_row[label_name].to_f
      end

      grouped_row_counts << result_group_sum
    end

    result_csv << grouped_row_counts
  end

  result_csv.close
end

# base_path = '/data/trained_models'
base_path = ARGV.shift

filename = 'total_threshold_counts.csv'
# filename = 'total_edge_counts.csv'
# filename = 'total_segmentation_summary.csv'

puts "Opening #{base_path}..."

count_filepaths = list_paths_to_file(base_path, filename)
puts "#{count_filepaths.size} folders to convert ..."

count_filepaths.each do |count_filepath|
  count_csv = open_csv_file(count_filepath)

  segementation_summary_path = count_filepath.gsub("total_threshold_counts.csv", "total_segmentation_summary.csv")
  segmentation_csv = open_csv_file(segementation_summary_path)

  edge_count_path = count_filepath.gsub("total_threshold_counts.csv", "total_edge_counts.csv")
  edge_count_csv = open_csv_file(edge_count_path)

  total_result_count_save_path = count_filepath.gsub("total_threshold_counts.csv", "total_result_grouped_percentages.csv")
  generate_grouped_percentages(count_csv, segmentation_csv, total_result_count_save_path, cityscapes_groups)

  total_edge_count_save_path = count_filepath.gsub("total_threshold_counts.csv", "total_edge_grouped_percentages.csv")
  generate_grouped_percentages(edge_count_csv, segmentation_csv, total_edge_count_save_path, cityscapes_groups)

  total_segmentation_group_counts_path = count_filepath.gsub("total_threshold_counts.csv", "total_segmentation_group_counts.csv")
  generate_grouped_counts(segmentation_csv, total_segmentation_group_counts_path, cityscapes_groups)
end

puts "Done."
