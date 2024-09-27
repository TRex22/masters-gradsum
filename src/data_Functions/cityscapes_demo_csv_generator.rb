require 'json'

# TODO:
# Camera
# gtFine
# gtCoarse
# gtBboxCityPersons
# blurred

CSV_HEADER = [
  'imagePath',
  'videoSequence'
].join(',')

DATA_FOLDERS = [
  'demoVideo/stuttgart_00',
  'demoVideo/stuttgart_01',
  'demoVideo/stuttgart_02'
]

DELIMITER = ','

def generate_path(base_path, folder_path)
  "#{base_path}/#{folder_path}"
end

def full_paths(base_path)
  DATA_FOLDERS.map do |folder_path|
    generate_path(base_path, folder_path)
  end
end

def list_files(full_path)
  `ls #{full_path}`.split("\n").map do |file_name|
    generate_path(full_path, file_name)
  end
end

def category(base_path)
  if base_path.include?('demoVideo')
    'demo_video'
  end
end

def city(base_path)
  DATA_FOLDERS.select { |folder|
    base_path.include?(folder)
  }.last
    .gsub('train_extra/', '')
    .gsub('train/', '')
    .gsub('test/', '')
    .gsub('val/', '')
    .gsub('demoVideo/', '')
    .capitalize
end

def generate_image_file_name(file_path)
  old_suffix = '_vehicle.json'
  new_suffix = '_leftImg8bit.png' # TODO: Make this configurable

  file_path.gsub(old_suffix, new_suffix)
end

def relative(path, base_path)
  path.gsub(base_path, '')
end

def convert_json_to_csv(base_path, file_path, json_obj)
  [
    relative(file_path, base_path),
    category(file_path),
    city(file_path),
    relative(generate_image_file_name(file_path), base_path),
    json_obj["gpsHeading"],
    json_obj["gpsLatitude"],
    json_obj["gpsLongitude"],
    json_obj["outsideTemperature"],
    json_obj["speed"],
    json_obj["yawRate"]
  ].join(',')
end

def read_files(base_path, file_paths)
  file_paths.map do |file_path|
    raw_file = File.read(file_path)
    json_obj = JSON.parse(raw_file)

    convert_json_to_csv(base_path, file_path, json_obj)
  end
end

def fetch_image_data(base_path)
  full_paths(base_path).map { |full_path|
    list_files(full_path)
  }.flatten.compact.map { |full_path|
    video_sequence = DATA_FOLDERS.find { |df| full_path.downcase.include?(df.downcase) }
    "#{full_path}#{DELIMITER}#{video_sequence}"
  }
end

def csv_path(base_path)
  "#{base_path}/telemetry_demo_video.csv"
end

def parse_and_save_json_data(base_path)
  lines = fetch_image_data(base_path)

  File.open(csv_path(base_path), "w+") do |f|
    ([CSV_HEADER] + lines).each do |line| f.puts(line) end
  end

  puts csv_path(base_path)
end

base_path = '/data/data/cityscapes/data_raw/leftImg8bit/'
parse_and_save_json_data(base_path)
