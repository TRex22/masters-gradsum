require 'json'

# TODO:
# Camera
# gtBboxCityPersons
# blurred

BASE_JSON_PATH = '/data/data/cityscapes/data_raw/vehicle'
BASE_SAVE_PATH = '/data/data/cityscapes/data_raw/leftImg8bit'
BASE_FINE_SEG_PATH = '/data/data/cityscapes/data_raw/gtFine'
BASE_COARSE_SEG_PATH = '/data/data/cityscapes/data_raw/gtCoarse'

CSV_HEADER = [
  'json_file_path',
  'category',
  'city',
  'imagePath',
  'gpsHeading',
  'gpsLatitude',
  'gpsLongitude',
  'outsideTemperature',
  'Speed',
  'Steering',
  'gtFine',
  'gtCoarse',
  'Segmentation Path' # Combined
].join(',')

DATA_FOLDERS = [
  'test/berlin',
  'test/bielefeld',
  'test/bonn',
  'test/leverkusen',
  'test/mainz',
  'test/munich',
  'train/aachen',
  'train/bochum',
  'train/bremen',
  'train/cologne',
  'train/darmstadt',
  'train/dusseldorf',
  'train/erfurt',
  'train/hamburg',
  'train/hanover',
  'train/jena',
  'train/krefeld',
  'train/monchengladbach',
  'train/strasbourg',
  'train/stuttgart',
  'train/tubingen',
  'train/ulm',
  'train/weimar',
  'train/zurich',
  'train_extra/augsburg',
  'train_extra/bamberg',
  'train_extra/dortmund',
  'train_extra/duisburg',
  'train_extra/freiburg',
  'train_extra/heilbronn',
  'train_extra/konigswinter',
  'train_extra/mannheim',
  'train_extra/nuremberg',
  'train_extra/saarbrucken',
  'train_extra/troisdorf',
  'train_extra/wurzburg',
  'train_extra/bad-honnef',
  'train_extra/bayreuth',
  'train_extra/dresden',
  'train_extra/erlangen',
  'train_extra/heidelberg',
  'train_extra/karlsruhe',
  'train_extra/konstanz',
  'train_extra/muhlheim-ruhr',
  'train_extra/oberhausen',
  'train_extra/schweinfurt',
  'train_extra/wuppertal',
  'val/frankfurt',
  'val/lindau',
  'val/munster'
]

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

def file_exists?(path)
  !file_does_not_exist?(path)
end

def file_does_not_exist?(path)
  `du #{path}`.empty?
end

def category(base_path)
  if base_path.include?('train_extra')
    'train_extra'
  elsif base_path.include?('train')
    'train'
  elsif base_path.include?('val')
    'val'
  elsif base_path.include?('test')
    'test'
  else
    ''
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

def generate_fine_seg_map_file_name(file_path)
  old_suffix = '_leftImg8bit.png'
  new_suffix = '_gtFine_labelIds.png' # TODO: Make this configurable
  # new_suffix = '_gtFine_instanceIds.png' # TODO: Make this configurable

  file_path.gsub(old_suffix, new_suffix)
end

def generate_coarse_seg_map_file_name(file_path)
  old_suffix = '_leftImg8bit.png'
  new_suffix = '_gtCoarse_labelIds.png' # TODO: Make this configurable
  # new_suffix = '_gtCoarse_instanceIds.png' # TODO: Make this configurable

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

def parse_json_data(json_folder_path)
  [CSV_HEADER] + full_paths(json_folder_path).map do |full_path|
    file_paths = list_files(full_path)
    read_files(json_folder_path, file_paths)
  end
end

def parse_segmentation_data(lines, fine_base_path, coarse_base_path)
  lines.each_with_index do |line, i|
    next if i == 0

    img_path = line.split(',')[3]
    fine_path = generate_fine_seg_map_file_name(img_path)

    if file_exists?("#{BASE_FINE_SEG_PATH}/#{fine_path}")
      fine_path = "/gtFine#{fine_path}"
      lines[i] = "#{lines[i]},#{fine_path},,#{fine_path}"
    else
      coarse_path = generate_coarse_seg_map_file_name(img_path)
      coarse_path = "/gtCoarse#{coarse_path}"
      lines[i] = "#{lines[i]},,#{coarse_path},#{coarse_path}"
    end
  end

  lines
end

def csv_path(base_path)
  "#{base_path}/telemetry.csv"
end

def parse_and_save_json_data(base_path, base_save_path, fine_base_path, coarse_base_path)
  lines = parse_json_data(base_path).flatten
  lines = parse_segmentation_data(lines, fine_base_path, coarse_base_path)

  File.open(csv_path(base_save_path), "w+") do |f|
    lines.each do |line| f.puts(line) end
  end

  puts csv_path(base_save_path)
end

parse_and_save_json_data(BASE_JSON_PATH, BASE_SAVE_PATH, BASE_FINE_SEG_PATH, BASE_COARSE_SEG_PATH)
