% Taken from: https://download.visinf.tu-darmstadt.de/data/from_games/
mapping = load('./fromgames_mapping.mat', 'cityscapesMap', 'camvidMap', 'classes');

% Save the colour map
% csvwrite('~/Downloads/fromgames_to_cityscapes_colour_map.csv', mapping.cityscapesMap)

label_files = dir('/mnt/excelsior/data/fromgames/labels_original/*.png');

for i = 1:length(label_files)
  filename = label_files(i).name;
  filepath = sprintf('/mnt/excelsior/data/fromgames/labels_original/%s', filename);
  label = imread(filepath);

  save_filepath = sprintf('/mnt/excelsior/data/fromgames/labels/%s', filename);
  imwrite(label, mapping.cityscapesMap, save_filepath, 'png', 'Compression', 'none');
end
