require 'progress_bar'

path = '/data/data/carla/imitation/processed/carlaimitation_256_60_3_old'
new_path = '/data/data/carla/imitation/processed/carlaimitation_256_60_3'
folders = ['SeqTrain', 'SeqVal']

files = `ls #{path}`;
files.size

files = files.split("\n");
files.size

bar = ProgressBar.new(files.size)
files.each do |filename|
  folder = ''

  if filename.include?('SeqTrain')
    folder = 'SeqTrain'
  else
    folder = 'SeqVal'
  end

  old_full_path = "#{path}/#{filename}"
  old_full_path = old_full_path.sub(folder, folder+'\\')

  new_filename = filename.sub(folder + '\\', '')
  new_full_path = "#{new_path}/#{folder}/#{new_filename}"

  `mv -f #{old_full_path} #{new_full_path}`

  bar.increment!
end;
