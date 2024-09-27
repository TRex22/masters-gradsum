require 'pulse/downloader'
save_path = "/mnt/data/home/jason/data/fromgames"

client = Pulse::Downloader::Client.new(url: 'https://download.visinf.tu-darmstadt.de/data/fromgames/', file_type: 'zip', verify_ssl: false, report_time: true, save_data: true, save_path: save_path, drop_exitsing_files_in_path: true, save_and_dont_return: true)
client.call!
