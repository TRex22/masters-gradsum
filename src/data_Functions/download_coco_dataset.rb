# https://cocodataset.org/#download

# 2014-2016 splits of data can be ignored

# Look at the 2017 data only


# # Images
# http://images.cocodataset.org/zips/train2014.zip


require 'pulse/downloader'
save_path = "/mnt/data/home/jason/data/coco"

# <div class="columnDownloads">
#   <h1>Images</h1>
#   <p class="fontSmall">
#     <a href="http://images.cocodataset.org/zips/train2014.zip">2014 Train images [83K/13GB]</a><br>
#     <a href="http://images.cocodataset.org/zips/val2014.zip">2014 Val images [41K/6GB]</a><br>
#     <a href="http://images.cocodataset.org/zips/test2014.zip">2014 Test images [41K/6GB]</a><br>
#     <a href="http://images.cocodataset.org/zips/test2015.zip">2015 Test images [81K/12GB]</a><br>
#     <a href="http://images.cocodataset.org/zips/train2017.zip">2017 Train images [118K/18GB]</a><br>
#     <a href="http://images.cocodataset.org/zips/val2017.zip">2017 Val images [5K/1GB]</a><br>
#     <a href="http://images.cocodataset.org/zips/test2017.zip">2017 Test images [41K/6GB]</a><br>
#     <a href="http://images.cocodataset.org/zips/unlabeled2017.zip">2017 Unlabeled images [123K/19GB]</a><br>
#   </p>
# </div>

# <div class="columnDownloads">
#   <h1>Annotations</h1>
#   <p class="fontSmall">
#     <a href="http://images.cocodataset.org/annotations/annotations_trainval2014.zip">2014 Train/Val annotations [241MB]</a><br>
#     <a href="http://images.cocodataset.org/annotations/image_info_test2014.zip">2014 Testing Image info [1MB]</a><br>
#     <a href="http://images.cocodataset.org/annotations/image_info_test2015.zip">2015 Testing Image info [2MB]</a><br>
#     <a href="http://images.cocodataset.org/annotations/annotations_trainval2017.zip">2017 Train/Val annotations [241MB]</a><br>
#     <a href="http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip">2017 Stuff Train/Val annotations [1.1GB]</a><br>
#     <a href="http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip">2017 Panoptic Train/Val annotations [821MB]</a><br>
#     <a href="http://images.cocodataset.org/annotations/image_info_test2017.zip">2017 Testing Image info [1MB]</a><br>
#     <a href="http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip">2017 Unlabeled Image info [4MB]</a><br>
#   </p>
# </div>

# url = 'https://cocodataset.org/#download'
urls = [
  'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
  'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',
  'http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip',
  'http://images.cocodataset.org/annotations/image_info_test2017.zip',
  'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip',
  'http://images.cocodataset.org/zips/train2017.zip',
  'http://images.cocodataset.org/zips/val2017.zip',
  'http://images.cocodataset.org/zips/test2017.zip',
  'http://images.cocodataset.org/zips/unlabeled2017.zip'
]

urls.map do |url|
  puts url
  client = Pulse::Downloader::Client
    .new(
      url: url,
      file_type: 'zip',
      verify_ssl: false,
      report_time: true,
      save_data: true,
      save_path: save_path,
      drop_exitsing_files_in_path: false,
      save_and_dont_return: true
    )
  client.call!
end


# Python
# import sys
# sys.path.insert(1, './data_Functions/')

# import torchvision
# import data_functions

# image_path = '/data/data/coco/data_raw/images/'
# annotation_path = '/data/data/coco/data_raw/annotations.json'

# data_functions.check_and_create_dir(image_path)
# data_functions.touch_file(annotation_path)
# # data_functions.check_and_create_dir(annotation_path)

# # CocoDetection(root, annFile, transform, …)

# # imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')

# # No transforms
# imagenet_data = torchvision.datasets.CocoDetection(image_path, annotation_path)
