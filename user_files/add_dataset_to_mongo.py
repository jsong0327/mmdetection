import sys
import os
import functools
import csv
from pymongo import MongoClient
from mmdet.apis import init_detector, inference_detector, show_result



def file_cmp(a, b):
  mod_a = a.split('_')
  mod_b = b.split('_')
  if int(mod_a[1]) <= int(mod_b[1]):
    return -1
  else:
    return 1


def isInt(s):
    try: 
      int(s)
      return True
    except ValueError:
      return False


def get_tsv_data(tsv_folder_path, video_number):
  tsv_file_path = str(video_number)
  while len(tsv_file_path) < 5:
    tsv_file_path = '0' + tsv_file_path
  tsv_file_path = tsv_folder_path + tsv_file_path + '.tsv'
  data = []
  with open(tsv_file_path) as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    i = 0
    for line in tsv_reader:
      if i == 0:
        pass
      else:
        myLine = {"startFrame": int(line[0]), "endFrame": int(line[2]), "startSecond": float(line[1]), "endSecond": float(line[3])}
        data.append(myLine)
      i += 1
  return data



client = MongoClient('127.0.0.1', 27017)
db = client.testdb
col = db.mmdetection
col.remove({})

if (len(sys.argv) != 3 \
    and (len(sys.argv) != 5 or (len(sys.argv) == 5 and sys.argv[3] != '-range'))) \
    or sys.argv[1] != '-folder':
  print("Write -folder FOLDERPATH -range STARTFOLDERNUMBER-ENDFOLDERNUMBER (range is optional)\n\
  ex) add_dataset_to_mongo.py -folder /dataset/ -range 1-100")
  exit(0)

if sys.argv[2][-1] != '/':
  sys.argv[2] = sys.argv[2] + '/'

startFolder = -1
endFolder = -1
if len(sys.argv) == 5:
  folder_range = sys.argv[4].split('-')
  if len(folder_range) != 2 or not isInt(folder_range[0]) or not isInt(folder_range[1]):
    print("Write range appropriately")
  startFolder = int(folder_range[0])
  endFolder = int(folder_range[1])



config_file = '../configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
checkpoint_file = '../checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

folder_path = sys.argv[2] + 'keyframes/'
tsv_path = sys.argv[2] + 'tsv/'
folder_list = os.listdir(folder_path)
folder_list.sort()
if startFolder == -1:
  startFolder = 0
  endFolder = len(folder_list)
print("Start folder: %d and end folder: %d" % (startFolder, endFolder))
for i in range(startFolder-1, endFolder):
  video_number = i + 1
  absoluteSubFolder = folder_path + folder_list[i] + '/'
  tmpFileList = os.listdir(absoluteSubFolder)
  tmpFileList = sorted(tmpFileList, key=functools.cmp_to_key(file_cmp))
  tsv_data = get_tsv_data(tsv_path, video_number)
  for img_name in tmpFileList:
    print(img_name)
    absolute_img_path = absoluteSubFolder + img_name
    result = inference_detector(model, absolute_img_path)
    frame_number = int(img_name.split('_')[1])
    print('--------------------------------------')
    endResult = show_result(absolute_img_path, result, model.CLASSES, out_file="result.jpg")
    print(endResult)
    print(tsv_data[frame_number - 1])
    # Remove position
    for objects in endResult:
      objects.pop('leftTop')
      objects.pop('rightBottom')

    # Add to database
    current_frame_tsv = tsv_data[frame_number-1]
    col.update(
      {"video": video_number, 
      "startFrame": current_frame_tsv["startFrame"]},
      {"video": video_number, 
      "startFrame": current_frame_tsv["startFrame"],
      "endFrame": current_frame_tsv["endFrame"],
      "startSecond": current_frame_tsv["startSecond"],
      "endSecond": current_frame_tsv["endSecond"],
      "object": endResult},
      upsert=True
    )
    print('--------------------------------------')    
    os.remove("result.jpg")

