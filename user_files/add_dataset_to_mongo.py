import sys
import os
import functools
import pymongo
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

folder_path = sys.argv[2]
folder_list = os.listdir(folder_path)
folder_list.sort()
if startFolder == -1:
  startFolder = 0
  endFolder = len(folder_list)
print("Start folder: %d and end folder: %d" % (startFolder, endFolder))
for i in range(startFolder-1, endFolder):
  absoluteSubFolder = folder_path + folder_list[i] + '/'
  tmpFileList = os.listdir(absoluteSubFolder)
  tmpFileList = sorted(tmpFileList, key=functools.cmp_to_key(file_cmp))
  for img_name in tmpFileList:
    print(img_name)
    absolute_img_path = absoluteSubFolder + img_name
    result = inference_detector(model, absolute_img_path)
    print('--------------------------------------')
    endResult = show_result(absolute_img_path, result, model.CLASSES, out_file="result.jpg")
    print(endResult)
    print('--------------------------------------')    
    os.remove("result.jpg")

