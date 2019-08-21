import sys
import os
import pymongo
from mmdet.apis import init_detector, inference_detector, show_result

if len(sys.argv) != 3 or sys.argv[1] != '-folder':
  print("Write -folder FOLDERPATH for folder\nex)add_dataset_to_mongo.py -folder /dataset/")
  exit(0)

if sys.argv[2][-1] != '/':
  sys.argv[2] = sys.argv[2] + '/'

config_file = '../configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
checkpoint_file = '../checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

folder_path = sys.argv[2]
folder_list = os.listdir(folder_path)
for subFolder in folder_list:
  absoluteSubFolder = folder_path + subFolder + '/'
  tmpFileList = os.listdir(absoluteSubFolder)
  for img_name in tmpFileList:
    absolute_img_path = absoluteSubFolder + img_name
    result = inference_detector(model, absolute_img_path)
    print('--------------------------------------')
    endResult = show_result(absolute_img_path, result, model.CLASSES, out_file="result.jpg")
    print(endResult)
    print('--------------------------------------')    
    os.remove("result.jpg")

