import sys
from mmdet.apis import init_detector, inference_detector, show_result

if len(sys.argv) != 3 or (sys.argv[1] != '-file' and sys.argv[1] != '-folder'):
  print("Write -file FILEPATH for one image and -folder FOLDERPATH for folder\nex) test_image.py -folder data/coco")
  exit(0)

config_file = 'configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
checkpoint_file = 'checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

if (sys.argv[1] == '-file'):
  # test a single image and show the results
  img = sys.argv[2]  # or img = mmcv.imread(img), which will only load it once
  result = inference_detector(model, img)
  show_result(img, result, model.CLASSES, out_file='result.jpg')

elif (sys.argv[1] == '-folder'):
  exit(0)
  # test a list of images and write the results to image files
  # imgs = ['/val2017/000000000285.jpg', '/val2017/000000000632.jpg']
  # for i, result in enumerate(inference_detector(model, imgs)):
  #     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
