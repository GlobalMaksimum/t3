# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3.7 MMDetect
#     language: python
#     name: mmdetect
# ---

# +
import sys
sys.path.insert(0,'..')

from mmdetection.mmdet.apis import init_detector, inference_detector, show_result_pyplot

# CONSTANTS
config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '../../models/work_dirs/faster_rcnn_r50_fpn_grid-yaya-t3-subset-visdrone/epoch_7.pth'

img = '../../data/ktr-test/test-grid/B23072019_V1_K1/frame23-02.jpg'

# or img = mmcv.imread(img), which will only load it once
out_file_name = 'result.png'
class_names = ['insan', 'arac'] 
auto_class = True # turn this on if you want to get class names directly from model and config

device = 'cuda:0'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device=device)
# -
if auto_class:
    class_names = model.CLASSES
# test a single image and show the results
result = inference_detector(model, img)
result

# remove `out_file` if you want it to show images on local
# show_result(img, result, model.CLASSES, score_thr=0.3, out_file=out_file_name)
show_result_pyplot(img, result, class_names, score_thr=0.7)
#show_result(img, result, class_names, score_thr=0.005)

# +
# from PIL import Image

# Image.open('../../data/ktr-test/test/B23072019_V1_K1/frame2200.jpg')
# -



