# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3.6 (TensorFlow GPU)
#     language: python
#     name: tensorflow_gpuenv
# ---

# +
import sys
sys.path.insert(0,'..')

from mmdetection.mmdet.apis import init_detector, inference_detector, show_result

# CONSTANTS
config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '../mmdetection/work_dirs/faster_rcnn_r50_fpn_1x/epoch_12.pth'
img = '../p.jpg'  # or img = mmcv.imread(img), which will only load it once
out_file_name = 'result_test.png'
class_names = ['person', 'vehicle'] 
auto_class = False # turn this on if you want to get class names directly from model and config


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

if auto_class:
    class_names = model.CLASSES

# test a single image and show the results
result = inference_detector(model, img)

# remove `out_file` if you want it to show images on local
# show_result(img, result, model.CLASSES, score_thr=0.3, out_file=out_file_name)
show_result(img, result, model.CLASSES, score_thr=0.3, out_file=out_file_name)
# -


