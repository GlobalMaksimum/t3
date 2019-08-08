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

import pickle
from mmdetection.mmdet.apis import init_detector, inference_detector, show_result
from tqdm import tqdm_notebook
# -

# CONSTANTS
prediction_file_path = 'preds.txt'
ground_truth_file_path = 'ground-truth.txt'
anns = pickle.load(open('../../data/t3-data/gonderilecek_veriler/test.pkl', 'rb'))
config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '../mmdetection/work_dirs/faster_rcnn_r50_fpn_1x/epoch_12.pth'
img_list = ['../../data/t3-data/gonderilecek_veriler/' + ann['filename'] for ann in anns]


# +
results = []
model = init_detector(config_file, checkpoint_file, device='cuda:0')


for img in tqdm_notebook(img_list):
    # inference on single image
    results.append(inference_detector(model, img))
    
# -


# PREDS.TXT
with open(prediction_file_path, 'w+') as f:
    for img_path, img_preds in zip(img_list, results):
        line = ' ' + '/'.join(img_path.split('/')[-2:])
        if len(img_preds[0]) > 0:
            for bbox in img_preds[0]:
                # TODO fix yaya, arac 0 and 1 issue
                line += ",{},{},{},{},{}".format(*bbox[:-1], 1)
        elif len(img_preds[1]) > 0:
            for bbox in img_preds[1]:
                line += ",{},{},{},{},{}".format(*bbox[:-1], 0)
        f.write(line)
        f.write('\n')


# GROUND TRUTH
with open(ground_truth_file_path, 'w+') as f:
    for img in anns:
        line = ' ' + img['filename']
        for i, bbox in enumerate(img['ann']['bboxes']):
            line += ",{},{},{},{},{}".format(*bbox, img['ann']['labels'][i])
        f.write(line)
