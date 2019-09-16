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

import pickle
from mmdetection.mmdet.apis import init_detector, inference_detector, show_result
from tqdm import tqdm_notebook
from glob import glob
import matplotlib.pyplot as plt

# %matplotlib inline

# +
# CONSTANTS
prediction_file_path = 'preds.txt'
ground_truth_file_path = 'ground-truth.txt'
# anns = pickle.load(open('../../data/t3-data/splits/all/test.pkl', 'rb'))
anns = pickle.load(open('../../data/t3-data/arac_splits/test.pkl', 'rb'))

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/old_workdir/faster_rcnn_r50_fpn_1x_visdrone/epoch_1.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_mix/epoch_4.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/old_workdir/faster_rcnn_r50_fpn_1x_visdrone_pretrained/epoch_14.pth'

# config_file = '../../src/configs/libra_rcnn/libra_retinanet_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/libra_retinanet_r50_fpn_1x/epoch_2.pth'

# config_file = '../configs/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x.py'
# checkpoint_file = '../../models/work_dirs/ga_retinanet_x101_32x4d_fpn_1x/latest.pth'


# config_file = '../mmdetection/configs/retinanet_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/retinanet_r50_fpn_1x_visdrone/epoch_2.pth'

config_file = '../../src/configs/cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = '../../models/work_dirs/cascade_rcnn_r50_fpn_1x-arac/latest.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r101_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/old_workdir/faster_rcnn_r101_fpn_1x/epoch_15.pth'

# config_file = '../../models/google-cloud-models/cascade-t3-vis/config.py'
# checkpoint_file = '../../models/google-cloud-models/cascade-t3-vis/epoch_1.pth'




# +
results = []
model = init_detector(config_file, checkpoint_file, device='cuda:0')


img_list = ['../../data/t3-data/' + ann['filename'] for ann in anns]

for img in tqdm_notebook(img_list):
    results.append(inference_detector(model, img))
# -


# PREDS.TXT
thres = [0.7]
with open(prediction_file_path, 'w+') as f:
    for img_path, img_preds in zip(img_list, results):
        line = '/'.join(img_path.split('/')[-2:])
        for i, preds in enumerate(img_preds[:2]):
            for bbox in filter(lambda x: x[-1] >= thres[i], preds):
                line += ",{},{},{},{},{}".format(*bbox[:-1], i+1)
        f.write(line)
        f.write('\n')


# GROUND TRUTH
with open(ground_truth_file_path, 'w+') as f:
    for img in anns:
        line = '/'.join(img['filename'].split('/')[-2:])
        for i, bbox in enumerate(img['ann']['bboxes']):
            line += ",{},{},{},{},{}".format(*bbox, img['ann']['labels'][i])
        f.write(line)
        f.write('\n')

# eval kodu: https://github.com/GlobalMaksimum/t3/blob/master/src/eval/evaluate.py

! /home/deep/miniconda3/envs/open-mmlab/bin/python ../eval/evaluate.py ground-truth.txt preds.txt

! /home/deep/miniconda3/envs/open-mmlab/bin/python ../eval/t3_evaluate.py ground-truth.txt preds.txt


