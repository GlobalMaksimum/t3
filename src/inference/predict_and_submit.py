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

# +
# CONSTANTS
prediction_file_path = 'preds.txt'
ground_truth_file_path = 'ground-truth.txt'
anns = pickle.load(open('../../data/test.pkl', 'rb'))

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/old_workdir/faster_rcnn_r50_fpn_1x_visdrone/epoch_1.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_mix/epoch_4.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_visdrone_pretrained/epoch_14.pth'

# config_file = '../mmdetection/configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/libra_faster_rcnn_r50_fpn_1x_visdrone/epoch_2.pth'

# config_file = '../mmdetection/configs/retinanet_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/retinanet_r50_fpn_1x_visdrone/epoch_2.pth'

config_file = '../mmdetection/configs/cascade_rcnn_x101_32x4d_fpn_1x.py'
checkpoint_file = '../mmdetection/work_dirs/old_workdir/cascade_rcnn_x101_32x4d_fpn_1x/epoch_14.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r101_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/faster_rcnn_r101_fpn_1x/epoch_15.pth'

img_list = ['../../data/t3-data/merged_veriler/' + ann['filename'] for ann in anns]
# img_list = ['../../data/' + ann['filename'] for ann in anns]


# +
results = []
model = init_detector(config_file, checkpoint_file, device='cuda:0')


for img in tqdm_notebook(img_list):
#     if 'T190619_V1_K1/frame15748.jpg' not in img:
#         continue
    # inference on single image
    results.append(inference_detector(model, img))
    
# -


# PREDS.TXT
thres = 0.8
with open(prediction_file_path, 'w+') as f:
    for img_path, img_preds in zip(img_list, results):
        line = '/'.join(img_path.split('/')[-2:])
        for i, preds in enumerate(img_preds[:2]):
            for bbox in filter(lambda x: x[-1] >= thres, preds):
                line += ",{},{},{},{},{}".format(*bbox[:-1], i)
        f.write(line)
        f.write('\n')


# GROUND TRUTH
with open(ground_truth_file_path, 'w+') as f:
    for img in anns:
        line = img['filename']
        for i, bbox in enumerate(img['ann']['bboxes']):
            line += ",{},{},{},{},{}".format(*bbox, img['ann']['labels'][i] - 1)
        f.write(line)
        f.write('\n')


