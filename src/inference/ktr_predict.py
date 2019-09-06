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

import matplotlib.pyplot as plt

# %matplotlib inline

# +
# CONSTANTS
prediction_file_path = 'preds.txt'
ground_truth_file_path = 'ground-truth.txt'

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/old_workdir/faster_rcnn_r50_fpn_1x_visdrone/epoch_1.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_mix/epoch_4.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/old_workdir/faster_rcnn_r50_fpn_1x_visdrone_pretrained/epoch_14.pth'

config_file = '../../src/configs/libra_rcnn/libra_retinanet_r50_fpn_1x.py'
checkpoint_file = '../mmdetection/work_dirs/libra_retinanet_r50_fpn_1x/epoch_2.pth'

# config_file = '../mmdetection/configs/retinanet_r50_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/retinanet_r50_fpn_1x_visdrone/epoch_2.pth'

# config_file = '../../src/configs/cascade_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../../models/work_dirs/cascade_rcnn_r50_fpn_1x-all/latest.pth'

# config_file = '../mmdetection/configs/faster_rcnn_r101_fpn_1x.py'
# checkpoint_file = '../mmdetection/work_dirs/old_workdir/faster_rcnn_r101_fpn_1x/epoch_15.pth'

# config_file = '../../models/google-cloud-models/cascade-t3-vis/config.py'
# checkpoint_file = '../../models/google-cloud-models/cascade-t3-vis/epoch_1.pth'

# img_list = ['../../data/t3-data/' + ann['filename'] for ann in anns]
# img_list = ['../../data/' + ann['filename'] for ann in anns]


# +
results = []
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_list = ['../../data/ktr-test/test/B23072019_V1_K1/frame2885.jpg']

for img in tqdm_notebook(img_list):
#     if 'T190619_V1_K1/frame15748.jpg' not in img:
#         continue
    # inference on single image
    results.append(inference_detector(model, img))


# +
# %matplotlib inline
import d2l
from mxnet import image

d2l.set_figsize((20, 9))
# img = image.imread('../data/external/visdrone/train/sequences/uav0000243_00001_v/0000690.jpg').asnumpy()
img = image.imread(img_list[0]).asnumpy()

# Save to the d2l package.
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)

for bbox in filter(lambda x: x[-1] >= 0.6, results[0][1]):
    fig.axes.add_patch(bbox_to_rect(bbox[:-1], 'red'))
# -















# +
# {
#  "frame_id": 568,
#  "objeler": [
#  {
#  "tur": "yaya",
#  "x1": 235,
#  "y1": 26,
#  "x2": 312,
#  "y2": 98
#  },
#  {
#  "tur": "arac",
#  "x1": 533,
#  "y1": 498,
#  "x2": 603,
#  "y2": 581
#  }
#  ]
# }

thres = [0.3, 0.7]
results = []
for img_path, img_preds in zip(img_list, results):
    frame = img_path.split('/')[-1].split('.')[0][5:]
    objects =[]
    for i, preds in enumerate(img_preds[:2]):
        for bbox in filter(lambda x: x[-1] >= thres[i], preds):
            objects.append({
                'tur': 'yaya' if i == 0 else 'arac',
                'x1': int(bbox[0]),
                'y1': int(bbox[1]),
                'x2': int(bbox[2]),
                'y2': int(bbox[3])
            })
            
    results.append({
        'frame_id': int(frame),
        'objeler': objects
    })

# -
results


