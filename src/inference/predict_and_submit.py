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
anns = pickle.load(open('../../data/t3-data/only_yaya_test_frames.pkl', 'rb'))

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

img_list = [f for f in glob('../../data/ktr-test/test/B23072019_V1_K1/**.jpg')]
# img_list = ['../../data/ktr-test/test/B23072019_V1_K1/frame3500.jpg']

for img in tqdm_notebook(img_list):
#     if 'T190619_V1_K1/frame15748.jpg' not in img:
#         continue
    # inference on single image
    results.append(inference_detector(model, img))
# -


[i for i,r in enumerate(results) if len(r[0]) > 0]

img_list[7]

print(results[7])

# +
N = 0 # nth image to be shown
threshold = 0.10


# %matplotlib inline
import cv2
import colorsys
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (15,12) # w, h

# img = image.imread('../data/external/visdrone/train/sequences/uav0000243_00001_v/0000690.jpg').asnumpy()

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
font_scale=0.85

im = cv2.imread(img_list[N])


for i in filter(lambda x: x[-1] >= threshold, results[N][0]):
    cv2.rectangle(im, (int(i[0]), int(i[1])),
                     (int(i[2]), int(i[3])), _GREEN, 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize('person', font, font_scale, 1)
    back_tl = int(i[0]), int(i[1] - 1.3 * txt_h)
    back_br = int(i[0] + txt_w), int(i[1])
    cv2.rectangle(im, back_tl, back_br, _GRAY, -1)
    
    cv2.rectangle(im, back_tl, back_br, _GRAY, -1)



    txt_tl = int(i[0]), int(i[1]) - int(0.3 * txt_h)

    cv2.putText(im, 'person: '+ str(i[4]) , txt_tl,
                             cv2.FONT_HERSHEY_DUPLEX, font_scale, _WHITE, 1, cv2.LINE_AA)


im2 = im[:,:,::-1]
plt.imshow(im2)
# -
plt.imshow(cv2.imread(img_list[N]))

# PREDS.TXT
thres = [0.1]
with open(prediction_file_path, 'w+') as f:
    for img_path, img_preds in zip(img_list, results):
        line = '/'.join(img_path.split('/')[-2:])
        for i, preds in enumerate(img_preds[:2]):
            for bbox in filter(lambda x: x[-1] >= thres[i], preds):
                line += ",{},{},{},{},{}".format(*bbox[:-1], i)
        f.write(line)
        f.write('\n')


# GROUND TRUTH
with open(ground_truth_file_path, 'w+') as f:
    for img in anns:
        line = '/'.join(img['filename'].split('/')[-2:])
        for i, bbox in enumerate(img['ann']['bboxes']):
            line += ",{},{},{},{},{}".format(*bbox, img['ann']['labels'][i] - 1)
        f.write(line)
        f.write('\n')

! /home/deep/miniconda3/envs/open-mmlab/bin/python ../eval/evaluate.py ground-truth.txt preds.txt

! /home/deep/miniconda3/envs/open-mmlab/bin/python ../eval/t3_evaluate.py ground-truth.txt preds.txt


