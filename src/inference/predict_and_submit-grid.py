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
import json
from mmdetection.mmdet.apis import init_detector, inference_detector, show_result
from tqdm import tqdm_notebook
from glob import glob
import matplotlib.pyplot as plt

import numpy as np

# %matplotlib inline
# -

# CONSTANTS
prediction_file_path = 'preds.txt'
ground_truth_file_path = 'ground-truth.txt'
anns = pickle.load(open('../../data/t3-data-grid/test.pkl', 'rb'))
tile_json = json.load(open('../../data/t3-data-grid/test_tiles.json', 'r'))


# +
frames_dict = {}

for i in tile_json.keys():
    frame = i.split('-')[0]
    if frame in frames_dict:
        frames_dict[frame].append(i)
    else:
        frames_dict[frame] = [i]

# +
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

# config_file = '../../src/configs/cascade_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../../models/work_dirs/cascade_rcnn_r50_fpn_1x-all/latest.pth'


# config_file = '../../models/google-cloud-models/cascade-t3-vis/config.py'
# checkpoint_file = '../../models/google-cloud-models/cascade-t3-vis/epoch_1.pth'

# config_file = '../../src/configs/cascade_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../../models/work_dirs/cascade_rcnn_r50_fpn_1x-cropped-t3only-nobp-yaya/latest.pth'

config_file = '../mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '../../models/work_dirs/faster_rcnn_r50_fpn_grid-yaya-t3-subset-visdrone/epoch_29.pth'


# +
results = {}
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_list = ['../../data/t3-data-grid/' + ann['filename'] for ann in anns]

for img in tqdm_notebook(img_list):
    results['/'.join(img.split('/')[-2:])[:-4]] = inference_detector(model, img)
# -


frame_bboxes = {}
for k, v in tqdm_notebook(frames_dict.items()):
    frame_bboxes[k] = []
    for vv in v:
        bboxes = results[vv][0]
        x, y, _, _ = tile_json[vv]
        for bbox in bboxes:
            frame_bboxes[k].append(bbox + [x, y, x, y, 0])

frame_bboxes['T190619_V5_K1/frame6032']


def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.6):
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return np.concatenate((boxes[pick].astype("int"), np.expand_dims(probs[pick], -1)), axis=1)


nms_frame_bboxes = {}
for frame, bboxes in frame_bboxes.items():
    if len(bboxes) > 0:
        boxes, probs = np.array(bboxes)[:,:4], np.array(bboxes)[:,-1]
        nms_frame_bboxes[frame] = non_max_suppression_fast(boxes, probs)
    else:
        nms_frame_bboxes[frame] = []

# PREDS.TXT
thres = [0.1]
with open(prediction_file_path, 'w+') as f:
    for img_path, img_preds in frame_bboxes.items():
        line = '/'.join(img_path.split('/')[-2:]) + '.jpg'
        for bbox in filter(lambda x: x[-1] >= thres[0], img_preds):
            line += ",{},{},{},{},{}".format(*bbox[:-1], 0)
        f.write(line)
        f.write('\n')        


# PREDS.TXT over NMS PREDICTIONS
thres = [0.2]
with open(prediction_file_path, 'w+') as f:
    for img_path, img_preds in nms_frame_bboxes.items():
        line = '/'.join(img_path.split('/')[-2:]) + '.jpg'
        for bbox in filter(lambda x: x[-1] >= thres[0], img_preds):
            line += ",{},{},{},{},{}".format(*bbox[:-1], 0)
        f.write(line)
        f.write('\n')        

anns = pickle.load(open('../../data/t3-data/only_yaya_test_frames.pkl', 'rb'))
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

len(frame_bboxes)

len(anns)

len(nms_frame_bboxes)

nms_bboxes['T190619_V5_K1/frame6004.jpg']

# +
import cv2
import os
from tqdm import tqdm_notebook as tqdm

frame_and_boxes = nms_frame_bboxes


# Input image base folder 
image_folder_base = '/home/deep/t3/data/t3-data/t3/'
# Output video base folder
video_output_base = './'


video_name = '{}{}.mp4'.format(video_output_base, 'faster_rcnn_test')
# sort images by frame

results = []

# get h,w to initialize canvas frame
frame = cv2.imread(os.path.join(image_folder_base, 'T190619_V5_K1/frame6004.jpg'))

# resize
height, width, layers = frame.shape
height //= 2 
width //= 2
# arguments: output_name, codec, fps, size
video = cv2.VideoWriter(video_name, 0x7634706d, 5, (width,height)) 

for frame_path in tqdm(frame_and_boxes.keys(), total=len(frame_and_boxes)):


    img_frame = cv2.imread(os.path.join(image_folder_base, frame_path + '.jpg'))

    bboxes = frame_and_boxes[frame_path][:,:-1]
    probs = frame_and_boxes[frame_path][:,-1]
    for bbox, prob in zip(bboxes, probs):
        if prob > 0.95:
            mins = int(bbox[0]), int(bbox[1])
            maxs = int(bbox[2]), int(bbox[3])
            img_frame = cv2.rectangle(img_frame, mins, maxs,(0, 0, 250),2)            
            txt = str(prob)
            font = cv2.FONT_HERSHEY_SIMPLEX
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, 0.35, 1)
            # Place text background.
            back_tl = bbox[0], bbox[1] - int(1.3 * txt_h)
            back_br = bbox[0] + txt_w, bbox[1]
            # Show text.
            txt_tl = int(bbox[0]), int(bbox[1] - int(0.3 * txt_h))
            cv2.putText(img_frame, f'{prob:.3f}', txt_tl, font, 0.6, (218, 227, 218), lineType=cv2.LINE_AA)

            
            

    font                   = cv2.FONT_HERSHEY_SIMPLEX

    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (218, 227, 218)
    lineType               = 2

    cv2.putText(img_frame,str(frame_path),
        bottomLeftCornerOfText, 
        font,
        fontScale,
        fontColor,
        lineType)
    
#     plt.rcParams["figure.figsize"] = (20,14)
#     plt.imshow(img_frame)
#     break
    
    
    video.write(cv2.resize(img_frame, (width, height)))


cv2.destroyAllWindows()
video.release()



# -

frame_and_boxes['T190619_V5_K1/frame6004'][0]

import pickle
pickle.dump(frame_bboxes, open('faster_rcnn_test_preds', 'wb'))

frame_bboxes = pickle.load(open('faster_rcnn_test_preds', 'rb'))

t['T190619_V5_K1/frame6004']


