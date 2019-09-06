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
from tqdm import tqdm_notebook as tqdm
from glob import glob

import cv2
import os

import matplotlib.pyplot as plt

# %matplotlib inline

# +
scene_names = ['B23072019_V1_K1', 'T190619_V4_K1']
# Input image base folder 
image_folder_base = '/home/deep/t3/data/ktr-test/test/'
# Output video base folder
video_output_base = './'

config_file = '../../models/google-cloud-models/cascade-t3-vis/config.py'
checkpoint_file = '../../models/google-cloud-models/cascade-t3-vis/epoch_1.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

thres = [0.4, 0.6]

for scene_name in scene_names:
    image_folder = '{}{}/'.format(image_folder_base, scene_name)
    video_name = '{}{}.mp4'.format(video_output_base, scene_name)

    # get names 
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # sort images by frame
    images = sorted(images, key=lambda x: int(x.split('.')[0][5:]))
    
    results = []
    
    for img in tqdm(images):
        results.append(inference_detector(model, f"{image_folder_base}{scene_name}/{img}"))

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # resize
    height, width, layers = frame.shape
    height //= 2 
    width //= 2
    # arguments: output_name, codec, fps, size
    video = cv2.VideoWriter(video_name, 0x7634706d, 5, (width,height)) 

    for idx, image in tqdm(enumerate(images), total=len(images)):
        img_frame = cv2.imread(os.path.join(image_folder, image))
        
        for i, preds in enumerate(results[idx][:2]):
            for bbox in filter(lambda x: x[-1] >= thres[i], preds):
                mins = int(bbox[0]), int(bbox[1])
                maxs = int(bbox[2]), int(bbox[3])
                img_frame = cv2.rectangle(img_frame, mins, maxs,(255 * i, 255 * (1 - i),0),3)

        
        video.write(cv2.resize(img_frame, (width, height)))

    cv2.destroyAllWindows()
    video.release()

