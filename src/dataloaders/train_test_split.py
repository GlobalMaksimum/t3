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
import cv2
import os
from tqdm import tqdm
import math
import json
import numpy as np
import pickle


# CONSTANTS
SCENE_NAMES = ['T190619_V1_K1', 'T190619_V2_K1', 'T190619_V3_K1', 'B160519_V1_K1']
SCENE_NAMES += ['T190619_V5_K1', 'B270619_V1_K1']
NUM_THROW_AWAY = 90 # how many frames will be removed between each split
TEST_SIZE, VAL_SIZE, TRAIN_SIZE = 0.4, 0.2, 0.4 # manually adjusted sample sizes

DATA_PATH = '../../data/t3-data/merged_veriler/'
FRAME_FILE_PATH = DATA_PATH + 'veriler.json'
TRAINING_ANNOT_PATH = DATA_PATH + 'training.pkl'
VAL_ANNOT_PATH = DATA_PATH + 'validation.pkl'
TEST_ANNOT_PATH = DATA_PATH + 'test.pkl'


# annotation files to be filled in
train_annot, val_annot, test_annot = [], [], []


# -

def t3_to_mmdetection_annotation(frame_annot):
    """
    Transforms single t3 object annotation to mmdetection custom dataset annotation type.
    Note that if class is `yaya`, it is represented with `0`. `Arac` is represented with `1`.
    
    Example:
    
    T3 Annotation Type:
    ------------------------------------------
    {'frame_url': 'T190619_V1_K1/frame4948.jpg',
     'frame_width': 1920,
     'frame_height': 900,
     'objeler': [{'tur': 'arac',
       'x0': 61.2,
       'y0': 417.1,
       'x1': 121.7055,
       'y1': 448.43},
      {'tur': 'arac',
       'x0': 1478.74,
       'y0': 359.07,
       'x1': 1544.8600000000001,
       'y1': 387.74},
      {'tur': 'arac', 'x0': 830.2, 'y0': 369.95, 'x1': 889.08, 'y1': 395.95}]}


    MMDetection Annotation Type
    -------------------------------------------
    {'filename': 'T190619_V1_K1/frame4948.jpg',
     'width': 1920,
     'height': 900,
     'ann': {'bboxes': array([[  61.2   ,  417.1   ,  121.7055,  448.43  ],
             [1478.74  ,  359.07  , 1544.86  ,  387.74  ],
             [ 830.2   ,  369.95  ,  889.08  ,  395.95  ]], dtype=float32),
      'labels': array([1, 1, 1]),
      'bboxes_ignore': [],
      'labels_ignore': []}}
    
    """
    mmdetection_annot = {}
    mmdetection_annot['filename'] = frame_annot['frame_url']
    mmdetection_annot['width'] = frame_annot['frame_width']
    mmdetection_annot['height'] = frame_annot['frame_height']
    mmdetection_annot['ann'] = {'bboxes': [], 'labels': [], 'bboxes_ignore': [], 'labels_ignore': [] }
    
    for obj in frame_annot['objeler']:
        bbox = [obj['x0'], obj['y0'], obj['x1'], obj['y1']]
        # 1 if class is yaya, 2 if arac
        if 'tur' in obj: # some objects' class are not labeled, we skip them
            mmdetection_annot['ann']['bboxes'].append(bbox)
            mmdetection_annot['ann']['labels'].append(1 if obj['tur'] != 'arac' else 2)
            
    # Type checks
    mmdetection_annot['ann']['bboxes'] = np.array(mmdetection_annot['ann']['bboxes']).astype('float32')
    mmdetection_annot['ann']['labels'] = np.array(mmdetection_annot['ann']['labels']).astype('int64')

    return mmdetection_annot


def main():
    for scene_name in SCENE_NAMES:
        image_folder = DATA_PATH + '{}/'.format(scene_name)

        # get names 
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

        # sort images by frame
        images = sorted(images, key=lambda x: int(x.split('.')[0][5:]))

        ### IDEA OF SPLITTING  ###
        # Split frames as:
        #
        # Validation(val_ratio * len) - Throw away-Train(train_ratio * len) - Throw away - Test(test_ratio * len)
        # 
        # Why throw away?: If we have n'th frame in validation and (n+1)'th frame in training this may cause overfitting.
        ##########################


        LEN_AFTER_TRASH = len(images) - 2 * NUM_THROW_AWAY

        num_val, num_train, num_test = math.floor(LEN_AFTER_TRASH * VAL_SIZE),\
                                       math.floor(LEN_AFTER_TRASH * TRAIN_SIZE), math.floor(LEN_AFTER_TRASH * TEST_SIZE) 

        val_upper = num_val 
        train_lower = val_upper + NUM_THROW_AWAY
        train_upper = train_lower + num_train
        test_lower = train_upper + NUM_THROW_AWAY

        val, train, test = images[:val_upper], images[train_lower:train_upper], images[test_lower:]

        val = [os.path.join(scene_name, i) for i in val]
        train = [os.path.join(scene_name, i) for i in train]
        test = [os.path.join(scene_name, i) for i in test]

        
        # Sanity checks to prevent leakage
        assert not len(set(val).intersection(set(train))), 'Same instance cannot be in both training and val set'
        assert not len(set(test).intersection(set(train))), 'Same instance cannot be in both training and test set'

        frames = json.load(open(FRAME_FILE_PATH))['frameler']
        # fill the frames with objects to dictionaries
        visited_frame_urls = []
        for frame in frames:
            # avoids duplicate frame annotations
            if frame['frame_url'] in val and frame['frame_url'] not in visited_frame_urls:
                val_annot.append(t3_to_mmdetection_annotation(frame))
            elif frame['frame_url'] in train and frame['frame_url'] not in visited_frame_urls:
                train_annot.append(t3_to_mmdetection_annotation(frame))
            elif frame['frame_url'] in test and frame['frame_url'] not in visited_frame_urls:
                test_annot.append(t3_to_mmdetection_annotation(frame))
            visited_frame_urls.append(frame['frame_url'])
        
        del visited_frame_urls
    
    
    # We use pickle instead of JSON beacuse bounding box arrays are required 
    # to be numpy arrays which is not possible with JSON
    # Output files to Pickle
    with open(TRAINING_ANNOT_PATH, 'wb') as handle:
        pickle.dump(train_annot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(VAL_ANNOT_PATH, 'wb') as handle:
        pickle.dump(val_annot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(TEST_ANNOT_PATH, 'wb') as handle:
        pickle.dump(test_annot, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()


