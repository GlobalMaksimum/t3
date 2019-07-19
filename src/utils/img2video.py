import cv2
import os
from tqdm import tqdm

# Folder names that contains scenes
scene_names = ['T190619_V1_K1', 'T190619_V2_K1', 'T190619_V3_K1', 'B160519_V1_K1']
# Input image base folder 
image_folder_base = '../../data/t3-data/gonderilecek_veriler/'
# Output video base folder
video_output_base = '../../data/t3-data/gonderilecek_veriler/'


for scene_name in scene_names:
    image_folder = '{}{}/'.format(image_folder_base, scene_name)
    video_name = '{}{}.mp4'.format(video_output_base, scene_name)

    # get names 
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # sort images by frame
    images = sorted(images, key=lambda x: int(x.split('.')[0][5:]))

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # resize
    height, width, layers = frame.shape
    height //= 2 
    width //= 2
    # arguments: output_name, codec, fps, size
    video = cv2.VideoWriter(video_name, 0x7634706d, 60, (width,height)) 

    for image in tqdm(images, total=len(images)):
        img_frame = cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height))
        video.write(img_frame)

    cv2.destroyAllWindows()
    video.release()