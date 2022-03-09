import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2


def color_map(N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap


def display_results(image, boxes, masks, class_ids, class_names, scores=None,
                        show_mask=True, show_bbox=True, display_img=True,
                        save_img=False, save_dir=None, img_name=None, ret_image=False):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset (Without Background)
        scores: (optional) confidence scores for each box
        show_mask, show_bbox: To show masks and bounding boxes or not
        display_img: To display the image in popup
        save_img: To save the predict image
        save_dir: If save_img is True, the directory where you want to save the predict image
        img_name: If save_img is True, the name of the predict image

        """
        n_instances = boxes.shape[0]
        colors = color_map()
        for k in range(n_instances):
            color = colors[class_ids[k]].astype(np.uint8)
            clr = (color[0], color[1], color[2])
            if show_bbox:
                box = boxes[k]
                cls = class_names[class_ids[k]-1]  # Skip the Background
                score = scores[k]
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, '{}: {:.3f}'.format(cls, score), (box[1], box[0]),
                            font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            if show_mask:
                mask = masks[:, :, k]
                color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                color_mask[mask] = color
                image = cv2.addWeighted(color_mask, 0.5, image.astype(np.uint8), 1, 0)

        if display_img:
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        if save_img:
            cv2.imwrite(os.path.join(save_dir, img_name), image)
        if ret_image:
            return image

        return None


# Root directory of the project
ROOT_DIR = os.path.abspath('.')

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config ?
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# Local path to trained weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'weights/mask_rcnn_maskrcnn_paper_0147.h5')

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, 'images')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on one image at a time.
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = 'resnet50'
    NUM_CLASSES = 1 + 8
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 1024

config = InferenceConfig()
config.display()

# Create model objects in inference mode
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

# Load weights trained on SmurfitKappa
model.load_weights(WEIGHTS_PATH, by_name=True)

# Class names
class_names = ['BG', '1_low', '1_long', '2_low', '2_high', '3_low', '3_high', '4_high']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# If True --> inference on image, if false, inference on video
imageMode = False

if imageMode:
    for image in file_names:
        img = skimage.io.imread(os.path.join(IMAGE_DIR, image))
        results = model.detect([img], verbose=1)
        r = results[0]
        display_results(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], save_dir=RESULTS_DIR, img_name='result_' + image + '.jpg')

else:
    VIDEOS_DIR = os.path.join(ROOT_DIR, 'videos')
    VIDEOS_RES = os.path.join(RESULTS_DIR, 'videos')
    video = cv2.VideoCapture(os.path.join(VIDEOS_DIR, 'video_test.avi'))
    frame_w = int(video.get(3))
    frame_h = int(video.get(4))
    out = cv2.VideoWriter(os.path.join(VIDEOS_RES, 'result.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_w,frame_h))

    success = True
    while success:
        success, frame = video.read()
        a = time.time()
        results = model.detect([frame], verbose=1)
        r = results[0]
        img = display_results(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], display_img=False, ret_image=True)
        b = time.time()
        print(f'TOTAL INFERENCE TIME: {b - a} ; FPS: {1 / (b - a)}')

        print(img.shape)
        cv2.imshow('video', img)
        # out.write(img)
        if cv2.waitKey(1) == ord('q') & 0xFF:
            break
    out.release()
    video.release()



        
        
