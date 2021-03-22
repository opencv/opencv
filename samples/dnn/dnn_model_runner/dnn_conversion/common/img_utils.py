import cv2
import numpy as np

from .test.configs.default_preprocess_config import BASE_IMG_SCALE_FACTOR


def read_rgb_img(img_file, is_bgr_to_rgb=True):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if is_bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_pytorch_preprocess(img):
    img = img.astype(np.float32)
    img *= BASE_IMG_SCALE_FACTOR
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    return img
