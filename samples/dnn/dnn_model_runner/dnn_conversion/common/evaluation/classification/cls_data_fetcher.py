import os
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np

from ...img_utils import read_rgb_img, get_pytorch_preprocess
from ...test.configs.default_preprocess_config import PYTORCH_RSZ_HEIGHT, PYTORCH_RSZ_WIDTH


class DataFetch(object):
    imgs_dir = ''
    frame_size = 0
    bgr_to_rgb = False

    __metaclass__ = ABCMeta

    @abstractmethod
    def preprocess(self, img):
        pass

    @staticmethod
    def reshape_img(img):
        img = img[:, :, 0:3].transpose(2, 0, 1)
        return np.expand_dims(img, 0)

    def center_crop(self, img):
        cols = img.shape[1]
        rows = img.shape[0]

        y1 = round((rows - self.frame_size) / 2)
        y2 = round(y1 + self.frame_size)
        x1 = round((cols - self.frame_size) / 2)
        x2 = round(x1 + self.frame_size)
        return img[y1:y2, x1:x2]

    def initial_preprocess(self, img):
        min_dim = min(img.shape[-3], img.shape[-2])
        resize_ratio = self.frame_size / float(min_dim)

        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
        img = self.center_crop(img)
        return img

    def get_preprocessed_img(self, img_path):
        image_data = read_rgb_img(img_path, self.bgr_to_rgb)
        image_data = self.preprocess(image_data)
        return self.reshape_img(image_data)

    def get_batch(self, img_names):
        assert type(img_names) is list
        batch = np.zeros((len(img_names), 3, self.frame_size, self.frame_size)).astype(np.float32)

        for i in range(len(img_names)):
            img_name = img_names[i]
            img_file = os.path.join(self.imgs_dir, img_name)
            assert os.path.exists(img_file)

            batch[i] = self.get_preprocessed_img(img_file)
        return batch


class PyTorchPreprocessedFetch(DataFetch):
    def __init__(self, pytorch_cls_config, preprocess_input=None):
        self.imgs_dir = pytorch_cls_config.img_root_dir
        self.frame_size = pytorch_cls_config.frame_size
        self.bgr_to_rgb = pytorch_cls_config.bgr_to_rgb
        self.preprocess_input = preprocess_input

    def preprocess(self, img):
        img = cv2.resize(img, (PYTORCH_RSZ_WIDTH, PYTORCH_RSZ_HEIGHT))
        img = self.center_crop(img)
        if self.preprocess_input:
            return self.presprocess_input(img)
        return get_pytorch_preprocess(img)


class TFPreprocessedFetch(DataFetch):
    def __init__(self, tf_cls_config, preprocess_input):
        self.imgs_dir = tf_cls_config.img_root_dir
        self.frame_size = tf_cls_config.frame_size
        self.bgr_to_rgb = tf_cls_config.bgr_to_rgb
        self.preprocess_input = preprocess_input

    def preprocess(self, img):
        img = self.initial_preprocess(img)
        return self.preprocess_input(img)
