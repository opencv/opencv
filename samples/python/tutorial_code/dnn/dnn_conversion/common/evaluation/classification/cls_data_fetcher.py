import os
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from albumentations import (
    Compose,
    Normalize,
)
from tensorflow.keras.applications.resnet import preprocess_input


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/imagenet_cls_test_alexnet.py#L26
class DataFetch(object):
    imgs_dir = ''
    frame_size = 0
    bgr_to_rgb = False

    __metaclass__ = ABCMeta

    @abstractmethod
    def preprocess(self, img):
        pass

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

        if self.bgr_to_rgb:
            img = img[..., ::-1]

        return img

    @staticmethod
    def reshape_img(img):
        # transform to HxWxC
        img = img[:, :, 0:3].transpose(2, 0, 1)
        return np.expand_dims(img, 0)

    def get_batch(self, imgs_names):
        assert type(imgs_names) is list
        batch = np.zeros((len(imgs_names), 3, self.frame_size, self.frame_size)).astype(np.float32)
        for i in range(len(imgs_names)):
            img_name = imgs_names[i]
            img_file = os.path.join(self.imgs_dir, img_name)
            assert os.path.exists(img_file)

            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            image_data = self.initial_preprocess(img)

            image_data = self.preprocess(image_data)

            batch[i] = self.reshape_img(image_data)

        return batch


class NormalizedValueFetch(DataFetch):
    def __init__(self, imgs_dir, frame_size, bgr_to_rgb):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.bgr_to_rgb = bgr_to_rgb

    def preprocess(self, img):
        # image_data = img.astype(np.float32)

        transform = Compose(
            {Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))},
        )

        # apply transformations
        image_data = transform(image=img)["image"]

        # image_data /= 255.0
        return image_data


class TFPreprocessedFetch(DataFetch):
    def __init__(self, imgs_dir, frame_size, bgr_to_rgb):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.bgr_to_rgb = bgr_to_rgb

    def preprocess(self, img):
        return preprocess_input(img)
