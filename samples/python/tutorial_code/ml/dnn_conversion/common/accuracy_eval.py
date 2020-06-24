import os
import sys
import time
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/imagenet_cls_test_alexnet.py#L98
def get_correct_answers(img_list, img_classes, net_output_blob):
    correct_answers = 0
    for i in range(len(img_list)):
        indexes = np.argsort(net_output_blob[i])[-5:]
        correct_index = img_classes[img_list[i]]
        if correct_index in indexes:
            correct_answers += 1
    return correct_answers


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/imagenet_cls_test_alexnet.py#L26
class DataFetch(object):
    imgs_dir = ''
    frame_size = 0
    bgr_to_rgb = False

    __metaclass__ = ABCMeta

    @abstractmethod
    def preprocess(self, img):
        pass

    def get_batch(self, imgs_names):
        assert type(imgs_names) is list
        batch = np.zeros((len(imgs_names), 3, self.frame_size, self.frame_size)).astype(np.float32)
        for i in range(len(imgs_names)):
            img_name = imgs_names[i]
            # changed to os.path.join
            img_file = os.path.join(self.imgs_dir, img_name)
            assert os.path.exists(img_file)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            min_dim = min(img.shape[-3], img.shape[-2])
            resize_ratio = self.frame_size / float(min_dim)
            img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
            cols = img.shape[1]
            rows = img.shape[0]
            y1 = round((rows - self.frame_size) / 2)
            y2 = round(y1 + self.frame_size)
            x1 = round((cols - self.frame_size) / 2)
            x2 = round(x1 + self.frame_size)
            img = img[y1:y2, x1:x2]
            if self.bgr_to_rgb:
                img = img[..., ::-1]
            image_data = img[:, :, 0:3].transpose(2, 0, 1)
            batch[i] = self.preprocess(image_data)
        return batch


class NormalizedValueFetch(DataFetch):
    def __init__(self, imgs_dir, frame_size, bgr_to_rgb):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.bgr_to_rgb = bgr_to_rgb

    def preprocess(self, img):
        image_data = np.array(img).astype(np.float32)
        image_data = np.expand_dims(image_data, 0)
        image_data /= 255.0
        return image_data


class MeanValueFetch(DataFetch):
    def __init__(self, frame_size, imgs_dir, bgr_to_rgb):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.mean_blob = np.ones((3, self.frame_size, self.frame_size)).astype(np.float32)
        self.mean_blob *= 117
        self.bgr_to_rgb = bgr_to_rgb

    def preprocess(self, img):
        return img - self.mean_blob


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/imagenet_cls_test_alexnet.py#L159
class ClsAccEvaluation:
    log = sys.stdout
    img_classes = {}
    batch_size = 0

    def __init__(self, log_path, img_classes_file, batch_size):
        self.log = open(log_path, 'w')
        self.img_classes = self.read_classes(img_classes_file)
        self.batch_size = batch_size
        # collect the accuracies for both models
        self.general_fw_accuracy = []

    @staticmethod
    def read_classes(img_classes_file):
        result = {}
        with open(img_classes_file) as file:
            for l in file.readlines():
                result[l.split()[0]] = int(l.split()[1])
        return result

    def process(self, frameworks, data_fetcher):
        sorted_imgs_names = sorted(self.img_classes.keys())
        correct_answers = [0] * len(frameworks)
        samples_handled = 0
        blobs_l1_diff = [0] * len(frameworks)
        blobs_l1_diff_count = [0] * len(frameworks)
        blobs_l_inf_diff = [sys.float_info.min] * len(frameworks)
        inference_time = [0.0] * len(frameworks)

        for x in range(0, len(sorted_imgs_names), self.batch_size):
            sublist = sorted_imgs_names[x:x + self.batch_size]
            batch = data_fetcher.get_batch(sublist)

            samples_handled += len(sublist)
            fw_accuracy = []
            frameworks_out = []
            for i in range(len(frameworks)):
                start = time.time()
                out = frameworks[i].get_output(batch)
                end = time.time()
                correct_answers[i] += get_correct_answers(sublist, self.img_classes, out)
                fw_accuracy.append(100 * correct_answers[i] / float(samples_handled))
                frameworks_out.append(out)
                inference_time[i] += end - start
                print(samples_handled, 'Accuracy for', frameworks[i].get_name() + ':', fw_accuracy[i], file=self.log)
                print("Inference time, ms ", frameworks[i].get_name(), inference_time[i] / samples_handled * 1000,
                      file=self.log)

                self.general_fw_accuracy.append(fw_accuracy)

            for i in range(1, len(frameworks)):
                log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
                diff = np.abs(frameworks_out[0] - frameworks_out[i])
                l1_diff = np.sum(diff) / diff.size
                print(samples_handled, "L1 difference", log_str, l1_diff, file=self.log)
                blobs_l1_diff[i] += l1_diff
                blobs_l1_diff_count[i] += 1
                if np.max(diff) > blobs_l_inf_diff[i]:
                    blobs_l_inf_diff[i] = np.max(diff)
                print(samples_handled, "L_INF difference", log_str, blobs_l_inf_diff[i], file=self.log)

            self.log.flush()

        for i in range(1, len(blobs_l1_diff)):
            log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
            print('Final l1 diff', log_str, blobs_l1_diff[i] / blobs_l1_diff_count[i], file=self.log)
