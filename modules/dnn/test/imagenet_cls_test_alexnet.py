from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
import sys
import os
import argparse
import time

try:
    import caffe
except ImportError:
    raise ImportError('Can\'t find Caffe Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "git/caffe/python" directory')
try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


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
            img_file = self.imgs_dir + img_name
            assert os.path.exists(img_file)
            img = cv.imread(img_file, cv.IMREAD_COLOR)
            min_dim = min(img.shape[-3], img.shape[-2])
            resize_ratio = self.frame_size / float(min_dim)
            img = cv.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
            cols = img.shape[1]
            rows = img.shape[0]
            y1 = (rows - self.frame_size) / 2
            y2 = y1 + self.frame_size
            x1 = (cols - self.frame_size) / 2
            x2 = x1 + self.frame_size
            img = img[y1:y2, x1:x2]
            if self.bgr_to_rgb:
                img = img[..., ::-1]
            image_data = img[:, :, 0:3].transpose(2, 0, 1)
            batch[i] = self.preprocess(image_data)
        return batch


class MeanBlobFetch(DataFetch):
    mean_blob = np.ndarray(())

    def __init__(self, frame_size, mean_blob_path, imgs_dir):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(mean_blob_path, 'rb').read()
        blob.ParseFromString(data)
        self.mean_blob = np.array(caffe.io.blobproto_to_array(blob))
        start = (self.mean_blob.shape[2] - self.frame_size) / 2
        stop = start + self.frame_size
        self.mean_blob = self.mean_blob[:, :, start:stop, start:stop][0]

    def preprocess(self, img):
        return img - self.mean_blob


class MeanChannelsFetch(MeanBlobFetch):
    def __init__(self, frame_size, imgs_dir):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.mean_blob = np.ones((3, self.frame_size, self.frame_size)).astype(np.float32)
        self.mean_blob[0] *= 104
        self.mean_blob[1] *= 117
        self.mean_blob[2] *= 123


class MeanValueFetch(MeanBlobFetch):
    def __init__(self, frame_size, imgs_dir, bgr_to_rgb):
        self.imgs_dir = imgs_dir
        self.frame_size = frame_size
        self.mean_blob = np.ones((3, self.frame_size, self.frame_size)).astype(np.float32)
        self.mean_blob *= 117
        self.bgr_to_rgb = bgr_to_rgb


def get_correct_answers(img_list, img_classes, net_output_blob):
    correct_answers = 0
    for i in range(len(img_list)):
        indexes = np.argsort(net_output_blob[i])[-5:]
        correct_index = img_classes[img_list[i]]
        if correct_index in indexes:
            correct_answers += 1
    return correct_answers


class Framework(object):
    in_blob_name = ''
    out_blob_name = ''

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_output(self, input_blob):
        pass


class CaffeModel(Framework):
    net = caffe.Net
    need_reshape = False

    def __init__(self, prototxt, caffemodel, in_blob_name, out_blob_name, need_reshape=False):
        caffe.set_mode_cpu()
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name
        self.need_reshape = need_reshape

    def get_name(self):
        return 'Caffe'

    def get_output(self, input_blob):
        if self.need_reshape:
            self.net.blobs[self.in_blob_name].reshape(*input_blob.shape)
        return self.net.forward_all(**{self.in_blob_name: input_blob})[self.out_blob_name]


class DnnCaffeModel(Framework):
    net = object

    def __init__(self, prototxt, caffemodel, in_blob_name, out_blob_name):
        self.net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name

    def get_name(self):
        return 'DNN'

    def get_output(self, input_blob):
        self.net.setInput(input_blob, self.in_blob_name)
        return self.net.forward(self.out_blob_name)

class DNNOnnxModel(Framework):
    net = object

    def __init__(self, onnx_file, in_blob_name, out_blob_name):
        self.net = cv.dnn.readNetFromONNX(onnx_file)
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name

    def get_name(self):
        return 'DNN (ONNX)'

    def get_output(self, input_blob):
        self.net.setInput(input_blob, self.in_blob_name)
        return self.net.forward(self.out_blob_name)


class ClsAccEvaluation:
    log = sys.stdout
    img_classes = {}
    batch_size = 0

    def __init__(self, log_path, img_classes_file, batch_size):
        self.log = open(log_path, 'w')
        self.img_classes = self.read_classes(img_classes_file)
        self.batch_size = batch_size

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

        for x in xrange(0, len(sorted_imgs_names), self.batch_size):
            sublist = sorted_imgs_names[x:x + self.batch_size]
            batch = data_fetcher.get_batch(sublist)

            samples_handled += len(sublist)

            frameworks_out = []
            fw_accuracy = []
            for i in range(len(frameworks)):
                start = time.time()
                out = frameworks[i].get_output(batch)
                end = time.time()
                correct_answers[i] += get_correct_answers(sublist, self.img_classes, out)
                fw_accuracy.append(100 * correct_answers[i] / float(samples_handled))
                frameworks_out.append(out)
                inference_time[i] += end - start
                print(samples_handled, 'Accuracy for', frameworks[i].get_name() + ':', fw_accuracy[i], file=self.log)
                print("Inference time, ms ", \
                    frameworks[i].get_name(), inference_time[i] / samples_handled * 1000, file=self.log)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", help="path to ImageNet validation subset images dir, ILSVRC2012_img_val dir")
    parser.add_argument("--img_cls_file", help="path to file with classes ids for images, val.txt file from this "
                                               "archive: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz")
    parser.add_argument("--prototxt", help="path to caffe prototxt, download it here: "
                                        "https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt")
    parser.add_argument("--caffemodel", help="path to caffemodel file, download it here: "
                                             "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel")
    parser.add_argument("--log", help="path to logging file")
    parser.add_argument("--mean", help="path to ImageNet mean blob caffe file, imagenet_mean.binaryproto file from"
                                       "this archive: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz")
    parser.add_argument("--batch_size", help="size of images in batch", default=1000)
    parser.add_argument("--frame_size", help="size of input image", default=227)
    parser.add_argument("--in_blob", help="name for input blob", default='data')
    parser.add_argument("--out_blob", help="name for output blob", default='prob')
    args = parser.parse_args()

    data_fetcher = MeanBlobFetch(args.frame_size, args.mean, args.imgs_dir)

    frameworks = [CaffeModel(args.prototxt, args.caffemodel, args.in_blob, args.out_blob),
                  DnnCaffeModel(args.prototxt, args.caffemodel, '', args.out_blob)]

    acc_eval = ClsAccEvaluation(args.log, args.img_cls_file, args.batch_size)
    acc_eval.process(frameworks, data_fetcher)
