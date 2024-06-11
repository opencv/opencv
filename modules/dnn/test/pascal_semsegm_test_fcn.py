from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
import sys
import argparse
import time

from imagenet_cls_test_alexnet import CaffeModel, DNNOnnxModel
try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')


def get_metrics(conf_mat):
    pix_accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    t = np.sum(conf_mat, 1)
    num_cl = np.count_nonzero(t)
    assert num_cl
    mean_accuracy = np.sum(np.nan_to_num(np.divide(np.diagonal(conf_mat), t))) / num_cl
    col_sum = np.sum(conf_mat, 0)
    mean_iou = np.sum(
        np.nan_to_num(np.divide(np.diagonal(conf_mat), (t + col_sum - np.diagonal(conf_mat))))) / num_cl
    return pix_accuracy, mean_accuracy, mean_iou


def eval_segm_result(net_out):
    assert type(net_out) is np.ndarray
    assert len(net_out.shape) == 4

    channels_dim = 1
    y_dim = channels_dim + 1
    x_dim = y_dim + 1
    res = np.zeros(net_out.shape).astype(int)
    for i in range(net_out.shape[y_dim]):
        for j in range(net_out.shape[x_dim]):
            max_ch = np.argmax(net_out[..., i, j])
            res[0, max_ch, i, j] = 1
    return res


def get_conf_mat(gt, prob):
    assert type(gt) is np.ndarray
    assert type(prob) is np.ndarray

    conf_mat = np.zeros((gt.shape[0], gt.shape[0]))
    for ch_gt in range(conf_mat.shape[0]):
        gt_channel = gt[ch_gt, ...]
        for ch_pr in range(conf_mat.shape[1]):
            prob_channel = prob[ch_pr, ...]
            conf_mat[ch_gt][ch_pr] = np.count_nonzero(np.multiply(gt_channel, prob_channel))
    return conf_mat


class MeanChannelsPreproc:
    def __init__(self):
        pass

    @staticmethod
    def process(img, framework):
        image_data = None
        if framework == "Caffe":
            image_data = cv.dnn.blobFromImage(img, scalefactor=1.0, mean=(123.0, 117.0, 104.0), swapRB=True)
        elif framework == "DNN (ONNX)":
            image_data = cv.dnn.blobFromImage(img, scalefactor=0.019, mean=(123.675, 116.28, 103.53), swapRB=True)
        else:
            raise ValueError("Unknown framework")
        return image_data


class DatasetImageFetch(object):
    __metaclass__ = ABCMeta
    data_prepoc = object

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @staticmethod
    def pix_to_c(pix):
        return pix[0] * 256 * 256 + pix[1] * 256 + pix[2]

    @staticmethod
    def color_to_gt(color_img, colors):
        num_classes = len(colors)
        gt = np.zeros((num_classes, color_img.shape[0], color_img.shape[1])).astype(int)
        for img_y in range(color_img.shape[0]):
            for img_x in range(color_img.shape[1]):
                c = DatasetImageFetch.pix_to_c(color_img[img_y][img_x])
                if c in colors:
                    cls = colors.index(c)
                    gt[cls][img_y][img_x] = 1
        return gt


class PASCALDataFetch(DatasetImageFetch):
    img_dir = ''
    segm_dir = ''
    names = []
    colors = []
    i = 0

    def __init__(self, img_dir, segm_dir, names_file, segm_cls_colors, preproc):
        self.img_dir = img_dir
        self.segm_dir = segm_dir
        self.colors = self.read_colors(segm_cls_colors)
        self.data_prepoc = preproc
        self.i = 0

        with open(names_file) as f:
            for l in f.readlines():
                self.names.append(l.rstrip())

    @staticmethod
    def read_colors(colors):
        result = []
        for color in colors:
            result.append(DatasetImageFetch.pix_to_c(color))
        return result

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.names):
            name = self.names[self.i]
            self.i += 1
            segm_file = self.segm_dir + name + ".png"
            img_file = self.img_dir + name + ".jpg"
            gt = self.color_to_gt(cv.imread(segm_file, cv.IMREAD_COLOR)[:, :, ::-1], self.colors)
            img = cv.imread(img_file, cv.IMREAD_COLOR)
            img_caffe = self.data_prepoc.process(img[:, :, ::-1], "Caffe")
            img_dnn = self.data_prepoc.process(img[:, :, ::-1], "DNN (ONNX)")
            img_dict = {
                "Caffe": img_caffe,
                "DNN (ONNX)": img_dnn
            }
            return img_dict, gt
        else:
            self.i = 0
            raise StopIteration

    def get_num_classes(self):
        return len(self.colors)


class SemSegmEvaluation:
    log = sys.stdout

    def __init__(self, log_path,):
        self.log = open(log_path, 'w')

    def process(self, frameworks, data_fetcher):
        samples_handled = 0

        conf_mats = [np.zeros((data_fetcher.get_num_classes(), data_fetcher.get_num_classes())) for i in range(len(frameworks))]
        blobs_l1_diff = [0] * len(frameworks)
        blobs_l1_diff_count = [0] * len(frameworks)
        blobs_l_inf_diff = [sys.float_info.min] * len(frameworks)
        inference_time = [0.0] * len(frameworks)

        for in_blob_dict, gt in data_fetcher:
            frameworks_out = []
            samples_handled += 1
            for i in range(len(frameworks)):
                start = time.time()
                framework_name = frameworks[i].get_name()
                out = frameworks[i].get_output(in_blob_dict[framework_name])
                end = time.time()
                segm = eval_segm_result(out)
                conf_mats[i] += get_conf_mat(gt, segm[0])
                frameworks_out.append(out)
                inference_time[i] += end - start

                pix_acc, mean_acc, miou = get_metrics(conf_mats[i])

                name = frameworks[i].get_name()
                print(samples_handled, 'Pixel accuracy, %s:' % name, 100 * pix_acc, file=self.log)
                print(samples_handled, 'Mean accuracy, %s:' % name, 100 * mean_acc, file=self.log)
                print(samples_handled, 'Mean IOU, %s:' % name, 100 * miou, file=self.log)
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

# PASCAL VOC 2012 classes colors
colors_pascal_voc_2012 = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", help="path to PASCAL VOC 2012 images dir, data/VOC2012/JPEGImages")
    parser.add_argument("--segm_dir", help="path to PASCAL VOC 2012 segmentation dir, data/VOC2012/SegmentationClass/")
    parser.add_argument("--val_names", help="path to file with validation set image names, download it here: "
                        "https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt")
    parser.add_argument("--prototxt", help="path to caffe prototxt, download it here: "
                        "https://github.com/opencv/opencv/blob/5.x/samples/data/dnn/fcn8s-heavy-pascal.prototxt")
    parser.add_argument("--caffemodel", help="path to caffemodel file, download it here: "
                                             "http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel")
    parser.add_argument("--onnxmodel", help="path to onnx model file, download it here: "
                                             "https://github.com/onnx/models/raw/491ce05590abb7551d7fae43c067c060eeb575a6/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx")
    parser.add_argument("--log", help="path to logging file", default='log.txt')
    parser.add_argument("--in_blob", help="name for input blob", default='data')
    parser.add_argument("--out_blob", help="name for output blob", default='score')
    args = parser.parse_args()

    prep = MeanChannelsPreproc()
    df = PASCALDataFetch(args.imgs_dir, args.segm_dir, args.val_names, colors_pascal_voc_2012, prep)

    fw = [CaffeModel(args.prototxt, args.caffemodel, args.in_blob, args.out_blob, True),
        DNNOnnxModel(args.onnxmodel, args.in_blob, args.out_blob)]

    segm_eval = SemSegmEvaluation(args.log)
    segm_eval.process(fw, df)
