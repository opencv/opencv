from abc import ABCMeta, abstractmethod
import numpy as np
import sys
import argparse
import time

from imagenet_cls_test_alexnet import CaffeModel, DnnCaffeModel
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
    res = np.zeros(net_out.shape).astype(np.int)
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
    def process(img):
        image_data = np.array(img).transpose(2, 0, 1).astype(np.float32)
        mean = np.ones(image_data.shape)
        mean[0] *= 104
        mean[1] *= 117
        mean[2] *= 123
        image_data -= mean
        image_data = np.expand_dims(image_data, 0)
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
        gt = np.zeros((num_classes, color_img.shape[0], color_img.shape[1])).astype(np.int)
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

    def __init__(self, img_dir, segm_dir, names_file, segm_cls_colors_file, preproc):
        self.img_dir = img_dir
        self.segm_dir = segm_dir
        self.colors = self.read_colors(segm_cls_colors_file)
        self.data_prepoc = preproc
        self.i = 0

        with open(names_file) as f:
            for l in f.readlines():
                self.names.append(l.rstrip())

    @staticmethod
    def read_colors(img_classes_file):
        result = []
        with open(img_classes_file) as f:
            for l in f.readlines():
                color = np.array(map(int, l.split()[1:]))
                result.append(DatasetImageFetch.pix_to_c(color))
        return result

    def __iter__(self):
        return self

    def next(self):
        if self.i < len(self.names):
            name = self.names[self.i]
            self.i += 1
            segm_file = self.segm_dir + name + ".png"
            img_file = self.img_dir + name + ".jpg"
            gt = self.color_to_gt(cv.imread(segm_file, cv.IMREAD_COLOR)[:, :, ::-1], self.colors)
            img = self.data_prepoc.process(cv.imread(img_file, cv.IMREAD_COLOR)[:, :, ::-1])
            return img, gt
        else:
            self.i = 0
            raise StopIteration

    def get_num_classes(self):
        return len(self.colors)


class SemSegmEvaluation:
    log = file

    def __init__(self, log_path,):
        self.log = open(log_path, 'w')

    def process(self, frameworks, data_fetcher):
        samples_handled = 0

        conf_mats = [np.zeros((data_fetcher.get_num_classes(), data_fetcher.get_num_classes())) for i in range(len(frameworks))]
        blobs_l1_diff = [0] * len(frameworks)
        blobs_l1_diff_count = [0] * len(frameworks)
        blobs_l_inf_diff = [sys.float_info.min] * len(frameworks)
        inference_time = [0.0] * len(frameworks)

        for in_blob, gt in data_fetcher:
            frameworks_out = []
            samples_handled += 1
            for i in range(len(frameworks)):
                start = time.time()
                out = frameworks[i].get_output(in_blob)
                end = time.time()
                segm = eval_segm_result(out)
                conf_mats[i] += get_conf_mat(gt, segm[0])
                frameworks_out.append(out)
                inference_time[i] += end - start

                pix_acc, mean_acc, miou = get_metrics(conf_mats[i])

                name = frameworks[i].get_name()
                print >> self.log, samples_handled, 'Pixel accuracy, %s:' % name, 100 * pix_acc
                print >> self.log, samples_handled, 'Mean accuracy, %s:' % name, 100 * mean_acc
                print >> self.log, samples_handled, 'Mean IOU, %s:' % name, 100 * miou
                print >> self.log, "Inference time, ms ", \
                    frameworks[i].get_name(), inference_time[i] / samples_handled * 1000

            for i in range(1, len(frameworks)):
                log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
                diff = np.abs(frameworks_out[0] - frameworks_out[i])
                l1_diff = np.sum(diff) / diff.size
                print >> self.log, samples_handled, "L1 difference", log_str, l1_diff
                blobs_l1_diff[i] += l1_diff
                blobs_l1_diff_count[i] += 1
                if np.max(diff) > blobs_l_inf_diff[i]:
                    blobs_l_inf_diff[i] = np.max(diff)
                print >> self.log, samples_handled, "L_INF difference", log_str, blobs_l_inf_diff[i]

            self.log.flush()

        for i in range(1, len(blobs_l1_diff)):
            log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
            print >> self.log, 'Final l1 diff', log_str, blobs_l1_diff[i] / blobs_l1_diff_count[i]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", help="path to PASCAL VOC 2012 images dir, data/VOC2012/JPEGImages")
    parser.add_argument("--segm_dir", help="path to PASCAL VOC 2012 segmentation dir, data/VOC2012/SegmentationClass/")
    parser.add_argument("--val_names", help="path to file with validation set image names, download it here: "
                        "https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt")
    parser.add_argument("--cls_file", help="path to file with colors for classes, download it here: "
                        "https://github.com/opencv/opencv/blob/master/samples/data/dnn/pascal-classes.txt")
    parser.add_argument("--prototxt", help="path to caffe prototxt, download it here: "
                        "https://github.com/opencv/opencv/blob/master/samples/data/dnn/fcn8s-heavy-pascal.prototxt")
    parser.add_argument("--caffemodel", help="path to caffemodel file, download it here: "
                                             "http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel")
    parser.add_argument("--log", help="path to logging file")
    parser.add_argument("--in_blob", help="name for input blob", default='data')
    parser.add_argument("--out_blob", help="name for output blob", default='score')
    args = parser.parse_args()

    prep = MeanChannelsPreproc()
    df = PASCALDataFetch(args.imgs_dir, args.segm_dir, args.val_names, args.cls_file, prep)

    fw = [CaffeModel(args.prototxt, args.caffemodel, args.in_blob, args.out_blob, True),
          DnnCaffeModel(args.prototxt, args.caffemodel, '', args.out_blob)]

    segm_eval = SemSegmEvaluation(args.log)
    segm_eval.process(fw, df)
