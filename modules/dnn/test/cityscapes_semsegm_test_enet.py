import numpy as np
import sys
import os
import fnmatch
import argparse

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')
try:
    import torch
except ImportError:
    raise ImportError('Can\'t find pytorch. Please install it by following instructions on the official site')

from torch.utils.serialization import load_lua
from pascal_semsegm_test_fcn import eval_segm_result, get_conf_mat, get_metrics, DatasetImageFetch, SemSegmEvaluation
from imagenet_cls_test_alexnet import Framework, DnnCaffeModel


class NormalizePreproc:
    def __init__(self):
        pass

    @staticmethod
    def process(img):
        image_data = np.array(img).transpose(2, 0, 1).astype(np.float32)
        image_data = np.expand_dims(image_data, 0)
        image_data /= 255.0
        return image_data


class CityscapesDataFetch(DatasetImageFetch):
    img_dir = ''
    segm_dir = ''
    segm_files = []
    colors = []
    i = 0

    def __init__(self, img_dir, segm_dir, preproc):
        self.img_dir = img_dir
        self.segm_dir = segm_dir
        self.segm_files = sorted([img for img in self.locate('*_color.png', segm_dir)])
        self.colors = self.get_colors()
        self.data_prepoc = preproc
        self.i = 0

    @staticmethod
    def get_colors():
        result = []
        colors_list = (
         (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
         (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
         (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32))

        for c in colors_list:
            result.append(DatasetImageFetch.pix_to_c(c))
        return result

    def __iter__(self):
        return self

    def next(self):
        if self.i < len(self.segm_files):
            segm_file = self.segm_files[self.i]
            segm = cv.imread(segm_file, cv.IMREAD_COLOR)[:, :, ::-1]
            segm = cv.resize(segm, (1024, 512), interpolation=cv.INTER_NEAREST)

            img_file = self.rreplace(self.img_dir + segm_file[len(self.segm_dir):], 'gtFine_color', 'leftImg8bit')
            assert os.path.exists(img_file)
            img = cv.imread(img_file, cv.IMREAD_COLOR)[:, :, ::-1]
            img = cv.resize(img, (1024, 512))

            self.i += 1
            gt = self.color_to_gt(segm, self.colors)
            img = self.data_prepoc.process(img)
            return img, gt
        else:
            self.i = 0
            raise StopIteration

    def get_num_classes(self):
        return len(self.colors)

    @staticmethod
    def locate(pattern, root_path):
        for path, dirs, files in os.walk(os.path.abspath(root_path)):
            for filename in fnmatch.filter(files, pattern):
                yield os.path.join(path, filename)

    @staticmethod
    def rreplace(s, old, new, occurrence=1):
        li = s.rsplit(old, occurrence)
        return new.join(li)


class TorchModel(Framework):
    net = object

    def __init__(self, model_file):
        self.net = load_lua(model_file)

    def get_name(self):
        return 'Torch'

    def get_output(self, input_blob):
        tensor = torch.FloatTensor(input_blob)
        out = self.net.forward(tensor).numpy()
        return out


class DnnTorchModel(DnnCaffeModel):
    net = cv.dnn.Net()

    def __init__(self, model_file):
        self.net = cv.dnn.readNetFromTorch(model_file)

    def get_output(self, input_blob):
        self.net.setBlob("", input_blob)
        self.net.forward()
        return self.net.getBlob(self.net.getLayerNames()[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", help="path to Cityscapes validation images dir, imgsfine/leftImg8bit/val")
    parser.add_argument("--segm_dir", help="path to Cityscapes dir with segmentation, gtfine/gtFine/val")
    parser.add_argument("--model", help="path to torch model, download it here: "
                        "https://www.dropbox.com/sh/dywzk3gyb12hpe5/AAD5YkUa8XgMpHs2gCRgmCVCa")
    parser.add_argument("--log", help="path to logging file")
    args = parser.parse_args()

    prep = NormalizePreproc()
    df = CityscapesDataFetch(args.imgs_dir, args.segm_dir, prep)

    fw = [TorchModel(args.model),
          DnnTorchModel(args.model)]

    segm_eval = SemSegmEvaluation(args.log)
    segm_eval.process(fw, df)
