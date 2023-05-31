import numpy as np
import sys
import os
import argparse
from imagenet_cls_test_alexnet import MeanChannelsFetch, CaffeModel, DnnCaffeModel, ClsAccEvaluation
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
    parser.add_argument("--batch_size", help="size of images in batch", default=500, type=int)
    parser.add_argument("--frame_size", help="size of input image", default=224, type=int)
    parser.add_argument("--in_blob", help="name for input blob", default='data')
    parser.add_argument("--out_blob", help="name for output blob", default='prob')
    args = parser.parse_args()

    data_fetcher = MeanChannelsFetch(args.frame_size, args.imgs_dir)

    frameworks = [CaffeModel(args.prototxt, args.caffemodel, args.in_blob, args.out_blob),
                  DnnCaffeModel(args.prototxt, args.caffemodel, '', args.out_blob)]

    acc_eval = ClsAccEvaluation(args.log, args.img_cls_file, args.batch_size)
    acc_eval.process(frameworks, data_fetcher)
