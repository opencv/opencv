import numpy as np
import sys
import os
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
from imagenet_cls_test_alexnet import MeanValueFetch, DnnCaffeModel, Framework, ClsAccEvaluation
try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

# If you've got an exception "Cannot load libmkl_avx.so or libmkl_def.so" or similar, try to export next variable
# before running the script:
# LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_sequential.so


class TensorflowModel(Framework):
    sess = tf.Session
    output = tf.Graph

    def __init__(self, model_file, in_blob_name, out_blob_name):
        self.in_blob_name = in_blob_name
        self.sess = tf.Session()
        with gfile.FastGFile(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        self.output = self.sess.graph.get_tensor_by_name(out_blob_name + ":0")

    def get_name(self):
        return 'Tensorflow'

    def get_output(self, input_blob):
        assert len(input_blob.shape) == 4
        batch_tf = input_blob.transpose(0, 2, 3, 1)
        out = self.sess.run(self.output,
                       {self.in_blob_name+':0': batch_tf})
        out = out[..., 1:1001]
        return out


class DnnTfInceptionModel(DnnCaffeModel):
    net = cv.dnn.Net()

    def __init__(self, model_file, in_blob_name, out_blob_name):
        self.net = cv.dnn.readNetFromTensorflow(model_file)
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name

    def get_output(self, input_blob):
        return super(DnnTfInceptionModel, self).get_output(input_blob)[..., 1:1001]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", help="path to ImageNet validation subset images dir, ILSVRC2012_img_val dir")
    parser.add_argument("--img_cls_file", help="path to file with classes ids for images, download it here:"
                            "https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/img_classes_inception.txt")
    parser.add_argument("--model", help="path to tensorflow model, download it here:"
                                        "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip")
    parser.add_argument("--log", help="path to logging file")
    parser.add_argument("--batch_size", help="size of images in batch", default=1)
    parser.add_argument("--frame_size", help="size of input image", default=224)
    parser.add_argument("--in_blob", help="name for input blob", default='input')
    parser.add_argument("--out_blob", help="name for output blob", default='softmax2')
    args = parser.parse_args()

    data_fetcher = MeanValueFetch(args.frame_size, args.imgs_dir, True)

    frameworks = [TensorflowModel(args.model, args.in_blob, args.out_blob),
                  DnnTfInceptionModel(args.model, '', args.out_blob)]

    acc_eval = ClsAccEvaluation(args.log, args.img_cls_file, args.batch_size)
    acc_eval.process(frameworks, data_fetcher)
