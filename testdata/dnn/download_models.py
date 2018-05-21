#!/usr/bin/env python

from __future__ import print_function
import hashlib
import sys
import tarfile
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


class Model:
    MB = 1024*1024
    BUFSIZE = 10*MB

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.url = kwargs.pop('url', None)
        self.filename = kwargs.pop('filename')
        self.sha = kwargs.pop('sha', None)
        self.archive = kwargs.pop('archive', None)
        self.member = kwargs.pop('member', None)

    def __str__(self):
        return 'Model <{}>'.format(self.name)

    def printRequest(self, r):
        def getMB(r):
            d = dict(r.info())
            for c in ['content-length', 'Content-Length']:
                if c in d:
                    return int(d[c]) / self.MB
            return '<unknown>'
        print('  {} {} [{} Mb]'.format(r.getcode(), r.msg, getMB(r)))

    def verify(self):
        if not self.sha:
            return False
        print('  expect {}'.format(m.sha))
        sha = hashlib.sha1()
        with open(self.filename, 'rb') as f:
            while True:
                buf = f.read(self.BUFSIZE)
                if not buf:
                    break
                sha.update(buf)
        print('  actual {}'.format(sha.hexdigest()))
        return self.sha == sha.hexdigest()

    def get(self):
        try:
            if self.verify():
                print('  hash match - skipping')
                return
        except Exception as e:
            print('  catch {}'.format(e))

        if self.archive or self.member:
            assert(self.archive and self.member)
            print('  hash check failed - extracting')
            print('  get {}'.format(self.member))
            self.extract()
        else:
            assert(self.url)
            print('  hash check failed - downloading')
            print('  get {}'.format(self.url))
            self.download()

        print(' done')
        print(' file {}'.format(self.filename))
        self.verify()

    def download(self):
        r = urlopen(self.url)
        self.printRequest(r)
        self.save(r)

    def extract(self):
        with tarfile.open(self.archive) as f:
            assert self.member in f.getnames()
            self.save(f.extractfile(self.member))

    def save(self, r):
        with open(self.filename, 'wb') as f:
            print('  progress ', end='')
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()

models = [
    Model(
        name='GoogleNet',
        url='http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        sha='405fc5acd08a3bb12de8ee5e23a96bec22f08204',
        filename='bvlc_googlenet.caffemodel'),
    Model(
        name='VGG16',
        url='http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel',
        sha='9363e1f6d65f7dba68c4f27a1e62105cdf6c4e24',
        filename='VGG_ILSVRC_16_layers.caffemodel'),
    Model(
        name='voc-fcn32s',
        url='http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel',
        sha='05eb922d7829c39448f57e5ab9d8cd75d6c0be6d',
        filename='fcn32s-heavy-pascal.caffemodel'),
    Model(
        name='Alexnet',
        url='http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
        sha='9116a64c0fbe4459d18f4bb6b56d647b63920377',
        filename='bvlc_alexnet.caffemodel'),
    Model(
        name='Inception',
        url='https://github.com/petewarden/tf_ios_makefile_example/raw/master/data/tensorflow_inception_graph.pb',
        sha='c8a5a000ee8d8dd75886f152a50a9c5b53d726a5',
        filename='tensorflow_inception_graph.pb'),
    Model(
        name='Enet',
        url='https://www.dropbox.com/sh/dywzk3gyb12hpe5/AABoUwqQGWvClUu27Z1EWeu9a/model-best.net?dl=1',
        sha='b4123a73bf464b9ebe9cfc4ab9c2d5c72b161315',
        filename='Enet-model-best.net'),
    Model(
        name='Fcn',
        url='http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel',
        sha='c449ea74dd7d83751d1357d6a8c323fcf4038962',
        filename='fcn8s-heavy-pascal.caffemodel'),
    Model(
        name='Ssd_vgg16',
        url='https://www.dropbox.com/s/8apyk3uzk2vl522/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel?dl=1',
        sha='0fc294d5257f3e0c8a3c5acaa1b1f6a9b0b6ade0',
        filename='VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel'),
    Model(
        name='ResNet50',
        url='https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117895&authkey=%21AAFW2%2DFVoxeVRck',
        sha='b7c79ccc21ad0479cddc0dd78b1d20c4d722908d',
        filename='ResNet-50-model.caffemodel'),
    Model(
        name='SqueezeNet_v1.1',
        url='https://raw.githubusercontent.com/DeepScale/SqueezeNet/b5c3f1a23713c8b3fd7b801d229f6b04c64374a5/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel',
        sha='3397f026368a45ae236403ccc81cfcbe8ebe1bd0',
        filename='squeezenet_v1.1.caffemodel'),
    Model(
        name='MobileNet-SSD',  # https://github.com/chuanqi305/MobileNet-SSD
        url='https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc',
        sha='994d30a8afaa9e754d17d2373b2d62a7dfbaaf7a',
        filename='MobileNetSSD_deploy.caffemodel'),
    Model(
        name='MobileNet-SSD',
        url='https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/daef68a6c2f5fbb8c88404266aa28180646d17e0/MobileNetSSD_deploy.prototxt',
        sha='d77c9cf09619470d49b82a9dd18704813a2043cd',
        filename='MobileNetSSD_deploy.prototxt'),
    Model(
        name='OpenFace',  # https://github.com/cmusatyalab/openface
        url='https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7',
        sha='ac8161a4376fb5a79ceec55d85bbb57ef81da9fe',
        filename='openface_nn4.small2.v1.t7'),
    Model(
        name='YoloV2voc',  # https://pjreddie.com/darknet/yolo/
        url='https://pjreddie.com/media/files/yolo-voc.weights',
        sha='1cc1a7f8ad12d563d85b76e9de025dc28ac397bb',
        filename='yolo-voc.weights'),
    Model(
        name='TinyYoloV2voc',  # https://pjreddie.com/darknet/yolo/
        url='https://pjreddie.com/media/files/tiny-yolo-voc.weights',
        sha='24b4bd049fc4fa5f5e95f684a8967e65c625dff9',
        filename='tiny-yolo-voc.weights'),
    Model(
        name='DenseNet-121',  # https://github.com/shicai/DenseNet-Caffe
        url='https://drive.google.com/uc?export=download&id=0B7ubpZO7HnlCcHlfNmJkU2VPelE',
        sha='02b520138e8a73c94473b05879978018fefe947b',
        filename='DenseNet_121.caffemodel'),
    Model(
        name='DenseNet-121',
        url='https://raw.githubusercontent.com/shicai/DenseNet-Caffe/master/DenseNet_121.prototxt',
        sha='4922099342af5993d9d09f63081c8a392f3c1cc6',
        filename='DenseNet_121.prototxt'),
    Model(
        name='Fast-Neural-Style',
        url='http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/starry_night.t7',
        sha='5b5e115253197b84d6c6ece1dafe6c15d7105ca6',
        filename='fast_neural_style_eccv16_starry_night.t7'),
    Model(
        name='Fast-Neural-Style',
        url='http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/feathers.t7',
        sha='9838007df750d483b5b5e90b92d76e8ada5a31c0',
        filename='fast_neural_style_instance_norm_feathers.t7'),
    Model(
        name='MobileNet-SSD (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz',
        sha='a88a18cca9fe4f9e496d73b8548bfd157ad286e2',
        filename='ssd_mobilenet_v1_coco_11_06_217.tar.gz'),
    Model(
        name='MobileNet-SSD (TensorFlow)',
        archive='ssd_mobilenet_v1_coco_11_06_217.tar.gz',
        member='ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb',
        sha='aaf36f068fab10359eadea0bc68388d96cf68139',
        filename='ssd_mobilenet_v1_coco.pb'),
    Model(
        name='Colorization',
        url='https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt',
        sha='f528334e386a69cbaaf237a7611d833bef8e5219',
        filename='colorization_deploy_v2.prototxt'),
    Model(
        name='Colorization',
        url='http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel',
        sha='21e61293a3fa6747308171c11b6dd18a68a26e7f',
        filename='colorization_release_v2.caffemodel'),
    Model(
        name='Face_detector',
        url='https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        sha='006baf926232df6f6332defb9c24f94bb9f3764e',
        filename='opencv_face_detector.prototxt'),
    Model(
        name='Face_detector',
        url='https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
        sha='15aa726b4d46d9f023526d85537db81cbc8dd566',
        filename='opencv_face_detector.caffemodel'),
    Model(
        name='Face_detector (FP16)',
        url='https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel',
        sha='31fc22bfdd907567a04bb45b7cfad29966caddc1',
        filename='opencv_face_detector_fp16.caffemodel'),
    Model(
        name='Face_detector (UINT8)',
        url='https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb',
        sha='4f2fdf6f231d759d7bbdb94353c5a68690f3d2ae',
        filename='opencv_face_detector_uint8.pb'),
    Model(
        name='InceptionV2-SSD (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz',
        sha='b9546dcd1ba99282b5bfa81c460008c885ca591b',
        filename='ssd_inception_v2_coco_2017_11_17.tar.gz'),
    Model(
        name='InceptionV2-SSD (TensorFlow)',
        archive='ssd_inception_v2_coco_2017_11_17.tar.gz',
        member='ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb',
        sha='554a75594e9fd1ccee291b3ba3f1190b868a54c9',
        filename='ssd_inception_v2_coco_2017_11_17.pb'),
    Model(
        name='Faster-RCNN',  # https://github.com/rbgirshick/py-faster-rcnn
        url='https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0',
        sha='51bca62727c3fe5d14b66e9331373c1e297df7d1',
        filename='faster_rcnn_models.tgz'),
    Model(
        name='Faster-RCNN VGG16',
        archive='faster_rcnn_models.tgz',
        member='faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel',
        sha='dd099979468aafba21f3952718a9ceffc7e57699',
        filename='VGG16_faster_rcnn_final.caffemodel'),
    Model(
        name='Faster-RCNN ZF',
        archive='faster_rcnn_models.tgz',
        member='faster_rcnn_models/ZF_faster_rcnn_final.caffemodel',
        sha='7af886686f149622ed7a41c08b96743c9f4130f5',
        filename='ZF_faster_rcnn_final.caffemodel'),
    Model(
        name='R-FCN',  # https://github.com/YuwenXiong/py-R-FCN
        url='https://onedrive.live.com/download?cid=10B28C0E28BF7B83&resid=10B28C0E28BF7B83%215317&authkey=%21AIeljruhoLuail8',
        sha='bb3180da68b2b71494f8d3eb8f51b2d47467da3e',
        filename='rfcn_models.tar.gz'),
    Model(
        name='R-FCN ResNet-50',
        archive='rfcn_models.tar.gz',
        member='rfcn_models/resnet50_rfcn_final.caffemodel',
        sha='e00beca7af2790801efb1724d77bddba89e7081c',
        filename='resnet50_rfcn_final.caffemodel'),
    Model(
        name='OpenPose/pose/coco',  # https://github.com/CMU-Perceptual-Computing-Lab/openpose
        url='http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel',
        sha='ac7e97da66f3ab8169af2e601384c144e23a95c1',
        filename='openpose_pose_coco.caffemodel'),
    Model(
        name='OpenPose/pose/mpi',  # https://github.com/CMU-Perceptual-Computing-Lab/openpose
        url='http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel',
        sha='a344f4da6b52892e44a0ca8a4c68ee605fc611cf',
        filename='openpose_pose_mpi.caffemodel'),
    Model(
        name='YOLOv3',  # https://pjreddie.com/darknet/yolo/
        url='https://pjreddie.com/media/files/yolov3.weights',
        sha='520878f12e97cf820529daea502acca380f1cb8e',
        filename='yolov3.weights'),
    Model(
        name='EAST',  # https://github.com/argman/EAST (a TensorFlow model), https://arxiv.org/abs/1704.03155v2 (a paper)
        url='https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1',
        sha='3ca8233d6edd748f7ed23246c8ca24cbf696bb94',
        filename='frozen_east_text_detection.tar.gz'),
    Model(
        name='EAST',
        archive='frozen_east_text_detection.tar.gz',
        member='frozen_east_text_detection.pb',
        sha='fffabf5ac36f37bddf68e34e84b45f5c4247ed06',
        filename='frozen_east_text_detection.pb'),
]

# Note: models will be downloaded to current working directory
#       expected working directory is opencv_extra/testdata/dnn
if __name__ == '__main__':
    for m in models:
        print(m)
        m.get()
