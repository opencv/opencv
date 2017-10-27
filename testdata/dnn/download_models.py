#!/usr/bin/env python

from __future__ import print_function
import hashlib
import sys
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


class Model:
    MB = 1024*1024
    BUFSIZE = 10*MB

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.url = kwargs.pop('url')
        self.filename = kwargs.pop('filename')
        self.sha = kwargs.pop('sha', None)

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

    def download(self):
        try:
            if self.verify():
                print('  hash match - skipping download')
                return
        except Exception as e:
            print('  catch {}'.format(e))
        print('  hash check failed - downloading')
        print('  get {}'.format(self.url))
        r = urlopen(self.url)
        self.printRequest(r)
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
            print(' done')
        print(' file {}'.format(self.filename))
        self.verify()


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
]

# Note: models will be downloaded to current working directory
#       expected working directory is opencv_extra/testdata/dnn
if __name__ == '__main__':
    for m in models:
        print(m)
        m.download()
