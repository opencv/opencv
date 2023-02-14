#!/usr/bin/env python

from __future__ import print_function
import hashlib
import os
import sys
import tarfile
import requests

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
        self.downloader = kwargs.pop('downloader', None)
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
        print('  expect {}'.format(self.sha))
        sha = hashlib.sha1()
        try:
            with open(self.filename, 'rb') as f:
                while True:
                    buf = f.read(self.BUFSIZE)
                    if not buf:
                        break
                    sha.update(buf)
            print('  actual {}'.format(sha.hexdigest()))
            self.sha_actual = sha.hexdigest()
            return self.sha == self.sha_actual
        except Exception as e:
            print('  catch {}'.format(e))

    def get(self):
        if self.verify():
            print('  hash match - skipping')
            return True

        basedir = os.path.dirname(self.filename)
        if basedir and not os.path.exists(basedir):
            print('  creating directory: ' + basedir)
            os.makedirs(basedir, exist_ok=True)

        if self.archive or self.member:
            assert(self.archive and self.member)
            print('  hash check failed - extracting')
            print('  get {}'.format(self.member))
            self.extract()
        elif self.url:
            print('  hash check failed - downloading')
            print('  get {}'.format(self.url))
            self.download()
        else:
            assert self.downloader
            print('  hash check failed - downloading')
            sz = self.downloader(self.filename)
            print('  size = %.2f Mb' % (sz / (1024.0 * 1024)))

        print(' done')
        print(' file {}'.format(self.filename))
        candidate_verify = self.verify()
        if not candidate_verify:
            self.handle_bad_download()
        return candidate_verify

    def download(self):
        try:
            r = urlopen(self.url, timeout=60)
            self.printRequest(r)
            self.save(r)
        except Exception as e:
            print('  catch {}'.format(e))

    def extract(self):
        try:
            with tarfile.open(self.archive) as f:
                assert self.member in f.getnames()
                self.save(f.extractfile(self.member))
        except Exception as e:
            print('  catch {}'.format(e))

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

    def handle_bad_download(self):
        if os.path.exists(self.filename):
            # rename file for further investigation
            try:
                # NB: using `self.sha_actual` may create unbounded number of files
                rename_target = self.filename + '.invalid'
                # TODO: use os.replace (Python 3.3+)
                try:
                    if os.path.exists(rename_target):  # avoid FileExistsError on Windows from os.rename()
                        os.remove(rename_target)
                finally:
                    os.rename(self.filename, rename_target)
                    print('  renaming invalid file to ' + rename_target)
            except:
                import traceback
                traceback.print_exc()
            finally:
                if os.path.exists(self.filename):
                    print('  deleting invalid file')
                    os.remove(self.filename)


def GDrive(gid):
    def download_gdrive(dst):
        session = requests.Session()  # re-use cookies

        URL = "https://docs.google.com/uc?export=download"
        response = session.get(URL, params = { 'id' : gid }, stream = True)

        def get_confirm_token(response):  # in case of large files
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None
        token = get_confirm_token(response)

        if token:
            params = { 'id' : gid, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        BUFSIZE = 1024 * 1024
        PROGRESS_SIZE = 10 * 1024 * 1024

        sz = 0
        progress_sz = PROGRESS_SIZE
        with open(dst, "wb") as f:
            for chunk in response.iter_content(BUFSIZE):
                if not chunk:
                    continue  # keep-alive

                f.write(chunk)
                sz += len(chunk)
                if sz >= progress_sz:
                    progress_sz += PROGRESS_SIZE
                    print('>', end='')
                    sys.stdout.flush()
        print('')
        return sz
    return download_gdrive


models = [
    Model(
        name='GoogleNet',
        url='http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        sha='405fc5acd08a3bb12de8ee5e23a96bec22f08204',
        filename='bvlc_googlenet.caffemodel'),
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
        name='Enet',  # https://github.com/e-lab/ENet-training
        url='https://www.dropbox.com/s/tdde0mawbi5dugq/Enet-model-best.net?dl=1',
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
        url='https://pjreddie.com/media/files/yolov2-tiny-voc.weights',
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
        name='MobileNet-SSD v1 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
        sha='6157ddb6da55db2da89dd561eceb7f944928e317',
        filename='ssd_mobilenet_v1_coco_2017_11_17.tar.gz'),
    Model(
        name='MobileNet-SSD v1 (TensorFlow)',
        archive='ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
        member='ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb',
        sha='9e4bcdd98f4c6572747679e4ce570de4f03a70e2',
        filename='ssd_mobilenet_v1_coco_2017_11_17.pb'),
    Model(
        name='MobileNet-SSD v2 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        sha='69c93d29e292bc9682396a5c78355b1dfe481b61',
        filename='ssd_mobilenet_v2_coco_2018_03_29.tar.gz'),
    Model(
        name='MobileNet-SSD v2 (TensorFlow)',
        archive='ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        member='ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
        sha='35d571ac314f1d32ae678a857f87cc0ef6b220e8',
        filename='ssd_mobilenet_v2_coco_2018_03_29.pb'),
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
    Model(
        name='Faster-RCNN, InveptionV2 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        sha='c710f25e5c6a3ce85fe793d5bf266d581ab1c230',
        filename='faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'),
    Model(
        name='Faster-RCNN, InveptionV2 (TensorFlow)',
        archive='faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        member='faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
        sha='f2e4bf386b9bb3e25ddfcbbd382c20f417e444f3',
        filename='faster_rcnn_inception_v2_coco_2018_01_28.pb'),
    Model(
        name='ssd_mobilenet_v1_ppn_coco (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
        sha='549ae0fd82c202786abe53c306b191c578599c44',
        filename='ssd_mobilenet_v1_ppn_coco.tar.gz'),
    Model(
        name='ssd_mobilenet_v1_ppn_coco (TensorFlow)',
        archive='ssd_mobilenet_v1_ppn_coco.tar.gz',
        member='ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb',
        sha='7943c51c6305b38173797d4afbf70697cf57ab48',
        filename='ssd_mobilenet_v1_ppn_coco.pb'),
    Model(
        name='mask_rcnn_inception_v2_coco_2018_01_28 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        sha='f8a920756744d0f7ee812b3ec2474979f74ab40c',
        filename='mask_rcnn_inception_v2_coco_2018_01_28.tar.gz'),
    Model(
        name='mask_rcnn_inception_v2_coco_2018_01_28 (TensorFlow)',
        archive='mask_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        member='mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
        sha='c8adff66a1e23e607f57cf1a7cfabad0faa371f9',
        filename='mask_rcnn_inception_v2_coco_2018_01_28.pb'),
    Model(
        name='faster_rcnn_resnet50_coco (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        sha='3066e8dd156b99c4b4d78a2ccd13e33fc263beb7',
        filename='faster_rcnn_resnet50_coco_2018_01_28.tar.gz'),
    Model(
        name='faster_rcnn_resnet50_coco (TensorFlow)',
        archive='faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        member='faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb',
        sha='27feaef9924650299b2ef5d29f041627b6f298b2',
        filename='faster_rcnn_resnet50_coco_2018_01_28.pb'),
    Model(
        name='AlexNet (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz',
        sha='c713be2852472582224fa7395e2ab4641f8b6356',
        filename='bvlc_alexnet.tar.gz'),
    Model(
        name='AlexNet (ONNX)',
        archive='bvlc_alexnet.tar.gz',
        member='bvlc_alexnet/model.onnx',
        sha='b256703f2b125d8681a0a6e5a40a6c9deb7d2b4b',
        filename='onnx/models/alexnet.onnx'),
    Model(
        name='GoogleNet (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_googlenet.tar.gz',
        sha='739732220ba2e3efa88f7c26f13badad9b7514bc',
        filename='bvlc_googlenet.tar.gz'),
    Model(
        name='GoogleNet (ONNX)',
        archive='bvlc_googlenet.tar.gz',
        member='bvlc_googlenet/model.onnx',
        sha='534a16d7e2472f6a9a1925a5ee6c9abc2f5c02b0',
        filename='onnx/models/googlenet.onnx'),
    Model(
        name='CaffeNet (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_caffenet.tar.gz',
        sha='f9f5dd60d4c9172a7e26bd4268eab7ecddb37393',
        filename='bvlc_reference_caffenet.tar.gz'),
    Model(
        name='CaffeNet (ONNX)',
        archive='bvlc_reference_caffenet.tar.gz',
        member='bvlc_reference_caffenet/model.onnx',
        sha='6b2be0cd598914e13b60787c63cba0533723d746',
        filename='onnx/models/caffenet.onnx'),
    Model(
        name='CaffeNet (ONNX)',
        archive='bvlc_reference_caffenet.tar.gz',
        member='bvlc_reference_caffenet/test_data_set_0/input_0.pb',
        sha='e5d6fb75a66ef157023a7fc2f88abdcb371f2f16',
        filename='onnx/data/input_caffenet.pb'),
    Model(
        name='CaffeNet (ONNX)',
        archive='bvlc_reference_caffenet.tar.gz',
        member='bvlc_reference_caffenet/test_data_set_0/output_0.pb',
        sha='eaff902ef71a648aaaeffa495e5fddf2dc0b77c1',
        filename='onnx/data/output_caffenet.pb'),
    Model(
        name='RCNN_ILSVRC13 (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_rcnn_ilsvrc13.tar.gz',
        sha='b1b27a41066c26f824d57e99036dc885459017f0',
        filename='bvlc_reference_rcnn_ilsvrc13.tar.gz'),
    Model(
        name='RCNN_ILSVRC13 (ONNX)',
        archive='bvlc_reference_rcnn_ilsvrc13.tar.gz',
        member='bvlc_reference_rcnn_ilsvrc13/model.onnx',
        sha='fbf174b62a1918bff43c0287e41fdc6017b46256',
        filename='onnx/models/rcnn_ilsvrc13.onnx'),
    Model(
        name='RCNN_ILSVRC13 (ONNX)',
        archive='bvlc_reference_rcnn_ilsvrc13.tar.gz',
        member='bvlc_reference_rcnn_ilsvrc13/test_data_set_0/input_0.pb',
        sha='dcfd587bede888606a7f10e9feadc7f25bed7da4',
        filename='onnx/data/input_rcnn_ilsvrc13.pb'),
    Model(
        name='RCNN_ILSVRC13 (ONNX)',
        archive='bvlc_reference_rcnn_ilsvrc13.tar.gz',
        member='bvlc_reference_rcnn_ilsvrc13/test_data_set_0/output_0.pb',
        sha='e09eea540b93a2f450e32db59e198ca96c3b8637',
        filename='onnx/data/output_rcnn_ilsvrc13.pb'),
    Model(
        name='ZFNet512 (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/zfnet512.tar.gz',
        sha='c040c455c8aac71c8cda57595b698b76449e4ff4',
        filename='zfnet512.tar.gz'),
    Model(
        name='ZFNet512 (ONNX)',
        archive='zfnet512.tar.gz',
        member='zfnet512/model.onnx',
        sha='c32b9ae0bbe65e2ee60f98639b170645000e2c75',
        filename='onnx/models/zfnet512.onnx'),
    Model(
        name='ZFNet512 (ONNX)',
        archive='zfnet512.tar.gz',
        member='zfnet512/test_data_set_0/input_0.pb',
        sha='2dc2c8020edbd84a52f0550d6666c9ae7e93c01f',
        filename='onnx/data/input_zfnet512.pb'),
    Model(
        name='ZFNet512 (ONNX)',
        archive='zfnet512.tar.gz',
        member='zfnet512/test_data_set_0/output_0.pb',
        sha='a74974096088954ca4e4e89bec212c1ac2ab0745',
        filename='onnx/data/output_zfnet512.pb'),
    Model(
        name='VGG16_bn (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.tar.gz',
        sha='60f4685aed632d2ce3b137017cf44ae1a5c55459',
        filename='vgg16-bn.tar.gz'),
    Model(
        name='VGG16_bn (ONNX)',
        archive='vgg16-bn.tar.gz',
        member='vgg16-bn/vgg16-bn.onnx',
        sha='e282e2137f1317d03ca1f2702e9cfddaf847e44d',
        filename='onnx/models/vgg16-bn.onnx'),
    Model(
        name='VGG16_bn (ONNX)',
        archive='vgg16-bn.tar.gz',
        member='vgg16-bn/test_data_set_0/input_0.pb',
        sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
        filename='onnx/data/input_vgg16-bn.pb'),
    Model(
        name='VGG16_bn (ONNX)',
        archive='vgg16-bn.tar.gz',
        member='vgg16-bn/test_data_set_0/output_0.pb',
        sha='418b1a426a2a4105cfd9a77a965ae67dc105891b',
        filename='onnx/data/output_vgg16-bn.pb'),
    Model(
        name='ResNet-18v1 (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz',
        sha='d132be4857d024de9caa21fd5300dee7c063bc35',
        filename='resnet18v1.tar.gz'),
    Model(
        name='ResNet-18v1 (ONNX)',
        archive='resnet18v1.tar.gz',
        member='resnet18v1/resnet18v1.onnx',
        sha='9d96d7142c5ce43aa61ce67124b8eb5530afff4c',
        filename='onnx/models/resnet18v1.onnx'),
    Model(
        name='ResNet-18v1 (ONNX)',
        archive='resnet18v1.tar.gz',
        member='resnet18v1/test_data_set_0/input_0.pb',
        sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
        filename='onnx/data/input_resnet18v1.pb'),
    Model(
        name='ResNet-18v1 (ONNX)',
        archive='resnet18v1.tar.gz',
        member='resnet18v1/test_data_set_0/output_0.pb',
        sha='70e0ad583cf922452ac6e52d882b5127db086a45',
        filename='onnx/data/output_resnet18v1.pb'),
    Model(
        name='ResNet-50v1 (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz',
        sha='a4ac2da7e0024d61fdb80481496ba966b48b9fea',
        filename='resnet50v1.tar.gz'),
    Model(
        name='ResNet-50v1 (ONNX)',
        archive='resnet50v1.tar.gz',
        member='resnet50v1/resnet50v1.onnx',
        sha='06aa26c6de448e11c64cd80cf06f5ab01de2ec9b',
        filename='onnx/models/resnet50v1.onnx'),
    Model(
        name='ResNet-50v1 (ONNX)',
        archive='resnet50v1.tar.gz',
        member='resnet50v1/test_data_set_0/input_0.pb',
        sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
        filename='onnx/data/input_resnet50v1.pb'),
    Model(
        name='ResNet-50v1 (ONNX)',
        archive='resnet50v1.tar.gz',
        member='resnet50v1/test_data_set_0/output_0.pb',
        sha='40deb324ddba7db4117568e1e3911e7a771fb260',
        filename='onnx/data/output_resnet50v1.pb'),
    Model(
        name='ResNet50-Int8 (ONNX)',
        url='https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-12-int8.tar.gz',
        sha='2ff2a58f4a27362ee6234915452e86287cdcf269',
        filename='resnet50-v1-12-int8.tar.gz'),
    Model(
        name='ResNet50-Int8 (ONNX)',
        archive='resnet50-v1-12-int8.tar.gz',
        member='resnet50-v1-12-int8/resnet50-v1-12-int8.onnx',
        sha='5fbeac70e1a3af3253c21e0e4008a784aa61929f',
        filename='onnx/models/resnet50_int8.onnx'),
    Model(
        name='ResNet50-Int8 (ONNX)',
        archive='resnet50-v1-12-int8.tar.gz',
        member='resnet50-v1-12-int8/test_data_set_0/input_0.pb',
        sha='0946521c8afcfea9340390298a41fb11496b3556',
        filename='onnx/data/input_resnet50_int8.pb'),
    Model(
        name='ResNet50-Int8 (ONNX)',
        archive='resnet50-v1-12-int8.tar.gz',
        member='resnet50-v1-12-int8/test_data_set_0/output_0.pb',
        sha='6d45d2f06150e9045631c7928093728b07c8b12d',
        filename='onnx/data/output_resnet50_int8.pb'),
    Model(
        name='ssd_mobilenet_v1_ppn_coco (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
        sha='549ae0fd82c202786abe53c306b191c578599c44',
        filename='ssd_mobilenet_v1_ppn_coco.tar.gz'),
    Model(
        name='ssd_mobilenet_v1_ppn_coco (TensorFlow)',
        archive='ssd_mobilenet_v1_ppn_coco.tar.gz',
        member='ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb',
        sha='7943c51c6305b38173797d4afbf70697cf57ab48',
        filename='ssd_mobilenet_v1_ppn_coco.pb'),
    Model(
        name='ResNet101_DUC_HDC (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/duc/ResNet101_DUC_HDC.tar.gz',
        sha='f8314f381939d01045ac31dbb53d7d35fe3ff9a0',
        filename='ResNet101_DUC_HDC.tar.gz'),
    Model(
        name='ResNet101_DUC_HDC (ONNX)',
        archive='ResNet101_DUC_HDC.tar.gz',
        member='ResNet101_DUC_HDC/ResNet101_DUC_HDC.onnx',
        sha='83f9cefdf3606a37dd4901a925bb9116795dae39',
        filename='onnx/models/resnet101_duc_hdc.onnx'),
    Model(
        name='ResNet101_DUC_HDC (ONNX)',
        archive='ResNet101_DUC_HDC.tar.gz',
        member='ResNet101_DUC_HDC/test_data_set_0/input_0.pb',
        sha='099d0e32742a2fa6a69c329f1bff699fb7266b33',
        filename='onnx/data/input_resnet101_duc_hdc.pb'),
    Model(
        name='ResNet101_DUC_HDC (ONNX)',
        archive='ResNet101_DUC_HDC.tar.gz',
        member='ResNet101_DUC_HDC/test_data_set_0/output_0.pb',
        sha='3713a21bb7228d3179721810bb72565aebee7033',
        filename='onnx/data/output_resnet101_duc_hdc.pb'),
    Model(
        name='TinyYolov2 (ONNX)',
        url='https://www.cntk.ai/OnnxModels/tiny_yolov2/opset_1/tiny_yolov2.tar.gz',
        sha='b9102abb8fa6f51368119b52146c30189353164a',
        filename='tiny_yolov2.tar.gz'),
    Model(
        name='TinyYolov2 (ONNX)',
        archive='tiny_yolov2.tar.gz',
        member='tiny_yolov2/model.onnx',
        sha='433fecbd32ac8b9be6f5ee10c39dcecf9dc5c151',
        filename='onnx/models/tiny_yolo2.onnx'),
    Model(
        name='TinyYolov2 (ONNX)',
        archive='tiny_yolov2.tar.gz',
        member='tiny_yolov2/test_data_set_0/input_0.pb',
        sha='a0412fde98ca21d726c0c86ef007c11aa4678e3c',
        filename='onnx/data/input_tiny_yolo2.pb'),
    Model(
        name='TinyYolov2 (ONNX)',
        archive='tiny_yolov2.tar.gz',
        member='tiny_yolov2/test_data_set_0/output_0.pb',
        sha='f9be0446cac76fe38bb23cb09ed23c317907f505',
        filename='onnx/data/output_tiny_yolo2.pb'),
    Model(
        name='CNN Mnist (ONNX)',
        url='https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz',
        sha='8bcd3372e44bd95dc8a211bc31fb3025d8edf9f9',
        filename='mnist.tar.gz'),
    Model(
        name='CNN Mnist (ONNX)',
        archive='mnist.tar.gz',
        member='mnist/model.onnx',
        sha='e4fb4914cd1d9e0faed3294e5cecfd1847339763',
        filename='onnx/models/cnn_mnist.onnx'),
    Model(
        name='CNN Mnist (ONNX)',
        archive='mnist.tar.gz',
        member='mnist/test_data_set_0/input_0.pb',
        sha='023f6c94951ab386957964e39727aa43d8c45ea8',
        filename='onnx/data/input_cnn_mnist.pb'),
    Model(
        name='CNN Mnist (ONNX)',
        archive='mnist.tar.gz',
        member='mnist/test_data_set_0/output_0.pb',
        sha='79f3028d97df835b058849d357e06d4c0bfcf5b3',
        filename='onnx/data/output_cnn_mnist.pb'),
    Model(
        name='MobileNetv2 (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz',
        sha='7f1429a8e1f3118a05943ff3ed54dbc9eb55691a',
        filename='mobilenetv2-1.0.tar.gz'),
    Model(
        name='MobileNetv2 (ONNX)',
        archive='mobilenetv2-1.0.tar.gz',
        member='mobilenetv2-1.0/mobilenetv2-1.0.onnx',
        sha='80c97941c3ce34d05bc3d3c9d6e04c44c15906bc',
        filename='onnx/models/mobilenetv2.onnx'),
    Model(
        name='MobileNetv2 (ONNX)',
        archive='mobilenetv2-1.0.tar.gz',
        member='mobilenetv2-1.0/test_data_set_0/input_0.pb',
        sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
        filename='onnx/data/input_mobilenetv2.pb'),
    Model(
        name='MobileNetv2 (ONNX)',
        archive='mobilenetv2-1.0.tar.gz',
        member='mobilenetv2-1.0/test_data_set_0/output_0.pb',
        sha='7e58c6faca7fc3b844e18364ae92606aa3f0b18e',
        filename='onnx/data/output_mobilenetv2.pb'),
    Model(
        name='LResNet100E-IR (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz',
        sha='b1178813b705d9d44ed806aa442f0b1cb11aea0a',
        filename='resnet100.tar.gz'),
    Model(
        name='LResNet100E-IR (ONNX)',
        archive='resnet100.tar.gz',
        member='resnet100/resnet100.onnx',
        sha='d307e426cf55cddf9f9292b5ffabb474eec93638',
        filename='onnx/models/LResNet100E_IR.onnx'),
    Model(
        name='LResNet100E-IR (ONNX)',
        archive='resnet100.tar.gz',
        member='resnet100/test_data_set_0/input_0.pb',
        sha='d80a849e000907734bd0061ba570f734784f7d38',
        filename='onnx/data/input_LResNet100E_IR.pb'),
    Model(
        name='LResNet100E-IR (ONNX)',
        archive='resnet100.tar.gz',
        member='resnet100/test_data_set_0/output_0.pb',
        sha='f54c73699d00b18b5c40e4ea895b1e88e7f8dea3',
        filename='onnx/data/output_LResNet100E_IR.pb'),
    Model(
        name='Emotion FERPlus (ONNX)',
        url='https://www.cntk.ai/OnnxModels/emotion_ferplus/opset_7/emotion_ferplus.tar.gz',
        sha='9ff80899c0cd468999db5d8ffde98780ef85455e',
        filename='emotion_ferplus.tar.gz'),
    Model(
        name='Emotion FERPlus (ONNX)',
        archive='emotion_ferplus.tar.gz',
        member='emotion_ferplus/model.onnx',
        sha='2ef5b3a6404a5feb8cc396d66c86838c4c750a7e',
        filename='onnx/models/emotion_ferplus.onnx'),
    Model(
        name='Emotion FERPlus (ONNX)',
        archive='emotion_ferplus.tar.gz',
        member='emotion_ferplus/test_data_set_0/input_0.pb',
        sha='29621536528116fc12f02bc81c7265f7ffe7c8bb',
        filename='onnx/data/input_emotion_ferplus.pb'),
    Model(
        name='Emotion FERPlus (ONNX)',
        archive='emotion_ferplus.tar.gz',
        member='emotion_ferplus/test_data_set_0/output_0.pb',
        sha='54f7892240d2d9298f5a8064a46fc3a8987015a5',
        filename='onnx/data/output_emotion_ferplus.pb'),
    Model(
        name='Squeezenet (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz',
        sha='57348321d4d460c07c41af814def3abe728b3a03',
        filename='squeezenet.tar.gz'),
    Model(
        name='Squeezenet (ONNX)',
        archive='squeezenet.tar.gz',
        member='squeezenet/model.onnx',
        sha='c3f272e672fa64a75fb4a2e48dd2ca25fcc76c49',
        filename='onnx/models/squeezenet.onnx'),
    Model(
        name='Squeezenet (ONNX)',
        archive='squeezenet.tar.gz',
        member='squeezenet/test_data_set_0/input_0.pb',
        sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
        filename='onnx/data/input_squeezenet.pb'),
    Model(
        name='Squeezenet (ONNX)',
        archive='squeezenet.tar.gz',
        member='squeezenet/test_data_set_0/output_0.pb',
        sha='e4f3c0c989cc7025ca94759492508d8f4ef3287b',
        filename='onnx/data/output_squeezenet.pb'),
    Model(
        name='DenseNet121 (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/densenet121.tar.gz',
        sha='338b70e871e73b0550fc8ccc0863b8382e90e8e5',
        filename='densenet121.tar.gz'),
    Model(
        name='DenseNet121 (ONNX)',
        archive='densenet121.tar.gz',
        member='densenet121/model.onnx',
        sha='2874279d0f56f15f4e7e9208526c1b35d85d5ad1',
        filename='onnx/models/densenet121.onnx'),
    Model(
        name='DenseNet121 (ONNX)',
        archive='densenet121.tar.gz',
        member='densenet121/test_data_set_0/input_0.pb',
        sha='d6146a5b08a85309a3b8ada313ae5887c2aa7e3e',
        filename='onnx/data/input_densenet121.pb'),
    Model(
        name='DenseNet121 (ONNX)',
        archive='densenet121.tar.gz',
        member='densenet121/test_data_set_0/output_0.pb',
        sha='f1fd0d5e8d48aff3df2c5c809ea24e982d72028e',
        filename='onnx/data/output_densenet121.pb'),
    Model(
        name='Inception v1 (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz',
        sha='94ecb2bd1426704dca578dc746e3c27bedf22352',
        filename='inception_v1.tar.gz'),
    Model(
        name='Inception v1 (ONNX)',
        archive='inception_v1.tar.gz',
        member='inception_v1/model.onnx',
        sha='f45896d8d35248a62ea551db922d982a90214517',
        filename='onnx/models/inception_v1.onnx'),
    Model(
        name='Inception v1 (ONNX)',
        archive='inception_v1.tar.gz',
        member='inception_v1/test_data_set_0/input_0.pb',
        sha='7ec7a82aa2fecd2c875b7b198ecd9a428bc9f462',
        filename='onnx/data/input_inception_v1.pb'),
    Model(
        name='Inception v1 (ONNX)',
        archive='inception_v1.tar.gz',
        member='inception_v1/test_data_set_0/output_0.pb',
        sha='870a30306bd2b82d5393a0ff5570b022681ef7b6',
        filename='onnx/data/output_inception_v1.pb'),
    Model(
        name='Inception v2 (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v2.tar.gz',
        sha='d07a442a84d939232c37c976fd8d624fa9f82026',
        filename='inception_v2.tar.gz'),
    Model(
        name='Inception v2 (ONNX)',
        archive='inception_v2.tar.gz',
        member='inception_v2/model.onnx',
        sha='cfa84f36bcae8910e0875872383991cb0c3b9a80',
        filename='onnx/models/inception_v2.onnx'),
    Model(
        name='Inception v2 (ONNX)',
        archive='inception_v2.tar.gz',
        member='inception_v2/test_data_set_0/input_0.pb',
        sha='f4ed6d838c20dbfc3bcf6abfd23c78d74892a5fe',
        filename='onnx/data/input_inception_v2.pb'),
    Model(
        name='Inception v2 (ONNX)',
        archive='inception_v2.tar.gz',
        member='inception_v2/test_data_set_0/output_0.pb',
        sha='cb75fb6db82290c49879380ce72c71e17eda76d0',
        filename='onnx/data/output_inception_v2.pb'),
    Model(
        name='Shufflenet (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz',
        sha='c99afcb7fcc809c0688cc99cb3709a052fde1de7',
        filename='shufflenet.tar.gz'),
    Model(
        name='Shufflenet (ONNX)',
        archive='shufflenet.tar.gz',
        member='shufflenet/model.onnx',
        sha='a781faf9f1fe6d001cd7b6b5a7d1a228da0ff17b',
        filename='onnx/models/shufflenet.onnx'),
    Model(
        name='Shufflenet (ONNX)',
        archive='shufflenet.tar.gz',
        member='shufflenet/test_data_set_0/input_0.pb',
        sha='27d31be9a084c1d1d1eacbd766f4c43d59352a07',
        filename='onnx/data/input_shufflenet.pb'),
    Model(
        name='Shufflenet (ONNX)',
        archive='shufflenet.tar.gz',
        member='shufflenet/test_data_set_0/output_0.pb',
        sha='6a33ed6ccef4c69a27a3993363c3f854d0f79bb0',
        filename='onnx/data/output_shufflenet.pb'),
    Model(
        name='ResNet-34_kinetics (ONNX)', # https://github.com/kenshohara/video-classification-3d-cnn-pytorch
        url='https://www.dropbox.com/s/065l4vr8bptzohb/resnet-34_kinetics.onnx?dl=1',
        sha='88897629e4abb0fddef939f0c2d668a4edeb0788',
        filename='resnet-34_kinetics.onnx'),
    Model(
        name='Alexnet Facial Keypoints (ONNX)', # https://github.com/ismalakazel/Facial-Keypoint-Detection
        url='https://drive.google.com/uc?export=dowload&id=1etGXT9WQK1KjDkJ0pUTH-CaHHva4p9cY',
        sha='e1b82b56b59ab96b50189e1b39487d91d4fa0eea',
        filename='onnx/models/facial_keypoints.onnx'),
    Model(
        name='LightWeight Human Pose Estimation (ONNX)', # https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
        url='https://drive.google.com/uc?export=dowload&id=1--Ij_gIzCeNA488u5TA4FqWMMdxBqOji',
        sha='5960f7aef233d75f8f4020be1fd911b2d93fbffc',
        filename='onnx/models/lightweight_pose_estimation_201912.onnx'),
    Model(
        name='EfficientDet-D0', # https://github.com/google/automl
        url='https://www.dropbox.com/s/9mqp99fd2tpuqn6/efficientdet-d0.pb?dl=1',
        sha='f178cc17b44e3ed2f3956a0adc1800a7d2a3b3ae',
        filename='efficientdet-d0.pb'),
    Model(
        name='YOLOv4',  # https://github.com/opencv/opencv/issues/17148
        url="https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
        sha='0143deb6c46fcc7f74dd35bf3c14edc3784e99ee',
        filename='yolov4.weights'),
    Model(
        name='YOLOv4-tiny-2020-12',  # https://github.com/opencv/opencv/issues/17148
        url='https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights',
        sha='451caaab22fb9831aa1a5ee9b5ba74a35ffa5dcb',
        filename='yolov4-tiny-2020-12.weights'),
    Model(
        name='YOLOv4x-mish',  # https://github.com/opencv/opencv/issues/18975
        url='https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights',
        sha='a6f2879af2241de2e9730d317a55db6afd0af00b',
        filename='yolov4x-mish.weights'),
    Model(
        name='GSOC2016-GOTURN',  # https://github.com/opencv/opencv_contrib/issues/941
        downloader=GDrive('1j4UTqVE4EGaUFiK7a5I_CYX7twO9c5br'),
        sha='49776d262993c387542f84d9cd16566840404f26',
        filename='gsoc2016-goturn/goturn.caffemodel'),
    Model(
        name='DaSiamRPM Tracker network (ONNX)',
        url='https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=1',
        sha='91b774fce7df4c0e4918469f0f482d9a27d0e2d4',
        filename='onnx/models/dasiamrpn_model.onnx'),
    Model(
        name='DaSiamRPM Tracker kernel_r1 (ONNX)',
        url='https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=1',
        sha='bb64620a54348657133eb28be2d3a2a8c76b84b3',
        filename='onnx/models/dasiamrpn_kernel_r1.onnx'),
    Model(
        name='DaSiamRPM Tracker kernel_cls1 (ONNX)',
        url='https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=1',
        sha='e9ccd270ce8059bdf7ed0d1845c03ef4a951ee0f',
        filename='onnx/models/dasiamrpn_kernel_cls1.onnx'),
    Model(
        name='crnn',
        url='https://drive.google.com/uc?export=dowload&id=1ooaLR-rkTl8jdpGy1DoQs0-X0lQsB6Fj',
        sha='270d92c9ccb670ada2459a25977e8deeaf8380d3',
        filename='onnx/models/crnn.onnx'),
    Model(
        name='DB_TD500_resnet50',
        url='https://drive.google.com/uc?export=dowload&id=19YWhArrNccaoSza0CfkXlA8im4-lAGsR',
        sha='1b4dd21a6baa5e3523156776970895bd3db6960a',
        filename='onnx/models/DB_TD500_resnet50.onnx'),
    Model(
        name='YuNet',
        url='https://github.com/ShiqiYu/libfacedetection.train/raw/1688402dbd9b9fc4a3a6793810f558b7407ff384/tasks/task1/onnx/yunet_120x160.onnx',
        sha='dfe691ae0c8e38d39d1a437e3f7e5fda7b256bdd',
        filename='onnx/models/yunet-202202.onnx'),
    Model(
        name='face_recognizer_fast',
        url='https://drive.google.com/uc?export=dowload&id=1ClK9WiB492c5OZFKveF3XiHCejoOxINW',
        sha='12ff8b1f5c8bff62e8dd91eabdacdfc998be255e',
        filename='onnx/models/face_recognizer_fast.onnx'),
    Model(
        name='MobileNetv2 FP16 (ONNX)',
        url='https://github.com/zihaomu/zihaomu/files/9393786/mobilenetv2_fp16_v7.tar.gz',
        sha='018d42b1b1283e6025a0455deffe9f0e9930e839',
        filename='mobilenetv2_fp16_v7.tar.gz'),
    Model(
        name='MobileNetv2 FP16 (ONNX)',
        archive='mobilenetv2_fp16_v7.tar.gz',
        member='mobilenetv2_fp16_v7/mobilenetv2_fp16.onnx',
        sha='ab9352de8e07b798417922f23e97c8488bd50017',
        filename='onnx/models/mobilenetv2_fp16.onnx'),
    Model(
        name='MobileNetv2 FP16 (ONNX)',
        archive='mobilenetv2_fp16_v7.tar.gz',
        member='mobilenetv2_fp16_v7/input_mobilenetv2_fp16.npy',
        sha='cbb97c31abc07ff8c68f5028c634d79f8b83b560',
        filename='onnx/data/input_mobilenetv2_fp16.npy'),
    Model(
        name='MobileNetv2 FP16 (ONNX)',
        archive='mobilenetv2_fp16_v7.tar.gz',
        member='mobilenetv2_fp16_v7/output_mobilenetv2_fp16.npy',
        sha='397560616c47b847340cec9561e12a13b29ae32e',
        filename='onnx/data/output_mobilenetv2_fp16.npy'),
    Model(
        name='wechat_qr_detect',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt',
        sha='a6936962139282d300ebbf15a54c2aa94b144bb7',
        filename='wechat_2021-01/detect.prototxt'),
    Model(
        name='wechat_qr_detect',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel',
        sha='d587623a055cbd58a648de62a8c703c7abb05f6d',
        filename='wechat_2021-01/detect.caffemodel'),
    Model(
        name='wechat_super_resolution',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt',
        sha='39e1f1031c842766f1cc126615fea8e8256facd2',
        filename='wechat_2021-01/sr.prototxt'),
    Model(
        name='wechat_super_resolution',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel',
        sha='2b181b55d1d7af718eaca6cabdeb741217b64c73',
        filename='wechat_2021-01/sr.caffemodel'),
    Model(
        name='yolov7_not_simplified',
        downloader=GDrive('1rm3mIqjJNu0xPTCjMKnXccspazV1B2zv'),
        sha='fcd0fa401c83bf2b29e18239a9c2c989c9b8669d',
        filename='onnx/models/yolov7_not_simplified.onnx'),
    Model(
        name='NanoTrackV1 (ONNX)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrack/models/nanotrack_backbone_sim.onnx',
        sha='9b083a2dbe10dcfe17e694879aa6749302a5888f',
        filename='onnx/models/nanotrack_backbone_sim.onnx'),
    Model(
        name='NanoTrackV1 (ONNX)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrack/models/nanotrack_head_sim.onnx',
        sha='8fa668893b27b726f9cab6695846b4690650a199',
        filename='onnx/models/nanotrack_head_sim.onnx'),
    Model(
        name='NanoTrackV2 (ONNX)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrackV2/models/nanotrack_backbone_sim_v2.onnx',
        sha='6e773a364457b78574f9f63a23b0659ee8646f8f',
        filename='onnx/models/nanotrack_backbone_sim_v2.onnx'),
    Model(
        name='NanoTrackV2 (ONNX)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrackV2/models/nanotrack_head_sim_v2.onnx',
        sha='39f168489671700cf739e402dfc67d41ce648aef',
        filename='onnx/models/nanotrack_head_sim_v2.onnx'),
    Model(
        name='Face Mesh (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite?generation=1668295060280094',
        sha='eb01d1d88c833aaea64c880506da72e4a4f43154',
        filename='tflite/face_landmark.tflite'),
    Model(
        name='Face Detection (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite?generation=1661875748538815',
        sha='e8f749fafc23bb88daac85bc9f7e0698436f29a0',
        filename='tflite/face_detection_short_range.tflite'),
    Model(
        name='Selfie Segmentation (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite?generation=1661875931201364',
        sha='8d497f51bd678fa5fb95c3871be72eb5d722b831',
        filename='tflite/selfie_segmentation.tflite'),
    Model(
        name='Hair Segmentation (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite?generation=1661875756623461',
        sha='bba28400dfc264b1ed7ee95df718fada1879644d',
        filename='tflite/hair_segmentation.tflite'),
]

# Note: models will be downloaded to current working directory
#       expected working directory is <testdata>/dnn
if __name__ == '__main__':

    selected_model_name = None
    if len(sys.argv) > 1:
        selected_model_name = sys.argv[1]
        print('Model: ' + selected_model_name)

    failedModels = []
    for m in models:
        print(m)
        if selected_model_name is not None and not m.name.startswith(selected_model_name):
            continue
        if not m.get():
            failedModels.append(m.filename)

    if failedModels:
        print("Following models have not been downloaded:")
        for f in failedModels:
            print("* {}".format(f))
        exit(15)
