#!/usr/bin/env python

import sys
import os
import hashlib
import argparse

CUR_DIR = os.getcwd()
# This directory contains result git-lfs cache
CACHE_DIR = CUR_DIR + '/.cache/onnx_models/'

class Model:
    MB = 1024*1024
    BUFSIZE = 10*MB
    # This directory contains large ONNX models that will be used
    MODELS_DIR = CUR_DIR + '/onnx_models/'

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.filepath = kwargs.pop('filepath')
        self.sha = kwargs.pop('sha')
        self.model_path = self.filepath + self.name + '.onnx'

    def __str__(self):
        return '[Model]: <{}>'.format(self.name + '\n          '+ self.filepath)

    def sys_lfs_call(self):
        return 'cd .cache/onnx_models/ && git lfs pull --include=/' + str(self.model_path) + ' --exclude="" '

    def verify(self, path):
        if not os.path.exists(path):
            return False
        if not self.sha:
            return False
        print('[Info]: Verifying file:')
        print('        Expected sha: {}'.format(self.sha))
        sha = hashlib.sha1()
        try:
            with open(path, 'rb') as f:
                while True:
                    buf = f.read(self.BUFSIZE)
                    if not buf:
                        break
                    sha.update(buf)
            print('        Actual sha: {}'.format(sha.hexdigest()))
            return self.sha == sha.hexdigest()
        except Exception as e:
            print('[Excn]: Catch {}'.format(e))

    def create_dir(self):
        model_dir_path = self.MODELS_DIR + self.filepath
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        else:
            print('[Warn]: Directory already contains a folder for {} - skipping'.format(self.name))

    def download(self):
        print('______________________{}______________________'.format(self.name))
        print('[Info]: Creating directory for {}'.format(self.name))
        self.create_dir()
        if self.verify(self.MODELS_DIR + self.model_path):
            print('[Warn]: Hash match - skipping')
            return True
        print('[Info]: Downloading model')
        # Pull large model file
        os.system(self.sys_lfs_call())
        if self.verify(CACHE_DIR + self.model_path):
        # Move large model file to onnx_models
            os.replace(CACHE_DIR + self.model_path, self.MODELS_DIR + self.model_path)
            return True
        return False

def download_cache():
    print('______________________{}______________________'.format('Download cache'))
    if not os.path.exists(CACHE_DIR):
        print('[Info]: Cloning onnx_models repository from https://github.com/onnx/models.git')
        os.system('git clone --recursive https://github.com/onnx/models.git .cache/onnx_models/')
    else:
        # pulling possible changes
        os.system('cd .cache/onnx_models/ && git checkout master && git pull https://github.com/onnx/models.git')
        print('[Warn]: Directory already contains the ".cache/onnx_models/" folder. Content updated.')

models = [
    Model(
        name ='ssd_mobilenet_v1_10',
        filepath ='vision/object_detection_segmentation/ssd-mobilenetv1/model/',
        sha = '6a2ba88990166b5212fc4115bb347dd1402fbf39'
    ),
    Model(
        name='squeezenet1.0-9',
        filepath='vision/classification/squeezenet/model/',
        sha = '7c4a0cc990d877f46105eb331bb71e2c90c0ecbb'
    ),
    Model(
        name='emotion-ferplus-8',
        filepath='vision/body_analysis/emotion_ferplus/model/',
        sha = '073ea68e09c0c8c21401b95e9cdccb42c639bc75'
    ),
    Model(
        name='FasterRCNN-10',
        filepath='vision/object_detection_segmentation/faster-rcnn/model/',
        sha = '7df2f48a6429ea412733af8ce9673a092a5f84c4'
    ),
    Model(
        name='yolov3-10',
        filepath='vision/object_detection_segmentation/yolov3/model/',
        sha = 'a3e31b46f37c2b5de0fc85b6b54571898e7bbbb7'
    ),
    Model(
        name='tinyyolov2-8',
        filepath='vision/object_detection_segmentation/tiny-yolov2/model/',
        sha = '7ad8395edc8057030d17c14459de6d07f4d11ac6'
    ),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Name of model to download')
    parser.add_argument('--models_list', default=False, action='store_true',
                        help='List of available models')
    args = parser.parse_args()

    selected_model_name = args.name
    print_all_models = args.models_list
    if selected_model_name is not None:
        print('Model: {}'.format(selected_model_name))

    if not print_all_models:
        download_cache()

    failedModels = []
    for m in models:
        if (not print_all_models):
            if selected_model_name is not None and not m.name.startswith(selected_model_name):
                continue
            if not m.download():
                failedModels.append(m.model_path)
        else:
            print(m)
    if failedModels:
        print("[Warn]: Following models have not been downloaded:")
        for f in failedModels:
            print("* {}".format(f))
print('[Info]: Done')
