#!/usr/bin/env python
from __future__ import print_function
import sys
if hasattr(sys, 'dont_write_bytecode'): sys.dont_write_bytecode = True  # Don't write cache .pyc files
import os

try:
    data_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    sys.path.append(data_path)
    from opencv_data_downloader import Downloader
except ImportError:
    raise ImportError('Can\'t find OpenCV downloader module (should be located in samples/data directory).')

if __name__ == '__main__':
    failed = False
    if not Downloader().download_metalink('download/opencv_dnn_face_detector_uint8.meta4'):
        failed = True
    elif not Downloader().download_metalink('download/opencv_dnn_face_detector_fp16.meta4'):
        failed = True
    #elif not Downloader().download_metalink('download/opencv_dnn_face_detector_fp32.meta4'):
    #    failed = True
    sys.exit(1 if failed else 0)
