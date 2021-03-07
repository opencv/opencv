#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Bindings(NewOpenCVTests):

    def check_name(self, name):
        #print(name)
        self.assertFalse(name == None)
        self.assertFalse(name == "")

    def test_registry(self):
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_ANY));
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_FFMPEG))
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_OPENCV_MJPEG))
        backends = cv.videoio_registry.getBackends()
        for backend in backends:
            self.check_name(cv.videoio_registry.getBackendName(backend))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
