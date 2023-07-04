# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import datetime

from tests_common import NewOpenCVTests

class get_cache_dir_test(NewOpenCVTests):
    def test_get_cache_dir(self):
        #New binding
        path = cv.utils.fs.getCacheDirectoryForDownloads()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.isdir(path))

    def get_cache_dir_imread_interop(self, ext):
        path = cv.utils.fs.getCacheDirectoryForDownloads()
        gold_image = np.ones((16, 16, 3), np.uint8)
        read_from_file = np.zeros((16, 16, 3), np.uint8)
        test_file_name = os.path.join(path, "test." + ext)
        try:
            cv.imwrite(test_file_name, gold_image)
            read_from_file = cv.imread(test_file_name)
        finally:
            os.remove(test_file_name)

        self.assertEqual(cv.norm(gold_image, read_from_file), 0)

    def test_get_cache_dir_imread_interop_png(self):
        self.get_cache_dir_imread_interop("png")

    def test_get_cache_dir_imread_interop_jpeg(self):
        self.get_cache_dir_imread_interop("jpg")

    def test_get_cache_dir_imread_interop_tiff(self):
        self.get_cache_dir_imread_interop("tif")

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
