# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import os
import datetime

from tests_common import NewOpenCVTests

class get_cache_dir_test(NewOpenCVTests):
    def test_get_cache_dir(self):
        #New binding
        path = cv.utils.fs.getCacheDirectory()
        try:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.isdir(path))
            os.rmdir(path)
        except:
            print("Tried to create cache directory ", path)
            pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()