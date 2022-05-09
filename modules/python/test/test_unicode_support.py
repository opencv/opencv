# -*- coding:utf-8 -*-
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import tempfile

from tests_common import NewOpenCVTests

str_unicode = u"测试tést"


def rmdir(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(directory)


class unicode_support_test(NewOpenCVTests):
    tmp_new = None
    tmp_old = None

    def setUp(self):
        self.tmp_old = os.environ["TMP"]
        self.tmp_new = tempfile.mkdtemp(str_unicode)
        os.environ["TMP"] = self.tmp_new

    def tearDown(self):
        rmdir(self.tmp_new)
        os.environ["TMP"] = self.tmp_old

    def test_fs_get_cache_dir(self):
        path = cv.utils.fs.getCacheDirectoryForDownloads()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.isdir(path))
        self.assertIn(self.tmp_new, path)

    def test_im_rw(self):
        data_write = np.random.randint(0, 255, (4, 4, 3))
        file_path = os.path.join(self.tmp_new, str_unicode + ".png")
        cv.imwrite(file_path, data_write)
        self.assertTrue(os.path.exists(file_path))
        data_read = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        self.assertTrue(np.array_equal(data_write, data_read))

    def test_video_write(self):
        file_path = os.path.join(self.tmp_new, str_unicode + ".mp4")
        cv.VideoWriter(file_path, cv.VideoWriter_fourcc('P', 'I', 'M', '1'), 1, (1, 1))
        self.assertTrue(os.path.exists(file_path))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
