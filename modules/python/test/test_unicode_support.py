# -*- coding:utf-8 -*-
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import tempfile
import sys

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
    tmp_path = None

    def setUp(self):
        if os.name != 'nt' and sys.getfilesystemencoding() != "utf-8":
            self.skipTest("environment variable `LANG` is not utf8")
            return
        self.tmp_path = tempfile.mkdtemp(str_unicode)

    def tearDown(self):
        rmdir(self.tmp_path)

    def test_fs_get_cache_dir(self):
        tmp_bak = os.environ["TMP"]
        os.environ["TMP"] = self.tmp_path

        path = cv.utils.fs.getCacheDirectoryForDownloads()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.isdir(path))
        self.assertIn(self.tmp_path, path)

        os.environ["TMP"] = tmp_bak

    def test_im_rw(self):
        data_write = np.random.randint(0, 255, (4, 4, 3))
        file_path = os.path.join(self.tmp_path, str_unicode + ".png")
        cv.imwrite(file_path, data_write)
        self.assertTrue(os.path.exists(file_path))
        data_read = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        self.assertTrue(np.array_equal(data_write, data_read))

    def test_video_write(self):
        file_path = os.path.join(self.tmp_path, str_unicode + ".mp4")
        cv.VideoWriter(file_path, cv.VideoWriter_fourcc('P', 'I', 'M', '1'), 1, (1, 1))
        self.assertTrue(os.path.exists(file_path))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
