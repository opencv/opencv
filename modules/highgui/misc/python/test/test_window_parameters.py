#!/usr/bin/env python
from __future__ import print_function

import os

# Force Qt's headless-safe platform plugin when nothing else is already
# configured. Without this, namedWindow() on a Qt-enabled build with no
# display server available (e.g. CI workers, which do not run the Python
# test suite under a virtual framebuffer) aborts the whole process instead
# of raising a catchable exception -- so this must be set *before* the
# first GUI call, and a plain try/except around namedWindow is not enough.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2 as cv

from tests_common import NewOpenCVTests


class window_parameters_test(NewOpenCVTests):

    def test_save_load_window_parameters(self):
        window_name = "test_save_load_window_parameters"
        try:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        except cv.error:
            self.skipTest("No GUI backend available in this build/environment")

        try:
            # Should not raise when the library is built with a backend that
            # supports persisting window state (currently: Qt).
            cv.saveWindowParameters(window_name)
            cv.loadWindowParameters(window_name)
        except cv.error as e:
            # Expected on builds without Qt support (see NO_QT_ERR_MSG in window.cpp)
            self.skipTest("saveWindowParameters/loadWindowParameters require the Qt "
                           "highgui backend: " + str(e))
        finally:
            cv.destroyWindow(window_name)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
