import unittest
import os
import sys
import time
import tracemalloc

import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..')))

from stitching_detailed.stitcher import Stitcher
# %%


class TestStitcher(unittest.TestCase):

    def test_performance(self):
        start = time.time()
        tracemalloc.start()

        stitcher = Stitcher(compose_megapix=3)
        result = stitcher.stitch(["boat5.jpg", "boat2.jpg",
                                  "boat3.jpg", "boat4.jpg",
                                  "boat1.jpg", "boat6.jpg"])
        cv.imwrite("time_test.jpg", result)

        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        time_needed = end - start

        self.assertLessEqual(current_memory / 10**6, 52)
        self.assertLessEqual(current_memory / 10**6, 136)
        self.assertLessEqual(time_needed, 3.77)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
