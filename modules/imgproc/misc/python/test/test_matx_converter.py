from __future__ import print_function
import cv2 as cv
from cv2 import testMatxPythonConverter
from tests_common import NewOpenCVTests


class MatxConverterTest(NewOpenCVTests):
    def test_matxconverter(self):
        samples = ['samples/data/lena.jpg', 'cv/cascadeandhog/images/mona-lisa.png']

        for sample in samples:
            img = self.get_sample(sample)
            out = testMatxPythonConverter(img)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()


