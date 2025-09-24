#!/usr/bin/env python

'''
Test for imread
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

from tests_common import NewOpenCVTests

class imread_test(NewOpenCVTests):
    def test_imread_to_buffer(self):
        path = self.extraTestDataPath + '/cv/shared/lena.png'
        ref = cv.imread(path)

        img = np.zeros_like(ref)
        cv.imread(path, img)
        self.assertEqual(cv.norm(ref, img, cv.NORM_INF), 0.0)

    def test_imread_with_meta(self):
        path = self.extraTestDataPath + '/highgui/readwrite/testExifOrientation_1.jpg'
        img, meta_types, meta_data = cv.imreadWithMetadata(path)
        self.assertTrue(img is not None)
        self.assertTrue(meta_types is not None)
        self.assertTrue(meta_data is not None)

        result, exif_entries = cv.decodeExif(meta_data[0][0])
        self.assertTrue(result)
        entry = exif_entries[0][0]
        self.assertEqual(entry.tagId, 274)
        self.assertEqual(entry.getValueAsInt(), 1)
        entry = exif_entries[0][4]
        self.assertEqual(entry.tagId, 305)
        self.assertEqual(entry.getTagIdAsString(), 'Software')
        self.assertEqual(entry.getValueAsString(), 'Adobe Photoshop CC 2015 (Windows)')

        path = self.extraTestDataPath + '/highgui/readwrite/testExifOrientation_1.png'
        img, meta_types, meta_data = cv.imreadWithMetadata(path)
        self.assertTrue(img is not None)
        self.assertTrue(meta_types is not None)
        self.assertTrue(meta_data is not None)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
