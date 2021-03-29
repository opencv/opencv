#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests


# Plaidml is an optional backend
pkgs = [
         ('ocl'    , cv.gapi.core.ocl.kernels()),
         ('cpu'    , cv.gapi.core.cpu.kernels()),
         ('fluid'  , cv.gapi.core.fluid.kernels())
         # ('plaidml', cv.gapi.core.plaidml.kernels())
     ]

# Test output GMat.
def custom_add(img1, img2, dtype):
    return cv.add(img1, img2)

# Test output GScalar.
def custom_mean(img):
    return cv.mean(img)

# Test output tuple of GMat's.
def custom_split3(img):
    # NB: cv.split return list but g-api requires tuple in multiple output case
    return tuple(cv.split(img))

# Test output GOpaque.
def custom_size(img):
    # NB: Take only H, W, because the operation should return cv::Size which is 2D.
    return img.shape[:2]

# Test output GArray.
def custom_goodFeaturesToTrack(img, max_corners, quality_lvl,
                               min_distance, mask, block_sz,
                               use_harris_detector, k):
    features = cv.goodFeaturesToTrack(img, max_corners, quality_lvl,
                                      min_distance, mask=mask,
                                      blockSize=block_sz,
                                      useHarrisDetector=use_harris_detector, k=k)
    # NB: The operation output is cv::GArray<cv::Pointf>, so it should be mapped
    # to python paramaters like this: [(1.2, 3.4), (5.2, 3.2)], because the cv::Point2f
    # according to opencv rules mapped to the tuple and cv::GArray<> mapped to the list.
    # OpenCV returns np.array with shape (n_features, 1, 2), so let's to convert it to list
    # tuples with size - n_features.
    features = list(map(tuple, features.reshape(features.shape[0], -1)))
    return features

# Test input scalar.
def custom_addC(img, sc, dtype):
    # NB: dtype is just ignored in this implementation.
    # More over from G-API kernel got scalar as tuples with 4 elements
    # where the last element is equal to zero, just cut him for broadcasting.
    return img + np.array(sc, dtype=np.uint8)[:-1]


# Test input opaque.
def custom_sizeR(rect):
    # NB: rect - is tuple (x, y, h, w)
    return (rect[2], rect[3])

# Test input array.
def custom_boundingRect(array):
    # NB: OpenCV - numpy array (n_points x 2).
    #     G-API  - array of tuples (n_points).
    return cv.boundingRect(np.array(array))


class gapi_sample_pipelines(NewOpenCVTests):

    # NB: This test check multiple outputs for operation
    def test_mean_over_r(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.imread(img_path)

        # # OpenCV
        _, _, r_ch = cv.split(in_mat)
        expected = cv.mean(r_ch)

        # G-API
        g_in = cv.GMat()
        b, g, r = cv.gapi.split3(g_in)
        g_out = cv.gapi.mean(r)
        comp = cv.GComputation(g_in, g_out)

        for pkg_name, pkg in pkgs:
            actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                             'Failed on ' + pkg_name + ' backend')


    def test_custom_mean(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.imread(img_path)

        # OpenCV
        expected = cv.mean(in_mat)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.mean(g_in)

        comp = cv.GComputation(g_in, g_out)

        pkg    = cv.gapi_wip_kernels((custom_mean, 'org.opencv.core.math.mean'))
        actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

        # Comparison
        self.assertEqual(expected, actual)


    def test_custom_add(self):
        sz = (3, 3)
        in_mat1 = np.full(sz, 45, dtype=np.uint8)
        in_mat2 = np.full(sz, 50 , dtype=np.uint8)

        # OpenCV
        expected = cv.add(in_mat1, in_mat2)

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

        pkg = cv.gapi_wip_kernels((custom_add, 'org.opencv.core.math.add'))
        actual = comp.apply(cv.gin(in_mat1, in_mat2), args=cv.compile_args(pkg))

        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_custom_size(self):
        sz = (100, 150, 3)
        in_mat = np.full(sz, 45, dtype=np.uint8)

        # OpenCV
        expected = (100, 150)

        # G-API
        g_in = cv.GMat()
        g_sz = cv.gapi.streaming.size(g_in)
        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_sz))

        pkg = cv.gapi_wip_kernels((custom_size, 'org.opencv.streaming.size'))
        actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_custom_goodFeaturesToTrack(self):
        # G-API
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.cvtColor(cv.imread(img_path), cv.COLOR_RGB2GRAY)

        # NB: goodFeaturesToTrack configuration
        max_corners         = 50
        quality_lvl         = 0.01
        min_distance        = 10
        block_sz            = 3
        use_harris_detector = True
        k                   = 0.04
        mask                = None

        # OpenCV
        expected = cv.goodFeaturesToTrack(in_mat, max_corners, quality_lvl,
                                          min_distance, mask=mask,
                                          blockSize=block_sz, useHarrisDetector=use_harris_detector, k=k)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.goodFeaturesToTrack(g_in, max_corners, quality_lvl,
                                            min_distance, mask, block_sz, use_harris_detector, k)

        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))
        pkg = cv.gapi_wip_kernels((custom_goodFeaturesToTrack, 'org.opencv.imgproc.feature.goodFeaturesToTrack'))
        actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

        # NB: OpenCV & G-API have different output types.
        # OpenCV - numpy array with shape (num_points, 1, 2)
        # G-API  - list of tuples with size - num_points
        # Comparison
        self.assertEqual(0.0, cv.norm(expected.flatten(),
                                      np.array(actual, dtype=np.float32).flatten(), cv.NORM_INF))


    def test_custom_addC(self):
        sz = (3, 3, 3)
        in_mat = np.full(sz, 45, dtype=np.uint8)
        sc = (50, 10, 20)

        # Numpy reference, make array from sc to keep uint8 dtype.
        expected = in_mat + np.array(sc, dtype=np.uint8)

        # G-API
        g_in = cv.GMat()
        g_sc = cv.GScalar()
        g_out = cv.gapi.addC(g_in, g_sc)
        comp = cv.GComputation(cv.GIn(g_in, g_sc), cv.GOut(g_out))

        pkg = cv.gapi_wip_kernels((custom_addC, 'org.opencv.core.math.addC'))
        actual = comp.apply(cv.gin(in_mat, sc), args=cv.compile_args(pkg))

        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_custom_sizeR(self):
        # x, y, h, w
        roi = (10, 15, 100, 150)

        expected = (100, 150)

        # G-API
        g_r  = cv.GOpaqueT(cv.gapi.CV_RECT)
        g_sz = cv.gapi.streaming.size(g_r)
        comp = cv.GComputation(cv.GIn(g_r), cv.GOut(g_sz))

        pkg = cv.gapi_wip_kernels((custom_sizeR, 'org.opencv.streaming.sizeR'))
        actual = comp.apply(cv.gin(roi), args=cv.compile_args(pkg))

        # cv.norm works with tuples ?
        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_custom_boundingRect(self):
        points = [(0,0), (0,1), (1,0), (1,1)]

        # OpenCV
        expected = cv.boundingRect(np.array(points))

        # G-API
        g_pts = cv.GArrayT(cv.gapi.CV_POINT)
        g_br  = cv.gapi.boundingRect(g_pts)
        comp = cv.GComputation(cv.GIn(g_pts), cv.GOut(g_br))

        pkg = cv.gapi_wip_kernels((custom_boundingRect, 'org.opencv.imgproc.shape.boundingRectVector32S'))
        actual = comp.apply(cv.gin(points), args=cv.compile_args(pkg))

        # cv.norm works with tuples ?
        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_multiple_custom_kernels(self):
        sz = (3, 3, 3)
        in_mat1 = np.full(sz, 45, dtype=np.uint8)
        in_mat2 = np.full(sz, 50 , dtype=np.uint8)

        # OpenCV
        expected = cv.mean(cv.split(cv.add(in_mat1, in_mat2))[1])

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_sum = cv.gapi.add(g_in1, g_in2)
        g_b, g_r, g_g = cv.gapi.split3(g_sum)
        g_mean = cv.gapi.mean(g_b)

        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_mean))


        pkg = cv.gapi_wip_kernels((custom_add   , 'org.opencv.core.math.add'),
                         (custom_mean  , 'org.opencv.core.math.mean'),
                         (custom_split3, 'org.opencv.core.transform.split3'))

        actual = comp.apply(cv.gin(in_mat1, in_mat2), args=cv.compile_args(pkg))

        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
