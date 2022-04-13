#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os
import sys
import unittest

from tests_common import NewOpenCVTests


try:

    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')

    # Plaidml is an optional backend
    pkgs = [
             ('ocl'    , cv.gapi.core.ocl.kernels()),
             ('cpu'    , cv.gapi.core.cpu.kernels()),
             ('fluid'  , cv.gapi.core.fluid.kernels())
             # ('plaidml', cv.gapi.core.plaidml.kernels())
           ]


    @cv.gapi.op('custom.add', in_types=[cv.GMat, cv.GMat, int], out_types=[cv.GMat])
    class GAdd:
        """Calculates sum of two matrices."""

        @staticmethod
        def outMeta(desc1, desc2, depth):
            return desc1


    @cv.gapi.kernel(GAdd)
    class GAddImpl:
        """Implementation for GAdd operation."""

        @staticmethod
        def run(img1, img2, dtype):
            return cv.add(img1, img2)


    @cv.gapi.op('custom.split3', in_types=[cv.GMat], out_types=[cv.GMat, cv.GMat, cv.GMat])
    class GSplit3:
        """Divides a 3-channel matrix into 3 single-channel matrices."""

        @staticmethod
        def outMeta(desc):
            out_desc = desc.withType(desc.depth, 1)
            return out_desc, out_desc, out_desc


    @cv.gapi.kernel(GSplit3)
    class GSplit3Impl:
        """Implementation for GSplit3 operation."""

        @staticmethod
        def run(img):
            # NB: cv.split return list but g-api requires tuple in multiple output case
            return tuple(cv.split(img))


    @cv.gapi.op('custom.mean', in_types=[cv.GMat], out_types=[cv.GScalar])
    class GMean:
        """Calculates the mean value M of matrix elements."""

        @staticmethod
        def outMeta(desc):
            return cv.empty_scalar_desc()


    @cv.gapi.kernel(GMean)
    class GMeanImpl:
        """Implementation for GMean operation."""

        @staticmethod
        def run(img):
            # NB: cv.split return list but g-api requires tuple in multiple output case
            return cv.mean(img)


    @cv.gapi.op('custom.addC', in_types=[cv.GMat, cv.GScalar, int], out_types=[cv.GMat])
    class GAddC:
        """Adds a given scalar value to each element of given matrix."""

        @staticmethod
        def outMeta(mat_desc, scalar_desc, dtype):
            return mat_desc


    @cv.gapi.kernel(GAddC)
    class GAddCImpl:
        """Implementation for GAddC operation."""

        @staticmethod
        def run(img, sc, dtype):
            # NB: dtype is just ignored in this implementation.
            # Moreover from G-API kernel got scalar as tuples with 4 elements
            # where the last element is equal to zero, just cut him for broadcasting.
            return img + np.array(sc, dtype=np.uint8)[:-1]


    @cv.gapi.op('custom.size', in_types=[cv.GMat], out_types=[cv.GOpaque.Size])
    class GSize:
        """Gets dimensions from input matrix."""

        @staticmethod
        def outMeta(mat_desc):
            return cv.empty_gopaque_desc()


    @cv.gapi.kernel(GSize)
    class GSizeImpl:
        """Implementation for GSize operation."""

        @staticmethod
        def run(img):
            # NB: Take only H, W, because the operation should return cv::Size which is 2D.
            return img.shape[:2]


    @cv.gapi.op('custom.sizeR', in_types=[cv.GOpaque.Rect], out_types=[cv.GOpaque.Size])
    class GSizeR:
        """Gets dimensions from rectangle."""

        @staticmethod
        def outMeta(opaq_desc):
            return cv.empty_gopaque_desc()


    @cv.gapi.kernel(GSizeR)
    class GSizeRImpl:
        """Implementation for GSizeR operation."""

        @staticmethod
        def run(rect):
            # NB: rect - is tuple (x, y, h, w)
            return (rect[2], rect[3])


    @cv.gapi.op('custom.boundingRect', in_types=[cv.GArray.Point], out_types=[cv.GOpaque.Rect])
    class GBoundingRect:
        """Calculates minimal up-right bounding rectangle for the specified
           9 point set or non-zero pixels of gray-scale image."""

        @staticmethod
        def outMeta(arr_desc):
            return cv.empty_gopaque_desc()


    @cv.gapi.kernel(GBoundingRect)
    class GBoundingRectImpl:
        """Implementation for GBoundingRect operation."""

        @staticmethod
        def run(array):
            # NB: OpenCV - numpy array (n_points x 2).
            #     G-API  - array of tuples (n_points).
            return cv.boundingRect(np.array(array))


    @cv.gapi.op('custom.goodFeaturesToTrack',
                in_types=[cv.GMat, int, float, float, int, bool, float],
                out_types=[cv.GArray.Point2f])
    class GGoodFeatures:
        """Finds the most prominent corners in the image
           or in the specified image region."""

        @staticmethod
        def outMeta(desc, max_corners, quality_lvl,
                    min_distance, block_sz,
                    use_harris_detector, k):
            return cv.empty_array_desc()


    @cv.gapi.kernel(GGoodFeatures)
    class GGoodFeaturesImpl:
        """Implementation for GGoodFeatures operation."""

        @staticmethod
        def run(img, max_corners, quality_lvl,
                min_distance, block_sz,
                use_harris_detector, k):
            features = cv.goodFeaturesToTrack(img, max_corners, quality_lvl,
                                              min_distance, mask=None,
                                              blockSize=block_sz,
                                              useHarrisDetector=use_harris_detector, k=k)
            # NB: The operation output is cv::GArray<cv::Pointf>, so it should be mapped
            # to python parameters like this: [(1.2, 3.4), (5.2, 3.2)], because the cv::Point2f
            # according to opencv rules mapped to the tuple and cv::GArray<> mapped to the list.
            # OpenCV returns np.array with shape (n_features, 1, 2), so let's to convert it to list
            # tuples with size == n_features.
            features = list(map(tuple, features.reshape(features.shape[0], -1)))
            return features


    # To validate invalid cases
    def create_op(in_types, out_types):
        @cv.gapi.op('custom.op', in_types=in_types, out_types=out_types)
        class Op:
            """Custom operation for testing."""

            @staticmethod
            def outMeta(desc):
                raise NotImplementedError("outMeta isn't implemented")
        return Op


    class gapi_sample_pipelines(NewOpenCVTests):

        def test_custom_op_add(self):
            sz = (3, 3)
            in_mat1 = np.full(sz, 45, dtype=np.uint8)
            in_mat2 = np.full(sz, 50, dtype=np.uint8)

            # OpenCV
            expected = cv.add(in_mat1, in_mat2)

            # G-API
            g_in1  = cv.GMat()
            g_in2  = cv.GMat()
            g_out = GAdd.on(g_in1, g_in2, cv.CV_8UC1)

            comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

            pkg = cv.gapi.kernels(GAddImpl)
            actual = comp.apply(cv.gin(in_mat1, in_mat2), args=cv.gapi.compile_args(pkg))

            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


        def test_custom_op_split3(self):
            sz = (4, 4)
            in_ch1 = np.full(sz, 1, dtype=np.uint8)
            in_ch2 = np.full(sz, 2, dtype=np.uint8)
            in_ch3 = np.full(sz, 3, dtype=np.uint8)
            # H x W x C
            in_mat = np.stack((in_ch1, in_ch2, in_ch3), axis=2)

            # G-API
            g_in  = cv.GMat()
            g_ch1, g_ch2, g_ch3 = GSplit3.on(g_in)

            comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_ch1, g_ch2, g_ch3))

            pkg = cv.gapi.kernels(GSplit3Impl)
            ch1, ch2, ch3 = comp.apply(cv.gin(in_mat), args=cv.gapi.compile_args(pkg))

            self.assertEqual(0.0, cv.norm(in_ch1, ch1, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(in_ch2, ch2, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(in_ch3, ch3, cv.NORM_INF))


        def test_custom_op_mean(self):
            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            in_mat = cv.imread(img_path)

            # OpenCV
            expected = cv.mean(in_mat)

            # G-API
            g_in  = cv.GMat()
            g_out = GMean.on(g_in)

            comp = cv.GComputation(g_in, g_out)

            pkg    = cv.gapi.kernels(GMeanImpl)
            actual = comp.apply(cv.gin(in_mat), args=cv.gapi.compile_args(pkg))

            # Comparison
            self.assertEqual(expected, actual)


        def test_custom_op_addC(self):
            sz = (3, 3, 3)
            in_mat = np.full(sz, 45, dtype=np.uint8)
            sc = (50, 10, 20)

            # Numpy reference, make array from sc to keep uint8 dtype.
            expected = in_mat + np.array(sc, dtype=np.uint8)

            # G-API
            g_in  = cv.GMat()
            g_sc  = cv.GScalar()
            g_out = GAddC.on(g_in, g_sc, cv.CV_8UC1)
            comp  = cv.GComputation(cv.GIn(g_in, g_sc), cv.GOut(g_out))

            pkg = cv.gapi.kernels(GAddCImpl)
            actual = comp.apply(cv.gin(in_mat, sc), args=cv.gapi.compile_args(pkg))

            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


        def test_custom_op_size(self):
            sz = (100, 150, 3)
            in_mat = np.full(sz, 45, dtype=np.uint8)

            # Open_cV
            expected = (100, 150)

            # G-API
            g_in = cv.GMat()
            g_sz = GSize.on(g_in)
            comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_sz))

            pkg = cv.gapi.kernels(GSizeImpl)
            actual = comp.apply(cv.gin(in_mat), args=cv.gapi.compile_args(pkg))

            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


        def test_custom_op_sizeR(self):
            # x, y, h, w
            roi = (10, 15, 100, 150)

            expected = (100, 150)

            # G-API
            g_r  = cv.GOpaque.Rect()
            g_sz = GSizeR.on(g_r)
            comp = cv.GComputation(cv.GIn(g_r), cv.GOut(g_sz))

            pkg = cv.gapi.kernels(GSizeRImpl)
            actual = comp.apply(cv.gin(roi), args=cv.gapi.compile_args(pkg))

            # cv.norm works with tuples ?
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


        def test_custom_op_boundingRect(self):
            points = [(0,0), (0,1), (1,0), (1,1)]

            # OpenCV
            expected = cv.boundingRect(np.array(points))

            # G-API
            g_pts = cv.GArray.Point()
            g_br  = GBoundingRect.on(g_pts)
            comp  = cv.GComputation(cv.GIn(g_pts), cv.GOut(g_br))

            pkg = cv.gapi.kernels(GBoundingRectImpl)
            actual = comp.apply(cv.gin(points), args=cv.gapi.compile_args(pkg))

            # cv.norm works with tuples ?
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


        def test_custom_op_goodFeaturesToTrack(self):
            # G-API
            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            in_mat = cv.cvtColor(cv.imread(img_path), cv.COLOR_RGB2GRAY)

            # NB: goodFeaturesToTrack configuration
            max_corners         = 50
            quality_lvl         = 0.01
            min_distance        = 10.0
            block_sz            = 3
            use_harris_detector = True
            k                   = 0.04

            # OpenCV
            expected = cv.goodFeaturesToTrack(in_mat, max_corners, quality_lvl,
                                              min_distance, mask=None,
                                              blockSize=block_sz, useHarrisDetector=use_harris_detector, k=k)

            # G-API
            g_in = cv.GMat()
            g_out = GGoodFeatures.on(g_in, max_corners, quality_lvl,
                                     min_distance, block_sz, use_harris_detector, k)

            comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))
            pkg = cv.gapi.kernels(GGoodFeaturesImpl)
            actual = comp.apply(cv.gin(in_mat), args=cv.gapi.compile_args(pkg))

            # NB: OpenCV & G-API have different output types.
            # OpenCV - numpy array with shape (num_points, 1, 2)
            # G-API  - list of tuples with size - num_points
            # Comparison
            self.assertEqual(0.0, cv.norm(expected.flatten(),
                                          np.array(actual, dtype=np.float32).flatten(), cv.NORM_INF))


        def test_invalid_op(self):
            # NB: Empty input types list
            with self.assertRaises(Exception): create_op(in_types=[], out_types=[cv.GMat])
            # NB: Empty output types list
            with self.assertRaises(Exception): create_op(in_types=[cv.GMat], out_types=[])

            # Invalid output types
            with self.assertRaises(Exception): create_op(in_types=[cv.GMat], out_types=[int])
            with self.assertRaises(Exception): create_op(in_types=[cv.GMat], out_types=[cv.GMat, int])
            with self.assertRaises(Exception): create_op(in_types=[cv.GMat], out_types=[str, cv.GScalar])


        def test_invalid_op_input(self):
            # NB: Check GMat/GScalar
            with self.assertRaises(Exception): create_op([cv.GMat]   , [cv.GScalar]).on(cv.GScalar())
            with self.assertRaises(Exception): create_op([cv.GScalar], [cv.GScalar]).on(cv.GMat())

            # NB: Check GOpaque
            op = create_op([cv.GOpaque.Rect], [cv.GMat])
            with self.assertRaises(Exception): op.on(cv.GOpaque.Bool())
            with self.assertRaises(Exception): op.on(cv.GOpaque.Int())
            with self.assertRaises(Exception): op.on(cv.GOpaque.Double())
            with self.assertRaises(Exception): op.on(cv.GOpaque.Float())
            with self.assertRaises(Exception): op.on(cv.GOpaque.String())
            with self.assertRaises(Exception): op.on(cv.GOpaque.Point())
            with self.assertRaises(Exception): op.on(cv.GOpaque.Point2f())
            with self.assertRaises(Exception): op.on(cv.GOpaque.Size())

            # NB: Check GArray
            op = create_op([cv.GArray.Rect], [cv.GMat])
            with self.assertRaises(Exception): op.on(cv.GArray.Bool())
            with self.assertRaises(Exception): op.on(cv.GArray.Int())
            with self.assertRaises(Exception): op.on(cv.GArray.Double())
            with self.assertRaises(Exception): op.on(cv.GArray.Float())
            with self.assertRaises(Exception): op.on(cv.GArray.String())
            with self.assertRaises(Exception): op.on(cv.GArray.Point())
            with self.assertRaises(Exception): op.on(cv.GArray.Point2f())
            with self.assertRaises(Exception): op.on(cv.GArray.Size())

            # Check other possible invalid options
            with self.assertRaises(Exception): op.on(cv.GMat())
            with self.assertRaises(Exception): op.on(cv.GScalar())

            with self.assertRaises(Exception): op.on(1)
            with self.assertRaises(Exception): op.on('foo')
            with self.assertRaises(Exception): op.on(False)

            with self.assertRaises(Exception): create_op([cv.GMat, int], [cv.GMat]).on(cv.GMat(), 'foo')
            with self.assertRaises(Exception): create_op([cv.GMat, int], [cv.GMat]).on(cv.GMat())


        def test_stateful_kernel(self):
            @cv.gapi.op('custom.sum', in_types=[cv.GArray.Int], out_types=[cv.GOpaque.Int])
            class GSum:
                @staticmethod
                def outMeta(arr_desc):
                    return cv.empty_gopaque_desc()


            @cv.gapi.kernel(GSum)
            class GSumImpl:
                last_result = 0

                @staticmethod
                def run(arr):
                    GSumImpl.last_result = sum(arr)
                    return GSumImpl.last_result


            g_in  = cv.GArray.Int()
            comp  = cv.GComputation(cv.GIn(g_in), cv.GOut(GSum.on(g_in)))

            s = comp.apply(cv.gin([1, 2, 3, 4]), args=cv.gapi.compile_args(cv.gapi.kernels(GSumImpl)))
            self.assertEqual(10, s)

            s = comp.apply(cv.gin([1, 2, 8, 7]), args=cv.gapi.compile_args(cv.gapi.kernels(GSumImpl)))
            self.assertEqual(18, s)

            self.assertEqual(18, GSumImpl.last_result)


        def test_opaq_with_custom_type(self):
            @cv.gapi.op('custom.op', in_types=[cv.GOpaque.Any, cv.GOpaque.String], out_types=[cv.GOpaque.Any])
            class GLookUp:
                @staticmethod
                def outMeta(opaq_desc0, opaq_desc1):
                    return cv.empty_gopaque_desc()

            @cv.gapi.kernel(GLookUp)
            class GLookUpImpl:
                @staticmethod
                def run(table, key):
                    return table[key]


            g_table = cv.GOpaque.Any()
            g_key   = cv.GOpaque.String()
            g_out   = GLookUp.on(g_table, g_key)

            comp = cv.GComputation(cv.GIn(g_table, g_key), cv.GOut(g_out))

            table = {
                        'int':   42,
                        'str':   'hello, world!',
                        'tuple': (42, 42)
                    }

            out = comp.apply(cv.gin(table, 'int'), args=cv.gapi.compile_args(cv.gapi.kernels(GLookUpImpl)))
            self.assertEqual(42, out)

            out = comp.apply(cv.gin(table, 'str'), args=cv.gapi.compile_args(cv.gapi.kernels(GLookUpImpl)))
            self.assertEqual('hello, world!', out)

            out = comp.apply(cv.gin(table, 'tuple'), args=cv.gapi.compile_args(cv.gapi.kernels(GLookUpImpl)))
            self.assertEqual((42, 42), out)


        def test_array_with_custom_type(self):
            @cv.gapi.op('custom.op', in_types=[cv.GArray.Any, cv.GArray.Any], out_types=[cv.GArray.Any])
            class GConcat:
                @staticmethod
                def outMeta(arr_desc0, arr_desc1):
                    return cv.empty_array_desc()

            @cv.gapi.kernel(GConcat)
            class GConcatImpl:
                @staticmethod
                def run(arr0, arr1):
                    return arr0 + arr1

            g_arr0 = cv.GArray.Any()
            g_arr1 = cv.GArray.Any()
            g_out  = GConcat.on(g_arr0, g_arr1)

            comp = cv.GComputation(cv.GIn(g_arr0, g_arr1), cv.GOut(g_out))

            arr0 = ((2, 2), 2.0)
            arr1 = (3,    'str')

            out = comp.apply(cv.gin(arr0, arr1),
                             args=cv.gapi.compile_args(cv.gapi.kernels(GConcatImpl)))

            self.assertEqual(arr0 + arr1, out)


        def test_raise_in_kernel(self):
            @cv.gapi.op('custom.op', in_types=[cv.GMat, cv.GMat], out_types=[cv.GMat])
            class GAdd:
                @staticmethod
                def outMeta(desc0, desc1):
                    return desc0

            @cv.gapi.kernel(GAdd)
            class GAddImpl:
                @staticmethod
                def run(img0, img1):
                    raise Exception('Error')
                    return img0 + img1

            g_in0 = cv.GMat()
            g_in1 = cv.GMat()
            g_out = GAdd.on(g_in0, g_in1)

            comp = cv.GComputation(cv.GIn(g_in0, g_in1), cv.GOut(g_out))

            img0 = np.array([1, 2, 3])
            img1 = np.array([1, 2, 3])

            with self.assertRaises(Exception): comp.apply(cv.gin(img0, img1),
                                                          args=cv.gapi.compile_args(
                                                              cv.gapi.kernels(GAddImpl)))


        def test_raise_in_outMeta(self):
            @cv.gapi.op('custom.op', in_types=[cv.GMat, cv.GMat], out_types=[cv.GMat])
            class GAdd:
                @staticmethod
                def outMeta(desc0, desc1):
                    raise NotImplementedError("outMeta isn't implemented")

            @cv.gapi.kernel(GAdd)
            class GAddImpl:
                @staticmethod
                def run(img0, img1):
                    return img0 + img1

            g_in0 = cv.GMat()
            g_in1 = cv.GMat()
            g_out = GAdd.on(g_in0, g_in1)

            comp = cv.GComputation(cv.GIn(g_in0, g_in1), cv.GOut(g_out))

            img0 = np.array([1, 2, 3])
            img1 = np.array([1, 2, 3])

            with self.assertRaises(Exception): comp.apply(cv.gin(img0, img1),
                                                          args=cv.gapi.compile_args(
                                                              cv.gapi.kernels(GAddImpl)))


        def test_invalid_outMeta(self):
            @cv.gapi.op('custom.op', in_types=[cv.GMat, cv.GMat], out_types=[cv.GMat])
            class GAdd:
                @staticmethod
                def outMeta(desc0, desc1):
                    # Invalid outMeta
                    return cv.empty_gopaque_desc()

            @cv.gapi.kernel(GAdd)
            class GAddImpl:
                @staticmethod
                def run(img0, img1):
                    return img0 + img1

            g_in0 = cv.GMat()
            g_in1 = cv.GMat()
            g_out = GAdd.on(g_in0, g_in1)

            comp = cv.GComputation(cv.GIn(g_in0, g_in1), cv.GOut(g_out))

            img0 = np.array([1, 2, 3])
            img1 = np.array([1, 2, 3])

            # FIXME: Cause Bad variant access.
            # Need to provide more descriptive error message.
            with self.assertRaises(Exception): comp.apply(cv.gin(img0, img1),
                                                          args=cv.gapi.compile_args(
                                                              cv.gapi.kernels(GAddImpl)))

        def test_pipeline_with_custom_kernels(self):
            @cv.gapi.op('custom.resize', in_types=[cv.GMat, tuple], out_types=[cv.GMat])
            class GResize:
                @staticmethod
                def outMeta(desc, size):
                    return desc.withSize(size)

            @cv.gapi.kernel(GResize)
            class GResizeImpl:
                @staticmethod
                def run(img, size):
                    return cv.resize(img, size)

            @cv.gapi.op('custom.transpose', in_types=[cv.GMat, tuple], out_types=[cv.GMat])
            class GTranspose:
                @staticmethod
                def outMeta(desc, order):
                    return desc

            @cv.gapi.kernel(GTranspose)
            class GTransposeImpl:
                @staticmethod
                def run(img, order):
                    return np.transpose(img, order)

            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            img      = cv.imread(img_path)
            size     = (32, 32)
            order    = (1, 0, 2)

            # Dummy pipeline just to validate this case:
            # gapi -> custom -> custom -> gapi

            # OpenCV
            expected = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            expected = cv.resize(expected, size)
            expected = np.transpose(expected, order)
            expected = cv.mean(expected)

            # G-API
            g_bgr        = cv.GMat()
            g_rgb        = cv.gapi.BGR2RGB(g_bgr)
            g_resized    = GResize.on(g_rgb, size)
            g_transposed = GTranspose.on(g_resized, order)
            g_mean       = cv.gapi.mean(g_transposed)

            comp = cv.GComputation(cv.GIn(g_bgr), cv.GOut(g_mean))
            actual = comp.apply(cv.gin(img), args=cv.gapi.compile_args(
                cv.gapi.kernels(GResizeImpl, GTransposeImpl)))

            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


except unittest.SkipTest as e:

    message = str(e)

    class TestSkip(unittest.TestCase):
        def setUp(self):
            self.skipTest('Skip tests: ' + message)

        def test_skip():
            pass

    pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
