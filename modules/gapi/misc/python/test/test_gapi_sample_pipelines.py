#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests

class garray:
    class rect:
        pass
    class point2f:
        pass

class gopaque:
    class size:
        pass
    class rect:
        pass

class gmat:
    pass

class gscalar:
    pass

def validate_type(argtype, actual):
    switcher = {
        gmat:    cv.GMat,
        gscalar: cv.GScalar,
        int:     int
    }

    expected = switcher.get(argtype, 'Invalid input type')
    if expected != type(actual):
        raise Exception('Invalid input type!')


# NB: Top lvl decorator takes arguments
def op(op_id, in_types, out_types):
    # NB: Second lvl decorator takes class to decorate
    def op_with_params(cls):
        # NB: Decorated class
        class decorated:
            id = op_id

            @staticmethod
            def on(*args):
                if len(args) != len(in_types):
                    raise Exception('Invalid number of input elements!')

                # for expected, actual in zip(in_types, args):
                    # validate_type(expected, actual)

                op = cv.gapi.wip.op(op_id, cls.outMeta, *args)

                out_protos = []
                for out_type in out_types:
                    if out_type == gmat:
                        out_protos.append(op.getGMat())
                    if out_type == gscalar:
                        out_protos.append(op.getGScalar())
                    if out_type == gopaque.size:
                        out_protos.append(op.getGOpaque(cv.gapi.CV_SIZE))
                    if out_type == gopaque.rect:
                        out_protos.append(op.getGOpaque(cv.gapi.CV_RECT))
                    if out_type == garray.point2f:
                        out_protos.append(op.getGArray(cv.gapi.CV_POINT2F))

                return tuple(out_protos) if len(out_protos) != 1 else out_protos[0]

            @staticmethod
            def outMeta(*args):
                return cls.outMeta(args)

        return decorated
    return op_with_params


def kernel(op_cls):
    # NB: Second lvl decorator takes class to decorate
    def kernel_with_params(cls):
        # NB: Decorated class
        class decorated:
            outMeta = op_cls.outMeta
            id      = op_cls.id

            @staticmethod
            def run(*args):
                return cls.run(*args)

        return decorated
    return kernel_with_params


# Plaidml is an optional backend
pkgs = [
         ('ocl'    , cv.gapi.core.ocl.kernels()),
         ('cpu'    , cv.gapi.core.cpu.kernels()),
         ('fluid'  , cv.gapi.core.fluid.kernels())
         # ('plaidml', cv.gapi.core.plaidml.kernels())
       ]


@op('custom.add', in_types=[gmat, gmat, int], out_types=[gmat])
class GAdd:
    """ Operation which represents addition in G-API graph """

    @staticmethod
    def outMeta(desc1, desc2, depth):
        return desc1


@kernel(GAdd)
class GAddImpl:
    """ Python kernel for GAdd operation """

    @staticmethod
    def run(img1, img2, dtype):
        return cv.add(img1, img2)


@op('custom.split3', in_types=[gmat], out_types=[gmat, gmat, gmat])
class GSplit3:
    """ Documentation """

    @staticmethod
    def outMeta(desc):
        out_desc = desc.withType(desc.depth, 1)
        return out_desc, out_desc, out_desc


@kernel(GSplit3)
class GSplit3Impl:
    """ Documentation """

    @staticmethod
    def run(img):
        # NB: cv.split return list but g-api requires tuple in multiple output case
        return tuple(cv.split(img))


@op('custom.mean', in_types=[gmat], out_types=[gscalar])
class GMean:
    """ Documentation """

    @staticmethod
    def outMeta(desc):
        return cv.empty_scalar_desc()


@kernel(GMean)
class GMeanImpl:
    """ Documentation """

    @staticmethod
    def run(img):
        # NB: cv.split return list but g-api requires tuple in multiple output case
        return cv.mean(img)


@op('custom.addC', in_types=[gmat, gscalar, int], out_types=[gmat])
class GAddC:
    """ Documentation """

    @staticmethod
    def outMeta(mat_desc, scalar_desc, dtype):
        return mat_desc


@kernel(GAddC)
class GAddCImpl:
    """ Documentation """

    @staticmethod
    def run(img, sc, dtype):
        # NB: dtype is just ignored in this implementation.
        # More over from G-API kernel got scalar as tuples with 4 elements
        # where the last element is equal to zero, just cut him for broadcasting.
        return img + np.array(sc, dtype=np.uint8)[:-1]


@op('custom.size', in_types=[gmat], out_types=[gopaque.size])
class GSize:
    """ Documentation """

    @staticmethod
    def outMeta(mat_desc):
        return cv.empty_gopaque_desc()


@kernel(GSize)
class GSizeImpl:
    """ Documentation """

    @staticmethod
    def run(img):
        # NB: Take only H, W, because the operation should return cv::Size which is 2D.
        return img.shape[:2]


@op('custom.sizeR', in_types=[garray.rect], out_types=[gopaque.size])
class GSizeR:
    """ Documentation """

    @staticmethod
    def outMeta(arr_desc):
        return cv.empty_gopaque_desc()


@kernel(GSizeR)
class GSizeRImpl:
    """ Documentation """

    @staticmethod
    def run(rect):
        # NB: rect - is tuple (x, y, h, w)
        return (rect[2], rect[3])


@op('custom.boundingRect', in_types=[garray.rect], out_types=[gopaque.rect])
class GBoundingRect:
    """ Documentation """

    @staticmethod
    def outMeta(arr_desc):
        return cv.empty_gopaque_desc()


@kernel(GBoundingRect)
class GBoundingRectImpl:
    """ Documentation """

    @staticmethod
    def run(array):
        # NB: OpenCV - numpy array (n_points x 2).
        #     G-API  - array of tuples (n_points).
        return cv.boundingRect(np.array(array))


@op('custom.goodFeaturesToTrack',
    in_types=[gmat, int, float, float, np.array, int, bool, float],
    out_types=[garray.point2f])
class GGoodFeatures:
    """ Documentation """

    @staticmethod
    def outMeta(desc, max_corners, quality_lvl,
                min_distance, mask, block_sz,
                use_harris_detector, k):
        return cv.empty_array_desc()


@kernel(GGoodFeatures)
class GGoodFeaturesImpl:
    """ Documentation """

    @staticmethod
    def run(img, max_corners, quality_lvl,
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
        # tuples with size == n_features.
        features = list(map(tuple, features.reshape(features.shape[0], -1)))
        return features


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

        pkg = cv.gapi.wip.kernels(GAddImpl)
        actual = comp.apply(cv.gin(in_mat1, in_mat2), args=cv.compile_args(pkg))

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

        pkg = cv.gapi.wip.kernels(GSplit3Impl)
        ch1, ch2, ch3 = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

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

        pkg    = cv.gapi.wip.kernels(GMeanImpl)
        actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

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

        pkg = cv.gapi.wip.kernels(GAddCImpl)
        actual = comp.apply(cv.gin(in_mat, sc), args=cv.compile_args(pkg))

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

        pkg = cv.gapi.wip.kernels(GSizeImpl)
        actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_custom_op_sizeR(self):
        # x, y, h, w
        roi = (10, 15, 100, 150)

        expected = (100, 150)

        # G-API
        g_r  = cv.GOpaqueT(cv.gapi.CV_RECT)
        g_sz = GSizeR.on(g_r)
        comp = cv.GComputation(cv.GIn(g_r), cv.GOut(g_sz))

        pkg = cv.gapi.wip.kernels(GSizeRImpl)
        actual = comp.apply(cv.gin(roi), args=cv.compile_args(pkg))

        # cv.norm works with tuples ?
        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_custom_op_boundingRect(self):
        points = [(0,0), (0,1), (1,0), (1,1)]

        # OpenCV
        expected = cv.boundingRect(np.array(points))

        # G-API
        g_pts = cv.GArrayT(cv.gapi.CV_POINT)
        g_br  = GBoundingRect.on(g_pts)
        comp = cv.GComputation(cv.GIn(g_pts), cv.GOut(g_br))

        pkg = cv.gapi.wip.kernels(GBoundingRectImpl)
        actual = comp.apply(cv.gin(points), args=cv.compile_args(pkg))

        # cv.norm works with tuples ?
        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_custom_op_goodFeaturesToTrack(self):
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
        g_out = GGoodFeatures.on(g_in, max_corners, quality_lvl,
                                 min_distance, mask, block_sz, use_harris_detector, k)

        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))
        pkg = cv.gapi.wip.kernels(GGoodFeaturesImpl)
        actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

        # NB: OpenCV & G-API have different output types.
        # OpenCV - numpy array with shape (num_points, 1, 2)
        # G-API  - list of tuples with size - num_points
        # Comparison
        self.assertEqual(0.0, cv.norm(expected.flatten(),
                                      np.array(actual, dtype=np.float32).flatten(), cv.NORM_INF))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
