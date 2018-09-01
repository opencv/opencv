#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class UMat(NewOpenCVTests):

    def test_umat_construct(self):
        data = np.random.random([512, 512])
        # UMat constructors
        data_um = cv.UMat(data)  # from ndarray
        data_sub_um = cv.UMat(data_um, (128, 256), (128, 256))  # from UMat
        data_dst_um = cv.UMat(128, 128, cv.CV_64F)  # from size/type
        # test continuous and submatrix flags
        assert data_um.isContinuous() and not data_um.isSubmatrix()
        assert not data_sub_um.isContinuous() and data_sub_um.isSubmatrix()
        # test operation on submatrix
        cv.multiply(data_sub_um, 2., dst=data_dst_um)
        assert np.allclose(2. * data[128:256, 128:256], data_dst_um.get())

    def test_umat_handle(self):
        a_um = cv.UMat(256, 256, cv.CV_32F)
        _ctx_handle = cv.UMat.context()  # obtain context handle
        _queue_handle = cv.UMat.queue()  # obtain queue handle
        _a_handle = a_um.handle(cv.ACCESS_READ)  # obtain buffer handle
        _offset = a_um.offset  # obtain buffer offset

    def test_umat_matching(self):
        img1 = self.get_sample("samples/data/right01.jpg")
        img2 = self.get_sample("samples/data/right02.jpg")

        orb = cv.ORB_create()

        img1, img2 = cv.UMat(img1), cv.UMat(img2)
        ps1, descs_umat1 = orb.detectAndCompute(img1, None)
        ps2, descs_umat2 = orb.detectAndCompute(img2, None)

        self.assertIsInstance(descs_umat1, cv.UMat)
        self.assertIsInstance(descs_umat2, cv.UMat)
        self.assertGreater(len(ps1), 0)
        self.assertGreater(len(ps2), 0)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        res_umats = bf.match(descs_umat1, descs_umat2)
        res = bf.match(descs_umat1.get(), descs_umat2.get())

        self.assertGreater(len(res), 0)
        self.assertEqual(len(res_umats), len(res))

    def test_umat_optical_flow(self):
        img1 = self.get_sample("samples/data/right01.jpg", cv.IMREAD_GRAYSCALE)
        img2 = self.get_sample("samples/data/right02.jpg", cv.IMREAD_GRAYSCALE)
        # Note, that if you want to see performance boost by OCL implementation - you need enough data
        # For example you can increase maxCorners param to 10000 and increase img1 and img2 in such way:
        # img = np.hstack([np.vstack([img] * 6)] * 6)

        feature_params = dict(maxCorners=239,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        p0 = cv.goodFeaturesToTrack(img1, mask=None, **feature_params)
        p0_umat = cv.goodFeaturesToTrack(cv.UMat(img1), mask=None, **feature_params)
        self.assertEqual(p0_umat.get().shape, p0.shape)

        p0 = np.array(sorted(p0, key=lambda p: tuple(p[0])))
        p0_umat = cv.UMat(np.array(sorted(p0_umat.get(), key=lambda p: tuple(p[0]))))
        self.assertTrue(np.allclose(p0_umat.get(), p0))

        _p1_mask_err = cv.calcOpticalFlowPyrLK(img1, img2, p0, None)

        _p1_mask_err_umat0 = list(map(lambda umat: umat.get(), cv.calcOpticalFlowPyrLK(img1, img2, p0_umat, None)))
        _p1_mask_err_umat1 = list(map(lambda umat: umat.get(), cv.calcOpticalFlowPyrLK(cv.UMat(img1), img2, p0_umat, None)))
        _p1_mask_err_umat2 = list(map(lambda umat: umat.get(), cv.calcOpticalFlowPyrLK(img1, cv.UMat(img2), p0_umat, None)))

        for _p1_mask_err_umat in [_p1_mask_err_umat0, _p1_mask_err_umat1, _p1_mask_err_umat2]:
            for data, data_umat in zip(_p1_mask_err, _p1_mask_err_umat):
                self.assertEqual(data.shape, data_umat.shape)
                self.assertEqual(data.dtype, data_umat.dtype)
        for _p1_mask_err_umat in [_p1_mask_err_umat1, _p1_mask_err_umat2]:
            for data_umat0, data_umat in zip(_p1_mask_err_umat0[:2], _p1_mask_err_umat[:2]):
                self.assertTrue(np.allclose(data_umat0, data_umat))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
