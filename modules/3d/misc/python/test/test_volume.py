import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class volume_test(NewOpenCVTests):
    def test_VolumeDefault(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)

        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        volume = cv.Volume()
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        Kraycast = np.array([[525. ,   0. , 319.5],
                             [  0. , 525. , 239.5],
                             [  0. ,   0. ,   1. ]])

        volume.raycastEx(Rt, size[0], size[1], Kraycast, points, normals)
        volume.raycast(Rt, points, normals)

    def test_VolumeTSDF(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)

        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        settings = cv.VolumeSettings(cv.VolumeType_TSDF)
        volume = cv.Volume(cv.VolumeType_TSDF, settings)
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        Kraycast = settings.getCameraRaycastIntrinsics()
        volume.raycastEx(Rt, size[0], size[1], Kraycast, points, normals)
        volume.raycast(Rt, points, normals)

    def test_VolumeHashTSDF(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)

        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        settings = cv.VolumeSettings(cv.VolumeType_HashTSDF)
        volume = cv.Volume(cv.VolumeType_HashTSDF, settings)
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        Kraycast = settings.getCameraRaycastIntrinsics()
        volume.raycastEx(Rt, size[0], size[1], Kraycast, points, normals)
        volume.raycast(Rt, points, normals)

    def test_VolumeColorTSDF(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)
        rgb = self.get_sample('cv/rgbd/rgb.png', cv.IMREAD_ANYCOLOR).astype(np.float32)

        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        settings = cv.VolumeSettings(cv.VolumeType_ColorTSDF)
        volume = cv.Volume(cv.VolumeType_ColorTSDF, settings)
        volume.integrateColor(depth, rgb, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)
        colors = np.zeros(size, np.float32)

        Kraycast = settings.getCameraRaycastIntrinsics()
        volume.raycastExColor(Rt, size[0], size[1], Kraycast, points, normals, colors)
        volume.raycastColor(Rt, points, normals, colors)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
