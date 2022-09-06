import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class volume_test(NewOpenCVTests):
    def test_VolumeDefault(self):
        depthPath = 'cv/rgbd/depth.png'
        depth = self.get_sample(depthPath, cv.IMREAD_ANYDEPTH).astype(np.float32)
        if depth.size <= 0:
            raise Exception('Failed to load depth file: %s' % depthPath)

        Rt = np.eye(4)
        volume = cv.Volume()
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        Kraycast = np.array([[525. ,   0. , 319.5],
                             [  0. , 525. , 239.5],
                             [  0. ,   0. ,   1. ]])

        volume.raycastEx(Rt, size[0], size[1], Kraycast, points, normals)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)

        volume.raycast(Rt, points, normals)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)

    def test_VolumeTSDF(self):
        depthPath = 'cv/rgbd/depth.png'
        depth = self.get_sample(depthPath, cv.IMREAD_ANYDEPTH).astype(np.float32)
        if depth.size <= 0:
            raise Exception('Failed to load depth file: %s' % depthPath)

        Rt = np.eye(4)

        settings = cv.VolumeSettings(cv.VolumeType_TSDF)
        volume = cv.Volume(cv.VolumeType_TSDF, settings)
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        Kraycast = settings.getCameraRaycastIntrinsics()
        volume.raycastEx(Rt, size[0], size[1], Kraycast, points, normals)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)

        volume.raycast(Rt, points, normals)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)

    def test_VolumeHashTSDF(self):
        depthPath = 'cv/rgbd/depth.png'
        depth = self.get_sample(depthPath, cv.IMREAD_ANYDEPTH).astype(np.float32)
        if depth.size <= 0:
            raise Exception('Failed to load depth file: %s' % depthPath)

        Rt = np.eye(4)
        settings = cv.VolumeSettings(cv.VolumeType_HashTSDF)
        volume = cv.Volume(cv.VolumeType_HashTSDF, settings)
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        Kraycast = settings.getCameraRaycastIntrinsics()
        volume.raycastEx(Rt, size[0], size[1], Kraycast, points, normals)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)

        volume.raycast(Rt, points, normals)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)

    def test_VolumeColorTSDF(self):
        depthPath = 'cv/rgbd/depth.png'
        rgbPath = 'cv/rgbd/rgb.png'
        depth = self.get_sample(depthPath, cv.IMREAD_ANYDEPTH).astype(np.float32)
        rgb = self.get_sample(rgbPath, cv.IMREAD_ANYCOLOR).astype(np.float32)

        if depth.size <= 0:
            raise Exception('Failed to load depth file: %s' % depthPath)
        if rgb.size <= 0:
            raise Exception('Failed to load RGB file: %s' % rgbPath)

        Rt = np.eye(4)
        settings = cv.VolumeSettings(cv.VolumeType_ColorTSDF)
        volume = cv.Volume(cv.VolumeType_ColorTSDF, settings)
        volume.integrateColor(depth, rgb, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)
        colors = np.zeros(size, np.float32)

        Kraycast = settings.getCameraRaycastIntrinsics()
        volume.raycastExColor(Rt, size[0], size[1], Kraycast, points, normals, colors)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)

        volume.raycastColor(Rt, points, normals, colors)

        self.assertEqual(points.shape, size)
        self.assertEqual(points.shape, normals.shape)
        self.assertEqual(points.shape, colors.shape)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
