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

        volume.raycast(Rt, size[0], size[1], points, normals)
        volume.raycast(Rt, points, normals)

    def test_VolumeTSDF(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)

        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        volume = cv.Volume(cv.TSDF)
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        volume.raycast(Rt, size[0], size[1], points, normals)
        volume.raycast(Rt, points, normals)

    def test_VolumeHashTSDF(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)

        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        volume = cv.Volume(cv.HashTSDF)
        volume.integrate(depth, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)

        volume.raycast(Rt, size[0], size[1], points, normals)
        volume.raycast(Rt, points, normals)

    def test_VolumeColorTSDF(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)
        rgb = self.get_sample('cv/rgbd/rgb.png', cv.IMREAD_ANYCOLOR).astype(np.float32)

        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        volume = cv.Volume(cv.ColorTSDF)
        volume.integrate(depth, rgb, Rt)

        size = (480, 640, 4)
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)
        colors = np.zeros(size, np.float32)

        volume.raycast(Rt, size[0], size[1], points, normals, colors)
        volume.raycast(Rt, points, normals, colors)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()