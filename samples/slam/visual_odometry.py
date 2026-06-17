'''
Monocular visual odometry with cv.slam.VisualOdometry (ALIKED + LightGlue).
'''

import time
import numpy as np
import cv2 as cv

ALIKED_MODEL    = '/media/user/path/to/models/aliked-n16rot-top1k-640.onnx'
LIGHTGLUE_MODEL = '/media/user/path/to/models/aliked_lightglue.onnx'
IMAGES_DIR      = '/media/user/path/to/dataset'
OUTPUT_DIR      = 'vo_out'

# KITTI-00: fx, fy, cx, cy
K = np.array([[718.856, 0.,      607.1928],
              [0.,      718.856, 185.2157],
              [0.,      0.,      1.      ]], dtype=np.float64)

# k1, k2, p1, p2, k3
DIST = np.array([-0.2811, 0.0723, -0.0003, 0.0001, 0.0], dtype=np.float64)


def make_detector():
    p = cv.ALIKED.Params()
    p.inputSize = (640, 640)
    p.engine    = cv.dnn.ENGINE_NEW
    return cv.ALIKED.create(ALIKED_MODEL, p)


def make_matcher():
    return cv.LightGlueMatcher.create(
        LIGHTGLUE_MODEL, 0.0,
        cv.dnn.DNN_BACKEND_DEFAULT,
        cv.dnn.DNN_TARGET_CPU)


def main():
    params = cv.slam.OdometryParams()
    params.minInitParallaxDeg = 1.5
    params.minInitPoints      = 50

    vo = cv.slam.VisualOdometry.create(
        make_detector(), make_matcher(),
        IMAGES_DIR, OUTPUT_DIR,
        K, DIST, params)

    t0 = time.perf_counter()
    ok = vo.run()
    elapsed = time.perf_counter() - t0

    print(f"run={'ok' if ok else 'FAILED'}  frames={len(vo.getTrajectory())}  elapsed={elapsed:.2f}s")
    print(f"output -> {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
