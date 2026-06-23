'''
Monocular visual odometry with cv.slam.VisualOdometry (ALIKED + LightGlue).

Example:
    python visual_odometry.py --aliked aliked.onnx --lightglue lg.onnx --images ./seq
'''

import argparse
import time
import numpy as np
import cv2 as cv


def build_K(fx, fy, cx, cy):
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description='Monocular visual odometry using ALIKED + LightGlue')
    parser.add_argument('--aliked',    required=True,
                        help='Path to ALIKED ONNX model')
    parser.add_argument('--lightglue', required=True,
                        help='Path to LightGlue ONNX model')
    parser.add_argument('--images',    required=True,
                        help='Path to directory with input images')
    parser.add_argument('--output',    default='vo_out',
                        help='Output directory for trajectory and map (default: vo_out)')
    parser.add_argument('--fx', type=float, default=718.856,
                        help='Camera focal length X (default: KITTI-00)')
    parser.add_argument('--fy', type=float, default=718.856,
                        help='Camera focal length Y (default: KITTI-00)')
    parser.add_argument('--cx', type=float, default=607.1928,
                        help='Camera principal point X (default: KITTI-00)')
    parser.add_argument('--cy', type=float, default=185.2157,
                        help='Camera principal point Y (default: KITTI-00)')
    parser.add_argument('--min-parallax', type=float, default=1.5,
                        help='Minimum initialisation parallax in degrees (default: 1.5)')
    parser.add_argument('--min-points', type=int, default=50,
                        help='Minimum initialisation map points (default: 50)')
    args = parser.parse_args()

    det_params = cv.ALIKED.Params()
    det_params.inputSize = (640, 640)
    det_params.engine    = cv.dnn.ENGINE_NEW
    detector = cv.ALIKED.create(args.aliked, det_params)

    matcher = cv.LightGlueMatcher.create(
        args.lightglue, 0.0,
        cv.dnn.DNN_BACKEND_DEFAULT,
        cv.dnn.DNN_TARGET_CPU)

    vo_params = cv.slam.OdometryParams()
    vo_params.minInitParallaxDeg = args.min_parallax
    vo_params.minInitPoints      = args.min_points

    K = build_K(args.fx, args.fy, args.cx, args.cy)

    vo = cv.slam.VisualOdometry.create(
        detector, matcher,
        args.images, args.output,
        K, np.array([]), vo_params)

    t0 = time.perf_counter()
    ok = vo.run()
    elapsed = time.perf_counter() - t0

    print(f"run={'ok' if ok else 'FAILED'}  frames={len(vo.getTrajectory())}  elapsed={elapsed:.2f}s")
    print(f"output -> {args.output}")


if __name__ == '__main__':
    main()
