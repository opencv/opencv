'''
Monocular visual odometry with cv.slam.VisualOdometry (ALIKED + LightGlue).

Example:
    python visual_odometry.py --aliked aliked.onnx --lightglue lg.onnx --images ./seq
'''

import argparse
import glob
import os
import sys
import time
import numpy as np
import cv2 as cv


def build_K(fx, fy, cx, cy):
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]], dtype=np.float64)


def list_image_files(images_dir):
    files = [f for f in sorted(glob.glob(os.path.join(images_dir, '*')))
             if cv.haveImageReader(f)]
    return files


def rotation_matrix_to_quaternion(R):
    # Shepperd's method: numerically stable for all rotations.
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def write_colmap_files(vo, K, image_size, pose_filenames, output_dir):
    # Note: cv.slam.Map isn't wrapped for Python (only aggregate counts are,
    # via getNumKeyframes/getNumMapPoints), so only camera intrinsics and the
    # trajectory (images.txt) can be exported here; point3d.txt (map points)
    # requires the C++ sample.
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'camera.txt'), 'w') as f:
        f.write(f"fx {K[0, 0]:.4f}\n")
        f.write(f"fy {K[1, 1]:.4f}\n")
        f.write(f"cx {K[0, 2]:.4f}\n")
        f.write(f"cy {K[1, 2]:.4f}\n")
        f.write(f"width {image_size[0]}\n")
        f.write(f"height {image_size[1]}\n")

    traj = vo.getTrajectory()
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(traj)}, mean observations per image: 0.0\n")
        for i, T in enumerate(traj):
            qw, qx, qy, qz = rotation_matrix_to_quaternion(T[:3, :3])
            name = (os.path.basename(pose_filenames[i]) if i < len(pose_filenames)
                    else f"pose_{i}")
            f.write(f"{i} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                    f"{T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} 1 {name}\n")


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
    det_params.engine    = cv.dnn.ENGINE_OPENCV
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
        detector, matcher, K, np.array([]), vo_params)

    image_files = list_image_files(args.images)
    if not image_files:
        print(f"no images found in {args.images}", file=sys.stderr)
        return 1

    print(f"images_folder = {args.images}")
    print(f"output_folder = {args.output}")
    print(f"found {len(image_files)} image(s)")

    # Tracks which input image each emitted trajectory pose came from, for images.txt.
    pose_filenames = []
    prev_traj_len = 0
    ref_filename = None
    image_size = (0, 0)
    n_emitted = 0

    t0 = time.perf_counter()
    for i, path in enumerate(image_files):
        img = cv.imread(path)
        if img is None:
            print(f"[FRAME {i}] file={path} imread failed", file=sys.stderr)
            continue
        image_size = (img.shape[1], img.shape[0])

        before = vo.getState()
        emitted = vo.processFrame(img)
        after = vo.getState()
        if emitted:
            n_emitted += 1

        # Track which input image maps to each trajectory pose.
        if before == cv.slam.NOT_INITIALIZED or \
           (before == cv.slam.TRACKING and after == cv.slam.INITIALIZING):
            ref_filename = path

        traj_len = len(vo.getTrajectory())
        added = traj_len - prev_traj_len
        if added == 1:
            pose_filenames.append(path)
        elif added == 2:
            pose_filenames.append(ref_filename)
            pose_filenames.append(path)
        prev_traj_len = traj_len

        print(f"[FRAME {i}] file={path}"
              f" emitted={'yes' if emitted else 'no'}"
              f" keyframes={vo.getNumKeyframes()}"
              f" map_points={vo.getNumMapPoints()}")
    elapsed = time.perf_counter() - t0
    ok = n_emitted > 0

    if ok and args.output:
        write_colmap_files(vo, K, image_size, pose_filenames, args.output)

    print(f"run={'ok' if ok else 'FAILED'}  frames={len(vo.getTrajectory())}  elapsed={elapsed:.2f}s")
    print(f"output -> {args.output}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
