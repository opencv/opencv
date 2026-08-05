'''
Helpers shared by the slam.py and visual_odometry.py samples.
'''

import os
import glob
import numpy as np
import cv2 as cv


def build_K(fx, fy, cx, cy):
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]], dtype=np.float64)


def parse_dist_coeffs(text):
    # "k1,k2,p1,p2[,k3,...]" -> 1xN row; empty input means no distortion.
    if not text:
        return np.array([])
    return np.array([float(tok) for tok in text.split(',') if tok.strip()], dtype=np.float64)


def list_image_files(images_dir):
    files = [f for f in sorted(glob.glob(os.path.join(images_dir, '*')))
             if cv.haveImageReader(f)]
    return files


def rotation_matrix_to_quaternion(R):
    # cv::Quat::createFromRotMat() (used by the C++ sample) is not exposed to Python,
    # so the conversion is done here with Shepperd's method: numerically stable for all
    # rotations.
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


def write_colmap_files(vo, K, dist_coeffs, image_size, pose_filenames, output_dir, trajectory):
    # Note: cv.slam.Map isn't wrapped for Python (only aggregate counts are,
    # via getNumKeyframes/getNumMapPoints), so only camera intrinsics and the
    # trajectory (images.txt) can be exported here; points3D.txt is written
    # empty (map points require the C++ sample) so the directory still loads.
    #
    # `trajectory` is passed in rather than read from `vo` because callers disagree on
    # which trajectory is "final": samples that run bundle adjustment/loop closure want
    # vo.getCorrectedTrajectory(), samples that only track want the raw vo.getTrajectory().
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        has_dist = dist_coeffs is not None and dist_coeffs.size > 0
        model = "FULL_OPENCV" if has_dist else "PINHOLE"
        params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        if has_dist:
            # FULL_OPENCV needs exactly 8 params (k1,k2,p1,p2,k3,k4,k5,k6); pad any shorter --dist with zeros.
            padded = np.zeros(8)
            padded[:min(8, dist_coeffs.size)] = dist_coeffs[:8]
            params.extend(padded.tolist())
        f.write(f"1 {model} {image_size[0]} {image_size[1]} "
                + " ".join(f"{p:.9f}" for p in params) + "\n")

    # IDs are 1-based because COLMAP reserves id 0 as "invalid".
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(trajectory)}, mean observations per image: 0\n")
        for i, T in enumerate(trajectory):
            T = np.asarray(T, dtype=np.float64).reshape(4, 4)
            qw, qx, qy, qz = rotation_matrix_to_quaternion(T[:3, :3])
            name = (os.path.basename(pose_filenames[i]) if i < len(pose_filenames)
                    else f"pose_{i}")
            f.write(f"{i + 1} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                    f"{T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} 1 {name}\n")
            # Second (POINTS2D) line intentionally blank: no per-frame keypoint list is
            # available here (see note above on why points3D.txt is empty).
            f.write("\n")

    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
