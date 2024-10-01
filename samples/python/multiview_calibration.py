# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import argparse
import glob
import json
import multiprocessing
import os
import sys
import time

from datetime import datetime

import cv2 as cv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
import math

def insideImageMask(pts, w, h):
    return np.logical_and(np.logical_and(pts[0] < w, pts[1] < h), np.logical_and(pts[0] > 0, pts[1] > 0))

def read_gt_rig(file, num_cameras, num_frames):
    Ks_gt = []
    distortions_gt = []
    rvecs_gt = []
    tvecs_gt = []
    rvecs0_gt = []
    tvecs0_gt = []
    with open(file, "r") as f:
        # Read in camera information
        for _ in range(num_cameras):
            f.readline() # camera label
            # 3 lines of K
            f.readline()
            K = np.zeros([3, 3])
            for i in range(3):
                K[i] = np.array([float(x) for x in f.readline().strip().split(" ")])
            Ks_gt.append(K)

            # 1 line of distortion
            f.readline()
            distortions_gt.append(np.array([float(x) for x in f.readline().strip().split(" ")]))

            # 3 line of rotation
            f.readline()
            R = np.zeros([3, 3])
            for i in range(3):
                R[i] = np.array([float(x) for x in f.readline().strip().split(" ")])
            rvecs_gt.append(R)

            # 1 line of translation
            f.readline()
            t = np.zeros([3, 1])
            for i in range(3):
                t[i] = np.array(float(f.readline().strip().split(" ")[0]))
            tvecs_gt.append(t)

        # Read in frame gt
        status = True
        for _ in range(num_frames):
            # 3 line of rotation
            f.readline()
            R = np.zeros([3, 3])
            for i in range(3):
                line = f.readline()
                if not line:
                    status = False
                    break
                R[i] = np.array([float(x) for x in line.strip().split(" ")])

            if not status:
                break

            rvecs0_gt.append(R)

            # 3 line of translation
            f.readline()
            t = np.zeros([3, 1])
            for i in range(3):
                t[i] = np.array(float(f.readline().strip().split(" ")[0]))
            tvecs0_gt.append(t)

    return Ks_gt, distortions_gt, rvecs_gt, tvecs_gt, rvecs0_gt, tvecs0_gt

def calc_angle(R1, R2):
    cos_r = ((R1.T @ R2).trace() - 1) / 2
    cos_r = min(max(cos_r, -1.), 1.)

    return np.degrees(math.acos(cos_r))

def calc_trans(R1, t1, R2, t2):
    return np.linalg.norm((R1.T @ t1 - R2.T @ t2))

def getDimBox(pts):
    return np.array([[pts[...,k].min(), pts[...,k].max()] for k in range(pts.shape[-1])])


def plotCamerasPosition(R, t, image_sizes, pairs, pattern, frame_idx, cam_ids, detection_mask):
    cam_box = np.array([
        [ 1,  1, 3],
        [ 1, -1, 3],
        [-1, -1, 3],
        [-1,  1, 3]
    ], dtype=np.float32)
    dist_to_pattern = np.linalg.norm(pattern.mean(0))
    cam_box *= 0.1 * dist_to_pattern
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax_lines = [None] * len(R)
    ax.set_title(f'Cameras position and pattern of frame {frame_idx}',
                 loc='center', wrap=True, fontsize=15)
    all_pts = [pattern]
    colors = np.random.RandomState(0).rand(len(R), 3)

    for i in range(len(R)):
        cam_box_i = cam_box.copy()
        cam_box_i[:,0] *= image_sizes[i][0] / max(image_sizes[i][1], image_sizes[i][0])
        cam_box_i[:,1] *= image_sizes[i][1] / max(image_sizes[i][1], image_sizes[i][0])
        cam_box_Rt = (R[i] @ cam_box_i.T + t[i]).T
        all_pts.append(np.concatenate((cam_box_Rt, t[i].T)))

        ax_lines[i] = ax.plot([t[i][0,0], cam_box_Rt[0,0]],
                              [t[i][1,0], cam_box_Rt[0,1]],
                              [t[i][2,0], cam_box_Rt[0,2]],
                              '-', color=colors[i])[0]

        ax.plot([t[i][0,0], cam_box_Rt[1,0]],
                [t[i][1,0], cam_box_Rt[1,1]],
                [t[i][2,0], cam_box_Rt[1,2]],
                '-', color=colors[i])
        ax.plot([t[i][0,0], cam_box_Rt[2,0]],
                [t[i][1,0], cam_box_Rt[2,1]],
                [t[i][2,0], cam_box_Rt[2,2]],
                '-', color=colors[i])
        ax.plot([t[i][0,0], cam_box_Rt[3,0]],
                [t[i][1,0], cam_box_Rt[3,1]],
                [t[i][2,0], cam_box_Rt[3,2]],
                '-', color=colors[i])

        ax.plot([cam_box_Rt[0,0], cam_box_Rt[1,0]],
                [cam_box_Rt[0,1], cam_box_Rt[1,1]],
                [cam_box_Rt[0,2], cam_box_Rt[1,2]],
                '-', color=colors[i])
        ax.plot([cam_box_Rt[1,0], cam_box_Rt[2,0]],
                [cam_box_Rt[1,1], cam_box_Rt[2,1]],
                [cam_box_Rt[1,2], cam_box_Rt[2,2]],
                '-', color=colors[i])
        ax.plot([cam_box_Rt[2,0], cam_box_Rt[3,0]],
                [cam_box_Rt[2,1], cam_box_Rt[3,1]],
                [cam_box_Rt[2,2], cam_box_Rt[3,2]],
                '-', color=colors[i])
        ax.plot([cam_box_Rt[3,0], cam_box_Rt[0,0]],
                [cam_box_Rt[3,1], cam_box_Rt[0,1]],
                [cam_box_Rt[3,2], cam_box_Rt[0,2]],
                '-', color=colors[i])

    # Plot lines between cameras
    base_width = 3 / detection_mask.shape[1]
    maps_pairs = set()
    for (i, j) in pairs:
        overlaps = np.sum((detection_mask[i] > 0) * (detection_mask[j] > 0))
        maps_pairs.add((np.minimum(i, j), np.maximum(i, j)))
        xs = [t[i][0,0], t[j][0,0]]
        ys = [t[i][1,0], t[j][1,0]]
        zs = [t[i][2,0], t[j][2,0]]
        edge_line = ax.plot(xs, ys, zs, '-', color='black', linewidth=overlaps * base_width)[0]

    # Plot all connected points
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            overlaps = np.sum((detection_mask[i] > 0) * (detection_mask[j] > 0))
            if overlaps == 0:
                continue
            xs = [t[i][0,0], t[j][0,0]]
            ys = [t[i][1,0], t[j][1,0]]
            zs = [t[i][2,0], t[j][2,0]]
            if (i, j) in maps_pairs:
                continue
            else:
                edge_line_extra = ax.plot(xs, ys, zs, '--', color='gray', linewidth=overlaps * base_width)[0]

    ax.scatter(pattern[:, 0], pattern[:, 1], pattern[:, 2], color='red', marker='o')
    ax.legend(ax_lines + [edge_line] + [edge_line_extra], cam_ids + ['stereo pair'] + ['full pairs'], fontsize=6)

    dim_box = getDimBox(np.concatenate((all_pts)))

    ax.set_xlim(dim_box[0])
    ax.set_ylim(dim_box[1])
    ax.set_zlim(dim_box[2])

    aspect = (
        dim_box[0, 1] - dim_box[0, 0],
        dim_box[1, 1] - dim_box[1, 0],
        dim_box[2, 1] - dim_box[2, 0],
    )
    ax.set_box_aspect(aspect)

    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('z', fontsize=16)

    ax.view_init(azim=90, elev=-40)


# [plot_detection]
def plotDetection(image_sizes, image_points):
    num_cameras = len(image_sizes)
    num_frames = len(image_points[0])

    for c in range(num_cameras):
        w, h = image_sizes[c]
        w = int(w / 10) + 1
        h = int(h / 10) + 1

        counts = np.zeros([h, w], dtype=np.int32)
        for f in range(num_frames):
            if len(image_points[c][f]):
                pos = np.floor(image_points[c][f] / 10).astype(np.int32)
                counts[pos[:,1], pos[:,0]] += 1

        vmax = np.max(counts)
        plt.figure()
        plt.imshow(counts, cmap='hot', interpolation='nearest',vmax=vmax)

        # Adding colorbar for reference
        plt.colorbar()
        plt.axis("off")
        savefile = "counts" + str(c) + ".png"
        print("Saving: " + savefile)
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        plt.close()

# [plot_detection]

def showUndistorted(image_points, Ks, distortions, image_names):
    detection_mask = getDetectionMask(image_points)
    for cam in range(len(image_points)):
        detected_imgs = np.where(detection_mask[cam])[0]
        random_frame = np.random.RandomState(0).choice(detected_imgs, 1, replace=False)[0]
        undistorted_pts = cv.undistortPoints(
            image_points[cam][random_frame][image_points[cam][random_frame][:,0] > 0],
            Ks[cam],
            distortions[cam],
            P=Ks[cam]
        )[:,0]

        fig = plt.figure()
        if image_names is not None:
            plt.imshow(cv.cvtColor(cv.undistort(
                cv.imread(image_names[cam][random_frame]),
                Ks[cam],
                distortions[cam]
            ), cv.COLOR_BGR2RGB))
        else:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x', fontsize=20)
            ax.set_ylabel('y', fontsize=20)

        plt.scatter(undistorted_pts[:,0], undistorted_pts[:,1], s=10)
        plt.title(
            f'Undistorted. Camera {cam_ids[cam]} frame {random_frame}',
            loc='center',
            wrap=True,
            fontsize=16
        )

        save_file = f'undistorted_{cam_ids[cam]}.png'
        print('Saving:', save_file)
        plt.savefig(save_file)


def plotProjection(points_2d, pattern_points, rvec0, tvec0, rvec1, tvec1,
                   K, dist_coeff, model, cam_idx, frame_idx, per_acc,
                   image=None):
    rvec2, tvec2 = cv.composeRT(rvec0, tvec0, rvec1, tvec1)[:2]

    if model == cv.CALIB_MODEL_FISHEYE:
        points_2d_est = cv.fisheye.projectPoints(
            pattern_points[:, None], rvec2, tvec2, K, dist_coeff.flatten()
        )[0].reshape(-1, 2)
    else:
        points_2d_est = cv.projectPoints(
            pattern_points, rvec2, tvec2, K, dist_coeff
        )[0].reshape(-1, 2)

    fig = plt.figure()
    errs = np.linalg.norm(points_2d - points_2d_est, axis=-1)
    mean_err = errs.mean()

    title = f"Comparison of given point (start) and back-projected (end). " \
        f"Cam. {cam_idx} frame {frame_idx} mean err. (px) {mean_err:.1f}. " \
        f"In top {per_acc:.0f}% accurate frames"

    dist_pattern = np.linalg.norm(points_2d_est.min(0) - points_2d_est.max(0))
    width = 2e-3 * dist_pattern
    head_width = 5 * width

    if image is None:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
    else:
        plt.imshow(image)
        ax = plt.gca()

    num_colors = 8
    cmap_fnc = lambda x : np.concatenate((x, 1-x, np.zeros_like(x)))
    cmap = cmap_fnc(np.linspace(0, 1, num_colors)[None, :])
    thrs = np.linspace(0, 10, num_colors)
    arrows = [None] * num_colors

    for k, (pt1, pt2) in enumerate(zip(points_2d, points_2d_est)):
        color = cmap[:, -1]
        for i, thr in enumerate(thrs):
            if errs[k] < thr:
                color = cmap[:, i]
                break
        arrow = ax.arrow(
            pt1[0], pt1[1], pt2[0]-pt1[0], pt2[1]-pt1[1],
            color=color, width=width, head_width=head_width,
        )
        for i, thr in enumerate(thrs):
            if errs[k] < thr:
                arrows[i] = arrow  # type: ignore
                break

    legend, legend_str = [], []
    for i in range(num_colors):
        if arrows[i] is not None:
            legend.append(arrows[i])
            if i == 0:
                legend_str.append(f'lower than {thrs[i]:.1f}')
            elif i == num_colors-1:
                legend_str.append(f'higher than {thrs[i]:.1f}')
            else:
                legend_str.append(f'between {thrs[i-1]:.1f} and {thrs[i]:.1f}')

    ax.legend(legend, legend_str, fontsize=10)
    ax.set_title(title, loc='center', wrap=True, fontsize=12)

    plt.savefig("projection_error.png")
    plt.close()

def getDetectionMask(image_points):
    detection_mask = np.zeros((len(image_points), len(image_points[0])), dtype=np.uint8)
# [detection_matrix]
    for i in range(len(image_points)):
        for j in range(len(image_points[0])):
            detection_mask[i,j] = int(len(image_points[i][j]) != 0)
# [detection_matrix]
    return detection_mask


def calibrateFromPoints(
        pattern_points,
        image_points,
        image_sizes,
        models,
        image_names=None,
        find_intrinsics_in_python=False,
        Ks=None,
        distortions=None
    ):
    """
    pattern_points: NUM_POINTS x 3 (numpy array)
    image_points: NUM_CAMERAS x NUM_FRAMES x NUM_POINTS x 2
    models: NUM_CAMERAS (cv.CALIB_MODEL_PINHOLE | cv.CALIB_MODEL_FISHEYE)
    image_sizes: NUM_CAMERAS x [width, height]
    """
    num_cameras = len(image_points)
    num_frames = len(image_points[0])
    detection_mask = getDetectionMask(image_points)
    pattern_points_all = [pattern_points] * num_frames
    with np.printoptions(threshold=np.inf):  # type: ignore
        print("detection mask Matrix:\n", str(detection_mask).replace('0\n ', '0').replace('1\n ', '1'))

    pinhole_flag = cv.CALIB_RATIONAL_MODEL
    fisheye_flag = cv.CALIB_RECOMPUTE_EXTRINSIC+cv.CALIB_FIX_SKEW
    if Ks is not None and distortions is not None:
        USE_INTRINSICS_GUESS = True
    else:
        USE_INTRINSICS_GUESS = find_intrinsics_in_python
        if find_intrinsics_in_python:
            Ks, distortions = [], []
            for c in range(num_cameras):
                if models[c] == cv.CALIB_MODEL_FISHEYE:
                    image_points_c = [
                        image_points[c][f][:, None] for f in range(num_frames) if len(image_points[c][f]) > 0
                    ]
                    repr_err_c, K, dist_coeff, _, _ = cv.fisheye.calibrate(
                        [pattern_points[:, None]] * len(image_points_c),
                        image_points_c,
                        image_sizes[c],
                        None,
                        None,
                        None,
                        None,
                        fisheye_flag
                    )
                else:
                    image_points_c = [
                        image_points[c][f] for f in range(num_frames) if len(image_points[c][f]) > 0
                    ]
                    repr_err_c, K, dist_coeff, _, _ = cv.calibrateCamera(
                        [pattern_points] * len(image_points_c),
                        image_points_c,
                        image_sizes[c],
                        None,
                        None,
                        flags=pinhole_flag
                    )
                print(f'Intrinsics calibration for camera {c}, reproj error {repr_err_c:.2f} (px)')
                Ks.append(K)
                distortions.append(dist_coeff)

    start_time = time.time()
#    try:
# [multiview_calib]
    rmse, Rs, Ts, Ks, distortions, rvecs0, tvecs0, errors_per_frame, output_pairs = \
            cv.calibrateMultiview(
                objPoints=pattern_points_all,
                imagePoints=image_points,
                imageSize=image_sizes,
                detectionMask=detection_mask,
                models=np.array(models, dtype=np.uint8),
                Rs=None,
                Ts=None,
                Ks=Ks,
                distortions=distortions,
                flagsForIntrinsics=np.array([pinhole_flag if models[x] == cv.CALIB_MODEL_PINHOLE else fisheye_flag for x in range(num_cameras)], dtype=int),
                flags = cv.CALIB_USE_INTRINSIC_GUESS if USE_INTRINSICS_GUESS else 0
            )
# [multiview_calib]
#    except Exception as e:
#        print("Multi-view calibration failed with the following exception:", e.__class__)
#        sys.exit(0)

    print('calibration time', time.time() - start_time, 'seconds')
    print('Rs', [Rs[x] for x in range(len(Rs))])
    print('Ts', [Ts[x].transpose() for x in range(len(Ts))])
    print('K', Ks)
    print('distortion', distortions)
    print('mean RMS error over all visible frames %.3E' % rmse)

    with np.printoptions(precision=2):
        print('mean RMS errors per camera', np.array([np.mean(errs[errs > 0]) for errs in errors_per_frame]))

    return {
        'Rs': Rs,
        'distortions': distortions,
        'Ks': Ks,
        'Ts': Ts,
        'rvecs0': rvecs0,
        'tvecs0': tvecs0,
        'errors_per_frame': errors_per_frame,
        'output_pairs': output_pairs,
        'image_points': image_points,
        'models': models,
        'image_sizes': image_sizes,
        'pattern_points': pattern_points,
        'detection_mask': detection_mask,
        'image_names': image_names,
    }


def visualizeResults(detection_mask, Rs, Ts, Ks, distortions, models,
                     image_points, errors_per_frame, rvecs0, tvecs0,
                     pattern_points, image_sizes, output_pairs, image_names, cam_ids):
    rvecs = [cv.Rodrigues(R)[0] for R in Rs]
    errors = errors_per_frame[errors_per_frame > 0]
    detection_mask_idxs = np.stack(np.where(detection_mask)) # 2 x M, first row is camera idx, second is frame idx

    # Get very first frame from first camera
    frame_idx = detection_mask_idxs[1, 0]
    pos = 0
    while rvecs0[frame_idx] is None:
        pos += 1
        frame_idx = detection_mask_idxs[1, pos]

    R_frame = cv.Rodrigues(rvecs0[frame_idx])[0]
    pattern_frame = (R_frame @ pattern_points.T + tvecs0[frame_idx]).T
    plotCamerasPosition(Rs, Ts, image_sizes, output_pairs, pattern_frame, frame_idx, cam_ids, detection_mask)

    save_file = 'cam_poses.png'
    print('Saving:', save_file)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    plt.close()

    # Generate and save undistorted images
    def plot(cam_idx, frame_idx):
        image = None
        if image_names is not None:
            image = cv.cvtColor(cv.imread(image_names[cam_idx][frame_idx]), cv.COLOR_BGR2RGB)
        mask = insideImageMask(image_points[cam_idx][frame_idx].T,
                               image_sizes[cam_idx][0], image_sizes[cam_idx][1])
        plotProjection(
            image_points[cam_idx][frame_idx][mask],
            pattern_points[mask],
            rvecs0[frame_idx],
            tvecs0[frame_idx],
            rvecs[cam_idx],
            Ts[cam_idx],
            Ks[cam_idx],
            distortions[cam_idx],
            models[cam_idx],
            cam_idx,
            frame_idx,
            (errors_per_frame[cam_idx, frame_idx] < errors).sum() * 100 / len(errors),
            image,
        )

    plot(detection_mask_idxs[0, pos], detection_mask_idxs[1, pos])
    showUndistorted(image_points, Ks, distortions, image_names)
    # plt.show()
    plotDetection(image_sizes, image_points)


def visualizeFromFile(file):
    file_read = cv.FileStorage(file, cv.FileStorage_READ)
    assert file_read.isOpened(), file
    read_keys = [
        'Rs', 'distortions', 'Ks', 'Ts', 'rvecs0', 'tvecs0',
        'errors_per_frame', 'output_pairs', 'image_points', 'models',
        'image_sizes', 'pattern_points', 'detection_mask', 'cam_ids',
    ]
    input = {}
    for key in read_keys:
        input[key] = file_read.getNode(key).mat()

    im_names_len = file_read.getNode('image_names').size()
    input['image_names'] = np.array(
        [file_read.getNode('image_names').at(i).string() for i in range(im_names_len)]
    ).reshape(input['image_points'].shape[:2])

    input['tvecs0'] = input['tvecs0'][..., None]
    input['Ts'] = input['Ts'][..., None]
    visualizeResults(**input)


def saveToFile(path_to_save, **kwargs):
    if path_to_save == '':
        path_to_save = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")+'.yaml'
    save_file = cv.FileStorage(path_to_save, cv.FileStorage_WRITE)

    kwargs['models'] = np.array(kwargs['models'], dtype=int)
    image_points = kwargs['image_points']

    for i in range(len(image_points)):
        for j in range(len(image_points[0])):
            if len(image_points[i][j]) == 0:
                image_points[i][j] = np.zeros((kwargs['pattern_points'].shape[0], 2))

    for key in kwargs.keys():
        if key == 'image_names':
            save_file.write('image_names', list(np.array(kwargs['image_names']).reshape(-1)))
        elif key == 'cam_ids':
            save_file.write('cam_ids', ','.join(cam_ids))
        elif key == 'distortions':
            value = kwargs[key]
            save_file.write('distortions', np.concatenate([x.reshape([-1,]) for x in value],axis=0))
        else:
            value = kwargs[key]
            if key in ('rvecs0', 'tvecs0'):
                # Replace None by [0, 0, 0]
                value = [arr if arr is not None else np.zeros((3, 1)) for arr in value]
            save_file.write(key, np.array(value))

    save_file.release()

def compareGT(gt_file, detection_mask, Rs, Ts, Ks, distortions, models,
                     image_points, errors_per_frame, rvecs0, tvecs0,
                     pattern_points, image_sizes, output_pairs, image_names, cam_ids):

    # Load the gt file
    Ks_gt, distortions_gt, rvecs_gt, tvecs_gt, rvecs0_gt, tvecs0_gt = read_gt_rig(gt_file, len(cam_ids), detection_mask[0].shape[0])

    # Compare the results and the gt
    err_r = np.zeros([len(cam_ids),])
    err_c = np.zeros([len(cam_ids),])
    for cam in range(len(cam_ids)):
        R = Rs[cam]

        # Convert angle from radians to degrees
        err_r[cam] = calc_angle(R, rvecs_gt[cam])
        err_c[cam] = calc_trans(R, Ts[cam], rvecs_gt[cam], tvecs_gt[cam])

    # Compute the distortion estimation error
    distortions = distortions
    Ks = Ks
    err_dist_mean = np.zeros([len(cam_ids),])
    err_dist_max = np.zeros([len(cam_ids),])
    err_dist_median = np.zeros([len(cam_ids),])
    for cam in range(len(cam_ids)):
        # Define the x and y coordinate vectors
        width = int(Ks_gt[cam][0, 2] * 2)
        height = int(Ks_gt[cam][1, 2] * 2)
# [vis_intrinsics_error]
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)

        # Generate the grid using np.meshgrid
        X, Y = np.meshgrid(x, y)

        points = np.concatenate([X[:,:,None], Y[:,:,None]], axis=2).reshape([-1, 1, 2])
        # Undistort the image points with the estimated distortions
        if models[cam] == cv.CALIB_MODEL_FISHEYE:
            points_undist = cv.fisheye.undistortPoints(points, Ks[cam],distortions[cam])
        else:
            points_undist = cv.undistortPoints(points, Ks[cam], distortions[cam])

        pt_norm = np.concatenate([points_undist, np.ones([points_undist.shape[0], 1, 1])], axis=2)

        # Distort the image points with the ground truth distortions
        if models[cam] == cv.CALIB_MODEL_FISHEYE:
            projected = cv.fisheye.projectPoints(pt_norm, np.zeros([3, 1]), np.zeros([3, 1]), Ks_gt[cam], distortions_gt[cam])[0]
        else:
            projected = cv.projectPoints(pt_norm, np.zeros([3, 1]), np.zeros([3, 1]), Ks_gt[cam], distortions_gt[cam])[0]

        errs_pt = np.linalg.norm(projected - points, axis=2)
        errs_pt = errs_pt.reshape([height, width])
        vmax = np.percentile(errs_pt, 95)

        plt.figure()
        plt.imshow(errs_pt, cmap='hot', interpolation='nearest',vmax=vmax)

        # Adding colorbar for reference
        plt.colorbar()
        savefile = "errors" + str(cam) + ".png"
        print("Saving: " + savefile)
        plt.savefig(savefile,dpi=300, bbox_inches='tight')
# [vis_intrinsics_error]

        err_dist_mean[cam] = np.mean(errs_pt)
        err_dist_max[cam] = np.max(errs_pt)
        err_dist_median[cam] = np.median(errs_pt)

    print("Distrotion error (mean, median):\n", " ".join([f'(%.4f, %.4f)' % (err_dist_mean[i], err_dist_median[i]) for i in range(len(cam_ids))]))
    print("Extrinsics error (R, C):\n", " ".join([f'(%.4f, %.4f)' % (err_r[i], err_c[i]) for i in range(len(cam_ids))]))
    print("Rotation error (mean, median):", f'(%.4f, %.4f)' % (np.mean(err_r), np.median(err_r)))
    print("Position error (mean, median):", f'(%.4f, %.4f)' % (np.mean(err_c), np.median(err_c)))

    if len(rvecs0_gt) > 0:
        # conver all things with respect to the first frame
        R0 = []
        for frame in range(0, len(rvecs0_gt)):
            if rvecs0[frame] is not None:
                R0.append(cv.Rodrigues(rvecs0[frame])[0])
            else:
                R0.append(None)

        # Compare the results and the gt
        err_r = np.zeros([detection_mask[0].shape[0],])
        err_c = np.zeros([detection_mask[0].shape[0],])
        for frame in range(detection_mask[0].shape[0]):
            # Convert angle from radians to degrees
            err_r[frame] = calc_angle(R0[frame], rvecs0_gt[frame])
            err_c[frame] = calc_trans(R0[frame], tvecs0[frame], rvecs0_gt[frame], tvecs0_gt[frame])

        print("Frame rotation error (mean, median):", f'(%.4f, %.4f)' % (np.mean(err_r), np.median(err_r)))
        print("Frame position error (mean, median):", f'(%.4f, %.4f)' % (np.mean(err_c), np.median(err_c)))

def chessboard_points(grid_size, dist_m):
    pattern = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    pattern[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * dist_m # only for (x,y,z=0)
    return pattern


def circles_grid_points(grid_size, dist_m):
    pattern = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            pattern.append([j * dist_m, i * dist_m, 0])
    return np.array(pattern, dtype=np.float32)


def asym_circles_grid_points(grid_size, dist_m):
    pattern = []
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            if i % 2 == 1:
                pattern.append([(j + .5)*dist_m, dist_m*(i//2 + .5), 0])
            else:
                pattern.append([j*dist_m, (i//2)*dist_m, 0])
    return np.array(pattern, dtype=np.float32)


def detect(cam_idx, frame_idx, img_name, pattern_type,
           grid_size, criteria, winsize, RESIZE_IMAGE, board_dict=None):
    assert os.path.exists(img_name), img_name
    img = cv.imread(img_name)
    img_size = img.shape[:2][::-1]

    scale = 1.0
    img_detection = img
    if RESIZE_IMAGE:
        scale = 1000.0 / max(img.shape[0], img.shape[1])
        if scale < 1.0:
            img_detection = cv.resize(
                img,
                (int(scale * img.shape[1]), int(scale * img.shape[0])),
                interpolation=cv.INTER_AREA
            )
# [detect_pattern]
    if pattern_type.lower() == 'checkerboard':
        ret, corners = cv.findChessboardCorners(
            cv.cvtColor(img_detection, cv.COLOR_BGR2GRAY), grid_size, None
        )
        if ret:
            if scale < 1.0:
                corners /= scale
            corners2 = cv.cornerSubPix(cv.cvtColor(img, cv.COLOR_BGR2GRAY),
                                       corners, winsize, (-1,-1), criteria)

    elif pattern_type.lower() == 'circles':
        ret, corners = cv.findCirclesGrid(
            img_detection, patternSize=grid_size, flags=cv.CALIB_CB_SYMMETRIC_GRID
        )
        if ret:
            corners2 = corners / scale

    elif pattern_type.lower() == 'acircles':
        ret, corners = cv.findCirclesGrid(
            img_detection, patternSize=grid_size, flags=cv.CALIB_CB_ASYMMETRIC_GRID
        )
        if ret:
            corners2 = corners / scale
    elif pattern_type.lower() == 'charuco':
        dictionary = cv.aruco.getPredefinedDictionary(board_dict["dictionary"])
        board = cv.aruco.CharucoBoard(
            size=(grid_size[0] + 1, grid_size[1] + 1),
            squareLength=board_dict["square_size"],
            markerLength=board_dict["marker_size"],
            dictionary=dictionary
        )

        # The found best practice is to refine detected Aruco marker with contour,
        # then refine subpix with the board functions
        detector_params = cv.aruco.DetectorParameters()
        charuco_params = cv.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
        refine_params = cv.aruco.RefineParameters()
        detector = cv.aruco.CharucoDetector(board, charuco_params, detector_params, refine_params)
        charucoCorners, charucoIds, _, _ = detector.detectBoard(img_detection)

        corners = np.ones([grid_size[0] * grid_size[1], 1, 2]) * -1
        ret = (not charucoIds is None) and charucoIds.flatten().size > 3

        if ret:
            corners[charucoIds.flatten()] = cv.cornerSubPix(cv.cvtColor(img, cv.COLOR_BGR2GRAY),
                                       charucoCorners / scale, winsize, (-1,-1), criteria)
            corners2 = corners

    else:
        raise ValueError("Calibration pattern is not supported!")
# [detect_pattern]
    if ret:
        # cv.drawChessboardCorners(img, grid_size, corners2, ret)
        # plt.imshow(img)
        # plt.show()
        return cam_idx, frame_idx, img_size, np.array(corners2, dtype=np.float32).reshape(-1, 2)
    else:
        # plt.imshow(img_detection)
        # plt.show()
        return cam_idx, frame_idx, img_size, np.array([], dtype=np.float32)


def calibrateFromImages(files_with_images, grid_size, pattern_type, models,
                        dist_m, winsize, points_json_file, debug_corners,
                        RESIZE_IMAGE, find_intrinsics_in_python,
                        is_parallel_detection=True, cam_ids=None, intrinsics_dir='', board_dict_path=None):
    """
    files_with_images: NUM_CAMERAS - path to file containing image names (NUM_FRAMES)
    grid_size: [width, height] -- size of grid pattern
    dist_m: length of a grid cell
    models: NUM_CAMERAS (cv.CALIB_MODEL_PINHOLE | cv.CALIB_MODEL_FISHEYE)
    """
# [calib_init]
    if pattern_type.lower() == 'checkerboard' or pattern_type.lower() == 'charuco':
        pattern = chessboard_points(grid_size, dist_m)
    elif pattern_type.lower() == 'circles':
        pattern = circles_grid_points(grid_size, dist_m)
    elif pattern_type.lower() == 'acircles':
        pattern = asym_circles_grid_points(grid_size, dist_m)
    else:
        raise NotImplementedError("Pattern type is not implemented!")

    if pattern_type.lower() == 'charuco':
        assert (board_dict_path is not None) and os.path.exists(board_dict_path)
        board_dict = json.load(open(board_dict_path, 'r'))
    else:
        board_dict = None

# [calib_init]

    assert len(files_with_images) == len(models) and len(grid_size) == 2
    if cam_ids is None:
        cam_ids = list(range(len(files_with_images)))

    all_images_names, input_data = [], []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    for cam_idx, filename in enumerate(files_with_images):
        assert os.path.exists(filename), filename
        print('cam_id:', cam_ids[cam_idx])

        images_names = open(filename, 'r').readlines()
        for i in range(len(images_names)):
            images_names[i] = images_names[i].strip()
            if images_names[i] != "":
                images_names[i] = "/".join(filename.split("/")[:-1] + [images_names[i]])
        all_images_names.append(images_names)
        if cam_idx > 0:
            # same number of images per file
            assert len(images_names) == len(all_images_names[0])
        for frame_idx, img_name in enumerate(images_names):
            input_data.append([cam_idx, frame_idx, img_name])

    image_sizes = [None] * len(files_with_images)
    image_points_cameras = [[np.array([], dtype=np.float32)] * len(images_names) for _ in files_with_images]

    if is_parallel_detection:
        parallel_job = joblib.Parallel(n_jobs=multiprocessing.cpu_count())
        output = parallel_job(
            joblib.delayed(detect)(
                cam_idx, frame_idx, img_name, pattern_type,
                grid_size, criteria, winsize, RESIZE_IMAGE, board_dict
            ) for cam_idx, frame_idx, img_name in input_data if img_name != ""
        )
        assert output is not None
        for cam_idx, frame_idx, img_size, corners in output:
            image_points_cameras[cam_idx][frame_idx] = corners
            if image_sizes[cam_idx] is None:
                image_sizes[cam_idx] = img_size
    else:
        for cam_idx, frame_idx, img_name in input_data:
            if img_name == "":
                continue
            _, _, img_size, corners = detect(
                cam_idx, frame_idx, img_name, pattern_type,
                grid_size, criteria, winsize, RESIZE_IMAGE, board_dict
            )
            image_points_cameras[cam_idx][frame_idx] = corners
            if image_sizes[cam_idx] is None:
                image_sizes[cam_idx] = img_size

    if debug_corners:
        # plots random image frames with detected points
        num_random_plots = 5
        visible_frames = []
        for c, pts_cam in enumerate(image_points_cameras):
            for f, pts_frame in enumerate(pts_cam):
                if pts_frame is not None and len(pts_frame) > 0:
                    visible_frames.append((c,f))
        random_images = np.random.RandomState(0).choice(
            range(len(visible_frames)), min(num_random_plots, len(visible_frames))
        )
        for idx in random_images:
            c, f = visible_frames[idx]
            img = cv.cvtColor(cv.imread(all_images_names[c][f]), cv.COLOR_BGR2RGB)
            if pattern_type.lower() != 'charuco':
                cv.drawChessboardCorners(img, grid_size, image_points_cameras[c][f], True)
            else:
                idx = image_points_cameras[c][f][:, 0] > 0
                cv.aruco.drawDetectedCornersCharuco(img, image_points_cameras[c][f][idx,None], np.arange(image_points_cameras[c][f].shape[0])[idx])
            plt.figure()
            plt.imshow(img)
        plt.show()

    if points_json_file:
        image_points_cameras_list = []
        for pts_cam in image_points_cameras:
            cam_pts = []
            for pts_frame in pts_cam:
                if pts_frame is not None:
                    cam_pts.append(pts_frame.tolist())
                else:
                    cam_pts.append([])
            image_points_cameras_list.append(cam_pts)

        with open(points_json_file, 'w') as wf:
            json.dump({
                'object_points': pattern.tolist(),
                'image_points': image_points_cameras_list,
                'image_sizes': image_sizes,
                'model': models,
                }, wf)

    Ks = None
    distortions = None
    if intrinsics_dir:
        # Read camera instrinsic matrices (Ks) and dictortions
        Ks, distortions = [], []
        for cam_id in cam_ids:
            input_file = os.path.join(intrinsics_dir, f"cameraParameters_{cam_id}.xml")
            print("Reading intrinsics from", input_file)
            storage = cv.FileStorage(input_file, cv.FileStorage_READ)
            camera_matrix = storage.getNode('cameraMatrix').mat()
            dist_coeffs = storage.getNode('dist_coeffs').mat()
            Ks.append(camera_matrix)
            distortions.append(dist_coeffs)
        find_intrinsics_in_python = True

    return calibrateFromPoints(
        pattern,
        image_points_cameras,
        image_sizes,
        models,
        all_images_names,
        find_intrinsics_in_python,
        Ks=Ks,
        distortions=distortions,
    )


def calibrateFromJSON(json_file, find_intrinsics_in_python):
    assert os.path.exists(json_file)
    data = json.load(open(json_file, 'r'))

    for i in range(len(data['image_points'])):
        for j in range(len(data['image_points'][i])):
            data['image_points'][i][j] = np.array(data['image_points'][i][j], dtype=np.float32)

    Ks = data['Ks'] if 'Ks' in data else None
    distortions = data['distortions'] if 'distortions' in data else None
    images_names = data['images_names'] if 'images_names' in data else None

    return calibrateFromPoints(
        np.array(data['object_points'], dtype=np.float32).T,
        data['image_points'],
        data['image_sizes'],
        data['models'],
        images_names,
        find_intrinsics_in_python,
        Ks,
        distortions,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default=None, help="json file with all data. Must have keys: 'object_points', 'image_points', 'image_sizes', 'is_fisheye'")
    parser.add_argument('--filenames', type=str, default=None, help='Txt files containg image lists, e.g., cam_1.txt,cam_2.txt,...,cam_N.txt for N cameras')
    parser.add_argument('--pattern_size', type=str, default=None, help='pattern size: width,height')
    parser.add_argument('--pattern_type', type=str, default=None, help='supported: checkerboard, circles, acircles, charuco')
    parser.add_argument('--is_fisheye', type=str, default=None, help='is_ mask, e.g., 0,1,...')
    parser.add_argument('--pattern_distance', type=float, default=None, help='distance between object / pattern points')
    parser.add_argument('--find_intrinsics_in_python', required=False, action='store_true', help='calibrate intrinsics in Python sample instead of C++')
    parser.add_argument('--winsize', type=str, default='5,5', help='window size for corners detection: w,h')
    parser.add_argument('--debug_corners', required=False, action='store_true', help='debug flag for corners detection visualization of images')
    parser.add_argument('--points_json_file', type=str, default='', help='if path is provided then image and object points will be saved to JSON file.')
    parser.add_argument('--path_to_save', type=str, default='', help='path and filename to save results in yaml file')
    parser.add_argument('--path_to_visualize', type=str, default='', help='path to results pickle file needed to run visualization')
    parser.add_argument('--visualize', required=False, action='store_true', help='visualization flag. If set, only runs visualization but path_to_visualize must be provided')
    parser.add_argument('--resize_image_detection', required=False, action='store_true', help='If set, an image will be resized to speed-up corners detection')
    parser.add_argument('--intrinsics_dir', type=str, default='', help='Path to measured intrinsics')
    parser.add_argument('--gt_file', type=str, default=None, help="ground truth")
    parser.add_argument('--board_dict_path', type=str, default=None, help="path to parameters of board dictionary")

    params, _ = parser.parse_known_args()
    print("params.board_dict_path:", params.board_dict_path)

    if params.visualize:
        assert os.path.exists(params.path_to_visualize), f'Path to result file does not exist: {params.path_to_visualize}'
        visualizeFromFile(params.path_to_visualize)
        sys.exit(0)

    if params.filenames is None:
        cam_files = sorted(glob.glob('cam_*.txt'))
        params.filenames = ','.join(cam_files)
        print('Found camera filenames:', params.filenames)
        params.is_fisheye = ','.join('0' * len(cam_files))
        print('Fisheye parameters:', params.is_fisheye)  # TODO: Calculate it automatically

    if params.json_file is not None:
        output = calibrateFromJSON(params.json_file, params.find_intrinsics_in_python)
        cam_ids = [str(x) for x in range(len(output['Rs']))]
        output['cam_ids'] = cam_ids
    else:
        print(params.pattern_size)
        if (params.pattern_type is None or params.pattern_size is None or params.pattern_distance is None):
            assert False and 'Either json file or all other parameters must be set'

        # cam_N.txt --> cam_N --> N
        cam_ids = [os.path.splitext(f)[0].split('_')[-1] for f in params.filenames.split(',')]

        output = calibrateFromImages(
            files_with_images=[x.strip() for x in params.filenames.split(',')],
            grid_size=[int(v) for v in params.pattern_size.split(',')],
            pattern_type=params.pattern_type,
            models=[int(v) for v in params.is_fisheye.split(',')],
            dist_m=params.pattern_distance,
            winsize=tuple([int(v) for v in params.winsize.split(',')]),
            points_json_file=params.points_json_file,
            debug_corners=params.debug_corners,
            RESIZE_IMAGE=params.resize_image_detection,
            find_intrinsics_in_python=params.find_intrinsics_in_python,
            cam_ids=cam_ids,
            intrinsics_dir=params.intrinsics_dir,
            board_dict_path=params.board_dict_path,
        )
        output['cam_ids'] = cam_ids

    # Evaluate the error
    if params.gt_file is not None:
        assert os.path.exists(params.gt_file), f'Path to gt file does not exist: {params.gt_file}'
        compareGT(params.gt_file, **output)

    if params.visualize:
        visualizeResults(**output)

    print('Saving:', params.path_to_save)
    saveToFile(params.path_to_save, **output)
