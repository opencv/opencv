'''
Modular monocular SLAM: tracking + local/global bundle adjustment + loop closure.

Feeds a directory of images through a pluggable feature detector + matcher pair,
estimates the camera trajectory, and writes the result to --output in COLMAP text
format. This sample wires up one ONNX detector/matcher pair by default; the
pipeline (cv.slam.VisualOdometry) accepts any cv2.Feature2D + cv2.DescriptorMatcher.

Example:
    python slam.py --aliked aliked.onnx --lightglue lg.onnx --images ./seq
Sample command (run on the GPU):
    python slam.py --aliked aliked.onnx --lightglue lg.onnx --images ./seq --target cuda
'''

import argparse
import sys
import time
import cv2 as cv

from slam_common import build_K, parse_dist_coeffs, list_image_files, write_colmap_files

BACKENDS = {
    'default': cv.dnn.DNN_BACKEND_DEFAULT,
    'openvino': cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
    'opencv': cv.dnn.DNN_BACKEND_OPENCV,
    'vkcom': cv.dnn.DNN_BACKEND_VKCOM,
    'cuda': cv.dnn.DNN_BACKEND_CUDA,
    'webnn': cv.dnn.DNN_BACKEND_WEBNN,
}

TARGETS = {
    'cpu': cv.dnn.DNN_TARGET_CPU,
    'opencl': cv.dnn.DNN_TARGET_OPENCL,
    'opencl_fp16': cv.dnn.DNN_TARGET_OPENCL_FP16,
    'vpu': cv.dnn.DNN_TARGET_MYRIAD,
    'vulkan': cv.dnn.DNN_TARGET_VULKAN,
    'cuda': cv.dnn.DNN_TARGET_CUDA,
    'cuda_fp16': cv.dnn.DNN_TARGET_CUDA_FP16,
}


def main():
    parser = argparse.ArgumentParser(description='Modular monocular SLAM')
    parser.add_argument('--aliked',    required=True,
                        help='Path to detector ONNX model')
    parser.add_argument('--lightglue', required=True,
                        help='Path to matcher ONNX model')
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
    parser.add_argument('--dist', default='',
                        help='Lens distortion coeffs, comma-separated k1,k2,p1,p2[,k3,...] (default: none)')
    parser.add_argument('--backend', default='default', choices=sorted(BACKENDS),
                        help='Choose a computation backend (default: default)')
    parser.add_argument('--target', default='cpu', choices=sorted(TARGETS),
                        help='Choose a target computation device (default: cpu)')
    parser.add_argument('--progress', type=lambda s: s.lower() != 'false', default=True,
                        help='Print per-frame progress logs to the console as they happen (default: true)')
    args = parser.parse_args()

    if args.progress:
        cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_INFO)

    backend_id = BACKENDS[args.backend]
    target_id = TARGETS[args.target]

    det_params = cv.ALIKED.Params()
    det_params.inputSize = (640, 640)
    det_params.engine    = cv.dnn.ENGINE_AUTO
    det_params.backend   = backend_id
    det_params.target    = target_id
    detector = cv.ALIKED.create(args.aliked, det_params)

    matcher = cv.LightGlueMatcher.create(args.lightglue, 0.0, backend_id, target_id)

    vo_params = cv.slam.OdometryParams()
    vo_params.minInitParallaxDeg = 1.5
    vo_params.minInitPoints      = 50
    vo_params.pnpReprojThresh    = 4.0
    vo_params.kfMaxFrames        = 30
    vo_params.localMapTopK       = 10

    K = build_K(args.fx, args.fy, args.cx, args.cy)
    dist_coeffs = parse_dist_coeffs(args.dist)

    vo = cv.slam.VisualOdometry.create(
        detector, matcher, K, dist_coeffs, vo_params)

    image_files = list_image_files(args.images)
    if not image_files:
        print(f"no images found in {args.images}", file=sys.stderr)
        return 1

    progress_line = ("Per-frame progress is printed below (OpenCV logs to stderr)."
                      " Pass --progress=false to silence it.\n"
                      if args.progress else
                      "Per-frame progress is disabled."
                      " Re-run with --progress=true (the default) to see it.\n")
    print(f"images folder : {args.images}\n"
          f"output folder : {args.output}\n"
          f"images found  : {len(image_files)}\n\n"
          f"Running tracking + local/global BA + loop closure.\n"
          f"{progress_line}")

    # Tracks which input image each emitted trajectory pose came from, for images.txt.
    pose_filenames = []
    prev_traj_len = 0
    ref_filename = None
    image_size = (0, 0)

    t0 = time.perf_counter()
    for i, path in enumerate(image_files):
        img = cv.imread(path)
        if img is None:
            print(f"[frame {i}] imread failed: {path}", file=sys.stderr)
            continue
        image_size = (img.shape[1], img.shape[0])

        before = vo.getState()
        vo.processFrame(img)
        after = vo.getState()

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

    # End-of-sequence global bundle adjustment over the whole map.
    vo.finalize()
    elapsed = time.perf_counter() - t0

    ok = len(vo.getTrajectory()) > 0
    exported = False
    if ok and args.output:
        # Corrected poses ride on the optimised keyframe graph, so they carry the
        # loop-closure and bundle-adjustment corrections; getTrajectory() is the raw
        # per-frame log.
        write_colmap_files(vo, K, dist_coeffs, image_size, pose_filenames, args.output,
                            vo.getCorrectedTrajectory())
        exported = True

    print("\n"
          "==================== SLAM Result ====================\n"
          f"status        : {'OK' if ok else 'FAILED'}\n"
          f"camera poses  : {len(vo.getTrajectory())}\n"
          f"keyframes     : {vo.getNumKeyframes()}\n"
          f"map points    : {vo.getNumMapPoints()}\n"
          f"elapsed time  : {elapsed:.2f} s")
    if exported:
        print(f"output        : {args.output}/{{cameras,images,points3D}}.txt")
    print("======================================================")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
