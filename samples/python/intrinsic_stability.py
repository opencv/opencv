#!/usr/bin/env python3

"""
This script evaluates intrinsic parameter stability using repeated calibration
on random frame subsets. Generates statistical summaries and plots for intrinsics,
distortion coefficients, and reprojection errors.
"""

import argparse
from pathlib import Path
import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import json


def load_data_from_yaml(yaml_path):
    fs = cv.FileStorage(str(yaml_path), cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Could not open {yaml_path}")
    img_pts_node = fs.getNode("image_points")
    ptn_pts_node = fs.getNode("pattern_points")
    img_sz_node = fs.getNode("image_sizes")

    image_points = img_pts_node.mat()
    pattern_points = ptn_pts_node.mat()
    image_sizes = img_sz_node.mat()
    fs.release()

    num_cams = int(image_sizes.shape[0])
    num_corners = int(pattern_points.shape[0])
    # reshape if image points are flattened
    if image_points.ndim == 3:
        total_rows = image_points.shape[0]
        num_frames = total_rows // (num_cams * num_corners)
        image_points = image_points.reshape(num_cams, num_frames, num_corners, 2)

    return image_points, pattern_points, image_sizes


def _one_run(img_pts, obj_pts, img_size, subset_size):
    # select random subset of frames
    num_frames = img_pts.shape[0]
    sel = sorted(random.sample(range(num_frames), subset_size))
    sub_img = [img_pts[i] for i in sel]
    sub_obj = [obj_pts for _ in sel]
    # run calibration on subset
    ret = cv.calibrateCamera(sub_obj, sub_img, tuple(img_size), None, None)
    rms, K, dist, _rvecs, _tvecs = ret
    return {"K": K, "dist": dist.flatten(), "rms": rms}


def sim_intrinsic_stability(
    img_pts, obj_pts, img_size, n_runs, subset_ratio, jobs, seed
):
    random.seed(seed)
    subset_size = max(3, int(subset_ratio * img_pts.shape[0]))
    # run simulations in parallel
    results = Parallel(n_jobs=jobs)(
        delayed(_one_run)(img_pts, obj_pts, img_size, subset_size)
        for _ in tqdm(range(n_runs), desc="Sim runs")
    )
    # stack results
    Ks = np.stack([r["K"] for r in results], axis=0)
    Dists = np.stack([r["dist"] for r in results], axis=0)
    Errs = np.array([r["rms"] for r in results])
    return Ks, Dists, Errs


def analyze_and_plot(yaml_path, n_runs, subset_ratio, jobs, out_dir, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_points, pattern_points, image_sizes = load_data_from_yaml(yaml_path)
    num_cams = image_points.shape[0]
    summary = {}

    for cam_idx in range(num_cams):
        # run stability simulation for each camera
        Ks, Dists, Errs = sim_intrinsic_stability(
            image_points[cam_idx],
            pattern_points,
            image_sizes[cam_idx],
            n_runs,
            subset_ratio,
            jobs,
            seed,
        )

        fx, fy = Ks[:, 0, 0], Ks[:, 1, 1]
        cx, cy = Ks[:, 0, 2], Ks[:, 1, 2]
        k1, k2, p1, p2 = Dists[:, :4].T  # first 4 distortion coefficients

        # compute mean and std for intrinsics and distortion
        stats = {
            "fx": (float(fx.mean()), float(fx.std())),
            "fy": (float(fy.mean()), float(fy.std())),
            "cx": (float(cx.mean()), float(cx.std())),
            "cy": (float(cy.mean()), float(cy.std())),
            "k1": (float(k1.mean()), float(k1.std())),
            "k2": (float(k2.mean()), float(k2.std())),
            "p1": (float(p1.mean()), float(p1.std())),
            "p2": (float(p2.mean()), float(p2.std())),
            "rms": (float(Errs.mean()), float(Errs.std())),
        }
        summary[f"camera_{cam_idx}"] = stats

        # plot intrinsics boxplot
        plt.figure(figsize=(8, 4))
        plt.boxplot([fx, fy, cx, cy], tick_labels=["fx", "fy", "cx", "cy"])
        plt.title(f"Cam {cam_idx} Intrinsic")
        plt.ylabel("pixels")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"cam{cam_idx}_intrinsics.png")
        plt.close()

        # plot distortion boxplot
        plt.figure(figsize=(8, 4))
        plt.boxplot([k1, k2, p1, p2], tick_labels=["k1", "k2", "p1", "p2"])
        plt.title(f"Cam {cam_idx} Distortion")
        plt.ylabel("coeff value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"cam{cam_idx}_distortion.png")
        plt.close()

        # plot histogram of RMS error
        plt.figure(figsize=(6, 3))
        plt.hist(Errs, bins=20)
        plt.title(f"Cam {cam_idx} RMS Errors")
        plt.xlabel("RMS error (px)")
        plt.tight_layout()
        plt.savefig(out_dir / f"cam{cam_idx}_rms_hist.png")
        plt.close()

    with open(out_dir / "stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved under {out_dir}")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Intrinsic Stability & QA/QC")
    p.add_argument("--yaml", required=True, help="Path to calibration YAML")
    p.add_argument("--runs", type=int, default=30, help="Number of random runs")
    p.add_argument("--ratio", type=float, default=0.7, help="Frame‐subset ratio")
    p.add_argument(
        "--jobs", type=int, default=1, help="Parallel jobs (-1 for all cores)"
    )
    p.add_argument("--out", default="stability_out", help="Output folder")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_and_plot(args.yaml, args.runs, args.ratio, args.jobs, args.out, args.seed)
