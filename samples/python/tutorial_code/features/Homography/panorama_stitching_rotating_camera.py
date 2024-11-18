#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def basicPanoramaStitching(img1Path, img2Path):
    img1 = cv.imread(cv.samples.findFile(img1Path))
    img2 = cv.imread(cv.samples.findFile(img2Path))

    # [camera-pose-from-Blender-at-location-1]
    c1Mo = np.array([[0.9659258723258972, 0.2588190734386444, 0.0, 1.5529145002365112],
                     [ 0.08852133899927139, -0.3303661346435547, -0.9396926164627075, -0.10281121730804443],
                     [-0.24321036040782928, 0.9076734185218811, -0.342020183801651, 6.130080699920654],
                     [0, 0, 0, 1]],dtype=np.float64)
    # [camera-pose-from-Blender-at-location-1]

    # [camera-pose-from-Blender-at-location-2]
    c2Mo = np.array([[0.9659258723258972, -0.2588190734386444, 0.0, -1.5529145002365112],
                     [-0.08852133899927139, -0.3303661346435547, -0.9396926164627075, -0.10281121730804443],
                     [0.24321036040782928, 0.9076734185218811, -0.342020183801651, 6.130080699920654],
                     [0, 0, 0, 1]],dtype=np.float64)
    # [camera-pose-from-Blender-at-location-2]

    # [camera-intrinsics-from-Blender]
    cameraMatrix = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0, 0, 1]], dtype=np.float32)
    # [camera-intrinsics-from-Blender]

    # [extract-rotation]
    R1 = c1Mo[0:3, 0:3]
    R2 = c2Mo[0:3, 0:3]
    #[extract-rotation]

    # [compute-rotation-displacement]
    R2 = R2.transpose()
    R_2to1 = np.dot(R1,R2)
    # [compute-rotation-displacement]

    # [compute-homography]
    H = cameraMatrix.dot(R_2to1).dot(np.linalg.inv(cameraMatrix))
    H = H / H[2][2]
    # [compute-homography]

    # [stitch]
    img_stitch = cv.warpPerspective(img2, H, (img2.shape[1]*2, img2.shape[0]))
    img_stitch[0:img1.shape[0], 0:img1.shape[1]] = img1
    # [stitch]

    img_space = np.zeros((img1.shape[0],50,3), dtype=np.uint8)
    img_compare = cv.hconcat([img1,img_space, img2])

    cv.imshow("Final", img_compare)
    cv.imshow("Panorama", img_stitch)
    cv.waitKey(0)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Code for homography tutorial. Example 5: basic panorama stitching from a rotating camera.")
    parser.add_argument("-I1","--image1", help = "path to first image", default="Blender_Suzanne1.jpg")
    parser.add_argument("-I2","--image2", help = "path to second image", default="Blender_Suzanne2.jpg")
    args = parser.parse_args()
    print("Panorama Stitching Started")
    basicPanoramaStitching(args.image1, args.image2)
    print("Panorama Stitching Completed Successfully")


if __name__ == '__main__':
    main()
