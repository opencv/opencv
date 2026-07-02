#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# gpu_hal_chain_demo.py
#
# Runs the same chain of image operations three ways and compares them:
#     resize -> GaussianBlur -> cvtColor -> threshold
#
#   (1) CPU         - ordinary cv2.<op> on a normal image
#   (2) Direct CUDA - cv2.cuda.<op> on a GpuMat (GPU-specific code)
#   (3) GPU HAL     - ordinary cv2.<op> on a GPU image (same as CPU code)
#
# HOW TO RUN (from /home/user/workspace):
#
#   LD_LIBRARY_PATH=build_cuda/lib \
#   PYTHONPATH=build_cuda/lib/python3 \
#   OPENCV_GPU_BACKEND_PATH=build_cuda/lib/libopencv_cuda_backend.so \
#   python3 gpu_hal_chain_demo.py
#
#   (The three settings point Python at the GPU build of OpenCV and tell
#    it where to find the CUDA plugin.)

import time
import numpy as np
import cv2

image = np.full((4320, 7680, 3), (60, 120, 200), np.uint8)   # an 8K image
newSize = (3840, 2160)
blurSize = (15, 15)
REPEAT = 30


def cpu_chain(img):
    a = cv2.resize(img, newSize)
    a = cv2.GaussianBlur(a, blurSize, 0)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    _, a = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY)
    return a


def cuda_chain(gpu_image):                  # gpu_image: cv2.cuda.GpuMat
    g = cv2.cuda.resize(gpu_image, newSize)
    g = blur.apply(g)
    g = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY)
    _, g = cv2.cuda.threshold(g, 128, 255, cv2.THRESH_BINARY)
    return g.download()


def hal_chain(gpu_image):                   # gpu_image: GPU image (UMat)
    g = cv2.resize(gpu_image, newSize)
    g = cv2.GaussianBlur(g, blurSize, 0)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    _, g = cv2.threshold(g, 128, 255, cv2.THRESH_BINARY)
    out = cv2.gpuDownload(g)
    return out.get() if isinstance(out, cv2.UMat) else out


def average_ms(run, make_input):
    arg = make_input()
    for _ in range(5):
        run(arg)
    start = time.perf_counter()
    for _ in range(REPEAT):
        run(arg)
    return (time.perf_counter() - start) * 1000 / REPEAT


# the Gaussian filter object the direct-CUDA path needs
blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, blurSize, 0)

# put the image in GPU memory (one copy each), kept ready for the runs
def gpumat():
    g = cv2.cuda.GpuMat(); g.upload(image); return g

def gpu_image():
    return cv2.gpuUpload(image)

# check all three produce the same picture
cpu  = cpu_chain(image)
cuda = cuda_chain(gpumat())
hal  = hal_chain(gpu_image())
same = (np.abs(cpu.astype(int) - cuda.astype(int)).max() <= 4 and
        np.abs(cpu.astype(int) - hal.astype(int)).max() <= 4)
print("all three give the same result:", "yes" if same else "NO", "\n")

# time them
cpu_ms  = average_ms(lambda _: cpu_chain(image), lambda: None)
cuda_ms = average_ms(cuda_chain, gpumat)
hal_ms  = average_ms(hal_chain, gpu_image)

print(f"  CPU          {cpu_ms:7.2f} ms")
print(f"  Direct CUDA  {cuda_ms:7.2f} ms")
print(f"  GPU HAL      {hal_ms:7.2f} ms")
