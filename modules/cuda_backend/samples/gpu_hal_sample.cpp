// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
 * gpu_hal_demo.cpp
 *
 * Shows the TWO ways to run an image operation on the GPU, and
 * confirms they give the same result:
 *
 *   (1) Direct CUDA - you manage a GpuMat and call cv::cuda::<op>
 *   (2) GPU HAL     - you call the ordinary cv::<op> on a GPU image
 *                     (the same code you'd write for the CPU)
 *
 * -------------------------------------------------------------
 * HOW TO BUILD  (run from /home/user/workspace):
 *
 *   g++ -std=c++14 -O2 gpu_hal_demo.cpp \
 *       -I opencv/modules/core/include \
 *       -I opencv/modules/imgproc/include \
 *       -I opencv_contrib/modules/cudawarping/include \
 *       -I opencv_contrib/modules/cudafilters/include \
 *       -I opencv_contrib/modules/cudaimgproc/include \
 *       -I opencv_contrib/modules/cudaarithm/include \
 *       -I build_cuda \
 *       build_cuda/lib/libopencv_core.so \
 *       build_cuda/lib/libopencv_imgproc.so \
 *       build_cuda/lib/libopencv_cudawarping.so \
 *       build_cuda/lib/libopencv_cudafilters.so \
 *       build_cuda/lib/libopencv_cudaimgproc.so \
 *       build_cuda/lib/libopencv_cudaarithm.so \
 *       -Wl,-rpath,build_cuda/lib \
 *       -o gpu_hal_demo
 *
 * HOW TO RUN:
 *
 *   OPENCV_GPU_BACKEND_PATH=build_cuda/lib/libopencv_cuda_backend.so \
 *       ./gpu_hal_demo
 *
 *   (OPENCV_GPU_BACKEND_PATH tells OpenCV where to find the CUDA
 *    plugin; without it the program prints "GPU backend not loaded".)
 */

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/core/hal/backend.hpp"
#include "opencv2/core/hal/backend_registry.hpp"
#include <cstdio>
#include <cstring>

// Copy a normal image into GPU memory so cv:: calls run on the GPU.
static cv::UMat uploadToGpu(const cv::Mat& img, cv::MatAllocator* gpu) {
    cv::UMat u;
    u.allocator = gpu;
    u.create(img.rows, img.cols, img.type());
    gpu->map(u.u, cv::ACCESS_WRITE);
    std::memcpy(u.u->data, img.data, u.u->size);
    gpu->unmap(u.u);
    return u;
}

// Copy a GPU image back to a normal image.
static cv::Mat downloadToHost(const cv::UMat& u, cv::MatAllocator* gpu) {
    gpu->map(u.u, cv::ACCESS_READ);
    return cv::Mat(u.rows, u.cols, u.type(), u.u->data, u.step[0]).clone();
}

// Print whether the two results are the same.
static void report(const char* op, const cv::Mat& viaCuda, const cv::Mat& viaHal) {
    cv::Mat diff;
    cv::absdiff(viaCuda, viaHal, diff);
    double maxDiff;
    cv::minMaxLoc(diff.reshape(1), nullptr, &maxDiff);
    printf("  %-13s direct CUDA %dx%d  |  GPU HAL %dx%d  |  same result: %s\n",
           op, viaCuda.cols, viaCuda.rows, viaHal.cols, viaHal.rows,
           maxDiff <= 4 ? "yes" : "NO");
}

int main()
{
    cv::MatAllocator* gpu = nullptr;
    if (cv::hal::Backend* b = cv::hal::findBackend())
        gpu = b->allocator();
    if (!gpu) {
        printf("GPU backend not loaded.\n");
        printf("Run with: OPENCV_GPU_BACKEND_PATH=build_cuda/lib/"
               "libopencv_cuda_backend.so ./gpu_hal_demo\n");
        return 1;
    }

    cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(60, 120, 200));
    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
    cv::Size newSize(640, 360);

    // ---- resize ----------------------------------------------------------
    {
        cv::cuda::GpuMat in, out;                       // (1) direct CUDA
        in.upload(image);
        cv::cuda::resize(in, out, newSize);
        cv::Mat viaCuda; out.download(viaCuda);

        cv::UMat g = uploadToGpu(image, gpu), gOut;     // (2) GPU HAL
        cv::resize(g, gOut, newSize);
        cv::Mat viaHal = downloadToHost(gOut, gpu);

        report("resize", viaCuda, viaHal);
    }

    // ---- GaussianBlur ----------------------------------------------------
    {
        cv::cuda::GpuMat in, out;                       // (1) direct CUDA
        in.upload(image);
        cv::Ptr<cv::cuda::Filter> blur =
            cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(15, 15), 0);
        blur->apply(in, out);
        cv::Mat viaCuda; out.download(viaCuda);

        cv::UMat g = uploadToGpu(image, gpu), gOut;     // (2) GPU HAL
        cv::GaussianBlur(g, gOut, cv::Size(15, 15), 0);
        cv::Mat viaHal = downloadToHost(gOut, gpu);

        report("GaussianBlur", viaCuda, viaHal);
    }

    // ---- cvtColor (color to gray) ---------------------------------------
    {
        cv::cuda::GpuMat in, out;                       // (1) direct CUDA
        in.upload(image);
        cv::cuda::cvtColor(in, out, cv::COLOR_BGR2GRAY);
        cv::Mat viaCuda; out.download(viaCuda);

        cv::UMat g = uploadToGpu(image, gpu), gOut;     // (2) GPU HAL
        cv::cvtColor(g, gOut, cv::COLOR_BGR2GRAY);
        cv::Mat viaHal = downloadToHost(gOut, gpu);

        report("cvtColor", viaCuda, viaHal);
    }

    // ---- threshold -------------------------------------------------------
    {
        cv::cuda::GpuMat in, out;                       // (1) direct CUDA
        in.upload(imageGray);
        cv::cuda::threshold(in, out, 128, 255, cv::THRESH_BINARY);
        cv::Mat viaCuda; out.download(viaCuda);

        cv::UMat g = uploadToGpu(imageGray, gpu), gOut; // (2) GPU HAL
        cv::threshold(g, gOut, 128, 255, cv::THRESH_BINARY);
        cv::Mat viaHal = downloadToHost(gOut, gpu);

        report("threshold", viaCuda, viaHal);
    }

    return 0;
}
