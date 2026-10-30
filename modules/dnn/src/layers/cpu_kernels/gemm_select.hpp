// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_GEMM_SELECT_HPP
#define OPENCV_DNN_GEMM_SELECT_HPP

#include <opencv2/core.hpp>

namespace cv { namespace dnn {

// GEMM implementation for fp32 layers, from OPENCV_DNN_GEMM_PATH=engine|mlas|auto (read once).
// "mlas" keeps the previous dnn behaviour: MLAS where it is built, the dnn kernels elsewhere.
enum GemmPath { GEMM_PATH_AUTO, GEMM_PATH_ENGINE, GEMM_PATH_MLAS };
GemmPath gemmPath();

// True when an M x N x K product should run on the core GEMM engine (cv::hal::gemm32f*).
// constB: B is a constant packed once. Always false off AArch64, so other platforms keep their paths.
bool gemmUseEngine(int M, int N, int K, bool constB);

// True when the engine should replace the dnn kernels (no MLAS involved): AArch64 unless the path is "mlas".
bool gemmEngineEnabled();

// Constant B packed once for the core GEMM engine: `count` slices of the same n x k shape.
class EngineGemmB
{
public:
    bool empty() const { return count == 0; }
    void pack(const float* B, size_t B_step, int n, int k, bool transB, int count = 1, size_t sliceElems = 0);
    void release();
    // C = alpha*op(A)*B_i + beta*C for slice i.
    void gemm(int i, const float* A, size_t A_step, bool transA, float* C, size_t C_step, int M,
              float alpha, float beta, bool multiThread) const;

    int n = 0, k = 0, count = 0;
private:
    Mat buf;
    size_t stride = 0;
};

}} // cv::dnn

#endif // OPENCV_DNN_GEMM_SELECT_HPP
