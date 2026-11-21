// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "matmulnbits_kernels.hpp"
#include "matmulnbits_kernels.simd.hpp"
#include "layers/cpu_kernels/matmulnbits_kernels.simd_declarations.hpp"

namespace cv { namespace dnn {

void matMulNBitsStripe(const float* a, const uchar* bq, const float* sc, const uchar* zp,
                       float* y, int M, int N, int K, int block_size, int bits,
                       int n_start, int n_end, float* wcol)
{
    CV_CPU_DISPATCH(matMulNBitsStripe_, (a, bq, sc, zp, y, M, N, K, block_size, bits,
                                         n_start, n_end, wcol), CV_CPU_DISPATCH_MODES_ALL);
}

int matMulNBitsGemvMinBlobBytes()
{
    CV_CPU_DISPATCH(matMulNBitsGemvMinBlobBytes_, (), CV_CPU_DISPATCH_MODES_ALL);
}

void matMulNBitsSplitA(const float* a, float* a_even, float* a_odd, int pairs)
{
    CV_CPU_DISPATCH(matMulNBitsSplitA_, (a, a_even, a_odd, pairs), CV_CPU_DISPATCH_MODES_ALL);
}

void matMulNBitsGemv(const float* a_even, const float* a_odd, const uchar* bq,
                     const float* sc, const uchar* zp, float* y, int K, int block_size,
                     int n_start, int n_end)
{
    CV_CPU_DISPATCH(matMulNBitsGemv_, (a_even, a_odd, bq, sc, zp, y, K, block_size,
                                       n_start, n_end), CV_CPU_DISPATCH_MODES_ALL);
}

}} // cv::dnn
