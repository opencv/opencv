// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_MATMULNBITS_KERNELS_HPP
#define OPENCV_DNN_MATMULNBITS_KERNELS_HPP

namespace cv { namespace dnn {

// Output columns [n_start, n_end) of Y = A @ dequant(B).T.
// wcol is caller-owned scratch of ceil(K/block_size)*block_size floats.
// zp is null for the symmetric form, where the zero point is 2^(bits-1).
void matMulNBitsStripe(const float* a, const uchar* bq, const float* sc, const uchar* zp,
                       float* y, int M, int N, int K, int block_size, int bits,
                       int n_start, int n_end, float* wcol);

// Below this per-block byte count the staged kernel is faster than the fused one.
int matMulNBitsGemvMinBlobBytes();

// Byte j holds weights 2j and 2j+1, so a_even[j] / a_odd[j] are what it multiplies.
void matMulNBitsSplitA(const float* a, float* a_even, float* a_odd, int pairs);

// Fused 4-bit M=1 GEMV: dequant stays in registers, no column is materialized.
void matMulNBitsGemv(const float* a_even, const float* a_odd, const uchar* bq,
                     const float* sc, const uchar* zp, float* y, int K, int block_size,
                     int n_start, int n_end);

}} // cv::dnn

#endif // OPENCV_DNN_MATMULNBITS_KERNELS_HPP
