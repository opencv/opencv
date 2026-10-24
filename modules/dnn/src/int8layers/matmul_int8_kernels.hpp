// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_INT8LAYERS_MATMUL_INT8_KERNELS_HPP__
#define __OPENCV_DNN_INT8LAYERS_MATMUL_INT8_KERNELS_HPP__

namespace cv
{
namespace dnn
{

enum { MATMUL_INT8_NR = 16, MATMUL_INT8_KR = 4 };

// Y[r,n] = sum_k (A[r,k] - a_zp) * B[k,n] + corr[n] * rowterm[r] in int32, corr may be null;
// packedB is int8 B as [N/NR][K/KR][NR][KR] rounded up and zero-padded; colsum and corr pad N alike
void matMulInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                     const int* corr, const int* rowterm, int* Y, size_t ldy,
                     int M, int N, int K, int a_zp, bool aIsU8);

}
}

#endif
