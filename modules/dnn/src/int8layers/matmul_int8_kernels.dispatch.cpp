// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "matmul_int8_kernels.hpp"

#include "matmul_int8_kernels.simd.hpp"
#include "int8layers/matmul_int8_kernels.simd_declarations.hpp"

namespace cv
{
namespace dnn
{

void matMulInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                     const int* corr, const int* rowterm, int* Y, size_t ldy,
                     int M, int N, int K, int a_zp, bool aIsU8)
{
    CV_CPU_DISPATCH(matMulInt8Int32, (A, lda, packedB, colsum, corr, rowterm, Y, ldy,
                                      M, N, K, a_zp, aIsU8),
                    CV_CPU_DISPATCH_MODES_ALL);
}

}
}
