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

// Where Y[r,n] goes: row-major with row stride ldy, or, with headD set, the [B,S,H,D] reshape of
// [B*S, H*headD] transposed to [B,H,S,D] (rows in blocks of headS); headD is a multiple of NR
struct MatMulInt8OutLayout
{
    size_t ldy = 0;
    int headS = 0, headD = 0, heads = 0;

    explicit MatMulInt8OutLayout(size_t ldy_) : ldy(ldy_) {}
    MatMulInt8OutLayout(int headS_, int heads_, int headD_)
        : headS(headS_), headD(headD_), heads(heads_) {}
    size_t rowOfs(int r) const
    {
        return headD ? ((size_t)(r / headS) * heads * headS + r % headS) * headD : (size_t)r * ldy;
    }
    size_t colOfs(int n) const
    {
        return headD ? (size_t)(n / headD) * headS * headD + n % headD : (size_t)n;
    }
};

// Packs nb [K, N] matrices of B - delta, one after another, into the panels matMulInt8Int32 reads,
// and writes their column sums; both buffers are zeroed by the caller and padded as described below
void matMulInt8PackB(const uchar* B, int nb, int K, int N, int delta, int8_t* packedB, int* colsum);

// Y[r,n] = sum_k (A[r,k] - a_zp) * B[k,n] + corr[n] * rowterm[r] in int32, corr may be null;
// packedB has N/NR panels (rounded up) of alignSize(K, KR) * NR bytes; colsum and corr pad N alike
// With scale set, Y is float: Y[r,n] * scale[n] + bias[n]; scale and bias pad N alike
void matMulInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                     const int* corr, const int* rowterm, void* Y, const MatMulInt8OutLayout& lay,
                     int M, int N, int K, int a_zp, bool aIsU8,
                     const float* scale = nullptr, const float* bias = nullptr);

}
}

#endif
