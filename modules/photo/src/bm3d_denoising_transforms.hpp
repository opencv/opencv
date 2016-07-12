/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_BM3D_DENOISING_TRANSFORMS_HPP__
#define __OPENCV_BM3D_DENOISING_TRANSFORMS_HPP__

using namespace cv;

template <typename T>
inline static void shrink(T &val, T &nonZeroCount, const T &threshold)
{
    if (std::abs(val) < threshold)
        val = 0;
    else
        ++nonZeroCount;
}

template <typename T>
inline static void hardThreshold2D(T *dst, T *thrMap, const int &templateWindowSizeSq)
{
    for (int i = 1; i < templateWindowSizeSq; ++i)
    {
        if (std::abs(dst[i] < thrMap[i]))
            dst[i] = 0;
    }
}

/// 1D and 2D threshold map coefficients. Implementation dependent, thus stored
/// together with transforms.

#define BM3D_MAX_3D_SIZE 8

const float sqrt2 = std::sqrt(2.0f);

// 2D map of threshold multipliers in case of 4x4 block size
static float kThrMap4x4[16] = {
    0.25f,           0.5f,       sqrt2 / 2.0f,    sqrt2 / 2.0f,
    0.5f,            1.0f,       sqrt2,           sqrt2,
    sqrt2 / 2.0f,    sqrt2,      2.0f,            2.0f,
    sqrt2 / 2.0f,    sqrt2,      2.0f,            2.0f
};

// 2D map of threshold multipliers in case of 8x8 block size
static float kThrMap8x8[64] = {
    0.125f,       0.25f,        sqrt2 / 4.0f, sqrt2 / 4.0f, 0.5f,  0.5f,  0.5f,  0.5f,
    0.25f,        0.5f,         sqrt2 / 2.0f, sqrt2 / 2.0f, 1.0f,  1.0f,  1.0f,  1.0f,
    sqrt2 / 4.0f, sqrt2 / 2.0f, 1.0f,         1.0f,         sqrt2, sqrt2, sqrt2, sqrt2,
    sqrt2 / 4.0f, sqrt2 / 2.0f, 1.0f,         1.0f,         sqrt2, sqrt2, sqrt2, sqrt2,
    0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f,
    0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f,
    0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f,
    0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f
};

// 1D map of threshold multipliers for up to 8 elements in a group
static const float kThrMap1D[(BM3D_MAX_3D_SIZE << 1) - 1] = {
    1.0f,  // 1 element
    sqrt2 / 2.0f,    sqrt2, // 2 elements
    0.5f,            1.0f,            sqrt2,       sqrt2,  // 4 elements
    sqrt2 / 4.0f,    sqrt2 / 2.0f,    1.0f,        1.0f,  sqrt2, sqrt2, sqrt2, sqrt2  // 8 elements
};

// 2D threshold multipliers for up to 8 elements in a group
static const float kCoeff[4] = {
    1.0f,                             // 1 element
    std::sqrt(2.0f * std::log(2.0f)), // 2 elements
    std::sqrt(2.0f * std::log(4.0f)), // 4 elements
    std::sqrt(2.0f * std::log(8.0f))  // 8 elements
};

/// Transforms for 4x4 2D block

// Forward transform 4x4 block
template <typename T, typename TT>
inline static void HaarColumn4x4(const T *src, TT *dst, const int &step)
{
    const T *src0 = src;
    const T *src1 = src + 1 * step;
    const T *src2 = src + 2 * step;
    const T *src3 = src + 3 * step;

    TT sum0 = (*src0 + *src1 + 1) >> 1;
    TT sum1 = (*src2 + *src3 + 1) >> 1;
    TT dif0 = *src0 - *src1;
    TT dif1 = *src2 - *src3;

    TT sum00 = (sum0 + sum1 + 1) >> 1;
    TT dif00 = sum0 - sum1;

    dst[0 * 4] = sum00;
    dst[1 * 4] = dif00;
    dst[2 * 4] = dif0;
    dst[3 * 4] = dif1;
}

template <typename TT>
inline static void HaarRow4x4(const TT *src, TT *dst)
{
    TT sum0 = (src[0] + src[1] + 1) >> 1;
    TT sum1 = (src[2] + src[3] + 1) >> 1;
    TT dif0 = src[0] - src[1];
    TT dif1 = src[2] - src[3];

    TT sum00 = (sum0 + sum1 + 1) >> 1;
    TT dif00 = sum0 - sum1;

    dst[0] = sum00;
    dst[1] = dif00;
    dst[2] = dif0;
    dst[3] = dif1;
}

template <typename T, typename TT>
inline static void Haar4x4(const T *ptr, TT *dst, const int &step)
{
    TT temp[16];

    // Transform columns first
    for (int i = 0; i < 4; ++i)
        HaarColumn4x4(ptr + i, temp + i, step);

    // Then transform rows
    for (int i = 0; i < 4; ++i)
        HaarRow4x4(temp + i * 4, dst + i * 4);
}

template <typename TT>
inline static void InvHaarColumn4x4(TT *src, TT *dst)
{
    TT src0 = src[0 * 4] * 2;
    TT src1 = src[1 * 4];
    TT src2 = src[2 * 4];
    TT src3 = src[3 * 4];

    TT sum0 = src0 + src1;
    TT dif0 = src0 - src1;

    dst[0 * 4] = (sum0 + src2) >> 1;
    dst[1 * 4] = (sum0 - src2) >> 1;
    dst[2 * 4] = (dif0 + src3) >> 1;
    dst[3 * 4] = (dif0 - src3) >> 1;
}

template <typename TT>
inline static void InvHaarRow4x4(TT *src, TT *dst)
{
    TT src0 = src[0] * 2;
    TT src1 = src[1];
    TT src2 = src[2];
    TT src3 = src[3];

    TT sum0 = src0 + src1;
    TT dif0 = src0 - src1;

    dst[0] = (sum0 + src2) >> 1;
    dst[1] = (sum0 - src2) >> 1;
    dst[2] = (dif0 + src3) >> 1;
    dst[3] = (dif0 - src3) >> 1;
}

template <typename TT>
inline static void InvHaar4x4(TT *src)
{
    TT temp[16];

    // Invert columns first
    for (int i = 0; i < 4; ++i)
        InvHaarColumn4x4(src + i, temp + i);

    // Then invert rows
    for (int i = 0; i < 4; ++i)
        InvHaarRow4x4(temp + i * 4, src + i * 4);
}

/// Transforms for 8x8 2D block

template <typename T, typename TT>
inline static void HaarColumn8x8(const T *src, TT *dst, const int &step)
{
    const T *src0 = src;
    const T *src1 = src + 1 * step;
    const T *src2 = src + 2 * step;
    const T *src3 = src + 3 * step;
    const T *src4 = src + 4 * step;
    const T *src5 = src + 5 * step;
    const T *src6 = src + 6 * step;
    const T *src7 = src + 7 * step;

    TT sum0 = (*src0 + *src1 + 1) >> 1;
    TT sum1 = (*src2 + *src3 + 1) >> 1;
    TT sum2 = (*src4 + *src5 + 1) >> 1;
    TT sum3 = (*src6 + *src7 + 1) >> 1;
    TT dif0 = *src0 - *src1;
    TT dif1 = *src2 - *src3;
    TT dif2 = *src4 - *src5;
    TT dif3 = *src6 - *src7;

    TT sum00 = (sum0 + sum1 + 1) >> 1;
    TT sum11 = (sum2 + sum3 + 1) >> 1;
    TT dif00 = sum0 - sum1;
    TT dif11 = sum2 - sum3;

    TT sum000 = (sum00 + sum11 + 1) >> 1;
    TT dif000 = sum00 - sum11;

    dst[0 * 8] = sum000;
    dst[1 * 8] = dif000;
    dst[2 * 8] = dif00;
    dst[3 * 8] = dif11;
    dst[4 * 8] = dif0;
    dst[5 * 8] = dif1;
    dst[6 * 8] = dif2;
    dst[7 * 8] = dif3;
}

template <typename TT>
inline static void HaarRow8x8(const TT *src, TT *dst)
{
    TT sum0 = (src[0] + src[1] + 1) >> 1;
    TT sum1 = (src[2] + src[3] + 1) >> 1;
    TT sum2 = (src[4] + src[5] + 1) >> 1;
    TT sum3 = (src[6] + src[7] + 1) >> 1;
    TT dif0 = src[0] - src[1];
    TT dif1 = src[2] - src[3];
    TT dif2 = src[4] - src[5];
    TT dif3 = src[6] - src[7];

    TT sum00 = (sum0 + sum1 + 1) >> 1;
    TT sum11 = (sum2 + sum3 + 1) >> 1;
    TT dif00 = sum0 - sum1;
    TT dif11 = sum2 - sum3;

    TT sum000 = (sum00 + sum11 + 1) >> 1;
    TT dif000 = sum00 - sum11;

    dst[0] = sum000;
    dst[1] = dif000;
    dst[2] = dif00;
    dst[3] = dif11;
    dst[4] = dif0;
    dst[5] = dif1;
    dst[6] = dif2;
    dst[7] = dif3;
}

template <typename T, typename TT>
inline static void Haar8x8(const T *ptr, TT *dst, const int &step)
{
    TT temp[64];

    // Transform columns first
    for (int i = 0; i < 8; ++i)
        HaarColumn8x8(ptr + i, temp + i, step);

    // Then transform rows
    for (int i = 0; i < 8; ++i)
        HaarRow8x8(temp + i * 8, dst + i * 8);
}

template <typename T>
inline static void InvHaarColumn8x8(T *src, T *dst)
{
    T src0 = src[0] * 2;
    T src1 = src[1 * 8];
    T src2 = src[2 * 8];
    T src3 = src[3 * 8];
    T src4 = src[4 * 8];
    T src5 = src[5 * 8];
    T src6 = src[6 * 8];
    T src7 = src[7 * 8];

    T sum0 = src0 + src1;
    T dif0 = src0 - src1;

    T sum00 = sum0 + src2;
    T dif00 = sum0 - src2;
    T sum11 = dif0 + src3;
    T dif11 = dif0 - src3;

    dst[0 * 8] = (sum00 + src4) >> 1;
    dst[1 * 8] = (sum00 - src4) >> 1;
    dst[2 * 8] = (dif00 + src5) >> 1;
    dst[3 * 8] = (dif00 - src5) >> 1;
    dst[4 * 8] = (sum11 + src6) >> 1;
    dst[5 * 8] = (sum11 - src6) >> 1;
    dst[6 * 8] = (dif11 + src7) >> 1;
    dst[7 * 8] = (dif11 - src7) >> 1;
}

template <typename T>
inline static void InvHaarRow8x8(T *src, T *dst)
{
    T src0 = src[0] * 2;
    T src1 = src[1];
    T src2 = src[2];
    T src3 = src[3];
    T src4 = src[4];
    T src5 = src[5];
    T src6 = src[6];
    T src7 = src[7];

    T sum0 = src0 + src1;
    T dif0 = src0 - src1;

    T sum00 = sum0 + src2;
    T dif00 = sum0 - src2;
    T sum11 = dif0 + src3;
    T dif11 = dif0 - src3;

    dst[0] = (sum00 + src4) >> 1;
    dst[1] = (sum00 - src4) >> 1;
    dst[2] = (dif00 + src5) >> 1;
    dst[3] = (dif00 - src5) >> 1;
    dst[4] = (sum11 + src6) >> 1;
    dst[5] = (sum11 - src6) >> 1;
    dst[6] = (dif11 + src7) >> 1;
    dst[7] = (dif11 - src7) >> 1;
}

template <typename TT>
inline static void InvHaar8x8(TT *src)
{
    TT temp[64];

    // Invert columns first
    for (int i = 0; i < 8; ++i)
        InvHaarColumn8x8(src + i, temp + i);

    // Then invert rows
    for (int i = 0; i < 8; ++i)
        InvHaarRow8x8(temp + i * 8, src + i * 8);
}

/// 1D forward transformations

template <typename T, typename DT, typename CT>
inline static short HaarTransformShrink2(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap)
{
    T sum = (z[0][n] + z[1][n] + 1) >> 1;
    T dif = z[0][n] - z[1][n];

    short nonZeroCount = 0;
    shrink(sum, nonZeroCount, *thrMap++);
    shrink(dif, nonZeroCount, *thrMap++);

    z[0][n] = sum;
    z[1][n] = dif;

    return nonZeroCount;
}

template <typename T, typename DT, typename CT>
inline static short HaarTransformShrink4(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap)
{
    T sum0 = (z[0][n] + z[1][n] + 1) >> 1;
    T sum1 = (z[2][n] + z[3][n] + 1) >> 1;
    T dif0 = z[0][n] - z[1][n];
    T dif1 = z[2][n] - z[3][n];

    T sum00 = (sum0 + sum1 + 1) >> 1;
    T dif00 = sum0 - sum1;

    short nonZeroCount = 0;
    shrink(sum00, nonZeroCount, *thrMap++);
    shrink(dif00, nonZeroCount, *thrMap++);
    shrink(dif0, nonZeroCount, *thrMap++);
    shrink(dif1, nonZeroCount, *thrMap++);

    z[0][n] = sum00;
    z[1][n] = dif00;
    z[2][n] = dif0;
    z[3][n] = dif1;

    return nonZeroCount;
}

template <typename T, typename DT, typename CT>
inline static short HaarTransformShrink8(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap)
{
    T sum0 = (z[0][n] + z[1][n] + 1) >> 1;
    T sum1 = (z[2][n] + z[3][n] + 1) >> 1;
    T sum2 = (z[4][n] + z[5][n] + 1) >> 1;
    T sum3 = (z[6][n] + z[7][n] + 1) >> 1;
    T dif0 = z[0][n] - z[1][n];
    T dif1 = z[2][n] - z[3][n];
    T dif2 = z[4][n] - z[5][n];
    T dif3 = z[6][n] - z[7][n];

    T sum00 = (sum0 + sum1 + 1) >> 1;
    T sum11 = (sum2 + sum3 + 1) >> 1;
    T dif00 = sum0 - sum1;
    T dif11 = sum2 - sum3;

    T sum000 = (sum00 + sum11 + 1) >> 1;
    T dif000 = sum00 - sum11;

    short nonZeroCount = 0;
    shrink(sum000, nonZeroCount, *thrMap++);
    shrink(dif000, nonZeroCount, *thrMap++);
    shrink(dif00, nonZeroCount, *thrMap++);
    shrink(dif11, nonZeroCount, *thrMap++);
    shrink(dif0, nonZeroCount, *thrMap++);
    shrink(dif1, nonZeroCount, *thrMap++);
    shrink(dif2, nonZeroCount, *thrMap++);
    shrink(dif3, nonZeroCount, *thrMap++);

    z[0][n] = sum000;
    z[1][n] = dif000;
    z[2][n] = dif00;
    z[3][n] = dif11;
    z[4][n] = dif0;
    z[5][n] = dif1;
    z[6][n] = dif2;
    z[7][n] = dif3;

    return nonZeroCount;
}

/// Functions for inverse 1D transforms

template <typename T, typename DT, typename CT>
inline static void InverseHaarTransform2(BlockMatch<T, DT, CT> *src, const int &n)
{
    T src0 = src[0][n] * 2;
    T src1 = src[1][n];

    src[0][n] = (src0 + src1) >> 1;
    src[1][n] = (src0 - src1) >> 1;
}

template <typename T, typename DT, typename CT>
inline static void InverseHaarTransform4(BlockMatch<T, DT, CT> *src, const int &n)
{
    T src0 = src[0][n] * 2;
    T src1 = src[1][n];
    T src2 = src[2][n];
    T src3 = src[3][n];

    T sum0 = src0 + src1;
    T dif0 = src0 - src1;

    src[0][n] = (sum0 + src2) >> 1;
    src[1][n] = (sum0 - src2) >> 1;
    src[2][n] = (dif0 + src3) >> 1;
    src[3][n] = (dif0 - src3) >> 1;
}

template <typename T, typename DT, typename CT>
inline static void InverseHaarTransform8(BlockMatch<T, DT, CT> *src, const int &n)
{
    T src0 = src[0][n] * 2;
    T src1 = src[1][n];
    T src2 = src[2][n];
    T src3 = src[3][n];
    T src4 = src[4][n];
    T src5 = src[5][n];
    T src6 = src[6][n];
    T src7 = src[7][n];

    T sum0 = src0 + src1;
    T dif0 = src0 - src1;

    T sum00 = sum0 + src2;
    T dif00 = sum0 - src2;
    T sum11 = dif0 + src3;
    T dif11 = dif0 - src3;

    src[0][n] = (sum00 + src4) >> 1;
    src[1][n] = (sum00 - src4) >> 1;
    src[2][n] = (dif00 + src5) >> 1;
    src[3][n] = (dif00 - src5) >> 1;
    src[4][n] = (sum11 + src6) >> 1;
    src[5][n] = (sum11 - src6) >> 1;
    src[6][n] = (dif11 + src7) >> 1;
    src[7][n] = (dif11 - src7) >> 1;
}

#endif