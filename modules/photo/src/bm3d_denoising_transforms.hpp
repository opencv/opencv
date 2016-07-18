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

#define BM3D_MAX_3D_SIZE 16

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

template <int N, typename T, typename DT, typename CT>
inline static short HardThreshold(BlockMatch<T, DT, CT> *z, const int &n, T *&thrMap)
{
    short nonZeroCount = 0;

    for (int i = 0; i < N; ++i)
        shrink(z[i][n], nonZeroCount, *thrMap++);

    return nonZeroCount;
}

template <int N, typename T, typename DT, typename CT>
inline static int WienerFiltering(BlockMatch<T, DT, CT> *zSrc, BlockMatch<T, DT, CT> *zBasic, const int &n, T *&thrMap)
{
    int wienerCoeffs = 0;

    for (int i = 0; i < N; ++i)
    {
        // Possible optimization point here to get rid of floats and casts
        int basicSq = zBasic[i][n] * zBasic[i][n];
        int sigmaSq = *thrMap * *thrMap;
        int denom = basicSq + sigmaSq;
        float wie = (denom == 0) ? 1.0f : ((float)basicSq / (float)denom);

        zBasic[i][n] = (T)(zSrc[i][n] * wie);
        wienerCoeffs += (int)wie;
        ++thrMap;
    }

    return wienerCoeffs;
}

/// 1D and 2D threshold map coefficients. Implementation dependent, thus stored
/// together with transforms.

static void calcHaarCoefficients1D(cv::Mat &coeff1D, const int &numberOfElements)
{
    // Generate base array and initialize with zeros
    cv::Mat baseArr = cv::Mat::zeros(numberOfElements, numberOfElements, CV_32FC1);

    // Calculate base array coefficients.
    int currentRow = 0;
    for (int i = numberOfElements; i > 0; i /= 2)
    {
        for (int k = 0, sign = -1; k < numberOfElements; ++k)
        {
            // Alternate sign every i-th element
            if (k % i == 0)
                sign *= -1;

            // Move to the next row every 2*i-th element
            if (k != 0 && (k % (2 * i) == 0))
                ++currentRow;

            baseArr.at<float>(currentRow, k) = sign * 1.0f / i;
        }
        ++currentRow;
    }

    // Square each elements of the base array
    float *ptr = baseArr.ptr<float>(0);
    for (unsigned i = 0; i < baseArr.total(); ++i)
        ptr[i] = ptr[i] * ptr[i];

    // Multiply baseArray with 1D vector of ones
    cv::Mat unitaryArr = cv::Mat::ones(numberOfElements, 1, CV_32FC1);
    coeff1D = baseArr * unitaryArr;
}

// Method to generate threshold coefficients for 1D transform depending on the number of elements.
static void fillHaarCoefficients1D(float *thrCoeff1D, int &idx, const int &numberOfElements)
{
    cv::Mat coeff1D;
    calcHaarCoefficients1D(coeff1D, numberOfElements);

    // Square root the array to get standard deviation
    float *ptr = coeff1D.ptr<float>(0);
    for (unsigned i = 0; i < coeff1D.total(); ++i)
    {
        ptr[i] = std::sqrt(ptr[i]);
        thrCoeff1D[idx++] = ptr[i];
    }
}

// Method to generate threshold coefficients for 2D transform depending on the number of elements.
static void fillHaarCoefficients2D(float *thrCoeff2D, const int &templateWindowSize)
{
    cv::Mat coeff1D;
    calcHaarCoefficients1D(coeff1D, templateWindowSize);

    // Calculate 2D array
    cv::Mat coeff1Dt;
    cv::transpose(coeff1D, coeff1Dt);
    cv::Mat coeff2D = coeff1D * coeff1Dt;

    // Square root the array to get standard deviation
    float *ptr = coeff2D.ptr<float>(0);
    for (unsigned i = 0; i < coeff2D.total(); ++i)
        thrCoeff2D[i] = std::sqrt(ptr[i]);
}

// Method to calculate 1D threshold map based on the maximum number of elements
// Allocates memory for the output array.
static void calcHaarThresholdMap1D(float *&thrMap1D, const int &numberOfElements)
{
    CV_Assert(numberOfElements <= BM3D_MAX_3D_SIZE && numberOfElements > 0);

    // Allocate memory for the array
    const int arrSize = (numberOfElements << 1) - 1;
    if (thrMap1D == NULL)
        thrMap1D = new float[arrSize];

    for (int i = 1, idx = 0; i <= numberOfElements; i *= 2)
        fillHaarCoefficients1D(thrMap1D, idx, i);
}

// Method to calculate 1D threshold map based on the maximum number of elements
// Allocates memory for the output array.
static void calcHaarThresholdMap2D(float *&thrMap2D, const int &templateWindowSize)
{
    // Allocate memory for the array
    if (thrMap2D == NULL)
        thrMap2D = new float[templateWindowSize * templateWindowSize];

    fillHaarCoefficients2D(thrMap2D, templateWindowSize);
}

// Method to calculate 3D threshold map based on the maximum number of elements.
// Allocates memory for the output array.
template <typename T>
static void calcHaarThresholdMap3D(
    T *&outThrMap1D,
    const float &hardThr1D,
    const int &templateWindowSize,
    const int &groupSize)
{
    const int templateWindowSizeSq = templateWindowSize * templateWindowSize;

    // Allocate memory for the output array
    if (outThrMap1D == NULL)
        outThrMap1D = new T[templateWindowSizeSq * ((groupSize << 1) - 1)];

    // Generate 1D coefficients map
    float *thrMap1D = NULL;
    calcHaarThresholdMap1D(thrMap1D, groupSize);

    // Generate 2D coefficients map
    float *thrMap2D = NULL;
    calcHaarThresholdMap2D(thrMap2D, templateWindowSize);

    // Generate 3D threshold map
    T *thrMapPtr1D = outThrMap1D;
    for (int i = 1, ii = 0; i <= groupSize; ++ii, i *= 2)
    {
        float coeff = (i == 1) ? 1.0f : std::sqrt(2.0f * std::log((float)i));
        for (int jj = 0; jj < templateWindowSizeSq; ++jj)
        {
            for (int ii1 = 0; ii1 < (1 << ii); ++ii1)
            {
                int indexIn1D = (1 << ii) - 1 + ii1;
                int indexIn2D = jj;
                int thr = static_cast<int>(thrMap1D[indexIn1D] * thrMap2D[indexIn2D] * hardThr1D * coeff);

                // Set DC component to zero
                if (jj == 0 && ii1 == 0)
                    thr = 0;

                *thrMapPtr1D++ = cv::saturate_cast<T>(thr);
            }
        }
    }

    delete[] thrMap1D;
    delete[] thrMap2D;
}

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
inline static void ForwardHaarTransform2(BlockMatch<T, DT, CT> *z, const int &n)
{
    T sum = (z[0][n] + z[1][n] + 1) >> 1;
    T dif = z[0][n] - z[1][n];

    z[0][n] = sum;
    z[1][n] = dif;
}

template <typename T, typename DT, typename CT>
inline static void ForwardHaarTransform4(BlockMatch<T, DT, CT> *z, const int &n)
{
    T sum0 = (z[0][n] + z[1][n] + 1) >> 1;
    T sum1 = (z[2][n] + z[3][n] + 1) >> 1;
    T dif0 = z[0][n] - z[1][n];
    T dif1 = z[2][n] - z[3][n];

    T sum00 = (sum0 + sum1 + 1) >> 1;
    T dif00 = sum0 - sum1;

    z[0][n] = sum00;
    z[1][n] = dif00;
    z[2][n] = dif0;
    z[3][n] = dif1;
}

template <typename T, typename DT, typename CT>
inline static void ForwardHaarTransform8(BlockMatch<T, DT, CT> *z, const int &n)
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

    z[0][n] = sum000;
    z[1][n] = dif000;
    z[2][n] = dif00;
    z[3][n] = dif11;
    z[4][n] = dif0;
    z[5][n] = dif1;
    z[6][n] = dif2;
    z[7][n] = dif3;
}

template <typename T, typename DT, typename CT>
inline static void ForwardHaarTransform16(BlockMatch<T, DT, CT> *z, const int &n)
{
    T sum0 = (z[0][n] + z[1][n] + 1) >> 1;
    T sum1 = (z[2][n] + z[3][n] + 1) >> 1;
    T sum2 = (z[4][n] + z[5][n] + 1) >> 1;
    T sum3 = (z[6][n] + z[7][n] + 1) >> 1;
    T sum4 = (z[8][n] + z[9][n] + 1) >> 1;
    T sum5 = (z[10][n] + z[11][n] + 1) >> 1;
    T sum6 = (z[12][n] + z[13][n] + 1) >> 1;
    T sum7 = (z[14][n] + z[15][n] + 1) >> 1;
    T dif0 = z[0][n] - z[1][n];
    T dif1 = z[2][n] - z[3][n];
    T dif2 = z[4][n] - z[5][n];
    T dif3 = z[6][n] - z[7][n];
    T dif4 = z[8][n] - z[9][n];
    T dif5 = z[10][n] - z[11][n];
    T dif6 = z[12][n] - z[13][n];
    T dif7 = z[14][n] - z[15][n];

    T sum00 = (sum0 + sum1 + 1) >> 1;
    T sum11 = (sum2 + sum3 + 1) >> 1;
    T sum22 = (sum4 + sum5 + 1) >> 1;
    T sum33 = (sum6 + sum7 + 1) >> 1;
    T dif00 = sum0 - sum1;
    T dif11 = sum2 - sum3;
    T dif22 = sum4 - sum5;
    T dif33 = sum6 - sum7;

    T sum000 = (sum00 + sum11 + 1) >> 1;
    T sum111 = (sum22 + sum33 + 1) >> 1;
    T dif000 = sum00 - sum11;
    T dif111 = sum22 - sum33;

    T sum0000 = (sum000 + sum111 + 1) >> 1;
    T dif0000 = dif000 - dif111;

    z[0][n] = sum0000;
    z[1][n] = dif0000;
    z[2][n] = dif000;
    z[3][n] = dif111;
    z[4][n] = dif00;
    z[5][n] = dif11;
    z[6][n] = dif22;
    z[7][n] = dif33;
    z[8][n] = dif0;
    z[9][n] = dif1;
    z[10][n] = dif2;
    z[11][n] = dif3;
    z[12][n] = dif4;
    z[13][n] = dif5;
    z[14][n] = dif6;
    z[15][n] = dif7;
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

template <typename T, typename DT, typename CT>
inline static void InverseHaarTransform16(BlockMatch<T, DT, CT> *src, const int &n)
{
    T src0 = src[0][n] * 2;
    T src1 = src[1][n];
    T src2 = src[2][n];
    T src3 = src[3][n];
    T src4 = src[4][n];
    T src5 = src[5][n];
    T src6 = src[6][n];
    T src7 = src[7][n];
    T src8 = src[8][n];
    T src9 = src[9][n];
    T src10 = src[10][n];
    T src11 = src[11][n];
    T src12 = src[12][n];
    T src13 = src[13][n];
    T src14 = src[14][n];
    T src15 = src[15][n];

    T sum0 = src0 + src1;
    T dif0 = src0 - src1;

    T sum00 = sum0 + src2;
    T dif00 = sum0 - src2;
    T sum11 = dif0 + src3;
    T dif11 = dif0 - src3;

    T sum000 = sum00 + src4;
    T dif000 = sum00 - src4;
    T sum111 = dif00 + src5;
    T dif111 = dif00 - src5;
    T sum222 = sum11 + src6;
    T dif222 = sum11 - src6;
    T sum333 = dif11 + src7;
    T dif333 = dif11 - src7;

    src[0][n] = (sum000 + src8) >> 1;
    src[1][n] = (sum000 - src8) >> 1;
    src[2][n] = (dif000 + src9) >> 1;
    src[3][n] = (dif000 - src9) >> 1;
    src[4][n] = (sum111 + src10) >> 1;
    src[5][n] = (sum111 - src10) >> 1;
    src[6][n] = (dif111 + src11) >> 1;
    src[7][n] = (dif111 - src11) >> 1;
    src[8][n] = (sum222 + src12) >> 1;
    src[9][n] = (sum222 - src12) >> 1;
    src[10][n] = (dif222 + src13) >> 1;
    src[11][n] = (dif222 - src13) >> 1;
    src[12][n] = (sum333 + src14) >> 1;
    src[13][n] = (sum333 - src14) >> 1;
    src[14][n] = (dif333 + src15) >> 1;
    src[15][n] = (dif333 - src15) >> 1;
}

#endif