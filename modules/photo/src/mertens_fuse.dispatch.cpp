// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"

#include "mertens_fuse.simd.hpp"
#include "mertens_fuse.simd_declarations.hpp" // CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,BASELINE

namespace cv {

typedef void (*MertensConvertGrayFunc)(const uchar*, size_t, float*, size_t, float*, size_t, int, int);
typedef void (*MertensPyrUpSubFunc)(const float*, size_t, float*, size_t, int, int, int, int, int);
typedef void (*MertensBlendAccFunc)(const float*, size_t, const float*, size_t, float*, size_t, int, int, int, bool);

static MertensConvertGrayFunc resolveConvertGray()
{
    CV_CPU_DISPATCH_FN(mertensConvertGray_, CV_CPU_DISPATCH_MODES_ALL);
}
static MertensPyrUpSubFunc resolvePyrUpSub()
{
    CV_CPU_DISPATCH_FN(mertensPyrUpSub_, CV_CPU_DISPATCH_MODES_ALL);
}
static MertensBlendAccFunc resolveBlendAcc()
{
    CV_CPU_DISPATCH_FN(mertensBlendAcc_, CV_CPU_DISPATCH_MODES_ALL);
}

void mertensConvertGray(const uchar* src, size_t srcStep, float* img, size_t imgStep,
                        float* gray, size_t grayStep, int rows, int cols);
void mertensPyrUpSub(const float* src, size_t srcStep, float* dst, size_t dstStep,
                     int srcRows, int srcCols, int dstRows, int dstCols, int channels);
void mertensBlendAcc(const float* img, size_t imgStep, const float* weight, size_t weightStep,
                     float* acc, size_t accStep, int rows, int cols, int channels, bool accumulate);

void mertensConvertGray(const uchar* src, size_t srcStep, float* img, size_t imgStep,
                        float* gray, size_t grayStep, int rows, int cols)
{
    static MertensConvertGrayFunc fn = resolveConvertGray();
    fn(src, srcStep, img, imgStep, gray, grayStep, rows, cols);
}

void mertensPyrUpSub(const float* src, size_t srcStep, float* dst, size_t dstStep,
                     int srcRows, int srcCols, int dstRows, int dstCols, int channels)
{
    static MertensPyrUpSubFunc fn = resolvePyrUpSub();
    fn(src, srcStep, dst, dstStep, srcRows, srcCols, dstRows, dstCols, channels);
}

void mertensBlendAcc(const float* img, size_t imgStep, const float* weight, size_t weightStep,
                     float* acc, size_t accStep, int rows, int cols, int channels, bool accumulate)
{
    static MertensBlendAccFunc fn = resolveBlendAcc();
    fn(img, imgStep, weight, weightStep, acc, accStep, rows, cols, channels, accumulate);
}

} // namespace cv
