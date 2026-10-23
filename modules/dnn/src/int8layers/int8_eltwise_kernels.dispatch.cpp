// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "int8_eltwise_kernels.hpp"

#include "int8_eltwise_kernels.simd.hpp"
#include "int8layers/int8_eltwise_kernels.simd_declarations.hpp"

namespace cv {
namespace dnn {

void addInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float c0, float c1, float offset, int lo, int hi, const uchar* lut)
{
    CV_CPU_DISPATCH(addInt8, (a, b, out, len, isU8, c0, c1, offset, lo, hi, lut),
                    CV_CPU_DISPATCH_MODES_ALL);
}

void mulInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float zp0, float zp1, float scale, float offset, int lo, int hi, const uchar* lut)
{
    CV_CPU_DISPATCH(mulInt8, (a, b, out, len, isU8, zp0, zp1, scale, offset, lo, hi, lut),
                    CV_CPU_DISPATCH_MODES_ALL);
}

void lutInt8(const void* src, void* dst, int len, bool isU8, const uchar* lut)
{
    CV_CPU_DISPATCH(lutInt8, (src, dst, len, isU8, lut), CV_CPU_DISPATCH_MODES_ALL);
}

}} // namespace cv::dnn
