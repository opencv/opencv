// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"

#include "mertens_weight.simd.hpp"
#include "mertens_weight.simd_declarations.hpp" // CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,BASELINE

namespace cv {

typedef void (*MertensWeightRowFunc)(const float*, const float*, const float*, const float*, float*, int, int, float, float, float, float*);

static MertensWeightRowFunc resolveMertensWeightRow()
{
    CV_CPU_DISPATCH_FN(mertensWeightRow_, CV_CPU_DISPATCH_MODES_ALL);
}

void mertensWeightRow(const float* grayUp, const float* grayMid, const float* grayDn,
                      const float* src, float* dst, int cols, int channels,
                      float wcon, float wsat, float wexp, float* sum);

void mertensWeightRow(const float* grayUp, const float* grayMid, const float* grayDn,
                      const float* src, float* dst, int cols, int channels,
                      float wcon, float wsat, float wexp, float* sum)
{
    static MertensWeightRowFunc fn = resolveMertensWeightRow();
    fn(grayUp, grayMid, grayDn, src, dst, cols, channels, wcon, wsat, wexp, sum);
}

} // namespace cv
