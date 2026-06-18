// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"

#include "equalize_hist.simd.hpp"
#include "equalize_hist.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,...,BASELINE

namespace cv {

namespace {

typedef void (*EqualizeHistLutFunc)( const uchar*, uchar*, int, const uchar* );

static EqualizeHistLutFunc resolveEqualizeHistLutFunc()
{
#if CV_TRY_AVX512_ICL
    if (cv::checkHardwareSupport(CV_CPU_AVX512_ICL))
        return opt_AVX512_ICL::equalizeHistLut_;
#endif
    return cpu_baseline::equalizeHistLut_;
}

} // namespace

void equalizeHistLut_dispatch( const uchar* src, uchar* dst, int len, const uchar* lut );

void equalizeHistLut_dispatch( const uchar* src, uchar* dst, int len, const uchar* lut )
{
    CV_INSTRUMENT_REGION();
    static EqualizeHistLutFunc fn = resolveEqualizeHistLutFunc();
    fn(src, dst, len, lut);
}

} // namespace cv
