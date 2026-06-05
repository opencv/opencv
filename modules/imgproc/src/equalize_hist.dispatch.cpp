// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"

#include "equalize_hist.simd.hpp"
#include "equalize_hist.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,...,BASELINE

namespace cv {

void equalizeHistLut_dispatch( const uchar* src, uchar* dst, int len, const uchar* lut );

void equalizeHistLut_dispatch( const uchar* src, uchar* dst, int len, const uchar* lut )
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(equalizeHistLut_, (src, dst, len, lut),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
