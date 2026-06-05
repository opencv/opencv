// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"

#include "lut.simd.hpp"
#include "lut.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,...,BASELINE based on CMakeLists.txt content

namespace cv {

void LUT8u_dispatch( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );

void LUT8u_dispatch( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(LUT8u_, (src, lut, dst, len, cn, lutcn),
        CV_CPU_DISPATCH_MODES_ALL);
}

void LUT16u_dispatch( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn );

void LUT16u_dispatch( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(LUT16u_, (src, lut, dst, len, cn, lutcn),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
