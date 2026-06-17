// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "pyramids_avx512_vbmi.simd.hpp"
#include "pyramids_avx512_vbmi.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,...,BASELINE

namespace cv {

// declared in pyramids.cpp
int PyrDownVecH_uchar_ushort_1_dispatch(const uchar* src, ushort* row, int width);
int PyrDownVecH_uchar_ushort_2_dispatch(const uchar* src, ushort* row, int width);
int PyrDownVecH_uchar_ushort_3_dispatch(const uchar* src, ushort* row, int width);
int PyrDownVecH_uchar_ushort_4_dispatch(const uchar* src, ushort* row, int width);

int PyrDownVecH_uchar_ushort_1_dispatch(const uchar* src, ushort* row, int width)
{
    CV_CPU_DISPATCH(PyrDownVecH_uchar_ushort_1_vbmi, (src, row, width),
        CV_CPU_DISPATCH_MODES_ALL);
}

int PyrDownVecH_uchar_ushort_2_dispatch(const uchar* src, ushort* row, int width)
{
    CV_CPU_DISPATCH(PyrDownVecH_uchar_ushort_2_vbmi, (src, row, width),
        CV_CPU_DISPATCH_MODES_ALL);
}

int PyrDownVecH_uchar_ushort_3_dispatch(const uchar* src, ushort* row, int width)
{
    CV_CPU_DISPATCH(PyrDownVecH_uchar_ushort_3_vbmi, (src, row, width),
        CV_CPU_DISPATCH_MODES_ALL);
}

int PyrDownVecH_uchar_ushort_4_dispatch(const uchar* src, ushort* row, int width)
{
    CV_CPU_DISPATCH(PyrDownVecH_uchar_ushort_4_vbmi, (src, row, width),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
