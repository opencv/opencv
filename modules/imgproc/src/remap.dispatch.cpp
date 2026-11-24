// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, all rights reserved.

#include "precomp.hpp"

#include "remap.simd.hpp"
#include "remap.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,...,BASELINE

namespace cv {

RemapBilinearC1Fn remapBilinearC1_fn()
{
    CV_CPU_DISPATCH(getRemapBilinearC1Fn_, (), CV_CPU_DISPATCH_MODES_ALL);
}

RemapWideC1Fn remapLanczos4C1_fn()
{
    CV_CPU_DISPATCH(getRemapLanczos4C1Fn_, (), CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
