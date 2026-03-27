// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "lkpyramid.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "lkpyramid.simd.hpp"
#include "lkpyramid.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL based on CMakeLists.txt

namespace cv {
namespace detail {

void ScharrDerivInvoker_impl(const Mat& src, Mat& dst, const Range& range)
{
    CV_CPU_DISPATCH(ScharrDerivInvoker_SIMD, (src, dst, range), CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace detail
} // namespace cv
