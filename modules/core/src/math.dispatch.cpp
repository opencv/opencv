// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Dispatch layer for the element-wise MATH + SELECT kernels (math.simd.hpp) - the sibling of
// arithm.dispatch.cpp: plain functions forwarding to the CPU-optimal kernel via CV_CPU_DISPATCH.
// getElemwiseFunc (arithm.dispatch.cpp) routes the corresponding TOps here.

#include "precomp.hpp"
#include "arithm_expr.hpp"
#include "math.simd.hpp"
#include "math.simd_declarations.hpp"

namespace cv { namespace ew {

TKernel getMathFunc(TOp op, int T)          { CV_CPU_DISPATCH(getMathFunc_,   (op, T),      CV_CPU_DISPATCH_MODES_ALL); }

}} // namespace cv::ew
