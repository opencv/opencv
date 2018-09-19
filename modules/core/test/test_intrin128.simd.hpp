// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#define CV__SIMD_FORCE_WIDTH 128
#include "opencv2/core/hal/intrin.hpp"
#undef CV__SIMD_FORCE_WIDTH

#if CV_SIMD_WIDTH != 16
#error "Invalid build configuration"
#endif

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace opencv_test { namespace hal { namespace intrin128 {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#include "test_intrin_utils.hpp"

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}} //namespace
