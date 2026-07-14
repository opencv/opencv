// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#if !defined CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY && \
    !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS // TODO? C++ fallback implementation for SIMD512

#define CV__SIMD_FORCE_WIDTH 512
#include "opencv2/core/hal/intrin.hpp"
#undef CV__SIMD_FORCE_WIDTH

#if CV_SIMD_WIDTH != 64
#error "Invalid build configuration"
#endif

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace opencv_test { namespace hal { namespace intrin512 {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#include "test_intrin_utils.hpp"

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}} //namespace
