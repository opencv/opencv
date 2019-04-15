// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128

#define CV_SIMD128 1
#define CV_SIMD128_64F 1
#define CV_SIMD128_FP16 0  // no native operations with FP16 type.

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#include "utils.hpp"
#include "types.hpp"
#include "memory.hpp"
#include "operators.hpp"
#include "misc.hpp"
#include "arithmetic.hpp"
#include "math.hpp"
#include "reorder.hpp"
#include "interleave.hpp"
#include "deinterleave.hpp"
#include "conversion.hpp"

inline void v_cleanup() {}

//! @name Check SIMD support
//! @{
//! @brief Check CPU capability of SIMD operation
static inline bool hasSIMD128()
{
    return (CV_CPU_HAS_SUPPORT_SSE2) ? true : false;
}
//! @}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::

#endif // CV_SIMD128