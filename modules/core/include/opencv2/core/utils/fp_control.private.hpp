// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_FP_CONTROL_UTILS_PRIVATE_HPP
#define OPENCV_CORE_FP_CONTROL_UTILS_PRIVATE_HPP

#include "fp_control_utils.hpp"

#if OPENCV_SUPPORTS_FP_DENORMALS_HINT == 0
  // disabled
#elif defined(OPENCV_IMPL_FP_HINTS)
  // custom
#elif defined(OPENCV_IMPL_FP_HINTS_X86)
  // custom
#elif defined(__SSE__) || defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
  #include <xmmintrin.h>
  #define OPENCV_IMPL_FP_HINTS_X86 1
  #define OPENCV_IMPL_FP_HINTS 1
#endif

#ifndef OPENCV_IMPL_FP_HINTS
#define OPENCV_IMPL_FP_HINTS 0
#endif
#ifndef OPENCV_IMPL_FP_HINTS_X86
#define OPENCV_IMPL_FP_HINTS_X86 0
#endif

#endif // OPENCV_CORE_FP_CONTROL_UTILS_PRIVATE_HPP
