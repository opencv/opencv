// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

// see "opencv2/core/hal/intrin.hpp"
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE hal_EMULATOR_CPP
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN namespace hal_EMULATOR_CPP {
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END }

// see "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"
//#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#define CV_FORCE_SIMD128_CPP
#undef CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
#undef CV_CPU_OPTIMIZATION_NAMESPACE_END
#define CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN namespace opt_EMULATOR_CPP {
#define CV_CPU_OPTIMIZATION_NAMESPACE_END }
#include "test_intrin128.simd.hpp"
#undef CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
#undef CV_CPU_OPTIMIZATION_NAMESPACE_END
#undef CV_CPU_DISPATCH_MODE
#undef CV_FORCE_SIMD128_CPP

// tests implementation is in test_intrin_utils.hpp
