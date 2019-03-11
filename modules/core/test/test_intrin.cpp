// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#include "test_intrin128.simd.hpp"
#include "test_intrin128.simd_declarations.hpp"

#undef CV_CPU_DISPATCH_MODES_ALL

#include "opencv2/core/cv_cpu_dispatch.h"
#include "test_intrin256.simd.hpp"
#include "test_intrin256.simd_declarations.hpp"

#ifdef _MSC_VER
# pragma warning(disable:4702)  // unreachable code
#endif

namespace opencv_test { namespace hal {

#define CV_CPU_CALL_BASELINE_(fn, args)  CV_CPU_CALL_BASELINE(fn, args)

#define DISPATCH_SIMD128(fn, cpu_opt) do { \
    CV_CPU_CALL_ ## cpu_opt ## _(fn, ()); \
    throw SkipTestException("SIMD128 (" #cpu_opt ") is not available or disabled"); \
} while(0)

#define DISPATCH_SIMD256(fn, cpu_opt) do { \
    CV_CPU_CALL_ ## cpu_opt ## _(fn, ()); \
    throw SkipTestException("SIMD256 (" #cpu_opt ") is not available or disabled"); \
} while(0)

#define DEFINE_SIMD_TESTS(simd_size, cpu_opt) \
TEST(hal_intrin ## simd_size, uint8x16_ ## cpu_opt)  { DISPATCH_SIMD ## simd_size(test_hal_intrin_uint8, cpu_opt); } \
TEST(hal_intrin ## simd_size, int8x16_ ## cpu_opt)   { DISPATCH_SIMD ## simd_size(test_hal_intrin_int8, cpu_opt); } \
TEST(hal_intrin ## simd_size, uint16x8_ ## cpu_opt)  { DISPATCH_SIMD ## simd_size(test_hal_intrin_uint16, cpu_opt); } \
TEST(hal_intrin ## simd_size, int16x8_ ## cpu_opt)   { DISPATCH_SIMD ## simd_size(test_hal_intrin_int16, cpu_opt); } \
TEST(hal_intrin ## simd_size, int32x4_ ## cpu_opt)   { DISPATCH_SIMD ## simd_size(test_hal_intrin_int32, cpu_opt); } \
TEST(hal_intrin ## simd_size, uint32x4_ ## cpu_opt)  { DISPATCH_SIMD ## simd_size(test_hal_intrin_uint32, cpu_opt); } \
TEST(hal_intrin ## simd_size, uint64x2_ ## cpu_opt)  { DISPATCH_SIMD ## simd_size(test_hal_intrin_uint64, cpu_opt); } \
TEST(hal_intrin ## simd_size, int64x2_ ## cpu_opt)   { DISPATCH_SIMD ## simd_size(test_hal_intrin_int64, cpu_opt); } \
TEST(hal_intrin ## simd_size, float32x4_ ## cpu_opt) { DISPATCH_SIMD ## simd_size(test_hal_intrin_float32, cpu_opt); } \
TEST(hal_intrin ## simd_size, float64x2_ ## cpu_opt) { DISPATCH_SIMD ## simd_size(test_hal_intrin_float64, cpu_opt); } \

namespace intrin128 {

DEFINE_SIMD_TESTS(128, BASELINE)

#if defined CV_CPU_DISPATCH_COMPILE_SSE2 || defined CV_CPU_BASELINE_COMPILE_SSE2
DEFINE_SIMD_TESTS(128, SSE2)
#endif
#if defined CV_CPU_DISPATCH_COMPILE_SSE3 || defined CV_CPU_BASELINE_COMPILE_SSE3
DEFINE_SIMD_TESTS(128, SSE3)
#endif
#if defined CV_CPU_DISPATCH_COMPILE_SSSE3 || defined CV_CPU_BASELINE_COMPILE_SSSE3
DEFINE_SIMD_TESTS(128, SSSE3)
#endif
#if defined CV_CPU_DISPATCH_COMPILE_SSE4_1 || defined CV_CPU_BASELINE_COMPILE_SSE4_1
DEFINE_SIMD_TESTS(128, SSE4_1)
#endif
#if defined CV_CPU_DISPATCH_COMPILE_SSE4_2 || defined CV_CPU_BASELINE_COMPILE_SSE4_2
DEFINE_SIMD_TESTS(128, SSE4_2)
#endif
#if defined CV_CPU_DISPATCH_COMPILE_AVX || defined CV_CPU_BASELINE_COMPILE_AVX
DEFINE_SIMD_TESTS(128, AVX)
#endif
#if defined CV_CPU_DISPATCH_COMPILE_AVX2 || defined CV_CPU_BASELINE_COMPILE_AVX2
DEFINE_SIMD_TESTS(128, AVX2)
#endif

TEST(hal_intrin128, float16x8_FP16)
{
    CV_CPU_CALL_FP16_(test_hal_intrin_float16, ());
    throw SkipTestException("Unsupported hardware: FP16 is not available");
}

} // namespace intrin128


namespace intrin256 {


// Not available due missing C++ backend for SIMD256
//DEFINE_SIMD_TESTS(256, BASELINE)

//#if defined CV_CPU_DISPATCH_COMPILE_AVX
//DEFINE_SIMD_TESTS(256, AVX)
//#endif

#if defined CV_CPU_DISPATCH_COMPILE_AVX2 || defined CV_CPU_BASELINE_COMPILE_AVX2
DEFINE_SIMD_TESTS(256, AVX2)
#endif

TEST(hal_intrin256, float16x16_FP16)
{
    //CV_CPU_CALL_FP16_(test_hal_intrin_float16, ());
    CV_CPU_CALL_AVX2_(test_hal_intrin_float16, ());
    throw SkipTestException("Unsupported hardware: FP16 is not available");
}


} // namespace intrin256

}} // namespace