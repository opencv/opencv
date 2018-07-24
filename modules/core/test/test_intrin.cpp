// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "test_intrin.simd.hpp"

#define CV_CPU_SIMD_FILENAME "test_intrin.simd.hpp"
#define CV_CPU_DISPATCH_MODE FP16
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODE AVX2
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

namespace opencv_test { namespace hal {
using namespace CV_CPU_OPTIMIZATION_NAMESPACE;

TEST(hal_intrin, uint8x16)
{ test_hal_intrin_uint8(); }

TEST(hal_intrin, int8x16)
{ test_hal_intrin_int8(); }

TEST(hal_intrin, uint16x8)
{ test_hal_intrin_uint16(); }

TEST(hal_intrin, int16x8)
{ test_hal_intrin_int16(); }

TEST(hal_intrin, int32x4)
{ test_hal_intrin_int32(); }

TEST(hal_intrin, uint32x4)
{ test_hal_intrin_uint32(); }

TEST(hal_intrin, uint64x2)
{ test_hal_intrin_uint64(); }

TEST(hal_intrin, int64x2)
{ test_hal_intrin_int64(); }

TEST(hal_intrin, float32x4)
{ test_hal_intrin_float32(); }

TEST(hal_intrin, float64x2)
{ test_hal_intrin_float64(); }

TEST(hal_intrin, float16x8)
{
    CV_CPU_CALL_FP16_(test_hal_intrin_float16, ());
    throw SkipTestException("Unsupported hardware: FP16 is not available");
}

#define DISPATCH_SIMD_MODES AVX2
#define DISPATCH_SIMD_NAME "SIMD256"
#define DISPATCH_SIMD(fun)                              \
    do {                                                \
        CV_CPU_DISPATCH(fun, (), DISPATCH_SIMD_MODES);  \
        throw SkipTestException(                        \
            "Unsupported hardware: "                    \
            DISPATCH_SIMD_NAME                          \
            " is not available"                         \
        );                                              \
    } while(0)

TEST(hal_intrin256, uint8x32)
{ DISPATCH_SIMD(test_hal_intrin_uint8); }

TEST(hal_intrin256, int8x32)
{ DISPATCH_SIMD(test_hal_intrin_int8); }

TEST(hal_intrin256, uint16x16)
{ DISPATCH_SIMD(test_hal_intrin_uint16); }

TEST(hal_intrin256, int16x16)
{ DISPATCH_SIMD(test_hal_intrin_int16); }

TEST(hal_intrin256, uint32x8)
{ DISPATCH_SIMD(test_hal_intrin_uint32); }

TEST(hal_intrin256, int32x8)
{ DISPATCH_SIMD(test_hal_intrin_int32); }

TEST(hal_intrin256, uint64x4)
{ DISPATCH_SIMD(test_hal_intrin_uint64); }

TEST(hal_intrin256, int64x4)
{ DISPATCH_SIMD(test_hal_intrin_int64); }

TEST(hal_intrin256, float32x8)
{ DISPATCH_SIMD(test_hal_intrin_float32); }

TEST(hal_intrin256, float64x4)
{ DISPATCH_SIMD(test_hal_intrin_float64); }

TEST(hal_intrin256, float16x16)
{
    if (!CV_CPU_HAS_SUPPORT_FP16)
        throw SkipTestException("Unsupported hardware: FP16 is not available");
    DISPATCH_SIMD(test_hal_intrin_float16);
}

}} // namespace