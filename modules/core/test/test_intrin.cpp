// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#include "test_intrin_utils.hpp"

#define CV_CPU_SIMD_FILENAME "test_intrin_utils.hpp"
#define CV_CPU_DISPATCH_MODE FP16
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"


using namespace cv;

namespace opencv_test { namespace hal {
using namespace CV_CPU_OPTIMIZATION_NAMESPACE;

//=============  8-bit integer =====================================================================

TEST(hal_intrin, uint8x16) {
    TheTest<v_uint8x16>()
        .test_loadstore()
        .test_interleave()
        .test_expand()
        .test_expand_q()
        .test_addsub()
        .test_addsub_wrap()
        .test_cmp()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<3>().test_pack<8>()
        .test_pack_u<1>().test_pack_u<2>().test_pack_u<3>().test_pack_u<8>()
        .test_unpack()
        .test_extract<0>().test_extract<1>().test_extract<8>().test_extract<15>()
        .test_rotate<0>().test_rotate<1>().test_rotate<8>().test_rotate<15>()
        ;
}

TEST(hal_intrin, int8x16) {
    TheTest<v_int8x16>()
        .test_loadstore()
        .test_interleave()
        .test_expand()
        .test_expand_q()
        .test_addsub()
        .test_addsub_wrap()
        .test_cmp()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_abs()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<3>().test_pack<8>()
        .test_unpack()
        .test_extract<0>().test_extract<1>().test_extract<8>().test_extract<15>()
        .test_rotate<0>().test_rotate<1>().test_rotate<8>().test_rotate<15>()
        ;
}

//============= 16-bit integer =====================================================================

TEST(hal_intrin, uint16x8) {
    TheTest<v_uint16x8>()
        .test_loadstore()
        .test_interleave()
        .test_expand()
        .test_addsub()
        .test_addsub_wrap()
        .test_mul()
        .test_mul_expand()
        .test_cmp()
        .test_shift<1>()
        .test_shift<8>()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_reduce()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<7>().test_pack<16>()
        .test_pack_u<1>().test_pack_u<2>().test_pack_u<7>().test_pack_u<16>()
        .test_unpack()
        .test_extract<0>().test_extract<1>().test_extract<4>().test_extract<7>()
        .test_rotate<0>().test_rotate<1>().test_rotate<4>().test_rotate<7>()
        ;
}

TEST(hal_intrin, int16x8) {
    TheTest<v_int16x8>()
        .test_loadstore()
        .test_interleave()
        .test_expand()
        .test_addsub()
        .test_addsub_wrap()
        .test_mul()
        .test_mul_expand()
        .test_cmp()
        .test_shift<1>()
        .test_shift<8>()
        .test_dot_prod()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_abs()
        .test_reduce()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<7>().test_pack<16>()
        .test_unpack()
        .test_extract<0>().test_extract<1>().test_extract<4>().test_extract<7>()
        .test_rotate<0>().test_rotate<1>().test_rotate<4>().test_rotate<7>()
        ;
}

//============= 32-bit integer =====================================================================

TEST(hal_intrin, uint32x4) {
    TheTest<v_uint32x4>()
        .test_loadstore()
        .test_interleave()
        .test_expand()
        .test_addsub()
        .test_mul()
        .test_mul_expand()
        .test_cmp()
        .test_shift<1>()
        .test_shift<8>()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_reduce()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<15>().test_pack<32>()
        .test_unpack()
        .test_extract<0>().test_extract<1>().test_extract<2>().test_extract<3>()
        .test_rotate<0>().test_rotate<1>().test_rotate<2>().test_rotate<3>()
        .test_transpose()
        ;
}

TEST(hal_intrin, int32x4) {
    TheTest<v_int32x4>()
        .test_loadstore()
        .test_interleave()
        .test_expand()
        .test_addsub()
        .test_mul()
        .test_abs()
        .test_cmp()
        .test_popcount()
        .test_shift<1>().test_shift<8>()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_reduce()
        .test_mask()
        .test_pack<1>().test_pack<2>().test_pack<15>().test_pack<32>()
        .test_unpack()
        .test_extract<0>().test_extract<1>().test_extract<2>().test_extract<3>()
        .test_rotate<0>().test_rotate<1>().test_rotate<2>().test_rotate<3>()
        .test_float_cvt32()
        .test_float_cvt64()
        .test_transpose()
        ;
}

//============= 64-bit integer =====================================================================

TEST(hal_intrin, uint64x2) {
    TheTest<v_uint64x2>()
        .test_loadstore()
        .test_addsub()
        .test_shift<1>().test_shift<8>()
        .test_logic()
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        ;
}

TEST(hal_intrin, int64x2) {
    TheTest<v_int64x2>()
        .test_loadstore()
        .test_addsub()
        .test_shift<1>().test_shift<8>()
        .test_logic()
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        ;
}

//============= Floating point =====================================================================

TEST(hal_intrin, float32x4) {
    TheTest<v_float32x4>()
        .test_loadstore()
        .test_interleave()
        .test_interleave_2channel()
        .test_addsub()
        .test_mul()
        .test_div()
        .test_cmp()
        .test_sqrt_abs()
        .test_min_max()
        .test_float_absdiff()
        .test_reduce()
        .test_mask()
        .test_unpack()
        .test_float_math()
        .test_float_cvt64()
        .test_matmul()
        .test_transpose()
        .test_reduce_sum4()
        ;
}

#if CV_SIMD128_64F
TEST(hal_intrin, float64x2) {
    TheTest<v_float64x2>()
        .test_loadstore()
        .test_addsub()
        .test_mul()
        .test_div()
        .test_cmp()
        .test_sqrt_abs()
        .test_min_max()
        .test_float_absdiff()
        .test_mask()
        .test_unpack()
        .test_float_math()
        .test_float_cvt32()
        ;
}
#endif

TEST(hal_intrin,float16x4)
{
    CV_CPU_CALL_FP16_(test_hal_intrin_float16x4, ());
    throw SkipTestException("Unsupported hardware: FP16 is not available");
}

}}
