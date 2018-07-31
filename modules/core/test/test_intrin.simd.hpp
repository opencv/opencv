// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "test_intrin_utils.hpp"

namespace opencv_test { namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void test_hal_intrin_uint8();
void test_hal_intrin_int8();
void test_hal_intrin_uint16();
void test_hal_intrin_int16();
void test_hal_intrin_uint32();
void test_hal_intrin_int32();
void test_hal_intrin_uint64();
void test_hal_intrin_int64();
void test_hal_intrin_float32();
void test_hal_intrin_float64();

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

//=============  8-bit integer =====================================================================

void test_hal_intrin_uint8()
{
    TheTest<v_uint8>()
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

#if CV_SIMD256
    TheTest<v_uint8>()
        .test_pack<9>().test_pack<10>().test_pack<13>().test_pack<15>()
        .test_pack_u<9>().test_pack_u<10>().test_pack_u<13>().test_pack_u<15>()
        .test_extract<16>().test_extract<17>().test_extract<23>().test_extract<31>()
        .test_rotate<16>().test_rotate<17>().test_rotate<23>().test_rotate<31>()
        ;
#endif
}

void test_hal_intrin_int8()
{
    TheTest<v_int8>()
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

void test_hal_intrin_uint16()
{
    TheTest<v_uint16>()
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

void test_hal_intrin_int16()
{
    TheTest<v_int16>()
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

void test_hal_intrin_uint32()
{
    TheTest<v_uint32>()
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

void test_hal_intrin_int32()
{
    TheTest<v_int32>()
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

void test_hal_intrin_uint64()
{
    TheTest<v_uint64>()
        .test_loadstore()
        .test_addsub()
        .test_shift<1>().test_shift<8>()
        .test_logic()
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        ;
}

void test_hal_intrin_int64()
{
    TheTest<v_int64>()
        .test_loadstore()
        .test_addsub()
        .test_shift<1>().test_shift<8>()
        .test_logic()
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        ;
}

//============= Floating point =====================================================================
void test_hal_intrin_float32()
{
    TheTest<v_float32>()
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
        .test_extract<0>().test_extract<1>().test_extract<2>().test_extract<3>()
        .test_rotate<0>().test_rotate<1>().test_rotate<2>().test_rotate<3>()
        ;

#if CV_SIMD256
    TheTest<v_float32>()
        .test_extract<4>().test_extract<5>().test_extract<6>().test_extract<7>()
        .test_rotate<4>().test_rotate<5>().test_rotate<6>().test_rotate<7>()
        ;
#endif
}

void test_hal_intrin_float64()
{
#if CV_SIMD_64F
    TheTest<v_float64>()
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
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        ;

#if CV_SIMD256
    TheTest<v_float64>()
        .test_extract<2>().test_extract<3>()
        .test_rotate<2>().test_rotate<3>()
        ;
#endif //CV_SIMD256

#endif
}

#if CV_FP16 && CV_SIMD_WIDTH > 16
void test_hal_intrin_float16()
{
    TheTest<v_float16>()
        .test_loadstore_fp16()
        .test_float_cvt_fp16()
        ;
}
#endif

#endif //CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}} //namespace