/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

#include "resize.hpp"

#include "opencv2/core/softfloat.hpp"
#include "fixedpoint.inl.hpp"

#include <iostream>
#include <array>

using namespace cv;

namespace
{

template <typename ET, bool needsign> struct fixedtype { typedef fixedpoint64 type; };
template <> struct fixedtype<uint32_t, false> { typedef ufixedpoint64 type; };
template <bool needsign> struct fixedtype<int16_t, needsign> { typedef fixedpoint32 type; };
template <> struct fixedtype<uint16_t, false> { typedef ufixedpoint32 type; };
template <bool needsign> struct fixedtype<int8_t, needsign> { typedef fixedpoint32 type; };
template <> struct fixedtype<uint8_t, false> { typedef ufixedpoint16 type; };

//FT is fixedtype<ET, needsign>::type
template <typename ET, typename FT, int n, bool mulall>
static void hlineResize(ET* src, int cn, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    for (; i < dst_min; i++, m += n) // Points that fall left from src image so became equal to leftmost src point
    {
        for (int j = 0; j < cn; j++, dst++)
        {
            *dst = src[j];
        }
    }
    for (; i < dst_max; i++, m += n)
    {
        ET* src_ofst = src + cn*ofst[i];
        for (int j = 0; j < cn; j++, dst++)
        {
            *dst = (mulall || !m[0].isZero()) ? m[0] * src_ofst[j] : FT::zero();
            for (int k = 1; k < n; k++)
            {
                *dst = *dst + ((mulall || !m[k].isZero()) ? m[k] * src_ofst[j+k*cn] : FT::zero());
            }
        }
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ET* src_last = src + cn*ofst[dst_width - 1];
    for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
    {
        for (int j = 0; j < cn; j++, dst++)
        {
            *dst = src_last[j];
        }
    }
}
template <typename ET, typename FT, int n, bool mulall, int cncnt> struct hline
{
    static void ResizeCn(ET* src, int cn, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        hlineResize<ET, FT, n, mulall>(src, cn, ofst, m, dst, dst_min, dst_max, dst_width);
    }
};
// Merges the four hand-unrolled hline<…,2,true,cncnt> specializations; compile-time cncnt unrolls the channel loop to byte-identical code.
template <typename ET, typename FT, int cncnt> struct hline<ET, FT, 2, true, cncnt>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0[cncnt];
        for (int c = 0; c < cncnt; c++) src0[c] = FT(src[c]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
            for (int c = 0; c < cncnt; c++)
                *(dst++) = src0[c];
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + cncnt*ofst[i];
            for (int c = 0; c < cncnt; c++)
                *(dst++) = m[0] * px[c] + m[1] * px[c + cncnt];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        ET* px_last = src + cncnt*ofst[dst_width - 1];
        for (int c = 0; c < cncnt; c++) src0[c] = px_last[c];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
            for (int c = 0; c < cncnt; c++)
                *(dst++) = src0[c];
    }
};
// Merged hline<…,4,true,cncnt>; preserves the original n=4 quirk of reading src (not src+cncnt*ofst[i]) byte-for-byte.
template <typename ET, typename FT, int cncnt> struct hline<ET, FT, 4, true, cncnt>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0[cncnt];
        for (int c = 0; c < cncnt; c++) src0[c] = FT(src[c]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
            for (int c = 0; c < cncnt; c++)
                *(dst++) = src0[c];
        for (; i < dst_max; i++, m += 4)
        {
            for (int c = 0; c < cncnt; c++)
                *(dst++) = m[0] * src[c] + m[1] * src[c + cncnt] + m[2] * src[c + 2*cncnt] + m[3] * src[c + 3*cncnt];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        ET* px_last = src + cncnt*ofst[dst_width - 1];
        for (int c = 0; c < cncnt; c++) src0[c] = px_last[c];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
            for (int c = 0; c < cncnt; c++)
                *(dst++) = src0[c];
    }
};
template <typename ET, typename FT, int n, bool mulall, int cncnt>
static void hlineResizeCn(ET* src, int cn, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
{
    hline<ET, FT, n, mulall, cncnt>::ResizeCn(src, cn, ofst, m, dst, dst_min, dst_max, dst_width);
}

template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 1>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    ufixedpoint16 src_0(src[0]);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_src_0 = vx_setall_u16(*((uint16_t*)&src_0));
    for (; i <= dst_min - VECSZ; i += VECSZ, m += 2*VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = src_0;
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - 2*VECSZ; i += 2*VECSZ, m += 4*VECSZ, dst += 2*VECSZ)
    {
        v_uint16 v_src0, v_src1;
        v_expand(vx_lut_pairs(src, ofst + i), v_src0, v_src1);
        v_store((uint16_t*)dst      , v_pack(v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), vx_load((int16_t*)m))),
                                             v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), vx_load((int16_t*)m + VECSZ)))));
        v_expand(vx_lut_pairs(src, ofst + i + VECSZ), v_src0, v_src1);
        v_store((uint16_t*)dst+VECSZ, v_pack(v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), vx_load((int16_t*)m + 2*VECSZ))),
                                             v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), vx_load((int16_t*)m + 3*VECSZ)))));
    }
    if (i <= dst_max - VECSZ)
    {
        v_uint16 v_src0, v_src1;
        v_expand(vx_lut_pairs(src, ofst + i), v_src0, v_src1);
        v_store((uint16_t*)dst, v_pack(v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), vx_load((int16_t*)m))),
                                       v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), vx_load((int16_t*)m + VECSZ)))));
        i += VECSZ; m += 2*VECSZ; dst += VECSZ;
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[1];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    src_0 = (src + ofst[dst_width - 1])[0];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_src_0 = vx_setall_u16(*((uint16_t*)&src_0));
    for (; i <= dst_width - VECSZ; i += VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = src_0;
    }
}
template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 2>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    union {
        uint32_t d;
        uint16_t w[2];
    } srccn;
    ((ufixedpoint16*)(srccn.w))[0] = src[0];
    ((ufixedpoint16*)(srccn.w))[1] = src[1];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_srccn = v_reinterpret_as_u16(vx_setall_u32(srccn.d));
    for (; i <= dst_min - VECSZ/2; i += VECSZ/2, m += VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - VECSZ/2; i += VECSZ/2, m += VECSZ, dst += VECSZ)
    {
        v_uint16 v_src0, v_src1;
        v_expand(v_interleave_pairs(v_reinterpret_as_u8(vx_lut_pairs((uint16_t*)src, ofst + i))), v_src0, v_src1);

        v_uint32 v_mul = vx_load((uint32_t*)m);//AaBbCcDd
        v_uint32 v_zip0, v_zip1;
        v_zip(v_mul, v_mul, v_zip0, v_zip1);//AaAaBbBb CcCcDdDd
        v_uint32 v_res0 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), v_reinterpret_as_s16(v_zip0)));
        v_uint32 v_res1 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), v_reinterpret_as_s16(v_zip1)));
        v_store((uint16_t*)dst, v_pack(v_res0, v_res1));//AB1AB2CD1CD2
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + 2 * ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[2];
        *(dst++) = m[0] * px[1] + m[1] * px[3];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ((ufixedpoint16*)(srccn.w))[0] = (src + 2 * ofst[dst_width - 1])[0]; ((ufixedpoint16*)(srccn.w))[1] = (src + 2 * ofst[dst_width - 1])[1];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_srccn = v_reinterpret_as_u16(vx_setall_u32(srccn.d));
    for (; i <= dst_width - VECSZ/2; i += VECSZ/2, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
    }
}
template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 3>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    union {
        uint64_t q;
        uint16_t w[4];
    } srccn;
    ((ufixedpoint16*)(srccn.w))[0] = src[0];
    ((ufixedpoint16*)(srccn.w))[1] = src[1];
    ((ufixedpoint16*)(srccn.w))[2] = src[2];
    ((ufixedpoint16*)(srccn.w))[3] = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_srccn = v_pack_triplets(v_reinterpret_as_u16(vx_setall_u64(srccn.q)));
    for (; i <= dst_min - (VECSZ+2)/3; i += VECSZ/4, m += VECSZ/2, dst += 3*VECSZ/4) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    CV_DECL_ALIGNED(CV_SIMD_WIDTH) int ofst3[VTraits<v_uint16>::max_nlanes/2];
    for (; i <= dst_max - (3*VECSZ/4 + (VECSZ+2)/3); i += VECSZ/2, m += VECSZ, dst += 3*VECSZ/2)
    {
        v_store(ofst3, v_mul(vx_load(ofst + i), vx_setall_s32(3)));
        v_uint8 v_src01, v_src23;
        v_uint16 v_src0, v_src1, v_src2, v_src3;
        v_zip(vx_lut_quads(src, ofst3), v_reinterpret_as_u8(v_shr<8>(v_reinterpret_as_u32(vx_lut_quads(src+2, ofst3)))), v_src01, v_src23);
        v_expand(v_src01, v_src0, v_src1);
        v_expand(v_src23, v_src2, v_src3);

        v_uint32 v_mul0, v_mul1, v_mul2, v_mul3, v_tmp;
        v_mul0 = vx_load((uint32_t*)m);//AaBbCcDd
        v_zip(v_mul0, v_mul0, v_mul3, v_tmp );//AaAaBbBb CcCcDdDd
        v_zip(v_mul3, v_mul3, v_mul0, v_mul1);//AaAaAaAa BbBbBbBb
        v_zip(v_tmp , v_tmp , v_mul2, v_mul3);//CcCcCcCc DdDdDdDd

        v_uint32 v_res0 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), v_reinterpret_as_s16(v_mul0)));
        v_uint32 v_res1 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), v_reinterpret_as_s16(v_mul1)));
        v_uint32 v_res2 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src2), v_reinterpret_as_s16(v_mul2)));
        v_uint32 v_res3 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src3), v_reinterpret_as_s16(v_mul3)));
        v_store((uint16_t*)dst            , v_pack_triplets(v_pack(v_res0, v_res1)));
        v_store((uint16_t*)dst + 3*VECSZ/4, v_pack_triplets(v_pack(v_res2, v_res3)));
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + 3 * ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[3];
        *(dst++) = m[0] * px[1] + m[1] * px[4];
        *(dst++) = m[0] * px[2] + m[1] * px[5];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ((ufixedpoint16*)(srccn.w))[0] = (src + 3*ofst[dst_width - 1])[0];
    ((ufixedpoint16*)(srccn.w))[1] = (src + 3*ofst[dst_width - 1])[1];
    ((ufixedpoint16*)(srccn.w))[2] = (src + 3*ofst[dst_width - 1])[2];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_srccn = v_pack_triplets(v_reinterpret_as_u16(vx_setall_u64(srccn.q)));
    for (; i <= dst_width - (VECSZ+2)/3; i += VECSZ/4, dst += 3*VECSZ/4) // Points that fall right from src image so became equal to rightmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
    }
}
template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 4>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    union {
        uint64_t q;
        uint16_t w[4];
    } srccn;
    ((ufixedpoint16*)(srccn.w))[0] = src[0];
    ((ufixedpoint16*)(srccn.w))[1] = src[1];
    ((ufixedpoint16*)(srccn.w))[2] = src[2];
    ((ufixedpoint16*)(srccn.w))[3] = src[3];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_srccn = v_reinterpret_as_u16(vx_setall_u64(srccn.q));
    for (; i <= dst_min - VECSZ/4; i += VECSZ/4, m += VECSZ/2, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[3];
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - VECSZ/2; i += VECSZ/2, m += VECSZ, dst += 2*VECSZ)
    {
        v_uint16 v_src0, v_src1, v_src2, v_src3;
        v_expand(v_interleave_quads(v_reinterpret_as_u8(vx_lut_pairs((uint32_t*)src, ofst + i))), v_src0, v_src1);
        v_expand(v_interleave_quads(v_reinterpret_as_u8(vx_lut_pairs((uint32_t*)src, ofst + i + VECSZ/4))), v_src2, v_src3);

        v_uint32 v_mul0, v_mul1, v_mul2, v_mul3, v_tmp;
        v_mul0 = vx_load((uint32_t*)m);//AaBbCcDd
        v_zip(v_mul0, v_mul0, v_mul3, v_tmp );//AaAaBbBb CcCcDdDd
        v_zip(v_mul3, v_mul3, v_mul0, v_mul1);//AaAaAaAa BbBbBbBb
        v_zip(v_tmp , v_tmp , v_mul2, v_mul3);//CcCcCcCc DdDdDdDd

        v_uint32 v_res0 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), v_reinterpret_as_s16(v_mul0)));
        v_uint32 v_res1 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), v_reinterpret_as_s16(v_mul1)));
        v_uint32 v_res2 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src2), v_reinterpret_as_s16(v_mul2)));
        v_uint32 v_res3 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src3), v_reinterpret_as_s16(v_mul3)));
        v_store((uint16_t*)dst        , v_pack(v_res0, v_res1));
        v_store((uint16_t*)dst + VECSZ, v_pack(v_res2, v_res3));
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + 4 * ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[4];
        *(dst++) = m[0] * px[1] + m[1] * px[5];
        *(dst++) = m[0] * px[2] + m[1] * px[6];
        *(dst++) = m[0] * px[3] + m[1] * px[7];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ((ufixedpoint16*)(srccn.w))[0] = (src + 4 * ofst[dst_width - 1])[0]; ((ufixedpoint16*)(srccn.w))[1] = (src + 4 * ofst[dst_width - 1])[1];
    ((ufixedpoint16*)(srccn.w))[2] = (src + 4 * ofst[dst_width - 1])[2]; ((ufixedpoint16*)(srccn.w))[3] = (src + 4 * ofst[dst_width - 1])[3];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_srccn = v_reinterpret_as_u16(vx_setall_u64(srccn.q));
    for (; i <= dst_width - VECSZ/4; i += VECSZ/4, dst += VECSZ) // Points that fall right from src image so became equal to rightmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[3];
    }
}
template <>
void hlineResizeCn<uint16_t, ufixedpoint32, 2, true, 1>(uint16_t* src, int, int *ofst, ufixedpoint32* m, ufixedpoint32* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    ufixedpoint32 src_0(src[0]);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint32>::vlanes();
    v_uint32 v_src_0 = vx_setall_u32(*((uint32_t*)&src_0));
    for (; i <= dst_min - VECSZ; i += VECSZ, m += 2*VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint32_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = src_0;
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - VECSZ; i += VECSZ, m += 2*VECSZ, dst += VECSZ)
    {
        v_uint32 v_src0, v_src1;
        v_expand(vx_lut_pairs(src, ofst + i), v_src0, v_src1);

        v_uint64 v_res0 = v_reinterpret_as_u64(v_mul(v_src0, vx_load((uint32_t *)m)));
        v_uint64 v_res1 = v_reinterpret_as_u64(v_mul(v_src1, vx_load((uint32_t *)m + VECSZ)));
        v_store((uint32_t*)dst, v_pack(v_add(v_and(v_res0, vx_setall_u64(0xFFFFFFFF)), v_shr<32>(v_res0)),
                                       v_add(v_and(v_res1, vx_setall_u64(0xFFFFFFFF)), v_shr<32>(v_res1))));
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint16_t* px = src + ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[1];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    src_0 = (src + ofst[dst_width - 1])[0];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_src_0 = vx_setall_u32(*((uint32_t*)&src_0));
    for (; i <= dst_width - VECSZ; i += VECSZ, dst += VECSZ)
    {
        v_store((uint32_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = src_0;
    }
}

template <typename ET, typename FT>
void vlineSet(FT* src, ET* dst, int dst_width)
{
    for (int i = 0; i < dst_width; i++)
        dst[i] = src[i];
}
template <>
void vlineSet<uint8_t, ufixedpoint16>(ufixedpoint16* src, uint8_t* dst, int dst_width)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint8>::vlanes();
    const v_uint16 v_fixedRound = vx_setall_u16((uint16_t)((1U << 8) >> 1));
    for (; i <= dst_width - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
    {
        v_uint16 v_src0 = vx_load((uint16_t*)src);
        v_uint16 v_src1 = vx_load((uint16_t*)src + VECSZ/2);

        v_uint16 v_res0 = v_shr<8>(v_add(v_src0, v_fixedRound));
        v_uint16 v_res1 = v_shr<8>(v_add(v_src1, v_fixedRound));

        v_store(dst, v_pack(v_res0, v_res1));
    }
#endif
    for (; i < dst_width; i++)
        *(dst++) = *(src++);
}

template <typename ET, typename FT, int n>
void vlineResize(FT* src, size_t src_step, FT* m, ET* dst, int dst_width)
{
    for (int i = 0; i < dst_width; i++)
    {
        typename FT::WT res = src[i] * m[0];
        for (int k = 1; k < n; k++)
            res = res + src[i + k*src_step] * m[k];
        dst[i] = res;
    }
}
template <>
void vlineResize<uint8_t, ufixedpoint16, 2>(ufixedpoint16* src, size_t src_step, ufixedpoint16* m, uint8_t* dst, int dst_width)
{
    int i = 0;
    ufixedpoint16* src1 = src + src_step;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint8>::vlanes();
    const v_int32 v_fixedRound = vx_setall_s32((int32_t)((1 << 16) >> 1));
    const v_int16 v_128    = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1<<15));
    const v_int8  v_128_16 = v_reinterpret_as_s8 (vx_setall_u8 ((uint8_t) 1<<7));

    v_int16 v_mul = v_reinterpret_as_s16(vx_setall_u32(((uint32_t*)m)[0]));
    for (; i <= dst_width - VECSZ; i += VECSZ, src += VECSZ, src1 += VECSZ, dst += VECSZ)
    {
        v_int16 v_src00 = vx_load((int16_t*)src);
        v_int16 v_src10 = vx_load((int16_t*)src1);
        v_int16 v_tmp0, v_tmp1;
        v_zip(v_add_wrap(v_src00,v_128), v_add_wrap(v_src10,v_128), v_tmp0, v_tmp1);

        v_int32 v_res0 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res1 = v_dotprod(v_tmp1, v_mul);

        v_int16 v_src01 = vx_load((int16_t*)src + VECSZ/2);
        v_int16 v_src11 = vx_load((int16_t*)src1 + VECSZ/2);
        v_zip(v_add_wrap(v_src01,v_128), v_add_wrap(v_src11,v_128), v_tmp0, v_tmp1);
        v_int32 v_res2 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res3 = v_dotprod(v_tmp1, v_mul);

        v_int8 v_res = v_pack(v_pack(v_shr<16>(v_add(v_res0, v_fixedRound)),
                                     v_shr<16>(v_add(v_res1, v_fixedRound))),
                              v_pack(v_shr<16>(v_add(v_res2, v_fixedRound)),
                                     v_shr<16>(v_add(v_res3, v_fixedRound))));

        v_store(dst, v_reinterpret_as_u8(v_sub_wrap(v_res, v_128_16)));
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = (uint8_t)(*(src++) * m[0] + *(src1++) * m[1]);
    }
}

template <typename ET> class interpolationLinear
{
public:
    static const int len = 2;
    static const bool needsign = false;
    interpolationLinear(double inv_scale, int srcsize, int dstsize) : scale(softdouble::one() / softdouble(inv_scale)), maxsize(srcsize), minofst(0), maxofst(dstsize) {}
    void getCoeffs(int val, int* offset, typename fixedtype<ET, needsign>::type* coeffs)
    {
        typedef typename fixedtype<ET, needsign>::type fixedpoint;
        softdouble fval = scale*(softdouble(val)+softdouble(0.5))-softdouble(0.5);
        int ival = cvFloor(fval);
        if (ival >= 0 && maxsize > 1)
        {
            if (ival < maxsize - 1)
            {
                *offset = ival;
                coeffs[1] = fval - softdouble(ival);
                coeffs[0] = fixedpoint::one() - coeffs[1];
            }
            else
            {
                *offset = maxsize - 1;
                maxofst = min(maxofst, val);
            }
        }
        else
        {
            minofst = max(minofst, val + 1);
        }
    }
    void getMinMax(int &min, int &max)
    {
        min = minofst;
        max = maxofst;
    }
protected:
    softdouble scale;
    int maxsize;
    int minofst, maxofst;
};

template <typename ET, typename FT, int interp_y_len>
class resize_bitExactInvoker :
    public ParallelLoopBody
{
public:
    typedef FT fixedpoint;
    typedef void(*hResizeFunc)(ET* src, int cn, int *ofst, fixedpoint* m, fixedpoint* dst, int dst_min, int dst_max, int dst_width);
    resize_bitExactInvoker(const uchar* _src, size_t _src_step, int _src_width, int _src_height,
                           uchar* _dst, size_t _dst_step, int _dst_width, int _dst_height,
                           int _cn, int *_xoffsets, int *_yoffsets, fixedpoint *_xcoeffs, fixedpoint *_ycoeffs,
                           int _min_x, int _max_x, int _min_y, int _max_y, hResizeFunc _hResize) : ParallelLoopBody(),
                           src(_src), src_step(_src_step), src_width(_src_width), src_height(_src_height),
                           dst(_dst), dst_step(_dst_step), dst_width(_dst_width), dst_height(_dst_height),
                           cn(_cn), xoffsets(_xoffsets), yoffsets(_yoffsets), xcoeffs(_xcoeffs), ycoeffs(_ycoeffs),
                           min_x(_min_x), max_x(_max_x), min_y(_min_y), max_y(_max_y), hResize(_hResize) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        AutoBuffer<fixedpoint> linebuf(interp_y_len * dst_width * cn);
        int last_eval = - interp_y_len;
        int evalbuf_start = 0;
        int rmin_y = max(min_y, range.start);
        int rmax_y = min(max_y, range.end);
        if (range.start < min_y)
        {
            last_eval = 1 - interp_y_len;
            evalbuf_start = 1;
            hResize((ET*)src, cn, xoffsets, xcoeffs, linebuf.data(), min_x, max_x, dst_width);
        }
        int dy = range.start;
        for (; dy < rmin_y; dy++)
            vlineSet<ET, FT>(linebuf.data(), (ET*)(dst + dst_step * dy), dst_width*cn);
        for (; dy < rmax_y; dy++)
        {
            int &iy = yoffsets[dy];

            int i;
            for (i = max(iy, last_eval + interp_y_len); i < min(iy + interp_y_len, src_height); i++, evalbuf_start = (evalbuf_start + 1) % interp_y_len)
                hResize((ET*)(src + i * src_step), cn, xoffsets, xcoeffs, linebuf.data() + evalbuf_start*(dst_width * cn), min_x, max_x, dst_width);
            evalbuf_start = (evalbuf_start + max(iy, src_height - interp_y_len) - max(last_eval, src_height - interp_y_len)) % interp_y_len;
            last_eval = iy;

            fixedpoint curcoeffs[interp_y_len];
            for (i = 0; i < evalbuf_start; i++)
                curcoeffs[i] = ycoeffs[ dy*interp_y_len - evalbuf_start + interp_y_len + i];
            for (; i < interp_y_len; i++)
                curcoeffs[i] = ycoeffs[ dy*interp_y_len - evalbuf_start + i];

            vlineResize<ET, FT, interp_y_len>(linebuf.data(), dst_width*cn, curcoeffs, (ET*)(dst + dst_step * dy), dst_width*cn);
        }
        fixedpoint *endline = linebuf.data();
        if (last_eval + interp_y_len > src_height)
            endline += dst_width*cn*((evalbuf_start + src_height - 1 - last_eval) % interp_y_len);
        else
            hResize((ET*)(src + (src_height - 1) * src_step), cn, xoffsets, xcoeffs, endline, min_x, max_x, dst_width);
        for (; dy < range.end; dy++)
            vlineSet<ET, FT>(endline, (ET*)(dst + dst_step * dy), dst_width*cn);
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vx_cleanup();
#endif
    }

private:
    const uchar* src;
    size_t src_step;
    int src_width, src_height;
    uchar* dst;
    size_t dst_step;
    int dst_width, dst_height, cn;
    int *xoffsets, *yoffsets;
    fixedpoint *xcoeffs, *ycoeffs;
    int min_x, max_x, min_y, max_y;
    hResizeFunc hResize;

    resize_bitExactInvoker(const resize_bitExactInvoker&);
    resize_bitExactInvoker& operator=(const resize_bitExactInvoker&);
};

// Geometry-only half of the bit-exact resize: tables plus its horizontal kernel.
struct BitExactTabs
{
    int* xoffsets;
    int* yoffsets;
    void* xcoeffs;
    void* ycoeffs;
    int min_x, max_x, min_y, max_y;
    void* hResize;   // type-erased: its signature depends on the run half's fixed-point type
};

template <typename ET, typename interpolation>
void resize_bitExact_build(int src_width, int src_height, int dst_width, int dst_height, int cn,
                           double inv_scale_x, double inv_scale_y,
                           AutoBuffer<uchar>& buf, BitExactTabs& tabs)
{
    typedef typename fixedtype<ET, interpolation::needsign>::type fixedpoint;
    void(*hResize)(ET* src, int cn, int *ofst, fixedpoint* m, fixedpoint* dst, int dst_min, int dst_max, int dst_width);
    switch (cn)
    {
    case  1: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 1> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 1>; break;
    case  2: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 2> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 2>; break;
    case  3: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 3> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 3>; break;
    case  4: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 4> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 4>; break;
    default: hResize = src_width > interpolation::len ? hlineResize<ET, fixedpoint, interpolation::len, true>      : hlineResize<ET, fixedpoint, interpolation::len, false>     ; break;
    }

    interpolation interp_x(inv_scale_x, src_width, dst_width);
    interpolation interp_y(inv_scale_y, src_height, dst_height);

    buf.allocate( dst_width * sizeof(int) +
                  dst_height * sizeof(int) +
                  dst_width * interp_x.len*sizeof(fixedpoint) +
                  dst_height * interp_y.len * sizeof(fixedpoint) );
    int* xoffsets = (int*)buf.data();
    int* yoffsets = xoffsets + dst_width;
    fixedpoint* xcoeffs = (fixedpoint*)(yoffsets + dst_height);
    fixedpoint* ycoeffs = xcoeffs + dst_width * interp_x.len;

    for (int dx = 0; dx < dst_width; dx++)
        interp_x.getCoeffs(dx, xoffsets+dx, xcoeffs+dx*interp_x.len);
    interp_x.getMinMax(tabs.min_x, tabs.max_x);
    for (int dy = 0; dy < dst_height; dy++)
        interp_y.getCoeffs(dy, yoffsets+dy, ycoeffs+dy*interp_y.len);
    interp_y.getMinMax(tabs.min_y, tabs.max_y);

    tabs.xoffsets = xoffsets;
    tabs.yoffsets = yoffsets;
    tabs.xcoeffs = xcoeffs;
    tabs.ycoeffs = ycoeffs;
    tabs.hResize = (void*)hResize;
}

template <typename ET, typename interpolation>
void resize_bitExact_run(const uchar* src, size_t src_step, int src_width, int src_height,
                         uchar* dst, size_t dst_step, int dst_width, int dst_height, int cn,
                         const BitExactTabs& tabs, const Range& range)
{
    typedef typename fixedtype<ET, interpolation::needsign>::type fixedpoint;
    typedef void(*hResizeFunc)(ET* src, int cn, int *ofst, fixedpoint* m,
                               fixedpoint* dst, int dst_min, int dst_max, int dst_width);

    resize_bitExactInvoker<ET, fixedpoint, interpolation::len> invoker(
        src, src_step, src_width, src_height, dst, dst_step, dst_width, dst_height, cn,
        tabs.xoffsets, tabs.yoffsets, (fixedpoint*)tabs.xcoeffs, (fixedpoint*)tabs.ycoeffs,
        tabs.min_x, tabs.max_x, tabs.min_y, tabs.max_y, (hResizeFunc)tabs.hResize);
    invoker(range);
}

typedef void(*be_build_func)(int src_width, int src_height, int dst_width, int dst_height, int cn,
                             double inv_scale_x, double inv_scale_y,
                             AutoBuffer<uchar>& buf, BitExactTabs& tabs);

typedef void(*be_run_func)(const uchar* src, size_t src_step, int src_width, int src_height,
                           uchar* dst, size_t dst_step, int dst_width, int dst_height, int cn,
                           const BitExactTabs& tabs, const Range& range);

}

namespace cv
{

/************** interpolation formulas and tables ***************/

const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

static inline void interpolateCubic( float x, float* coeffs, float A = -0.75f )
{
    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void interpolateLanczos4( float x, float* coeffs )
{
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2]=
    {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

    float sum = 0;
    double y0=-(x+3)*CV_PI*0.25, s0 = std::sin(y0), c0= std::cos(y0);
    for(int i = 0; i < 8; i++ )
    {
        float y0_ = (x+3-i);
        if (fabs(y0_) >= 1e-6f)
        {
            double y = -y0_*CV_PI*0.25;
            coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
        }
        else
        {
            // special handling for 'x' values:
            // - ~0.0: 0 0 0 1 0 0 0 0
            // - ~1.0: 0 0 0 0 1 0 0 0
            coeffs[i] = 1e30f;
        }
        sum += coeffs[i];
    }

    sum = 1.f/sum;
    for(int i = 0; i < 8; i++ )
        coeffs[i] *= sum;
}

template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
};

/****************************************************************************************\
*                                         Resize                                         *
\****************************************************************************************/

class resizeNNInvoker :
    public ParallelLoopBody
{
public:
    resizeNNInvoker(const Mat& _src, Mat &_dst, int *_x_ofs, const int* _y_ofs) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        y_ofs(_y_ofs)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int y, x, pix_size = (int)src.elemSize();

        for( y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.data + dst.step*y;
            const uchar* S = src.ptr(y_ofs[y]);

            switch( pix_size )
            {
            case 1:
                for( x = 0; x <= dsize.width - 2; x += 2 )
                {
                    uchar t0 = S[x_ofs[x]];
                    uchar t1 = S[x_ofs[x+1]];
                    D[x] = t0;
                    D[x+1] = t1;
                }

                for( ; x < dsize.width; x++ )
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
                for( x = 0; x < dsize.width; x++ )
                    *(ushort*)(D + x*2) = *(ushort*)(S + x_ofs[x]);
                break;
            case 3:
                for( x = 0; x < dsize.width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofs[x];
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
                for( x = 0; x < dsize.width; x++ )
                    *(int*)(D + x*4) = *(int*)(S + x_ofs[x]);
                break;
            case 6:
                for( x = 0; x < dsize.width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofs[x]);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
                for( x = 0; x < dsize.width; x++, D += 8 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1];
                }
                break;
            case 12:
                for( x = 0; x < dsize.width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            default:
                for( x = 0; x < dsize.width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofs[x];
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int* y_ofs;

    resizeNNInvoker(const resizeNNInvoker&);
    resizeNNInvoker& operator=(const resizeNNInvoker&);
};

static void
resizeNN( const Mat& src, Mat& dst, int* x_ofs, const int* y_ofs, const Range& range )
{
    int pix_size = (int)src.elemSize();
#if CV_TRY_AVX2
    if(CV_CPU_HAS_SUPPORT_AVX2 && ((pix_size == 2) || (pix_size == 4)))
    {
        if(pix_size == 2)
            opt_AVX2::resizeNN2_AVX2(range, src, dst, x_ofs, y_ofs);
        else
            opt_AVX2::resizeNN4_AVX2(range, src, dst, x_ofs, y_ofs);
    }
    else
#endif
#if CV_TRY_SSE4_1
    if(CV_CPU_HAS_SUPPORT_SSE4_1 && ((pix_size == 2) || (pix_size == 4)))
    {
        if(pix_size == 2)
            opt_SSE4_1::resizeNN2_SSE4_1(range, src, dst, x_ofs, y_ofs);
        else
            opt_SSE4_1::resizeNN4_SSE4_1(range, src, dst, x_ofs, y_ofs);
    }
    else
#endif
#if CV_TRY_LASX
    if(CV_CPU_HAS_SUPPORT_LASX && ((pix_size == 2) || (pix_size == 4)))
    {
        if(pix_size == 2)
            opt_LASX::resizeNN2_LASX(range, src, dst, x_ofs, y_ofs);
        else
            opt_LASX::resizeNN4_LASX(range, src, dst, x_ofs, y_ofs);
    }
    else
#endif
    {
        resizeNNInvoker invoker(src, dst, x_ofs, y_ofs);
        invoker(range);
    }
}

class resizeNN_bitexactInvoker : public ParallelLoopBody
{
public:
    resizeNN_bitexactInvoker(const Mat& _src, Mat& _dst, int* _x_ofse, int* _y_ofse)
        : src(_src), dst(_dst), x_ofse(_x_ofse), y_ofse(_y_ofse) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int pix_size = (int)src.elemSize();
        for( int y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.ptr(y);
            int sy = y_ofse[y];
            const uchar* S = src.ptr(sy);

            int x = 0;
            switch( pix_size )
            {
            case 1:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes() )
                    v_store(D + x, vx_lut(S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    D[x] = S[x_ofse[x]];
                break;
            case 2:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes() )
                    v_store((ushort*)D + x, vx_lut((ushort*)S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    *((ushort*)D + x) = *((ushort*)S + x_ofse[x]);
                break;
            case 3:
                for( ; x < dsize.width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofse[x] * 3;
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes() )
                    v_store((uint32_t*)D + x, vx_lut((uint32_t*)S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    *((uint32_t*)D + x) = *((uint32_t*)S + x_ofse[x]);
                break;
            case 6:
                for( ; x < dsize.width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofse[x]*6);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint64>::vlanes(); x += VTraits<v_uint64>::vlanes() )
                    v_store((uint64_t*)D + x, vx_lut((uint64_t*)S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    *((uint64_t*)D + x) = *((uint64_t*)S + x_ofse[x]);
                break;
            case 12:
                for( ; x < dsize.width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofse[x]*12);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            default:
                for( x = 0; x < dsize.width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofse[x] * pix_size;
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }
            }
        }
    }
private:
    const Mat& src;
    Mat& dst;
    int* x_ofse;
    int* y_ofse;
};

static void resizeNN_bitexact_tab(int src_dim, int dst_dim, int* ofse)
{
    // Match Pillow's nearest-neighbor resize path: start at half a pixel,
    // truncate the current coordinate, then increment by scale. softdouble
    // keeps the IEEE-754 rounding deterministic across platforms.
    const softdouble scale = softdouble(src_dim) / softdouble(dst_dim);
    softdouble f = scale * softdouble(0.5);
    for( int i = 0; i < dst_dim; i++ )
    {
        ofse[i] = std::min(cvFloor(f), src_dim-1);
        f += scale;
    }
}

static void resizeNN_bitexact( const Mat& src, Mat& dst, int* x_ofse, int* y_ofse,
                               const Range& range )
{
    resizeNN_bitexactInvoker invoker(src, dst, x_ofse, y_ofse);
    invoker(range);
}

struct VResizeNoVec
{
    template<typename WT, typename T, typename BT>
    int operator()(const WT**, T*, const BT*, int ) const
    {
        return 0;
    }
};

struct HResizeNoVec
{
    template<typename T, typename WT, typename AT> inline
    int operator()(const T**, WT**, int, const int*,
        const AT*, int, int, int, int, int) const
    {
        return 0;
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)

struct VResizeLinearVec_32s8u
{
    int operator()(const int** src, uchar* dst, const short* beta, int width) const
    {
        const int *S0 = src[0], *S1 = src[1];
        int x = 0;
        v_int16 b0 = vx_setall_s16(beta[0]), b1 = vx_setall_s16(beta[1]);

        if( (((size_t)S0|(size_t)S1)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
            for( ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
                v_store(dst + x, v_rshr_pack_u<2>(v_add(v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S0 + x)), v_shr<4>(vx_load_aligned(S0 + x + VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S1 + x)), v_shr<4>(vx_load_aligned(S1 + x + VTraits<v_int32>::vlanes()))), b1)),
                                                  v_add(v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S0 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load_aligned(S0 + x + 3 * VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S1 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load_aligned(S1 + x + 3 * VTraits<v_int32>::vlanes()))), b1))));
        else
            for( ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
                v_store(dst + x, v_rshr_pack_u<2>(v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x)), v_shr<4>(vx_load(S0 + x + VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x)), v_shr<4>(vx_load(S1 + x + VTraits<v_int32>::vlanes()))), b1)),
                                                  v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load(S0 + x + 3 * VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load(S1 + x + 3 * VTraits<v_int32>::vlanes()))), b1))));

        for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
            v_rshr_pack_u_store<2>(dst + x, v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x)), v_shr<4>(vx_load(S0 + x + VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x)), v_shr<4>(vx_load(S1 + x + VTraits<v_int32>::vlanes()))), b1)));

        return x;
    }
};

// Unified float vertical-resize kernel collapsing the nine VResize*Vec_32f* structs:
// an N-tap muladd chain (exact original nesting -> bit-identical) + a pack policy.
template <int K, int N, bool Aligned>
static inline v_float32 vresizeVecChain(const float** S, const float* beta, int x)
{
    v_float32 s;
    if constexpr (Aligned)
        s = vx_load_aligned(S[K] + x);
    else
        s = vx_load(S[K] + x);
    v_float32 bK = vx_setall_f32(beta[K]);
    if constexpr (K + 1 == N)
        return v_mul(s, bK);
    else
        return v_muladd(s, bK, vresizeVecChain<K + 1, N, Aligned>(S, beta, x));
}

struct VPackF32
{
    typedef float DT;
    enum { groups = 1 };
    static int step() { return VTraits<v_float32>::vlanes(); }
    static void store(DT* dst, const v_float32& a0) { v_store(dst, a0); }
};

template <typename DT_, bool Unsigned>
struct VPack16  // 16u (v_pack_u) / 16s (v_pack) - selected at compile time
{
    typedef DT_ DT;
    enum { groups = 2 };
    static int step() { return VTraits<v_int16>::vlanes(); }
    static void store(DT* dst, const v_float32& a0, const v_float32& a1) { if constexpr (Unsigned) v_store(dst, v_pack_u(v_round(a0), v_round(a1))); else v_store(dst, v_pack(v_round(a0), v_round(a1))); }
    static void store_low(DT* dst, const v_int32& t0) { if constexpr (Unsigned) v_store_low(dst, v_pack_u(t0, t0)); else v_store_low(dst, v_pack(t0, t0)); }
};

template <int N, typename Pack, bool Aligned>
static inline void vresizeVecStore(typename Pack::DT* dst, const float** src, const float* beta, int x)
{
    if constexpr (Pack::groups == 1)
        Pack::store(dst, vresizeVecChain<0, N, Aligned>(src, beta, x));
    else
        Pack::store(dst, vresizeVecChain<0, N, Aligned>(src, beta, x),
                         vresizeVecChain<0, N, Aligned>(src, beta, x + VTraits<v_float32>::vlanes()));
}

template <int N, typename Pack>
struct VResizeVec_32f
{
    int operator()(const float** src, typename Pack::DT* dst, const float* beta, int width) const
    {
        int x = 0;
        const int step = Pack::step();
        if constexpr (N == 2)
        {
            // Linear-only aligned fast path (matches the original, which only the N=2 kernels had).
            const float *S0 = src[0], *S1 = src[1];
            if( (((size_t)S0|(size_t)S1)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
                for( ; x <= width - step; x += step )
                    vresizeVecStore<N, Pack, true>(dst + x, src, beta, x);
            else
                for( ; x <= width - step; x += step )
                    vresizeVecStore<N, Pack, false>(dst + x, src, beta, x);
        }
        else
        {
            for( ; x <= width - step; x += step )
                vresizeVecStore<N, Pack, false>(dst + x, src, beta, x);
        }
        if constexpr (N == 2 && Pack::groups == 2)
        {
            // Linear 16u/16s float-width tail (matches the original).
            for( ; x <= width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes() )
                Pack::store_low(dst + x, v_round(vresizeVecChain<0, N, false>(src, beta, x)));
        }
        return x;
    }
};

typedef VResizeVec_32f<2, VPack16<ushort, true> > VResizeLinearVec_32f16u;
typedef VResizeVec_32f<2, VPack16<short, false> > VResizeLinearVec_32f16s;
typedef VResizeVec_32f<2, VPackF32> VResizeLinearVec_32f;


struct VResizeCubicVec_32s8u
{
    int operator()(const int** src, uchar* dst, const short* beta, int width) const
    {
        const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        float scale = 1.f/(INTER_RESIZE_COEF_SCALE*INTER_RESIZE_COEF_SCALE);

        v_float32 b0 = vx_setall_f32(beta[0] * scale), b1 = vx_setall_f32(beta[1] * scale),
                  b2 = vx_setall_f32(beta[2] * scale), b3 = vx_setall_f32(beta[3] * scale);

        if( (((size_t)S0|(size_t)S1|(size_t)S2|(size_t)S3)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
            for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
                v_pack_u_store(dst + x, v_pack(v_round(v_muladd(v_cvt_f32(vx_load_aligned(S0 + x                    )),  b0,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S1 + x                    )),  b1,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S2 + x                    )),  b2,
                                                                v_mul(v_cvt_f32(vx_load_aligned(S3 + x)), b3))))),
                                               v_round(v_muladd(v_cvt_f32(vx_load_aligned(S0 + x + VTraits<v_float32>::vlanes())),  b0,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S1 + x + VTraits<v_float32>::vlanes())),  b1,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S2 + x + VTraits<v_float32>::vlanes())),  b2,
                                                                v_mul(v_cvt_f32(vx_load_aligned(S3 + x + VTraits<v_float32>::vlanes())), b3)))))));
        else
            for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
                v_pack_u_store(dst + x, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S0 + x                    )),  b0,
                                                       v_muladd(v_cvt_f32(vx_load(S1 + x                    )),  b1,
                                                       v_muladd(v_cvt_f32(vx_load(S2 + x                    )),  b2,
                                                                v_mul(v_cvt_f32(vx_load(S3 + x)), b3))))),
                                               v_round(v_muladd(v_cvt_f32(vx_load(S0 + x + VTraits<v_float32>::vlanes())),  b0,
                                                       v_muladd(v_cvt_f32(vx_load(S1 + x + VTraits<v_float32>::vlanes())),  b1,
                                                       v_muladd(v_cvt_f32(vx_load(S2 + x + VTraits<v_float32>::vlanes())),  b2,
                                                                v_mul(v_cvt_f32(vx_load(S3 + x + VTraits<v_float32>::vlanes())), b3)))))));
        return x;
    }
};

typedef VResizeVec_32f<4, VPack16<ushort, true> > VResizeCubicVec_32f16u;
typedef VResizeVec_32f<4, VPack16<short, false> > VResizeCubicVec_32f16s;
typedef VResizeVec_32f<4, VPackF32> VResizeCubicVec_32f;


#if CV_TRY_SSE4_1

struct VResizeLanczos4Vec_32f16u
{
    int operator()(const float** src, ushort* dst, const float* beta, int width) const
    {
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::VResizeLanczos4Vec_32f16u_SSE41(src, dst, beta, width);
        else
            return 0;
    }
};

#else

typedef VResizeVec_32f<8, VPack16<ushort, true> > VResizeLanczos4Vec_32f16u;

#endif

typedef VResizeVec_32f<8, VPack16<short, false> > VResizeLanczos4Vec_32f16s;
typedef VResizeVec_32f<8, VPackF32> VResizeLanczos4Vec_32f;

#else

typedef VResizeNoVec VResizeLinearVec_32s8u;
typedef VResizeNoVec VResizeLinearVec_32f16u;
typedef VResizeNoVec VResizeLinearVec_32f16s;
typedef VResizeNoVec VResizeLinearVec_32f;

typedef VResizeNoVec VResizeCubicVec_32s8u;
typedef VResizeNoVec VResizeCubicVec_32f16u;
typedef VResizeNoVec VResizeCubicVec_32f16s;
typedef VResizeNoVec VResizeCubicVec_32f;

typedef VResizeNoVec VResizeLanczos4Vec_32f16u;
typedef VResizeNoVec VResizeLanczos4Vec_32f16s;
typedef VResizeNoVec VResizeLanczos4Vec_32f;

#endif

#if CV_SIMD128

template<typename ST, typename DT, typename AT, typename DVT>
struct HResizeLinearVec_X4
{
    int operator()(const ST** src, DT** dst, int count, const int* xofs,
        const AT* alpha, int, int, int cn, int, int xmax) const
    {
        const int nlanes = 4;
        const int len0 = xmax & -nlanes;
        int dx = 0, k = 0;

        for( ; k <= (count - 2); k+=2 )
        {
            const ST *S0 = src[k];
            DT *D0 = dst[k];
            const ST *S1 = src[k+1];
            DT *D1 = dst[k+1];

            for( dx = 0; dx < len0; dx += nlanes )
            {
                int sx0 = xofs[dx+0];
                int sx1 = xofs[dx+1];
                int sx2 = xofs[dx+2];
                int sx3 = xofs[dx+3];
                DVT a_even;
                DVT a_odd;

                v_load_deinterleave(&alpha[dx*2], a_even, a_odd);
                DVT s0(S0[sx0], S0[sx1], S0[sx2], S0[sx3]);
                DVT s1(S0[sx0+cn], S0[sx1+cn], S0[sx2+cn], S0[sx3+cn]);
                DVT s0_u(S1[sx0], S1[sx1], S1[sx2], S1[sx3]);
                DVT s1_u(S1[sx0+cn], S1[sx1+cn], S1[sx2+cn], S1[sx3+cn]);
                v_store(&D1[dx], v_add(v_mul(s0_u, a_even), v_mul(s1_u, a_odd)));
                v_store(&D0[dx], v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
            }
        }
        for( ; k < count; k++ )
        {
            const ST *S = src[k];
            DT *D = dst[k];
            for( dx = 0; dx < len0; dx += nlanes )
            {
                int sx0 = xofs[dx+0];
                int sx1 = xofs[dx+1];
                int sx2 = xofs[dx+2];
                int sx3 = xofs[dx+3];
                DVT a_even;
                DVT a_odd;

                v_load_deinterleave(&alpha[dx*2], a_even, a_odd);
                DVT s0(S[sx0], S[sx1], S[sx2], S[sx3]);
                DVT s1(S[sx0+cn], S[sx1+cn], S[sx2+cn], S[sx3+cn]);
                v_store(&D[dx], v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
            }
        }
        return dx;
    }
};

struct HResizeLinearVecU8_X4
{
    int operator()(const uchar** src, int** dst, int count, const int* xofs,
        const short* alpha/*[xmax]*/, int /*smax*/, int dmax, int cn, int /*xmin*/, int xmax) const
    {
        int dx = 0, k = 0;

        if(cn == 1)
        {
            const int step = 8;
            const int len0 = xmax & -step;
            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_lut_pairs(S0, xofs+dx), sl, sh);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D0[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                    v_expand(v_lut_pairs(S1, xofs+dx), sl, sh);
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D1[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_lut_pairs(S, xofs+dx), sl, sh);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
        }
        else if(cn == 2)
        {
            const int step = 8;
            const int len0 = xmax & -step;
            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; dx < len0; dx += step )
                {
                    int ofs[4] = { xofs[dx], xofs[dx + 2], xofs[dx + 4], xofs[dx + 6] };
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_interleave_pairs(v_lut_quads(S0, ofs)), sl, sh);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D0[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                    v_expand(v_interleave_pairs(v_lut_quads(S1, ofs)), sl, sh);
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D1[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; dx < len0; dx += step )
                {
                    int ofs[4] = { xofs[dx], xofs[dx + 2], xofs[dx + 4], xofs[dx + 6] };
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_interleave_pairs(v_lut_quads(S, ofs)), sl, sh);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
        }
        else if(cn == 3)
        {
            /* Peek at the last x offset to find the maximal s offset.  We know the loop
               will terminate prior to value which may be 1 or more elements prior to the
               final valid offset. xofs[] is constucted to be an array of increasingly
               large offsets (i.e xofs[x] <= xofs[x+1] for x < xmax). */
            int smax = xofs[dmax-cn];

            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; (xofs[dx] + cn) < smax; dx += cn )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S0 + xofs[dx]), v_shl<16>(v_load_expand_q(S0 + xofs[dx] + cn)))), a));
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S1 + xofs[dx]), v_shl<16>(v_load_expand_q(S1 + xofs[dx] + cn)))), a));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; (xofs[dx] + cn) < smax; dx += cn )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S + xofs[dx]), v_shl<16>(v_load_expand_q(S + xofs[dx] + cn)))), a));
                }
            }
            /* Debug check to ensure truthiness that we never vector the final value. */
            CV_DbgAssert(dx < dmax);
        }
        else if(cn == 4)
        {
            const int step = 4;
            const int len0 = xmax & -step;
            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S0+xofs[dx]))), a));
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S1+xofs[dx]))), a));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S+xofs[dx]))), a));
                }
            }
        }
        else
        {
            return 0;  // images with channels >4 are out of optimization scope
        }
        return dx;
    }
};

typedef HResizeLinearVec_X4<float,float,float,v_float32x4> HResizeLinearVec_32f;
typedef HResizeLinearVec_X4<ushort,float,float,v_float32x4> HResizeLinearVec_16u32f;
typedef HResizeLinearVec_X4<short,float,float,v_float32x4> HResizeLinearVec_16s32f;
typedef HResizeLinearVecU8_X4 HResizeLinearVec_8u32s;

#else

typedef HResizeNoVec HResizeLinearVec_8u32s;
typedef HResizeNoVec HResizeLinearVec_16u32f;
typedef HResizeNoVec HResizeLinearVec_16s32f;
typedef HResizeNoVec HResizeLinearVec_32f;

#endif

typedef HResizeNoVec HResizeLinearVec_64f;


template<typename T, typename WT, typename AT, int ONE, class VecOp>
struct HResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        int dx, k;
        VecOp vecOp;

        int dx0 = vecOp(src, dst, count,
            xofs, alpha, swidth, dwidth, cn, xmin, xmax );

        for( k = 0; k <= count - 2; k+=2 )
        {
            const T *S0 = src[k], *S1 = src[k+1];
            WT *D0 = dst[k], *D1 = dst[k+1];
            for( dx = dx0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                WT a0 = alpha[dx*2], a1 = alpha[dx*2+1];
                WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
                WT t1 = S1[sx]*a0 + S1[sx + cn]*a1;
                D0[dx] = t0; D1[dx] = t1;
            }

            for( ; dx < dwidth; dx++ )
            {
                int sx = xofs[dx];
                D0[dx] = WT(S0[sx]*ONE); D1[dx] = WT(S1[sx]*ONE);
            }
        }

        for( ; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            for( dx = dx0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                D[dx] = S[sx]*alpha[dx*2] + S[sx+cn]*alpha[dx*2+1];
            }

            for( ; dx < dwidth; dx++ )
                D[dx] = WT(S[xofs[dx]]*ONE);
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1];
        const WT *S0 = src[0], *S1 = src[1];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT t0, t1;
            t0 = S0[x]*b0 + S1[x]*b1;
            t1 = S0[x+1]*b0 + S1[x+1]*b1;
            dst[x] = castOp(t0); dst[x+1] = castOp(t1);
            t0 = S0[x+2]*b0 + S1[x+2]*b1;
            t1 = S0[x+3]*b0 + S1[x+3]*b1;
            dst[x+2] = castOp(t0); dst[x+3] = castOp(t1);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1);
    }
};

template<>
struct VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>, VResizeLinearVec_32s8u>
{
    typedef uchar value_type;
    typedef int buf_type;
    typedef short alpha_type;

    void operator()(const buf_type** src, value_type* dst, const alpha_type* beta, int width ) const
    {
        alpha_type b0 = beta[0], b1 = beta[1];
        const buf_type *S0 = src[0], *S1 = src[1];
        VResizeLinearVec_32s8u vecOp;

        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            dst[x+0] = uchar(( ((b0 * (S0[x+0] >> 4)) >> 16) + ((b1 * (S1[x+0] >> 4)) >> 16) + 2)>>2);
            dst[x+1] = uchar(( ((b0 * (S0[x+1] >> 4)) >> 16) + ((b1 * (S1[x+1] >> 4)) >> 16) + 2)>>2);
            dst[x+2] = uchar(( ((b0 * (S0[x+2] >> 4)) >> 16) + ((b1 * (S1[x+2] >> 4)) >> 16) + 2)>>2);
            dst[x+3] = uchar(( ((b0 * (S0[x+3] >> 4)) >> 16) + ((b1 * (S1[x+3] >> 4)) >> 16) + 2)>>2);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
    }
};


template<typename T, typename WT, typename AT, int K>
struct HResizeFilterK
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        for( int k = 0; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for(;;)
            {
                for( ; dx < limit; dx++, alpha += K )
                {
                    int j, sx = xofs[dx] - cn*(K/2 - 1);
                    WT v = 0;
                    for( j = 0; j < K; j++ )
                    {
                        int sxj = sx + j*cn;
                        if( (unsigned)sxj >= (unsigned)swidth )
                        {
                            while( sxj < 0 )
                                sxj += cn;
                            while( sxj >= swidth )
                                sxj -= cn;
                        }
                        v += S[sxj]*alpha[j];
                    }
                    D[dx] = v;
                }
                if( limit == dwidth )
                    break;
                for( ; dx < xmax; dx++, alpha += K )
                {
                    int sx = xofs[dx] - cn*(K/2 - 1);
                    WT v = S[sx]*alpha[0];
                    for( int j = 1; j < K; j++ )
                        v += S[sx + j*cn]*alpha[j];
                    D[dx] = v;
                }
                limit = dwidth;
            }
            alpha -= dwidth*K;
        }
    }
};

template<typename T, typename WT, typename AT> using HResizeCubic = HResizeFilterK<T, WT, AT, 4>;


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
        const WT *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp(src, dst, beta, width);
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1 + S2[x]*b2 + S3[x]*b3);
    }
};


template<typename T, typename WT, typename AT> using HResizeLanczos4 = HResizeFilterK<T, WT, AT, 8>;


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        CastOp castOp;
        VecOp vecOp;
        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT b = beta[0];
            const WT* S = src[0];
            WT s0 = S[x]*b, s1 = S[x+1]*b, s2 = S[x+2]*b, s3 = S[x+3]*b;

            for( int k = 1; k < 8; k++ )
            {
                b = beta[k]; S = src[k];
                s0 += S[x]*b; s1 += S[x+1]*b;
                s2 += S[x+2]*b; s3 += S[x+3]*b;
            }

            dst[x] = castOp(s0); dst[x+1] = castOp(s1);
            dst[x+2] = castOp(s2); dst[x+3] = castOp(s3);
        }
        #endif
        for( ; x < width; x++ )
        {
            dst[x] = castOp(src[0][x]*beta[0] + src[1][x]*beta[1] +
                src[2][x]*beta[2] + src[3][x]*beta[3] + src[4][x]*beta[4] +
                src[5][x]*beta[5] + src[6][x]*beta[6] + src[7][x]*beta[7]);
        }
    }
};


static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}

static const int MAX_ESIZE=16;

template <typename HResize, typename VResize>
class resizeGeneric_Invoker :
    public ParallelLoopBody
{
public:
    typedef typename HResize::value_type T;
    typedef typename HResize::buf_type WT;
    typedef typename HResize::alpha_type AT;

    resizeGeneric_Invoker(const Mat& _src, Mat &_dst, const int *_xofs, const int *_yofs,
        const AT* _alpha, const AT* __beta, const Size& _ssize, const Size &_dsize,
        int _ksize, int _xmin, int _xmax) :
        ParallelLoopBody(), src(_src), dst(_dst), xofs(_xofs), yofs(_yofs),
        alpha(_alpha), _beta(__beta), ssize(_ssize), dsize(_dsize),
        ksize(_ksize), xmin(_xmin), xmax(_xmax)
    {
        CV_Assert(ksize <= MAX_ESIZE);
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        int dy, cn = src.channels();
        HResize hresize;
        VResize vresize;

        int bufstep = (int)alignSize(dsize.width, 16);
        AutoBuffer<WT> _buffer(bufstep*ksize);
        const T* srows[MAX_ESIZE]={0};
        WT* rows[MAX_ESIZE]={0};
        int prev_sy[MAX_ESIZE];

        for(int k = 0; k < ksize; k++ )
        {
            prev_sy[k] = -1;
            rows[k] = _buffer.data() + bufstep*k;
        }

        const AT* beta = _beta + ksize * range.start;

        for( dy = range.start; dy < range.end; dy++, beta += ksize )
        {
            int sy0 = yofs[dy], k0=ksize, k1=0, ksize2 = ksize/2;

            for(int k = 0; k < ksize; k++ )
            {
                int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
                for( k1 = std::max(k1, k); k1 < ksize; k1++ )
                {
                    if( k1 < MAX_ESIZE && sy == prev_sy[k1] ) // if the sy-th row has been computed already, reuse it.
                    {
                        if( k1 > k )
                            memcpy( rows[k], rows[k1], bufstep*sizeof(rows[0][0]) );
                        break;
                    }
                }
                if( k1 == ksize )
                    k0 = std::min(k0, k); // remember the first row that needs to be computed
                srows[k] = src.template ptr<T>(sy);
                prev_sy[k] = sy;
            }

            if( k0 < ksize )
                hresize( (const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha),
                        ssize.width, dsize.width, cn, xmin, xmax );
            vresize( (const WT**)rows, (T*)(dst.data + dst.step*dy), beta, dsize.width );
        }
    }

private:
    Mat src;
    Mat dst;
    const int* xofs, *yofs;
    const AT* alpha, *_beta;
    Size ssize, dsize;
    const int ksize, xmin, xmax;

    resizeGeneric_Invoker& operator = (const resizeGeneric_Invoker&);
};

template<class HResize, class VResize>
static void resizeGeneric_( const Mat& src, Mat& dst,
                            const int* xofs, const void* _alpha,
                            const int* yofs, const void* _beta,
                            int xmin, int xmax, int ksize, const Range& range )
{
    typedef typename HResize::alpha_type AT;

    const AT* beta = (const AT*)_beta;
    Size ssize = src.size(), dsize = dst.size();
    int cn = src.channels();
    ssize.width *= cn;
    dsize.width *= cn;
    xmin *= cn;
    xmax *= cn;
    // image resize is a separable operation. In case of not too strong

    resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
        ssize, dsize, ksize, xmin, xmax);
    invoker(range);
}

template <typename T, typename WT>
struct ResizeAreaFastNoVec
{
    ResizeAreaFastNoVec(int, int) { }
    ResizeAreaFastNoVec(int, int, int, int) { }
    int operator() (const T*, T*, int) const
    { return 0; }
};

#if CV_NEON

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const uchar* S, uchar* D, int w) const
    {
        int dx = 0;
        const uchar* S0 = S, * S1 = S0 + step;

        uint16x8_t v_2 = vdupq_n_u16(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 16; dx += 16, S0 += 32, S1 += 32, D += 16)
            {
                uint8x16x2_t v_row0 = vld2q_u8(S0), v_row1 = vld2q_u8(S1);

                uint16x8_t v_dst0 = vaddl_u8(vget_low_u8(v_row0.val[0]), vget_low_u8(v_row0.val[1]));
                v_dst0 = vaddq_u16(v_dst0, vaddl_u8(vget_low_u8(v_row1.val[0]), vget_low_u8(v_row1.val[1])));
                v_dst0 = vshrq_n_u16(vaddq_u16(v_dst0, v_2), 2);

                uint16x8_t v_dst1 = vaddl_u8(vget_high_u8(v_row0.val[0]), vget_high_u8(v_row0.val[1]));
                v_dst1 = vaddq_u16(v_dst1, vaddl_u8(vget_high_u8(v_row1.val[0]), vget_high_u8(v_row1.val[1])));
                v_dst1 = vshrq_n_u16(vaddq_u16(v_dst1, v_2), 2);

                vst1q_u8(D, vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint8x16_t v_row0 = vld1q_u8(S0), v_row1 = vld1q_u8(S1);

                uint16x8_t v_row00 = vmovl_u8(vget_low_u8(v_row0));
                uint16x8_t v_row01 = vmovl_u8(vget_high_u8(v_row0));
                uint16x8_t v_row10 = vmovl_u8(vget_low_u8(v_row1));
                uint16x8_t v_row11 = vmovl_u8(vget_high_u8(v_row1));

                uint16x4_t v_p0 = vadd_u16(vadd_u16(vget_low_u16(v_row00), vget_high_u16(v_row00)),
                                           vadd_u16(vget_low_u16(v_row10), vget_high_u16(v_row10)));
                uint16x4_t v_p1 = vadd_u16(vadd_u16(vget_low_u16(v_row01), vget_high_u16(v_row01)),
                                           vadd_u16(vget_low_u16(v_row11), vget_high_u16(v_row11)));
                uint16x8_t v_dst = vshrq_n_u16(vaddq_u16(vcombine_u16(v_p0, v_p1), v_2), 2);

                vst1_u8(D, vmovn_u16(v_dst));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const ushort * S, ushort * D, int w) const
    {
        int dx = 0;
        const ushort * S0 = S, * S1 = (const ushort *)((const uchar *)(S0) + step);

        uint32x4_t v_2 = vdupq_n_u32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint16x8x2_t v_row0 = vld2q_u16(S0), v_row1 = vld2q_u16(S1);

                uint32x4_t v_dst0 = vaddl_u16(vget_low_u16(v_row0.val[0]), vget_low_u16(v_row0.val[1]));
                v_dst0 = vaddq_u32(v_dst0, vaddl_u16(vget_low_u16(v_row1.val[0]), vget_low_u16(v_row1.val[1])));
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_2), 2);

                uint32x4_t v_dst1 = vaddl_u16(vget_high_u16(v_row0.val[0]), vget_high_u16(v_row0.val[1]));
                v_dst1 = vaddq_u32(v_dst1, vaddl_u16(vget_high_u16(v_row1.val[0]), vget_high_u16(v_row1.val[1])));
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_2), 2);

                vst1q_u16(D, vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                uint16x8_t v_row0 = vld1q_u16(S0), v_row1 = vld1q_u16(S1);
                uint32x4_t v_dst = vaddq_u32(vaddl_u16(vget_low_u16(v_row0), vget_high_u16(v_row0)),
                                             vaddl_u16(vget_low_u16(v_row1), vget_high_u16(v_row1)));
                vst1_u16(D, vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const short * S, short * D, int w) const
    {
        int dx = 0;
        const short * S0 = S, * S1 = (const short *)((const uchar *)(S0) + step);

        int32x4_t v_2 = vdupq_n_s32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                int16x8x2_t v_row0 = vld2q_s16(S0), v_row1 = vld2q_s16(S1);

                int32x4_t v_dst0 = vaddl_s16(vget_low_s16(v_row0.val[0]), vget_low_s16(v_row0.val[1]));
                v_dst0 = vaddq_s32(v_dst0, vaddl_s16(vget_low_s16(v_row1.val[0]), vget_low_s16(v_row1.val[1])));
                v_dst0 = vshrq_n_s32(vaddq_s32(v_dst0, v_2), 2);

                int32x4_t v_dst1 = vaddl_s16(vget_high_s16(v_row0.val[0]), vget_high_s16(v_row0.val[1]));
                v_dst1 = vaddq_s32(v_dst1, vaddl_s16(vget_high_s16(v_row1.val[0]), vget_high_s16(v_row1.val[1])));
                v_dst1 = vshrq_n_s32(vaddq_s32(v_dst1, v_2), 2);

                vst1q_s16(D, vcombine_s16(vmovn_s32(v_dst0), vmovn_s32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                int16x8_t v_row0 = vld1q_s16(S0), v_row1 = vld1q_s16(S1);
                int32x4_t v_dst = vaddq_s32(vaddl_s16(vget_low_s16(v_row0), vget_high_s16(v_row0)),
                                            vaddl_s16(vget_low_s16(v_row1), vget_high_s16(v_row1)));
                vst1_s16(D, vmovn_s32(vshrq_n_s32(vaddq_s32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        float32x4_t v_025 = vdupq_n_f32(0.25f);

        if (cn == 1)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4x2_t v_row0 = vld2q_f32(S0), v_row1 = vld2q_f32(S1);

                float32x4_t v_dst0 = vaddq_f32(v_row0.val[0], v_row0.val[1]);
                float32x4_t v_dst1 = vaddq_f32(v_row1.val[0], v_row1.val[1]);

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4_t v_dst0 = vaddq_f32(vld1q_f32(S0), vld1q_f32(S0 + 4));
                float32x4_t v_dst1 = vaddq_f32(vld1q_f32(S1), vld1q_f32(S1 + 4));

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#elif CV_SIMD

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) :
        cn(_cn), step(_step) {}

    int operator() (const uchar* S, uchar* D, int w) const
    {
        int dx = 0;
        const uchar* S0 = S;
        const uchar* S1 = S0 + step;

        if (cn == 1)
        {
            v_uint16 masklow = vx_setall_u16(0x00ff);
            for ( ; dx <= w - VTraits<v_uint16>::vlanes(); dx += VTraits<v_uint16>::vlanes(), S0 += VTraits<v_uint8>::vlanes(), S1 += VTraits<v_uint8>::vlanes(), D += VTraits<v_uint16>::vlanes())
            {
                v_uint16 r0 = v_reinterpret_as_u16(vx_load(S0));
                v_uint16 r1 = v_reinterpret_as_u16(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<8>(r0), v_and(r0, masklow)), v_shr<8>(r1)), v_and(r1, masklow)));
            }
        }
        else if (cn == 3)
        {
            if (CV_SIMD_WIDTH > 64)
                return 0;
            for ( ; dx <= w - 3*VTraits<v_uint8>::vlanes(); dx += 3*VTraits<v_uint8>::vlanes(), S0 += 6*VTraits<v_uint8>::vlanes(), S1 += 6*VTraits<v_uint8>::vlanes(), D += 3*VTraits<v_uint8>::vlanes())
            {
                v_uint16 t0, t1, t2, t3, t4, t5;
                v_uint16 s0, s1, s2, s3, s4, s5;
                s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                s1 = v_add(vx_load_expand(S0 + VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + VTraits<v_uint16>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_uint16>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_uint16>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 4 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 4 * VTraits<v_uint16>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 5 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 5 * VTraits<v_uint16>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_uint16 bl, gl, rl;
#if CV_SIMD_WIDTH == 16
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#elif CV_SIMD_WIDTH == 32
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#elif CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#endif
                s0 = v_add(vx_load_expand(S0 + 6 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 6 * VTraits<v_uint16>::vlanes()));
                s1 = v_add(vx_load_expand(S0 + 7 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 7 * VTraits<v_uint16>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 8 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 8 * VTraits<v_uint16>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 9 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 9 * VTraits<v_uint16>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 10 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 10 * VTraits<v_uint16>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 11 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 11 * VTraits<v_uint16>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_uint16 bh, gh, rh;
#if CV_SIMD_WIDTH == 16
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#elif CV_SIMD_WIDTH == 32
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#elif CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#endif
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
        }
        else
        {
            CV_Assert(cn == 4);
            for ( ; dx <= w - VTraits<v_uint8>::vlanes(); dx += VTraits<v_uint8>::vlanes(), S0 += 2*VTraits<v_uint8>::vlanes(), S1 += 2*VTraits<v_uint8>::vlanes(), D += VTraits<v_uint8>::vlanes())
            {
                v_uint32 r00, r01, r10, r11;
                v_load_deinterleave((uint32_t*)S0, r00, r01);
                v_load_deinterleave((uint32_t*)S1, r10, r11);

                v_uint16 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_u8(r00), r00l, r00h);
                v_expand(v_reinterpret_as_u8(r01), r01l, r01h);
                v_expand(v_reinterpret_as_u8(r10), r10l, r10h);
                v_expand(v_reinterpret_as_u8(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(v_add(v_add(r00l, r01l), r10l), r11l), v_add(v_add(v_add(r00h, r01h), r10h), r11h)));
            }
        }

        return dx;
    }

private:
    int cn;
    int step;
};

#if CV_SIMD_WIDTH == 32 || CV_SIMD_WIDTH == 64
// Shared cn==3 wide-vector area-fast kernel for the 16u/16s twins (CV_SIMD_WIDTH 32/64): pure type
// substitution over {element-vector VT, accumulator WT, element ET}, force-inlined so each caller's codegen is unchanged.
template <typename VT, typename WT, typename ET>
static CV_ALWAYS_INLINE void resizeAreaFast_cn3_wide(int& dx, const ET* S0, const ET* S1, ET* D, int w)
{
    for ( ; dx <= w - 3*VTraits<VT>::vlanes(); dx += 3*VTraits<VT>::vlanes(), S0 += 6*VTraits<VT>::vlanes(), S1 += 6*VTraits<VT>::vlanes(), D += 3*VTraits<VT>::vlanes())
    {
        WT t0, t1, t2, t3, t4, t5;
        WT s0, s1, s2, s3, s4, s5;
        s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
        s1 = v_add(vx_load_expand(S0 + VTraits<WT>::vlanes()), vx_load_expand(S1 + VTraits<WT>::vlanes()));
        s2 = v_add(vx_load_expand(S0 + 2 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 2 * VTraits<WT>::vlanes()));
        s3 = v_add(vx_load_expand(S0 + 3 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 3 * VTraits<WT>::vlanes()));
        s4 = v_add(vx_load_expand(S0 + 4 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 4 * VTraits<WT>::vlanes()));
        s5 = v_add(vx_load_expand(S0 + 5 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 5 * VTraits<WT>::vlanes()));
        v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
        v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
        WT bl, gl, rl;
        v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
        bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
        v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
        bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#endif
        s0 = v_add(vx_load_expand(S0 + 6 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 6 * VTraits<WT>::vlanes()));
        s1 = v_add(vx_load_expand(S0 + 7 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 7 * VTraits<WT>::vlanes()));
        s2 = v_add(vx_load_expand(S0 + 8 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 8 * VTraits<WT>::vlanes()));
        s3 = v_add(vx_load_expand(S0 + 9 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 9 * VTraits<WT>::vlanes()));
        s4 = v_add(vx_load_expand(S0 + 10 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 10 * VTraits<WT>::vlanes()));
        s5 = v_add(vx_load_expand(S0 + 11 * VTraits<WT>::vlanes()), vx_load_expand(S1 + 11 * VTraits<WT>::vlanes()));
        v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
        v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
        WT bh, gh, rh;
        v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
        bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
        v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
        bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#endif
        v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
    }
}
#endif

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) :
        cn(_cn), step(_step) {}

    int operator() (const ushort* S, ushort* D, int w) const
    {
        int dx = 0;
        const ushort* S0 = (const ushort*)S;
        const ushort* S1 = (const ushort*)((const uchar*)(S) + step);

        if (cn == 1)
        {
            v_uint32 masklow = vx_setall_u32(0x0000ffff);
            for (; dx <= w - VTraits<v_uint32>::vlanes(); dx += VTraits<v_uint32>::vlanes(), S0 += VTraits<v_uint16>::vlanes(), S1 += VTraits<v_uint16>::vlanes(), D += VTraits<v_uint32>::vlanes())
            {
                v_uint32 r0 = v_reinterpret_as_u32(vx_load(S0));
                v_uint32 r1 = v_reinterpret_as_u32(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<16>(r0), v_and(r0, masklow)), v_shr<16>(r1)), v_and(r1, masklow)));
            }
        }
        else if (cn == 3)
        {
#if CV_SIMD_WIDTH == 16
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
#if CV_SSE4_1
            {
                v_uint32 r0, r1, r2, r3;
                v_expand(vx_load(S0), r0, r1);
                v_expand(vx_load(S1), r2, r3);
                r0 = v_add(r0, r2); r1 = v_add(r1, r3);
                v_rshr_pack_store<2>(D, v_add(r0, v_rotate_left<1>(r1, r0)));
            }
#else
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_load_expand(S0), v_load_expand(S0 + 3)), v_load_expand(S1)), v_load_expand(S1 + 3)));
#endif
#elif CV_SIMD_WIDTH == 32 || CV_SIMD_WIDTH == 64
            resizeAreaFast_cn3_wide<v_uint16, v_uint32>(dx, S0, S1, D, w);
#elif CV_SIMD_WIDTH >= 64
            v_uint32 masklow = vx_setall_u32(0x0000ffff);
            for ( ; dx <= w - 3*VTraits<v_uint16>::vlanes(); dx += 3*VTraits<v_uint16>::vlanes(), S0 += 6*VTraits<v_uint16>::vlanes(), S1 += 6*VTraits<v_uint16>::vlanes(), D += 3*VTraits<v_uint16>::vlanes())
            {
                v_uint16 b0, g0, r0, b1, g1, r1;
                v_load_deinterleave(S0, b0, g0, r0);
                v_load_deinterleave(S1, b1, g1, r1);
                v_uint32 bl = (v_reinterpret_as_u32(b0) >> 16) + (v_reinterpret_as_u32(b0) & masklow) + (v_reinterpret_as_u32(b1) >> 16) + (v_reinterpret_as_u32(b1) & masklow);
                v_uint32 gl = (v_reinterpret_as_u32(g0) >> 16) + (v_reinterpret_as_u32(g0) & masklow) + (v_reinterpret_as_u32(g1) >> 16) + (v_reinterpret_as_u32(g1) & masklow);
                v_uint32 rl = (v_reinterpret_as_u32(r0) >> 16) + (v_reinterpret_as_u32(r0) & masklow) + (v_reinterpret_as_u32(r1) >> 16) + (v_reinterpret_as_u32(r1) & masklow);
                v_load_deinterleave(S0 + 3*VTraits<v_uint16>::vlanes(), b0, g0, r0);
                v_load_deinterleave(S1 + 3*VTraits<v_uint16>::vlanes(), b1, g1, r1);
                v_uint32 bh = (v_reinterpret_as_u32(b0) >> 16) + (v_reinterpret_as_u32(b0) & masklow) + (v_reinterpret_as_u32(b1) >> 16) + (v_reinterpret_as_u32(b1) & masklow);
                v_uint32 gh = (v_reinterpret_as_u32(g0) >> 16) + (v_reinterpret_as_u32(g0) & masklow) + (v_reinterpret_as_u32(g1) >> 16) + (v_reinterpret_as_u32(g1) & masklow);
                v_uint32 rh = (v_reinterpret_as_u32(r0) >> 16) + (v_reinterpret_as_u32(r0) & masklow) + (v_reinterpret_as_u32(r1) >> 16) + (v_reinterpret_as_u32(r1) & masklow);
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#endif
        }
        else
        {
            CV_Assert(cn == 4);
#if CV_SIMD_WIDTH >= 64
            for ( ; dx <= w - VTraits<v_uint16>::vlanes(); dx += VTraits<v_uint16>::vlanes(), S0 += 2*VTraits<v_uint16>::vlanes(), S1 += 2*VTraits<v_uint16>::vlanes(), D += VTraits<v_uint16>::vlanes())
            {
                v_uint64 r00, r01, r10, r11;
                v_load_deinterleave((uint64_t*)S0, r00, r01);
                v_load_deinterleave((uint64_t*)S1, r10, r11);

                v_uint32 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_u16(r00), r00l, r00h);
                v_expand(v_reinterpret_as_u16(r01), r01l, r01h);
                v_expand(v_reinterpret_as_u16(r10), r10l, r10h);
                v_expand(v_reinterpret_as_u16(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(r00l, r01l, r10l, r11l), v_add(r00h, r01h, r10h, r11h)));
            }
#else
            for ( ; dx <= w - VTraits<v_uint32>::vlanes(); dx += VTraits<v_uint32>::vlanes(), S0 += VTraits<v_uint16>::vlanes(), S1 += VTraits<v_uint16>::vlanes(), D += VTraits<v_uint32>::vlanes())
            {
                v_uint32 r0, r1, r2, r3;
                v_expand(vx_load(S0), r0, r1);
                v_expand(vx_load(S1), r2, r3);
                r0 = v_add(r0, r2); r1 = v_add(r1, r3);
                v_uint32 v_d;
#if CV_SIMD_WIDTH == 16
                v_d = v_add(r0, r1);
#elif CV_SIMD_WIDTH == 32
                v_uint32 t0, t1;
                v_recombine(r0, r1, t0, t1);
                v_d = v_add(t0, t1);
#endif
                v_rshr_pack_store<2>(D, v_d);
            }
#endif
        }

        return dx;
    }

private:
    int cn;
    int step;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) :
        cn(_cn), step(_step) {}

    int operator() (const short* S, short* D, int w) const
    {
        int dx = 0;
        const short* S0 = (const short*)S;
        const short* S1 = (const short*)((const uchar*)(S) + step);

        if (cn == 1)
        {
            v_int32 masklow = vx_setall_s32(0x0000ffff);
            for (; dx <= w - VTraits<v_int32>::vlanes(); dx += VTraits<v_int32>::vlanes(), S0 += VTraits<v_int16>::vlanes(), S1 += VTraits<v_int16>::vlanes(), D += VTraits<v_int32>::vlanes())
            {
                v_int32 r0 = v_reinterpret_as_s32(vx_load(S0));
                v_int32 r1 = v_reinterpret_as_s32(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<16>(r0), v_shr<16>(v_shl<16>(v_and(r0, masklow)))), v_shr<16>(r1)), v_shr<16>(v_shl<16>(v_and(r1, masklow)))));
            }
        }
        else if (cn == 3)
        {
#if CV_SIMD_WIDTH == 16
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_load_expand(S0), v_load_expand(S0 + 3)), v_load_expand(S1)), v_load_expand(S1 + 3)));
#elif CV_SIMD_WIDTH == 32 || CV_SIMD_WIDTH == 64
            resizeAreaFast_cn3_wide<v_int16, v_int32>(dx, S0, S1, D, w);
#elif CV_SIMD_WIDTH >= 64
            for ( ; dx <= w - 3*VTraits<v_int16>::vlanes(); dx += 3*VTraits<v_int16>::vlanes(), S0 += 6*VTraits<v_int16>::vlanes(), S1 += 6*VTraits<v_int16>::vlanes(), D += 3*VTraits<v_int16>::vlanes())
            {
                v_int16 b0, g0, r0, b1, g1, r1;
                v_load_deinterleave(S0, b0, g0, r0);
                v_load_deinterleave(S1, b1, g1, r1);
                v_int32 bl = (v_reinterpret_as_s32(b0) >> 16) + ((v_reinterpret_as_s32(b0) << 16) >> 16) + (v_reinterpret_as_s32(b1) >> 16) + ((v_reinterpret_as_s32(b1) << 16) >> 16);
                v_int32 gl = (v_reinterpret_as_s32(g0) >> 16) + ((v_reinterpret_as_s32(g0) << 16) >> 16) + (v_reinterpret_as_s32(g1) >> 16) + ((v_reinterpret_as_s32(g1) << 16) >> 16);
                v_int32 rl = (v_reinterpret_as_s32(r0) >> 16) + ((v_reinterpret_as_s32(r0) << 16) >> 16) + (v_reinterpret_as_s32(r1) >> 16) + ((v_reinterpret_as_s32(r1) << 16) >> 16);
                v_load_deinterleave(S0 + 3*VTraits<v_int16>::vlanes(), b0, g0, r0);
                v_load_deinterleave(S1 + 3*VTraits<v_int16>::vlanes(), b1, g1, r1);
                v_int32 bh = (v_reinterpret_as_s32(b0) >> 16) + ((v_reinterpret_as_s32(b0) << 16) >> 16) + (v_reinterpret_as_s32(b1) >> 16) + ((v_reinterpret_as_s32(b1) << 16) >> 16);
                v_int32 gh = (v_reinterpret_as_s32(g0) >> 16) + ((v_reinterpret_as_s32(g0) << 16) >> 16) + (v_reinterpret_as_s32(g1) >> 16) + ((v_reinterpret_as_s32(g1) << 16) >> 16);
                v_int32 rh = (v_reinterpret_as_s32(r0) >> 16) + ((v_reinterpret_as_s32(r0) << 16) >> 16) + (v_reinterpret_as_s32(r1) >> 16) + ((v_reinterpret_as_s32(r1) << 16) >> 16);
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#endif
        }
        else
        {
            CV_Assert(cn == 4);
            for (; dx <= w - VTraits<v_int16>::vlanes(); dx += VTraits<v_int16>::vlanes(), S0 += 2 * VTraits<v_int16>::vlanes(), S1 += 2 * VTraits<v_int16>::vlanes(), D += VTraits<v_int16>::vlanes())
            {
#if CV_SIMD_WIDTH >= 64
                v_int64 r00, r01, r10, r11;
                v_load_deinterleave((int64_t*)S0, r00, r01);
                v_load_deinterleave((int64_t*)S1, r10, r11);

                v_int32 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_s16(r00), r00l, r00h);
                v_expand(v_reinterpret_as_s16(r01), r01l, r01h);
                v_expand(v_reinterpret_as_s16(r10), r10l, r10h);
                v_expand(v_reinterpret_as_s16(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(r00l, r01l, r10l, r11l), v_add(r00h, r01h, r10h, r11h)));
#else
                v_int32 r0, r1, r2, r3;
                r0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                r1 = v_add(vx_load_expand(S0 + VTraits<v_int32>::vlanes()), vx_load_expand(S1 + VTraits<v_int32>::vlanes()));
                r2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_int32>::vlanes()));
                r3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_int32>::vlanes()));
                v_int32 dl, dh;
#if CV_SIMD_WIDTH == 16
                dl = v_add(r0, r1); dh = v_add(r2, r3);
#elif CV_SIMD_WIDTH == 32
                v_int32 t0, t1, t2, t3;
                v_recombine(r0, r1, t0, t1); v_recombine(r2, r3, t2, t3);
                dl = v_add(t0, t1); dh = v_add(t2, t3);
#endif
                v_store(D, v_rshr_pack<2>(dl, dh));
#endif
            }
        }

        return dx;
    }

private:
    int cn;
    int step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        if (cn == 1)
        {
            v_float32 v_025 = vx_setall_f32(0.25f);
            for ( ; dx <= w - VTraits<v_float32>::vlanes(); dx += VTraits<v_float32>::vlanes(), S0 += 2*VTraits<v_float32>::vlanes(), S1 += 2*VTraits<v_float32>::vlanes(), D += VTraits<v_float32>::vlanes())
            {
                v_float32 v_row00, v_row01, v_row10, v_row11;
                v_load_deinterleave(S0, v_row00, v_row01);
                v_load_deinterleave(S1, v_row10, v_row11);
                v_store(D, v_mul(v_add(v_add(v_row00, v_row01), v_add(v_row10, v_row11)), v_025));
            }
        }
        else if (cn == 4)
        {
#if CV_SIMD_WIDTH == 16
            v_float32 v_025 = vx_setall_f32(0.25f);
            for (; dx <= w - VTraits<v_float32>::vlanes(); dx += VTraits<v_float32>::vlanes(), S0 += 2*VTraits<v_float32>::vlanes(), S1 += 2*VTraits<v_float32>::vlanes(), D += VTraits<v_float32>::vlanes())
                v_store(D, v_mul(v_add(v_add(vx_load(S0), vx_load(S0 + VTraits<v_float32>::vlanes())), v_add(vx_load(S1), vx_load(S1 + VTraits<v_float32>::vlanes()))), v_025));
#elif CV_SIMD256
            v_float32x8 v_025 = v256_setall_f32(0.25f);
            for (; dx <= w - VTraits<v_float32x8>::vlanes(); dx += VTraits<v_float32x8>::vlanes(), S0 += 2*VTraits<v_float32x8>::vlanes(), S1 += 2*VTraits<v_float32x8>::vlanes(), D += VTraits<v_float32x8>::vlanes())
            {
                v_float32x8 dst0, dst1;
                v_recombine(v_add(v256_load(S0), v256_load(S1)), v_add(v256_load(S0 + VTraits<v_float32x8>::vlanes()), v256_load(S1 + VTraits<v_float32x8>::vlanes())), dst0, dst1);
                v_store(D, v_mul(v_add(dst0, dst1), v_025));
            }
#endif
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#else

typedef ResizeAreaFastNoVec<uchar, uchar> ResizeAreaFastVec_SIMD_8u;
typedef ResizeAreaFastNoVec<ushort, ushort> ResizeAreaFastVec_SIMD_16u;
typedef ResizeAreaFastNoVec<short, short> ResizeAreaFastVec_SIMD_16s;
typedef ResizeAreaFastNoVec<float, float> ResizeAreaFastVec_SIMD_32f;

#endif

template<typename T, typename SIMDVecOp>
struct ResizeAreaFastVec
{
    ResizeAreaFastVec(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step), vecOp(_cn, _step)
    {
        fast_mode = scale_x == 2 && scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
    }

    int operator() (const T* S, T* D, int w) const
    {
        if (!fast_mode)
            return 0;

        const T* nextS = (const T*)((const uchar*)S + step);
        int dx = vecOp(S, D, w);

        if (cn == 1)
            for( ; dx < w; ++dx )
            {
                int index = dx*2;
                D[dx] = (T)((S[index] + S[index+1] + nextS[index] + nextS[index+1] + 2) >> 2);
            }
        else if (cn == 3)
            for( ; dx < w; dx += 3 )
            {
                int index = dx*2;
                D[dx] = (T)((S[index] + S[index+3] + nextS[index] + nextS[index+3] + 2) >> 2);
                D[dx+1] = (T)((S[index+1] + S[index+4] + nextS[index+1] + nextS[index+4] + 2) >> 2);
                D[dx+2] = (T)((S[index+2] + S[index+5] + nextS[index+2] + nextS[index+5] + 2) >> 2);
            }
        else
            {
                CV_Assert(cn == 4);
                for( ; dx < w; dx += 4 )
                {
                    int index = dx*2;
                    D[dx] = (T)((S[index] + S[index+4] + nextS[index] + nextS[index+4] + 2) >> 2);
                    D[dx+1] = (T)((S[index+1] + S[index+5] + nextS[index+1] + nextS[index+5] + 2) >> 2);
                    D[dx+2] = (T)((S[index+2] + S[index+6] + nextS[index+2] + nextS[index+6] + 2) >> 2);
                    D[dx+3] = (T)((S[index+3] + S[index+7] + nextS[index+3] + nextS[index+7] + 2) >> 2);
                }
            }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    bool fast_mode;
    int step;
    SIMDVecOp vecOp;
};

template <typename T, typename WT, typename VecOp>
class resizeAreaFast_Invoker :
    public ParallelLoopBody
{
public:
    resizeAreaFast_Invoker(const Mat &_src, Mat &_dst,
        int _scale_x, int _scale_y, const int* _ofs, const int* _xofs) :
        ParallelLoopBody(), src(_src), dst(_dst), scale_x(_scale_x),
        scale_y(_scale_y), ofs(_ofs), xofs(_xofs)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int cn = src.channels();
        int area = scale_x*scale_y;
        float scale = 1.f/(area);
        int dwidth1 = (ssize.width/scale_x)*cn;
        dsize.width *= cn;
        ssize.width *= cn;
        int dy, dx, k = 0;

        VecOp vop(scale_x, scale_y, src.channels(), (int)src.step/*, area_ofs*/);

        for( dy = range.start; dy < range.end; dy++ )
        {
            T* D = (T*)(dst.data + dst.step*dy);
            int sy0 = dy*scale_y;
            int w = sy0 + scale_y <= ssize.height ? dwidth1 : 0;

            if( sy0 >= ssize.height )
            {
                for( dx = 0; dx < dsize.width; dx++ )
                    D[dx] = 0;
                continue;
            }

            dx = vop(src.template ptr<T>(sy0), D, w);
            for( ; dx < w; dx++ )
            {
                const T* S = src.template ptr<T>(sy0) + xofs[dx];
                WT sum = 0;
                k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= area - 4; k += 4 )
                    sum += S[ofs[k]] + S[ofs[k+1]] + S[ofs[k+2]] + S[ofs[k+3]];
                #endif
                for( ; k < area; k++ )
                    sum += S[ofs[k]];

                D[dx] = saturate_cast<T>(sum * scale);
            }

            for( ; dx < dsize.width; dx++ )
            {
                WT sum = 0;
                int count = 0, sx0 = xofs[dx];
                if( sx0 >= ssize.width )
                    D[dx] = 0;

                for( int sy = 0; sy < scale_y; sy++ )
                {
                    if( sy0 + sy >= ssize.height )
                        break;
                    const T* S = src.template ptr<T>(sy0 + sy) + sx0;
                    for( int sx = 0; sx < scale_x*cn; sx += cn )
                    {
                        if( sx0 + sx >= ssize.width )
                            break;
                        sum += S[sx];
                        count++;
                    }
                }

                D[dx] = saturate_cast<T>((float)sum/count);
            }
        }
    }

private:
    Mat src;
    Mat dst;
    int scale_x, scale_y;
    const int *ofs, *xofs;
};

template<typename T, typename WT, typename VecOp>
static void resizeAreaFast_( const Mat& src, Mat& dst, const int* ofs, const int* xofs,
                             int scale_x, int scale_y, const Range& range )
{
    resizeAreaFast_Invoker<T, WT, VecOp> invoker(src, dst, scale_x,
        scale_y, ofs, xofs);
    invoker(range);
}

struct DecimateAlpha
{
    int si, di;
    float alpha;
};


namespace inter_area {
#if (CV_SIMD || CV_SIMD_SCALABLE)
inline void saturate_store(const float* src, uchar* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp2 = v_round(vx_load(src + 2 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp3 = v_round(vx_load(src + 3 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack(v_pack_u(tmp0, tmp1), v_pack_u(tmp2, tmp3)));
}

inline void saturate_store(const float* src, ushort* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack_u(tmp0, tmp1));
}

inline void saturate_store(const float* src, short* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack(tmp0, tmp1));
}

static inline v_float32 vx_setall(float coeff) { return vx_setall_f32(coeff); }

template <typename T>
struct VArea {};

template <>
struct VArea<float> {
    typedef v_float32 vWT;
};
#endif

#if (CV_SIMD128_64F || CV_SIMD_SCALABLE_64F)
static inline v_float64 vx_setall(double coeff) { return vx_setall_f64(coeff); }

template <>
struct VArea<double> {
    typedef v_float64 vWT;
};

#else
inline void mul(const double* buf, int width, double beta, double* sum) {
    for (int dx = 0; dx < width; ++dx) {
        sum[dx] = beta * buf[dx];
    }
}

inline void muladd(const double* buf, int width, double beta, double* sum) {
    for (int dx = 0; dx < width; ++dx) {
        sum[dx] += beta * buf[dx];
    }
}
#endif

template <typename T, typename WT>
inline void saturate_store(const WT* sum, int width, T* D) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes() * sizeof(WT) / sizeof(T);
    for (; dx + step < width; dx += step) {
        saturate_store(sum + dx, D + dx);
    }
#endif
    for (; dx < width; ++dx) {
        D[dx] = saturate_cast<T>(sum[dx]);
    }
}

// Optimization when T == WT.
template <typename WT>
inline void saturate_store(const WT* sum, int width, WT* D) {
    std::copy(sum, sum + width, D);
}

template <typename WT>
inline void mul(const WT* buf, int width, WT beta, WT* sum) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes();
    for (; dx + step < width; dx += step) {
        vx_store(sum + dx, v_mul(vx_setall(beta), vx_load(buf + dx)));
    }
#endif
    for (; dx < width; ++dx) {
        sum[dx] = beta * buf[dx];
    }
}

template <typename WT>
inline void muladd(const WT* buf, int width, WT beta, WT* sum) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes();
    for (; dx + step < width; dx += step) {
        vx_store(sum + dx, v_add(vx_load(sum + dx), v_mul(vx_setall(beta), vx_load(buf + dx))));
    }
#endif
    for (; dx < width; ++dx) {
        sum[dx] += beta * buf[dx];
    }
}

}  // namespace inter_area

template<typename T, typename WT> class ResizeArea_Invoker :
    public ParallelLoopBody
{
public:
    ResizeArea_Invoker( const Mat& _src, Mat& _dst,
                        const DecimateAlpha* _xtab, int _xtab_size,
                        const DecimateAlpha* _ytab, int _ytab_size,
                        const int* _tabofs )
    {
        src = &_src;
        dst = &_dst;
        xtab0 = _xtab;
        xtab_size0 = _xtab_size;
        ytab = _ytab;
        ytab_size = _ytab_size;
        tabofs = _tabofs;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst->size();
        int cn = dst->channels();
        dsize.width *= cn;
        AutoBuffer<WT> _buffer(dsize.width*2);
        const DecimateAlpha* xtab = xtab0;
        int xtab_size = xtab_size0;
        WT *buf = _buffer.data(), *sum = buf + dsize.width;
        int j_start = tabofs[range.start], j_end = tabofs[range.end], j, k, dx, prev_dy = ytab[j_start].di;

        for( dx = 0; dx < dsize.width; dx++ )
            sum[dx] = (WT)0;

        for( j = j_start; j < j_end; j++ )
        {
            WT beta = ytab[j].alpha;
            int dy = ytab[j].di;
            int sy = ytab[j].si;

            {
                const T* S = src->template ptr<T>(sy);
                for( dx = 0; dx < dsize.width; dx++ )
                    buf[dx] = (WT)0;

                if( cn == 1 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        buf[dxn] += S[xtab[k].si]*alpha;
                    }
                else if( cn == 2 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                    }
                else if( cn == 3 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        WT t2 = buf[dxn+2] + S[sxn+2]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1; buf[dxn+2] = t2;
                    }
                else if( cn == 4 )
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                        t0 = buf[dxn+2] + S[sxn+2]*alpha;
                        t1 = buf[dxn+3] + S[sxn+3]*alpha;
                        buf[dxn+2] = t0; buf[dxn+3] = t1;
                    }
                }
                else
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        for( int c = 0; c < cn; c++ )
                            buf[dxn + c] += S[sxn + c]*alpha;
                    }
                }
            }

            if( dy != prev_dy )
            {
                inter_area::saturate_store(sum, dsize.width, dst->template ptr<T>(prev_dy));
                inter_area::mul(buf, dsize.width, beta, sum);
                prev_dy = dy;
            }
            else
            {
                inter_area::muladd(buf, dsize.width, beta, sum);
            }
        }

        inter_area::saturate_store(sum, dsize.width, dst->template ptr<T>(prev_dy));
    }

private:
    const Mat* src;
    Mat* dst;
    const DecimateAlpha* xtab0;
    const DecimateAlpha* ytab;
    int xtab_size0, ytab_size;
    const int* tabofs;
};


template <typename T, typename WT>
static void resizeArea_( const Mat& src, Mat& dst,
                         const DecimateAlpha* xtab, int xtab_size,
                         const DecimateAlpha* ytab, int ytab_size,
                         const int* tabofs, const Range& range )
{
    ResizeArea_Invoker<T, WT> invoker(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs);
    invoker(range);
}


typedef void (*ResizeFunc)( const Mat& src, Mat& dst,
                            const int* xofs, const void* alpha,
                            const int* yofs, const void* beta,
                            int xmin, int xmax, int ksize, const Range& range );

typedef void (*ResizeAreaFastFunc)( const Mat& src, Mat& dst,
                                    const int* ofs, const int *xofs,
                                    int scale_x, int scale_y, const Range& range );

typedef void (*ResizeAreaFunc)( const Mat& src, Mat& dst,
                                const DecimateAlpha* xtab, int xtab_size,
                                const DecimateAlpha* ytab, int ytab_size,
                                const int* yofs, const Range& range);


// Shared resize-area coefficient-table math for the CPU and OpenCL callers; the
// per-entry emit (struct vs parallel arrays + ofs_tab) is picked via if constexpr.
template <bool OCL>
static int computeResizeAreaTabImpl( int ssize, int dsize, int cn, double scale,
                                     DecimateAlpha* tab, int* map_tab, float* alpha_tab, int* ofs_tab )
{
    int k = 0, dx = 0;
    for( ; dx < dsize; dx++ )
    {
        if constexpr (OCL) ofs_tab[dx] = k;

        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        auto emit = [&](int si, float alpha)
        {
            if constexpr (OCL)
            {
                map_tab[k] = si;
                alpha_tab[k] = alpha;
            }
            else
            {
                CV_Assert( k < ssize*2 );
                tab[k].di = dx * cn;
                tab[k].si = si * cn;
                tab[k].alpha = alpha;
            }
            k++;
        };

        if( sx1 - fsx1 > 1e-3 )
            emit(sx1 - 1, (float)((sx1 - fsx1) / cellWidth));

        for( int sx = sx1; sx < sx2; sx++ )
            emit(sx, float(1.0 / cellWidth));

        if( fsx2 - sx2 > 1e-3 )
            emit(sx2, (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth));
    }
    if constexpr (OCL) ofs_tab[dsize] = k;
    return k;
}

static int computeResizeAreaTab( int ssize, int dsize, int cn, double scale, DecimateAlpha* tab )
{
    return computeResizeAreaTabImpl<false>(ssize, dsize, cn, scale, tab, nullptr, nullptr, nullptr);
}

#ifdef HAVE_OPENCL
static void ocl_computeResizeAreaTabs(int ssize, int dsize, double scale, int * const map_tab,
                                      float * const alpha_tab, int * const ofs_tab)
{
    computeResizeAreaTabImpl<true>(ssize, dsize, 1, scale, nullptr, map_tab, alpha_tab, ofs_tab);
}

static bool ocl_resize( InputArray _src, OutputArray _dst, Size dsize,
                        double fx, double fy, int interpolation)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    double inv_fx = 1.0 / fx, inv_fy = 1.0 / fy;
    float inv_fxf = (float)inv_fx, inv_fyf = (float)inv_fy;
    int iscale_x = saturate_cast<int>(inv_fx), iscale_y = saturate_cast<int>(inv_fx);
    bool is_area_fast = std::abs(inv_fx - iscale_x) < DBL_EPSILON &&
        std::abs(inv_fy - iscale_y) < DBL_EPSILON;

    // in case of scale_x && scale_y is equal to 2
    // INTER_AREA (fast) also is equal to INTER_LINEAR
    if( interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 )
        /*interpolation = INTER_AREA*/CV_UNUSED(0); // INTER_AREA is slower

    if( !(cn <= 4 &&
           (interpolation == INTER_NEAREST || interpolation == INTER_LINEAR ||
            (interpolation == INTER_AREA && inv_fx >= 1 && inv_fy >= 1) )) )
        return false;

    UMat src = _src.getUMat();
    _dst.create(dsize, type);
    UMat dst = _dst.getUMat();

    Size ssize = src.size();
    ocl::Kernel k;
    size_t globalsize[] = { (size_t)dst.cols, (size_t)dst.rows };

    ocl::Image2D srcImage;

    // See if this could be done with a sampler.  We stick with integer
    // datatypes because the observed error is low.
    bool useSampler = (interpolation == INTER_LINEAR && ocl::Device::getDefault().imageSupport() &&
                       ocl::Image2D::canCreateAlias(src) && depth <= 4 &&
                       ocl::Image2D::isFormatSupported(depth, cn, true) &&
                       src.offset==0);
    if (useSampler)
    {
        int wdepth = std::max(depth, CV_32S);
        char buf[2][50];
        cv::String compileOpts = format("-D USE_SAMPLER -D SRC_DEPTH=%d -D T=%s -D T1=%s "
                        "-D CONVERT_TO_DT=%s -D CN=%d",
                        depth, ocl::typeToStr(type), ocl::typeToStr(depth),
                        ocl::convertTypeStr(wdepth, depth, cn, buf[1], sizeof(buf[1])),
                        cn);
        k.create("resizeSampler", ocl::imgproc::resize_oclsrc, compileOpts);

        if (k.empty())
            useSampler = false;
        else
        {
            // Convert the input into an OpenCL image type, using normalized channel data types
            // and aliasing the UMat.
            srcImage = ocl::Image2D(src, true, true);
            k.args(srcImage, ocl::KernelArg::WriteOnly(dst),
                   (float)inv_fx, (float)inv_fy);
        }
    }

    if (interpolation == INTER_LINEAR && !useSampler)
    {
        char buf[2][50];

        // integer path is slower because of CPU part, so it's disabled
        if (depth == CV_8U && ((void)0, 0))
        {
            AutoBuffer<uchar> _buffer((dsize.width + dsize.height)*(sizeof(int) + sizeof(short)*2));
            int* xofs = (int*)_buffer.data(), * yofs = xofs + dsize.width;
            short* ialpha = (short*)(yofs + dsize.height), * ibeta = ialpha + dsize.width*2;
            float fxx, fyy;
            int sx, sy;

            for (int dx = 0; dx < dsize.width; dx++)
            {
                fxx = (float)((dx+0.5)*inv_fx - 0.5);
                sx = cvFloor(fxx);
                fxx -= sx;

                if (sx < 0)
                    fxx = 0, sx = 0;

                if (sx >= ssize.width-1)
                    fxx = 0, sx = ssize.width-1;

                xofs[dx] = sx;
                ialpha[dx*2 + 0] = saturate_cast<short>((1.f - fxx) * INTER_RESIZE_COEF_SCALE);
                ialpha[dx*2 + 1] = saturate_cast<short>(fxx         * INTER_RESIZE_COEF_SCALE);
            }

            for (int dy = 0; dy < dsize.height; dy++)
            {
                fyy = (float)((dy+0.5)*inv_fy - 0.5);
                sy = cvFloor(fyy);
                fyy -= sy;

                yofs[dy] = sy;
                ibeta[dy*2 + 0] = saturate_cast<short>((1.f - fyy) * INTER_RESIZE_COEF_SCALE);
                ibeta[dy*2 + 1] = saturate_cast<short>(fyy         * INTER_RESIZE_COEF_SCALE);
            }

            int wdepth = std::max(depth, CV_32S), wtype = CV_MAKETYPE(wdepth, cn);
            UMat coeffs;
            Mat(1, static_cast<int>(_buffer.size()), CV_8UC1, _buffer.data()).copyTo(coeffs);

            k.create("resizeLN", ocl::imgproc::resize_oclsrc,
                     format("-D INTER_LINEAR_INTEGER -D SRC_DEPTH=%d -D T=%s -D T1=%s "
                            "-D WT=%s -D CONVERT_TO_WT=%s -D CONVERT_TO_DT=%s -D CN=%d "
                            "-D INTER_RESIZE_COEF_BITS=%d",
                            depth, ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                            ocl::convertTypeStr(depth, wdepth, cn, buf[0], sizeof(buf[0])),
                            ocl::convertTypeStr(wdepth, depth, cn, buf[1], sizeof(buf[1])),
                            cn, INTER_RESIZE_COEF_BITS));
            if (k.empty())
                return false;

            k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
                   ocl::KernelArg::PtrReadOnly(coeffs));
        }
        else
        {
            int wdepth = depth <= CV_8S ? CV_32S : std::max(depth, CV_32F);
            int wtype = CV_MAKETYPE(wdepth, cn);
            k.create("resizeLN", ocl::imgproc::resize_oclsrc,
                     format("-D INTER_LINEAR -D SRC_DEPTH=%d -D T=%s -D T1=%s "
                            "-D WT=%s -D CONVERT_TO_WT=%s -D CONVERT_TO_DT=%s -D CN=%d "
                            "-D INTER_RESIZE_COEF_BITS=%d",
                            depth, ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                            ocl::convertTypeStr(depth, wdepth, cn, buf[0], sizeof(buf[0])),
                            ocl::convertTypeStr(wdepth, depth, cn, buf[1], sizeof(buf[1])),
                            cn, INTER_RESIZE_COEF_BITS));
            if (k.empty())
                return false;

            k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
                   (float)inv_fx, (float)inv_fy);
        }
    }
    else if (interpolation == INTER_NEAREST)
    {
        k.create("resizeNN", ocl::imgproc::resize_oclsrc,
                 format("-D INTER_NEAREST -D T=%s -D T1=%s -D CN=%d",
                        ocl::vecopTypeToStr(type), ocl::vecopTypeToStr(depth), cn));
        if (k.empty())
            return false;

        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
               (float)inv_fx, (float)inv_fy);
    }
    else if (interpolation == INTER_AREA)
    {
        int wdepth = std::max(depth, is_area_fast ? CV_32S : CV_32F);
        int wtype = CV_MAKE_TYPE(wdepth, cn);

        char cvt[2][50];
        String buildOption = format("-D INTER_AREA -D T=%s -D T1=%s -D WTV=%s -D CONVERT_TO_WTV=%s -D CN=%d",
                                    ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                                    ocl::convertTypeStr(depth, wdepth, cn, cvt[0], sizeof(cvt[0])), cn);

        UMat alphaOcl, tabofsOcl, mapOcl;
        UMat dmap, smap;

        if (is_area_fast)
        {
            int wdepth2 = std::max(CV_32F, depth), wtype2 = CV_MAKE_TYPE(wdepth2, cn);
            buildOption = buildOption + format(" -D CONVERT_TO_T=%s -D WT2V=%s -D CONVERT_TO_WT2V=%s -D INTER_AREA_FAST"
                                                " -D XSCALE=%d -D YSCALE=%d -D SCALE=%ff",
                                                ocl::convertTypeStr(wdepth2, depth, cn, cvt[0], sizeof(cvt[0])),
                                                ocl::typeToStr(wtype2), ocl::convertTypeStr(wdepth, wdepth2, cn, cvt[1], sizeof(cvt[1])),
                                    iscale_x, iscale_y, 1.0f / (iscale_x * iscale_y));

            k.create("resizeAREA_FAST", ocl::imgproc::resize_oclsrc, buildOption);
            if (k.empty())
                return false;
        }
        else
        {
            buildOption = buildOption + format(" -D CONVERT_TO_T=%s", ocl::convertTypeStr(wdepth, depth, cn, cvt[0], sizeof(cvt[0])));
            k.create("resizeAREA", ocl::imgproc::resize_oclsrc, buildOption);
            if (k.empty())
                return false;

            int xytab_size = (ssize.width + ssize.height) << 1;
            int tabofs_size = dsize.height + dsize.width + 2;

            AutoBuffer<int> _xymap_tab(xytab_size), _xyofs_tab(tabofs_size);
            AutoBuffer<float> _xyalpha_tab(xytab_size);
            int * xmap_tab = _xymap_tab.data(), * ymap_tab = _xymap_tab.data() + (ssize.width << 1);
            float * xalpha_tab = _xyalpha_tab.data(), * yalpha_tab = _xyalpha_tab.data() + (ssize.width << 1);
            int * xofs_tab = _xyofs_tab.data(), * yofs_tab = _xyofs_tab.data() + dsize.width + 1;

            ocl_computeResizeAreaTabs(ssize.width, dsize.width, inv_fx, xmap_tab, xalpha_tab, xofs_tab);
            ocl_computeResizeAreaTabs(ssize.height, dsize.height, inv_fy, ymap_tab, yalpha_tab, yofs_tab);

            // loading precomputed arrays to GPU
            Mat(1, xytab_size, CV_32FC1, _xyalpha_tab.data()).copyTo(alphaOcl);
            Mat(1, xytab_size, CV_32SC1, _xymap_tab.data()).copyTo(mapOcl);
            Mat(1, tabofs_size, CV_32SC1, _xyofs_tab.data()).copyTo(tabofsOcl);
        }

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnly(src), dstarg = ocl::KernelArg::WriteOnly(dst);

        if (is_area_fast)
            k.args(srcarg, dstarg);
        else
            k.args(srcarg, dstarg, inv_fxf, inv_fyf, ocl::KernelArg::PtrReadOnly(tabofsOcl),
                   ocl::KernelArg::PtrReadOnly(mapOcl), ocl::KernelArg::PtrReadOnly(alphaOcl));

        return k.run(2, globalsize, NULL, false);
    }

    return k.run(2, globalsize, 0, false);
}

#endif

//==================================================================================================

namespace {

// ---- ResizeParams: validation, output geometry, and the dst->src coordinate map ----

// The flag the kernel tables are indexed by. As in classic cv::resize, INTER_LINEAR_EXACT on a
// float degrades to INTER_LINEAR, which has no fixed-point kernel to be exact about.
int resolveInterpolation(const ResizeParams& params, int depth)
{
    if (params.interpolation == INTER_LINEAR_EXACT && (depth == CV_32F || depth == CV_64F))
        return INTER_LINEAR;
    return params.interpolation;
}

// Rejects what no table builder covers. Called once per resize(), before any plan is built.
void checkResizeParams(const ResizeParams& params)
{
    if (params.antialias)
        CV_Error(Error::StsNotImplemented, "ResizeParams::antialias is not implemented yet");
    // The bit-exact kernels have no coordinate-mode parameter, so they only serve PIXEL_CENTER.
    if (params.coordMode != ResizeCoord::PIXEL_CENTER &&
        params.interpolation != INTER_NEAREST && params.interpolation != INTER_LINEAR &&
        params.interpolation != INTER_CUBIC && params.interpolation != INTER_LANCZOS4)
        CV_Error(Error::StsNotImplemented,
                 "a ResizeParams::coordMode other than ResizeCoord::PIXEL_CENTER needs "
                 "INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_LANCZOS4 "
                 "(the _EXACT flags included)");
}

// Follows classic cv::resize's rule for deriving the output size from fx/fy.
Size resolveDstSize(Size ssize, const ResizeParams& params)
{
    CV_Assert(!ssize.empty());
    if (!params.dsize.empty())
        return params.dsize;
    CV_Assert(params.fx > 0 && params.fy > 0);
    Size dsize(saturate_cast<int>(ssize.width*params.fx), saturate_cast<int>(ssize.height*params.fy));
    CV_Assert(!dsize.empty());
    return dsize;
}

// dst/src per axis. A given dsize overrides fx/fy, as in classic cv::resize, except for the ONNX
// modes: their dsize is usually floor(size*scale), which does not divide back to the true scale.
void resolveScales(Size ssize, Size dsize, const ResizeParams& params,
                   double& inv_scale_x, double& inv_scale_y)
{
    const bool fromScale = params.dsize.empty() ||
        (params.coordMode != ResizeCoord::PIXEL_CENTER && params.fx > 0 && params.fy > 0);
    inv_scale_x = fromScale ? params.fx : (double)dsize.width /ssize.width;
    inv_scale_y = fromScale ? params.fy : (double)dsize.height/ssize.height;
    CV_Assert(inv_scale_x > 0 && inv_scale_y > 0);
}

// dst -> src along one axis. ResizeCoord collapses into this at build time, so only the table
// builders ever see the mode and no kernel does -- which is why every mode costs the same.
struct AxisMap
{
    ResizeCoord mode;
    double scale;       // source units per destination unit
    double symOffset;   // HALF_PIXEL_SYMMETRIC's recentering shift
    bool degenerate;    // PYTORCH_HALF_PIXEL on a one-sample output axis

    double operator ()(int dst) const
    {
        switch (mode)
        {
        case ResizeCoord::ASYMMETRIC:
        case ResizeCoord::ALIGN_CORNERS:        // scale already rewritten by makeAxisMap()
            return dst*scale;
        case ResizeCoord::TF_HALF_PIXEL_FOR_NN:
            return (dst + 0.5)*scale;
        case ResizeCoord::HALF_PIXEL_SYMMETRIC:
            return symOffset + (dst + 0.5)*scale - 0.5;
        case ResizeCoord::PYTORCH_HALF_PIXEL:
            return degenerate ? 0.0 : (dst + 0.5)*scale - 0.5;
        default:                                // PIXEL_CENTER and HALF_PIXEL agree
            return (dst + 0.5)*scale - 0.5;
        }
    }

    // Every mode above is affine, so src = dst*a + b too -- a form that vectorizes where
    // operator() does not. Only INTER_NEAREST uses it; the separable kernels need operator()'s
    // exact expression. Its PIXEL_CENTER becomes ASYMMETRIC, where b is 0.0 and this is exact.
    void affine(double& a, double& b) const
    {
        a = degenerate ? 0.0 : scale;
        switch (mode)
        {
        case ResizeCoord::ASYMMETRIC:
        case ResizeCoord::ALIGN_CORNERS:        b = 0.0;                                  break;
        case ResizeCoord::TF_HALF_PIXEL_FOR_NN: b = 0.5*scale;                            break;
        case ResizeCoord::HALF_PIXEL_SYMMETRIC: b = symOffset + 0.5*scale - 0.5;          break;
        case ResizeCoord::PYTORCH_HALF_PIXEL:   b = degenerate ? 0.0 : 0.5*scale - 0.5;   break;
        default:                                b = 0.5*scale - 0.5;                      break;
        }
    }
};

// trueOutLen is the output length before dsize was floored. ALIGN_CORNERS divides by it, so only
// an explicit fx/fy pins that mode down.
AxisMap makeAxisMap(ResizeCoord mode, double scale, int inLen, int outLen, double trueOutLen)
{
    AxisMap m;
    m.mode = mode;
    m.scale = scale;
    m.symOffset = 0;
    m.degenerate = false;
    if (mode == ResizeCoord::ALIGN_CORNERS)
        m.scale = trueOutLen > 1 ? (inLen - 1)/(trueOutLen - 1) : 0.0;
    else if (mode == ResizeCoord::HALF_PIXEL_SYMMETRIC)
        m.symOffset = inLen*0.5 - outLen*scale*0.5;
    else if (mode == ResizeCoord::PYTORCH_HALF_PIXEL)
        m.degenerate = outLen <= 1;
    return m;
}

// Fills one axis of the INTER_NEAREST source-index table; mul turns an index into the byte offset
// the x axis needs, or is 1 for y. ONNX nearest_mode is resolved before the loop, not per entry:
// this table is built serially while the kernel that reads it runs on every thread.
void buildNearestTab(int* tab, int outLen, int inLen, const AxisMap& map,
                     ResizeNearest mode, int mul)
{
    double a, b;
    map.affine(a, b);
    const int last = inLen - 1;

    // Clamping in double and then truncating vectorizes where cvFloor/cvCeil do not, and gives the
    // same answer: over [0, last] a truncating cast is the floor, outside it the clamp decides.
    // Clamping first also keeps the cast in range.
    const double lastd = last;
    switch (mode)
    {
    case ResizeNearest::FLOOR:
        for (int i = 0; i < outLen; i++)
            tab[i] = (int)std::min(std::max(i*a + b, 0.0), lastd)*mul;
        break;
    case ResizeNearest::CEIL:
        for (int i = 0; i < outLen; i++)
            tab[i] = (int)std::ceil(std::min(std::max(i*a + b, 0.0), lastd))*mul;
        break;
    default:
        // The two round-to-nearest rules differ only in which way an exact .5 goes; the epsilon
        // is what counts as exact, matching the ONNX Resize layer.
        {
            const int tie = mode == ResizeNearest::ROUND_PREFER_CEIL ? 1 : 0;
            for (int i = 0; i < outLen; i++)
            {
                const double src = i*a + b;
                const int f = cvFloor(src);
                const int idx = std::abs(src - f - 0.5) <= 1e-6 ? f + tie : cvRound(src);
                tab[i] = std::min(std::max(idx, 0), last)*mul;
            }
        }
        break;
    }
}

// ONNX exclude_outside: zero the taps that fall off the edge and renormalise the rest. The taps
// span [first, first+ksize) and the kernels clamp them to the edge, so a zeroed weight is inert.
void excludeOutsideTaps(float* coeffs, int ksize, int first, int len)
{
    float sum = 0.f;
    for (int k = 0; k < ksize; k++)
    {
        if ((unsigned)(first + k) >= (unsigned)len)
            coeffs[k] = 0.f;
        sum += coeffs[k];
    }
    if (sum != 0.f)
        for (int k = 0; k < ksize; k++)
            coeffs[k] /= sum;
}

// Geometry-only half of a resize: kernel choice plus tables; run() fills any row range.
class ResizePlan
{
public:
    ResizePlan() : type(0), depth(0), cn(0), srcStep(0), kind(KIND_NONE), func(0), areaFastFunc(0),
                   areaFunc(0), beRun(0), xofs(0), yofs(0), alpha(0), beta(0),
                   xmin(0), xmax(0), ksize(0), nnXofs(0), nnYofs(0), nnBeX(0), nnBeY(0),
                   afOfs(0), afXofs(0), iscale_x(0), iscale_y(0), xtab(0), ytab(0),
                   xtab_size(0), ytab_size(0), tabofs(0) {}

    // src_step is part of the geometry: the INTER_AREA fast kernel bakes it into offsets.
    void build(int src_type, Size _ssize, size_t _src_step, Size _dsize, const ResizeParams& params);

    // Row copy, matching the shortcut cv::resize takes when dsize equals the source size.
    void buildCopy(int src_type, Size size)
    {
        type = src_type; depth = CV_MAT_DEPTH(src_type); cn = CV_MAT_CN(src_type);
        ssize = dsize = size;
        kind = KIND_COPY;
    }

    // Fills dst rows [rows.start,rows.end) serially. runResize() owns the only parallel loop.
    void run(const uchar* src_data, size_t src_step,
             uchar* dst_data, size_t dst_step, const Range& rows) const;

private:
    enum Kind
    {
        KIND_NONE = 0,
        KIND_COPY,
        KIND_NEAREST,
        KIND_NEAREST_EXACT,
        KIND_LINEAR_EXACT,
        KIND_AREA_FAST,
        KIND_AREA,
        KIND_GENERIC
    };

    int type, depth, cn;
    Size ssize, dsize;
    size_t srcStep;
    Kind kind;

    // KIND_GENERIC: separable INTER_LINEAR/INTER_CUBIC/INTER_LANCZOS4 (and INTER_AREA upscaling).
    ResizeFunc func;
    ResizeAreaFastFunc areaFastFunc;
    ResizeAreaFunc areaFunc;
    be_run_func beRun;

    AutoBuffer<uchar> tabBuf;
    int *xofs, *yofs;
    void *alpha, *beta;
    int xmin, xmax, ksize;

    // KIND_NEAREST: one source index per output column and per output row
    AutoBuffer<int> nnBuf;
    int *nnXofs, *nnYofs;

    // KIND_NEAREST_EXACT
    cv::utils::BufferArea nnBeArea;
    int *nnBeX, *nnBeY;

    // KIND_LINEAR_EXACT
    AutoBuffer<uchar> beBuf;
    BitExactTabs beTabs;

    // KIND_AREA_FAST
    AutoBuffer<int> afBuf;
    int *afOfs, *afXofs;
    int iscale_x, iscale_y;

    // KIND_AREA
    AutoBuffer<DecimateAlpha> areaBuf;
    AutoBuffer<int> areaTabofs;
    DecimateAlpha *xtab, *ytab;
    int xtab_size, ytab_size;
    int* tabofs;
};

void ResizePlan::build(int src_type, Size _ssize, size_t _src_step, Size _dsize,
                       const ResizeParams& params)
{
    type = src_type;
    depth = CV_MAT_DEPTH(src_type);
    cn = CV_MAT_CN(src_type);
    ssize = _ssize;
    srcStep = _src_step;
    dsize = _dsize;

    static ResizeFunc linear_tab[CV_DEPTH_MAX] =
    {
        resizeGeneric_<
            HResizeLinear<uchar, int, short,
                INTER_RESIZE_COEF_SCALE,
                HResizeLinearVec_8u32s>,
            VResizeLinear<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeLinearVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeLinear<ushort, float, float, 1,
                HResizeLinearVec_16u32f>,
            VResizeLinear<ushort, float, float, Cast<float, ushort>,
                VResizeLinearVec_32f16u> >,
        resizeGeneric_<
            HResizeLinear<short, float, float, 1,
                HResizeLinearVec_16s32f>,
            VResizeLinear<short, float, float, Cast<float, short>,
                VResizeLinearVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeLinear<float, float, float, 1,
                HResizeLinearVec_32f>,
            VResizeLinear<float, float, float, Cast<float, float>,
                VResizeLinearVec_32f> >,
        resizeGeneric_<
            HResizeLinear<double, double, float, 1,
                HResizeNoVec>,
            VResizeLinear<double, double, float, Cast<double, double>,
                VResizeNoVec> >,
        0
    };

    static ResizeFunc cubic_tab[CV_DEPTH_MAX] =
    {
        resizeGeneric_<
            HResizeCubic<uchar, int, short>,
            VResizeCubic<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeCubicVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeCubic<ushort, float, float>,
            VResizeCubic<ushort, float, float, Cast<float, ushort>,
            VResizeCubicVec_32f16u> >,
        resizeGeneric_<
            HResizeCubic<short, float, float>,
            VResizeCubic<short, float, float, Cast<float, short>,
            VResizeCubicVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeCubic<float, float, float>,
            VResizeCubic<float, float, float, Cast<float, float>,
            VResizeCubicVec_32f> >,
        resizeGeneric_<
            HResizeCubic<double, double, float>,
            VResizeCubic<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeFunc lanczos4_tab[CV_DEPTH_MAX] =
    {
        resizeGeneric_<HResizeLanczos4<uchar, int, short>,
            VResizeLanczos4<uchar, int, short,
            FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
            VResizeNoVec> >,
        0,
        resizeGeneric_<HResizeLanczos4<ushort, float, float>,
            VResizeLanczos4<ushort, float, float, Cast<float, ushort>,
            VResizeLanczos4Vec_32f16u> >,
        resizeGeneric_<HResizeLanczos4<short, float, float>,
            VResizeLanczos4<short, float, float, Cast<float, short>,
            VResizeLanczos4Vec_32f16s> >,
        0,
        resizeGeneric_<HResizeLanczos4<float, float, float>,
            VResizeLanczos4<float, float, float, Cast<float, float>,
            VResizeLanczos4Vec_32f> >,
        resizeGeneric_<HResizeLanczos4<double, double, float>,
            VResizeLanczos4<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeAreaFastFunc areafast_tab[CV_DEPTH_MAX] =
    {
        resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar, ResizeAreaFastVec_SIMD_8u> >,
        0,
        resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort, ResizeAreaFastVec_SIMD_16u> >,
        resizeAreaFast_<short, float, ResizeAreaFastVec<short, ResizeAreaFastVec_SIMD_16s> >,
        0,
        resizeAreaFast_<float, float, ResizeAreaFastVec_SIMD_32f>,
        resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
        0
    };

    static ResizeAreaFunc area_tab[CV_DEPTH_MAX] =
    {
        resizeArea_<uchar, float>, 0, resizeArea_<ushort, float>,
        resizeArea_<short, float>, 0, resizeArea_<float, float>,
        resizeArea_<double, double>, 0
    };

    static be_build_func linear_exact_build_tab[CV_DEPTH_MAX] =
    {
        resize_bitExact_build<uchar, interpolationLinear<uchar> >,
        resize_bitExact_build<schar, interpolationLinear<schar> >,
        resize_bitExact_build<ushort, interpolationLinear<ushort> >,
        resize_bitExact_build<short, interpolationLinear<short> >,
        resize_bitExact_build<int, interpolationLinear<int> >,
        0,
        0,
        0
    };

    static be_run_func linear_exact_run_tab[CV_DEPTH_MAX] =
    {
        resize_bitExact_run<uchar, interpolationLinear<uchar> >,
        resize_bitExact_run<schar, interpolationLinear<schar> >,
        resize_bitExact_run<ushort, interpolationLinear<ushort> >,
        resize_bitExact_run<short, interpolationLinear<short> >,
        resize_bitExact_run<int, interpolationLinear<int> >,
        0,
        0,
        0
    };

    double inv_scale_x, inv_scale_y;
    resolveScales(ssize, dsize, params, inv_scale_x, inv_scale_y);
    int interpolation = resolveInterpolation(params, depth);

    const bool classicCoord = params.coordMode == ResizeCoord::PIXEL_CENTER;
    const double trueDstW = !classicCoord && params.fx > 0 ? ssize.width *params.fx : (double)dsize.width;
    const double trueDstH = !classicCoord && params.fy > 0 ? ssize.height*params.fy : (double)dsize.height;

    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;

    int iscale_x_ = saturate_cast<int>(scale_x);
    int iscale_y_ = saturate_cast<int>(scale_y);

    bool is_area_fast = std::abs(scale_x - iscale_x_) < DBL_EPSILON &&
            std::abs(scale_y - iscale_y_) < DBL_EPSILON;

    if (interpolation == INTER_LINEAR_EXACT)
    {
        // in case of inv_scale_x && inv_scale_y is equal to 0.5
        // INTER_AREA (fast) is equal to bit exact INTER_LINEAR
        if (is_area_fast && iscale_x_ == 2 && iscale_y_ == 2 && cn != 2)//Area resize implementation for 2-channel images isn't bit-exact
            interpolation = INTER_AREA;
        else
        {
            be_build_func build_func = linear_exact_build_tab[depth];
            beRun = linear_exact_run_tab[depth];
            CV_Assert(build_func != 0 && beRun != 0);
            build_func(ssize.width, ssize.height, dsize.width, dsize.height, cn,
                       inv_scale_x, inv_scale_y, beBuf, beTabs);
            kind = KIND_LINEAR_EXACT;
            return;
        }
    }

    if( interpolation == INTER_NEAREST )
    {
        // Classic INTER_NEAREST samples at dst*scale and rounds down, which is exactly ONNX's
        // asymmetric coordinate mode with nearest_mode=floor.
        const ResizeCoord nnCoord = classicCoord ? ResizeCoord::ASYMMETRIC : params.coordMode;
        const ResizeNearest nnRound = classicCoord ? ResizeNearest::FLOOR : params.nearestMode;
        const AxisMap mapX = makeAxisMap(nnCoord, scale_x, ssize.width,  dsize.width,  trueDstW);
        const AxisMap mapY = makeAxisMap(nnCoord, scale_y, ssize.height, dsize.height, trueDstH);
        const int pix_size = (int)CV_ELEM_SIZE(type);

        nnBuf.allocate(dsize.width + dsize.height);
        nnXofs = nnBuf.data();
        nnYofs = nnXofs + dsize.width;
        buildNearestTab(nnXofs, dsize.width,  ssize.width,  mapX, nnRound, pix_size);
        buildNearestTab(nnYofs, dsize.height, ssize.height, mapY, nnRound, 1);
        kind = KIND_NEAREST;
        return;
    }

    if( interpolation == INTER_NEAREST_EXACT )
    {
        nnBeArea.allocate(nnBeX, dsize.width, CV_SIMD_WIDTH);
        nnBeArea.allocate(nnBeY, dsize.height, CV_SIMD_WIDTH);
        nnBeArea.commit();
        resizeNN_bitexact_tab(ssize.width, dsize.width, nnBeX);
        resizeNN_bitexact_tab(ssize.height, dsize.height, nnBeY);
        kind = KIND_NEAREST_EXACT;
        return;
    }

    int k, sx, sy, dx, dy;

    {
        // in case of scale_x && scale_y is equal to 2
        // INTER_AREA (fast) also is equal to INTER_LINEAR
        if( interpolation == INTER_LINEAR && is_area_fast && iscale_x_ == 2 && iscale_y_ == 2 )
            interpolation = INTER_AREA;

        // true "area" interpolation is only implemented for the case (scale_x >= 1 && scale_y >= 1).
        // In other cases it is emulated using some variant of bilinear interpolation
        if( interpolation == INTER_AREA && scale_x >= 1 && scale_y >= 1 )
        {
            if( is_area_fast )
            {
                int area = iscale_x_*iscale_y_;
                afBuf.allocate(area + dsize.width*cn);
                afOfs = afBuf.data();
                afXofs = afOfs + area;
                areaFastFunc = areafast_tab[depth];
                CV_Assert( areaFastFunc != 0 );

                size_t srcstep = srcStep / CV_ELEM_SIZE1(type);
                for( sy = 0, k = 0; sy < iscale_y_; sy++ )
                    for( sx = 0; sx < iscale_x_; sx++ )
                        afOfs[k++] = (int)(sy*srcstep + sx*cn);

                for( dx = 0; dx < dsize.width; dx++ )
                {
                    int j = dx * cn;
                    sx = iscale_x_ * j;
                    for( k = 0; k < cn; k++ )
                        afXofs[j + k] = sx + k;
                }

                iscale_x = iscale_x_;
                iscale_y = iscale_y_;
                kind = KIND_AREA_FAST;
                return;
            }

            areaFunc = area_tab[depth];
            CV_Assert( areaFunc != 0 && cn <= 4 );

            areaBuf.allocate((ssize.width + ssize.height)*2);
            xtab = areaBuf.data();
            ytab = xtab + ssize.width*2;

            xtab_size = computeResizeAreaTab(ssize.width, dsize.width, cn, scale_x, xtab);
            ytab_size = computeResizeAreaTab(ssize.height, dsize.height, 1, scale_y, ytab);

            areaTabofs.allocate(dsize.height + 1);
            tabofs = areaTabofs.data();
            for( k = 0, dy = 0; k < ytab_size; k++ )
            {
                if( k == 0 || ytab[k].di != ytab[k-1].di )
                {
                    CV_Assert( ytab[k].di == dy );
                    tabofs[dy++] = k;
                }
            }
            tabofs[dy] = ytab_size;
            kind = KIND_AREA;
            return;
        }
    }

    xmin = 0;
    xmax = dsize.width;
    int width = dsize.width*cn;
    bool area_mode = interpolation == INTER_AREA;
    bool fixpt = depth == CV_8U;
    float fx, fy;
    // INTER_AREA keeps its own decimation geometry; every other kernel reads its coordinates here.
    const AxisMap mapX = makeAxisMap(params.coordMode, scale_x, ssize.width,  dsize.width,  trueDstW);
    const AxisMap mapY = makeAxisMap(params.coordMode, scale_y, ssize.height, dsize.height, trueDstH);
    if( interpolation == INTER_CUBIC )
        ksize = 4, func = cubic_tab[depth];
    else if( interpolation == INTER_LANCZOS4 )
        ksize = 8, func = lanczos4_tab[depth];
    else if( interpolation == INTER_LINEAR || interpolation == INTER_AREA )
        ksize = 2, func = linear_tab[depth];
    else
        CV_Error( cv::Error::StsBadArg, "Unknown interpolation method" );
    int ksize2 = ksize/2;
    // With two taps, dropping the outside one and renormalising is edge replication, which the
    // kernels already do; only the wider kernels see a difference.
    const bool exclude = params.excludeOutside && ksize > 2;
    const float cubicA = params.cubicCoeffA;

    CV_Assert( func != 0 );

    tabBuf.allocate((width + dsize.height)*(sizeof(int) + sizeof(float)*ksize));
    xofs = (int*)tabBuf.data();
    yofs = xofs + width;
    float* alphaf = (float*)(yofs + dsize.height);
    short* ialpha = (short*)alphaf;
    float* betaf = alphaf + width*ksize;
    short* ibeta = ialpha + width*ksize;
    float cbuf[MAX_ESIZE] = {0};

    for( dx = 0; dx < dsize.width; dx++ )
    {
        if( !area_mode )
        {
            fx = (float)mapX(dx);
            sx = cvFloor(fx);
            fx -= sx;
        }
        else
        {
            sx = cvFloor(dx*scale_x);
            fx = (float)((dx+1) - (sx+1)*inv_scale_x);
            fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
        }

        if( sx < ksize2-1 )
        {
            xmin = dx+1;
            if( sx < 0 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = 0;
        }

        if( sx + ksize2 >= ssize.width )
        {
            xmax = std::min( xmax, dx );
            if( sx >= ssize.width-1 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = ssize.width-1;
        }

        const int sx0 = sx;
        for( k = 0, sx *= cn; k < cn; k++ )
            xofs[dx*cn + k] = sx + k;

        if( interpolation == INTER_CUBIC )
            interpolateCubic( fx, cbuf, cubicA );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fx, cbuf );
        else
        {
            cbuf[0] = 1.f - fx;
            cbuf[1] = fx;
        }
        if( exclude )
            excludeOutsideTaps( cbuf, ksize, sx0 - ksize2 + 1, ssize.width );
        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
            for( ; k < cn*ksize; k++ )
                ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                alphaf[dx*cn*ksize + k] = cbuf[k];
            for( ; k < cn*ksize; k++ )
                alphaf[dx*cn*ksize + k] = alphaf[dx*cn*ksize + k - ksize];
        }
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        if( !area_mode )
        {
            fy = (float)mapY(dy);
            sy = cvFloor(fy);
            fy -= sy;
        }
        else
        {
            sy = cvFloor(dy*scale_y);
            fy = (float)((dy+1) - (sy+1)*inv_scale_y);
            fy = fy <= 0 ? 0.f : fy - cvFloor(fy);
        }

        yofs[dy] = sy;
        if( interpolation == INTER_CUBIC )
            interpolateCubic( fy, cbuf, cubicA );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fy, cbuf );
        else
        {
            cbuf[0] = 1.f - fy;
            cbuf[1] = fy;
        }
        if( exclude )
            excludeOutsideTaps( cbuf, ksize, sy - ksize2 + 1, ssize.height );

        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                betaf[dy*ksize + k] = cbuf[k];
        }
    }

    alpha = fixpt ? (void*)ialpha : (void*)alphaf;
    beta  = fixpt ? (void*)ibeta  : (void*)betaf;
    kind = KIND_GENERIC;
}

void ResizePlan::run(const uchar* src_data, size_t src_step,
                     uchar* dst_data, size_t dst_step, const Range& rows) const
{
    if (rows.start >= rows.end)
        return;
    CV_DbgAssert(kind != KIND_AREA_FAST || src_step == srcStep);

    Mat src(ssize, type, const_cast<uchar*>(src_data), src_step);
    Mat dst(dsize, type, dst_data, dst_step);

    switch (kind)
    {
    case KIND_COPY:
        src.rowRange(rows).copyTo(dst.rowRange(rows));
        break;
    case KIND_NEAREST:
        resizeNN(src, dst, nnXofs, nnYofs, rows);
        break;
    case KIND_NEAREST_EXACT:
        resizeNN_bitexact(src, dst, nnBeX, nnBeY, rows);
        break;
    case KIND_LINEAR_EXACT:
        beRun(src_data, src_step, ssize.width, ssize.height,
              dst_data, dst_step, dsize.width, dsize.height, cn, beTabs, rows);
        break;
    case KIND_AREA_FAST:
        areaFastFunc(src, dst, afOfs, afXofs, iscale_x, iscale_y, rows);
        break;
    case KIND_AREA:
        areaFunc(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs, rows);
        break;
    case KIND_GENERIC:
        func(src, dst, xofs, alpha, yofs, beta, xmin, xmax, ksize, rows);
        break;
    default:
        CV_Error(cv::Error::StsInternal, "resize: the plan was never built");
    }
}


// ---- batch scheduling ----

// One image of a batch, bound to the plan its geometry resolved to.
struct BatchItem
{
    const uchar* srcData;
    size_t srcStep;
    uchar* dstData;
    size_t dstStep;
    const ResizePlan* plan;
    int firstRow;   // this image's first destination row in the batch-wide row list
    int rows;
};

// The only parallel loop in resize. Every destination row of every image forms one flat list and
// chunks are cut from that, so a chunk may straddle images and threads get equal work whether the
// batch is two large images or three hundred small ones. A single image is a batch of one.
void runResize(const BatchItem* items, size_t count, int totalRows, double totalPixels)
{
    if (count == 0 || totalRows <= 0)
        return;

    // ~64K output pixels per chunk, the rule the kernels were tuned against. A batch also asks for
    // a few chunks per thread, since a chunk carries a fixed cost per image it spans and sizing by
    // pixels alone under-fills a many-core machine. One image keeps the historical rule exactly.
    double nstripes = totalPixels/(double)(1 << 16);
    if (count > 1)
        nstripes = std::min((double)totalRows, std::max(nstripes, 4.0*getNumThreads()));

    parallel_for_(Range(0, totalRows), [&](const Range& chunk) {
        // The chunk is a slice of the flat row list, so find the image its first row falls in.
        size_t i = (size_t)(std::upper_bound(items, items + count, chunk.start,
                        [](int row, const BatchItem& it) { return row < it.firstRow; }) - items) - 1;
        for (; i < count && items[i].firstRow < chunk.end; i++)
        {
            const BatchItem& it = items[i];
            const Range rows(std::max(chunk.start - it.firstRow, 0),
                             std::min(chunk.end - it.firstRow, it.rows));
            it.plan->run(it.srcData, it.srcStep, it.dstData, it.dstStep, rows);
        }
    }, nstripes);
}

// One plan per distinct geometry. A uniform batch -- the common case, and the whole of the
// tensor path -- leaves the linear scan a single entry, so the tables are built exactly once.
struct PlanCache
{
    struct Key
    {
        Size ssize, dsize;
        size_t srcStep;
        int type;
        bool operator == (const Key& o) const
        { return ssize == o.ssize && dsize == o.dsize && srcStep == o.srcStep && type == o.type; }
    };

    std::vector<Key> keys;
    std::vector<Ptr<ResizePlan> > plans;

    // The returned plan lives as long as the cache.
    const ResizePlan* get(const ResizeParams& params, int type, Size ssize, size_t srcStep, Size dsize)
    {
        Key key; key.ssize = ssize; key.dsize = dsize; key.srcStep = srcStep; key.type = type;
        for (size_t i = 0; i < keys.size(); i++)
            if (keys[i] == key)
                return plans[i].get();

        Ptr<ResizePlan> p = makePtr<ResizePlan>();
        // Classic cv::resize short-circuits an equal-size resize to a copy. Most coordinate modes
        // are not the identity at scale 1, so only PIXEL_CENTER gets the shortcut.
        if (params.coordMode == ResizeCoord::PIXEL_CENTER && ssize == dsize)
            p->buildCopy(type, ssize);
        else
            p->build(type, ssize, srcStep, dsize, params);

        keys.push_back(key);
        plans.push_back(p);
        return p.get();
    }
};

// Collects a batch: hands out plans, lays the images onto the flat row list, totals the work.
// The two batch kinds then differ only in where their pixels come from.
struct BatchBuilder
{
    PlanCache cache;
    AutoBuffer<BatchItem> items;
    size_t count;
    int totalRows;
    double totalPixels;

    explicit BatchBuilder(size_t n) : items(n), count(0), totalRows(0), totalPixels(0) {}

    void add(const ResizePlan* plan, const uchar* srcData, size_t srcStep,
             uchar* dstData, size_t dstStep, Size dsize)
    {
        CV_Assert((int64)totalRows + dsize.height <= INT_MAX);

        BatchItem& it = items[count++];
        it.plan = plan;
        it.srcData = srcData;
        it.srcStep = srcStep;
        it.dstData = dstData;
        it.dstStep = dstStep;
        it.firstRow = totalRows;
        it.rows = dsize.height;

        totalRows += dsize.height;
        totalPixels += dsize.area();
    }

    // Same, resolving the geometry to a cached plan first.
    void add(const ResizeParams& params, int type, const uchar* srcData, size_t srcStep, Size ssize,
             uchar* dstData, size_t dstStep, Size dsize)
    {
        add(cache.get(params, type, ssize, srcStep, dsize), srcData, srcStep, dstData, dstStep, dsize);
    }

    void run() const { runResize(items.data(), count, totalRows, totalPixels); }
};

// One image: a batch of one. The plan stays on the stack rather than in the cache, so the classic
// single-image path allocates nothing extra for going through the batch machinery.
void resizeOne(int type, const uchar* srcData, size_t srcStep, Size ssize,
               uchar* dstData, size_t dstStep, Size dsize, const ResizeParams& params)
{
    ResizePlan plan;
    plan.build(type, ssize, srcStep, dsize, params);

    BatchBuilder batch(1);
    batch.add(&plan, srcData, srcStep, dstData, dstStep, dsize);
    batch.run();
}

// Byte offset of the p-th 2D plane. Walks the axes rather than assuming a dense buffer, so
// views work too.
size_t planeOffset(const Mat& m, int p)
{
    size_t ofs = 0;
    for (int i = m.dims - 3; i >= 0; i--)
    {
        ofs += (size_t)(p % m.size[i]) * m.step[i];
        p /= m.size[i];
    }
    return ofs;
}

// N-D tensor batch: the leading dimensions are batch axes, the last two are height and width.
void resizeTensorBatch(const Mat& src, const ResizeParams& params, OutputArray _dst)
{
    const int D = src.dims;
    const Size ssize(src.size[D - 1], src.size[D - 2]);
    const Size dsize = resolveDstSize(ssize, params);

    int64 planes = 1;
    for (int i = 0; i < D - 2; i++)
        planes *= src.size[i];
    CV_Assert(planes > 0 && planes * dsize.height <= INT_MAX);
    const int numPlanes = (int)planes;

    int dstShape[CV_MAX_DIM];
    for (int i = 0; i < D; i++)
        dstShape[i] = src.size[i];
    dstShape[D - 2] = dsize.height;
    dstShape[D - 1] = dsize.width;
    _dst.create(D, dstShape, src.type());
    Mat dst = _dst.getMat();

    const size_t srcStep = src.step[D - 2], dstStep = dst.step[D - 2];

    // Every plane shares one geometry, so the cache builds one plan for the whole tensor.
    BatchBuilder batch(numPlanes);
    for (int p = 0; p < numPlanes; p++)
        batch.add(params, src.type(), src.data + planeOffset(src, p), srcStep, ssize,
                  dst.data + planeOffset(dst, p), dstStep, dsize);
    batch.run();
}

// vector<Mat> batch: the elements may differ in size, so only the matching ones share a plan.
void resizeMatVectorBatch(const std::vector<Mat>& srcs, const ResizeParams& params, OutputArray _dst)
{
    const size_t n = srcs.size();
    CV_Assert(n > 0);
    CV_Assert((int64)n <= INT_MAX);

    const int type = srcs[0].type();
    _dst.create((int)n, 1, 0);

    // getMatRef() is an out-of-line call into core; over thousands of elements that dispatch costs
    // more than the resizing, so take it once here. Any other output kind keeps the accessor.
    std::vector<Mat>* dstv = _dst.kind() == _InputArray::STD_VECTOR_MAT
                           ? (std::vector<Mat>*)_dst.getObj() : 0;

    // Plans are built here, before any thread runs; a same-shape batch shares a single one.
    BatchBuilder batch(n);
    for (size_t i = 0; i < n; i++)
    {
        const Mat& src = srcs[i];
        CV_Assert(src.dims == 2 && !src.empty() && src.type() == type);
        const Size dsize = resolveDstSize(src.size(), params);
        Mat& dst = dstv ? (*dstv)[i] : _dst.getMatRef((int)i);
        dst.create(dsize, type);

        batch.add(params, type, src.data, src.step, src.size(), dst.data, dst.step, dsize);
    }
    batch.run();
}

} // namespace

namespace hal {

void resize(int src_type,
            const uchar * src_data, size_t src_step, int src_width, int src_height,
            uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
            double inv_scale_x, double inv_scale_y, int interpolation)
{
    CV_INSTRUMENT_REGION();

    CV_Assert((dst_width > 0 && dst_height > 0) || (inv_scale_x > 0 && inv_scale_y > 0));
    if (inv_scale_x < DBL_EPSILON || inv_scale_y < DBL_EPSILON)
    {
        inv_scale_x = static_cast<double>(dst_width) / src_width;
        inv_scale_y = static_cast<double>(dst_height) / src_height;
    }

    CALL_HAL(resize, cv_hal_resize, src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);

    // This entry point is given the scale, so leave dsize empty and let the plan derive it.
    ResizeParams params;
    params.fx = inv_scale_x;
    params.fy = inv_scale_y;
    params.interpolation = interpolation;

    const Size ssize(src_width, src_height);
    const Size dsize = resolveDstSize(ssize, params);
    CV_Assert( !dsize.empty() );

    resizeOne(src_type, src_data, src_step, ssize, dst_data, dst_step, dsize, params);
}


} // cv::hal::

} // cv::

//==================================================================================================

void cv::resize( InputArray _src, OutputArray _dst, Size dsize,
                 double inv_scale_x, double inv_scale_y, int interpolation )
{
    CV_INSTRUMENT_REGION();

    // Batch input (N-D tensor or vector<Mat>/<UMat>): delegate to the ResizeParams overload.
    if (_src.isMatVector() || _src.isUMatVector() || _src.dims() > 2)
    {
        cv::resize(_src, _dst, ResizeParams(dsize, inv_scale_x, inv_scale_y, interpolation));
        return;
    }

    Size ssize = _src.size();

    CV_Assert( !ssize.empty() );
    if( dsize.empty() )
    {
        CV_Assert(inv_scale_x > 0); CV_Assert(inv_scale_y > 0);
        dsize = Size(saturate_cast<int>(ssize.width*inv_scale_x),
                     saturate_cast<int>(ssize.height*inv_scale_y));
        CV_Assert( !dsize.empty() );
    }
    else
    {
        inv_scale_x = (double)dsize.width/ssize.width;
        inv_scale_y = (double)dsize.height/ssize.height;
        CV_Assert(inv_scale_x > 0); CV_Assert(inv_scale_y > 0);
    }

    if (interpolation == INTER_LINEAR_EXACT && (_src.depth() == CV_32F || _src.depth() == CV_64F))
        interpolation = INTER_LINEAR; // If depth isn't supported fallback to generic resize

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat() && _src.cols() > 10 && _src.rows() > 10,
               ocl_resize(_src, _dst, dsize, inv_scale_x, inv_scale_y, interpolation))

    // Fake reference to source. Resolves issue 13577 in case of src == dst.
    UMat srcUMat;
    if (_src.isUMat())
        srcUMat = _src.getUMat();

    Mat src = _src.getMat();
    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

    if (dsize == ssize)
    {
        // Source and destination are of same size. Use simple copy.
        src.copyTo(dst);
        return;
    }

    hal::resize(src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows, inv_scale_x, inv_scale_y, interpolation);
}

//==================================================================================================

cv::ResizeParams::ResizeParams()
    : dsize(), fx(0), fy(0), interpolation(INTER_LINEAR),
      coordMode(ResizeCoord::PIXEL_CENTER), nearestMode(ResizeNearest::ROUND_PREFER_FLOOR),
      cubicCoeffA(-0.75f), excludeOutside(false), antialias(false), hint(cv::ALGO_HINT_DEFAULT)
{
}

cv::ResizeParams::ResizeParams(Size _dsize, double _fx, double _fy, int _interpolation)
    : dsize(_dsize), fx(_fx), fy(_fy), interpolation(_interpolation),
      coordMode(ResizeCoord::PIXEL_CENTER), nearestMode(ResizeNearest::ROUND_PREFER_FLOOR),
      cubicCoeffA(-0.75f), excludeOutside(false), antialias(false), hint(cv::ALGO_HINT_DEFAULT)
{
}

void cv::resize( InputArray _src, OutputArray _dst, const ResizeParams& params )
{
    CV_INSTRUMENT_REGION();

    checkResizeParams(params);

    if (_src.isUMatVector())
    {
        // OpenCL elements go one at a time: the plans below address host memory.
        std::vector<UMat> srcs;
        _src.getUMatVector(srcs);
        CV_Assert(!srcs.empty());
        _dst.create((int)srcs.size(), 1, 0);
        for (size_t i = 0; i < srcs.size(); i++)
            cv::resize(srcs[i], _dst.getUMatRef((int)i), params);
        return;
    }

    if (_src.isMatVector())
    {
        std::vector<Mat> srcs;
        _src.getMatVector(srcs);
        resizeMatVectorBatch(srcs, params, _dst);
        return;
    }

    if (_src.dims() > 2)
    {
        resizeTensorBatch(_src.getMat(), params, _dst);
        return;
    }

    if (params.coordMode == ResizeCoord::PIXEL_CENTER)
    {
        // A single 2D image with the classic convention: go through classic cv::resize, which
        // owns the OpenCL, IPP and custom-HAL fast paths.
        cv::resize(_src, _dst, params.dsize, params.fx, params.fy,
                   resolveInterpolation(params, _src.depth()));
        return;
    }

    const Size ssize = _src.size();
    const Size dsize = resolveDstSize(ssize, params);

    Mat src = _src.getMat();
    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

    resizeOne(src.type(), src.data, src.step, ssize, dst.data, dst.step, dsize, params);
}
