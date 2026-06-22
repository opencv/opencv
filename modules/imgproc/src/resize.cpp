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
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
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

#include "opencv2/core/openvx/ovx_defs.hpp"
#include "resize.hpp"

#include "opencv2/core/softfloat.hpp"
#include "fixedpoint.inl.hpp"

using namespace cv;

namespace cv {
namespace resize_fixedpoint {

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
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 1>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[1];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + ofst[dst_width - 1])[0];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 2>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + 2*ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[2];
            *(dst++) = m[0] * px[1] + m[1] * px[3];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 2*ofst[dst_width - 1])[0];
        src1 = (src + 2*ofst[dst_width - 1])[1];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 3>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + 3*ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[3];
            *(dst++) = m[0] * px[1] + m[1] * px[4];
            *(dst++) = m[0] * px[2] + m[1] * px[5];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 3*ofst[dst_width - 1])[0];
        src1 = (src + 3*ofst[dst_width - 1])[1];
        src2 = (src + 3*ofst[dst_width - 1])[2];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 4>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]), src3(src[3]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + 4*ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[4];
            *(dst++) = m[0] * px[1] + m[1] * px[5];
            *(dst++) = m[0] * px[2] + m[1] * px[6];
            *(dst++) = m[0] * px[3] + m[1] * px[7];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 4*ofst[dst_width - 1])[0];
        src1 = (src + 4*ofst[dst_width - 1])[1];
        src2 = (src + 4*ofst[dst_width - 1])[2];
        src3 = (src + 4*ofst[dst_width - 1])[3];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 1>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[1] + m[2] * src[2] + m[3] * src[3];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + ofst[dst_width - 1])[0];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 2>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + 2*ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[2] + m[2] * src[4] + m[3] * src[6];
            *(dst++) = m[0] * src[1] + m[1] * src[3] + m[2] * src[5] + m[3] * src[7];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 2*ofst[dst_width - 1])[0];
        src1 = (src + 2*ofst[dst_width - 1])[1];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 3>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + 3*ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[3] + m[2] * src[6] + m[3] * src[ 9];
            *(dst++) = m[0] * src[1] + m[1] * src[4] + m[2] * src[7] + m[3] * src[10];
            *(dst++) = m[0] * src[2] + m[1] * src[5] + m[2] * src[8] + m[3] * src[11];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 3*ofst[dst_width - 1])[0];
        src1 = (src + 3*ofst[dst_width - 1])[1];
        src2 = (src + 3*ofst[dst_width - 1])[2];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 4>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]), src3(src[3]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + 4*ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[4] + m[2] * src[ 8] + m[3] * src[12];
            *(dst++) = m[0] * src[1] + m[1] * src[5] + m[2] * src[ 9] + m[3] * src[13];
            *(dst++) = m[0] * src[2] + m[1] * src[6] + m[2] * src[10] + m[3] * src[14];
            *(dst++) = m[0] * src[3] + m[1] * src[7] + m[2] * src[11] + m[3] * src[15];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 4*ofst[dst_width - 1])[0];
        src1 = (src + 4*ofst[dst_width - 1])[1];
        src2 = (src + 4*ofst[dst_width - 1])[2];
        src3 = (src + 4*ofst[dst_width - 1])[3];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
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

template <typename ET, typename interpolation>
void resize_bitExact(const uchar* src, size_t src_step, int src_width, int src_height,
                           uchar* dst, size_t dst_step, int dst_width, int dst_height,
                     int cn, double inv_scale_x, double inv_scale_y)
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

    AutoBuffer<uchar> buf( dst_width * sizeof(int) +
                           dst_height * sizeof(int) +
                           dst_width * interp_x.len*sizeof(fixedpoint) +
                           dst_height * interp_y.len * sizeof(fixedpoint) );
    int* xoffsets = (int*)buf.data();
    int* yoffsets = xoffsets + dst_width;
    fixedpoint* xcoeffs = (fixedpoint*)(yoffsets + dst_height);
    fixedpoint* ycoeffs = xcoeffs + dst_width * interp_x.len;

    int min_x, max_x, min_y, max_y;
    for (int dx = 0; dx < dst_width; dx++)
        interp_x.getCoeffs(dx, xoffsets+dx, xcoeffs+dx*interp_x.len);
    interp_x.getMinMax(min_x, max_x);
    for (int dy = 0; dy < dst_height; dy++)
        interp_y.getCoeffs(dy, yoffsets+dy, ycoeffs+dy*interp_y.len);
    interp_y.getMinMax(min_y, max_y);

    resize_bitExactInvoker<ET, fixedpoint, interpolation::len> invoker(src, src_step, src_width, src_height, dst, dst_step, dst_width, dst_height, cn,
                                                                       xoffsets, yoffsets, xcoeffs, ycoeffs, min_x, max_x, min_y, max_y, hResize);
    Range range(0, dst_height);
    parallel_for_(range, invoker, dst_width * dst_height / (double)(1 << 16));
}

typedef void(*be_resize_func)(const uchar* src, size_t src_step, int src_width, int src_height,
                                    uchar* dst, size_t dst_step, int dst_width, int dst_height,
                              int cn, double inv_scale_x, double inv_scale_y);

} // namespace resize_fixedpoint

static resize_fixedpoint::be_resize_func resizeLinearExact_tab[] =
{
    resize_fixedpoint::resize_bitExact<uchar, resize_fixedpoint::interpolationLinear<uchar> >,
    resize_fixedpoint::resize_bitExact<schar, resize_fixedpoint::interpolationLinear<schar> >,
    resize_fixedpoint::resize_bitExact<ushort, resize_fixedpoint::interpolationLinear<ushort> >,
    resize_fixedpoint::resize_bitExact<short, resize_fixedpoint::interpolationLinear<short> >,
    resize_fixedpoint::resize_bitExact<int, resize_fixedpoint::interpolationLinear<int> >,
    0, 0, 0
};

bool resizeLinearExact(int src_type,
                       const uchar* src_data, size_t src_step, int src_width, int src_height,
                       uchar* dst_data, size_t dst_step, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int* interpolation)
{
    int depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);

    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;
    int iscale_x = saturate_cast<int>(scale_x);
    int iscale_y = saturate_cast<int>(scale_y);
    bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
                        std::abs(scale_y - iscale_y) < DBL_EPSILON;

    if (is_area_fast && iscale_x == 2 && iscale_y == 2 && cn != 2)
    {
        *interpolation = INTER_AREA;
        return false;
    }

    resize_fixedpoint::be_resize_func func = resizeLinearExact_tab[depth];
    if (!func)
        return false;

    func(src_data, src_step, src_width, src_height,
         dst_data, dst_step, dst_width, dst_height,
         cn, inv_scale_x, inv_scale_y);
    return true;
}

#ifdef HAVE_OPENCL
const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
#endif

/************** interpolation formulas and tables ***************/

#ifdef HAVE_OPENCL
static void ocl_computeResizeAreaTabs(int ssize, int dsize, double scale, int * const map_tab,
                                      float * const alpha_tab, int * const ofs_tab)
{
    int k = 0, dx = 0;
    for ( ; dx < dsize; dx++)
    {
        ofs_tab[dx] = k;

        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if (sx1 - fsx1 > 1e-3)
        {
            map_tab[k] = sx1 - 1;
            alpha_tab[k++] = (float)((sx1 - fsx1) / cellWidth);
        }

        for (int sx = sx1; sx < sx2; sx++)
        {
            map_tab[k] = sx;
            alpha_tab[k++] = float(1.0 / cellWidth);
        }

        if (fsx2 - sx2 > 1e-3)
        {
            map_tab[k] = sx2;
            alpha_tab[k++] = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    ofs_tab[dx] = k;
}

static bool ocl_resize( InputArray _src, OutputArray _dst, Size dsize,
                        double fx, double fy, int interpolation)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    double inv_fx = 1.0 / fx, inv_fy = 1.0 / fy;
    float inv_fxf = (float)inv_fx, inv_fyf = (float)inv_fy;
    int iscale_x = saturate_cast<int>(inv_fx), iscale_y = saturate_cast<int>(inv_fy);
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

void resize( InputArray _src, OutputArray _dst, Size dsize,
                 double inv_scale_x, double inv_scale_y, int interpolation )
{
    CV_INSTRUMENT_REGION();

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

} // namespace cv

#ifndef OPENCV_EXCLUDE_C_API

CV_IMPL void
cvResize( const CvArr* srcarr, CvArr* dstarr, int method )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() );
    cv::resize( src, dst, dst.size(), (double)dst.cols/src.cols,
        (double)dst.rows/src.rows, method );
}

#endif
/* End of file. */
