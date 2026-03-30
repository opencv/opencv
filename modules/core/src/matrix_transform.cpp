// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/detail/dispatch_helper.impl.hpp"

#include <algorithm> // std::swap_ranges
#include <numeric> // std::accumulate

namespace cv {

////////////////////////////////////// transpose /////////////////////////////////////////
#if CV_SIMD128
static void transpose_8bit_simd(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz)
{
    const int m = sz.width, n = sz.height;
    int i = 0;
    for (; i <= m - 16; i += 16)
    {
        int j = 0;
        for (; j <= n - 16; j += 16)
        {
            v_uint8x16 r0  = v_load(src + i + sstep*(j+ 0));
            v_uint8x16 r1  = v_load(src + i + sstep*(j+ 1));
            v_uint8x16 r2  = v_load(src + i + sstep*(j+ 2));
            v_uint8x16 r3  = v_load(src + i + sstep*(j+ 3));
            v_uint8x16 r4  = v_load(src + i + sstep*(j+ 4));
            v_uint8x16 r5  = v_load(src + i + sstep*(j+ 5));
            v_uint8x16 r6  = v_load(src + i + sstep*(j+ 6));
            v_uint8x16 r7  = v_load(src + i + sstep*(j+ 7));
            v_uint8x16 r8  = v_load(src + i + sstep*(j+ 8));
            v_uint8x16 r9  = v_load(src + i + sstep*(j+ 9));
            v_uint8x16 r10 = v_load(src + i + sstep*(j+10));
            v_uint8x16 r11 = v_load(src + i + sstep*(j+11));
            v_uint8x16 r12 = v_load(src + i + sstep*(j+12));
            v_uint8x16 r13 = v_load(src + i + sstep*(j+13));
            v_uint8x16 r14 = v_load(src + i + sstep*(j+14));
            v_uint8x16 r15 = v_load(src + i + sstep*(j+15));

            v_uint8x16 t0, t1, t2, t3, t4, t5, t6, t7,
                       t8, t9, t10, t11, t12, t13, t14, t15;

            v_zip(r0,  r1,  t0,  t1);
            v_zip(r2,  r3,  t2,  t3);
            v_zip(r4,  r5,  t4,  t5);
            v_zip(r6,  r7,  t6,  t7);
            v_zip(r8,  r9,  t8,  t9);
            v_zip(r10, r11, t10, t11);
            v_zip(r12, r13, t12, t13);
            v_zip(r14, r15, t14, t15);

            v_uint16x8 s0, s1, s2, s3, s4, s5, s6, s7,
                       s8, s9, s10, s11, s12, s13, s14, s15;
            v_zip(v_reinterpret_as_u16(t0),  v_reinterpret_as_u16(t2),  s0,  s1);
            v_zip(v_reinterpret_as_u16(t1),  v_reinterpret_as_u16(t3),  s2,  s3);
            v_zip(v_reinterpret_as_u16(t4),  v_reinterpret_as_u16(t6),  s4,  s5);
            v_zip(v_reinterpret_as_u16(t5),  v_reinterpret_as_u16(t7),  s6,  s7);
            v_zip(v_reinterpret_as_u16(t8),  v_reinterpret_as_u16(t10), s8,  s9);
            v_zip(v_reinterpret_as_u16(t9),  v_reinterpret_as_u16(t11), s10, s11);
            v_zip(v_reinterpret_as_u16(t12), v_reinterpret_as_u16(t14), s12, s13);
            v_zip(v_reinterpret_as_u16(t13), v_reinterpret_as_u16(t15), s14, s15);

            v_uint32x4 u0, u1, u2, u3, u4, u5, u6, u7,
                       u8, u9, u10, u11, u12, u13, u14, u15;

            v_zip(v_reinterpret_as_u32(s0),  v_reinterpret_as_u32(s4),  u0,  u1);
            v_zip(v_reinterpret_as_u32(s1),  v_reinterpret_as_u32(s5),  u2,  u3);
            v_zip(v_reinterpret_as_u32(s2),  v_reinterpret_as_u32(s6),  u4,  u5);
            v_zip(v_reinterpret_as_u32(s3),  v_reinterpret_as_u32(s7),  u6,  u7);
            v_zip(v_reinterpret_as_u32(s8),  v_reinterpret_as_u32(s12), u8,  u9);
            v_zip(v_reinterpret_as_u32(s9),  v_reinterpret_as_u32(s13), u10, u11);
            v_zip(v_reinterpret_as_u32(s10), v_reinterpret_as_u32(s14), u12, u13);
            v_zip(v_reinterpret_as_u32(s11), v_reinterpret_as_u32(s15), u14, u15);

            v_uint32x4 v0  = v_combine_low (u0,  u8);
            v_uint32x4 v1  = v_combine_high(u0,  u8);
            v_uint32x4 v2  = v_combine_low (u1,  u9);
            v_uint32x4 v3  = v_combine_high(u1,  u9);
            v_uint32x4 v4  = v_combine_low (u2,  u10);
            v_uint32x4 v5  = v_combine_high(u2,  u10);
            v_uint32x4 v6  = v_combine_low (u3,  u11);
            v_uint32x4 v7  = v_combine_high(u3,  u11);
            v_uint32x4 v8  = v_combine_low (u4,  u12);
            v_uint32x4 v9  = v_combine_high(u4,  u12);
            v_uint32x4 v10 = v_combine_low (u5,  u13);
            v_uint32x4 v11 = v_combine_high(u5,  u13);
            v_uint32x4 v12 = v_combine_low (u6,  u14);
            v_uint32x4 v13 = v_combine_high(u6,  u14);
            v_uint32x4 v14 = v_combine_low (u7,  u15);
            v_uint32x4 v15 = v_combine_high(u7,  u15);

            v_store(dst + dstep*(i+ 0) + j, v_reinterpret_as_u8(v0));
            v_store(dst + dstep*(i+ 1) + j, v_reinterpret_as_u8(v1));
            v_store(dst + dstep*(i+ 2) + j, v_reinterpret_as_u8(v2));
            v_store(dst + dstep*(i+ 3) + j, v_reinterpret_as_u8(v3));
            v_store(dst + dstep*(i+ 4) + j, v_reinterpret_as_u8(v4));
            v_store(dst + dstep*(i+ 5) + j, v_reinterpret_as_u8(v5));
            v_store(dst + dstep*(i+ 6) + j, v_reinterpret_as_u8(v6));
            v_store(dst + dstep*(i+ 7) + j, v_reinterpret_as_u8(v7));
            v_store(dst + dstep*(i+ 8) + j, v_reinterpret_as_u8(v8));
            v_store(dst + dstep*(i+ 9) + j, v_reinterpret_as_u8(v9));
            v_store(dst + dstep*(i+10) + j, v_reinterpret_as_u8(v10));
            v_store(dst + dstep*(i+11) + j, v_reinterpret_as_u8(v11));
            v_store(dst + dstep*(i+12) + j, v_reinterpret_as_u8(v12));
            v_store(dst + dstep*(i+13) + j, v_reinterpret_as_u8(v13));
            v_store(dst + dstep*(i+14) + j, v_reinterpret_as_u8(v14));
            v_store(dst + dstep*(i+15) + j, v_reinterpret_as_u8(v15));
        }
        for (; j < n; j++)
            for (int k = 0; k < 16; k++)
                dst[dstep*(i+k) + j] = src[i + sstep*j + k];
    }
    for (; i < m; i++)
        for (int j = 0; j < n; j++)
            dst[dstep*i + j] = src[i + sstep*j];
}

static void transpose_16bit_simd(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz)
{
    const ushort* src16 = reinterpret_cast<const ushort*>(src);
    ushort* dst16 = reinterpret_cast<ushort*>(dst);

    const size_t sstep_e = sstep / sizeof(ushort);
    const size_t dstep_e = dstep / sizeof(ushort);

    const int m = sz.width, n = sz.height;
    int i = 0;

    for (; i <= m - 8; i += 8)
    {
        int j = 0;
        for (; j <= n - 8; j += 8)
        {
            v_uint16x8 r0 = v_load(src16 + i + sstep_e*(j+0));
            v_uint16x8 r1 = v_load(src16 + i + sstep_e*(j+1));
            v_uint16x8 r2 = v_load(src16 + i + sstep_e*(j+2));
            v_uint16x8 r3 = v_load(src16 + i + sstep_e*(j+3));
            v_uint16x8 r4 = v_load(src16 + i + sstep_e*(j+4));
            v_uint16x8 r5 = v_load(src16 + i + sstep_e*(j+5));
            v_uint16x8 r6 = v_load(src16 + i + sstep_e*(j+6));
            v_uint16x8 r7 = v_load(src16 + i + sstep_e*(j+7));

            v_uint16x8 t0, t1, t2, t3, t4, t5, t6, t7;
            v_zip(r0, r1, t0, t1);
            v_zip(r2, r3, t2, t3);
            v_zip(r4, r5, t4, t5);
            v_zip(r6, r7, t6, t7);
            v_uint32x4 u0, u1, u2, u3, u4, u5, u6, u7;
            v_zip(v_reinterpret_as_u32(t0), v_reinterpret_as_u32(t4), u0, u1);
            v_zip(v_reinterpret_as_u32(t1), v_reinterpret_as_u32(t5), u2, u3);
            v_zip(v_reinterpret_as_u32(t2), v_reinterpret_as_u32(t6), u4, u5);
            v_zip(v_reinterpret_as_u32(t3), v_reinterpret_as_u32(t7), u6, u7);
            v_uint32x4 v0, v1, v2, v3, v4, v5, v6, v7;
            v_zip(u0, u4, v0, v1);
            v_zip(u1, u5, v2, v3);
            v_zip(u2, u6, v4, v5);
            v_zip(u3, u7, v6, v7);

            v_store(dst16 + dstep_e*(i+0) + j, v_reinterpret_as_u16(v0));
            v_store(dst16 + dstep_e*(i+1) + j, v_reinterpret_as_u16(v1));
            v_store(dst16 + dstep_e*(i+2) + j, v_reinterpret_as_u16(v2));
            v_store(dst16 + dstep_e*(i+3) + j, v_reinterpret_as_u16(v3));
            v_store(dst16 + dstep_e*(i+4) + j, v_reinterpret_as_u16(v4));
            v_store(dst16 + dstep_e*(i+5) + j, v_reinterpret_as_u16(v5));
            v_store(dst16 + dstep_e*(i+6) + j, v_reinterpret_as_u16(v6));
            v_store(dst16 + dstep_e*(i+7) + j, v_reinterpret_as_u16(v7));
        }
        for (; j < n; j++)
            for (int k = 0; k < 8; k++)
                dst16[dstep_e*(i+k) + j] = src16[i + sstep_e*j + k];
    }
    for (; i < m; i++)
        for (int j = 0; j < n; j++)
            dst16[dstep_e*i + j] = src16[i + sstep_e*j];
}

static void transpose_32bit_simd(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz)
{
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);

    const size_t sstep_e = sstep / sizeof(uint32_t);
    const size_t dstep_e = dstep / sizeof(uint32_t);

    const int m = sz.width, n = sz.height;
    int i = 0;
    for (; i <= m - 4; i += 4)
    {
        int j = 0;
        for (; j <= n - 4; j += 4)
        {
            v_uint32x4 r0 = v_load(src32 + i + sstep_e*(j+0));
            v_uint32x4 r1 = v_load(src32 + i + sstep_e*(j+1));
            v_uint32x4 r2 = v_load(src32 + i + sstep_e*(j+2));
            v_uint32x4 r3 = v_load(src32 + i + sstep_e*(j+3));
            v_uint32x4 o0, o1, o2, o3;
            v_transpose4x4(r0, r1, r2, r3, o0, o1, o2, o3);

            v_store(dst32 + dstep_e*(i+0) + j, o0);
            v_store(dst32 + dstep_e*(i+1) + j, o1);
            v_store(dst32 + dstep_e*(i+2) + j, o2);
            v_store(dst32 + dstep_e*(i+3) + j, o3);
        }
        for (; j < n; j++)
            for (int k = 0; k < 4; k++)
                dst32[dstep_e*(i+k) + j] = src32[i + sstep_e*j + k];
    }
    for (; i < m; i++)
        for (int j = 0; j < n; j++)
            dst32[dstep_e*i + j] = src32[i + sstep_e*j];
}

static void transpose_48bit_simd(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz)
{
    const short* src16 = reinterpret_cast<const short*>(src);
    short* dst16 = reinterpret_cast<short*>(dst);

    const size_t sstep_e = sstep / sizeof(short);
    const size_t dstep_e = dstep / sizeof(short);

    const int m = sz.width, n = sz.height;
    int i = 0;

    for (; i <= m - 8; i += 8)
    {
        int j = 0;
        for (; j <= n - 8; j += 8)
        {
            v_int16x8 C0_0, C1_0, C2_0;
            v_int16x8 C0_1, C1_1, C2_1;
            v_int16x8 C0_2, C1_2, C2_2;
            v_int16x8 C0_3, C1_3, C2_3;
            v_int16x8 C0_4, C1_4, C2_4;
            v_int16x8 C0_5, C1_5, C2_5;
            v_int16x8 C0_6, C1_6, C2_6;
            v_int16x8 C0_7, C1_7, C2_7;

            v_load_deinterleave(src16 + sstep_e*(j+0) + i*3, C0_0, C1_0, C2_0);
            v_load_deinterleave(src16 + sstep_e*(j+1) + i*3, C0_1, C1_1, C2_1);
            v_load_deinterleave(src16 + sstep_e*(j+2) + i*3, C0_2, C1_2, C2_2);
            v_load_deinterleave(src16 + sstep_e*(j+3) + i*3, C0_3, C1_3, C2_3);
            v_load_deinterleave(src16 + sstep_e*(j+4) + i*3, C0_4, C1_4, C2_4);
            v_load_deinterleave(src16 + sstep_e*(j+5) + i*3, C0_5, C1_5, C2_5);
            v_load_deinterleave(src16 + sstep_e*(j+6) + i*3, C0_6, C1_6, C2_6);
            v_load_deinterleave(src16 + sstep_e*(j+7) + i*3, C0_7, C1_7, C2_7);

            v_uint16x8 t0, t1, t2, t3, t4, t5, t6, t7;
            v_zip(v_reinterpret_as_u16(C0_0), v_reinterpret_as_u16(C0_1), t0, t1);
            v_zip(v_reinterpret_as_u16(C0_2), v_reinterpret_as_u16(C0_3), t2, t3);
            v_zip(v_reinterpret_as_u16(C0_4), v_reinterpret_as_u16(C0_5), t4, t5);
            v_zip(v_reinterpret_as_u16(C0_6), v_reinterpret_as_u16(C0_7), t6, t7);
            v_uint32x4 u0, u1, u2, u3, u4, u5, u6, u7;
            v_zip(v_reinterpret_as_u32(t0), v_reinterpret_as_u32(t4), u0, u1);
            v_zip(v_reinterpret_as_u32(t1), v_reinterpret_as_u32(t5), u2, u3);
            v_zip(v_reinterpret_as_u32(t2), v_reinterpret_as_u32(t6), u4, u5);
            v_zip(v_reinterpret_as_u32(t3), v_reinterpret_as_u32(t7), u6, u7);
            v_uint32x4 s0, s1, s2, s3, s4, s5, s6, s7;
            v_zip(u0, u4, s0, s1); v_zip(u1, u5, s2, s3);
            v_zip(u2, u6, s4, s5); v_zip(u3, u7, s6, s7);
            v_int16x8 r0_0 = v_reinterpret_as_s16(s0), r0_1 = v_reinterpret_as_s16(s1);
            v_int16x8 r0_2 = v_reinterpret_as_s16(s2), r0_3 = v_reinterpret_as_s16(s3);
            v_int16x8 r0_4 = v_reinterpret_as_s16(s4), r0_5 = v_reinterpret_as_s16(s5);
            v_int16x8 r0_6 = v_reinterpret_as_s16(s6), r0_7 = v_reinterpret_as_s16(s7);

            v_zip(v_reinterpret_as_u16(C1_0), v_reinterpret_as_u16(C1_1), t0, t1);
            v_zip(v_reinterpret_as_u16(C1_2), v_reinterpret_as_u16(C1_3), t2, t3);
            v_zip(v_reinterpret_as_u16(C1_4), v_reinterpret_as_u16(C1_5), t4, t5);
            v_zip(v_reinterpret_as_u16(C1_6), v_reinterpret_as_u16(C1_7), t6, t7);
            v_zip(v_reinterpret_as_u32(t0), v_reinterpret_as_u32(t4), u0, u1);
            v_zip(v_reinterpret_as_u32(t1), v_reinterpret_as_u32(t5), u2, u3);
            v_zip(v_reinterpret_as_u32(t2), v_reinterpret_as_u32(t6), u4, u5);
            v_zip(v_reinterpret_as_u32(t3), v_reinterpret_as_u32(t7), u6, u7);
            v_zip(u0, u4, s0, s1); v_zip(u1, u5, s2, s3);
            v_zip(u2, u6, s4, s5); v_zip(u3, u7, s6, s7);
            v_int16x8 r1_0 = v_reinterpret_as_s16(s0), r1_1 = v_reinterpret_as_s16(s1);
            v_int16x8 r1_2 = v_reinterpret_as_s16(s2), r1_3 = v_reinterpret_as_s16(s3);
            v_int16x8 r1_4 = v_reinterpret_as_s16(s4), r1_5 = v_reinterpret_as_s16(s5);
            v_int16x8 r1_6 = v_reinterpret_as_s16(s6), r1_7 = v_reinterpret_as_s16(s7);

            v_zip(v_reinterpret_as_u16(C2_0), v_reinterpret_as_u16(C2_1), t0, t1);
            v_zip(v_reinterpret_as_u16(C2_2), v_reinterpret_as_u16(C2_3), t2, t3);
            v_zip(v_reinterpret_as_u16(C2_4), v_reinterpret_as_u16(C2_5), t4, t5);
            v_zip(v_reinterpret_as_u16(C2_6), v_reinterpret_as_u16(C2_7), t6, t7);
            v_zip(v_reinterpret_as_u32(t0), v_reinterpret_as_u32(t4), u0, u1);
            v_zip(v_reinterpret_as_u32(t1), v_reinterpret_as_u32(t5), u2, u3);
            v_zip(v_reinterpret_as_u32(t2), v_reinterpret_as_u32(t6), u4, u5);
            v_zip(v_reinterpret_as_u32(t3), v_reinterpret_as_u32(t7), u6, u7);
            v_zip(u0, u4, s0, s1); v_zip(u1, u5, s2, s3);
            v_zip(u2, u6, s4, s5); v_zip(u3, u7, s6, s7);
            v_int16x8 r2_0 = v_reinterpret_as_s16(s0), r2_1 = v_reinterpret_as_s16(s1);
            v_int16x8 r2_2 = v_reinterpret_as_s16(s2), r2_3 = v_reinterpret_as_s16(s3);
            v_int16x8 r2_4 = v_reinterpret_as_s16(s4), r2_5 = v_reinterpret_as_s16(s5);
            v_int16x8 r2_6 = v_reinterpret_as_s16(s6), r2_7 = v_reinterpret_as_s16(s7);

            v_store_interleave(dst16 + dstep_e*(i+0) + j*3, r0_0, r1_0, r2_0);
            v_store_interleave(dst16 + dstep_e*(i+1) + j*3, r0_1, r1_1, r2_1);
            v_store_interleave(dst16 + dstep_e*(i+2) + j*3, r0_2, r1_2, r2_2);
            v_store_interleave(dst16 + dstep_e*(i+3) + j*3, r0_3, r1_3, r2_3);
            v_store_interleave(dst16 + dstep_e*(i+4) + j*3, r0_4, r1_4, r2_4);
            v_store_interleave(dst16 + dstep_e*(i+5) + j*3, r0_5, r1_5, r2_5);
            v_store_interleave(dst16 + dstep_e*(i+6) + j*3, r0_6, r1_6, r2_6);
            v_store_interleave(dst16 + dstep_e*(i+7) + j*3, r0_7, r1_7, r2_7);
        }
        for (; j < n; j++)
            for (int k = 0; k < 8; k++)
            {
                dst16[dstep_e*(i+k) + j*3 + 0] = src16[sstep_e*j + (i+k)*3 + 0];
                dst16[dstep_e*(i+k) + j*3 + 1] = src16[sstep_e*j + (i+k)*3 + 1];
                dst16[dstep_e*(i+k) + j*3 + 2] = src16[sstep_e*j + (i+k)*3 + 2];
            }
    }
    for (; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            dst16[dstep_e*i + j*3 + 0] = src16[sstep_e*j + i*3 + 0];
            dst16[dstep_e*i + j*3 + 1] = src16[sstep_e*j + i*3 + 1];
            dst16[dstep_e*i + j*3 + 2] = src16[sstep_e*j + i*3 + 2];
        }
}
#endif

template<typename T> static void
transpose_( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz )
{
#if CV_SIMD128
    switch (sizeof(T))
    {
        case 1: transpose_8bit_simd(src, sstep, dst, dstep, sz);  return;
        case 2: transpose_16bit_simd(src, sstep, dst, dstep, sz); return;
        case 4: transpose_32bit_simd(src, sstep, dst, dstep, sz); return;
        case 6: transpose_48bit_simd(src, sstep, dst, dstep, sz); return;
        default: break;
    }
#endif

    int i = 0, j, m = sz.width, n = sz.height;

    #if CV_ENABLE_UNROLLED
    for(; i <= m - 4; i += 4 )
    {
        T* d0 = (T*)(dst + dstep*i);
        T* d1 = (T*)(dst + dstep*(i+1));
        T* d2 = (T*)(dst + dstep*(i+2));
        T* d3 = (T*)(dst + dstep*(i+3));

        for( j = 0; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
            d1[j] = s0[1]; d1[j+1] = s1[1]; d1[j+2] = s2[1]; d1[j+3] = s3[1];
            d2[j] = s0[2]; d2[j+1] = s1[2]; d2[j+2] = s2[2]; d2[j+3] = s3[2];
            d3[j] = s0[3]; d3[j+1] = s1[3]; d3[j+2] = s2[3]; d3[j+3] = s3[3];
        }

        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0]; d1[j] = s0[1]; d2[j] = s0[2]; d3[j] = s0[3];
        }
    }
    #endif
    for( ; i < m; i++ )
    {
        T* d0 = (T*)(dst + dstep*i);
        j = 0;
        #if CV_ENABLE_UNROLLED
        for(; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
        }
        #endif
        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0];
        }
    }
}

template<typename T> static void
transposeI_( uchar* data, size_t step, int n )
{
    for( int i = 0; i < n; i++ )
    {
        T* row = (T*)(data + step*i);
        uchar* data1 = data + i*sizeof(T);
        for( int j = i+1; j < n; j++ )
            std::swap( row[j], *(T*)(data1 + step*j) );
    }
}

typedef void (*TransposeFunc)( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz );
typedef void (*TransposeInplaceFunc)( uchar* data, size_t step, int n );

#define DEF_TRANSPOSE_FUNC(suffix, type) \
static void transpose_##suffix( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz ) \
{ transpose_<type>(src, sstep, dst, dstep, sz); } \
\
static void transposeI_##suffix( uchar* data, size_t step, int n ) \
{ transposeI_<type>(data, step, n); }

DEF_TRANSPOSE_FUNC(8u, uchar)
DEF_TRANSPOSE_FUNC(16u, ushort)
DEF_TRANSPOSE_FUNC(8uC3, Vec3b)
DEF_TRANSPOSE_FUNC(32s, int)
DEF_TRANSPOSE_FUNC(16uC3, Vec3s)
DEF_TRANSPOSE_FUNC(32sC2, Vec2i)
DEF_TRANSPOSE_FUNC(32sC3, Vec3i)
DEF_TRANSPOSE_FUNC(32sC4, Vec4i)
DEF_TRANSPOSE_FUNC(32sC6, Vec6i)
DEF_TRANSPOSE_FUNC(32sC8, Vec8i)

static TransposeFunc transposeTab[] =
{
    0, transpose_8u, transpose_16u, transpose_8uC3, transpose_32s, 0, transpose_16uC3, 0,
    transpose_32sC2, 0, 0, 0, transpose_32sC3, 0, 0, 0, transpose_32sC4,
    0, 0, 0, 0, 0, 0, 0, transpose_32sC6, 0, 0, 0, 0, 0, 0, 0, transpose_32sC8
};

static TransposeInplaceFunc transposeInplaceTab[] =
{
    0, transposeI_8u, transposeI_16u, transposeI_8uC3, transposeI_32s, 0, transposeI_16uC3, 0,
    transposeI_32sC2, 0, 0, 0, transposeI_32sC3, 0, 0, 0, transposeI_32sC4,
    0, 0, 0, 0, 0, 0, 0, transposeI_32sC6, 0, 0, 0, 0, 0, 0, 0, transposeI_32sC8
};

#ifdef HAVE_OPENCL

static bool ocl_transpose( InputArray _src, OutputArray _dst )
{
    const ocl::Device & dev = ocl::Device::getDefault();
    const int TILE_DIM = 32, BLOCK_ROWS = 8;
    int type = _src.type(), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type),
        rowsPerWI = dev.isIntel() ? 4 : 1;

    UMat src = _src.getUMat();
    _dst.create(src.cols, src.rows, type);
    UMat dst = _dst.getUMat();

    String kernelName("transpose");
    bool inplace = dst.u == src.u;

    if (inplace)
    {
        CV_Assert(dst.cols == dst.rows);
        kernelName += "_inplace";
    }
    else
    {
        // check required local memory size
        size_t required_local_memory = (size_t) TILE_DIM*(TILE_DIM+1)*CV_ELEM_SIZE(type);
        if (required_local_memory > ocl::Device::getDefault().localMemSize())
            return false;
    }

    String deviceMacro;
    if (dev.isIntel())
        deviceMacro = " -D INTEL_GPU";
    else
        deviceMacro = "";

    ocl::Kernel k(kernelName.c_str(), ocl::core::transpose_oclsrc,
                  format("-D T=%s -D T1=%s -D cn=%d -D TILE_DIM=%d -D BLOCK_ROWS=%d -D rowsPerWI=%d%s%s",
                         ocl::memopTypeToStr(type), ocl::memopTypeToStr(depth),
                         cn, TILE_DIM, BLOCK_ROWS, rowsPerWI, inplace ? " -D INPLACE" : "", deviceMacro.c_str()));
    if (k.empty())
        return false;

    if (inplace)
        k.args(ocl::KernelArg::ReadWriteNoSize(dst), dst.rows);
    else
        k.args(ocl::KernelArg::ReadOnly(src),
               ocl::KernelArg::WriteOnlyNoSize(dst));

    size_t localsize[2]  = { TILE_DIM, BLOCK_ROWS };
    size_t globalsize[2] = { (size_t)src.cols, inplace ? ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI : (divUp((size_t)src.rows, TILE_DIM) * BLOCK_ROWS) };

    if (inplace && dev.isIntel())
    {
        localsize[0] = 16;
        localsize[1] = dev.maxWorkGroupSize() / localsize[0];
    }

    return k.run(2, globalsize, localsize, false);
}

#endif

void transpose( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), esz = CV_ELEM_SIZE(type);
    CV_Assert( _src.dims() <= 2 && esz <= 32 );

    CV_OCL_RUN(_dst.isUMat(),
               ocl_transpose(_src, _dst))

    Mat src = _src.getMat();
    if( src.empty() )
    {
        _dst.release();
        return;
    }

    _dst.create(src.cols, src.rows, src.type());
    Mat dst = _dst.getMat();

    // handle the case of single-column/single-row matrices, stored in STL vectors.
    if( src.rows != dst.cols || src.cols != dst.rows )
    {
        CV_Assert( src.size() == dst.size() && (src.cols == 1 || src.rows == 1) );
        src.copyTo(dst);
        return;
    }

    CALL_HAL(transpose2d, cv_hal_transpose2d, src.data, src.step, dst.data, dst.step, src.cols, src.rows, esz);

    if( dst.data == src.data )
    {
        TransposeInplaceFunc func = transposeInplaceTab[esz];
        CV_Assert( func != 0 );
        CV_Assert( dst.cols == dst.rows );
        func( dst.ptr(), dst.step, dst.rows );
    }
    else
    {
        TransposeFunc func = transposeTab[esz];
        CV_Assert( func != 0 );
        func( src.ptr(), src.step, dst.ptr(), dst.step, src.size() );
    }
}


void transposeND(InputArray src_, const std::vector<int>& order, OutputArray dst_)
{
    Mat inp = src_.getMat();
    CV_Assert(inp.isContinuous());
    CV_CheckEQ(inp.channels(), 1, "Input array should be single-channel");
    CV_CheckEQ(order.size(), static_cast<size_t>(inp.dims), "Number of dimensions shouldn't change");

    auto order_ = order;
    std::sort(order_.begin(), order_.end());
    for (size_t i = 0; i < order_.size(); ++i)
    {
        CV_CheckEQ(static_cast<size_t>(order_[i]), i, "New order should be a valid permutation of the old one");
    }

    std::vector<int> newShape(order.size());
    for (size_t i = 0; i < order.size(); ++i)
    {
        newShape[i] = inp.size[order[i]];
    }

    dst_.create(static_cast<int>(newShape.size()), newShape.data(), inp.type());
    Mat out = dst_.getMat();
    CV_Assert(out.isContinuous());
    CV_Assert(inp.data != out.data);

    int continuous_idx = 0;
    for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i)
    {
        if (order[i] != i)
        {
            continuous_idx = i + 1;
            break;
        }
    }

    size_t continuous_size = continuous_idx == 0 ? out.total() : out.step1(continuous_idx - 1);
    size_t outer_size = out.total() / continuous_size;

    std::vector<size_t> steps(order.size());
    for (int i = 0; i < static_cast<int>(steps.size()); ++i)
    {
        steps[i] = inp.step1(order[i]);
    }

    auto* src = inp.ptr<const unsigned char>();
    auto* dst = out.ptr<unsigned char>();

    size_t src_offset = 0;
    size_t es = out.elemSize();
    for (size_t i = 0; i < outer_size; ++i)
    {
        std::memcpy(dst, src + es * src_offset, es * continuous_size);
        dst += es * continuous_size;
        for (int j = continuous_idx - 1; j >= 0; --j)
        {
            src_offset += steps[j];
            if ((src_offset / steps[j]) % out.size[j] != 0)
            {
                break;
            }
            src_offset -= steps[j] * out.size[j];
        }
    }
}


// Generic scalar fallback
CV_ALWAYS_INLINE void flipHoriz_generic( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz )
{
    int i, j, limit = (int)(((size.width + 1)/2)*esz);
    int height = size.height;
    AutoBuffer<int> _tab(size.width*esz);
    int* tab = _tab.data();

    for( i = 0; i < size.width; i++ )
        for( size_t k = 0; k < esz; k++ )
            tab[i*esz + k] = (int)((size.width - i - 1)*esz + k);

    for( ; height--; src += sstep, dst += dstep )
    {
        for( i = 0; i < limit; i++ )
        {
            j = tab[i];
            uchar t0 = src[i], t1 = src[j];
            dst[i] = t1; dst[j] = t0;
        }
    }
}

#if CV_SIMD || CV_SIMD_SCALABLE
template<typename V> CV_ALWAYS_INLINE void flipHoriz_single( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size )
{
    typedef typename VTraits<V>::lane_type T;
    const int vlanes = VTraits<v_uint8>::vlanes();
    int end = (int)(size.width * sizeof(T));
    int width = (end + 1) / 2;
    int width_simd = width & -vlanes;
    int height = size.height;

#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(T)>(src, dst));
#endif

    for( ; height--; src += sstep, dst += dstep )
    {
        int i = 0, j = end;
        for( ; i < width_simd; i += vlanes, j -= vlanes )
        {
            V t0 = vx_load((const T*)(src + i));
            V t1 = vx_load((const T*)(src + j - vlanes));
            t0 = v_reverse(t0);
            t1 = v_reverse(t1);
            v_store((T*)(dst + j - vlanes), t0);
            v_store((T*)(dst + i), t1);
        }

        // Scalar tail loop
        if (isAligned<sizeof(T)>(src, dst))
        {
            for ( ; i < width; i += sizeof(T), j -= sizeof(T) )
            {
                T t0 = *((const T*)(src + i));
                T t1 = *((const T*)(src + j - sizeof(T)));
                *((T*)(dst + j - sizeof(T))) = t0;
                *((T*)(dst + i)) = t1;
            }
        }
        else
        {
            for ( ; i < width; i += sizeof(T), j -= sizeof(T) )
            {
                for (int k = 0; k < (int)sizeof(T); k++)
                {
                    uchar t0 = src[i + k];
                    uchar t1 = src[j + k - sizeof(T)];
                    dst[j + k - sizeof(T)] = t0;
                    dst[i + k] = t1;
                }
            }
        }
    }
}

// SIMD for C3
template<typename V>
CV_ALWAYS_INLINE void flipHoriz_c3( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size )
{
    typedef typename VTraits<V>::lane_type T;
    const int vlanes = VTraits<V>::vlanes();
    const int stride = 3 * sizeof(T);
    int width = size.width;
    int centre = (width + 1) / 2;
    int width_simd = (centre / vlanes) * vlanes;
    int height = size.height;

    for( ; height--; src += sstep, dst += dstep )
    {
        int i = 0;
        for( ; i < width_simd; i += vlanes )
        {
            V r0, g0, b0;
            v_load_deinterleave((const T*)(src + i * stride), r0, g0, b0);
            V r1, g1, b1;
            v_load_deinterleave((const T*)(src + (width - i - vlanes) * stride), r1, g1, b1);
            r0 = v_reverse(r0);
            g0 = v_reverse(g0);
            b0 = v_reverse(b0);
            r1 = v_reverse(r1);
            g1 = v_reverse(g1);
            b1 = v_reverse(b1);
            v_store_interleave((T*)(dst + (width - i - vlanes) * stride), r0, g0, b0);
            v_store_interleave((T*)(dst + i * stride), r1, g1, b1);
        }
        // Scalar tail loop for remaining pixels
        for( ; i < centre; i++ )
        {
            int j = width - i - 1;
            T c0 = ((const T*)(src + i * stride))[0];
            T c1 = ((const T*)(src + i * stride))[1];
            T c2 = ((const T*)(src + i * stride))[2];
            T c3 = ((const T*)(src + j * stride))[0];
            T c4 = ((const T*)(src + j * stride))[1];
            T c5 = ((const T*)(src + j * stride))[2];
            ((T*)(dst + j * stride))[0] = c0;
            ((T*)(dst + j * stride))[1] = c1;
            ((T*)(dst + j * stride))[2] = c2;
            ((T*)(dst + i * stride))[0] = c3;
            ((T*)(dst + i * stride))[1] = c4;
            ((T*)(dst + i * stride))[2] = c5;
        }
    }
}

// SIMD flip when ESZ multiple of vlanes
template<size_t ESZ>
CV_ALWAYS_INLINE void flipHoriz_vlanes_match( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size)
{
    const int vlanes = VTraits<v_uint8>::vlanes();
    int end = (int)(size.width * ESZ);
    int width = end / 2;
    int height = size.height;
    int eSize = (int)ESZ;
    for( ; height--; src += sstep, dst += dstep )
    {
        for( int i = 0, j = end - eSize; i < width; i += eSize, j -= eSize )
        {
            for( int k = 0; k < eSize; k += vlanes )
            {
                v_uint8 t0 = vx_load(src + i + k);
                v_uint8 t1 = vx_load(src + j + k);
                v_store(dst + j + k, t0);
                v_store(dst + i + k, t1);
            }
        }
    }
}

#if CV_SIMD128
// SIMD flip when ESZ=16 (128-bit)
template<size_t ESZ>
CV_ALWAYS_INLINE void flipHoriz_vlanes_match_128( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size)
{
    const int vlanes16 = VTraits<v_uint8x16>::vlanes();
    int end = (int)(size.width * ESZ);
    int width = end / 2;
    int height = size.height;
    for( ; height--; src += sstep, dst += dstep )
    {
        for( int i = 0, j = end - vlanes16; i < width; i += vlanes16, j -= vlanes16 )
        {
            v_uint8x16 t0 = v_load(src + i);
            v_uint8x16 t1 = v_load(src + j);
            v_store(dst + j, t0);
            v_store(dst + i, t1);
        }
    }
}
#endif // CV_SIMD128

// SIMD flip for ESZ=16,32
template<size_t ESZ>
CV_ALWAYS_INLINE void flipHoriz_vlanes_dispatch( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size )
{
    const int vlanes = VTraits<v_uint8>::vlanes();
#if CV_SIMD128
    const int vlanes16 = VTraits<v_uint8x16>::vlanes();
#endif
    if ( (ESZ == (size_t)vlanes) || (ESZ == 2 * (size_t)vlanes))
    {
        flipHoriz_vlanes_match<ESZ>(src, sstep, dst, dstep, size);
        return;
    }
#if CV_SIMD128
    else if (ESZ == vlanes16)
    {
        flipHoriz_vlanes_match_128<ESZ>(src, sstep, dst, dstep, size);
        return;
    }
#endif
    flipHoriz_generic(src, sstep, dst, dstep, size, ESZ);
}

#if CV_SIMD128
// SIMD flip for ESZ=24 (CV_64FC3)
CV_ALWAYS_INLINE void flipHoriz_24( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size )
{
#if CV_STRONG_ALIGNMENT
    // This kernel performs 64-bit scalar loads/stores, so require 8-byte alignment.
    if (!isAligned<8>(((size_t)src)|((size_t)dst)|sstep|dstep))
    {
        flipHoriz_generic(src, sstep, dst, dstep, size, 24);
        return;
    }
#endif
    const int lanes16 = 16;
    int end = (int)(size.width * 24);
    int width = (end + 1) / 2;
    int height = size.height;
    for( ; height--; src += sstep, dst += dstep )
    {
        for ( int i = 0, j = end; i < width; i += lanes16 + 8, j -= lanes16 + 8 )
        {
            v_uint8x16 t0 = v_load(src + i);
            uint64_t t2 = *reinterpret_cast<const uint64_t*>(src + i + lanes16);
            v_uint8x16 t1 = v_load(src + j - lanes16 - 8);
            uint64_t t3 = *reinterpret_cast<const uint64_t*>(src + j - 8);
            v_store(dst + j - lanes16 - 8, t0);
            *reinterpret_cast<uint64_t*>(dst + j - 8) = t2;
            v_store(dst + i, t1);
            *reinterpret_cast<uint64_t*>(dst + i + lanes16) = t3;
        }
    }
}
#endif // CV_SIMD128
#endif // CV_SIMD || CV_SIMD_SCALABLE

static void flipHoriz( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz )
{
#if CV_SIMD || CV_SIMD_SCALABLE
    // SIMD-optimized dispatch
    switch(esz)
    {
        case 1:   flipHoriz_single<v_uint8>(src, sstep, dst, dstep, size); return;            // CV_8UC1: 8-bit, 1 channel
        case 2:   flipHoriz_single<v_uint16>(src, sstep, dst, dstep, size); return;           // CV_8UC2, CV_16UC1: 8-bit 2-channel or 16-bit 1-channel
        case 3:   flipHoriz_c3<v_uint8>(src, sstep, dst, dstep, size); return;                // CV_8UC3: 8-bit, 3 channels
        case 4:   flipHoriz_single<v_uint32>(src, sstep, dst, dstep, size); return;           // CV_8UC4, CV_16UC2, CV_32SC1, CV_32FC1: 8-bit 4-channel, 16-bit 2-channel, or 32-bit 1-channel
        case 6:   flipHoriz_c3<v_uint16>(src, sstep, dst, dstep, size); return;               // CV_16UC3, CV_16SC3: 16-bit, 3 channels
        case 8:   flipHoriz_single<v_uint64>(src, sstep, dst, dstep, size); return;           // CV_16UC4, CV_32SC2, CV_32FC2, CV_64FC1: 16-bit 4-channel, 32-bit 2-channel, or 64-bit 1-channel
        case 12:  flipHoriz_c3<v_uint32>(src, sstep, dst, dstep, size); return;               // CV_32SC3, CV_32FC3: 32-bit, 3 channels
        case 16:  flipHoriz_vlanes_dispatch<16>(src, sstep, dst, dstep, size); return;        // CV_32SC4, CV_32FC4, CV_64FC2: 32-bit 4-channel or 64-bit 2-channel
#if CV_SIMD128
        case 24:  flipHoriz_24(src, sstep, dst, dstep, size); return;                         // CV_64FC3: 64-bit, 3 channels
#endif
        case 32:  flipHoriz_vlanes_dispatch<32>(src, sstep, dst, dstep, size); return;        // CV_64FC4: 64-bit, 4 channels
        default:
            break; // Fall through to generic implementation
    }
#endif
    // Fallback: generic scalar
    flipHoriz_generic(src, sstep, dst, dstep, size, esz);
}

static void
flipVert( const uchar* src0, size_t sstep, uchar* dst0, size_t dstep, Size size, size_t esz )
{
    const uchar* src1 = src0 + (size.height - 1)*sstep;
    uchar* dst1 = dst0 + (size.height - 1)*dstep;
    size.width *= (int)esz;

    for( int y = 0; y < (size.height + 1)/2; y++, src0 += sstep, src1 -= sstep,
                                                  dst0 += dstep, dst1 -= dstep )
    {
        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_STRONG_ALIGNMENT
        if (isAligned<sizeof(int)>(src0, src1, dst0, dst1))
#endif
        {
            for (; i <= size.width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes())
            {
                v_int32 t0 = v_reinterpret_as_s32(vx_load(src0 + i));
                v_int32 t1 = v_reinterpret_as_s32(vx_load(src1 + i));
                v_store(dst0 + i, v_reinterpret_as_u8(t1));
                v_store(dst1 + i, v_reinterpret_as_u8(t0));
            }
        }
#if CV_STRONG_ALIGNMENT
        else
        {
            for (; i <= size.width - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes())
            {
                v_uint8 t0 = vx_load(src0 + i);
                v_uint8 t1 = vx_load(src1 + i);
                v_store(dst0 + i, t1);
                v_store(dst1 + i, t0);
            }
        }
#endif
#endif

        if (isAligned<sizeof(int)>(src0, src1, dst0, dst1))
        {
            for( ; i <= size.width - 16; i += 16 )
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;

                t0 = ((int*)(src0 + i))[1];
                t1 = ((int*)(src1 + i))[1];

                ((int*)(dst0 + i))[1] = t1;
                ((int*)(dst1 + i))[1] = t0;

                t0 = ((int*)(src0 + i))[2];
                t1 = ((int*)(src1 + i))[2];

                ((int*)(dst0 + i))[2] = t1;
                ((int*)(dst1 + i))[2] = t0;

                t0 = ((int*)(src0 + i))[3];
                t1 = ((int*)(src1 + i))[3];

                ((int*)(dst0 + i))[3] = t1;
                ((int*)(dst1 + i))[3] = t0;
            }

            for( ; i <= size.width - 4; i += 4 )
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;
            }
        }

        for( ; i < size.width; i++ )
        {
            uchar t0 = src0[i];
            uchar t1 = src1[i];

            dst0[i] = t1;
            dst1[i] = t0;
        }
    }
}

#ifdef HAVE_OPENCL

enum { FLIP_COLS = 1 << 0, FLIP_ROWS = 1 << 1, FLIP_BOTH = FLIP_ROWS | FLIP_COLS };

static bool ocl_flip(InputArray _src, OutputArray _dst, int flipCode )
{
    CV_Assert(flipCode >= -1 && flipCode <= 1);

    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            flipType, kercn = std::min(ocl::predictOptimalVectorWidth(_src, _dst), 4);

    bool doubleSupport = dev.doubleFPConfig() > 0;
    if (!doubleSupport && depth == CV_64F)
        kercn = cn;

    if (cn > 4)
        return false;

    const char * kernelName;
    if (flipCode == 0)
        kernelName = "arithm_flip_rows", flipType = FLIP_ROWS;
    else if (flipCode > 0)
        kernelName = "arithm_flip_cols", flipType = FLIP_COLS;
    else
        kernelName = "arithm_flip_rows_cols", flipType = FLIP_BOTH;

    int pxPerWIy = (dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU)) ? 4 : 1;
    kercn = (cn!=3 || flipType == FLIP_ROWS) ? std::max(kercn, cn) : cn;

    ocl::Kernel k(kernelName, ocl::core::flip_oclsrc,
        format( "-D T=%s -D T1=%s -D DEPTH=%d -D cn=%d -D PIX_PER_WI_Y=%d -D kercn=%d",
                kercn != cn ? ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)) : ocl::vecopTypeToStr(CV_MAKE_TYPE(depth, kercn)),
                kercn != cn ? ocl::typeToStr(depth) : ocl::vecopTypeToStr(depth), depth, cn, pxPerWIy, kercn));
    if (k.empty())
        return false;

    Size size = _src.size();
    _dst.create(size, type);
    UMat src = _src.getUMat(), dst = _dst.getUMat();

    int cols = size.width * cn / kercn, rows = size.height;
    cols = flipType == FLIP_COLS ? (cols + 1) >> 1 : cols;
    rows = flipType & FLIP_ROWS ? (rows + 1) >> 1 : rows;

    k.args(ocl::KernelArg::ReadOnlyNoSize(src),
           ocl::KernelArg::WriteOnly(dst, cn, kercn), rows, cols);

    size_t maxWorkGroupSize = dev.maxWorkGroupSize();
    CV_Assert(maxWorkGroupSize % 4 == 0);

    size_t globalsize[2] = { (size_t)cols, ((size_t)rows + pxPerWIy - 1) / pxPerWIy },
            localsize[2] = { maxWorkGroupSize / 4, 4 };
    return k.run(2, globalsize, (flipType == FLIP_COLS) && !dev.isIntel() ? localsize : NULL, false);
}

#endif

void flip( InputArray _src, OutputArray _dst, int flip_mode )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( _src.dims() <= 2 );
    Size size = _src.size();

    if (flip_mode < 0)
    {
        if (size.width == 1)
            flip_mode = 0;
        if (size.height == 1)
            flip_mode = 1;
    }

    if ((size.width == 1 && flip_mode > 0) ||
        (size.height == 1 && flip_mode == 0))
    {
        return _src.copyTo(_dst);
    }

    CV_OCL_RUN( _dst.isUMat(), ocl_flip(_src, _dst, flip_mode))

    Mat src = _src.getMat();
    int type = src.type();
    _dst.create( size, type );
    Mat dst = _dst.getMat();

    CALL_HAL(flip, cv_hal_flip, type, src.ptr(), src.step, src.cols, src.rows,
             dst.ptr(), dst.step, flip_mode);

    size_t esz = CV_ELEM_SIZE(type);

    if( flip_mode <= 0 )
        flipVert( src.ptr(), src.step, dst.ptr(), dst.step, src.size(), esz );
    else
        flipHoriz( src.ptr(), src.step, dst.ptr(), dst.step, src.size(), esz );

    if( flip_mode < 0 )
        flipHoriz( dst.ptr(), dst.step, dst.ptr(), dst.step, dst.size(), esz );
}

static void
flipNDImpl(uchar* data, const int* shape, const size_t* step, int axis)
{
    int total = 1;
    for (int i = 0; i < axis; ++i)
        total *= shape[i];

    int shape_at_axis = shape[axis];
    size_t step_at_axis = step[axis];
    size_t offset = 0;
    size_t offset_increment = axis == 0 ? 0 : step[axis - 1];
    for (int i = 0; i < total; ++i, offset += offset_increment)
        for (int j = 0, k = shape_at_axis - 1; j < shape_at_axis / 2; ++j, --k)
            std::swap_ranges(data + offset + j * step_at_axis,
                             data + offset + j * step_at_axis + step_at_axis,
                             data + offset + k * step_at_axis);
}

void flipND(InputArray _src, OutputArray _dst, int _axis)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();

    // verify axis
    int ndim = src.dims;
    CV_CheckLT(_axis, ndim, "flipND: given axis is out of range");
    CV_CheckGE(_axis, -ndim, "flipND: given axis is out of range");
    int axis = (_axis + ndim) % ndim;

    // in-place flip
    _src.copyTo(_dst);

    // return the src if it has only one element on the flip axis
    const auto shape = src.size.p;
    if (shape[axis] == 1)
        return ;

    // call impl
    Mat dst = _dst.getMat();
    flipNDImpl(dst.ptr(), dst.size.p, dst.step.p, axis);
}

/*
    This function first prepends 1 to each tensor shape to have a common max_ndims dimension, then flatten non-broadcast dimensions.
*/
static bool _flatten_for_broadcast(int narrays, int max_ndims, const int* ndims, const int** orig_shape,
                                   int** flatten_shape, size_t** flatten_step) {
    int i, j, k;

    // step 1.
    // * make all inputs and the output max_ndims-dimensional.
    // * compute proper step's
    for (i = max_ndims - 1; i >= 0; i-- ) {
        for (k = 0; k < narrays; k++) {
            j = ndims[k] - (max_ndims - i);
            int sz_i = j >= 0 ? orig_shape[k][j] : 1;
            size_t st_i = i == max_ndims - 1 ? 1 : flatten_step[k][i+1] * flatten_shape[k][i+1];
            flatten_shape[k][i] = sz_i;
            flatten_step[k][i] = st_i;
            if (flatten_shape[k][i] == 0)
                return false;
        }
    }

    // step 2. Let's do the flattening first,
    // since we'd need proper values of steps to check continuity.
    // this loop is probably the most tricky part
    // in the whole implementation of broadcasting.
    j = max_ndims-1;
    for (i = j - 1; i >= 0; i--) {
        bool all_contiguous = true, all_scalars = true, all_consistent = true;
        for(k = 0; k < narrays; k++) {
            size_t st = flatten_step[k][j] * flatten_shape[k][j];
            bool prev_scalar = flatten_shape[k][j] == 1;
            bool scalar = flatten_shape[k][i] == 1;
            all_contiguous = all_contiguous && (st == flatten_step[k][i]);
            all_scalars = all_scalars && scalar;
            all_consistent = all_consistent && (scalar == prev_scalar);
        }
        if (all_contiguous && (all_consistent || all_scalars)) {
            for(k = 0; k < narrays; k++)
                flatten_shape[k][j] *= flatten_shape[k][i];
        } else {
            j--;
            if (i < j) {
                for(k = 0; k < narrays; k++) {
                    flatten_shape[k][j] = flatten_shape[k][i];
                    flatten_step[k][j] = flatten_step[k][i];
                }
            }
        }
    }

    // step 3. Set some step's to 0's.
    for (i = max_ndims-1; i >= j; i--) {
        for (k = 0; k < narrays; k++)
            flatten_step[k][i] = flatten_shape[k][i] == 1 ? 0 : flatten_step[k][i];
    }
    for (; i >= 0; i--) {
        for (k = 0; k < narrays; k++) {
            flatten_step[k][i] = 0;
            flatten_shape[k][i] = 1;
        }
    }
    return true;
}

void broadcast(InputArray _src, InputArray _shape, OutputArray _dst) {
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    CV_CheckTrue(src.isContinuous(), "broadcast: input array must be contiguous");
    CV_CheckChannelsEQ(src.channels(), 1, "broadcast: input array must be single channel");

    Mat shape = _shape.getMat();
    CV_CheckTypeEQ(shape.type(), CV_32S, "broadcast: target shape must be of type int32");
    const auto dims_shape = static_cast<int>(shape.total());
    const auto *ptr_shape = shape.ptr<int>();

    // check valid shape, 1D/0D Mat would fail in the following checks
    const auto dims_src = src.dims;
    CV_CheckLE(dims_src, dims_shape,
               "broadcast: dimension of input array must be less than or equal to dimension of target shape");
    std::vector<int> shape_src{src.size.p, src.size.p + dims_src};
    if (shape_src.size() < static_cast<size_t>(dims_shape)) {
        shape_src.insert(shape_src.begin(), dims_shape - shape_src.size(), 1);
    }
    for (int i = 0; i < static_cast<int>(shape_src.size()); ++i) {
        const auto *shape_target = ptr_shape;
        if (shape_src[i] != 1) {
            CV_CheckEQ(shape_src[i], shape_target[i], "target shape must be equal to input shape or 1");
        }
    }

    // impl
    _dst.create(dims_shape, shape.ptr<int>(), src.type());
    Mat dst = _dst.getMat();
    if (dst.total() == 0)
        return;
    std::vector<int> is_same_shape(dims_shape, 0);
    for (int i = 0; i < static_cast<int>(shape_src.size()); ++i) {
        if (shape_src[i] == ptr_shape[i]) {
            is_same_shape[i] = 1;
        }
    }
    // copy if same shape
    if (std::accumulate(is_same_shape.begin(), is_same_shape.end(), 1, std::multiplies<int>()) != 0) {
        const auto *p_src = src.ptr<const char>();
        auto *p_dst = dst.ptr<char>();
        std::memcpy(p_dst, p_src, dst.total() * dst.elemSize());
        return;
    }
    // other cases
    int max_ndims = std::max(dims_src, dims_shape);
    if (max_ndims < 2 && src.total() == 1) {
        const char* p_src = src.ptr<const char>();
        char* p_dst = dst.ptr<char>();
        size_t esz = src.elemSize();
        for (size_t j = 0; j < dst.total(); j++)
            std::memcpy(p_dst + j * esz, p_src, esz);
        return;
    }
    const int all_ndims[2] = {src.dims, dst.dims};
    const int* orig_shapes[2] = {src.size.p, dst.size.p};
    cv::AutoBuffer<size_t> buff(max_ndims * 4);
    int* flatten_shapes[2] = {(int*)buff.data(), (int*)(buff.data() + max_ndims)};
    size_t* flatten_steps[2] = {(size_t*)(buff.data() + 2 * max_ndims), (size_t*)(buff.data() + 3 * max_ndims)};
    if (_flatten_for_broadcast(2, max_ndims, all_ndims, orig_shapes, flatten_shapes, flatten_steps)) {
        size_t src_dp = flatten_steps[0][max_ndims - 1];
        size_t dst_dp = flatten_steps[1][max_ndims - 1];
        CV_Assert(dst_dp == 1 || dst_dp == 0);
        CV_Assert(max_ndims >= 2); // >= 3?
        size_t rowstep_src = flatten_steps[0][max_ndims - 2];
        size_t rowstep_dst = flatten_steps[1][max_ndims - 2];
        const char* ptr_src = src.ptr<const char>();
        char* ptr_dst = dst.ptr<char>();
        size_t esz = src.elemSize();
        int nrows = flatten_shapes[1][max_ndims - 2];
        int ncols = flatten_shapes[1][max_ndims - 1];
        int nplanes = 1;
        CV_Check(esz, esz == 1 || esz == 2 || esz == 4 || esz == 8, "broadcast: not supported data type");

        for (int k = 0; k < max_ndims - 2; k++) {
            nplanes *= flatten_shapes[1][k];
        }
        for (int plane_idx = 0; plane_idx < nplanes; plane_idx++) {
            size_t offset_src = 0, offset_dst = 0;
            size_t idx = (size_t)plane_idx;
            for (int k = max_ndims - 3; k >= 0; k--) {
                size_t prev_idx = idx / flatten_shapes[1][k];
                size_t i_k = (int)(idx - prev_idx * flatten_shapes[1][k]);
                offset_src += i_k * flatten_steps[0][k];
                offset_dst += i_k * flatten_steps[1][k];
                idx = prev_idx;
            }

            #define OPENCV_CORE_BROADCAST_LOOP(_Tp) \
                for (int i = 0; i < nrows; i++) {   \
                    const _Tp *ptr_src_ = (const _Tp*)ptr_src + offset_src + rowstep_src * i; \
                    _Tp *ptr_dst_ = (_Tp*)ptr_dst + offset_dst + rowstep_dst * i; \
                    if (src_dp == 1) { \
                        for (int j = 0; j < ncols; j++) { \
                            ptr_dst_[j] = ptr_src_[j]; \
                        } \
                    } else { \
                        _Tp x = *ptr_src_; \
                        for (int j = 0; j < ncols; j++) { \
                            ptr_dst_[j] = x; \
                        } \
                    } \
                }

            if (esz == 1) {
                OPENCV_CORE_BROADCAST_LOOP(int8_t);
            } else if (esz == 2) {
                OPENCV_CORE_BROADCAST_LOOP(int16_t);
            } else if (esz == 4) {
                OPENCV_CORE_BROADCAST_LOOP(int32_t);
            } else if (esz == 8) {
                OPENCV_CORE_BROADCAST_LOOP(int64_t);
            } else {
                CV_Error(cv::Error::StsNotImplemented, "");
            }
            #undef OPENCV_CORE_BROADCAST_LOOP
        }
    } else {
        // initial copy (src to dst)
        std::vector<size_t> step_src{src.step.p, src.step.p + dims_src};
        if (step_src.size() < static_cast<size_t>(dims_shape)) {
            step_src.insert(step_src.begin(), dims_shape - step_src.size(), step_src[0]);
        }
        for (size_t i = 0; i < src.total(); ++i) {
            size_t t = i;
            size_t src_offset = 0, dst_offset = 0;
            for (int j = static_cast<int>(shape_src.size() - 1); j >= 0; --j) {
                size_t idx = t / shape_src[j];
                size_t offset = static_cast<size_t>(t - idx * shape_src[j]);
                src_offset += offset * step_src[j];
                dst_offset += offset * dst.step[j];
                t = idx;
            }
            const auto *p_src = src.ptr<const char>();
            auto *p_dst = dst.ptr<char>();
            std::memcpy(p_dst + dst_offset, p_src + src_offset, dst.elemSize());
        }
        // broadcast copy (dst inplace)
        std::vector<int> cumulative_shape(dims_shape, 1);
        int total = static_cast<int>(dst.total());
        for (int i = dims_shape - 1; i >= 0; --i) {
            cumulative_shape[i] = static_cast<int>(total / ptr_shape[i]);
            total = cumulative_shape[i];
        }
        for (int i = dims_shape - 1; i >= 0; --i) {
            if (is_same_shape[i] == 1) {
                continue;
            }
            auto step = dst.step[i];
            auto *p_dst = dst.ptr<char>();
            for (int j = 0; j < cumulative_shape[i]; j++) {
                for (int k = 0; k < ptr_shape[i] - 1; k++) {
                    std::memcpy(p_dst + step, p_dst, step);
                    p_dst += step;
                }
                p_dst += step;
            }
        }
    }
}

void broadcast(InputArray _src, const MatShape& _shape, OutputArray _dst)
{
    if (_shape.dims < 0) {
        _dst.release();
    } else {
        Mat shape(1, _shape.dims, CV_32S, (int*)_shape.p);
        broadcast(_src, shape, _dst);
    }
}

static void rotateImpl(InputArray _src, OutputArray _dst, int rotateMode)
{
    switch (rotateMode)
    {
    case ROTATE_90_CLOCKWISE:
        transpose(_src, _dst);
        flip(_dst, _dst, 1);
        break;
    case ROTATE_180:
        flip(_src, _dst, -1);
        break;
    case ROTATE_90_COUNTERCLOCKWISE:
        transpose(_src, _dst);
        flip(_dst, _dst, 0);
        break;
    default:
        break;
    }
}

void rotate(InputArray _src, OutputArray _dst, int rotateMode)
{
    CV_Assert(_src.dims() <= 2);
    int angle;

    if (_dst.isUMat())
    {
        rotateImpl(_src, _dst, rotateMode);
        return;
    }

    Mat src = _src.getMat();
    int type = src.type();
    if( src.empty() )
    {
        _dst.release();
        return;
    }

    switch (rotateMode)
    {
    case ROTATE_90_CLOCKWISE:
        _dst.create(src.cols, src.rows, type);
        angle = 90;
        break;
    case ROTATE_180:
        _dst.create(src.rows, src.cols, type);
        angle = 180;
        break;
    case ROTATE_90_COUNTERCLOCKWISE:
        _dst.create(src.cols, src.rows, type);
        angle = 270;
        break;
    default:
        _dst.create(src.rows, src.cols, type);
        angle = 0;
        break;
    }

    Mat dst = _dst.getMat();
    CALL_HAL(rotate90, cv_hal_rotate90, type, src.ptr(), src.step, src.cols, src.rows,
             dst.ptr(), dst.step, angle);

    // use src (Mat) since _src (InputArray) is updated by _dst.create() when in-place
    rotateImpl(src, _dst, rotateMode);
}

}  // namespace
