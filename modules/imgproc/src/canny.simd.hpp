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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <deque>

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

int canny_simd_width();

void canny_calc_magnitude(const short* _dx, const short* _dy, int* _mag_n,
                          int width, bool L2gradient);

void canny_nms_row(const int* _mag_a, const int* _mag_p, const int* _mag_n,
                   const short* _dx, const short* _dy, uchar* _pmap,
                   int width, int low, int high, std::deque<uchar*>& stack);

void canny_finalize_row(const uchar* pmap, uchar* pdst, int width);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#define CANNY_PUSH(map, stack) *map = 2, stack.push_back(map)

#define CANNY_CHECK(m, high, map, stack) \
    if (m > high) \
        CANNY_PUSH(map, stack); \
    else \
        *map = 0

int canny_simd_width()
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    return VTraits<v_uint8>::vlanes();
#else
    return 1;
#endif
}

// stage 1: magnitude of the gradient for one row (L1 = |dx|+|dy|, L2 = dx*dx+dy*dy).
// width == src.cols * cn.
void canny_calc_magnitude(const short* _dx, const short* _dy, int* _mag_n,
                          int width, bool L2gradient)
{
    int j = 0;
    if (L2gradient)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for ( ; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
        {
            v_int16 v_dx = vx_load((const short*)(_dx + j));
            v_int16 v_dy = vx_load((const short*)(_dy + j));

            v_int32 v_dxp_low, v_dxp_high;
            v_int32 v_dyp_low, v_dyp_high;
            v_expand(v_dx, v_dxp_low, v_dxp_high);
            v_expand(v_dy, v_dyp_low, v_dyp_high);

            v_store_aligned((int *)(_mag_n + j), v_add(v_mul(v_dxp_low, v_dxp_low), v_mul(v_dyp_low, v_dyp_low)));
            v_store_aligned((int *)(_mag_n + j + VTraits<v_int32>::vlanes()), v_add(v_mul(v_dxp_high, v_dxp_high), v_mul(v_dyp_high, v_dyp_high)));
        }
#endif
        for ( ; j < width; ++j)
            _mag_n[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
    }
    else
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for(; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
        {
            v_int16 v_dx = vx_load((const short *)(_dx + j));
            v_int16 v_dy = vx_load((const short *)(_dy + j));

            v_dx = v_reinterpret_as_s16(v_abs(v_dx));
            v_dy = v_reinterpret_as_s16(v_abs(v_dy));

            v_int32 v_dx_ml, v_dy_ml, v_dx_mh, v_dy_mh;
            v_expand(v_dx, v_dx_ml, v_dx_mh);
            v_expand(v_dy, v_dy_ml, v_dy_mh);

            v_store_aligned((int *)(_mag_n + j), v_add(v_dx_ml, v_dy_ml));
            v_store_aligned((int *)(_mag_n + j + VTraits<v_int32>::vlanes()), v_add(v_dx_mh, v_dy_mh));
        }
#endif
        for ( ; j < width; ++j)
            _mag_n[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
    }
}

// stage 2: non-maxima suppression + double thresholding for one row.
// Fills _pmap with: 0 (maybe edge), 1 (not edge), 2 (strong edge). Strong edges
// are pushed onto `stack` for the subsequent hysteresis pass.
void canny_nms_row(const int* _mag_a, const int* _mag_p, const int* _mag_n,
                   const short* _dx, const short* _dy, uchar* _pmap,
                   int width, int low, int high, std::deque<uchar*>& stack)
{
    const int TG22 = 13573;
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    // smask[i] selects, after a v_scan_forward hit at lane l, all lanes >= l so
    // that the already-handled lanes are cleared from the candidate mask.
    schar smask[2 * VTraits<v_int8>::max_nlanes];
    for (int i = 0; i < VTraits<v_int8>::vlanes(); ++i)
    {
        smask[i] = 0;
        smask[i + VTraits<v_int8>::vlanes()] = (schar)-1;
    }

    const v_int32 v_low = vx_setall_s32(low);
    const v_int8 v_one = vx_setall_s8(1);

    for (; j <= width - VTraits<v_int8>::vlanes(); j += VTraits<v_int8>::vlanes())
    {
        v_store_aligned((signed char*)(_pmap + j), v_one);
        v_int8 v_cmp = v_pack(v_pack(v_gt(vx_load_aligned((const int *)(_mag_a + j)), v_low),
                                     v_gt(vx_load_aligned((const int *)(_mag_a + j + VTraits<v_int32>::vlanes())), v_low)),
                              v_pack(v_gt(vx_load_aligned((const int *)(_mag_a + j + 2 * VTraits<v_int32>::vlanes())), v_low),
                                     v_gt(vx_load_aligned((const int *)(_mag_a + j + 3 * VTraits<v_int32>::vlanes())), v_low)));
        while (v_check_any(v_cmp))
        {
            int l = v_scan_forward(v_cmp);
            v_cmp = v_and(v_cmp, vx_load(smask + VTraits<v_int8>::vlanes() - 1 - l));
            int k = j + l;

            int m = _mag_a[k];
            short xs = _dx[k];
            short ys = _dy[k];
            int x = (int)std::abs(xs);
            int y = (int)std::abs(ys) << 15;

            int tg22x = x * TG22;

            if (y < tg22x)
            {
                if (m > _mag_a[k - 1] && m >= _mag_a[k + 1])
                {
                    CANNY_CHECK(m, high, (_pmap+k), stack);
                }
            }
            else
            {
                int tg67x = tg22x + (x << 16);
                if (y > tg67x)
                {
                    if (m > _mag_p[k] && m >= _mag_n[k])
                    {
                        CANNY_CHECK(m, high, (_pmap+k), stack);
                    }
                }
                else
                {
                    int s = (xs ^ ys) < 0 ? -1 : 1;
                    if(m > _mag_p[k - s] && m > _mag_n[k + s])
                    {
                        CANNY_CHECK(m, high, (_pmap+k), stack);
                    }
                }
            }
        }
    }
#endif
    for (; j < width; j++)
    {
        int m = _mag_a[j];

        if (m > low)
        {
            short xs = _dx[j];
            short ys = _dy[j];
            int x = (int)std::abs(xs);
            int y = (int)std::abs(ys) << 15;

            int tg22x = x * TG22;

            if (y < tg22x)
            {
                if (m > _mag_a[j - 1] && m >= _mag_a[j + 1])
                {
                    CANNY_CHECK(m, high, (_pmap+j), stack);
                    continue;
                }
            }
            else
            {
                int tg67x = tg22x + (x << 16);
                if (y > tg67x)
                {
                    if (m > _mag_p[j] && m >= _mag_n[j])
                    {
                        CANNY_CHECK(m, high, (_pmap+j), stack);
                        continue;
                    }
                }
                else
                {
                    int s = (xs ^ ys) < 0 ? -1 : 1;
                    if(m > _mag_p[j - s] && m > _mag_n[j + s])
                    {
                        CANNY_CHECK(m, high, (_pmap+j), stack);
                        continue;
                    }
                }
            }
        }
        _pmap[j] = 1;
    }
}

// final pass: turn the edge map (value 2 == edge) into a 0/255 output row.
void canny_finalize_row(const uchar* pmap, uchar* pdst, int width)
{
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    {
        const v_uint8 v_zero = vx_setzero_u8();
        const v_uint8 v_ff = v_not(v_zero);
        const v_uint8 v_two = vx_setall_u8(2);

        for (; j <= width - VTraits<v_uint8>::vlanes(); j += VTraits<v_uint8>::vlanes())
        {
            v_uint8 v_pmap = vx_load_aligned((const unsigned char*)(pmap + j));
            v_pmap = v_select(v_eq(v_pmap, v_two), v_ff, v_zero);
            v_store((pdst + j), v_pmap);
        }

        if (j <= width - VTraits<v_uint8>::vlanes()/2)
        {
            v_uint8 v_pmap = vx_load_low((const unsigned char*)(pmap + j));
            v_pmap = v_select(v_eq(v_pmap, v_two), v_ff, v_zero);
            v_store_low((pdst + j), v_pmap);
            j += VTraits<v_uint8>::vlanes()/2;
        }
    }
#endif
    for (; j < width; j++)
    {
        pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
