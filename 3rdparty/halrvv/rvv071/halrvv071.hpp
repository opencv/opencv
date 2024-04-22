/*
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2016, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 */

#ifndef OPENCV_HALRVV071_HAL_HPP_INCLUDED
#define OPENCV_HALRVV071_HAL_HPP_INCLUDED

#if __GNUC__ > 8 || \
    (__GNUC__ == 8 && (__GNUC_MINOR__ > 4 || \
                       (__GNUC_MINOR__ == 4 && \
                        __GNUC_PATCHLEVEL__ > 0)))

#include "opencv2/core/hal/interface.h"

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic == 7001
#include <riscv_vector.h>

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR cvt_hal_BGRtoBGR

static unsigned char index_array_32 [32] 
                        { 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 18, 17, 16, 19, 22, 21, 20, 23, 26, 25, 24, 27, 30, 29, 28, 31  };

static unsigned char index_array_24 [24]
                        { 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21  };

static void cvt_vector(const uchar* src, uchar * dst, uchar * index, int n, int scn, int dcn, int bi, int vsize_pixels, int vsize)
{
    vuint8m2_t vec_index = vle8_v_u8m2(index, vsize);

    int i = 0;

    for( ; i <= n-vsize; i += vsize_pixels, src += vsize, dst += vsize)
    {
        vuint8m2_t vec_src = vle8_v_u8m2(src, vsize);
        vuint8m2_t vec_dst = vrgather_vv_u8m2(vec_src, vec_index, vsize);
        vse8_v_u8m2(dst, vec_dst, vsize);
    }

    for ( ; i < n; i++, src += scn, dst += dcn )
    {
        uchar t0 = src[0], t1 = src[1], t2 = src[2];
        dst[bi  ] = t0;
        dst[1]    = t1;
        dst[bi^2] = t2;
        if(dcn == 4)
        {
            uchar d = scn == 4 ? src[3] : UCHAR_MAX;
            dst[3] = d;
        }
    }
}

static void cvt_scalar(const uchar* src, uchar * dst, int n, int scn, int dcn, int bi)
{
    for (int i = 0; i < n; i++, src += scn, dst += dcn)
    {
        uchar t0 = src[0], t1 = src[1], t2 = src[2];
        dst[bi  ] = t0;
        dst[1]    = t1;
        dst[bi^2] = t2;
        if(dcn == 4)
        {
            uchar d = scn == 4 ? src[3] : UCHAR_MAX;
            dst[3] = d;
        }
    }
}

static int cvt_hal_BGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue)
{
    int blueIdx = swapBlue ? 2 : 0;
    if (scn == dcn)
    {   
        const int vsize_pixels = 8;
        const int vsize = vsize_pixels*scn;
    
        unsigned char* index;
        if (scn == 4)
        {
            index = index_array_32;
        }
        else
        {
            index = index_array_24;
        }

        size_t vl = vsetvl_e8m2(vsize);

        for(int i = 0; i < height; i++, src_data += src_step, dst_data += dst_step)
            cvt_vector(src_data, dst_data, index, width, scn, dcn, blueIdx, vsize_pixels, vsize);
    }
    else
    {
        for(int i = 0; i < height; i++, src_data += src_step, dst_data += dst_step)
            cvt_scalar(src_data, dst_data, width, scn, dcn, blueIdx);
    }

    return CV_HAL_ERROR_OK;
}

#endif /* RVV 0.7.1 status */
#endif /* GCC version check*/

#endif