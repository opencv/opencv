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
 * Copyright (C) 2012-2015, NVIDIA Corporation, all rights reserved.
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


/* This is FAST corner detector, contributed to OpenCV by the author, Edward Rosten.
   Below is the original copyright and the references */

/*
Copyright (c) 2006, 2008 Edward Rosten
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 *Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

 *Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

 *Neither the name of the University of Cambridge nor the names of
  its contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
The references are:
 * Machine learning for high-speed corner detection,
   E. Rosten and T. Drummond, ECCV 2006
 * Faster and better: A machine learning approach to corner detection
   E. Rosten, R. Porter and T. Drummond, PAMI, 2009
*/

#include "common.hpp"

#include <vector>
#include <cstring>

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON
namespace
{

void makeOffsets(ptrdiff_t pixel[], ptrdiff_t row_stride)
{
    pixel[0] = 0 + row_stride * 3;
    pixel[1] = 1 + row_stride * 3;
    pixel[2] = 2 + row_stride * 2;
    pixel[3] = 3 + row_stride * 1;
    pixel[4] = 3 + row_stride * 0;
    pixel[5] = 3 + row_stride * -1;
    pixel[6] = 2 + row_stride * -2;
    pixel[7] = 1 + row_stride * -3;
    pixel[8] = 0 + row_stride * -3;
    pixel[9] = -1 + row_stride * -3;
    pixel[10] = -2 + row_stride * -2;
    pixel[11] = -3 + row_stride * -1;
    pixel[12] = -3 + row_stride * 0;
    pixel[13] = -3 + row_stride * 1;
    pixel[14] = -2 + row_stride * 2;
    pixel[15] = -1 + row_stride * 3;
}

u8 cornerScore(const u8* ptr, const ptrdiff_t pixel[])
{
    const s32 K = 8, N = 16 + K + 1;
    s32 k, v = ptr[0];
    s16 d[(N + 7) & ~7];
    for( k = 0; k < N; k++ )
        d[k] = (s16)(v - ptr[pixel[k]]);

    int16x8_t q0 = vdupq_n_s16((s16)(-1000));
    int16x8_t q1 = vdupq_n_s16((s16)(1000));

    int16x8_t d0_7   = vld1q_s16(d +  0);
    int16x8_t d8_15  = vld1q_s16(d +  8);
    int16x8_t d16_23 = vld1q_s16(d + 16);
    int16x8_t d24    = vld1q_s16(d + 24);

    //k == 0
    int16x8_t v0k0 = vextq_s16(d0_7, d8_15, 1);
    int16x8_t v1k0 = vextq_s16(d0_7, d8_15, 2);
    int16x8_t ak0 = vminq_s16(v0k0, v1k0);
    int16x8_t bk0 = vmaxq_s16(v0k0, v1k0);

    v0k0 = vextq_s16(d0_7, d8_15, 3);
    ak0 = vminq_s16(ak0, v0k0);
    bk0 = vmaxq_s16(bk0, v0k0);

    v1k0 = vextq_s16(d0_7, d8_15, 4);
    ak0 = vminq_s16(ak0, v1k0);
    bk0 = vmaxq_s16(bk0, v1k0);

    v0k0 = vextq_s16(d0_7, d8_15, 5);
    ak0 = vminq_s16(ak0, v0k0);
    bk0 = vmaxq_s16(bk0, v0k0);

    v1k0 = vextq_s16(d0_7, d8_15, 6);
    ak0 = vminq_s16(ak0, v1k0);
    bk0 = vmaxq_s16(bk0, v1k0);

    v0k0 = vextq_s16(d0_7, d8_15, 7);
    ak0 = vminq_s16(ak0, v0k0);
    bk0 = vmaxq_s16(bk0, v0k0);

    ak0 = vminq_s16(ak0, d8_15);
    bk0 = vmaxq_s16(bk0, d8_15);

    q0 = vmaxq_s16(q0, vminq_s16(ak0, d0_7));
    q1 = vminq_s16(q1, vmaxq_s16(bk0, d0_7));

    v1k0 = vextq_s16(d8_15, d16_23, 1);
    q0 = vmaxq_s16(q0, vminq_s16(ak0, v1k0));
    q1 = vminq_s16(q1, vmaxq_s16(bk0, v1k0));

    //k == 8
    int16x8_t v0k8 = v1k0;
    int16x8_t v1k8 = vextq_s16(d8_15, d16_23, 2);
    int16x8_t ak8 = vminq_s16(v0k8, v1k8);
    int16x8_t bk8 = vmaxq_s16(v0k8, v1k8);

    v0k8 = vextq_s16(d8_15, d16_23, 3);
    ak8 = vminq_s16(ak8, v0k8);
    bk8 = vmaxq_s16(bk8, v0k8);

    v1k8 = vextq_s16(d8_15, d16_23, 4);
    ak8 = vminq_s16(ak8, v1k8);
    bk8 = vmaxq_s16(bk8, v1k8);

    v0k8 = vextq_s16(d8_15, d16_23, 5);
    ak8 = vminq_s16(ak8, v0k8);
    bk8 = vmaxq_s16(bk8, v0k8);

    v1k8 = vextq_s16(d8_15, d16_23, 6);
    ak8 = vminq_s16(ak8, v1k8);
    bk8 = vmaxq_s16(bk8, v1k8);

    v0k8 = vextq_s16(d8_15, d16_23, 7);
    ak8 = vminq_s16(ak8, v0k8);
    bk8 = vmaxq_s16(bk8, v0k8);

    ak8 = vminq_s16(ak8, d16_23);
    bk8 = vmaxq_s16(bk8, d16_23);

    q0 = vmaxq_s16(q0, vminq_s16(ak8, d8_15));
    q1 = vminq_s16(q1, vmaxq_s16(bk8, d8_15));

    v1k8 = vextq_s16(d16_23, d24, 1);
    q0 = vmaxq_s16(q0, vminq_s16(ak8, v1k8));
    q1 = vminq_s16(q1, vmaxq_s16(bk8, v1k8));

    //fin
    int16x8_t q = vmaxq_s16(q0, vsubq_s16(vmovq_n_s16(0), q1));
    int16x4_t q2 = vmax_s16(vget_low_s16(q), vget_high_s16(q));
    int32x4_t q2w = vmovl_s16(q2);
    int32x2_t q4 = vmax_s32(vget_low_s32(q2w), vget_high_s32(q2w));
    int32x2_t q8 = vmax_s32(q4, vreinterpret_s32_s64(vshr_n_s64(vreinterpret_s64_s32(q4), 32)));

    return (u8)(vget_lane_s32(q8, 0) - 1);
}

} //namespace
#endif

void FAST(const Size2D &size,
          u8 *srcBase, ptrdiff_t srcStride,
          KeypointStore *keypoints,
          u8 threshold, bool nonmax_suppression)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    //keypoints.clear();

    const s32 K = 8, N = 16 + K + 1;
    ptrdiff_t i, j, k, pixel[N];
    makeOffsets(pixel, srcStride);
    for(k = 16; k < N; k++)
        pixel[k] = pixel[k - 16];

    uint8x16_t delta = vdupq_n_u8(128);
    uint8x16_t t = vdupq_n_u8(threshold);
    uint8x16_t K16 = vdupq_n_u8((u8)K);

    u8 threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (u8)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    std::vector<u8> _buf((size.width+16)*3*(sizeof(ptrdiff_t) + sizeof(u8)) + 128);
    u8* buf[3];
    buf[0] = &_buf[0]; buf[1] = buf[0] + size.width; buf[2] = buf[1] + size.width;
    ptrdiff_t* cpbuf[3];
    cpbuf[0] = (ptrdiff_t*)internal::alignPtr(buf[2] + size.width, sizeof(ptrdiff_t)) + 1;
    cpbuf[1] = cpbuf[0] + size.width + 1;
    cpbuf[2] = cpbuf[1] + size.width + 1;
    memset(buf[0], 0, size.width*3);

    for(i = 3; i < (ptrdiff_t)size.height-2; i++)
    {
        const u8* ptr = internal::getRowPtr(srcBase, srcStride, i) + 3;
        u8* curr = buf[(i - 3)%3];
        ptrdiff_t* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, size.width);
        ptrdiff_t ncorners = 0;

        if( i < (ptrdiff_t)size.height - 3 )
        {
            j = 3;

            for(; j < (ptrdiff_t)size.width - 16 - 3; j += 16, ptr += 16)
            {
                internal::prefetch(ptr);
                internal::prefetch(ptr + pixel[0]);
                internal::prefetch(ptr + pixel[2]);

                uint8x16_t v0 = vld1q_u8(ptr);
                int8x16_t v1 = vreinterpretq_s8_u8(veorq_u8(vqsubq_u8(v0, t), delta));
                int8x16_t v2 = vreinterpretq_s8_u8(veorq_u8(vqaddq_u8(v0, t), delta));

                int8x16_t x0 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[0]), delta));
                int8x16_t x1 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[4]), delta));
                int8x16_t x2 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[8]), delta));
                int8x16_t x3 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[12]), delta));

                uint8x16_t m0 =   vandq_u8(vcgtq_s8(x0, v2), vcgtq_s8(x1, v2));
                uint8x16_t m1 =   vandq_u8(vcgtq_s8(v1, x0), vcgtq_s8(v1, x1));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_s8(x1, v2), vcgtq_s8(x2, v2)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_s8(v1, x1), vcgtq_s8(v1, x2)));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_s8(x2, v2), vcgtq_s8(x3, v2)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_s8(v1, x2), vcgtq_s8(v1, x3)));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_s8(x3, v2), vcgtq_s8(x0, v2)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_s8(v1, x3), vcgtq_s8(v1, x0)));
                m0 = vorrq_u8(m0, m1);

                u64 mask[2];
                vst1q_u64(mask, vreinterpretq_u64_u8(m0));

                if( mask[0] == 0 )
                {
                    if (mask[1] != 0)
                    {
                        j -= 8;
                        ptr -= 8;
                    }
                    continue;
                }

                uint8x16_t c0 = vmovq_n_u8(0);
                uint8x16_t c1 = vmovq_n_u8(0);
                uint8x16_t max0 = vmovq_n_u8(0);
                uint8x16_t max1 = vmovq_n_u8(0);
                for( k = 0; k < N; k++ )
                {
                    int8x16_t x = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(ptr + pixel[k]), delta));
                    m0 = vcgtq_s8(x, v2);
                    m1 = vcgtq_s8(v1, x);

                    c0 = vandq_u8(vsubq_u8(c0, m0), m0);
                    c1 = vandq_u8(vsubq_u8(c1, m1), m1);

                    max0 = vmaxq_u8(max0, c0);
                    max1 = vmaxq_u8(max1, c1);
                }

                max0 = vmaxq_u8(max0, max1);
                u8 m[16];
                vst1q_u8(m, vcgtq_u8(max0, K16));

                for( k = 0; k < 16; ++k )
                    if(m[k])
                    {
                        cornerpos[ncorners++] = j+k;
                        if(nonmax_suppression)
                            curr[j+k] = cornerScore(ptr+k, pixel);
                    }
            }

            for( ; j < (s32)size.width - 3; j++, ptr++ )
            {
                s32 v = ptr[0];
                const u8* tab = &threshold_tab[0] - v + 255;
                s32 d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    s32 vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        s32 x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = cornerScore(ptr, pixel);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    s32 vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        s32 x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = cornerScore(ptr, pixel);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        const u8* prev = buf[(i - 4 + 3)%3];
        const u8* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            s32 score = prev[j];
            if( !nonmax_suppression ||
                    (score > prev[j+1] && score > prev[j-1] &&
                     score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                     score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints->push((f32)j, (f32)(i-1), 7.f, -1, (f32)score);
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)keypoints;
    (void)threshold;
    (void)nonmax_suppression;
#endif
}

} // namespace CAROTENE_NS
