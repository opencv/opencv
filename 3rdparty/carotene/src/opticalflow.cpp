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

#include "common.hpp"
#include "saturate_cast.hpp"
#include <vector>
#include <float.h> // For FLT_EPSILON

namespace CAROTENE_NS {

#define CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

/*
 *        Pyramidal Lucas-Kanade Optical Flow level processing
 */
void pyrLKOptFlowLevel(const Size2D &size, s32 cn,
                       const u8 *prevData, ptrdiff_t prevStride,
                       const s16 *prevDerivData, ptrdiff_t prevDerivStride,
                       const u8 *nextData, ptrdiff_t nextStride,
                       u32 ptCount,
                       const f32 *prevPts, f32 *nextPts,
                       u8 *status, f32 *err,
                       const Size2D &winSize,
                       u32 terminationCount, f64 terminationEpsilon,
                       bool getMinEigenVals,
                       f32 minEigThreshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    f32 halfWinX = (winSize.width-1)*0.5f, halfWinY = (winSize.height-1)*0.5f;
    s32 cn2 = cn*2;

    std::vector<s16> _buf(winSize.total()*(cn + cn2));
    s16* IWinBuf = &_buf[0];
    s32  IWinBufStride = winSize.width*cn;
    s16* derivIWinBuf = &_buf[winSize.total()*cn];
    s32  derivIWinBufStride = winSize.width*cn2;

    for( u32 ptidx = 0; ptidx < ptCount; ptidx++ )
    {
        f32 prevPtX = prevPts[ptref+0];
        f32 prevPtY = prevPts[ptref+1];
        f32 nextPtX = nextPts[ptref+0];
        f32 nextPtY = nextPts[ptref+1];

        s32 iprevPtX, iprevPtY;
        s32 inextPtX, inextPtY;
        prevPtX -= halfWinX;
        prevPtY -= halfWinY;
        iprevPtX = floor(prevPtX);
        iprevPtY = floor(prevPtY);

        if( iprevPtX < -(s32)winSize.width || iprevPtX >= (s32)size.width ||
            iprevPtY < -(s32)winSize.height || iprevPtY >= (s32)size.height )
        {
            if( status )
                status[ptidx] = false;
            if( err )
                err[ptidx] = 0;
            continue;
        }

        f32 a = prevPtX - iprevPtX;
        f32 b = prevPtY - iprevPtY;
        const s32 W_BITS = 14, W_BITS1 = 14;
        const f32 FLT_SCALE = 1.f/(1 << 20);
        s32 iw00 = round((1.f - a)*(1.f - b)*(1 << W_BITS));
        s32 iw01 = round(a*(1.f - b)*(1 << W_BITS));
        s32 iw10 = round((1.f - a)*b*(1 << W_BITS));
        s32 iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        s32 dstep = prevDerivStride/sizeof(s16);
        f32 A11 = 0, A12 = 0, A22 = 0;

        int16x4_t viw00 = vmov_n_s16((s16)iw00);
        int16x4_t viw01 = vmov_n_s16((s16)iw01);
        int16x4_t viw10 = vmov_n_s16((s16)iw10);
        int16x4_t viw11 = vmov_n_s16((s16)iw11);

        float32x4_t vA11 = vmovq_n_f32(0);
        float32x4_t vA12 = vmovq_n_f32(0);
        float32x4_t vA22 = vmovq_n_f32(0);

        s32 wwcn = winSize.width*cn;

        // extract the patch from the first image, compute covariation matrix of derivatives
        s32 x = 0;
        for(s32 y = 0; y < (s32)winSize.height; y++ )
        {
            const u8* src = prevData + prevStride*(y + iprevPtY) + iprevPtX*cn;
            const s16* dsrc = prevDerivData + dstep*(y + iprevPtY) + iprevPtX*cn2;

            s16* Iptr = IWinBuf + y*IWinBufStride;
            s16* dIptr = derivIWinBuf + y*derivIWinBufStride;

            internal::prefetch(src + x + prevStride * 2, 0);
            for(x = 0; x <= wwcn - 8; x += 8)
            {
                uint8x8_t vsrc00 = vld1_u8(src + x);
                uint8x8_t vsrc10 = vld1_u8(src + x + prevStride);
                uint8x8_t vsrc01 = vld1_u8(src + x + cn);
                uint8x8_t vsrc11 = vld1_u8(src + x + prevStride + cn);

                int16x8_t vs00 = vreinterpretq_s16_u16(vmovl_u8(vsrc00));
                int16x8_t vs10 = vreinterpretq_s16_u16(vmovl_u8(vsrc10));
                int16x8_t vs01 = vreinterpretq_s16_u16(vmovl_u8(vsrc01));
                int16x8_t vs11 = vreinterpretq_s16_u16(vmovl_u8(vsrc11));

                int32x4_t vsuml = vmull_s16(vget_low_s16(vs00), viw00);
                int32x4_t vsumh = vmull_s16(vget_high_s16(vs10), viw10);

                vsuml = vmlal_s16(vsuml, vget_low_s16(vs01), viw01);
                vsumh = vmlal_s16(vsumh, vget_high_s16(vs11), viw11);

                vsuml = vmlal_s16(vsuml, vget_low_s16(vs10), viw10);
                vsumh = vmlal_s16(vsumh, vget_high_s16(vs00), viw00);

                vsuml = vmlal_s16(vsuml, vget_low_s16(vs11), viw11);
                vsumh = vmlal_s16(vsumh, vget_high_s16(vs01), viw01);

                int16x4_t vsumnl = vrshrn_n_s32(vsuml, W_BITS1-5);
                int16x4_t vsumnh = vrshrn_n_s32(vsumh, W_BITS1-5);

                vst1q_s16(Iptr + x, vcombine_s16(vsumnl, vsumnh));
            }
            for(; x <= wwcn - 4; x += 4)
            {
                uint8x8_t vsrc00 = vld1_u8(src + x);
                uint8x8_t vsrc10 = vld1_u8(src + x + prevStride);
                uint8x8_t vsrc01 = vld1_u8(src + x + cn);
                uint8x8_t vsrc11 = vld1_u8(src + x + prevStride + cn);

                int16x4_t vs00 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vsrc00)));
                int16x4_t vs10 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vsrc10)));
                int16x4_t vs01 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vsrc01)));
                int16x4_t vs11 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vsrc11)));

                int32x4_t vsuml1 = vmull_s16(vs00, viw00);
                int32x4_t vsuml2 = vmull_s16(vs01, viw01);
                vsuml1 = vmlal_s16(vsuml1, vs10, viw10);
                vsuml2 = vmlal_s16(vsuml2, vs11, viw11);
                int32x4_t vsuml = vaddq_s32(vsuml1, vsuml2);

                int16x4_t vsumnl = vrshrn_n_s32(vsuml, W_BITS1-5);

                vst1_s16(Iptr + x, vsumnl);
            }

            internal::prefetch(dsrc + dstep * 2, 0);
            for(x = 0; x <= wwcn - 4; x += 4, dsrc += 4*2, dIptr += 4*2 )
            {
#if 0
                __asm__ (
                    "vld2.16 {d0-d1}, [%[dsrc00]]                         \n\t"
                    "vld2.16 {d2-d3}, [%[dsrc10]]                         \n\t"
                    "vld2.16 {d4-d5}, [%[dsrc01]]                         \n\t"
                    "vld2.16 {d6-d7}, [%[dsrc11]]                         \n\t"
                    "vmull.s16 q4, d3, %P[viw10]                           \n\t"
                    "vmull.s16 q5, d0, %P[viw00]                           \n\t"
                    "vmlal.s16 q4, d7, %P[viw11]                           \n\t"
                    "vmlal.s16 q5, d4, %P[viw01]                           \n\t"
                    "vmlal.s16 q4, d1, %P[viw00]                           \n\t"
                    "vmlal.s16 q5, d2, %P[viw10]                           \n\t"
                    "vmlal.s16 q4, d5, %P[viw01]                           \n\t"
                    "vmlal.s16 q5, d6, %P[viw11]                            \n\t"
                    "vrshrn.s32 d13, q4, %[W_BITS1]                       \n\t"
                    "vrshrn.s32 d12, q5, %[W_BITS1]                       \n\t"
                    "vmull.s16 q3, d13, d13                               \n\t"
                    "vmull.s16 q4, d12, d12                               \n\t"
                    "vmull.s16 q5, d13, d12                               \n\t"
                    "vcvt.f32.s32 q3, q3                                  \n\t"
                    "vcvt.f32.s32 q4, q4                                  \n\t"
                    "vcvt.f32.s32 q5, q5                                  \n\t"
                    "vadd.f32 %q[vA22], q3                                \n\t"
                    "vadd.f32 %q[vA11], q4                                \n\t"
                    "vadd.f32 %q[vA12], q5                                \n\t"
                    "vst2.16 {d12-d13}, [%[out]]                          \n\t"
                    : [vA22] "=w" (vA22),
                      [vA11] "=w" (vA11),
                      [vA12] "=w" (vA12)
                    : "0" (vA22),
                      "1" (vA11),
                      "2" (vA12),
                      [out] "r" (dIptr),
                      [dsrc00] "r" (dsrc),
                      [dsrc10] "r" (dsrc + dstep),
                      [dsrc01] "r" (dsrc + cn2),
                      [dsrc11] "r" (dsrc + dstep + cn2),
                      [viw00] "w" (viw00),
                      [viw10] "w" (viw10),
                      [viw01] "w" (viw01),
                      [viw11] "w" (viw11),
                      [W_BITS1] "I" (W_BITS1)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13"
                );
#else
                int16x4x2_t vdsrc00 = vld2_s16(dsrc);
                int16x4x2_t vdsrc10 = vld2_s16(dsrc + dstep);
                int16x4x2_t vdsrc01 = vld2_s16(dsrc + cn2);
                int16x4x2_t vdsrc11 = vld2_s16(dsrc + dstep + cn2);

                int32x4_t vsumy = vmull_s16(vdsrc10.val[1], viw10);
                int32x4_t vsumx = vmull_s16(vdsrc00.val[0], viw00);

                vsumy = vmlal_s16(vsumy, vdsrc11.val[1], viw11);
                vsumx = vmlal_s16(vsumx, vdsrc01.val[0], viw01);

                vsumy = vmlal_s16(vsumy, vdsrc00.val[1], viw00);
                vsumx = vmlal_s16(vsumx, vdsrc10.val[0], viw10);

                vsumy = vmlal_s16(vsumy, vdsrc01.val[1], viw01);
                vsumx = vmlal_s16(vsumx, vdsrc11.val[0], viw11);

                int16x4_t vsumny = vrshrn_n_s32(vsumy, W_BITS1);
                int16x4_t vsumnx = vrshrn_n_s32(vsumx, W_BITS1);

                int32x4_t va22i = vmull_s16(vsumny, vsumny);
                int32x4_t va11i = vmull_s16(vsumnx, vsumnx);
                int32x4_t va12i = vmull_s16(vsumnx, vsumny);

                float32x4_t va22f = vcvtq_f32_s32(va22i);
                float32x4_t va11f = vcvtq_f32_s32(va11i);
                float32x4_t va12f = vcvtq_f32_s32(va12i);

                vA22 = vaddq_f32(vA22, va22f);
                vA11 = vaddq_f32(vA11, va11f);
                vA12 = vaddq_f32(vA12, va12f);

                int16x4x2_t vsum;
                vsum.val[0] = vsumnx;
                vsum.val[1] = vsumny;
                vst2_s16(dIptr, vsum);
#endif
            }

            for( ; x < wwcn; x++, dsrc += 2, dIptr += 2 )
            {
                s32 ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                      src[x+prevStride]*iw10 + src[x+prevStride+cn]*iw11, W_BITS1-5);
                s32 ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                       dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
                s32 iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                       dsrc[dstep+cn2+1]*iw11, W_BITS1);
                Iptr[x] = (s16)ival;
                dIptr[0] = (s16)ixval;
                dIptr[1] = (s16)iyval;

                A11 += (f32)(ixval*ixval);
                A12 += (f32)(ixval*iyval);
                A22 += (f32)(iyval*iyval);
            }
        }

        f32 A11buf[2], A12buf[2], A22buf[2];
        vst1_f32(A11buf, vadd_f32(vget_low_f32(vA11), vget_high_f32(vA11)));
        vst1_f32(A12buf, vadd_f32(vget_low_f32(vA12), vget_high_f32(vA12)));
        vst1_f32(A22buf, vadd_f32(vget_low_f32(vA22), vget_high_f32(vA22)));
        A11 += A11buf[0] + A11buf[1];
        A12 += A12buf[0] + A12buf[1];
        A22 += A22buf[0] + A22buf[1];

        A11 *= FLT_SCALE;
        A12 *= FLT_SCALE;
        A22 *= FLT_SCALE;

        f32 D = A11*A22 - A12*A12;
        f32 minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winSize.width*winSize.height);

        if( err && getMinEigenVals )
            err[ptidx] = (f32)minEig;

        if( minEig < minEigThreshold || D < FLT_EPSILON )
        {
            if( tatus )
                status[ptidx] = false;
            continue;
        }

        D = 1.f/D;

        nextPtX -= halfWinX;
        nextPtY -= halfWinY;
        f32 prevDeltaX = 0;
        f32 prevDeltaY = 0;

        for(u32 j = 0; j < terminationCount; j++ )
        {
            inextPtX = floor(nextPtX);
            inextPtY = floor(nextPtY);

            if( inextPtX < -(s32)winSize.width || inextPtX >= (s32)size.width ||
               inextPtY < -(s32)winSize.height || inextPtY >= (s32)size.height )
            {
                if( status )
                    status[ptidx] = false;
                break;
            }

            a = nextPtX - inextPtX;
            b = nextPtY - inextPtY;
            iw00 = round((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = round(a*(1.f - b)*(1 << W_BITS));
            iw10 = round((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            f32 b1 = 0, b2 = 0;

            viw00 = vmov_n_s16((s16)iw00);
            viw01 = vmov_n_s16((s16)iw01);
            viw10 = vmov_n_s16((s16)iw10);
            viw11 = vmov_n_s16((s16)iw11);

            float32x4_t vb1 = vmovq_n_f32(0);
            float32x4_t vb2 = vmovq_n_f32(0);

            for(s32 y = 0; y < (s32)winSize.height; y++ )
            {
                const u8* Jptr = nextData + nextStride*(y + inextPtY) + inextPtX*cn;
                const s16* Iptr = IWinBuf + y*IWinBufStride;
                const s16* dIptr = derivIWinBuf + y*derivIWinBufStride;

                x = 0;

                internal::prefetch(Jptr, nextStride * 2);
                internal::prefetch(Iptr, IWinBufStride/2);
                internal::prefetch(dIptr, derivIWinBufStride/2);

                for( ; x <= wwcn - 8; x += 8, dIptr += 8*2 )
                {
                    uint8x8_t vj00 = vld1_u8(Jptr + x);
                    uint8x8_t vj10 = vld1_u8(Jptr + x + nextStride);
                    uint8x8_t vj01 = vld1_u8(Jptr + x + cn);
                    uint8x8_t vj11 = vld1_u8(Jptr + x + nextStride + cn);
                    int16x8_t vI = vld1q_s16(Iptr + x);
                    int16x8x2_t vDerivI = vld2q_s16(dIptr);

                    int16x8_t vs00 = vreinterpretq_s16_u16(vmovl_u8(vj00));
                    int16x8_t vs10 = vreinterpretq_s16_u16(vmovl_u8(vj10));
                    int16x8_t vs01 = vreinterpretq_s16_u16(vmovl_u8(vj01));
                    int16x8_t vs11 = vreinterpretq_s16_u16(vmovl_u8(vj11));

                    int32x4_t vsuml = vmull_s16(vget_low_s16(vs00), viw00);
                    int32x4_t vsumh = vmull_s16(vget_high_s16(vs10), viw10);

                    vsuml = vmlal_s16(vsuml, vget_low_s16(vs01), viw01);
                    vsumh = vmlal_s16(vsumh, vget_high_s16(vs11), viw11);

                    vsuml = vmlal_s16(vsuml, vget_low_s16(vs10), viw10);
                    vsumh = vmlal_s16(vsumh, vget_high_s16(vs00), viw00);

                    vsuml = vmlal_s16(vsuml, vget_low_s16(vs11), viw11);
                    vsumh = vmlal_s16(vsumh, vget_high_s16(vs01), viw01);

                    int16x4_t vsumnl = vrshrn_n_s32(vsuml, W_BITS1-5);
                    int16x4_t vsumnh = vrshrn_n_s32(vsumh, W_BITS1-5);

                    int16x8_t diff = vqsubq_s16(vcombine_s16(vsumnl, vsumnh), vI);

                    int32x4_t vb1l = vmull_s16(vget_low_s16(diff), vget_low_s16(vDerivI.val[0]));
                    int32x4_t vb2h = vmull_s16(vget_high_s16(diff), vget_high_s16(vDerivI.val[1]));
                    int32x4_t vb1i = vmlal_s16(vb1l, vget_high_s16(diff), vget_high_s16(vDerivI.val[0]));
                    int32x4_t vb2i = vmlal_s16(vb2h, vget_low_s16(diff), vget_low_s16(vDerivI.val[1]));

                    float32x4_t vb1f = vcvtq_f32_s32(vb1i);
                    float32x4_t vb2f = vcvtq_f32_s32(vb2i);

                    vb1 = vaddq_f32(vb1, vb1f);
                    vb2 = vaddq_f32(vb2, vb2f);
                }

                for( ; x < wwcn; x++, dIptr += 2 )
                {
                    s32 diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+nextStride]*iw10 + Jptr[x+nextStride+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    b1 += (f32)(diff*dIptr[0]);
                    b2 += (f32)(diff*dIptr[1]);
                }
            }

            f32 bbuf[2];
            float32x2_t vb = vpadd_f32(vadd_f32(vget_low_f32(vb1), vget_high_f32(vb1)), vadd_f32(vget_low_f32(vb2), vget_high_f32(vb2)));
            vst1_f32(bbuf, vb);
            b1 += bbuf[0];
            b2 += bbuf[1];

            b1 *= FLT_SCALE;
            b2 *= FLT_SCALE;

            f32 deltaX = (f32)((A12*b2 - A22*b1) * D);
            f32 deltaY = (f32)((A12*b1 - A11*b2) * D);

            nextPtX += deltaX;
            nextPtY += deltaY;
            nextPts[ptref+0] = nextPtX + halfWinX;
            nextPts[ptref+1] = nextPtY + halfWinY;

            if( ((double)deltaX*deltaX + (double)deltaY*deltaY) <= terminationEpsilon )
                break;

            if( j > 0 && std::abs(deltaX + prevDeltaX) < 0.01 &&
               std::abs(deltaY + prevDeltaY) < 0.01 )
            {
                nextPts[ptref+0] -= deltaX*0.5f;
                nextPts[ptref+1] -= deltaY*0.5f;
                break;
            }
            prevDeltaX = deltaX;
            prevDeltaY = deltaY;
        }
    }
#else
    (void)size;
    (void)cn;
    (void)prevData;
    (void)prevStride;
    (void)prevDerivData;
    (void)prevDerivStride;
    (void)nextData;
    (void)nextStride;
    (void)prevPts;
    (void)nextPts;
    (void)status;
    (void)err;
    (void)winSize;
    (void)terminationCount;
    (void)terminationEpsilon;
    (void)getMinEigenVals;
    (void)minEigThreshold;
    (void)ptCount;
#endif
}

}//CAROTENE_NS

