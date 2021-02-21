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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
#include <float.h>
#include <stdio.h>
#include "lkpyramid.hpp"
#include "opencl_kernels_video.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

namespace
{
static void calcScharrDeriv(const cv::Mat& src, cv::Mat& dst)
{
    using namespace cv;
    using cv::detail::deriv_type;
    int rows = src.rows, cols = src.cols, cn = src.channels(), depth = src.depth();
    CV_Assert(depth == CV_8U);
    dst.create(rows, cols, CV_MAKETYPE(DataType<deriv_type>::depth, cn*2));
    parallel_for_(Range(0, rows), cv::detail::ScharrDerivInvoker(src, dst), cv::getNumThreads());
}

}//namespace

void cv::detail::ScharrDerivInvoker::operator()(const Range& range) const
{
    using cv::detail::deriv_type;
    int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols*cn;

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra() && tegra::calcSharrDeriv(src, dst))
        return;
#endif

    int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf.data() + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

#if CV_SIMD128
    v_int16x8 c3 = v_setall_s16(3), c10 = v_setall_s16(10);
#endif

    for( y = range.start; y < range.end; y++ )
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        deriv_type* drow = (deriv_type *)dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;
#if CV_SIMD128
        {
            for( ; x <= colsn - 8; x += 8 )
            {
                v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(srow0 + x));
                v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(srow1 + x));
                v_int16x8 s2 = v_reinterpret_as_s16(v_load_expand(srow2 + x));

                v_int16x8 t1 = s2 - s0;
                v_int16x8 t0 = v_mul_wrap(s0 + s2, c3) + v_mul_wrap(s1, c10);

                v_store(trow0 + x, t0);
                v_store(trow1 + x, t1);
            }
        }
#endif

        for( ; x < colsn; x++ )
        {
            int t0 = (srow0[x] + srow2[x])*3 + srow1[x]*10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        int x0 = (cols > 1 ? 1 : 0)*cn, x1 = (cols > 1 ? cols-2 : 0)*cn;
        for( int k = 0; k < cn; k++ )
        {
            trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
#if CV_SIMD128
        {
            for( ; x <= colsn - 8; x += 8 )
            {
                v_int16x8 s0 = v_load(trow0 + x - cn);
                v_int16x8 s1 = v_load(trow0 + x + cn);
                v_int16x8 s2 = v_load(trow1 + x - cn);
                v_int16x8 s3 = v_load(trow1 + x);
                v_int16x8 s4 = v_load(trow1 + x + cn);

                v_int16x8 t0 = s1 - s0;
                v_int16x8 t1 = v_mul_wrap(s2 + s4, c3) + v_mul_wrap(s3, c10);

                v_store_interleave((drow + x*2), t0, t1);
            }
        }
#endif
        for( ; x < colsn; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+cn] - trow0[x-cn]);
            deriv_type t1 = (deriv_type)((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
            drow[x*2] = t0; drow[x*2+1] = t1;
        }
    }
}

cv::detail::LKTrackerInvoker::LKTrackerInvoker(
                      const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                      const Point2f* _prevPts, Point2f* _nextPts,
                      uchar* _status, float* _err,
                      Size _winSize, TermCriteria _criteria,
                      int _level, int _maxLevel, int _flags, float _minEigThreshold )
{
    prevImg = &_prevImg;
    prevDeriv = &_prevDeriv;
    nextImg = &_nextImg;
    prevPts = _prevPts;
    nextPts = _nextPts;
    status = _status;
    err = _err;
    winSize = _winSize;
    criteria = _criteria;
    level = _level;
    maxLevel = _maxLevel;
    flags = _flags;
    minEigThreshold = _minEigThreshold;
}

#if defined __arm__ && !CV_NEON
typedef int64 acctype;
typedef int itemtype;
#else
typedef float acctype;
typedef float itemtype;
#endif

void cv::detail::LKTrackerInvoker::operator()(const Range& range) const
{
    CV_INSTRUMENT_REGION();

    Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
    const Mat& I = *prevImg;
    const Mat& J = *nextImg;
    const Mat& derivI = *prevDeriv;

    int j, cn = I.channels(), cn2 = cn*2;
    cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2));
    int derivDepth = DataType<deriv_type>::depth;

    Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), _buf.data());
    Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), _buf.data() + winSize.area()*cn);

    for( int ptidx = range.start; ptidx < range.end; ptidx++ )
    {
        Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
        Point2f nextPt;
        if( level == maxLevel )
        {
            if( flags & OPTFLOW_USE_INITIAL_FLOW )
                nextPt = nextPts[ptidx]*(float)(1./(1 << level));
            else
                nextPt = prevPt;
        }
        else
            nextPt = nextPts[ptidx]*2.f;
        nextPts[ptidx] = nextPt;

        Point2i iprevPt, inextPt;
        prevPt -= halfWin;
        iprevPt.x = cvFloor(prevPt.x);
        iprevPt.y = cvFloor(prevPt.y);

        if( iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
            iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows )
        {
            if( level == 0 )
            {
                if( status )
                    status[ptidx] = false;
                if( err )
                    err[ptidx] = 0;
            }
            continue;
        }

        float a = prevPt.x - iprevPt.x;
        float b = prevPt.y - iprevPt.y;
        const int W_BITS = 14, W_BITS1 = 14;
        const float FLT_SCALE = 1.f/(1 << 20);
        int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
        int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
        int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
        int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        int dstep = (int)(derivI.step/derivI.elemSize1());
        int stepI = (int)(I.step/I.elemSize1());
        int stepJ = (int)(J.step/J.elemSize1());
        acctype iA11 = 0, iA12 = 0, iA22 = 0;
        float A11, A12, A22;

#if CV_SIMD128 && !CV_NEON
        v_int16x8 qw0((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
        v_int16x8 qw1((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
        v_int32x4 qdelta_d = v_setall_s32(1 << (W_BITS1-1));
        v_int32x4 qdelta = v_setall_s32(1 << (W_BITS1-5-1));
        v_float32x4 qA11 = v_setzero_f32(), qA12 = v_setzero_f32(), qA22 = v_setzero_f32();
#endif

#if CV_NEON

        float CV_DECL_ALIGNED(16) nA11[] = { 0, 0, 0, 0 }, nA12[] = { 0, 0, 0, 0 }, nA22[] = { 0, 0, 0, 0 };
        const int shifter1 = -(W_BITS - 5); //negative so it shifts right
        const int shifter2 = -(W_BITS);

        const int16x4_t d26 = vdup_n_s16((int16_t)iw00);
        const int16x4_t d27 = vdup_n_s16((int16_t)iw01);
        const int16x4_t d28 = vdup_n_s16((int16_t)iw10);
        const int16x4_t d29 = vdup_n_s16((int16_t)iw11);
        const int32x4_t q11 = vdupq_n_s32((int32_t)shifter1);
        const int32x4_t q12 = vdupq_n_s32((int32_t)shifter2);

#endif

        // extract the patch from the first image, compute covariation matrix of derivatives
        int x, y;
        for( y = 0; y < winSize.height; y++ )
        {
            const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x*cn;
            const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y)*dstep + iprevPt.x*cn2;

            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

            x = 0;

#if CV_SIMD128 && !CV_NEON
            for( ; x <= winSize.width*cn - 8; x += 8, dsrc += 8*2, dIptr += 8*2 )
            {
                v_int32x4 t0, t1;
                v_int16x8 v00, v01, v10, v11, t00, t01, t10, t11;

                v00 = v_reinterpret_as_s16(v_load_expand(src + x));
                v01 = v_reinterpret_as_s16(v_load_expand(src + x + cn));
                v10 = v_reinterpret_as_s16(v_load_expand(src + x + stepI));
                v11 = v_reinterpret_as_s16(v_load_expand(src + x + stepI + cn));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta) + v_dotprod(t11, qw1);
                t0 = t0 >> (W_BITS1-5);
                t1 = t1 >> (W_BITS1-5);
                v_store(Iptr + x, v_pack(t0, t1));

                v00 = v_reinterpret_as_s16(v_load(dsrc));
                v01 = v_reinterpret_as_s16(v_load(dsrc + cn2));
                v10 = v_reinterpret_as_s16(v_load(dsrc + dstep));
                v11 = v_reinterpret_as_s16(v_load(dsrc + dstep + cn2));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta_d) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta_d) + v_dotprod(t11, qw1);
                t0 = t0 >> W_BITS1;
                t1 = t1 >> W_BITS1;
                v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                v_store(dIptr, v00);

                v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
                v_expand(v00, t1, t0);

                v_float32x4 fy = v_cvt_f32(t0);
                v_float32x4 fx = v_cvt_f32(t1);

                qA22 = v_muladd(fy, fy, qA22);
                qA12 = v_muladd(fx, fy, qA12);
                qA11 = v_muladd(fx, fx, qA11);

                v00 = v_reinterpret_as_s16(v_load(dsrc + 4*2));
                v01 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + cn2));
                v10 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + dstep));
                v11 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + dstep + cn2));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta_d) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta_d) + v_dotprod(t11, qw1);
                t0 = t0 >> W_BITS1;
                t1 = t1 >> W_BITS1;
                v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                v_store(dIptr + 4*2, v00);

                v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
                v_expand(v00, t1, t0);

                fy = v_cvt_f32(t0);
                fx = v_cvt_f32(t1);

                qA22 = v_muladd(fy, fy, qA22);
                qA12 = v_muladd(fx, fy, qA12);
                qA11 = v_muladd(fx, fx, qA11);
            }
#endif

#if CV_NEON
            for( ; x <= winSize.width*cn - 4; x += 4, dsrc += 4*2, dIptr += 4*2 )
            {

                uint8x8_t d0 = vld1_u8(&src[x]);
                uint8x8_t d2 = vld1_u8(&src[x+cn]);
                uint16x8_t q0 = vmovl_u8(d0);
                uint16x8_t q1 = vmovl_u8(d2);

                int32x4_t q5 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26);
                int32x4_t q6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27);

                uint8x8_t d4 = vld1_u8(&src[x + stepI]);
                uint8x8_t d6 = vld1_u8(&src[x + stepI + cn]);
                uint16x8_t q2 = vmovl_u8(d4);
                uint16x8_t q3 = vmovl_u8(d6);

                int32x4_t q7 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28);
                int32x4_t q8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29);

                q5 = vaddq_s32(q5, q6);
                q7 = vaddq_s32(q7, q8);
                q5 = vaddq_s32(q5, q7);

                int16x4x2_t d0d1 = vld2_s16(dsrc);
                int16x4x2_t d2d3 = vld2_s16(&dsrc[cn2]);

                q5 = vqrshlq_s32(q5, q11);

                int32x4_t q4 = vmull_s16(d0d1.val[0], d26);
                q6 = vmull_s16(d0d1.val[1], d26);

                int16x4_t nd0 = vmovn_s32(q5);

                q7 = vmull_s16(d2d3.val[0], d27);
                q8 = vmull_s16(d2d3.val[1], d27);

                vst1_s16(&Iptr[x], nd0);

                int16x4x2_t d4d5 = vld2_s16(&dsrc[dstep]);
                int16x4x2_t d6d7 = vld2_s16(&dsrc[dstep+cn2]);

                q4 = vaddq_s32(q4, q7);
                q6 = vaddq_s32(q6, q8);

                q7 = vmull_s16(d4d5.val[0], d28);
                int32x4_t q14 = vmull_s16(d4d5.val[1], d28);
                q8 = vmull_s16(d6d7.val[0], d29);
                int32x4_t q15 = vmull_s16(d6d7.val[1], d29);

                q7 = vaddq_s32(q7, q8);
                q14 = vaddq_s32(q14, q15);

                q4 = vaddq_s32(q4, q7);
                q6 = vaddq_s32(q6, q14);

                float32x4_t nq0 = vld1q_f32(nA11);
                float32x4_t nq1 = vld1q_f32(nA12);
                float32x4_t nq2 = vld1q_f32(nA22);

                q4 = vqrshlq_s32(q4, q12);
                q6 = vqrshlq_s32(q6, q12);

                q7 = vmulq_s32(q4, q4);
                q8 = vmulq_s32(q4, q6);
                q15 = vmulq_s32(q6, q6);

                nq0 = vaddq_f32(nq0, vcvtq_f32_s32(q7));
                nq1 = vaddq_f32(nq1, vcvtq_f32_s32(q8));
                nq2 = vaddq_f32(nq2, vcvtq_f32_s32(q15));

                vst1q_f32(nA11, nq0);
                vst1q_f32(nA12, nq1);
                vst1q_f32(nA22, nq2);

                int16x4_t d8 = vmovn_s32(q4);
                int16x4_t d12 = vmovn_s32(q6);

                int16x4x2_t d8d12;
                d8d12.val[0] = d8; d8d12.val[1] = d12;
                vst2_s16(dIptr, d8d12);
            }
#endif

            for( ; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2 )
            {
                int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                      src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
                int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                       dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
                int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                       dsrc[dstep+cn2+1]*iw11, W_BITS1);

                Iptr[x] = (short)ival;
                dIptr[0] = (short)ixval;
                dIptr[1] = (short)iyval;

                iA11 += (itemtype)(ixval*ixval);
                iA12 += (itemtype)(ixval*iyval);
                iA22 += (itemtype)(iyval*iyval);
            }
        }

#if CV_SIMD128 && !CV_NEON
        iA11 += v_reduce_sum(qA11);
        iA12 += v_reduce_sum(qA12);
        iA22 += v_reduce_sum(qA22);
#endif

#if CV_NEON
        iA11 += nA11[0] + nA11[1] + nA11[2] + nA11[3];
        iA12 += nA12[0] + nA12[1] + nA12[2] + nA12[3];
        iA22 += nA22[0] + nA22[1] + nA22[2] + nA22[3];
#endif

        A11 = iA11*FLT_SCALE;
        A12 = iA12*FLT_SCALE;
        A22 = iA22*FLT_SCALE;

        float D = A11*A22 - A12*A12;
        float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winSize.width*winSize.height);

        if( err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0 )
            err[ptidx] = (float)minEig;

        if( minEig < minEigThreshold || D < FLT_EPSILON )
        {
            if( level == 0 && status )
                status[ptidx] = false;
            continue;
        }

        D = 1.f/D;

        nextPt -= halfWin;
        Point2f prevDelta;

        for( j = 0; j < criteria.maxCount; j++ )
        {
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);

            if( inextPt.x < -winSize.width || inextPt.x >= J.cols ||
               inextPt.y < -winSize.height || inextPt.y >= J.rows )
            {
                if( level == 0 && status )
                    status[ptidx] = false;
                break;
            }

            a = nextPt.x - inextPt.x;
            b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            acctype ib1 = 0, ib2 = 0;
            float b1, b2;
#if CV_SIMD128 && !CV_NEON
            qw0 = v_int16x8((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
            qw1 = v_int16x8((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
            v_float32x4 qb0 = v_setzero_f32(), qb1 = v_setzero_f32();
#endif

#if CV_NEON
            float CV_DECL_ALIGNED(16) nB1[] = { 0,0,0,0 }, nB2[] = { 0,0,0,0 };

            const int16x4_t d26_2 = vdup_n_s16((int16_t)iw00);
            const int16x4_t d27_2 = vdup_n_s16((int16_t)iw01);
            const int16x4_t d28_2 = vdup_n_s16((int16_t)iw10);
            const int16x4_t d29_2 = vdup_n_s16((int16_t)iw11);

#endif

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x*cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

                x = 0;

#if CV_SIMD128 && !CV_NEON
                for( ; x <= winSize.width*cn - 8; x += 8, dIptr += 8*2 )
                {
                    v_int16x8 diff0 = v_reinterpret_as_s16(v_load(Iptr + x)), diff1, diff2;
                    v_int16x8 v00 = v_reinterpret_as_s16(v_load_expand(Jptr + x));
                    v_int16x8 v01 = v_reinterpret_as_s16(v_load_expand(Jptr + x + cn));
                    v_int16x8 v10 = v_reinterpret_as_s16(v_load_expand(Jptr + x + stepJ));
                    v_int16x8 v11 = v_reinterpret_as_s16(v_load_expand(Jptr + x + stepJ + cn));

                    v_int32x4 t0, t1;
                    v_int16x8 t00, t01, t10, t11;
                    v_zip(v00, v01, t00, t01);
                    v_zip(v10, v11, t10, t11);

                    t0 = v_dotprod(t00, qw0, qdelta) + v_dotprod(t10, qw1);
                    t1 = v_dotprod(t01, qw0, qdelta) + v_dotprod(t11, qw1);
                    t0 = t0 >> (W_BITS1-5);
                    t1 = t1 >> (W_BITS1-5);
                    diff0 = v_pack(t0, t1) - diff0;
                    v_zip(diff0, diff0, diff2, diff1); // It0 It0 It1 It1 ...
                    v00 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                    v01 = v_reinterpret_as_s16(v_load(dIptr + 8));
                    v_zip(v00, v01, v10, v11);
                    v_zip(diff2, diff1, v00, v01);
                    qb0 += v_cvt_f32(v_dotprod(v00, v10));
                    qb1 += v_cvt_f32(v_dotprod(v01, v11));
                }
#endif

#if CV_NEON
                for( ; x <= winSize.width*cn - 8; x += 8, dIptr += 8*2 )
                {

                    uint8x8_t d0 = vld1_u8(&Jptr[x]);
                    uint8x8_t d2 = vld1_u8(&Jptr[x+cn]);
                    uint8x8_t d4 = vld1_u8(&Jptr[x+stepJ]);
                    uint8x8_t d6 = vld1_u8(&Jptr[x+stepJ+cn]);

                    uint16x8_t q0 = vmovl_u8(d0);
                    uint16x8_t q1 = vmovl_u8(d2);
                    uint16x8_t q2 = vmovl_u8(d4);
                    uint16x8_t q3 = vmovl_u8(d6);

                    int32x4_t nq4 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26_2);
                    int32x4_t nq5 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q0)), d26_2);

                    int32x4_t nq6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27_2);
                    int32x4_t nq7 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q1)), d27_2);

                    int32x4_t nq8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28_2);
                    int32x4_t nq9 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q2)), d28_2);

                    int32x4_t nq10 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29_2);
                    int32x4_t nq11 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q3)), d29_2);

                    nq4 = vaddq_s32(nq4, nq6);
                    nq5 = vaddq_s32(nq5, nq7);
                    nq8 = vaddq_s32(nq8, nq10);
                    nq9 = vaddq_s32(nq9, nq11);

                    int16x8_t q6 = vld1q_s16(&Iptr[x]);

                    nq4 = vaddq_s32(nq4, nq8);
                    nq5 = vaddq_s32(nq5, nq9);

                    nq8 = vmovl_s16(vget_high_s16(q6));
                    nq6 = vmovl_s16(vget_low_s16(q6));

                    nq4 = vqrshlq_s32(nq4, q11);
                    nq5 = vqrshlq_s32(nq5, q11);

                    int16x8x2_t q0q1 = vld2q_s16(dIptr);
                    float32x4_t nB1v = vld1q_f32(nB1);
                    float32x4_t nB2v = vld1q_f32(nB2);

                    nq4 = vsubq_s32(nq4, nq6);
                    nq5 = vsubq_s32(nq5, nq8);

                    int32x4_t nq2 = vmovl_s16(vget_low_s16(q0q1.val[0]));
                    int32x4_t nq3 = vmovl_s16(vget_high_s16(q0q1.val[0]));

                    nq7 = vmovl_s16(vget_low_s16(q0q1.val[1]));
                    nq8 = vmovl_s16(vget_high_s16(q0q1.val[1]));

                    nq9 = vmulq_s32(nq4, nq2);
                    nq10 = vmulq_s32(nq5, nq3);

                    nq4 = vmulq_s32(nq4, nq7);
                    nq5 = vmulq_s32(nq5, nq8);

                    nq9 = vaddq_s32(nq9, nq10);
                    nq4 = vaddq_s32(nq4, nq5);

                    nB1v = vaddq_f32(nB1v, vcvtq_f32_s32(nq9));
                    nB2v = vaddq_f32(nB2v, vcvtq_f32_s32(nq4));

                    vst1q_f32(nB1, nB1v);
                    vst1q_f32(nB2, nB2v);
                }
#endif

                for( ; x < winSize.width*cn; x++, dIptr += 2 )
                {
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    ib1 += (itemtype)(diff*dIptr[0]);
                    ib2 += (itemtype)(diff*dIptr[1]);
                }
            }

#if CV_SIMD128 && !CV_NEON
            v_float32x4 qf0, qf1;
            v_recombine(v_interleave_pairs(qb0 + qb1), v_setzero_f32(), qf0, qf1);
            ib1 += v_reduce_sum(qf0);
            ib2 += v_reduce_sum(qf1);
#endif

#if CV_NEON

            ib1 += (float)(nB1[0] + nB1[1] + nB1[2] + nB1[3]);
            ib2 += (float)(nB2[0] + nB2[1] + nB2[2] + nB2[3]);
#endif

            b1 = ib1*FLT_SCALE;
            b2 = ib2*FLT_SCALE;

            Point2f delta( (float)((A12*b2 - A22*b1) * D),
                          (float)((A12*b1 - A11*b2) * D));
            //delta = -delta;

            nextPt += delta;
            nextPts[ptidx] = nextPt + halfWin;

            if( delta.ddot(delta) <= criteria.epsilon )
                break;

            if( j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
               std::abs(delta.y + prevDelta.y) < 0.01 )
            {
                nextPts[ptidx] -= delta*0.5f;
                break;
            }
            prevDelta = delta;
        }

        CV_Assert(status != NULL);
        if( status[ptidx] && err && level == 0 && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0 )
        {
            Point2f nextPoint = nextPts[ptidx] - halfWin;
            Point inextPoint;

            inextPoint.x = cvFloor(nextPoint.x);
            inextPoint.y = cvFloor(nextPoint.y);

            if( inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
                inextPoint.y < -winSize.height || inextPoint.y >= J.rows )
            {
                if( status )
                    status[ptidx] = false;
                continue;
            }

            float aa = nextPoint.x - inextPoint.x;
            float bb = nextPoint.y - inextPoint.y;
            iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
            iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
            iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float errval = 0.f;

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPoint.y)*stepJ + inextPoint.x*cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);

                for( x = 0; x < winSize.width*cn; x++ )
                {
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    errval += std::abs((float)diff);
                }
            }
            err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height);
        }
    }
}

int cv::buildOpticalFlowPyramid(InputArray _img, OutputArrayOfArrays pyramid, Size winSize, int maxLevel, bool withDerivatives,
                                int pyrBorder, int derivBorder, bool tryReuseInputImage)
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();
    CV_Assert(img.depth() == CV_8U && winSize.width > 2 && winSize.height > 2 );
    int pyrstep = withDerivatives ? 2 : 1;

    pyramid.create(1, (maxLevel + 1) * pyrstep, 0 /*type*/, -1, true, 0);

    int derivType = CV_MAKETYPE(DataType<cv::detail::deriv_type>::depth, img.channels() * 2);

    //level 0
    bool lvl0IsSet = false;
    if(tryReuseInputImage && img.isSubmatrix() && (pyrBorder & BORDER_ISOLATED) == 0)
    {
        Size wholeSize;
        Point ofs;
        img.locateROI(wholeSize, ofs);
        if (ofs.x >= winSize.width && ofs.y >= winSize.height
              && ofs.x + img.cols + winSize.width <= wholeSize.width
              && ofs.y + img.rows + winSize.height <= wholeSize.height)
        {
            pyramid.getMatRef(0) = img;
            lvl0IsSet = true;
        }
    }

    if(!lvl0IsSet)
    {
        Mat& temp = pyramid.getMatRef(0);

        if(!temp.empty())
            temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
        if(temp.type() != img.type() || temp.cols != winSize.width*2 + img.cols || temp.rows != winSize.height * 2 + img.rows)
            temp.create(img.rows + winSize.height*2, img.cols + winSize.width*2, img.type());

        if(pyrBorder == BORDER_TRANSPARENT)
            img.copyTo(temp(Rect(winSize.width, winSize.height, img.cols, img.rows)));
        else
            copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder);
        temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
    }

    Size sz = img.size();
    Mat prevLevel = pyramid.getMatRef(0);
    Mat thisLevel = prevLevel;

    for(int level = 0; level <= maxLevel; ++level)
    {
        if (level != 0)
        {
            Mat& temp = pyramid.getMatRef(level * pyrstep);

            if(!temp.empty())
                temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
            if(temp.type() != img.type() || temp.cols != winSize.width*2 + sz.width || temp.rows != winSize.height * 2 + sz.height)
                temp.create(sz.height + winSize.height*2, sz.width + winSize.width*2, img.type());

            thisLevel = temp(Rect(winSize.width, winSize.height, sz.width, sz.height));
            pyrDown(prevLevel, thisLevel, sz);

            if(pyrBorder != BORDER_TRANSPARENT)
                copyMakeBorder(thisLevel, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder|BORDER_ISOLATED);
            temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
        }

        if(withDerivatives)
        {
            Mat& deriv = pyramid.getMatRef(level * pyrstep + 1);

            if(!deriv.empty())
                deriv.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
            if(deriv.type() != derivType || deriv.cols != winSize.width*2 + sz.width || deriv.rows != winSize.height * 2 + sz.height)
                deriv.create(sz.height + winSize.height*2, sz.width + winSize.width*2, derivType);

            Mat derivI = deriv(Rect(winSize.width, winSize.height, sz.width, sz.height));
            calcScharrDeriv(thisLevel, derivI);

            if(derivBorder != BORDER_TRANSPARENT)
                copyMakeBorder(derivI, deriv, winSize.height, winSize.height, winSize.width, winSize.width, derivBorder|BORDER_ISOLATED);
            deriv.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
        }

        sz = Size((sz.width+1)/2, (sz.height+1)/2);
        if( sz.width <= winSize.width || sz.height <= winSize.height )
        {
            pyramid.create(1, (level + 1) * pyrstep, 0 /*type*/, -1, true, 0);//check this
            return level;
        }

        prevLevel = thisLevel;
    }

    return maxLevel;
}

namespace cv
{
namespace
{
    class SparsePyrLKOpticalFlowImpl : public SparsePyrLKOpticalFlow
    {
        struct dim3
        {
            unsigned int x, y, z;
            dim3() : x(0), y(0), z(0) { }
        };
    public:
        SparsePyrLKOpticalFlowImpl(Size winSize_ = Size(21,21),
                         int maxLevel_ = 3,
                         TermCriteria criteria_ = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                         int flags_ = 0,
                         double minEigThreshold_ = 1e-4) :
          winSize(winSize_), maxLevel(maxLevel_), criteria(criteria_), flags(flags_), minEigThreshold(minEigThreshold_)
#ifdef HAVE_OPENCL
          , iters(criteria_.maxCount), derivLambda(criteria_.epsilon), useInitialFlow(0 != (flags_ & OPTFLOW_LK_GET_MIN_EIGENVALS))
#endif
        {
        }

        virtual Size getWinSize() const CV_OVERRIDE { return winSize;}
        virtual void setWinSize(Size winSize_) CV_OVERRIDE { winSize = winSize_;}

        virtual int getMaxLevel() const CV_OVERRIDE { return maxLevel;}
        virtual void setMaxLevel(int maxLevel_) CV_OVERRIDE { maxLevel = maxLevel_;}

        virtual TermCriteria getTermCriteria() const CV_OVERRIDE { return criteria;}
        virtual void setTermCriteria(TermCriteria& crit_) CV_OVERRIDE { criteria=crit_;}

        virtual int getFlags() const CV_OVERRIDE { return flags; }
        virtual void setFlags(int flags_) CV_OVERRIDE { flags=flags_;}

        virtual double getMinEigThreshold() const CV_OVERRIDE { return minEigThreshold;}
        virtual void setMinEigThreshold(double minEigThreshold_) CV_OVERRIDE { minEigThreshold=minEigThreshold_;}

        virtual void calc(InputArray prevImg, InputArray nextImg,
                          InputArray prevPts, InputOutputArray nextPts,
                          OutputArray status,
                          OutputArray err = cv::noArray()) CV_OVERRIDE;

        virtual String getDefaultName() const CV_OVERRIDE { return "SparseOpticalFlow.SparsePyrLKOpticalFlow"; }

    private:
#ifdef HAVE_OPENCL
        bool checkParam()
        {
            iters = std::min(std::max(iters, 0), 100);

            derivLambda = std::min(std::max(derivLambda, 0.0), 1.0);
            if (derivLambda < 0)
                return false;
            if (maxLevel < 0 || winSize.width <= 2 || winSize.height <= 2)
                return false;
            if (winSize.width < 8 || winSize.height < 8 ||
                winSize.width > 24 || winSize.height > 24)
                return false;
            calcPatchSize();
            if (patch.x <= 0 || patch.x >= 6 || patch.y <= 0 || patch.y >= 6)
                return false;
            return true;
        }

        bool sparse(const UMat &prevImg, const UMat &nextImg, const UMat &prevPts, UMat &nextPts, UMat &status, UMat &err)
        {
            if (!checkParam())
                return false;

            UMat temp1 = (useInitialFlow ? nextPts : prevPts).reshape(1);
            UMat temp2 = nextPts.reshape(1);
            multiply(1.0f / (1 << maxLevel) /2.0f, temp1, temp2);

            status.setTo(Scalar::all(1));

            // build the image pyramids.
            std::vector<UMat> prevPyr; prevPyr.resize(maxLevel + 1);
            std::vector<UMat> nextPyr; nextPyr.resize(maxLevel + 1);

            // allocate buffers with aligned pitch to be able to use cl_khr_image2d_from_buffer extension
            // This is the required pitch alignment in pixels
            int pitchAlign = (int)ocl::Device::getDefault().imagePitchAlignment();
            if (pitchAlign>0)
            {
                prevPyr[0] = UMat(prevImg.rows,(prevImg.cols+pitchAlign-1)&(-pitchAlign),CV_32FC1).colRange(0,prevImg.cols);
                nextPyr[0] = UMat(nextImg.rows,(nextImg.cols+pitchAlign-1)&(-pitchAlign),CV_32FC1).colRange(0,nextImg.cols);
                for (int level = 1; level <= maxLevel; ++level)
                {
                    int cols,rows;
                    // allocate buffers with aligned pitch to be able to use image on buffer extension
                    cols = (prevPyr[level - 1].cols+1)/2;
                    rows = (prevPyr[level - 1].rows+1)/2;
                    prevPyr[level] = UMat(rows,(cols+pitchAlign-1)&(-pitchAlign),prevPyr[level-1].type()).colRange(0,cols);
                    cols = (nextPyr[level - 1].cols+1)/2;
                    rows = (nextPyr[level - 1].rows+1)/2;
                    nextPyr[level] = UMat(rows,(cols+pitchAlign-1)&(-pitchAlign),nextPyr[level-1].type()).colRange(0,cols);
                }
            }

            prevImg.convertTo(prevPyr[0], CV_32F);
            nextImg.convertTo(nextPyr[0], CV_32F);

            for (int level = 1; level <= maxLevel; ++level)
            {
                pyrDown(prevPyr[level - 1], prevPyr[level]);
                pyrDown(nextPyr[level - 1], nextPyr[level]);
            }

            // dI/dx ~ Ix, dI/dy ~ Iy
            for (int level = maxLevel; level >= 0; level--)
            {
                if (!lkSparse_run(prevPyr[level], nextPyr[level], prevPts,
                                  nextPts, status, err,
                                  static_cast<int>(prevPts.total()),
                                  level))
                    return false;
            }
            return true;
        }
#endif

        Size winSize;
        int maxLevel;
        TermCriteria criteria;
        int flags;
        double minEigThreshold;
#ifdef HAVE_OPENCL
        int iters;
        double derivLambda;
        bool useInitialFlow;
        dim3 patch;
        void calcPatchSize()
        {
            dim3 block;

            if (winSize.width > 32 && winSize.width > 2 * winSize.height)
            {
                block.x = 32;
                block.y = 8;
            }
            else
            {
                block.x = 16;
                block.y = 16;
            }

            patch.x = (winSize.width  + block.x - 1) / block.x;
            patch.y = (winSize.height + block.y - 1) / block.y;

            block.z = patch.z = 1;
        }

        bool lkSparse_run(UMat &I, UMat &J, const UMat &prevPts, UMat &nextPts, UMat &status, UMat& err,
            int ptcount, int level)
        {
            size_t localThreads[3]  = { 8, 8};
            size_t globalThreads[3] = { 8 * (size_t)ptcount, 8};
            char calcErr = (0 == level) ? 1 : 0;

            int wsx = 1, wsy = 1;
            if(winSize.width < 16)
                wsx = 0;
            if(winSize.height < 16)
                wsy = 0;
            cv::String build_options;
            if (isDeviceCPU())
                build_options = " -D CPU";
            else
                build_options = cv::format("-D WSX=%d -D WSY=%d",
                                           wsx, wsy);

            ocl::Kernel kernel;
            if (!kernel.create("lkSparse", cv::ocl::video::pyrlk_oclsrc, build_options))
                return false;

            CV_Assert(I.depth() == CV_32F && J.depth() == CV_32F);
            ocl::Image2D imageI(I, false, ocl::Image2D::canCreateAlias(I));
            ocl::Image2D imageJ(J, false, ocl::Image2D::canCreateAlias(J));

            int idxArg = 0;
            idxArg = kernel.set(idxArg, imageI); //image2d_t I
            idxArg = kernel.set(idxArg, imageJ); //image2d_t J
            idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadOnly(prevPts)); // __global const float2* prevPts
            idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadWrite(nextPts)); // __global const float2* nextPts
            idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadWrite(status)); // __global uchar* status
            idxArg = kernel.set(idxArg, ocl::KernelArg::PtrReadWrite(err)); // __global float* err
            idxArg = kernel.set(idxArg, (int)level); // const int level
            idxArg = kernel.set(idxArg, (int)I.rows); // const int rows
            idxArg = kernel.set(idxArg, (int)I.cols); // const int cols
            idxArg = kernel.set(idxArg, (int)patch.x); // int PATCH_X
            idxArg = kernel.set(idxArg, (int)patch.y); // int PATCH_Y
            idxArg = kernel.set(idxArg, (int)winSize.width); // int c_winSize_x
            idxArg = kernel.set(idxArg, (int)winSize.height); // int c_winSize_y
            idxArg = kernel.set(idxArg, (int)iters); // int c_iters
            idxArg = kernel.set(idxArg, (char)calcErr); //char calcErr
            return kernel.run(2, globalThreads, localThreads, true); // sync=true because ocl::Image2D lifetime is not handled well for temp UMat
        }
    private:
        inline static bool isDeviceCPU()
        {
            return (cv::ocl::Device::TYPE_CPU == cv::ocl::Device::getDefault().type());
        }


    bool ocl_calcOpticalFlowPyrLK(InputArray _prevImg, InputArray _nextImg,
                                         InputArray _prevPts, InputOutputArray _nextPts,
                                         OutputArray _status, OutputArray _err)
    {
        if (0 != (OPTFLOW_LK_GET_MIN_EIGENVALS & flags))
            return false;
        if (!cv::ocl::Device::getDefault().imageSupport())
            return false;
        if (_nextImg.size() != _prevImg.size())
            return false;
        int typePrev = _prevImg.type();
        int typeNext = _nextImg.type();
        if ((1 != CV_MAT_CN(typePrev)) || (1 != CV_MAT_CN(typeNext)))
            return false;
        if ((0 != CV_MAT_DEPTH(typePrev)) || (0 != CV_MAT_DEPTH(typeNext)))
            return false;

        if (_prevPts.empty() || _prevPts.type() != CV_32FC2 || (!_prevPts.isContinuous()))
            return false;
        if ((1 != _prevPts.size().height) && (1 != _prevPts.size().width))
            return false;
        size_t npoints = _prevPts.total();
        if (useInitialFlow)
        {
            if (_nextPts.empty() || _nextPts.type() != CV_32FC2 || (!_prevPts.isContinuous()))
                return false;
            if ((1 != _nextPts.size().height) && (1 != _nextPts.size().width))
                return false;
            if (_nextPts.total() != npoints)
                return false;
        }
        else
        {
            _nextPts.create(_prevPts.size(), _prevPts.type());
        }

        if (!checkParam())
            return false;

        UMat umatErr;
        if (_err.needed())
        {
            _err.create((int)npoints, 1, CV_32FC1);
            umatErr = _err.getUMat();
        }
        else
            umatErr.create((int)npoints, 1, CV_32FC1);

        _status.create((int)npoints, 1, CV_8UC1);
        UMat umatNextPts = _nextPts.getUMat();
        UMat umatStatus = _status.getUMat();
        UMat umatPrevPts;
        _prevPts.getMat().copyTo(umatPrevPts);
        return sparse(_prevImg.getUMat(), _nextImg.getUMat(), umatPrevPts, umatNextPts, umatStatus, umatErr);
    }
#endif

#ifdef HAVE_OPENVX
    bool openvx_pyrlk(InputArray _prevImg, InputArray _nextImg, InputArray _prevPts, InputOutputArray _nextPts,
                             OutputArray _status, OutputArray _err)
    {
        using namespace ivx;

        // Pyramids as inputs are not acceptable because there's no (direct or simple) way
        // to build vx_pyramid on user data
        if(_prevImg.kind() != _InputArray::MAT || _nextImg.kind() != _InputArray::MAT)
            return false;

        Mat prevImgMat = _prevImg.getMat(), nextImgMat = _nextImg.getMat();

        if(prevImgMat.type() != CV_8UC1 || nextImgMat.type() != CV_8UC1)
            return false;

        if (ovx::skipSmallImages<VX_KERNEL_OPTICAL_FLOW_PYR_LK>(prevImgMat.cols, prevImgMat.rows))
            return false;

        CV_Assert(prevImgMat.size() == nextImgMat.size());
        Mat prevPtsMat = _prevPts.getMat();
        int checkPrev = prevPtsMat.checkVector(2, CV_32F, false);
        CV_Assert( checkPrev >= 0 );
        size_t npoints = checkPrev;

        if( !(flags & OPTFLOW_USE_INITIAL_FLOW) )
            _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);
        Mat nextPtsMat = _nextPts.getMat();
        CV_Assert( nextPtsMat.checkVector(2, CV_32F, false) == (int)npoints );

        _status.create((int)npoints, 1, CV_8U, -1, true);
        Mat statusMat = _status.getMat();
        uchar* status = statusMat.ptr();
        for(size_t i = 0; i < npoints; i++ )
            status[i] = true;

        // OpenVX doesn't return detection errors
        if( _err.needed() )
        {
            return false;
        }

        try
        {
            Context context = ovx::getOpenVXContext();

            if(context.vendorID() == VX_ID_KHRONOS)
            {
                // PyrLK in OVX 1.0.1 performs vxCommitImagePatch incorrecty and crashes
                if(VX_VERSION == VX_VERSION_1_0)
                    return false;
                // Implementation ignores border mode
                // So check that minimal size of image in pyramid is big enough
                int width = prevImgMat.cols, height = prevImgMat.rows;
                for(int i = 0; i < maxLevel+1; i++)
                {
                    if(width < winSize.width + 1 || height < winSize.height + 1)
                        return false;
                    else
                    {
                        width /= 2; height /= 2;
                    }
                }
            }

            Image prevImg = Image::createFromHandle(context, Image::matTypeToFormat(prevImgMat.type()),
                                                    Image::createAddressing(prevImgMat), (void*)prevImgMat.data);
            Image nextImg = Image::createFromHandle(context, Image::matTypeToFormat(nextImgMat.type()),
                                                    Image::createAddressing(nextImgMat), (void*)nextImgMat.data);

            Graph graph = Graph::create(context);

            Pyramid prevPyr = Pyramid::createVirtual(graph, (vx_size)maxLevel+1, VX_SCALE_PYRAMID_HALF,
                                                     prevImg.width(), prevImg.height(), prevImg.format());
            Pyramid nextPyr = Pyramid::createVirtual(graph, (vx_size)maxLevel+1, VX_SCALE_PYRAMID_HALF,
                                                     nextImg.width(), nextImg.height(), nextImg.format());

            ivx::Node::create(graph, VX_KERNEL_GAUSSIAN_PYRAMID, prevImg, prevPyr);
            ivx::Node::create(graph, VX_KERNEL_GAUSSIAN_PYRAMID, nextImg, nextPyr);

            Array prevPts = Array::create(context, VX_TYPE_KEYPOINT, npoints);
            Array estimatedPts = Array::create(context, VX_TYPE_KEYPOINT, npoints);
            Array nextPts = Array::create(context, VX_TYPE_KEYPOINT, npoints);

            std::vector<vx_keypoint_t> vxPrevPts(npoints), vxEstPts(npoints), vxNextPts(npoints);
            for(size_t i = 0; i < npoints; i++)
            {
                vx_keypoint_t& prevPt = vxPrevPts[i]; vx_keypoint_t& estPt  = vxEstPts[i];
                prevPt.x = prevPtsMat.at<Point2f>(i).x; prevPt.y = prevPtsMat.at<Point2f>(i).y;
                 estPt.x = nextPtsMat.at<Point2f>(i).x;  estPt.y = nextPtsMat.at<Point2f>(i).y;
                prevPt.tracking_status = estPt.tracking_status = vx_true_e;
            }
            prevPts.addItems(vxPrevPts); estimatedPts.addItems(vxEstPts);

            if( (criteria.type & TermCriteria::COUNT) == 0 )
                criteria.maxCount = 30;
            else
                criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
            if( (criteria.type & TermCriteria::EPS) == 0 )
                criteria.epsilon = 0.01;
            else
                criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
            criteria.epsilon *= criteria.epsilon;

            vx_enum termEnum = (criteria.type == TermCriteria::COUNT) ? VX_TERM_CRITERIA_ITERATIONS :
                               (criteria.type == TermCriteria::EPS) ? VX_TERM_CRITERIA_EPSILON :
                               VX_TERM_CRITERIA_BOTH;

            //minEigThreshold is fixed to 0.0001f
            ivx::Scalar termination = ivx::Scalar::create<VX_TYPE_ENUM>(context, termEnum);
            ivx::Scalar epsilon = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, criteria.epsilon);
            ivx::Scalar numIterations = ivx::Scalar::create<VX_TYPE_UINT32>(context, criteria.maxCount);
            ivx::Scalar useInitial = ivx::Scalar::create<VX_TYPE_BOOL>(context, (vx_bool)(flags & OPTFLOW_USE_INITIAL_FLOW));
            //assume winSize is square
            ivx::Scalar windowSize = ivx::Scalar::create<VX_TYPE_SIZE>(context, (vx_size)winSize.width);

            ivx::Node::create(graph, VX_KERNEL_OPTICAL_FLOW_PYR_LK, prevPyr, nextPyr, prevPts, estimatedPts,
                              nextPts, termination, epsilon, numIterations, useInitial, windowSize);

            graph.verify();
            graph.process();

            nextPts.copyTo(vxNextPts);
            for(size_t i = 0; i < npoints; i++)
            {
                vx_keypoint_t kp = vxNextPts[i];
                nextPtsMat.at<Point2f>(i) = Point2f(kp.x, kp.y);
                statusMat.at<uchar>(i) = (bool)kp.tracking_status;
            }

#ifdef VX_VERSION_1_1
        //we should take user memory back before release
        //(it's not done automatically according to standard)
        prevImg.swapHandle(); nextImg.swapHandle();
#endif
        }
        catch (const RuntimeError & e)
        {
            VX_DbgThrow(e.what());
        }
        catch (const WrapperError & e)
        {
            VX_DbgThrow(e.what());
        }

        return true;
    }
#endif
};



void SparsePyrLKOpticalFlowImpl::calc( InputArray _prevImg, InputArray _nextImg,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err)
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(ocl::isOpenCLActivated() &&
               (_prevImg.isUMat() || _nextImg.isUMat()) &&
               ocl::Image2D::isFormatSupported(CV_32F, 1, false),
               ocl_calcOpticalFlowPyrLK(_prevImg, _nextImg, _prevPts, _nextPts, _status, _err))

    // Disabled due to bad accuracy
    CV_OVX_RUN(false,
               openvx_pyrlk(_prevImg, _nextImg, _prevPts, _nextPts, _status, _err))

    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<cv::detail::deriv_type>::depth;

    CV_Assert( maxLevel >= 0 && winSize.width > 2 && winSize.height > 2 );

    int level=0, i, npoints;
    CV_Assert( (npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0 );

    if( npoints == 0 )
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    if( !(flags & OPTFLOW_USE_INITIAL_FLOW) )
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert( nextPtsMat.checkVector(2, CV_32F, true) == npoints );

    const Point2f* prevPts = prevPtsMat.ptr<Point2f>();
    Point2f* nextPts = nextPtsMat.ptr<Point2f>();

    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.ptr();
    float* err = 0;

    for( i = 0; i < npoints; i++ )
        status[i] = true;

    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = errMat.ptr<float>();
    }

    std::vector<Mat> prevPyr, nextPyr;
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    if(_prevImg.kind() == _InputArray::STD_VECTOR_MAT)
    {
        _prevImg.getMatVector(prevPyr);

        levels1 = int(prevPyr.size()) - 1;
        CV_Assert(levels1 >= 0);

        if (levels1 % 2 == 1 && prevPyr[0].channels() * 2 == prevPyr[1].channels() && prevPyr[1].depth() == derivDepth)
        {
            lvlStep1 = 2;
            levels1 /= 2;
        }

        // ensure that pyramid has required padding
        if(levels1 > 0)
        {
            Size fullSize;
            Point ofs;
            prevPyr[lvlStep1].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
                && ofs.x + prevPyr[lvlStep1].cols + winSize.width <= fullSize.width
                && ofs.y + prevPyr[lvlStep1].rows + winSize.height <= fullSize.height);
        }

        if(levels1 < maxLevel)
            maxLevel = levels1;
    }

    if(_nextImg.kind() == _InputArray::STD_VECTOR_MAT)
    {
        _nextImg.getMatVector(nextPyr);

        levels2 = int(nextPyr.size()) - 1;
        CV_Assert(levels2 >= 0);

        if (levels2 % 2 == 1 && nextPyr[0].channels() * 2 == nextPyr[1].channels() && nextPyr[1].depth() == derivDepth)
        {
            lvlStep2 = 2;
            levels2 /= 2;
        }

        // ensure that pyramid has required padding
        if(levels2 > 0)
        {
            Size fullSize;
            Point ofs;
            nextPyr[lvlStep2].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
                && ofs.x + nextPyr[lvlStep2].cols + winSize.width <= fullSize.width
                && ofs.y + nextPyr[lvlStep2].rows + winSize.height <= fullSize.height);
        }

        if(levels2 < maxLevel)
            maxLevel = levels2;
    }

    if (levels1 < 0)
        maxLevel = buildOpticalFlowPyramid(_prevImg, prevPyr, winSize, maxLevel, false);

    if (levels2 < 0)
        maxLevel = buildOpticalFlowPyramid(_nextImg, nextPyr, winSize, maxLevel, false);

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    // dI/dx ~ Ix, dI/dy ~ Iy
    Mat derivIBuf;
    if(lvlStep1 == 1)
        derivIBuf.create(prevPyr[0].rows + winSize.height*2, prevPyr[0].cols + winSize.width*2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));

    for( level = maxLevel; level >= 0; level-- )
    {
        Mat derivI;
        if(lvlStep1 == 1)
        {
            Size imgSize = prevPyr[level * lvlStep1].size();
            Mat _derivI( imgSize.height + winSize.height*2,
                imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.ptr() );
            derivI = _derivI(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
            calcScharrDeriv(prevPyr[level * lvlStep1], derivI);
            copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_CONSTANT|BORDER_ISOLATED);
        }
        else
            derivI = prevPyr[level * lvlStep1 + 1];

        CV_Assert(prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size());
        CV_Assert(prevPyr[level * lvlStep1].type() == nextPyr[level * lvlStep2].type());

#ifdef HAVE_TEGRA_OPTIMIZATION
        typedef tegra::LKTrackerInvoker<cv::detail::LKTrackerInvoker> LKTrackerInvoker;
#else
        typedef cv::detail::LKTrackerInvoker LKTrackerInvoker;
#endif

        parallel_for_(Range(0, npoints), LKTrackerInvoker(prevPyr[level * lvlStep1], derivI,
                                                          nextPyr[level * lvlStep2], prevPts, nextPts,
                                                          status, err,
                                                          winSize, criteria, level, maxLevel,
                                                          flags, (float)minEigThreshold));
    }
}

} // namespace
} // namespace cv
cv::Ptr<cv::SparsePyrLKOpticalFlow> cv::SparsePyrLKOpticalFlow::create(Size winSize, int maxLevel, TermCriteria crit, int flags, double minEigThreshold){
    return makePtr<SparsePyrLKOpticalFlowImpl>(winSize,maxLevel,crit,flags,minEigThreshold);
}
void cv::calcOpticalFlowPyrLK( InputArray _prevImg, InputArray _nextImg,
                               InputArray _prevPts, InputOutputArray _nextPts,
                               OutputArray _status, OutputArray _err,
                               Size winSize, int maxLevel,
                               TermCriteria criteria,
                               int flags, double minEigThreshold )
{
    Ptr<cv::SparsePyrLKOpticalFlow> optflow = cv::SparsePyrLKOpticalFlow::create(winSize,maxLevel,criteria,flags,minEigThreshold);
    optflow->calc(_prevImg,_nextImg,_prevPts,_nextPts,_status,_err);
}

namespace cv
{

static void
getRTMatrix( const std::vector<Point2f> a, const std::vector<Point2f> b,
             int count, Mat& M, bool fullAffine )
{
    CV_Assert( M.isContinuous() );

    if( fullAffine )
    {
        double sa[6][6]={{0.}}, sb[6]={0.};
        Mat A( 6, 6, CV_64F, &sa[0][0] ), B( 6, 1, CV_64F, sb );
        Mat MM = M.reshape(1, 6);

        for( int i = 0; i < count; i++ )
        {
            sa[0][0] += a[i].x*a[i].x;
            sa[0][1] += a[i].y*a[i].x;
            sa[0][2] += a[i].x;

            sa[1][1] += a[i].y*a[i].y;
            sa[1][2] += a[i].y;

            sb[0] += a[i].x*b[i].x;
            sb[1] += a[i].y*b[i].x;
            sb[2] += b[i].x;
            sb[3] += a[i].x*b[i].y;
            sb[4] += a[i].y*b[i].y;
            sb[5] += b[i].y;
        }

        sa[3][4] = sa[4][3] = sa[1][0] = sa[0][1];
        sa[3][5] = sa[5][3] = sa[2][0] = sa[0][2];
        sa[4][5] = sa[5][4] = sa[2][1] = sa[1][2];

        sa[3][3] = sa[0][0];
        sa[4][4] = sa[1][1];
        sa[5][5] = sa[2][2] = count;

        solve( A, B, MM, DECOMP_EIG );
    }
    else
    {
        double sa[4][4]={{0.}}, sb[4]={0.}, m[4] = {0};
        Mat A( 4, 4, CV_64F, sa ), B( 4, 1, CV_64F, sb );
        Mat MM( 4, 1, CV_64F, m );

        for( int i = 0; i < count; i++ )
        {
            sa[0][0] += a[i].x*a[i].x + a[i].y*a[i].y;
            sa[0][2] += a[i].x;
            sa[0][3] += a[i].y;

            sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
            sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
            sb[2] += b[i].x;
            sb[3] += b[i].y;
        }

        sa[1][1] = sa[0][0];
        sa[2][1] = sa[1][2] = -sa[0][3];
        sa[3][1] = sa[1][3] = sa[2][0] = sa[0][2];
        sa[2][2] = sa[3][3] = count;
        sa[3][0] = sa[0][3];

        solve( A, B, MM, DECOMP_EIG );

        double* om = M.ptr<double>();
        om[0] = om[4] = m[0];
        om[1] = -m[1];
        om[3] = m[1];
        om[2] = m[2];
        om[5] = m[3];
    }
}

}

cv::Mat cv::estimateRigidTransform( InputArray src1, InputArray src2, bool fullAffine )
{
    return estimateRigidTransform(src1, src2, fullAffine, 500, 0.5, 3);
}

cv::Mat cv::estimateRigidTransform( InputArray src1, InputArray src2, bool fullAffine, int ransacMaxIters, double ransacGoodRatio,
                                    const int ransacSize0)
{
    CV_INSTRUMENT_REGION();

    Mat M(2, 3, CV_64F), A = src1.getMat(), B = src2.getMat();

    const int COUNT = 15;
    const int WIDTH = 160, HEIGHT = 120;

    std::vector<Point2f> pA, pB;
    std::vector<int> good_idx;
    std::vector<uchar> status;

    double scale = 1.;
    int i, j, k, k1;

    RNG rng((uint64)-1);
    int good_count = 0;

    if( ransacSize0 < 3 )
        CV_Error( Error::StsBadArg, "ransacSize0 should have value bigger than 2.");

    if( ransacGoodRatio > 1 || ransacGoodRatio < 0)
        CV_Error( Error::StsBadArg, "ransacGoodRatio should have value between 0 and 1");

    if( A.size() != B.size() )
        CV_Error( Error::StsUnmatchedSizes, "Both input images must have the same size" );

    if( A.type() != B.type() )
        CV_Error( Error::StsUnmatchedFormats, "Both input images must have the same data type" );

    int count = A.checkVector(2);

    if( count > 0 )
    {
        A.reshape(2, count).convertTo(pA, CV_32F);
        B.reshape(2, count).convertTo(pB, CV_32F);
    }
    else if( A.depth() == CV_8U )
    {
        int cn = A.channels();
        CV_Assert( cn == 1 || cn == 3 || cn == 4 );
        Size sz0 = A.size();
        Size sz1(WIDTH, HEIGHT);

        scale = std::max(1., std::max( (double)sz1.width/sz0.width, (double)sz1.height/sz0.height ));

        sz1.width = cvRound( sz0.width * scale );
        sz1.height = cvRound( sz0.height * scale );

        bool equalSizes = sz1.width == sz0.width && sz1.height == sz0.height;

        if( !equalSizes || cn != 1 )
        {
            Mat sA, sB;

            if( cn != 1 )
            {
                Mat gray;
                cvtColor(A, gray, COLOR_BGR2GRAY);
                resize(gray, sA, sz1, 0., 0., INTER_AREA);
                cvtColor(B, gray, COLOR_BGR2GRAY);
                resize(gray, sB, sz1, 0., 0., INTER_AREA);
            }
            else
            {
                resize(A, sA, sz1, 0., 0., INTER_AREA);
                resize(B, sB, sz1, 0., 0., INTER_AREA);
            }

            A = sA;
            B = sB;
        }

        int count_y = COUNT;
        int count_x = cvRound((double)COUNT*sz1.width/sz1.height);
        count = count_x * count_y;

        pA.resize(count);
        pB.resize(count);
        status.resize(count);

        for( i = 0, k = 0; i < count_y; i++ )
            for( j = 0; j < count_x; j++, k++ )
            {
                pA[k].x = (j+0.5f)*sz1.width/count_x;
                pA[k].y = (i+0.5f)*sz1.height/count_y;
            }

        // find the corresponding points in B
        calcOpticalFlowPyrLK(A, B, pA, pB, status, noArray(), Size(21, 21), 3,
                             TermCriteria(TermCriteria::MAX_ITER,40,0.1));

        // repack the remained points
        for( i = 0, k = 0; i < count; i++ )
            if( status[i] )
            {
                if( i > k )
                {
                    pA[k] = pA[i];
                    pB[k] = pB[i];
                }
                k++;
            }
        count = k;
        pA.resize(count);
        pB.resize(count);
    }
    else
        CV_Error( Error::StsUnsupportedFormat, "Both input images must have either 8uC1 or 8uC3 type" );

    good_idx.resize(count);

    if( count < ransacSize0 )
        return Mat();

    Rect brect = boundingRect(pB);

    std::vector<Point2f> a(ransacSize0);
    std::vector<Point2f> b(ransacSize0);

    // RANSAC stuff:
    // 1. find the consensus
    for( k = 0; k < ransacMaxIters; k++ )
    {
        std::vector<int> idx(ransacSize0);
        // choose random 3 non-complanar points from A & B
        for( i = 0; i < ransacSize0; i++ )
        {
            for( k1 = 0; k1 < ransacMaxIters; k1++ )
            {
                idx[i] = rng.uniform(0, count);

                for( j = 0; j < i; j++ )
                {
                    if( idx[j] == idx[i] )
                        break;
                    // check that the points are not very close one each other
                    if( fabs(pA[idx[i]].x - pA[idx[j]].x) +
                        fabs(pA[idx[i]].y - pA[idx[j]].y) < FLT_EPSILON )
                        break;
                    if( fabs(pB[idx[i]].x - pB[idx[j]].x) +
                        fabs(pB[idx[i]].y - pB[idx[j]].y) < FLT_EPSILON )
                        break;
                }

                if( j < i )
                    continue;

                if( i+1 == ransacSize0 )
                {
                    // additional check for non-complanar vectors
                    a[0] = pA[idx[0]];
                    a[1] = pA[idx[1]];
                    a[2] = pA[idx[2]];

                    b[0] = pB[idx[0]];
                    b[1] = pB[idx[1]];
                    b[2] = pB[idx[2]];

                    double dax1 = a[1].x - a[0].x, day1 = a[1].y - a[0].y;
                    double dax2 = a[2].x - a[0].x, day2 = a[2].y - a[0].y;
                    double dbx1 = b[1].x - b[0].x, dby1 = b[1].y - b[0].y;
                    double dbx2 = b[2].x - b[0].x, dby2 = b[2].y - b[0].y;
                    const double eps = 0.01;

                    if( fabs(dax1*day2 - day1*dax2) < eps*std::sqrt(dax1*dax1+day1*day1)*std::sqrt(dax2*dax2+day2*day2) ||
                        fabs(dbx1*dby2 - dby1*dbx2) < eps*std::sqrt(dbx1*dbx1+dby1*dby1)*std::sqrt(dbx2*dbx2+dby2*dby2) )
                        continue;
                }
                break;
            }

            if( k1 >= ransacMaxIters )
                break;
        }

        if( i < ransacSize0 )
            continue;

        // estimate the transformation using 3 points
        getRTMatrix( a, b, 3, M, fullAffine );

        const double* m = M.ptr<double>();
        for( i = 0, good_count = 0; i < count; i++ )
        {
            if( std::abs( m[0]*pA[i].x + m[1]*pA[i].y + m[2] - pB[i].x ) +
                std::abs( m[3]*pA[i].x + m[4]*pA[i].y + m[5] - pB[i].y ) < std::max(brect.width,brect.height)*0.05 )
                good_idx[good_count++] = i;
        }

        if( good_count >= count*ransacGoodRatio )
            break;
    }

    if( k >= ransacMaxIters )
        return Mat();

    if( good_count < count )
    {
        for( i = 0; i < good_count; i++ )
        {
            j = good_idx[i];
            pA[i] = pA[j];
            pB[i] = pB[j];
        }
    }

    getRTMatrix( pA, pB, good_count, M, fullAffine );
    M.at<double>(0, 2) /= scale;
    M.at<double>(1, 2) /= scale;

    return M;
}

/* End of file. */
