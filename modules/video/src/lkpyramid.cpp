/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

namespace cv
{

typedef short deriv_type;
    
static void calcSharrDeriv(const Mat& src, Mat& dst)
{
    int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols*cn, depth = src.depth();
    CV_Assert(depth == CV_8U);
    dst.create(rows, cols, CV_MAKETYPE(DataType<deriv_type>::depth, cn*2));
    
    int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);
    
#if CV_SSE2
    __m128i z = _mm_setzero_si128(), c3 = _mm_set1_epi16(3), c10 = _mm_set1_epi16(10);
#endif
    
    for( y = 0; y < rows; y++ )
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        deriv_type* drow = dst.ptr<deriv_type>(y);
        
        // do vertical convolution
        x = 0;
#if CV_SSE2
        for( ; x <= colsn - 8; x += 8 )
        {
            __m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
            __m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
            __m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
            __m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s0, s2), c3), _mm_mullo_epi16(s1, c10));
            __m128i t1 = _mm_sub_epi16(s2, s0);
            _mm_store_si128((__m128i*)(trow0 + x), t0);
            _mm_store_si128((__m128i*)(trow1 + x), t1);
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
#if CV_SSE2
        for( ; x <= colsn - 8; x += 8 )
        {
            __m128i s0 = _mm_loadu_si128((const __m128i*)(trow0 + x - cn));
            __m128i s1 = _mm_loadu_si128((const __m128i*)(trow0 + x + cn));
            __m128i s2 = _mm_loadu_si128((const __m128i*)(trow1 + x - cn));
            __m128i s3 = _mm_load_si128((const __m128i*)(trow1 + x));
            __m128i s4 = _mm_loadu_si128((const __m128i*)(trow1 + x + cn));
            
            __m128i t0 = _mm_sub_epi16(s1, s0);
            __m128i t1 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s2, s4), c3), _mm_mullo_epi16(s3, c10));
            __m128i t2 = _mm_unpacklo_epi16(t0, t1);
            t0 = _mm_unpackhi_epi16(t0, t1);
            // this can probably be replaced with aligned stores if we aligned dst properly.
            _mm_storeu_si128((__m128i*)(drow + x*2), t2);
            _mm_storeu_si128((__m128i*)(drow + x*2 + 8), t0);
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

    
struct LKTrackerInvoker
{
    LKTrackerInvoker( const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
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
    
    void operator()(const BlockedRange& range) const
    {
        Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
        const Mat& I = *prevImg;
        const Mat& J = *nextImg;
        const Mat& derivI = *prevDeriv;
        
        int j, cn = I.channels(), cn2 = cn*2;
        cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2));
        int derivDepth = DataType<deriv_type>::depth;
        
        Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf);
        Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf + winSize.area()*cn);
        
        for( int ptidx = range.begin(); ptidx < range.end(); ptidx++ )
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
            int step = (int)(I.step/I.elemSize1());
            CV_Assert( step == (int)(J.step/J.elemSize1()) );
            float A11 = 0, A12 = 0, A22 = 0;
            
#if CV_SSE2
            __m128i qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
            __m128i qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
            __m128i z = _mm_setzero_si128();
            __m128i qdelta_d = _mm_set1_epi32(1 << (W_BITS1-1));
            __m128i qdelta = _mm_set1_epi32(1 << (W_BITS1-5-1));
            __m128 qA11 = _mm_setzero_ps(), qA12 = _mm_setzero_ps(), qA22 = _mm_setzero_ps();
#endif
            
            // extract the patch from the first image, compute covariation matrix of derivatives
            int x, y;
            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* src = (const uchar*)I.data + (y + iprevPt.y)*step + iprevPt.x*cn;
                const deriv_type* dsrc = (const deriv_type*)derivI.data + (y + iprevPt.y)*dstep + iprevPt.x*cn2;
                
                deriv_type* Iptr = (deriv_type*)(IWinBuf.data + y*IWinBuf.step);
                deriv_type* dIptr = (deriv_type*)(derivIWinBuf.data + y*derivIWinBuf.step);
                
                x = 0;
                
#if CV_SSE2
                for( ; x <= winSize.width*cn - 4; x += 4, dsrc += 4*2, dIptr += 4*2 )
                {
                    __m128i v00, v01, v10, v11, t0, t1;
                    
                    v00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x)), z);
                    v01 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + cn)), z);
                    v10 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + step)), z);
                    v11 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + step + cn)), z);
                    
                    t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                    t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
                    _mm_storel_epi64((__m128i*)(Iptr + x), _mm_packs_epi32(t0,t0));
                    
                    v00 = _mm_loadu_si128((const __m128i*)(dsrc));
                    v01 = _mm_loadu_si128((const __m128i*)(dsrc + cn2));
                    v10 = _mm_loadu_si128((const __m128i*)(dsrc + dstep));
                    v11 = _mm_loadu_si128((const __m128i*)(dsrc + dstep + cn2));
                    
                    t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                    t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
                    t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta_d), W_BITS1);
                    t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta_d), W_BITS1);
                    v00 = _mm_packs_epi32(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                    
                    _mm_storeu_si128((__m128i*)dIptr, v00);
                    t0 = _mm_srai_epi32(v00, 16); // Iy0 Iy1 Iy2 Iy3
                    t1 = _mm_srai_epi32(_mm_slli_epi32(v00, 16), 16); // Ix0 Ix1 Ix2 Ix3
                    
                    __m128 fy = _mm_cvtepi32_ps(t0);
                    __m128 fx = _mm_cvtepi32_ps(t1);
                    
                    qA22 = _mm_add_ps(qA22, _mm_mul_ps(fy, fy));
                    qA12 = _mm_add_ps(qA12, _mm_mul_ps(fx, fy));
                    qA11 = _mm_add_ps(qA11, _mm_mul_ps(fx, fx));
                }
#endif
                
                for( ; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2 )
                {
                    int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                          src[x+step]*iw10 + src[x+step+cn]*iw11, W_BITS1-5);
                    int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                           dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
                    int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                           dsrc[dstep+cn2+1]*iw11, W_BITS1);
                    
                    Iptr[x] = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;
                    
                    A11 += (float)(ixval*ixval);
                    A12 += (float)(ixval*iyval);
                    A22 += (float)(iyval*iyval);
                }
            }
            
#if CV_SSE2
            float CV_DECL_ALIGNED(16) A11buf[4], A12buf[4], A22buf[4];
            _mm_store_ps(A11buf, qA11);
            _mm_store_ps(A12buf, qA12);
            _mm_store_ps(A22buf, qA22);
            A11 += A11buf[0] + A11buf[1] + A11buf[2] + A11buf[3];
            A12 += A12buf[0] + A12buf[1] + A12buf[2] + A12buf[3];
            A22 += A22buf[0] + A22buf[1] + A22buf[2] + A22buf[3];
#endif
            
            A11 *= FLT_SCALE;
            A12 *= FLT_SCALE;
            A22 *= FLT_SCALE;
            
            float D = A11*A22 - A12*A12;
            float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                            4.f*A12*A12))/(2*winSize.width*winSize.height);
            
            if( err && (flags & CV_LKFLOW_GET_MIN_EIGENVALS) != 0 )
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
                float b1 = 0, b2 = 0;
#if CV_SSE2
                qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
                qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
                __m128 qb0 = _mm_setzero_ps(), qb1 = _mm_setzero_ps();
#endif
                
                for( y = 0; y < winSize.height; y++ )
                {
                    const uchar* Jptr = (const uchar*)J.data + (y + inextPt.y)*step + inextPt.x*cn;
                    const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data + y*IWinBuf.step);
                    const deriv_type* dIptr = (const deriv_type*)(derivIWinBuf.data + y*derivIWinBuf.step);
                    
                    x = 0;
                    
#if CV_SSE2
                    for( ; x <= winSize.width*cn - 8; x += 8, dIptr += 8*2 )
                    {
                        __m128i diff0 = _mm_loadu_si128((const __m128i*)(Iptr + x)), diff1;
                        __m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x)), z);
                        __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + cn)), z);
                        __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + step)), z);
                        __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + step + cn)), z);
                        
                        __m128i t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                                   _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                        __m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                                   _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
                        t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
                        t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS1-5);
                        diff0 = _mm_subs_epi16(_mm_packs_epi32(t0, t1), diff0);
                        diff1 = _mm_unpackhi_epi16(diff0, diff0);
                        diff0 = _mm_unpacklo_epi16(diff0, diff0); // It0 It0 It1 It1 ...
                        v00 = _mm_loadu_si128((const __m128i*)(dIptr)); // Ix0 Iy0 Ix1 Iy1 ... 
                        v01 = _mm_loadu_si128((const __m128i*)(dIptr + 8));
                        v10 = _mm_mullo_epi16(v00, diff0);
                        v11 = _mm_mulhi_epi16(v00, diff0);
                        v00 = _mm_unpacklo_epi16(v10, v11);
                        v10 = _mm_unpackhi_epi16(v10, v11);
                        qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
                        qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
                        v10 = _mm_mullo_epi16(v01, diff1);
                        v11 = _mm_mulhi_epi16(v01, diff1);
                        v00 = _mm_unpacklo_epi16(v10, v11);
                        v10 = _mm_unpackhi_epi16(v10, v11);
                        qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
                        qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
                    }
#endif
                    
                    for( ; x < winSize.width*cn; x++, dIptr += 2 )
                    {
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                              Jptr[x+step]*iw10 + Jptr[x+step+cn]*iw11,
                                              W_BITS1-5) - Iptr[x];
                        b1 += (float)(diff*dIptr[0]);
                        b2 += (float)(diff*dIptr[1]);
                    }
                }
                
#if CV_SSE2
                float CV_DECL_ALIGNED(16) bbuf[4];
                _mm_store_ps(bbuf, _mm_add_ps(qb0, qb1));
                b1 += bbuf[0] + bbuf[2];
                b2 += bbuf[1] + bbuf[3];
#endif

                b1 *= FLT_SCALE;
                b2 *= FLT_SCALE;
                
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
            
            if( status[ptidx] && err && level == 0 && (flags & CV_LKFLOW_GET_MIN_EIGENVALS) == 0 )
            {
                Point2f nextPt = nextPts[ptidx];
                Point inextPt;
                
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);
                
                if( inextPt.x < -winSize.width || inextPt.x >= J.cols ||
                    inextPt.y < -winSize.height || inextPt.y >= J.rows )
                {
                    if( status )
                        status[ptidx] = false;
                    continue;
                }
                
                float a = nextPt.x - inextPt.x;
                float b = nextPt.y - inextPt.y;
                iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                float errval = 0.f;
                
                for( y = 0; y < winSize.height; y++ )
                {
                    const uchar* Jptr = (const uchar*)J.data + (y + inextPt.y)*step + inextPt.x*cn;
                    const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data + y*IWinBuf.step);
                    
                    for( x = 0; x < winSize.width*cn; x++ )
                    {
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                              Jptr[x+step]*iw10 + Jptr[x+step+cn]*iw11,
                                              W_BITS1-5) - Iptr[x];
                        errval += std::abs((float)diff);
                    }
                }
                err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height);
            }
        }
    }
    
    const Mat* prevImg;
    const Mat* nextImg;
    const Mat* prevDeriv;
    const Point2f* prevPts;
    Point2f* nextPts;
    uchar* status;
    float* err;
    Size winSize;
    TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
};
    
}

void cv::calcOpticalFlowPyrLK( InputArray _prevImg, InputArray _nextImg,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err,
                           Size winSize, int maxLevel,
                           TermCriteria criteria,
                           double derivLambda,
                           int flags, double minEigThreshold )
{
#ifdef HAVE_TEGRA_OPTIMIZATION__DISABLED
    if (tegra::calcOpticalFlowPyrLK(_prevImg, _nextImg, _prevPts, _nextPts, _status, _err, winSize, maxLevel, criteria, derivLambda, flags))
        return;
#endif
    Mat prevImg = _prevImg.getMat(), nextImg = _nextImg.getMat(), prevPtsMat = _prevPts.getMat();
    derivLambda = std::min(std::max(derivLambda, 0.), 1.);
    const int derivDepth = DataType<deriv_type>::depth;

    CV_Assert( derivLambda >= 0 );
    CV_Assert( maxLevel >= 0 && winSize.width > 2 && winSize.height > 2 );
    CV_Assert( prevImg.size() == nextImg.size() &&
        prevImg.type() == nextImg.type() );

    int level=0, i, k, npoints, cn = prevImg.channels(), cn2 = cn*2;
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
    
    const Point2f* prevPts = (const Point2f*)prevPtsMat.data;
    Point2f* nextPts = (Point2f*)nextPtsMat.data;
    
    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.data;
    float* err = 0;
    
    for( i = 0; i < npoints; i++ )
        status[i] = true;
    
    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = (float*)errMat.data;
    }

    vector<Mat> prevPyr(maxLevel+1), nextPyr(maxLevel+1);
    
    // build the image pyramids.
    // we pad each level with +/-winSize.{width|height}
    // pixels to simplify the further patch extraction.
    // Thanks to the reference counting, "temp" mat (the pyramid layer + border)
    // will not be deallocated, since {prevPyr|nextPyr}[level] will be a ROI in "temp".
    for( k = 0; k < 2; k++ )
    {
        Size sz = prevImg.size();
        vector<Mat>& pyr = k == 0 ? prevPyr : nextPyr;
        Mat& img0 = k == 0 ? prevImg : nextImg;
        
        for( level = 0; level <= maxLevel; level++ )
        {
            Mat temp(sz.height + winSize.height*2,
                     sz.width + winSize.width*2,
                     img0.type());
            pyr[level] = temp(Rect(winSize.width, winSize.height, sz.width, sz.height));
            if( level == 0 )
                img0.copyTo(pyr[level]);
            else
                pyrDown(pyr[level-1], pyr[level], pyr[level].size());
            copyMakeBorder(pyr[level], temp, winSize.height, winSize.height,
                           winSize.width, winSize.width, BORDER_REFLECT_101|BORDER_ISOLATED);
            sz = Size((sz.width+1)/2, (sz.height+1)/2);
            if( sz.width <= winSize.width || sz.height <= winSize.height )
            {
                maxLevel = level;
                break;
            }
        }
    }
    // dI/dx ~ Ix, dI/dy ~ Iy
    Mat derivIBuf((prevImg.rows + winSize.height*2),
             (prevImg.cols + winSize.width*2),
             CV_MAKETYPE(derivDepth, cn2));

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    for( level = maxLevel; level >= 0; level-- )
    {
        Size imgSize = prevPyr[level].size();
        Mat _derivI( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.data );
        Mat derivI = _derivI(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        calcSharrDeriv(prevPyr[level], derivI);
        copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_CONSTANT|BORDER_ISOLATED);
        
        parallel_for(BlockedRange(0, npoints), LKTrackerInvoker(prevPyr[level], derivI,
                                                                nextPyr[level], prevPts, nextPts,
                                                                status, err,
                                                                winSize, criteria, level, maxLevel,
                                                                flags, (float)minEigThreshold));
    }
}


static int icvMinimalPyramidSize( CvSize imgSize )
{
    return cvAlign(imgSize.width,8) * imgSize.height / 3;
}


static void
icvInitPyramidalAlgorithm( const CvMat* imgA, const CvMat* imgB,
                           CvMat* pyrA, CvMat* pyrB,
                           int level, CvTermCriteria * criteria,
                           int max_iters, int flags,
                           uchar *** imgI, uchar *** imgJ,
                           int **step, CvSize** size,
                           double **scale, cv::AutoBuffer<uchar>* buffer )
{
    const int ALIGN = 8;
    int pyrBytes, bufferBytes = 0, elem_size;
    int level1 = level + 1;

    int i;
    CvSize imgSize, levelSize;

    *imgI = *imgJ = 0;
    *step = 0;
    *scale = 0;
    *size = 0;

    /* check input arguments */
    if( ((flags & CV_LKFLOW_PYR_A_READY) != 0 && !pyrA) ||
        ((flags & CV_LKFLOW_PYR_B_READY) != 0 && !pyrB) )
        CV_Error( CV_StsNullPtr, "Some of the precomputed pyramids are missing" );

    if( level < 0 )
        CV_Error( CV_StsOutOfRange, "The number of pyramid levels is negative" );

    switch( criteria->type )
    {
    case CV_TERMCRIT_ITER:
        criteria->epsilon = 0.f;
        break;
    case CV_TERMCRIT_EPS:
        criteria->max_iter = max_iters;
        break;
    case CV_TERMCRIT_ITER | CV_TERMCRIT_EPS:
        break;
    default:
        assert( 0 );
        CV_Error( CV_StsBadArg, "Invalid termination criteria" );
    }

    /* compare squared values */
    criteria->epsilon *= criteria->epsilon;

    /* set pointers and step for every level */
    pyrBytes = 0;

    imgSize = cvGetSize(imgA);
    elem_size = CV_ELEM_SIZE(imgA->type);
    levelSize = imgSize;

    for( i = 1; i < level1; i++ )
    {
        levelSize.width = (levelSize.width + 1) >> 1;
        levelSize.height = (levelSize.height + 1) >> 1;

        int tstep = cvAlign(levelSize.width,ALIGN) * elem_size;
        pyrBytes += tstep * levelSize.height;
    }

    assert( pyrBytes <= imgSize.width * imgSize.height * elem_size * 4 / 3 );

    /* buffer_size = <size for patches> + <size for pyramids> */
    bufferBytes = (int)((level1 >= 0) * ((pyrA->data.ptr == 0) +
        (pyrB->data.ptr == 0)) * pyrBytes +
        (sizeof(imgI[0][0]) * 2 + sizeof(step[0][0]) +
         sizeof(size[0][0]) + sizeof(scale[0][0])) * level1);

    buffer->allocate( bufferBytes );

    *imgI = (uchar **) (uchar*)(*buffer);
    *imgJ = *imgI + level1;
    *step = (int *) (*imgJ + level1);
    *scale = (double *) (*step + level1);
    *size = (CvSize *)(*scale + level1);

    imgI[0][0] = imgA->data.ptr;
    imgJ[0][0] = imgB->data.ptr;
    step[0][0] = imgA->step;
    scale[0][0] = 1;
    size[0][0] = imgSize;

    if( level > 0 )
    {
        uchar *bufPtr = (uchar *) (*size + level1);
        uchar *ptrA = pyrA->data.ptr;
        uchar *ptrB = pyrB->data.ptr;

        if( !ptrA )
        {
            ptrA = bufPtr;
            bufPtr += pyrBytes;
        }

        if( !ptrB )
            ptrB = bufPtr;

        levelSize = imgSize;

        /* build pyramids for both frames */
        for( i = 1; i <= level; i++ )
        {
            int levelBytes;
            CvMat prev_level, next_level;

            levelSize.width = (levelSize.width + 1) >> 1;
            levelSize.height = (levelSize.height + 1) >> 1;

            size[0][i] = levelSize;
            step[0][i] = cvAlign( levelSize.width, ALIGN ) * elem_size;
            scale[0][i] = scale[0][i - 1] * 0.5;

            levelBytes = step[0][i] * levelSize.height;
            imgI[0][i] = (uchar *) ptrA;
            ptrA += levelBytes;

            if( !(flags & CV_LKFLOW_PYR_A_READY) )
            {
                prev_level = cvMat( size[0][i-1].height, size[0][i-1].width, CV_8UC1 );
                next_level = cvMat( size[0][i].height, size[0][i].width, CV_8UC1 );
                cvSetData( &prev_level, imgI[0][i-1], step[0][i-1] );
                cvSetData( &next_level, imgI[0][i], step[0][i] );
                cvPyrDown( &prev_level, &next_level );
            }

            imgJ[0][i] = (uchar *) ptrB;
            ptrB += levelBytes;

            if( !(flags & CV_LKFLOW_PYR_B_READY) )
            {
                prev_level = cvMat( size[0][i-1].height, size[0][i-1].width, CV_8UC1 );
                next_level = cvMat( size[0][i].height, size[0][i].width, CV_8UC1 );
                cvSetData( &prev_level, imgJ[0][i-1], step[0][i-1] );
                cvSetData( &next_level, imgJ[0][i], step[0][i] );
                cvPyrDown( &prev_level, &next_level );
            }
        }
    }
}


/* compute dI/dx and dI/dy */
static void
icvCalcIxIy_32f( const float* src, int src_step, float* dstX, float* dstY, int dst_step,
                 CvSize src_size, const float* smooth_k, float* buffer0 )
{
    int src_width = src_size.width, dst_width = src_size.width-2;
    int x, height = src_size.height - 2;
    float* buffer1 = buffer0 + src_width;

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dstX[0]);

    for( ; height--; src += src_step, dstX += dst_step, dstY += dst_step )
    {
        const float* src2 = src + src_step;
        const float* src3 = src + src_step*2;

        for( x = 0; x < src_width; x++ )
        {
            float t0 = (src3[x] + src[x])*smooth_k[0] + src2[x]*smooth_k[1];
            float t1 = src3[x] - src[x];
            buffer0[x] = t0; buffer1[x] = t1;
        }

        for( x = 0; x < dst_width; x++ )
        {
            float t0 = buffer0[x+2] - buffer0[x];
            float t1 = (buffer1[x] + buffer1[x+2])*smooth_k[0] + buffer1[x+1]*smooth_k[1];
            dstX[x] = t0; dstY[x] = t1;
        }
    }
}


#undef CV_8TO32F
#define CV_8TO32F(a) (a)

static const void*
icvAdjustRect( const void* srcptr, int src_step, int pix_size,
              CvSize src_size, CvSize win_size,
              CvPoint ip, CvRect* pRect )
{
    CvRect rect;
    const char* src = (const char*)srcptr;
    
    if( ip.x >= 0 )
    {
        src += ip.x*pix_size;
        rect.x = 0;
    }
    else
    {
        rect.x = -ip.x;
        if( rect.x > win_size.width )
            rect.x = win_size.width;
    }
    
    if( ip.x + win_size.width < src_size.width )
        rect.width = win_size.width;
    else
    {
        rect.width = src_size.width - ip.x - 1;
        if( rect.width < 0 )
        {
            src += rect.width*pix_size;
            rect.width = 0;
        }
        assert( rect.width <= win_size.width );
    }
    
    if( ip.y >= 0 )
    {
        src += ip.y * src_step;
        rect.y = 0;
    }
    else
        rect.y = -ip.y;
    
    if( ip.y + win_size.height < src_size.height )
        rect.height = win_size.height;
    else
    {
        rect.height = src_size.height - ip.y - 1;
        if( rect.height < 0 )
        {
            src += rect.height*src_step;
            rect.height = 0;
        }
    }
    
    *pRect = rect;
    return src - rect.x*pix_size;
}


static CvStatus CV_STDCALL icvGetRectSubPix_8u32f_C1R
( const uchar* src, int src_step, CvSize src_size,
 float* dst, int dst_step, CvSize win_size, CvPoint2D32f center )
{
    CvPoint ip;
    float  a12, a22, b1, b2;
    float a, b;
    double s = 0;
    int i, j;
    
    center.x -= (win_size.width-1)*0.5f;
    center.y -= (win_size.height-1)*0.5f;
    
    ip.x = cvFloor( center.x );
    ip.y = cvFloor( center.y );
    
    if( win_size.width <= 0 || win_size.height <= 0 )
        return CV_BADRANGE_ERR;
    
    a = center.x - ip.x;
    b = center.y - ip.y;
    a = MAX(a,0.0001f);
    a12 = a*(1.f-b);
    a22 = a*b;
    b1 = 1.f - b;
    b2 = b;
    s = (1. - a)/a;
    
    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);
    
    if( 0 <= ip.x && ip.x + win_size.width < src_size.width &&
       0 <= ip.y && ip.y + win_size.height < src_size.height )
    {
        // extracted rectangle is totally inside the image
        src += ip.y * src_step + ip.x;
        
#if 0
        if( icvCopySubpix_8u32f_C1R_p &&
           icvCopySubpix_8u32f_C1R_p( src, src_step, dst,
                                     dst_step*sizeof(dst[0]), win_size, a, b ) >= 0 )
            return CV_OK;
#endif
        
        for( ; win_size.height--; src += src_step, dst += dst_step )
        {
            float prev = (1 - a)*(b1*CV_8TO32F(src[0]) + b2*CV_8TO32F(src[src_step]));
            for( j = 0; j < win_size.width; j++ )
            {
                float t = a12*CV_8TO32F(src[j+1]) + a22*CV_8TO32F(src[j+1+src_step]);
                dst[j] = prev + t;
                prev = (float)(t*s);
            }
        }
    }
    else
    {
        CvRect r;
        
        src = (const uchar*)icvAdjustRect( src, src_step*sizeof(*src),
                                          sizeof(*src), src_size, win_size,ip, &r);
        
        for( i = 0; i < win_size.height; i++, dst += dst_step )
        {
            const uchar *src2 = src + src_step;
            
            if( i < r.y || i >= r.height )
                src2 -= src_step;
            
            for( j = 0; j < r.x; j++ )
            {
                float s0 = CV_8TO32F(src[r.x])*b1 +
                CV_8TO32F(src2[r.x])*b2;
                
                dst[j] = (float)(s0);
            }
            
            if( j < r.width )
            {
                float prev = (1 - a)*(b1*CV_8TO32F(src[j]) + b2*CV_8TO32F(src2[j]));
                
                for( ; j < r.width; j++ )
                {
                    float t = a12*CV_8TO32F(src[j+1]) + a22*CV_8TO32F(src2[j+1]);
                    dst[j] = prev + t;
                    prev = (float)(t*s);
                }
            }
            
            for( ; j < win_size.width; j++ )
            {
                float s0 = CV_8TO32F(src[r.width])*b1 +
                CV_8TO32F(src2[r.width])*b2;
                
                dst[j] = (float)(s0);
            }
            
            if( i < r.height )
                src = src2;
        }
    }
    
    return CV_OK;
}


#define ICV_32F8U(x)  ((uchar)cvRound(x))

#define ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( flavor, srctype, dsttype,      \
worktype, cast_macro, cvt )    \
static CvStatus CV_STDCALL                                                   \
icvGetQuadrangleSubPix_##flavor##_C1R                                       \
( const srctype * src, int src_step, CvSize src_size,                       \
dsttype *dst, int dst_step, CvSize win_size, const float *matrix )        \
{                                                                           \
int x, y;                                                               \
double dx = (win_size.width - 1)*0.5;                                   \
double dy = (win_size.height - 1)*0.5;                                  \
double A11 = matrix[0], A12 = matrix[1], A13 = matrix[2]-A11*dx-A12*dy; \
double A21 = matrix[3], A22 = matrix[4], A23 = matrix[5]-A21*dx-A22*dy; \
\
src_step /= sizeof(srctype);                                            \
dst_step /= sizeof(dsttype);                                            \
\
for( y = 0; y < win_size.height; y++, dst += dst_step )                 \
{                                                                       \
double xs = A12*y + A13;                                            \
double ys = A22*y + A23;                                            \
double xe = A11*(win_size.width-1) + A12*y + A13;                   \
double ye = A21*(win_size.width-1) + A22*y + A23;                   \
\
if( (unsigned)(cvFloor(xs)-1) < (unsigned)(src_size.width - 3) &&   \
(unsigned)(cvFloor(ys)-1) < (unsigned)(src_size.height - 3) &&  \
(unsigned)(cvFloor(xe)-1) < (unsigned)(src_size.width - 3) &&   \
(unsigned)(cvFloor(ye)-1) < (unsigned)(src_size.height - 3))    \
{                                                                   \
for( x = 0; x < win_size.width; x++ )                           \
{                                                               \
int ixs = cvFloor( xs );                                    \
int iys = cvFloor( ys );                                    \
const srctype *ptr = src + src_step*iys + ixs;              \
double a = xs - ixs, b = ys - iys, a1 = 1.f - a;            \
worktype p0 = cvt(ptr[0])*a1 + cvt(ptr[1])*a;               \
worktype p1 = cvt(ptr[src_step])*a1 + cvt(ptr[src_step+1])*a;\
xs += A11;                                                  \
ys += A21;                                                  \
\
dst[x] = cast_macro(p0 + b * (p1 - p0));                    \
}                                                               \
}                                                                   \
else                                                                \
{                                                                   \
for( x = 0; x < win_size.width; x++ )                           \
{                                                               \
int ixs = cvFloor( xs ), iys = cvFloor( ys );               \
double a = xs - ixs, b = ys - iys, a1 = 1.f - a;            \
const srctype *ptr0, *ptr1;                                 \
worktype p0, p1;                                            \
xs += A11; ys += A21;                                       \
\
if( (unsigned)iys < (unsigned)(src_size.height-1) )         \
ptr0 = src + src_step*iys, ptr1 = ptr0 + src_step;      \
else                                                        \
ptr0 = ptr1 = src + (iys < 0 ? 0 : src_size.height-1)*src_step; \
\
if( (unsigned)ixs < (unsigned)(src_size.width-1) )          \
{                                                           \
p0 = cvt(ptr0[ixs])*a1 + cvt(ptr0[ixs+1])*a;            \
p1 = cvt(ptr1[ixs])*a1 + cvt(ptr1[ixs+1])*a;            \
}                                                           \
else                                                        \
{                                                           \
ixs = ixs < 0 ? 0 : src_size.width - 1;                 \
p0 = cvt(ptr0[ixs]); p1 = cvt(ptr1[ixs]);               \
}                                                           \
dst[x] = cast_macro(p0 + b * (p1 - p0));                    \
}                                                               \
}                                                                   \
}                                                                       \
\
return CV_OK;                                                           \
}

ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( 8u32f, uchar, float, double, CV_CAST_32F, CV_8TO32F )


CV_IMPL void
cvCalcOpticalFlowPyrLK( const void* arrA, const void* arrB,
                        void* /*pyrarrA*/, void* /*pyrarrB*/,
                        const CvPoint2D32f * featuresA,
                        CvPoint2D32f * featuresB,
                        int count, CvSize winSize, int level,
                        char *status, float *error,
                        CvTermCriteria criteria, int flags )
{
    if( count <= 0 )
        return;
    CV_Assert( featuresA && featuresB );
    cv::Mat A = cv::cvarrToMat(arrA), B = cv::cvarrToMat(arrB);
    cv::Mat ptA(count, 1, CV_32FC2, (void*)featuresA);
    cv::Mat ptB(count, 1, CV_32FC2, (void*)featuresB);
    cv::Mat st, err;
    
    if( status )
        st = cv::Mat(count, 1, CV_8U, (void*)status);
    if( error )
        err = cv::Mat(count, 1, CV_32F, (void*)error);
    cv::calcOpticalFlowPyrLK( A, B, ptA, ptB, status ? cv::_OutputArray(st) : cv::_OutputArray(),
                              error ? cv::_OutputArray(err) : cv::_OutputArray(),
                              winSize, level, criteria, flags);
}


/* Affine tracking algorithm */

CV_IMPL void
cvCalcAffineFlowPyrLK( const void* arrA, const void* arrB,
                       void* pyrarrA, void* pyrarrB,
                       const CvPoint2D32f * featuresA,
                       CvPoint2D32f * featuresB,
                       float *matrices, int count,
                       CvSize winSize, int level,
                       char *status, float *error,
                       CvTermCriteria criteria, int flags )
{
    const int MAX_ITERS = 100;

    cv::AutoBuffer<char> _status;
    cv::AutoBuffer<uchar> buffer;
    cv::AutoBuffer<uchar> pyr_buffer;

    CvMat stubA, *imgA = (CvMat*)arrA;
    CvMat stubB, *imgB = (CvMat*)arrB;
    CvMat pstubA, *pyrA = (CvMat*)pyrarrA;
    CvMat pstubB, *pyrB = (CvMat*)pyrarrB;

    static const float smoothKernel[] = { 0.09375, 0.3125, 0.09375 };  /* 3/32, 10/32, 3/32 */

    int bufferBytes = 0;

    uchar **imgI = 0;
    uchar **imgJ = 0;
    int *step = 0;
    double *scale = 0;
    CvSize* size = 0;

    float *patchI;
    float *patchJ;
    float *Ix;
    float *Iy;

    int i, j, k, l;

    CvSize patchSize = cvSize( winSize.width * 2 + 1, winSize.height * 2 + 1 );
    int patchLen = patchSize.width * patchSize.height;
    int patchStep = patchSize.width * sizeof( patchI[0] );

    CvSize srcPatchSize = cvSize( patchSize.width + 2, patchSize.height + 2 );
    int srcPatchLen = srcPatchSize.width * srcPatchSize.height;
    int srcPatchStep = srcPatchSize.width * sizeof( patchI[0] );
    CvSize imgSize;
    float eps = (float)MIN(winSize.width, winSize.height);

    imgA = cvGetMat( imgA, &stubA );
    imgB = cvGetMat( imgB, &stubB );

    if( CV_MAT_TYPE( imgA->type ) != CV_8UC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( !CV_ARE_TYPES_EQ( imgA, imgB ))
        CV_Error( CV_StsUnmatchedFormats, "" );

    if( !CV_ARE_SIZES_EQ( imgA, imgB ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( imgA->step != imgB->step )
        CV_Error( CV_StsUnmatchedSizes, "imgA and imgB must have equal steps" );

    if( !matrices )
        CV_Error( CV_StsNullPtr, "" );

    imgSize = cvGetMatSize( imgA );

    if( pyrA )
    {
        pyrA = cvGetMat( pyrA, &pstubA );

        if( pyrA->step*pyrA->height < icvMinimalPyramidSize( imgSize ) )
            CV_Error( CV_StsBadArg, "pyramid A has insufficient size" );
    }
    else
    {
        pyrA = &pstubA;
        pyrA->data.ptr = 0;
    }

    if( pyrB )
    {
        pyrB = cvGetMat( pyrB, &pstubB );

        if( pyrB->step*pyrB->height < icvMinimalPyramidSize( imgSize ) )
            CV_Error( CV_StsBadArg, "pyramid B has insufficient size" );
    }
    else
    {
        pyrB = &pstubB;
        pyrB->data.ptr = 0;
    }

    if( count == 0 )
        return;

    /* check input arguments */
    if( !featuresA || !featuresB || !matrices )
        CV_Error( CV_StsNullPtr, "" );

    if( winSize.width <= 1 || winSize.height <= 1 )
        CV_Error( CV_StsOutOfRange, "the search window is too small" );

    if( count < 0 )
        CV_Error( CV_StsOutOfRange, "" );

    icvInitPyramidalAlgorithm( imgA, imgB,
        pyrA, pyrB, level, &criteria, MAX_ITERS, flags,
        &imgI, &imgJ, &step, &size, &scale, &pyr_buffer );

    /* buffer_size = <size for patches> + <size for pyramids> */
    bufferBytes = (srcPatchLen + patchLen*3)*sizeof(patchI[0]) + (36*2 + 6)*sizeof(double);

    buffer.allocate(bufferBytes);

    if( !status )
    {
        _status.allocate(count);
        status = _status;
    }

    patchI = (float *)(uchar*)buffer;
    patchJ = patchI + srcPatchLen;
    Ix = patchJ + patchLen;
    Iy = Ix + patchLen;

    if( status )
        memset( status, 1, count );

    if( !(flags & CV_LKFLOW_INITIAL_GUESSES) )
    {
        memcpy( featuresB, featuresA, count * sizeof( featuresA[0] ));
        for( i = 0; i < count * 4; i += 4 )
        {
            matrices[i] = matrices[i + 3] = 1.f;
            matrices[i + 1] = matrices[i + 2] = 0.f;
        }
    }

    for( i = 0; i < count; i++ )
    {
        featuresB[i].x = (float)(featuresB[i].x * scale[level] * 0.5);
        featuresB[i].y = (float)(featuresB[i].y * scale[level] * 0.5);
    }

    /* do processing from top pyramid level (smallest image)
       to the bottom (original image) */
    for( l = level; l >= 0; l-- )
    {
        CvSize levelSize = size[l];
        int levelStep = step[l];

        /* find flow for each given point at the particular level */
        for( i = 0; i < count; i++ )
        {
            CvPoint2D32f u;
            float Av[6];
            double G[36];
            double meanI = 0, meanJ = 0;
            int x, y;
            int pt_status = status[i];
            CvMat mat;

            if( !pt_status )
                continue;

            Av[0] = matrices[i*4];
            Av[1] = matrices[i*4+1];
            Av[3] = matrices[i*4+2];
            Av[4] = matrices[i*4+3];

            Av[2] = featuresB[i].x += featuresB[i].x;
            Av[5] = featuresB[i].y += featuresB[i].y;

            u.x = (float) (featuresA[i].x * scale[l]);
            u.y = (float) (featuresA[i].y * scale[l]);

            if( u.x < -eps || u.x >= levelSize.width+eps ||
                u.y < -eps || u.y >= levelSize.height+eps ||
                icvGetRectSubPix_8u32f_C1R( imgI[l], levelStep,
                levelSize, patchI, srcPatchStep, srcPatchSize, u ) < 0 )
            {
                /* point is outside the image. take the next */
                if( l == 0 )
                    status[i] = 0;
                continue;
            }

            icvCalcIxIy_32f( patchI, srcPatchStep, Ix, Iy,
                (srcPatchSize.width-2)*sizeof(patchI[0]), srcPatchSize,
                smoothKernel, patchJ );

            /* repack patchI (remove borders) */
            for( k = 0; k < patchSize.height; k++ )
                memcpy( patchI + k * patchSize.width,
                        patchI + (k + 1) * srcPatchSize.width + 1, patchStep );

            memset( G, 0, sizeof( G ));

            /* calculate G matrix */
            for( y = -winSize.height, k = 0; y <= winSize.height; y++ )
            {
                for( x = -winSize.width; x <= winSize.width; x++, k++ )
                {
                    double ixix = ((double) Ix[k]) * Ix[k];
                    double ixiy = ((double) Ix[k]) * Iy[k];
                    double iyiy = ((double) Iy[k]) * Iy[k];

                    double xx, xy, yy;

                    G[0] += ixix;
                    G[1] += ixiy;
                    G[2] += x * ixix;
                    G[3] += y * ixix;
                    G[4] += x * ixiy;
                    G[5] += y * ixiy;

                    // G[6] == G[1]
                    G[7] += iyiy;
                    // G[8] == G[4]
                    // G[9] == G[5]
                    G[10] += x * iyiy;
                    G[11] += y * iyiy;

                    xx = x * x;
                    xy = x * y;
                    yy = y * y;

                    // G[12] == G[2]
                    // G[13] == G[8] == G[4]
                    G[14] += xx * ixix;
                    G[15] += xy * ixix;
                    G[16] += xx * ixiy;
                    G[17] += xy * ixiy;

                    // G[18] == G[3]
                    // G[19] == G[9]
                    // G[20] == G[15]
                    G[21] += yy * ixix;
                    // G[22] == G[17]
                    G[23] += yy * ixiy;

                    // G[24] == G[4]
                    // G[25] == G[10]
                    // G[26] == G[16]
                    // G[27] == G[22]
                    G[28] += xx * iyiy;
                    G[29] += xy * iyiy;

                    // G[30] == G[5]
                    // G[31] == G[11]
                    // G[32] == G[17]
                    // G[33] == G[23]
                    // G[34] == G[29]
                    G[35] += yy * iyiy;

                    meanI += patchI[k];
                }
            }

            meanI /= patchSize.width*patchSize.height;

            G[8] = G[4];
            G[9] = G[5];
            G[22] = G[17];

            // fill part of G below its diagonal
            for( y = 1; y < 6; y++ )
                for( x = 0; x < y; x++ )
                    G[y * 6 + x] = G[x * 6 + y];

            cvInitMatHeader( &mat, 6, 6, CV_64FC1, G );

            if( cvInvert( &mat, &mat, CV_SVD ) < 1e-4 )
            {
                /* bad matrix. take the next point */
                if( l == 0 )
                    status[i] = 0;
                continue;
            }

            for( j = 0; j < criteria.max_iter; j++ )
            {
                double b[6] = {0,0,0,0,0,0}, eta[6];
                double t0, t1, s = 0;

                if( Av[2] < -eps || Av[2] >= levelSize.width+eps ||
                    Av[5] < -eps || Av[5] >= levelSize.height+eps ||
                    icvGetQuadrangleSubPix_8u32f_C1R( imgJ[l], levelStep,
                    levelSize, patchJ, patchStep, patchSize, Av ) < 0 )
                {
                    pt_status = 0;
                    break;
                }

                for( y = -winSize.height, k = 0, meanJ = 0; y <= winSize.height; y++ )
                    for( x = -winSize.width; x <= winSize.width; x++, k++ )
                        meanJ += patchJ[k];

                meanJ = meanJ / (patchSize.width * patchSize.height) - meanI;

                for( y = -winSize.height, k = 0; y <= winSize.height; y++ )
                {
                    for( x = -winSize.width; x <= winSize.width; x++, k++ )
                    {
                        double t = patchI[k] - patchJ[k] + meanJ;
                        double ixt = Ix[k] * t;
                        double iyt = Iy[k] * t;

                        s += t;

                        b[0] += ixt;
                        b[1] += iyt;
                        b[2] += x * ixt;
                        b[3] += y * ixt;
                        b[4] += x * iyt;
                        b[5] += y * iyt;
                    }
                }

                for( k = 0; k < 6; k++ )
                    eta[k] = G[k*6]*b[0] + G[k*6+1]*b[1] + G[k*6+2]*b[2] +
                        G[k*6+3]*b[3] + G[k*6+4]*b[4] + G[k*6+5]*b[5];

                Av[2] = (float)(Av[2] + Av[0] * eta[0] + Av[1] * eta[1]);
                Av[5] = (float)(Av[5] + Av[3] * eta[0] + Av[4] * eta[1]);

                t0 = Av[0] * (1 + eta[2]) + Av[1] * eta[4];
                t1 = Av[0] * eta[3] + Av[1] * (1 + eta[5]);
                Av[0] = (float)t0;
                Av[1] = (float)t1;

                t0 = Av[3] * (1 + eta[2]) + Av[4] * eta[4];
                t1 = Av[3] * eta[3] + Av[4] * (1 + eta[5]);
                Av[3] = (float)t0;
                Av[4] = (float)t1;

                if( eta[0] * eta[0] + eta[1] * eta[1] < criteria.epsilon )
                    break;
            }

            if( pt_status != 0 || l == 0 )
            {
                status[i] = (char)pt_status;
                featuresB[i].x = Av[2];
                featuresB[i].y = Av[5];
            
                matrices[i*4] = Av[0];
                matrices[i*4+1] = Av[1];
                matrices[i*4+2] = Av[3];
                matrices[i*4+3] = Av[4];
            }

            if( pt_status && l == 0 && error )
            {
                /* calc error */
                double err = 0;

                for( y = 0, k = 0; y < patchSize.height; y++ )
                {
                    for( x = 0; x < patchSize.width; x++, k++ )
                    {
                        double t = patchI[k] - patchJ[k] + meanJ;
                        err += t * t;
                    }
                }
                error[i] = (float)sqrt(err);
            }
        }
    }
}



static void
icvGetRTMatrix( const CvPoint2D32f* a, const CvPoint2D32f* b,
                int count, CvMat* M, int full_affine )
{
    if( full_affine )
    {
        double sa[36], sb[6];
        CvMat A = cvMat( 6, 6, CV_64F, sa ), B = cvMat( 6, 1, CV_64F, sb );
        CvMat MM = cvMat( 6, 1, CV_64F, M->data.db );

        int i;

        memset( sa, 0, sizeof(sa) );
        memset( sb, 0, sizeof(sb) );

        for( i = 0; i < count; i++ )
        {
            sa[0] += a[i].x*a[i].x;
            sa[1] += a[i].y*a[i].x;
            sa[2] += a[i].x;

            sa[6] += a[i].x*a[i].y;
            sa[7] += a[i].y*a[i].y;
            sa[8] += a[i].y;

            sa[12] += a[i].x;
            sa[13] += a[i].y;
            sa[14] += 1;

            sb[0] += a[i].x*b[i].x;
            sb[1] += a[i].y*b[i].x;
            sb[2] += b[i].x;
            sb[3] += a[i].x*b[i].y;
            sb[4] += a[i].y*b[i].y;
            sb[5] += b[i].y;
        }

        sa[21] = sa[0];
        sa[22] = sa[1];
        sa[23] = sa[2];
        sa[27] = sa[6];
        sa[28] = sa[7];
        sa[29] = sa[8];
        sa[33] = sa[12];
        sa[34] = sa[13];
        sa[35] = sa[14];

        cvSolve( &A, &B, &MM, CV_SVD );
    }
    else
    {
        double sa[16], sb[4], m[4], *om = M->data.db;
        CvMat A = cvMat( 4, 4, CV_64F, sa ), B = cvMat( 4, 1, CV_64F, sb );
        CvMat MM = cvMat( 4, 1, CV_64F, m );

        int i;

        memset( sa, 0, sizeof(sa) );
        memset( sb, 0, sizeof(sb) );

        for( i = 0; i < count; i++ )
        {
            sa[0] += a[i].x*a[i].x + a[i].y*a[i].y;
            sa[1] += 0;
            sa[2] += a[i].x;
            sa[3] += a[i].y;

            sa[4] += 0;
            sa[5] += a[i].x*a[i].x + a[i].y*a[i].y;
            sa[6] += -a[i].y;
            sa[7] += a[i].x;

            sa[8] += a[i].x;
            sa[9] += -a[i].y;
            sa[10] += 1;
            sa[11] += 0;

            sa[12] += a[i].y;
            sa[13] += a[i].x;
            sa[14] += 0;
            sa[15] += 1;

            sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
            sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
            sb[2] += b[i].x;
            sb[3] += b[i].y;
        }

        cvSolve( &A, &B, &MM, CV_SVD );

        om[0] = om[4] = m[0];
        om[1] = -m[1];
        om[3] = m[1];
        om[2] = m[2];
        om[5] = m[3];
    }
}


CV_IMPL int
cvEstimateRigidTransform( const CvArr* matA, const CvArr* matB, CvMat* matM, int full_affine )
{
    const int COUNT = 15;
    const int WIDTH = 160, HEIGHT = 120;
    const int RANSAC_MAX_ITERS = 500;
    const int RANSAC_SIZE0 = 3;
    const double RANSAC_GOOD_RATIO = 0.5;

    cv::Ptr<CvMat> sA, sB;
    cv::AutoBuffer<CvPoint2D32f> pA, pB;
    cv::AutoBuffer<int> good_idx;
    cv::AutoBuffer<char> status;
    cv::Ptr<CvMat> gray;

    CvMat stubA, *A = cvGetMat( matA, &stubA );
    CvMat stubB, *B = cvGetMat( matB, &stubB );
    CvSize sz0, sz1;
    int cn, equal_sizes;
    int i, j, k, k1;
    int count_x, count_y, count = 0;
    double scale = 1;
    CvRNG rng = cvRNG(-1);
    double m[6]={0};
    CvMat M = cvMat( 2, 3, CV_64F, m );
    int good_count = 0;
    CvRect brect;

    if( !CV_IS_MAT(matM) )
        CV_Error( matM ? CV_StsBadArg : CV_StsNullPtr, "Output parameter M is not a valid matrix" );

    if( !CV_ARE_SIZES_EQ( A, B ) )
        CV_Error( CV_StsUnmatchedSizes, "Both input images must have the same size" );

    if( !CV_ARE_TYPES_EQ( A, B ) )
        CV_Error( CV_StsUnmatchedFormats, "Both input images must have the same data type" );

    if( CV_MAT_TYPE(A->type) == CV_8UC1 || CV_MAT_TYPE(A->type) == CV_8UC3 )
    {
        cn = CV_MAT_CN(A->type);
        sz0 = cvGetSize(A);
        sz1 = cvSize(WIDTH, HEIGHT);

        scale = MAX( (double)sz1.width/sz0.width, (double)sz1.height/sz0.height );
        scale = MIN( scale, 1. );
        sz1.width = cvRound( sz0.width * scale );
        sz1.height = cvRound( sz0.height * scale );

        equal_sizes = sz1.width == sz0.width && sz1.height == sz0.height;

        if( !equal_sizes || cn != 1 )
        {
            sA = cvCreateMat( sz1.height, sz1.width, CV_8UC1 );
            sB = cvCreateMat( sz1.height, sz1.width, CV_8UC1 );

            if( cn != 1 )
            {
                gray = cvCreateMat( sz0.height, sz0.width, CV_8UC1 );
                cvCvtColor( A, gray, CV_BGR2GRAY );
                cvResize( gray, sA, CV_INTER_AREA );
                cvCvtColor( B, gray, CV_BGR2GRAY );
                cvResize( gray, sB, CV_INTER_AREA );
                gray.release();
            }
            else
            {
                cvResize( A, sA, CV_INTER_AREA );
                cvResize( B, sB, CV_INTER_AREA );
            }
           
            A = sA;
            B = sB;
        }

        count_y = COUNT;
        count_x = cvRound((double)COUNT*sz1.width/sz1.height);
        count = count_x * count_y;

        pA.allocate(count);
        pB.allocate(count);
        status.allocate(count);

        for( i = 0, k = 0; i < count_y; i++ )
            for( j = 0; j < count_x; j++, k++ )
            {
                pA[k].x = (j+0.5f)*sz1.width/count_x;
                pA[k].y = (i+0.5f)*sz1.height/count_y;
            }

        // find the corresponding points in B
        cvCalcOpticalFlowPyrLK( A, B, 0, 0, pA, pB, count, cvSize(10,10), 3,
                                status, 0, cvTermCriteria(CV_TERMCRIT_ITER,40,0.1), 0 );

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
    }
    else if( CV_MAT_TYPE(A->type) == CV_32FC2 || CV_MAT_TYPE(A->type) == CV_32SC2 )
    {
        count = A->cols*A->rows;
        CvMat _pA, _pB;
        pA.allocate(count);
        pB.allocate(count);
        _pA = cvMat( A->rows, A->cols, CV_32FC2, pA );
        _pB = cvMat( B->rows, B->cols, CV_32FC2, pB );
        cvConvert( A, &_pA );
        cvConvert( B, &_pB );
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Both input images must have either 8uC1 or 8uC3 type" );

    good_idx.allocate(count);

    if( count < RANSAC_SIZE0 )
        return 0;
    
    CvMat _pB = cvMat(1, count, CV_32FC2, pB);    
    brect = cvBoundingRect(&_pB, 1);

    // RANSAC stuff:
    // 1. find the consensus
    for( k = 0; k < RANSAC_MAX_ITERS; k++ )
    {
        int idx[RANSAC_SIZE0];
        CvPoint2D32f a[3];
        CvPoint2D32f b[3];

        memset( a, 0, sizeof(a) );
        memset( b, 0, sizeof(b) );

        // choose random 3 non-complanar points from A & B
        for( i = 0; i < RANSAC_SIZE0; i++ )
        {
            for( k1 = 0; k1 < RANSAC_MAX_ITERS; k1++ )
            {
                idx[i] = cvRandInt(&rng) % count;
                
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

                if( i+1 == RANSAC_SIZE0 )
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

                    if( fabs(dax1*day2 - day1*dax2) < eps*sqrt(dax1*dax1+day1*day1)*sqrt(dax2*dax2+day2*day2) ||
                        fabs(dbx1*dby2 - dby1*dbx2) < eps*sqrt(dbx1*dbx1+dby1*dby1)*sqrt(dbx2*dbx2+dby2*dby2) )
                        continue;
                }
                break;
            }

            if( k1 >= RANSAC_MAX_ITERS )
                break;
        }

        if( i < RANSAC_SIZE0 )
            continue;

        // estimate the transformation using 3 points
        icvGetRTMatrix( a, b, 3, &M, full_affine );

        for( i = 0, good_count = 0; i < count; i++ )
        {
            if( fabs( m[0]*pA[i].x + m[1]*pA[i].y + m[2] - pB[i].x ) +
                fabs( m[3]*pA[i].x + m[4]*pA[i].y + m[5] - pB[i].y ) < MAX(brect.width,brect.height)*0.05 )
                good_idx[good_count++] = i;
        }

        if( good_count >= count*RANSAC_GOOD_RATIO )
            break;
    }

    if( k >= RANSAC_MAX_ITERS )
        return 0;

    if( good_count < count )
    {
        for( i = 0; i < good_count; i++ )
        {
            j = good_idx[i];
            pA[i] = pA[j];
            pB[i] = pB[j];
        }
    }

    icvGetRTMatrix( pA, pB, good_count, &M, full_affine );
    m[2] /= scale;
    m[5] /= scale;
    cvConvert( &M, matM );
    
    return 1;
}

cv::Mat cv::estimateRigidTransform( InputArray src1,
                                    InputArray src2,
                                    bool fullAffine )
{
    Mat M(2, 3, CV_64F), A = src1.getMat(), B = src2.getMat();
    CvMat matA = A, matB = B, matM = M;
    cvEstimateRigidTransform(&matA, &matB, &matM, fullAffine);
    return M;
}

/* End of file. */
