//M*//////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

/****************************************************************************************\
*    Very fast SAD-based (Sum-of-Absolute-Diffrences) stereo correspondence algorithm.   *
*    Contributed by Kurt Konolige                                                        *
\****************************************************************************************/

#include "precomp.hpp"
#include <stdio.h>
#include <limits>
#include "opencl_kernels_calib3d.hpp"

namespace cv
{

struct StereoBMParams
{
    StereoBMParams(int _numDisparities=64, int _SADWindowSize=21)
    {
        preFilterType = StereoBM::PREFILTER_XSOBEL;
        preFilterSize = 9;
        preFilterCap = 31;
        SADWindowSize = _SADWindowSize;
        minDisparity = 0;
        numDisparities = _numDisparities > 0 ? _numDisparities : 64;
        textureThreshold = 10;
        uniquenessRatio = 15;
        speckleRange = speckleWindowSize = 0;
        roi1 = roi2 = Rect(0,0,0,0);
        disp12MaxDiff = -1;
        dispType = CV_16S;
    }

    int preFilterType;
    int preFilterSize;
    int preFilterCap;
    int SADWindowSize;
    int minDisparity;
    int numDisparities;
    int textureThreshold;
    int uniquenessRatio;
    int speckleRange;
    int speckleWindowSize;
    Rect roi1, roi2;
    int disp12MaxDiff;
    int dispType;
};

static bool ocl_prefilter_norm(InputArray _input, OutputArray _output, int winsize, int prefilterCap)
{
    ocl::Kernel k("prefilter_norm", ocl::calib3d::stereobm_oclsrc);
    if(k.empty())
        return false;

    int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
    scale_g *= scale_s;

    UMat input = _input.getUMat(), output;
    _output.create(input.size(), input.type());
    output = _output.getUMat();

    size_t globalThreads[3] = { input.cols, input.rows, 1 };

    k.args(ocl::KernelArg::PtrReadOnly(input), ocl::KernelArg::PtrWriteOnly(output), input.rows, input.cols,
        prefilterCap, winsize, scale_g, scale_s);

    return k.run(2, globalThreads, NULL, false);
}

static void prefilterNorm( const Mat& src, Mat& dst, int winsize, int ftzero, uchar* buf )
{
    int x, y, wsz2 = winsize/2;
    int* vsum = (int*)alignPtr(buf + (wsz2 + 1)*sizeof(vsum[0]), 32);
    int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
    const int OFS = 256*5, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ];
    const uchar* sptr = src.ptr();
    int srcstep = (int)src.step;
    Size size = src.size();

    scale_g *= scale_s;

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero*2 : x - OFS + ftzero);

    for( x = 0; x < size.width; x++ )
        vsum[x] = (ushort)(sptr[x]*(wsz2 + 2));

    for( y = 1; y < wsz2; y++ )
    {
        for( x = 0; x < size.width; x++ )
            vsum[x] = (ushort)(vsum[x] + sptr[srcstep*y + x]);
    }

    for( y = 0; y < size.height; y++ )
    {
        const uchar* top = sptr + srcstep*MAX(y-wsz2-1,0);
        const uchar* bottom = sptr + srcstep*MIN(y+wsz2,size.height-1);
        const uchar* prev = sptr + srcstep*MAX(y-1,0);
        const uchar* curr = sptr + srcstep*y;
        const uchar* next = sptr + srcstep*MIN(y+1,size.height-1);
        uchar* dptr = dst.ptr<uchar>(y);

        for( x = 0; x < size.width; x++ )
            vsum[x] = (ushort)(vsum[x] + bottom[x] - top[x]);

        for( x = 0; x <= wsz2; x++ )
        {
            vsum[-x-1] = vsum[0];
            vsum[size.width+x] = vsum[size.width-1];
        }

        int sum = vsum[0]*(wsz2 + 1);
        for( x = 1; x <= wsz2; x++ )
            sum += vsum[x];

        int val = ((curr[0]*5 + curr[1] + prev[0] + next[0])*scale_g - sum*scale_s) >> 10;
        dptr[0] = tab[val + OFS];

        for( x = 1; x < size.width-1; x++ )
        {
            sum += vsum[x+wsz2] - vsum[x-wsz2-1];
            val = ((curr[x]*4 + curr[x-1] + curr[x+1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
            dptr[x] = tab[val + OFS];
        }

        sum += vsum[x+wsz2] - vsum[x-wsz2-1];
        val = ((curr[x]*5 + curr[x-1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
        dptr[x] = tab[val + OFS];
    }
}

static bool ocl_prefilter_xsobel(InputArray _input, OutputArray _output, int prefilterCap)
{
    ocl::Kernel k("prefilter_xsobel", ocl::calib3d::stereobm_oclsrc);
    if(k.empty())
        return false;

    UMat input = _input.getUMat(), output;
    _output.create(input.size(), input.type());
    output = _output.getUMat();

    size_t globalThreads[3] = { input.cols, input.rows, 1 };

    k.args(ocl::KernelArg::PtrReadOnly(input), ocl::KernelArg::PtrWriteOnly(output), input.rows, input.cols, prefilterCap);

    return k.run(2, globalThreads, NULL, false);
}

static void
prefilterXSobel( const Mat& src, Mat& dst, int ftzero )
{
    int x, y;
    const int OFS = 256*4, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ];
    Size size = src.size();

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero*2 : x - OFS + ftzero);
    uchar val0 = tab[0 + OFS];

#if CV_SSE2
    volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
#endif

    for( y = 0; y < size.height-1; y += 2 )
    {
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
        const uchar* srow2 = y < size.height-1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
        const uchar* srow3 = y < size.height-2 ? srow1 + src.step*2 : srow1;
        uchar* dptr0 = dst.ptr<uchar>(y);
        uchar* dptr1 = dptr0 + dst.step;

        dptr0[0] = dptr0[size.width-1] = dptr1[0] = dptr1[size.width-1] = val0;
        x = 1;

#if CV_SSE2
        if( useSIMD )
        {
            __m128i z = _mm_setzero_si128(), ftz = _mm_set1_epi16((short)ftzero),
            ftz2 = _mm_set1_epi8(cv::saturate_cast<uchar>(ftzero*2));
            for( ; x <= size.width-9; x += 8 )
            {
                __m128i c0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x - 1)), z);
                __m128i c1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x - 1)), z);
                __m128i d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x + 1)), z);
                __m128i d1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x + 1)), z);

                d0 = _mm_sub_epi16(d0, c0);
                d1 = _mm_sub_epi16(d1, c1);

                __m128i c2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x - 1)), z);
                __m128i c3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x - 1)), z);
                __m128i d2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x + 1)), z);
                __m128i d3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x + 1)), z);

                d2 = _mm_sub_epi16(d2, c2);
                d3 = _mm_sub_epi16(d3, c3);

                __m128i v0 = _mm_add_epi16(d0, _mm_add_epi16(d2, _mm_add_epi16(d1, d1)));
                __m128i v1 = _mm_add_epi16(d1, _mm_add_epi16(d3, _mm_add_epi16(d2, d2)));
                v0 = _mm_packus_epi16(_mm_add_epi16(v0, ftz), _mm_add_epi16(v1, ftz));
                v0 = _mm_min_epu8(v0, ftz2);

                _mm_storel_epi64((__m128i*)(dptr0 + x), v0);
                _mm_storel_epi64((__m128i*)(dptr1 + x), _mm_unpackhi_epi64(v0, v0));
            }
        }
#endif

        for( ; x < size.width-1; x++ )
        {
            int d0 = srow0[x+1] - srow0[x-1], d1 = srow1[x+1] - srow1[x-1],
            d2 = srow2[x+1] - srow2[x-1], d3 = srow3[x+1] - srow3[x-1];
            int v0 = tab[d0 + d1*2 + d2 + OFS];
            int v1 = tab[d1 + d2*2 + d3 + OFS];
            dptr0[x] = (uchar)v0;
            dptr1[x] = (uchar)v1;
        }
    }

    for( ; y < size.height; y++ )
    {
        uchar* dptr = dst.ptr<uchar>(y);
        for( x = 0; x < size.width; x++ )
            dptr[x] = val0;
    }
}


static const int DISPARITY_SHIFT = 4;

#if CV_SSE2
static void findStereoCorrespondenceBM_SSE2( const Mat& left, const Mat& right,
                                            Mat& disp, Mat& cost, StereoBMParams& state,
                                            uchar* buf, int _dy0, int _dy1 )
{
    const int ALIGN = 16;
    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);
    int ndisp = state.numDisparities;
    int mindisp = state.minDisparity;
    int lofs = MAX(ndisp - 1 + mindisp, 0);
    int rofs = -MIN(ndisp - 1 + mindisp, 0);
    int width = left.cols, height = left.rows;
    int width1 = width - rofs - ndisp + 1;
    int ftzero = state.preFilterCap;
    int textureThreshold = state.textureThreshold;
    int uniquenessRatio = state.uniquenessRatio;
    short FILTERED = (short)((mindisp - 1) << DISPARITY_SHIFT);

    ushort *sad, *hsad0, *hsad, *hsad_sub;
    int *htext;
    uchar *cbuf0, *cbuf;
    const uchar* lptr0 = left.ptr() + lofs;
    const uchar* rptr0 = right.ptr() + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    short* dptr = disp.ptr<short>();
    int sstep = (int)left.step;
    int dstep = (int)(disp.step/sizeof(dptr[0]));
    int cstep = (height + dy0 + dy1)*ndisp;
    short costbuf = 0;
    int coststep = cost.data ? (int)(cost.step/sizeof(costbuf)) : 0;
    const int TABSZ = 256;
    uchar tab[TABSZ];
    const __m128i d0_8 = _mm_setr_epi16(0,1,2,3,4,5,6,7), dd_8 = _mm_set1_epi16(8);

    sad = (ushort*)alignPtr(buf + sizeof(sad[0]), ALIGN);
    hsad0 = (ushort*)alignPtr(sad + ndisp + 1 + dy0*ndisp, ALIGN);
    htext = (int*)alignPtr((int*)(hsad0 + (height+dy1)*ndisp) + wsz2 + 2, ALIGN);
    cbuf0 = (uchar*)alignPtr((uchar*)(htext + height + wsz2 + 2) + dy0*ndisp, ALIGN);

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)std::abs(x - ftzero);

    // initialize buffers
    memset( hsad0 - dy0*ndisp, 0, (height + dy0 + dy1)*ndisp*sizeof(hsad0[0]) );
    memset( htext - wsz2 - 1, 0, (height + wsz + 1)*sizeof(htext[0]) );

    for( x = -wsz2-1; x < wsz2; x++ )
    {
        hsad = hsad0 - dy0*ndisp; cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0*ndisp;
        lptr = lptr0 + MIN(MAX(x, -lofs), width-lofs-1) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x, -rofs), width-rofs-1) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            __m128i lv = _mm_set1_epi8((char)lval), z = _mm_setzero_si128();
            for( d = 0; d < ndisp; d += 16 )
            {
                __m128i rv = _mm_loadu_si128((const __m128i*)(rptr + d));
                __m128i hsad_l = _mm_load_si128((__m128i*)(hsad + d));
                __m128i hsad_h = _mm_load_si128((__m128i*)(hsad + d + 8));
                __m128i diff = _mm_adds_epu8(_mm_subs_epu8(lv, rv), _mm_subs_epu8(rv, lv));
                _mm_store_si128((__m128i*)(cbuf + d), diff);
                hsad_l = _mm_add_epi16(hsad_l, _mm_unpacklo_epi8(diff,z));
                hsad_h = _mm_add_epi16(hsad_h, _mm_unpackhi_epi8(diff,z));
                _mm_store_si128((__m128i*)(hsad + d), hsad_l);
                _mm_store_si128((__m128i*)(hsad + d + 8), hsad_h);
            }
            htext[y] += tab[lval];
        }
    }

    // initialize the left and right borders of the disparity map
    for( y = 0; y < height; y++ )
    {
        for( x = 0; x < lofs; x++ )
            dptr[y*dstep + x] = FILTERED;
        for( x = lofs + width1; x < width; x++ )
            dptr[y*dstep + x] = FILTERED;
    }
    dptr += lofs;

    for( x = 0; x < width1; x++, dptr++ )
    {
        short* costptr = cost.data ? cost.ptr<short>() + lofs + x : &costbuf;
        int x0 = x - wsz2 - 1, x1 = x + wsz2;
        const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        hsad = hsad0 - dy0*ndisp;
        lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width-1-lofs) - dy0*sstep;
        lptr = lptr0 + MIN(MAX(x1, -lofs), width-1-lofs) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-1-rofs) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp,
            hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            __m128i lv = _mm_set1_epi8((char)lval), z = _mm_setzero_si128();
            for( d = 0; d < ndisp; d += 16 )
            {
                __m128i rv = _mm_loadu_si128((const __m128i*)(rptr + d));
                __m128i hsad_l = _mm_load_si128((__m128i*)(hsad + d));
                __m128i hsad_h = _mm_load_si128((__m128i*)(hsad + d + 8));
                __m128i cbs = _mm_load_si128((const __m128i*)(cbuf_sub + d));
                __m128i diff = _mm_adds_epu8(_mm_subs_epu8(lv, rv), _mm_subs_epu8(rv, lv));
                __m128i diff_h = _mm_sub_epi16(_mm_unpackhi_epi8(diff, z), _mm_unpackhi_epi8(cbs, z));
                _mm_store_si128((__m128i*)(cbuf + d), diff);
                diff = _mm_sub_epi16(_mm_unpacklo_epi8(diff, z), _mm_unpacklo_epi8(cbs, z));
                hsad_h = _mm_add_epi16(hsad_h, diff_h);
                hsad_l = _mm_add_epi16(hsad_l, diff);
                _mm_store_si128((__m128i*)(hsad + d), hsad_l);
                _mm_store_si128((__m128i*)(hsad + d + 8), hsad_h);
            }
            htext[y] += tab[lval] - tab[lptr_sub[0]];
        }

        // fill borders
        for( y = dy1; y <= wsz2; y++ )
            htext[height+y] = htext[height+dy1-1];
        for( y = -wsz2-1; y < -dy0; y++ )
            htext[y] = htext[-dy0];

        // initialize sums
        for( d = 0; d < ndisp; d++ )
            sad[d] = (ushort)(hsad0[d-ndisp*dy0]*(wsz2 + 2 - dy0));

        hsad = hsad0 + (1 - dy0)*ndisp;
        for( y = 1 - dy0; y < wsz2; y++, hsad += ndisp )
            for( d = 0; d < ndisp; d += 16 )
            {
                __m128i s0 = _mm_load_si128((__m128i*)(sad + d));
                __m128i s1 = _mm_load_si128((__m128i*)(sad + d + 8));
                __m128i t0 = _mm_load_si128((__m128i*)(hsad + d));
                __m128i t1 = _mm_load_si128((__m128i*)(hsad + d + 8));
                s0 = _mm_add_epi16(s0, t0);
                s1 = _mm_add_epi16(s1, t1);
                _mm_store_si128((__m128i*)(sad + d), s0);
                _mm_store_si128((__m128i*)(sad + d + 8), s1);
            }
        int tsum = 0;
        for( y = -wsz2-1; y < wsz2; y++ )
            tsum += htext[y];

        // finally, start the real processing
        for( y = 0; y < height; y++ )
        {
            int minsad = INT_MAX, mind = -1;
            hsad = hsad0 + MIN(y + wsz2, height+dy1-1)*ndisp;
            hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0)*ndisp;
            __m128i minsad8 = _mm_set1_epi16(SHRT_MAX);
            __m128i mind8 = _mm_set1_epi16(0), d8 = d0_8, mask;

            for( d = 0; d < ndisp; d += 16 )
            {
                __m128i u0 = _mm_load_si128((__m128i*)(hsad_sub + d));
                __m128i u1 = _mm_load_si128((__m128i*)(hsad + d));

                __m128i v0 = _mm_load_si128((__m128i*)(hsad_sub + d + 8));
                __m128i v1 = _mm_load_si128((__m128i*)(hsad + d + 8));

                __m128i usad8 = _mm_load_si128((__m128i*)(sad + d));
                __m128i vsad8 = _mm_load_si128((__m128i*)(sad + d + 8));

                u1 = _mm_sub_epi16(u1, u0);
                v1 = _mm_sub_epi16(v1, v0);
                usad8 = _mm_add_epi16(usad8, u1);
                vsad8 = _mm_add_epi16(vsad8, v1);

                mask = _mm_cmpgt_epi16(minsad8, usad8);
                minsad8 = _mm_min_epi16(minsad8, usad8);
                mind8 = _mm_max_epi16(mind8, _mm_and_si128(mask, d8));

                _mm_store_si128((__m128i*)(sad + d), usad8);
                _mm_store_si128((__m128i*)(sad + d + 8), vsad8);

                mask = _mm_cmpgt_epi16(minsad8, vsad8);
                minsad8 = _mm_min_epi16(minsad8, vsad8);

                d8 = _mm_add_epi16(d8, dd_8);
                mind8 = _mm_max_epi16(mind8, _mm_and_si128(mask, d8));
                d8 = _mm_add_epi16(d8, dd_8);
            }

            tsum += htext[y + wsz2] - htext[y - wsz2 - 1];
            if( tsum < textureThreshold )
            {
                dptr[y*dstep] = FILTERED;
                continue;
            }

            ushort CV_DECL_ALIGNED(16) minsad_buf[8], mind_buf[8];
            _mm_store_si128((__m128i*)minsad_buf, minsad8);
            _mm_store_si128((__m128i*)mind_buf, mind8);
            for( d = 0; d < 8; d++ )
                if(minsad > (int)minsad_buf[d] || (minsad == (int)minsad_buf[d] && mind > mind_buf[d]))
                {
                    minsad = minsad_buf[d];
                    mind = mind_buf[d];
                }

            if( uniquenessRatio > 0 )
            {
                int thresh = minsad + (minsad * uniquenessRatio/100);
                __m128i thresh8 = _mm_set1_epi16((short)(thresh + 1));
                __m128i d1 = _mm_set1_epi16((short)(mind-1)), d2 = _mm_set1_epi16((short)(mind+1));
                __m128i dd_16 = _mm_add_epi16(dd_8, dd_8);
                d8 = _mm_sub_epi16(d0_8, dd_16);

                for( d = 0; d < ndisp; d += 16 )
                {
                    __m128i usad8 = _mm_load_si128((__m128i*)(sad + d));
                    __m128i vsad8 = _mm_load_si128((__m128i*)(sad + d + 8));
                    mask = _mm_cmpgt_epi16( thresh8, _mm_min_epi16(usad8,vsad8));
                    d8 = _mm_add_epi16(d8, dd_16);
                    if( !_mm_movemask_epi8(mask) )
                        continue;
                    mask = _mm_cmpgt_epi16( thresh8, usad8);
                    mask = _mm_and_si128(mask, _mm_or_si128(_mm_cmpgt_epi16(d1,d8), _mm_cmpgt_epi16(d8,d2)));
                    if( _mm_movemask_epi8(mask) )
                        break;
                    __m128i t8 = _mm_add_epi16(d8, dd_8);
                    mask = _mm_cmpgt_epi16( thresh8, vsad8);
                    mask = _mm_and_si128(mask, _mm_or_si128(_mm_cmpgt_epi16(d1,t8), _mm_cmpgt_epi16(t8,d2)));
                    if( _mm_movemask_epi8(mask) )
                        break;
                }
                if( d < ndisp )
                {
                    dptr[y*dstep] = FILTERED;
                    continue;
                }
            }

            if( 0 < mind && mind < ndisp - 1 )
            {
                int p = sad[mind+1], n = sad[mind-1];
                d = p + n - 2*sad[mind] + std::abs(p - n);
                dptr[y*dstep] = (short)(((ndisp - mind - 1 + mindisp)*256 + (d != 0 ? (p-n)*256/d : 0) + 15) >> 4);
            }
            else
                dptr[y*dstep] = (short)((ndisp - mind - 1 + mindisp)*16);
            costptr[y*coststep] = sad[mind];
        }
    }
}
#endif

static void
findStereoCorrespondenceBM( const Mat& left, const Mat& right,
                           Mat& disp, Mat& cost, const StereoBMParams& state,
                           uchar* buf, int _dy0, int _dy1 )
{
    const int ALIGN = 16;
    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);
    int ndisp = state.numDisparities;
    int mindisp = state.minDisparity;
    int lofs = MAX(ndisp - 1 + mindisp, 0);
    int rofs = -MIN(ndisp - 1 + mindisp, 0);
    int width = left.cols, height = left.rows;
    int width1 = width - rofs - ndisp + 1;
    int ftzero = state.preFilterCap;
    int textureThreshold = state.textureThreshold;
    int uniquenessRatio = state.uniquenessRatio;
    short FILTERED = (short)((mindisp - 1) << DISPARITY_SHIFT);

    int *sad, *hsad0, *hsad, *hsad_sub, *htext;
    uchar *cbuf0, *cbuf;
    const uchar* lptr0 = left.ptr() + lofs;
    const uchar* rptr0 = right.ptr() + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    short* dptr = disp.ptr<short>();
    int sstep = (int)left.step;
    int dstep = (int)(disp.step/sizeof(dptr[0]));
    int cstep = (height+dy0+dy1)*ndisp;
    int costbuf = 0;
    int coststep = cost.data ? (int)(cost.step/sizeof(costbuf)) : 0;
    const int TABSZ = 256;
    uchar tab[TABSZ];

    sad = (int*)alignPtr(buf + sizeof(sad[0]), ALIGN);
    hsad0 = (int*)alignPtr(sad + ndisp + 1 + dy0*ndisp, ALIGN);
    htext = (int*)alignPtr((int*)(hsad0 + (height+dy1)*ndisp) + wsz2 + 2, ALIGN);
    cbuf0 = (uchar*)alignPtr((uchar*)(htext + height + wsz2 + 2) + dy0*ndisp, ALIGN);

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)std::abs(x - ftzero);

    // initialize buffers
    memset( hsad0 - dy0*ndisp, 0, (height + dy0 + dy1)*ndisp*sizeof(hsad0[0]) );
    memset( htext - wsz2 - 1, 0, (height + wsz + 1)*sizeof(htext[0]) );

    for( x = -wsz2-1; x < wsz2; x++ )
    {
        hsad = hsad0 - dy0*ndisp; cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0*ndisp;
        lptr = lptr0 + std::min(std::max(x, -lofs), width-lofs-1) - dy0*sstep;
        rptr = rptr0 + std::min(std::max(x, -rofs), width-rofs-1) - dy0*sstep;
        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] = (int)(hsad[d] + diff);
            }
            htext[y] += tab[lval];
        }
    }

    // initialize the left and right borders of the disparity map
    for( y = 0; y < height; y++ )
    {
        for( x = 0; x < lofs; x++ )
            dptr[y*dstep + x] = FILTERED;
        for( x = lofs + width1; x < width; x++ )
            dptr[y*dstep + x] = FILTERED;
    }
    dptr += lofs;

    for( x = 0; x < width1; x++, dptr++ )
    {
        int* costptr = cost.data ? cost.ptr<int>() + lofs + x : &costbuf;
        int x0 = x - wsz2 - 1, x1 = x + wsz2;
        const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        hsad = hsad0 - dy0*ndisp;
        lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width-1-lofs) - dy0*sstep;
        lptr = lptr0 + MIN(MAX(x1, -lofs), width-1-lofs) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-1-rofs) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp,
            hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] = hsad[d] + diff - cbuf_sub[d];
            }
            htext[y] += tab[lval] - tab[lptr_sub[0]];
        }

        // fill borders
        for( y = dy1; y <= wsz2; y++ )
            htext[height+y] = htext[height+dy1-1];
        for( y = -wsz2-1; y < -dy0; y++ )
            htext[y] = htext[-dy0];

        // initialize sums
        for( d = 0; d < ndisp; d++ )
            sad[d] = (int)(hsad0[d-ndisp*dy0]*(wsz2 + 2 - dy0));

        hsad = hsad0 + (1 - dy0)*ndisp;
        for( y = 1 - dy0; y < wsz2; y++, hsad += ndisp )
            for( d = 0; d < ndisp; d++ )
                sad[d] = (int)(sad[d] + hsad[d]);
        int tsum = 0;
        for( y = -wsz2-1; y < wsz2; y++ )
            tsum += htext[y];

        // finally, start the real processing
        for( y = 0; y < height; y++ )
        {
            int minsad = INT_MAX, mind = -1;
            hsad = hsad0 + MIN(y + wsz2, height+dy1-1)*ndisp;
            hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0)*ndisp;

            for( d = 0; d < ndisp; d++ )
            {
                int currsad = sad[d] + hsad[d] - hsad_sub[d];
                sad[d] = currsad;
                if( currsad < minsad )
                {
                    minsad = currsad;
                    mind = d;
                }
            }

            tsum += htext[y + wsz2] - htext[y - wsz2 - 1];
            if( tsum < textureThreshold )
            {
                dptr[y*dstep] = FILTERED;
                continue;
            }

            if( uniquenessRatio > 0 )
            {
                int thresh = minsad + (minsad * uniquenessRatio/100);
                for( d = 0; d < ndisp; d++ )
                {
                    if( sad[d] <= thresh && (d < mind-1 || d > mind+1))
                        break;
                }
                if( d < ndisp )
                {
                    dptr[y*dstep] = FILTERED;
                    continue;
                }
            }

            {
                sad[-1] = sad[1];
                sad[ndisp] = sad[ndisp-2];
                int p = sad[mind+1], n = sad[mind-1];
                d = p + n - 2*sad[mind] + std::abs(p - n);
                dptr[y*dstep] = (short)(((ndisp - mind - 1 + mindisp)*256 + (d != 0 ? (p-n)*256/d : 0) + 15) >> 4);
                costptr[y*coststep] = sad[mind];
            }
        }
    }
}

static bool ocl_prefiltering(InputArray left0, InputArray right0, OutputArray left, OutputArray right, StereoBMParams* state)
{
    if( state->preFilterType == StereoBM::PREFILTER_NORMALIZED_RESPONSE )
    {
        if(!ocl_prefilter_norm( left0, left, state->preFilterSize, state->preFilterCap))
            return false;
        if(!ocl_prefilter_norm( right0, right, state->preFilterSize, state->preFilterCap))
            return false;
    }
    else
    {
        if(!ocl_prefilter_xsobel( left0, left, state->preFilterCap ))
            return false;
        if(!ocl_prefilter_xsobel( right0, right, state->preFilterCap))
            return false;
    }
    return true;
}

struct PrefilterInvoker : public ParallelLoopBody
{
    PrefilterInvoker(const Mat& left0, const Mat& right0, Mat& left, Mat& right,
                     uchar* buf0, uchar* buf1, StereoBMParams* _state)
    {
        imgs0[0] = &left0; imgs0[1] = &right0;
        imgs[0] = &left; imgs[1] = &right;
        buf[0] = buf0; buf[1] = buf1;
        state = _state;
    }

    void operator()( const Range& range ) const
    {
        for( int i = range.start; i < range.end; i++ )
        {
            if( state->preFilterType == StereoBM::PREFILTER_NORMALIZED_RESPONSE )
                prefilterNorm( *imgs0[i], *imgs[i], state->preFilterSize, state->preFilterCap, buf[i] );
            else
                prefilterXSobel( *imgs0[i], *imgs[i], state->preFilterCap );
        }
    }

    const Mat* imgs0[2];
    Mat* imgs[2];
    uchar* buf[2];
    StereoBMParams* state;
};

static bool ocl_stereobm( InputArray _left, InputArray _right,
                       OutputArray _disp, StereoBMParams* state)
{
    int ndisp = state->numDisparities;
    int mindisp = state->minDisparity;
    int wsz = state->SADWindowSize;
    int wsz2 = wsz/2;

    int sizeX = std::max(11, 27 - ocl::Device::getDefault().maxComputeUnits() ), sizeY = sizeX-1, N = ndisp*2;

    ocl::Kernel k("stereoBM", ocl::calib3d::stereobm_oclsrc, cv::format("-D csize=%d -D wsz=%d", (2*sizeY)*ndisp, wsz) );
    if(k.empty())
        return false;

    UMat left = _left.getUMat(), right = _right.getUMat();
    int cols = left.cols, rows = left.rows;

    _disp.create(_left.size(), CV_16S);
    _disp.setTo((mindisp - 1)<<4);
    Rect roi = Rect(Point(wsz2 + mindisp + ndisp - 1, wsz2), Point(cols-wsz2-mindisp, rows-wsz2) );
    UMat disp = (_disp.getUMat())(roi);

    int globalX = disp.cols/sizeX, globalY = disp.rows/sizeY;
    globalX += (disp.cols%sizeX) > 0 ? 1 : 0;
    globalY += (disp.rows%sizeY) > 0 ? 1 : 0;
    size_t globalThreads[3] = { globalX, globalY, N};
    size_t localThreads[3] = {1, 1, N};

    int idx = 0;
    idx = k.set(idx, ocl::KernelArg::PtrReadOnly(left));
    idx = k.set(idx, ocl::KernelArg::PtrReadOnly(right));
    idx = k.set(idx, ocl::KernelArg::WriteOnlyNoSize(disp));
    idx = k.set(idx, rows);
    idx = k.set(idx, cols);
    idx = k.set(idx, mindisp);
    idx = k.set(idx, ndisp);
    idx = k.set(idx, state->preFilterCap);
    idx = k.set(idx, state->textureThreshold);
    idx = k.set(idx, state->uniquenessRatio);
    idx = k.set(idx, sizeX);
    idx = k.set(idx, sizeY);
    idx = k.set(idx, wsz);

    return k.run(3, globalThreads, localThreads, false);
}

struct FindStereoCorrespInvoker : public ParallelLoopBody
{
    FindStereoCorrespInvoker( const Mat& _left, const Mat& _right,
                             Mat& _disp, StereoBMParams* _state,
                             int _nstripes, size_t _stripeBufSize,
                             bool _useShorts, Rect _validDisparityRect,
                             Mat& _slidingSumBuf, Mat& _cost )
    {
        left = &_left; right = &_right;
        disp = &_disp; state = _state;
        nstripes = _nstripes; stripeBufSize = _stripeBufSize;
        useShorts = _useShorts;
        validDisparityRect = _validDisparityRect;
        slidingSumBuf = &_slidingSumBuf;
        cost = &_cost;
    }

    void operator()( const Range& range ) const
    {
        int cols = left->cols, rows = left->rows;
        int _row0 = std::min(cvRound(range.start * rows / nstripes), rows);
        int _row1 = std::min(cvRound(range.end * rows / nstripes), rows);
        uchar *ptr = slidingSumBuf->ptr() + range.start * stripeBufSize;
        int FILTERED = (state->minDisparity - 1)*16;

        Rect roi = validDisparityRect & Rect(0, _row0, cols, _row1 - _row0);
        if( roi.height == 0 )
            return;
        int row0 = roi.y;
        int row1 = roi.y + roi.height;

        Mat part;
        if( row0 > _row0 )
        {
            part = disp->rowRange(_row0, row0);
            part = Scalar::all(FILTERED);
        }
        if( _row1 > row1 )
        {
            part = disp->rowRange(row1, _row1);
            part = Scalar::all(FILTERED);
        }

        Mat left_i = left->rowRange(row0, row1);
        Mat right_i = right->rowRange(row0, row1);
        Mat disp_i = disp->rowRange(row0, row1);
        Mat cost_i = state->disp12MaxDiff >= 0 ? cost->rowRange(row0, row1) : Mat();

#if CV_SSE2
        if( useShorts )
            findStereoCorrespondenceBM_SSE2( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1 );
        else
#endif
            findStereoCorrespondenceBM( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1 );

        if( state->disp12MaxDiff >= 0 )
            validateDisparity( disp_i, cost_i, state->minDisparity, state->numDisparities, state->disp12MaxDiff );

        if( roi.x > 0 )
        {
            part = disp_i.colRange(0, roi.x);
            part = Scalar::all(FILTERED);
        }
        if( roi.x + roi.width < cols )
        {
            part = disp_i.colRange(roi.x + roi.width, cols);
            part = Scalar::all(FILTERED);
        }
    }

protected:
    const Mat *left, *right;
    Mat* disp, *slidingSumBuf, *cost;
    StereoBMParams *state;

    int nstripes;
    size_t stripeBufSize;
    bool useShorts;
    Rect validDisparityRect;
};

class StereoBMImpl : public StereoBM
{
public:
    StereoBMImpl()
    {
        params = StereoBMParams();
    }

    StereoBMImpl( int _numDisparities, int _SADWindowSize )
    {
        params = StereoBMParams(_numDisparities, _SADWindowSize);
    }

    void compute( InputArray leftarr, InputArray rightarr, OutputArray disparr )
    {
        int dtype = disparr.fixedType() ? disparr.type() : params.dispType;
        Size leftsize = leftarr.size();

        if (leftarr.size() != rightarr.size())
            CV_Error( Error::StsUnmatchedSizes, "All the images must have the same size" );

        if (leftarr.type() != CV_8UC1 || rightarr.type() != CV_8UC1)
            CV_Error( Error::StsUnsupportedFormat, "Both input images must have CV_8UC1" );

        if (dtype != CV_16SC1 && dtype != CV_32FC1)
            CV_Error( Error::StsUnsupportedFormat, "Disparity image must have CV_16SC1 or CV_32FC1 format" );

        if( params.preFilterType != PREFILTER_NORMALIZED_RESPONSE &&
            params.preFilterType != PREFILTER_XSOBEL )
            CV_Error( Error::StsOutOfRange, "preFilterType must be = CV_STEREO_BM_NORMALIZED_RESPONSE" );

        if( params.preFilterSize < 5 || params.preFilterSize > 255 || params.preFilterSize % 2 == 0 )
            CV_Error( Error::StsOutOfRange, "preFilterSize must be odd and be within 5..255" );

        if( params.preFilterCap < 1 || params.preFilterCap > 63 )
            CV_Error( Error::StsOutOfRange, "preFilterCap must be within 1..63" );

        if( params.SADWindowSize < 5 || params.SADWindowSize > 255 || params.SADWindowSize % 2 == 0 ||
            params.SADWindowSize >= std::min(leftsize.width, leftsize.height) )
            CV_Error( Error::StsOutOfRange, "SADWindowSize must be odd, be within 5..255 and be not larger than image width or height" );

        if( params.numDisparities <= 0 || params.numDisparities % 16 != 0 )
            CV_Error( Error::StsOutOfRange, "numDisparities must be positive and divisble by 16" );

        if( params.textureThreshold < 0 )
            CV_Error( Error::StsOutOfRange, "texture threshold must be non-negative" );

        if( params.uniquenessRatio < 0 )
            CV_Error( Error::StsOutOfRange, "uniqueness ratio must be non-negative" );

        int FILTERED = (params.minDisparity - 1) << DISPARITY_SHIFT;

        if(ocl::useOpenCL() && disparr.isUMat() && params.textureThreshold == 0)
        {
            UMat left, right;
            if(ocl_prefiltering(leftarr, rightarr, left, right, &params))
            {
                if(ocl_stereobm(left, right, disparr, &params))
                {
                    if( params.speckleRange >= 0 && params.speckleWindowSize > 0 )
                        filterSpeckles(disparr.getMat(), FILTERED, params.speckleWindowSize, params.speckleRange, slidingSumBuf);
                    if (dtype == CV_32F)
                        disparr.getUMat().convertTo(disparr, CV_32FC1, 1./(1 << DISPARITY_SHIFT), 0);
                    return;
                }
            }
        }

        Mat left0 = leftarr.getMat(), right0 = rightarr.getMat();
        disparr.create(left0.size(), dtype);
        Mat disp0 = disparr.getMat();

        preFilteredImg0.create( left0.size(), CV_8U );
        preFilteredImg1.create( left0.size(), CV_8U );
        cost.create( left0.size(), CV_16S );

        Mat left = preFilteredImg0, right = preFilteredImg1;

        int mindisp = params.minDisparity;
        int ndisp = params.numDisparities;

        int width = left0.cols;
        int height = left0.rows;
        int lofs = std::max(ndisp - 1 + mindisp, 0);
        int rofs = -std::min(ndisp - 1 + mindisp, 0);
        int width1 = width - rofs - ndisp + 1;

        if( lofs >= width || rofs >= width || width1 < 1 )
        {
            disp0 = Scalar::all( FILTERED * ( disp0.type() < CV_32F ? 1 : 1./(1 << DISPARITY_SHIFT) ) );
            return;
        }

        Mat disp = disp0;
        if( dtype == CV_32F )
        {
            dispbuf.create(disp0.size(), CV_16S);
            disp = dispbuf;
        }

        int wsz = params.SADWindowSize;
        int bufSize0 = (int)((ndisp + 2)*sizeof(int));
        bufSize0 += (int)((height+wsz+2)*ndisp*sizeof(int));
        bufSize0 += (int)((height + wsz + 2)*sizeof(int));
        bufSize0 += (int)((height+wsz+2)*ndisp*(wsz+2)*sizeof(uchar) + 256);

        int bufSize1 = (int)((width + params.preFilterSize + 2) * sizeof(int) + 256);
        int bufSize2 = 0;
        if( params.speckleRange >= 0 && params.speckleWindowSize > 0 )
            bufSize2 = width*height*(sizeof(Point_<short>) + sizeof(int) + sizeof(uchar));

#if CV_SSE2
        bool useShorts = params.preFilterCap <= 31 && params.SADWindowSize <= 21 && checkHardwareSupport(CV_CPU_SSE2);
#else
        const bool useShorts = false;
#endif

        const double SAD_overhead_coeff = 10.0;
        double N0 = 8000000 / (useShorts ? 1 : 4);  // approx tbb's min number instructions reasonable for one thread
        double maxStripeSize = std::min(std::max(N0 / (width * ndisp), (wsz-1) * SAD_overhead_coeff), (double)height);
        int nstripes = cvCeil(height / maxStripeSize);
        int bufSize = std::max(bufSize0 * nstripes, std::max(bufSize1 * 2, bufSize2));

        if( slidingSumBuf.cols < bufSize )
            slidingSumBuf.create( 1, bufSize, CV_8U );

        uchar *_buf = slidingSumBuf.ptr();

        parallel_for_(Range(0, 2), PrefilterInvoker(left0, right0, left, right, _buf, _buf + bufSize1, &params), 1);

        Rect validDisparityRect(0, 0, width, height), R1 = params.roi1, R2 = params.roi2;
        validDisparityRect = getValidDisparityROI(R1.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                                  R2.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                                  params.minDisparity, params.numDisparities,
                                                  params.SADWindowSize);

        parallel_for_(Range(0, nstripes),
                      FindStereoCorrespInvoker(left, right, disp, &params, nstripes,
                                               bufSize0, useShorts, validDisparityRect,
                                               slidingSumBuf, cost));

        if( params.speckleRange >= 0 && params.speckleWindowSize > 0 )
            filterSpeckles(disp, FILTERED, params.speckleWindowSize, params.speckleRange, slidingSumBuf);

        if (disp0.data != disp.data)
            disp.convertTo(disp0, disp0.type(), 1./(1 << DISPARITY_SHIFT), 0);
    }

    AlgorithmInfo* info() const { return 0; }

    int getMinDisparity() const { return params.minDisparity; }
    void setMinDisparity(int minDisparity) { params.minDisparity = minDisparity; }

    int getNumDisparities() const { return params.numDisparities; }
    void setNumDisparities(int numDisparities) { params.numDisparities = numDisparities; }

    int getBlockSize() const { return params.SADWindowSize; }
    void setBlockSize(int blockSize) { params.SADWindowSize = blockSize; }

    int getSpeckleWindowSize() const { return params.speckleWindowSize; }
    void setSpeckleWindowSize(int speckleWindowSize) { params.speckleWindowSize = speckleWindowSize; }

    int getSpeckleRange() const { return params.speckleRange; }
    void setSpeckleRange(int speckleRange) { params.speckleRange = speckleRange; }

    int getDisp12MaxDiff() const { return params.disp12MaxDiff; }
    void setDisp12MaxDiff(int disp12MaxDiff) { params.disp12MaxDiff = disp12MaxDiff; }

    int getPreFilterType() const { return params.preFilterType; }
    void setPreFilterType(int preFilterType) { params.preFilterType = preFilterType; }

    int getPreFilterSize() const { return params.preFilterSize; }
    void setPreFilterSize(int preFilterSize) { params.preFilterSize = preFilterSize; }

    int getPreFilterCap() const { return params.preFilterCap; }
    void setPreFilterCap(int preFilterCap) { params.preFilterCap = preFilterCap; }

    int getTextureThreshold() const { return params.textureThreshold; }
    void setTextureThreshold(int textureThreshold) { params.textureThreshold = textureThreshold; }

    int getUniquenessRatio() const { return params.uniquenessRatio; }
    void setUniquenessRatio(int uniquenessRatio) { params.uniquenessRatio = uniquenessRatio; }

    int getSmallerBlockSize() const { return 0; }
    void setSmallerBlockSize(int) {}

    Rect getROI1() const { return params.roi1; }
    void setROI1(Rect roi1) { params.roi1 = roi1; }

    Rect getROI2() const { return params.roi2; }
    void setROI2(Rect roi2) { params.roi2 = roi2; }

    void write(FileStorage& fs) const
    {
        fs << "name" << name_
        << "minDisparity" << params.minDisparity
        << "numDisparities" << params.numDisparities
        << "blockSize" << params.SADWindowSize
        << "speckleWindowSize" << params.speckleWindowSize
        << "speckleRange" << params.speckleRange
        << "disp12MaxDiff" << params.disp12MaxDiff
        << "preFilterType" << params.preFilterType
        << "preFilterSize" << params.preFilterSize
        << "preFilterCap" << params.preFilterCap
        << "textureThreshold" << params.textureThreshold
        << "uniquenessRatio" << params.uniquenessRatio;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert( n.isString() && String(n) == name_ );
        params.minDisparity = (int)fn["minDisparity"];
        params.numDisparities = (int)fn["numDisparities"];
        params.SADWindowSize = (int)fn["blockSize"];
        params.speckleWindowSize = (int)fn["speckleWindowSize"];
        params.speckleRange = (int)fn["speckleRange"];
        params.disp12MaxDiff = (int)fn["disp12MaxDiff"];
        params.preFilterType = (int)fn["preFilterType"];
        params.preFilterSize = (int)fn["preFilterSize"];
        params.preFilterCap = (int)fn["preFilterCap"];
        params.textureThreshold = (int)fn["textureThreshold"];
        params.uniquenessRatio = (int)fn["uniquenessRatio"];
        params.roi1 = params.roi2 = Rect();
    }

    StereoBMParams params;
    Mat preFilteredImg0, preFilteredImg1, cost, dispbuf;
    Mat slidingSumBuf;

    static const char* name_;
};

const char* StereoBMImpl::name_ = "StereoMatcher.BM";

}

cv::Ptr<cv::StereoBM> cv::createStereoBM(int _numDisparities, int _SADWindowSize)
{
    return makePtr<StereoBMImpl>(_numDisparities, _SADWindowSize);
}

/* End of file. */
