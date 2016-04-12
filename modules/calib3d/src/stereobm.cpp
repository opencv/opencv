//M*//////////////////////////////////////////////////////////////////////////////////////
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

/****************************************************************************************\
*    Very fast SAD-based (Sum-of-Absolute-Diffrences) stereo correspondence algorithm.   *
*    Contributed by Kurt Konolige                                                        *
\****************************************************************************************/

#include "precomp.hpp"
#include <stdio.h>

//#undef CV_SSE2
//#define CV_SSE2 0
//#include "emmintrin.h"


#include <limits>

CV_IMPL CvStereoBMState* cvCreateStereoBMState( int /*preset*/, int numberOfDisparities )
{
    CvStereoBMState* state = (CvStereoBMState*)cvAlloc( sizeof(*state) );
    if( !state )
        return 0;

    state->preFilterType = CV_STEREO_BM_XSOBEL; //CV_STEREO_BM_NORMALIZED_RESPONSE;
    state->preFilterSize = 9;
    state->preFilterCap = 31;
    state->SADWindowSize = 15;
    state->minDisparity = 0;
    state->numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : 64;
    state->textureThreshold = 10;
    state->uniquenessRatio = 15;
    state->speckleRange = state->speckleWindowSize = 0;
    state->trySmallerWindows = 0;
    state->roi1 = state->roi2 = cvRect(0,0,0,0);
    state->disp12MaxDiff = -1;

    state->preFilteredImg0 = state->preFilteredImg1 = state->slidingSumBuf =
    state->disp = state->cost = 0;

    return state;
}

CV_IMPL void cvReleaseStereoBMState( CvStereoBMState** state )
{
    if( !state )
        CV_Error( CV_StsNullPtr, "" );

    if( !*state )
        return;

    cvReleaseMat( &(*state)->preFilteredImg0 );
    cvReleaseMat( &(*state)->preFilteredImg1 );
    cvReleaseMat( &(*state)->slidingSumBuf );
    cvReleaseMat( &(*state)->disp );
    cvReleaseMat( &(*state)->cost );
    cvFree( state );
}

namespace cv
{

static void prefilterNorm( const Mat& src, Mat& dst, int winsize, int ftzero, uchar* buf )
{
    int x, y, wsz2 = winsize/2;
    int* vsum = (int*)alignPtr(buf + (wsz2 + 1)*sizeof(vsum[0]), 32);
    int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
    const int OFS = 256*5, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ];
    const uchar* sptr = src.data;
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
        x = 0;

        for( ; x < size.width; x++ )
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
                ftz2 = _mm_set1_epi8(CV_CAST_8U(ftzero*2));
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
    #if CV_NEON
        int16x8_t ftz = vdupq_n_s16 ((short) ftzero);
        uint8x8_t ftz2 = vdup_n_u8 (cv::saturate_cast<uchar>(ftzero*2));

        for(; x <=size.width-9; x += 8 )
        {
            uint8x8_t c0 = vld1_u8 (srow0 + x - 1);
            uint8x8_t c1 = vld1_u8 (srow1 + x - 1);
            uint8x8_t d0 = vld1_u8 (srow0 + x + 1);
            uint8x8_t d1 = vld1_u8 (srow1 + x + 1);

            int16x8_t t0 = vreinterpretq_s16_u16 (vsubl_u8 (d0, c0));
            int16x8_t t1 = vreinterpretq_s16_u16 (vsubl_u8 (d1, c1));

            uint8x8_t c2 = vld1_u8 (srow2 + x - 1);
            uint8x8_t c3 = vld1_u8 (srow3 + x - 1);
            uint8x8_t d2 = vld1_u8 (srow2 + x + 1);
            uint8x8_t d3 = vld1_u8 (srow3 + x + 1);

            int16x8_t t2 = vreinterpretq_s16_u16 (vsubl_u8 (d2, c2));
            int16x8_t t3 = vreinterpretq_s16_u16 (vsubl_u8 (d3, c3));

            int16x8_t v0 = vaddq_s16 (vaddq_s16 (t2, t0), vaddq_s16 (t1, t1));
            int16x8_t v1 = vaddq_s16 (vaddq_s16 (t3, t1), vaddq_s16 (t2, t2));


            uint8x8_t v0_u8 = vqmovun_s16 (vaddq_s16 (v0, ftz));
            uint8x8_t v1_u8 = vqmovun_s16 (vaddq_s16 (v1, ftz));
            v0_u8 =  vmin_u8 (v0_u8, ftz2);
            v1_u8 =  vmin_u8 (v1_u8, ftz2);
            vqmovun_s16 (vaddq_s16 (v1, ftz));

            vst1_u8 (dptr0 + x, v0_u8);
            vst1_u8 (dptr1 + x, v1_u8);
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

#if CV_NEON
    uint8x16_t val0_16 = vdupq_n_u8 (val0);
#endif

    for( ; y < size.height; y++ )
    {
        uchar* dptr = dst.ptr<uchar>(y);
        x = 0;
    #if CV_NEON
        for(; x <= size.width-16; x+=16 )
            vst1q_u8 (dptr + x, val0_16);
    #endif
        for(; x < size.width; x++ )
            dptr[x] = val0;
    }
}


static const int DISPARITY_SHIFT = 4;

#if CV_SSE2
static void findStereoCorrespondenceBM_SSE2( const Mat& left, const Mat& right,
                                             Mat& disp, Mat& cost, CvStereoBMState& state,
                                             uchar* buf, int _dy0, int _dy1 )
{
    const int ALIGN = 16;
    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);
    int ndisp = state.numberOfDisparities;
    int mindisp = state.minDisparity;
    int lofs = MAX(ndisp - 1 + mindisp, 0);
    int rofs = -MIN(ndisp - 1 + mindisp, 0);
    int width = left.cols, height = left.rows;
    int width1 = width - rofs - ndisp + 1;
    int ftzero = state.preFilterCap;
    int textureThreshold = state.textureThreshold;
    int uniquenessRatio = state.uniquenessRatio*256/100;
    short FILTERED = (short)((mindisp - 1) << DISPARITY_SHIFT);

    ushort *sad, *hsad0, *hsad, *hsad_sub;
    int *htext;
    uchar *cbuf0, *cbuf;
    const uchar* lptr0 = left.data + lofs;
    const uchar* rptr0 = right.data + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    short* dptr = (short*)disp.data;
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
    cbuf0 = (uchar*)alignPtr(htext + height + wsz2 + 2 + dy0*ndisp, ALIGN);

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)std::abs(x - ftzero);

    // initialize buffers
    memset( hsad0 - dy0*ndisp, 0, (height + dy0 + dy1)*ndisp*sizeof(hsad0[0]) );
    memset( htext - wsz2 - 1, 0, (height + wsz + 1)*sizeof(htext[0]) );

    for( x = -wsz2-1; x < wsz2; x++ )
    {
        hsad = hsad0 - dy0*ndisp; cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0*ndisp;
        lptr = lptr0 + MIN(MAX(x, -lofs), width-lofs-1) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x, -rofs), width-rofs-ndisp) - dy0*sstep;

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
        short* costptr = cost.data ? (short*)cost.data + lofs + x : &costbuf;
        int x0 = x - wsz2 - 1, x1 = x + wsz2;
        const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        hsad = hsad0 - dy0*ndisp;
        lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width-1-lofs) - dy0*sstep;
        lptr = lptr0 + MIN(MAX(x1, -lofs), width-1-lofs) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-ndisp-rofs) - dy0*sstep;

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

            __m128i minsad82 = _mm_unpackhi_epi64(minsad8, minsad8);
            __m128i mind82 = _mm_unpackhi_epi64(mind8, mind8);
            mask = _mm_cmpgt_epi16(minsad8, minsad82);
            mind8 = _mm_xor_si128(mind8,_mm_and_si128(_mm_xor_si128(mind82,mind8),mask));
            minsad8 = _mm_min_epi16(minsad8, minsad82);

            minsad82 = _mm_shufflelo_epi16(minsad8, _MM_SHUFFLE(3,2,3,2));
            mind82 = _mm_shufflelo_epi16(mind8, _MM_SHUFFLE(3,2,3,2));
            mask = _mm_cmpgt_epi16(minsad8, minsad82);
            mind8 = _mm_xor_si128(mind8,_mm_and_si128(_mm_xor_si128(mind82,mind8),mask));
            minsad8 = _mm_min_epi16(minsad8, minsad82);

            minsad82 = _mm_shufflelo_epi16(minsad8, 1);
            mind82 = _mm_shufflelo_epi16(mind8, 1);
            mask = _mm_cmpgt_epi16(minsad8, minsad82);
            mind8 = _mm_xor_si128(mind8,_mm_and_si128(_mm_xor_si128(mind82,mind8),mask));
            mind = (short)_mm_cvtsi128_si32(mind8);
            minsad = sad[mind];

            if( uniquenessRatio > 0 )
            {
                int thresh = minsad + ((minsad * uniquenessRatio) >> 8);
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
                            Mat& disp, Mat& cost, const CvStereoBMState& state,
                            uchar* buf, int _dy0, int _dy1 )
{

    const int ALIGN = 16;
    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);
    int ndisp = state.numberOfDisparities;
    int mindisp = state.minDisparity;
    int lofs = MAX(ndisp - 1 + mindisp, 0);
    int rofs = -MIN(ndisp - 1 + mindisp, 0);
    int width = left.cols, height = left.rows;
    int width1 = width - rofs - ndisp + 1;
    int ftzero = state.preFilterCap;
    int textureThreshold = state.textureThreshold;
    int uniquenessRatio = state.uniquenessRatio;
    short FILTERED = (short)((mindisp - 1) << DISPARITY_SHIFT);

#if CV_NEON
    CV_Assert (ndisp % 8 == 0);
    int32_t d0_4_temp [4];
    for (int i = 0; i < 4; i ++)
        d0_4_temp[i] = i;
    int32x4_t d0_4 = vld1q_s32 (d0_4_temp);
    int32x4_t dd_4 = vdupq_n_s32 (4);
#endif

    int *sad, *hsad0, *hsad, *hsad_sub, *htext;
    uchar *cbuf0, *cbuf;
    const uchar* lptr0 = left.data + lofs;
    const uchar* rptr0 = right.data + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    short* dptr = (short*)disp.data;
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
        rptr = rptr0 + std::min(std::max(x, -rofs), width-rofs-ndisp) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0];
        #if CV_NEON
            int16x8_t lv = vdupq_n_s16 ((int16_t)lval);

            for( d = 0; d < ndisp; d += 8 )
            {
                int16x8_t rv = vreinterpretq_s16_u16 (vmovl_u8 (vld1_u8 (rptr + d)));
                int32x4_t hsad_l = vld1q_s32 (hsad + d);
                int32x4_t hsad_h = vld1q_s32 (hsad + d + 4);
                int16x8_t diff = vabdq_s16 (lv, rv);
                vst1_u8 (cbuf + d, vmovn_u16(vreinterpretq_u16_s16(diff)));
                hsad_l = vaddq_s32 (hsad_l, vmovl_s16(vget_low_s16 (diff)));
                hsad_h = vaddq_s32 (hsad_h, vmovl_s16(vget_high_s16 (diff)));
                vst1q_s32 ((hsad + d), hsad_l);
                vst1q_s32 ((hsad + d + 4), hsad_h);
            }
        #else
            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] = (int)(hsad[d] + diff);
            }
        #endif
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
        int* costptr = cost.data ? (int*)cost.data + lofs + x : &costbuf;
        int x0 = x - wsz2 - 1, x1 = x + wsz2;
        const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        hsad = hsad0 - dy0*ndisp;
        lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width-1-lofs) - dy0*sstep;
        lptr = lptr0 + MIN(MAX(x1, -lofs), width-1-lofs) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-ndisp-rofs) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp,
             hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep )
        {
            int lval = lptr[0];
        #if CV_NEON
            int16x8_t lv = vdupq_n_s16 ((int16_t)lval);
            for( d = 0; d < ndisp; d += 8 )
            {
                int16x8_t rv = vreinterpretq_s16_u16 (vmovl_u8 (vld1_u8 (rptr + d)));
                int32x4_t hsad_l = vld1q_s32 (hsad + d);
                int32x4_t hsad_h = vld1q_s32 (hsad + d + 4);
                int16x8_t cbs = vreinterpretq_s16_u16 (vmovl_u8 (vld1_u8 (cbuf_sub + d)));
                int16x8_t diff = vabdq_s16 (lv, rv);
                int32x4_t diff_h = vsubl_s16 (vget_high_s16 (diff), vget_high_s16 (cbs));
                int32x4_t diff_l = vsubl_s16 (vget_low_s16 (diff), vget_low_s16 (cbs));
                vst1_u8 (cbuf + d, vmovn_u16(vreinterpretq_u16_s16(diff)));
                hsad_h = vaddq_s32 (hsad_h, diff_h);
                hsad_l = vaddq_s32 (hsad_l, diff_l);
                vst1q_s32 ((hsad + d), hsad_l);
                vst1q_s32 ((hsad + d + 4), hsad_h);
            }
        #else
            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] = hsad[d] + diff - cbuf_sub[d];
            }
        #endif
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
        {
        #if CV_NEON
            for( d = 0; d <= ndisp-8; d += 8 )
            {
                int32x4_t s0 = vld1q_s32 (sad + d);
                int32x4_t s1 = vld1q_s32 (sad + d + 4);
                int32x4_t t0 = vld1q_s32 (hsad + d);
                int32x4_t t1 = vld1q_s32 (hsad + d + 4);
                s0 = vaddq_s32 (s0, t0);
                s1 = vaddq_s32 (s1, t1);
                vst1q_s32 (sad + d, s0);
                vst1q_s32 (sad + d + 4, s1);
            }
        #else
            for( d = 0; d < ndisp; d++ )
                sad[d] = (int)(sad[d] + hsad[d]);
        #endif
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
        #if CV_NEON
            int32x4_t minsad4 = vdupq_n_s32 (INT_MAX);
            int32x4_t mind4 = vdupq_n_s32(0), d4 = d0_4;

            for( d = 0; d <= ndisp-8; d += 8 )
            {
                int32x4_t u0 = vld1q_s32 (hsad_sub + d);
                int32x4_t u1 = vld1q_s32 (hsad + d);

                int32x4_t v0 = vld1q_s32 (hsad_sub + d + 4);
                int32x4_t v1 = vld1q_s32 (hsad + d + 4);

                int32x4_t usad4 = vld1q_s32(sad + d);
                int32x4_t vsad4 = vld1q_s32(sad + d + 4);

                u1 = vsubq_s32 (u1, u0);
                v1 = vsubq_s32 (v1, v0);
                usad4 = vaddq_s32 (usad4, u1);
                vsad4 = vaddq_s32 (vsad4, v1);

                uint32x4_t mask = vcgtq_s32 (minsad4, usad4);
                minsad4 = vminq_s32 (minsad4, usad4);
                mind4 = vbslq_s32(mask, d4, mind4);

                vst1q_s32 (sad + d, usad4);
                vst1q_s32 (sad + d + 4, vsad4);
                d4 = vaddq_s32 (d4, dd_4);

                mask = vcgtq_s32 (minsad4, vsad4);
                minsad4 = vminq_s32 (minsad4, vsad4);
                mind4 = vbslq_s32(mask, d4, mind4);

                d4 = vaddq_s32 (d4, dd_4);

            }
            int32x2_t mind4_h = vget_high_s32 (mind4);
            int32x2_t mind4_l = vget_low_s32 (mind4);
            int32x2_t minsad4_h = vget_high_s32 (minsad4);
            int32x2_t minsad4_l = vget_low_s32 (minsad4);

            uint32x2_t mask = vorr_u32 (vclt_s32 (minsad4_h, minsad4_l), vand_u32 (vceq_s32 (minsad4_h, minsad4_l), vclt_s32 (mind4_h, mind4_l)));
            mind4_h = vbsl_s32 (mask, mind4_h, mind4_l);
            minsad4_h = vbsl_s32 (mask, minsad4_h, minsad4_l);

            mind4_l = vext_s32 (mind4_h,mind4_h,1);
            minsad4_l = vext_s32 (minsad4_h,minsad4_h,1);

            mask = vorr_u32 (vclt_s32 (minsad4_h, minsad4_l), vand_u32 (vceq_s32 (minsad4_h, minsad4_l), vclt_s32 (mind4_h, mind4_l)));
            mind4_h = vbsl_s32 (mask, mind4_h, mind4_l);
            minsad4_h = vbsl_s32 (mask, minsad4_h, minsad4_l);

            mind = (int) vget_lane_s32 (mind4_h, 0);
            minsad = sad[mind];

        #else
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
        #endif
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

struct PrefilterInvoker
{
    PrefilterInvoker(const Mat& left0, const Mat& right0, Mat& left, Mat& right,
                     uchar* buf0, uchar* buf1, CvStereoBMState* _state )
    {
        imgs0[0] = &left0; imgs0[1] = &right0;
        imgs[0] = &left; imgs[1] = &right;
        buf[0] = buf0; buf[1] = buf1;
        state = _state;
    }

    void operator()( int ind ) const
    {
        if( state->preFilterType == CV_STEREO_BM_NORMALIZED_RESPONSE )
            prefilterNorm( *imgs0[ind], *imgs[ind], state->preFilterSize, state->preFilterCap, buf[ind] );
        else
            prefilterXSobel( *imgs0[ind], *imgs[ind], state->preFilterCap );
    }

    const Mat* imgs0[2];
    Mat* imgs[2];
    uchar* buf[2];
    CvStereoBMState *state;
};


struct FindStereoCorrespInvoker : ParallelLoopBody
{
    FindStereoCorrespInvoker( const Mat& _left, const Mat& _right,
                              Mat& _disp, CvStereoBMState* _state,
                              int _nstripes, int _stripeBufSize,
                              bool _useShorts, Rect _validDisparityRect )
    {
        left = &_left; right = &_right;
        disp = &_disp; state = _state;
        nstripes = _nstripes; stripeBufSize = _stripeBufSize;
        useShorts = _useShorts;
        validDisparityRect = _validDisparityRect;
    }

    void operator()( const Range& range ) const
    {
        int cols = left->cols, rows = left->rows;
        int _row0 = min(cvRound(range.start * rows / nstripes), rows);
        int _row1 = min(cvRound(range.end * rows / nstripes), rows);
        uchar *ptr = state->slidingSumBuf->data.ptr + range.start * stripeBufSize;
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
        Mat cost_i = state->disp12MaxDiff >= 0 ? Mat(state->cost).rowRange(row0, row1) : Mat();

#if CV_SSE2
        if( useShorts )
            findStereoCorrespondenceBM_SSE2( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1 );
        else
#endif
            findStereoCorrespondenceBM( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1 );

        if( state->disp12MaxDiff >= 0 )
            validateDisparity( disp_i, cost_i, state->minDisparity, state->numberOfDisparities, state->disp12MaxDiff );

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
    Mat* disp;
    CvStereoBMState *state;

    int nstripes;
    int stripeBufSize;
    bool useShorts;
    Rect validDisparityRect;
};

static void findStereoCorrespondenceBM( const Mat& left0, const Mat& right0, Mat& disp0, CvStereoBMState* state)
{
    if (left0.size() != right0.size() || disp0.size() != left0.size())
        CV_Error( CV_StsUnmatchedSizes, "All the images must have the same size" );

    if (left0.type() != CV_8UC1 || right0.type() != CV_8UC1)
        CV_Error( CV_StsUnsupportedFormat, "Both input images must have CV_8UC1" );

    if (disp0.type() != CV_16SC1 && disp0.type() != CV_32FC1)
        CV_Error( CV_StsUnsupportedFormat, "Disparity image must have CV_16SC1 or CV_32FC1 format" );

    if( !state )
        CV_Error( CV_StsNullPtr, "Stereo BM state is NULL." );

    if( state->preFilterType != CV_STEREO_BM_NORMALIZED_RESPONSE && state->preFilterType != CV_STEREO_BM_XSOBEL )
        CV_Error( CV_StsOutOfRange, "preFilterType must be = CV_STEREO_BM_NORMALIZED_RESPONSE" );

    if( state->preFilterSize < 5 || state->preFilterSize > 255 || state->preFilterSize % 2 == 0 )
        CV_Error( CV_StsOutOfRange, "preFilterSize must be odd and be within 5..255" );

    if( state->preFilterCap < 1 || state->preFilterCap > 63 )
        CV_Error( CV_StsOutOfRange, "preFilterCap must be within 1..63" );

    if( state->SADWindowSize < 5 || state->SADWindowSize > 255 || state->SADWindowSize % 2 == 0 ||
        state->SADWindowSize >= min(left0.cols, left0.rows) )
        CV_Error( CV_StsOutOfRange, "SADWindowSize must be odd, be within 5..255 and be not larger than image width or height" );

    if( state->numberOfDisparities <= 0 || state->numberOfDisparities % 16 != 0 )
        CV_Error( CV_StsOutOfRange, "numberOfDisparities must be positive and divisble by 16" );

    if( state->textureThreshold < 0 )
        CV_Error( CV_StsOutOfRange, "texture threshold must be non-negative" );

    if( state->uniquenessRatio < 0 )
        CV_Error( CV_StsOutOfRange, "uniqueness ratio must be non-negative" );

    if( !state->preFilteredImg0 || state->preFilteredImg0->cols * state->preFilteredImg0->rows < left0.cols * left0.rows )
    {
        cvReleaseMat( &state->preFilteredImg0 );
        cvReleaseMat( &state->preFilteredImg1 );
        cvReleaseMat( &state->cost );

        state->preFilteredImg0 = cvCreateMat( left0.rows, left0.cols, CV_8U );
        state->preFilteredImg1 = cvCreateMat( left0.rows, left0.cols, CV_8U );
        state->cost = cvCreateMat( left0.rows, left0.cols, CV_16S );
    }
    Mat left(left0.size(), CV_8U, state->preFilteredImg0->data.ptr);
    Mat right(right0.size(), CV_8U, state->preFilteredImg1->data.ptr);

    int mindisp = state->minDisparity;
    int ndisp = state->numberOfDisparities;

    int width = left0.cols;
    int height = left0.rows;
    int lofs = max(ndisp - 1 + mindisp, 0);
    int rofs = -min(ndisp - 1 + mindisp, 0);
    int width1 = width - rofs - ndisp + 1;
    int FILTERED = (state->minDisparity - 1) << DISPARITY_SHIFT;

    if( lofs >= width || rofs >= width || width1 < 1 )
    {
        disp0 = Scalar::all( FILTERED * ( disp0.type() < CV_32F ? 1 : 1./(1 << DISPARITY_SHIFT) ) );
        return;
    }

    Mat disp = disp0;

    if( disp0.type() == CV_32F)
    {
        if( !state->disp || state->disp->rows != disp0.rows || state->disp->cols != disp0.cols )
        {
            cvReleaseMat( &state->disp );
            state->disp = cvCreateMat(disp0.rows, disp0.cols, CV_16S);
        }
        disp = cv::cvarrToMat(state->disp);
    }

    int wsz = state->SADWindowSize;
    int bufSize0 = (int)((ndisp + 2)*sizeof(int));
    bufSize0 += (int)((height+wsz+2)*ndisp*sizeof(int));
    bufSize0 += (int)((height + wsz + 2)*sizeof(int));
    bufSize0 += (int)((height+wsz+2)*ndisp*(wsz+2)*sizeof(uchar) + 256);

    int bufSize1 = (int)((width + state->preFilterSize + 2) * sizeof(int) + 256);
    int bufSize2 = 0;
    if( state->speckleRange >= 0 && state->speckleWindowSize > 0 )
        bufSize2 = width*height*(sizeof(cv::Point_<short>) + sizeof(int) + sizeof(uchar));

#if CV_SSE2
    bool useShorts = state->preFilterCap <= 31 && state->SADWindowSize <= 21 && checkHardwareSupport(CV_CPU_SSE2);
#else
    const bool useShorts = false;
#endif

    const double SAD_overhead_coeff = 10.0;
    double N0 = 8000000 / (useShorts ? 1 : 4);  // approx tbb's min number instructions reasonable for one thread
    double maxStripeSize = min(max(N0 / (width * ndisp), (wsz-1) * SAD_overhead_coeff), (double)height);
    int nstripes = cvCeil(height / maxStripeSize);

    int bufSize = max(bufSize0 * nstripes, max(bufSize1 * 2, bufSize2));

    if( !state->slidingSumBuf || state->slidingSumBuf->cols < bufSize )
    {
        cvReleaseMat( &state->slidingSumBuf );
        state->slidingSumBuf = cvCreateMat( 1, bufSize, CV_8U );
    }

    uchar *_buf = state->slidingSumBuf->data.ptr;
    int idx[] = {0,1};
    parallel_do(idx, idx+2, PrefilterInvoker(left0, right0, left, right, _buf, _buf + bufSize1, state));

    Rect validDisparityRect(0, 0, width, height), R1 = state->roi1, R2 = state->roi2;
    validDisparityRect = getValidDisparityROI(R1.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                              R2.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                              state->minDisparity, state->numberOfDisparities,
                                              state->SADWindowSize);

    parallel_for_(Range(0, nstripes),
                  FindStereoCorrespInvoker(left, right, disp, state, nstripes,
                                           bufSize0, useShorts, validDisparityRect));

    if( state->speckleRange >= 0 && state->speckleWindowSize > 0 )
    {
        Mat buf(state->slidingSumBuf);
        filterSpeckles(disp, FILTERED, state->speckleWindowSize, state->speckleRange, buf);
    }

    if (disp0.data != disp.data)
        disp.convertTo(disp0, disp0.type(), 1./(1 << DISPARITY_SHIFT), 0);
}

StereoBM::StereoBM()
{ state = cvCreateStereoBMState(); }

StereoBM::StereoBM(int _preset, int _ndisparities, int _SADWindowSize)
{ init(_preset, _ndisparities, _SADWindowSize); }

void StereoBM::init(int _preset, int _ndisparities, int _SADWindowSize)
{
    state = cvCreateStereoBMState(_preset, _ndisparities);
    state->SADWindowSize = _SADWindowSize;
}

void StereoBM::operator()( InputArray _left, InputArray _right,
                           OutputArray _disparity, int disptype )
{
    Mat left = _left.getMat(), right = _right.getMat();
    CV_Assert( disptype == CV_16S || disptype == CV_32F );
    _disparity.create(left.size(), disptype);
    Mat disparity = _disparity.getMat();

    findStereoCorrespondenceBM(left, right, disparity, state);
}

template<> void Ptr<CvStereoBMState>::delete_obj()
{ cvReleaseStereoBMState(&obj); }

}

CV_IMPL void cvFindStereoCorrespondenceBM( const CvArr* leftarr, const CvArr* rightarr,
                                           CvArr* disparr, CvStereoBMState* state )
{
    cv::Mat left = cv::cvarrToMat(leftarr),
        right = cv::cvarrToMat(rightarr),
        disp = cv::cvarrToMat(disparr);
    cv::findStereoCorrespondenceBM(left, right, disp, state);
}

/* End of file. */
