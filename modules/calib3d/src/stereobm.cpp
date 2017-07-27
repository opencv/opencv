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
#include "opencv2/core/hal/intrin.hpp"

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

#ifdef HAVE_OPENCL
static bool ocl_prefilter_norm(InputArray _input, OutputArray _output, int winsize, int prefilterCap)
{
    ocl::Kernel k("prefilter_norm", ocl::calib3d::stereobm_oclsrc, cv::format("-D WSZ=%d", winsize));
    if(k.empty())
        return false;

    int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
    scale_g *= scale_s;

    UMat input = _input.getUMat(), output;
    _output.create(input.size(), input.type());
    output = _output.getUMat();

    size_t globalThreads[3] = { (size_t)input.cols, (size_t)input.rows, 1 };

    k.args(ocl::KernelArg::PtrReadOnly(input), ocl::KernelArg::PtrWriteOnly(output), input.rows, input.cols,
        prefilterCap, scale_g, scale_s);

    return k.run(2, globalThreads, NULL, false);
}
#endif

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

#ifdef HAVE_OPENCL
static bool ocl_prefilter_xsobel(InputArray _input, OutputArray _output, int prefilterCap)
{
    ocl::Kernel k("prefilter_xsobel", ocl::calib3d::stereobm_oclsrc);
    if(k.empty())
        return false;

    UMat input = _input.getUMat(), output;
    _output.create(input.size(), input.type());
    output = _output.getUMat();

    size_t globalThreads[3] = { (size_t)input.cols, (size_t)input.rows, 1 };

    k.args(ocl::KernelArg::PtrReadOnly(input), ocl::KernelArg::PtrWriteOnly(output), input.rows, input.cols, prefilterCap);

    return k.run(2, globalThreads, NULL, false);
}
#endif

static void
prefilterXSobel( const Mat& src, Mat& dst, int ftzero )
{
    int x, y;
    const int OFS = 256*4, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ] = { 0 };
    Size size = src.size();

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero*2 : x - OFS + ftzero);
    uchar val0 = tab[0 + OFS];

#if CV_SIMD128
    bool useSIMD = hasSIMD128();
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

#if CV_SIMD128
        if( useSIMD )
        {
            v_int16x8 ftz = v_setall_s16((short) ftzero);
            v_int16x8 ftz2 = v_setall_s16((short)(ftzero*2));
            v_int16x8 z = v_setzero_s16();

            for(; x <= (size.width - 1) - 8; x += 8 )
            {
                v_int16x8 s00 = v_reinterpret_as_s16(v_load_expand(srow0 + x + 1));
                v_int16x8 s01 = v_reinterpret_as_s16(v_load_expand(srow0 + x - 1));
                v_int16x8 s10 = v_reinterpret_as_s16(v_load_expand(srow1 + x + 1));
                v_int16x8 s11 = v_reinterpret_as_s16(v_load_expand(srow1 + x - 1));
                v_int16x8 s20 = v_reinterpret_as_s16(v_load_expand(srow2 + x + 1));
                v_int16x8 s21 = v_reinterpret_as_s16(v_load_expand(srow2 + x - 1));
                v_int16x8 s30 = v_reinterpret_as_s16(v_load_expand(srow3 + x + 1));
                v_int16x8 s31 = v_reinterpret_as_s16(v_load_expand(srow3 + x - 1));

                v_int16x8 d0 = s00 - s01;
                v_int16x8 d1 = s10 - s11;
                v_int16x8 d2 = s20 - s21;
                v_int16x8 d3 = s30 - s31;

                v_uint16x8 v0 = v_reinterpret_as_u16(v_max(v_min(d0 + d1 + d1 + d2 + ftz, ftz2), z));
                v_uint16x8 v1 = v_reinterpret_as_u16(v_max(v_min(d1 + d2 + d2 + d3 + ftz, ftz2), z));

                v_pack_store(dptr0 + x, v0);
                v_pack_store(dptr1 + x, v1);
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
        x = 0;
#if CV_SIMD128
        if( useSIMD )
        {
            v_uint8x16 val0_16 = v_setall_u8(val0);
            for(; x <= size.width-16; x+=16 )
                v_store(dptr + x, val0_16);
        }
#endif
        for(; x < size.width; x++ )
            dptr[x] = val0;
    }
}


static const int DISPARITY_SHIFT_16S = 4;
static const int DISPARITY_SHIFT_32S = 8;

#if CV_SIMD128
static void findStereoCorrespondenceBM_SIMD( const Mat& left, const Mat& right,
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
    short FILTERED = (short)((mindisp - 1) << DISPARITY_SHIFT_16S);

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
    const v_int16x8 d0_8 = v_int16x8(0,1,2,3,4,5,6,7), dd_8 = v_setall_s16(8);

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
        rptr = rptr0 + MIN(MAX(x, -rofs), width-rofs-ndisp) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            v_uint8x16 lv = v_setall_u8((uchar)lval);
            for( d = 0; d < ndisp; d += 16 )
            {
                v_uint8x16 rv = v_load(rptr + d);
                v_uint16x8 hsad_l = v_load(hsad + d);
                v_uint16x8 hsad_h = v_load(hsad + d + 8);
                v_uint8x16 diff = v_absdiff(lv, rv);
                v_store(cbuf + d, diff);
                v_uint16x8 diff0, diff1;
                v_expand(diff, diff0, diff1);
                hsad_l += diff0;
                hsad_h += diff1;
                v_store(hsad + d, hsad_l);
                v_store(hsad + d + 8, hsad_h);
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
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-ndisp-rofs) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp,
            hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            v_uint8x16 lv = v_setall_u8((uchar)lval);
            for( d = 0; d < ndisp; d += 16 )
            {
                v_uint8x16 rv = v_load(rptr + d);
                v_uint16x8 hsad_l = v_load(hsad + d);
                v_uint16x8 hsad_h = v_load(hsad + d + 8);
                v_uint8x16 cbs = v_load(cbuf_sub + d);
                v_uint8x16 diff = v_absdiff(lv, rv);
                v_int16x8 diff_l, diff_h, cbs_l, cbs_h;
                v_store(cbuf + d, diff);
                v_expand(v_reinterpret_as_s8(diff), diff_l, diff_h);
                v_expand(v_reinterpret_as_s8(cbs), cbs_l, cbs_h);
                diff_l -= cbs_l;
                diff_h -= cbs_h;
                hsad_h = v_reinterpret_as_u16(v_reinterpret_as_s16(hsad_h) + diff_h);
                hsad_l = v_reinterpret_as_u16(v_reinterpret_as_s16(hsad_l) + diff_l);
                v_store(hsad + d, hsad_l);
                v_store(hsad + d + 8, hsad_h);
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
            for( d = 0; d <= ndisp-16; d += 16 )
            {
                v_uint16x8 s0 = v_load(sad + d);
                v_uint16x8 s1 = v_load(sad + d + 8);
                v_uint16x8 t0 = v_load(hsad + d);
                v_uint16x8 t1 = v_load(hsad + d + 8);
                s0 = s0 + t0;
                s1 = s1 + t1;
                v_store(sad + d, s0);
                v_store(sad + d + 8, s1);
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
            v_int16x8 minsad8 = v_setall_s16(SHRT_MAX);
            v_int16x8 mind8 = v_setall_s16(0), d8 = d0_8;

            for( d = 0; d < ndisp; d += 16 )
            {
                v_int16x8 u0 = v_reinterpret_as_s16(v_load(hsad_sub + d));
                v_int16x8 u1 = v_reinterpret_as_s16(v_load(hsad + d));

                v_int16x8 v0 = v_reinterpret_as_s16(v_load(hsad_sub + d + 8));
                v_int16x8 v1 = v_reinterpret_as_s16(v_load(hsad + d + 8));

                v_int16x8 usad8 = v_reinterpret_as_s16(v_load(sad + d));
                v_int16x8 vsad8 = v_reinterpret_as_s16(v_load(sad + d + 8));

                u1 -= u0;
                v1 -= v0;
                usad8 += u1;
                vsad8 += v1;

                v_int16x8 mask = minsad8 > usad8;
                minsad8 = v_min(minsad8, usad8);
                mind8 = v_max(mind8, (mask& d8));

                v_store(sad + d, v_reinterpret_as_u16(usad8));
                v_store(sad + d + 8, v_reinterpret_as_u16(vsad8));

                mask = minsad8 > vsad8;
                minsad8 = v_min(minsad8, vsad8);

                d8 = d8 + dd_8;
                mind8 = v_max(mind8, (mask & d8));
                d8 = d8 + dd_8;
            }

            tsum += htext[y + wsz2] - htext[y - wsz2 - 1];
            if( tsum < textureThreshold )
            {
                dptr[y*dstep] = FILTERED;
                continue;
            }

            ushort CV_DECL_ALIGNED(16) minsad_buf[8], mind_buf[8];
            v_store(minsad_buf, v_reinterpret_as_u16(minsad8));
            v_store(mind_buf, v_reinterpret_as_u16(mind8));
            for( d = 0; d < 8; d++ )
                if(minsad > (int)minsad_buf[d] || (minsad == (int)minsad_buf[d] && mind > mind_buf[d]))
                {
                    minsad = minsad_buf[d];
                    mind = mind_buf[d];
                }

            if( uniquenessRatio > 0 )
            {
                int thresh = minsad + (minsad * uniquenessRatio/100);
                v_int32x4 thresh4 = v_setall_s32(thresh + 1);
                v_int32x4 d1 = v_setall_s32(mind-1), d2 = v_setall_s32(mind+1);
                v_int32x4 dd_4 = v_setall_s32(4);
                v_int32x4 d4 = v_int32x4(0,1,2,3);
                v_int32x4 mask4;

                for( d = 0; d < ndisp; d += 8 )
                {
                    v_int16x8 sad8 = v_reinterpret_as_s16(v_load(sad + d));
                    v_int32x4 sad4_l, sad4_h;
                    v_expand(sad8, sad4_l, sad4_h);
                    mask4 = thresh4 > sad4_l;
                    mask4 = mask4 & ((d1 > d4) | (d4 > d2));
                    if( v_signmask(mask4) )
                        break;
                    d4 += dd_4;
                    mask4 = thresh4 > sad4_h;
                    mask4 = mask4 & ((d1 > d4) | (d4 > d2));
                    if( v_signmask(mask4) )
                        break;
                    d4 += dd_4;
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

template <typename mType>
static void
findStereoCorrespondenceBM( const Mat& left, const Mat& right,
                           Mat& disp, Mat& cost, const StereoBMParams& state,
                           uchar* buf, int _dy0, int _dy1, const int disp_shift )
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
    mType FILTERED = (mType)((mindisp - 1) << disp_shift);

#if CV_SIMD128
    bool useSIMD = hasSIMD128();
    if( useSIMD )
    {
        CV_Assert (ndisp % 8 == 0);
    }
#endif

    int *sad, *hsad0, *hsad, *hsad_sub, *htext;
    uchar *cbuf0, *cbuf;
    const uchar* lptr0 = left.ptr() + lofs;
    const uchar* rptr0 = right.ptr() + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    mType* dptr = disp.ptr<mType>();
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
            d = 0;
#if CV_SIMD128
            if( useSIMD )
            {
                v_uint8x16 lv = v_setall_u8((uchar)lval);

                for( ; d <= ndisp - 16; d += 16 )
                {
                    v_uint8x16 rv = v_load(rptr + d);
                    v_int32x4 hsad_0 = v_load(hsad + d);
                    v_int32x4 hsad_1 = v_load(hsad + d + 4);
                    v_int32x4 hsad_2 = v_load(hsad + d + 8);
                    v_int32x4 hsad_3 = v_load(hsad + d + 12);
                    v_uint8x16 diff = v_absdiff(lv, rv);
                    v_store(cbuf + d, diff);

                    v_uint16x8 diff0, diff1;
                    v_uint32x4 diff00, diff01, diff10, diff11;
                    v_expand(diff, diff0, diff1);
                    v_expand(diff0, diff00, diff01);
                    v_expand(diff1, diff10, diff11);

                    hsad_0 += v_reinterpret_as_s32(diff00);
                    hsad_1 += v_reinterpret_as_s32(diff01);
                    hsad_2 += v_reinterpret_as_s32(diff10);
                    hsad_3 += v_reinterpret_as_s32(diff11);

                    v_store(hsad + d, hsad_0);
                    v_store(hsad + d + 4, hsad_1);
                    v_store(hsad + d + 8, hsad_2);
                    v_store(hsad + d + 12, hsad_3);
                }
            }
#endif
            for( ; d < ndisp; d++ )
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
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-ndisp-rofs) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp,
            hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            d = 0;
#if CV_SIMD128
            if( useSIMD )
            {
                v_uint8x16 lv = v_setall_u8((uchar)lval);
                for( ; d <= ndisp - 16; d += 16 )
                {
                    v_uint8x16 rv = v_load(rptr + d);
                    v_int32x4 hsad_0 = v_load(hsad + d);
                    v_int32x4 hsad_1 = v_load(hsad + d + 4);
                    v_int32x4 hsad_2 = v_load(hsad + d + 8);
                    v_int32x4 hsad_3 = v_load(hsad + d + 12);
                    v_uint8x16 cbs = v_load(cbuf_sub + d);
                    v_uint8x16 diff = v_absdiff(lv, rv);
                    v_store(cbuf + d, diff);

                    v_uint16x8 diff0, diff1, cbs0, cbs1;
                    v_int32x4 diff00, diff01, diff10, diff11, cbs00, cbs01, cbs10, cbs11;
                    v_expand(diff, diff0, diff1);
                    v_expand(cbs, cbs0, cbs1);
                    v_expand(v_reinterpret_as_s16(diff0), diff00, diff01);
                    v_expand(v_reinterpret_as_s16(diff1), diff10, diff11);
                    v_expand(v_reinterpret_as_s16(cbs0), cbs00, cbs01);
                    v_expand(v_reinterpret_as_s16(cbs1), cbs10, cbs11);

                    v_int32x4 diff_0 = diff00 - cbs00;
                    v_int32x4 diff_1 = diff01 - cbs01;
                    v_int32x4 diff_2 = diff10 - cbs10;
                    v_int32x4 diff_3 = diff11 - cbs11;
                    hsad_0 += diff_0;
                    hsad_1 += diff_1;
                    hsad_2 += diff_2;
                    hsad_3 += diff_3;

                    v_store(hsad + d, hsad_0);
                    v_store(hsad + d + 4, hsad_1);
                    v_store(hsad + d + 8, hsad_2);
                    v_store(hsad + d + 12, hsad_3);
                }
            }
#endif
            for( ; d < ndisp; d++ )
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
        {
            d = 0;
#if CV_SIMD128
            if( useSIMD )
            {
                for( d = 0; d <= ndisp-8; d += 8 )
                {
                    v_int32x4 s0 = v_load(sad + d);
                    v_int32x4 s1 = v_load(sad + d + 4);
                    v_int32x4 t0 = v_load(hsad + d);
                    v_int32x4 t1 = v_load(hsad + d + 4);
                    s0 += t0;
                    s1 += t1;
                    v_store(sad + d, s0);
                    v_store(sad + d + 4, s1);
                }
            }
#endif
            for( ; d < ndisp; d++ )
                sad[d] = (int)(sad[d] + hsad[d]);
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
            d = 0;
#if CV_SIMD128
            if( useSIMD )
            {
                v_int32x4 d0_4 = v_int32x4(0, 1, 2, 3);
                v_int32x4 dd_4 = v_setall_s32(4);
                v_int32x4 minsad4 = v_setall_s32(INT_MAX);
                v_int32x4 mind4 = v_setall_s32(0), d4 = d0_4;

                for( ; d <= ndisp - 8; d += 8 )
                {
                    v_int32x4 u0 = v_load(hsad_sub + d);
                    v_int32x4 u1 = v_load(hsad + d);

                    v_int32x4 v0 = v_load(hsad_sub + d + 4);
                    v_int32x4 v1 = v_load(hsad + d + 4);

                    v_int32x4 usad4 = v_load(sad + d);
                    v_int32x4 vsad4 = v_load(sad + d + 4);

                    u1 -= u0;
                    v1 -= v0;
                    usad4 += u1;
                    vsad4 += v1;

                    v_store(sad + d, usad4);
                    v_store(sad + d + 4, vsad4);

                    v_int32x4 mask = minsad4 > usad4;
                    minsad4 = v_min(minsad4, usad4);
                    mind4 = v_select(mask, d4, mind4);
                    d4 += dd_4;

                    mask = minsad4 > vsad4;
                    minsad4 = v_min(minsad4, vsad4);
                    mind4 = v_select(mask, d4, mind4);
                    d4 += dd_4;
                }

                int CV_DECL_ALIGNED(16) minsad_buf[4], mind_buf[4];
                v_store(minsad_buf, minsad4);
                v_store(mind_buf, mind4);
                if(minsad_buf[0] < minsad || (minsad == minsad_buf[0] && mind_buf[0] < mind)) { minsad = minsad_buf[0]; mind = mind_buf[0]; }
                if(minsad_buf[1] < minsad || (minsad == minsad_buf[1] && mind_buf[1] < mind)) { minsad = minsad_buf[1]; mind = mind_buf[1]; }
                if(minsad_buf[2] < minsad || (minsad == minsad_buf[2] && mind_buf[2] < mind)) { minsad = minsad_buf[2]; mind = mind_buf[2]; }
                if(minsad_buf[3] < minsad || (minsad == minsad_buf[3] && mind_buf[3] < mind)) { minsad = minsad_buf[3]; mind = mind_buf[3]; }
            }
#endif
            for( ; d < ndisp; d++ )
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
                    if( (d < mind-1 || d > mind+1) && sad[d] <= thresh)
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
                dptr[y*dstep] = (mType)(((ndisp - mind - 1 + mindisp)*256 + (d != 0 ? (p-n)*256/d : 0) + 15)
                                 >> (DISPARITY_SHIFT_32S - disp_shift));
                costptr[y*coststep] = sad[mind];
            }
        }
    }
}

#ifdef HAVE_OPENCL
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
#endif

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

#ifdef HAVE_OPENCL
static bool ocl_stereobm( InputArray _left, InputArray _right,
                       OutputArray _disp, StereoBMParams* state)
{
    int ndisp = state->numDisparities;
    int mindisp = state->minDisparity;
    int wsz = state->SADWindowSize;
    int wsz2 = wsz/2;

    ocl::Device devDef = ocl::Device::getDefault();
    int sizeX = devDef.isIntel() ? 32 : std::max(11, 27 - devDef.maxComputeUnits()),
        sizeY = sizeX - 1,
        N = ndisp * 2;

    cv::String opt = cv::format("-D DEFINE_KERNEL_STEREOBM -D MIN_DISP=%d -D NUM_DISP=%d"
                                " -D BLOCK_SIZE_X=%d -D BLOCK_SIZE_Y=%d -D WSZ=%d",
                                mindisp, ndisp,
                                sizeX, sizeY, wsz);
    ocl::Kernel k("stereoBM", ocl::calib3d::stereobm_oclsrc, opt);
    if(k.empty())
        return false;

    UMat left = _left.getUMat(), right = _right.getUMat();
    int cols = left.cols, rows = left.rows;

    _disp.create(_left.size(), CV_16S);
    _disp.setTo((mindisp - 1) << 4);
    Rect roi = Rect(Point(wsz2 + mindisp + ndisp - 1, wsz2), Point(cols-wsz2-mindisp, rows-wsz2) );
    UMat disp = (_disp.getUMat())(roi);

    int globalX = (disp.cols + sizeX - 1) / sizeX,
        globalY = (disp.rows + sizeY - 1) / sizeY;
    size_t globalThreads[3] = {(size_t)N, (size_t)globalX, (size_t)globalY};
    size_t localThreads[3]  = {(size_t)N, 1, 1};

    int idx = 0;
    idx = k.set(idx, ocl::KernelArg::PtrReadOnly(left));
    idx = k.set(idx, ocl::KernelArg::PtrReadOnly(right));
    idx = k.set(idx, ocl::KernelArg::WriteOnlyNoSize(disp));
    idx = k.set(idx, rows);
    idx = k.set(idx, cols);
    idx = k.set(idx, state->textureThreshold);
    idx = k.set(idx, state->uniquenessRatio);
    return k.run(3, globalThreads, localThreads, false);
}
#endif

struct FindStereoCorrespInvoker : public ParallelLoopBody
{
    FindStereoCorrespInvoker( const Mat& _left, const Mat& _right,
                             Mat& _disp, StereoBMParams* _state,
                             int _nstripes, size_t _stripeBufSize,
                             bool _useShorts, Rect _validDisparityRect,
                             Mat& _slidingSumBuf, Mat& _cost )
    {
        CV_Assert( _disp.type() == CV_16S || _disp.type() == CV_32S );
        left = &_left; right = &_right;
        disp = &_disp; state = _state;
        nstripes = _nstripes; stripeBufSize = _stripeBufSize;
        useShorts = _useShorts;
        validDisparityRect = _validDisparityRect;
        slidingSumBuf = &_slidingSumBuf;
        cost = &_cost;
#if CV_SIMD128
        useSIMD = hasSIMD128();
#endif
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

#if CV_SIMD128
        if( useSIMD && useShorts )
        {
            findStereoCorrespondenceBM_SIMD( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1 );
        }
        else
#endif
        {
            if( disp_i.type() == CV_16S )
                findStereoCorrespondenceBM<short>( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1, DISPARITY_SHIFT_16S );
            else
                findStereoCorrespondenceBM<int>( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1, DISPARITY_SHIFT_32S );
        }

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
    bool useSIMD;
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
        CV_INSTRUMENT_REGION()

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

        int disp_shift;
        if (dtype == CV_16SC1)
            disp_shift = DISPARITY_SHIFT_16S;
        else
            disp_shift = DISPARITY_SHIFT_32S;


        int FILTERED = (params.minDisparity - 1) << disp_shift;

#ifdef HAVE_OPENCL
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
                        disparr.getUMat().convertTo(disparr, CV_32FC1, 1./(1 << disp_shift), 0);
                    CV_IMPL_ADD(CV_IMPL_OCL);
                    return;
                }
            }
        }
#endif

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
            disp0 = Scalar::all( FILTERED * ( disp0.type() < CV_32F ? 1 : 1./(1 << disp_shift) ) );
            return;
        }

        Mat disp = disp0;
        if( dtype == CV_32F )
        {
            dispbuf.create(disp0.size(), CV_32S);
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

        bool useShorts = params.preFilterCap <= 31 && params.SADWindowSize <= 21;
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
        validDisparityRect = getValidDisparityROI(R1.area() > 0 ? R1 : validDisparityRect,
                                                  R2.area() > 0 ? R2 : validDisparityRect,
                                                  params.minDisparity, params.numDisparities,
                                                  params.SADWindowSize);

        parallel_for_(Range(0, nstripes),
                      FindStereoCorrespInvoker(left, right, disp, &params, nstripes,
                                               bufSize0, useShorts, validDisparityRect,
                                               slidingSumBuf, cost));

        if( params.speckleRange >= 0 && params.speckleWindowSize > 0 )
            filterSpeckles(disp, FILTERED, params.speckleWindowSize, params.speckleRange, slidingSumBuf);

        if (disp0.data != disp.data)
            disp.convertTo(disp0, disp0.type(), 1./(1 << disp_shift), 0);
    }

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
        writeFormat(fs);
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

Ptr<StereoBM> StereoBM::create(int _numDisparities, int _SADWindowSize)
{
    return makePtr<StereoBMImpl>(_numDisparities, _SADWindowSize);
}

}

/* End of file. */
