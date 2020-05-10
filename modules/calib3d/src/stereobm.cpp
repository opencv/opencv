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
#include <vector>
#include "opencl_kernels_calib3d.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

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

    inline bool useShorts() const
    {
        return preFilterCap <= 31 && SADWindowSize <= 21;
    }
    inline bool useFilterSpeckles() const
    {
        return speckleRange >= 0 && speckleWindowSize > 0;
    }
    inline bool useNormPrefilter() const
    {
        return preFilterType == StereoBM::PREFILTER_NORMALIZED_RESPONSE;
    }
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

static void prefilterNorm( const Mat& src, Mat& dst, int winsize, int ftzero, int *buf )
{
    int x, y, wsz2 = winsize/2;
    int* vsum = buf + (wsz2 + 1);
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

#if CV_SIMD
        {
            v_int16 ftz = vx_setall_s16((short) ftzero);
            v_int16 ftz2 = vx_setall_s16((short)(ftzero*2));
            v_int16 z = vx_setzero_s16();

            for(; x <= (size.width - 1) - v_int16::nlanes; x += v_int16::nlanes)
            {
                v_int16 s00 = v_reinterpret_as_s16(vx_load_expand(srow0 + x + 1));
                v_int16 s01 = v_reinterpret_as_s16(vx_load_expand(srow0 + x - 1));
                v_int16 s10 = v_reinterpret_as_s16(vx_load_expand(srow1 + x + 1));
                v_int16 s11 = v_reinterpret_as_s16(vx_load_expand(srow1 + x - 1));
                v_int16 s20 = v_reinterpret_as_s16(vx_load_expand(srow2 + x + 1));
                v_int16 s21 = v_reinterpret_as_s16(vx_load_expand(srow2 + x - 1));
                v_int16 s30 = v_reinterpret_as_s16(vx_load_expand(srow3 + x + 1));
                v_int16 s31 = v_reinterpret_as_s16(vx_load_expand(srow3 + x - 1));

                v_int16 d0 = s00 - s01;
                v_int16 d1 = s10 - s11;
                v_int16 d2 = s20 - s21;
                v_int16 d3 = s30 - s31;

                v_uint16 v0 = v_reinterpret_as_u16(v_max(v_min(d0 + d1 + d1 + d2 + ftz, ftz2), z));
                v_uint16 v1 = v_reinterpret_as_u16(v_max(v_min(d1 + d2 + d2 + d3 + ftz, ftz2), z));

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
#if CV_SIMD
        {
            v_uint8 val0_16 = vx_setall_u8(val0);
            for(; x <= size.width-v_uint8::nlanes; x+=v_uint8::nlanes)
                v_store(dptr + x, val0_16);
        }
#endif
        for(; x < size.width; x++ )
            dptr[x] = val0;
    }
}


static const int DISPARITY_SHIFT_16S = 4;
static const int DISPARITY_SHIFT_32S = 8;

template <typename T>
struct dispShiftTemplate
{ };

template<>
struct dispShiftTemplate<short>
{
    enum { value = DISPARITY_SHIFT_16S };
};

template<>
struct dispShiftTemplate<int>
{
    enum { value = DISPARITY_SHIFT_32S };
};

template <typename T>
inline T dispDescale(int /*v1*/, int /*v2*/, int /*d*/);

template<>
inline short dispDescale(int v1, int v2, int d)
{
    return (short)((v1*256 + (d != 0 ? v2*256/d : 0) + 15) >> 4);
}

template <>
inline int dispDescale(int v1, int v2, int d)
{
    return (int)(v1*256 + (d != 0 ? v2*256/d : 0)); // no need to add 127, this will be converted to float
}


class BufferBM
{
    static const int TABSZ = 256;
public:
    std::vector<int*> sad;
    std::vector<int*> hsad;
    std::vector<int*> htext;
    std::vector<uchar*> cbuf0;
    std::vector<ushort*> sad_short;
    std::vector<ushort*> hsad_short;
    int *prefilter[2];
    uchar tab[TABSZ];
private:
    utils::BufferArea area;

public:
    BufferBM(size_t nstripes, size_t width, size_t height, const StereoBMParams& params)
        : sad(nstripes, NULL),
        hsad(nstripes, NULL),
        htext(nstripes, NULL),
        cbuf0(nstripes, NULL),
        sad_short(nstripes, NULL),
        hsad_short(nstripes, NULL)
    {
        const int wsz = params.SADWindowSize;
        const int ndisp = params.numDisparities;
        const int ftzero = params.preFilterCap;
        for (size_t i = 0; i < nstripes; ++i)
        {
            // 1D: [1][  ndisp  ][1]
#if CV_SIMD
            if (params.useShorts())
                area.allocate(sad_short[i], ndisp + 2);
            else
#endif
                area.allocate(sad[i], ndisp + 2);

            // 2D: [ wsz/2 + 1 ][   height   ][ wsz/2 + 1 ] * [ ndisp ]
#if CV_SIMD
            if (params.useShorts())
                area.allocate(hsad_short[i], (height + wsz + 2) * ndisp);
            else
#endif
                area.allocate(hsad[i], (height + wsz + 2) * ndisp);

            // 1D: [ wsz/2 + 1 ][   height   ][ wsz/2 + 1 ]
            area.allocate(htext[i], (height + wsz + 2));

            // 3D: [ wsz/2 + 1 ][   height   ][ wsz/2 + 1 ] * [ ndisp ] * [ wsz/2 + 1 ][ wsz/2 + 1 ]
            area.allocate(cbuf0[i], ((height + wsz + 2) * ndisp * (wsz + 2) + 256));
        }
        if (params.useNormPrefilter())
        {
            for (size_t i = 0; i < 2; ++i)
                area.allocate(prefilter[0], width + params.preFilterSize + 2);
        }
        area.commit();

        // static table
        for (int x = 0; x < TABSZ; x++)
            tab[x] = (uchar)std::abs(x - ftzero);
    }
};

#if CV_SIMD
template <typename dType>
static void findStereoCorrespondenceBM_SIMD( const Mat& left, const Mat& right,
                                            Mat& disp, Mat& cost, const StereoBMParams& state,
                                            int _dy0, int _dy1, const BufferBM & bufX, size_t bufNum )
{
    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);
    int ndisp = state.numDisparities;
    int mindisp = state.minDisparity;
    int lofs = MAX(ndisp - 1 + mindisp, 0);
    int rofs = -MIN(ndisp - 1 + mindisp, 0);
    int width = left.cols, height = left.rows;
    int width1 = width - rofs - ndisp + 1;
    int textureThreshold = state.textureThreshold;
    int uniquenessRatio = state.uniquenessRatio;
    const int disp_shift = dispShiftTemplate<dType>::value;
    dType FILTERED = (dType)((mindisp - 1) << disp_shift);

    ushort *hsad, *hsad_sub;
    uchar *cbuf;
    const uchar* lptr0 = left.ptr() + lofs;
    const uchar* rptr0 = right.ptr() + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    dType* dptr = disp.ptr<dType>();
    int sstep = (int)left.step;
    int dstep = (int)(disp.step/sizeof(dptr[0]));
    int cstep = (height + dy0 + dy1)*ndisp;
    short costbuf = 0;
    int coststep = cost.data ? (int)(cost.step/sizeof(costbuf)) : 0;
    const uchar * tab = bufX.tab;
    short v_seq[v_int16::nlanes];
    for (short i = 0; i < v_int16::nlanes; ++i)
        v_seq[i] = i;

    ushort *sad = bufX.sad_short[bufNum] + 1;
    ushort *hsad0 = bufX.hsad_short[bufNum] + (wsz2 + 1) * ndisp;
    int *htext = bufX.htext[bufNum] + (wsz2 + 1);
    uchar *cbuf0 = bufX.cbuf0[bufNum] + (wsz2 + 1) * ndisp;

    // initialize buffers
    memset(sad - 1, 0, (ndisp + 2) * sizeof(sad[0]));
    memset(hsad0 - dy0 * ndisp, 0, (height + wsz + 2) * ndisp * sizeof(hsad[0]));
    memset(htext - dy0, 0, (height + wsz + 2) * sizeof(htext[0]));

    for( x = -wsz2-1; x < wsz2; x++ )
    {
        hsad = hsad0 - dy0*ndisp; cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0*ndisp;
        lptr = lptr0 + MIN(MAX(x, -lofs), width-lofs-1) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x, -rofs), width-rofs-ndisp) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            v_uint8 lv = vx_setall_u8((uchar)lval);
            for( d = 0; d <= ndisp - v_uint8::nlanes; d += v_uint8::nlanes )
            {
                v_uint8 diff = v_absdiff(lv, vx_load(rptr + d));
                v_store(cbuf + d, diff);
                v_store(hsad + d, vx_load(hsad + d) + v_expand_low(diff));
                v_store(hsad + d + v_uint16::nlanes, vx_load(hsad + d + v_uint16::nlanes) + v_expand_high(diff));
            }
            if( d <= ndisp - v_uint16::nlanes )
            {
                v_uint8 diff = v_absdiff(lv, vx_load_low(rptr + d));
                v_store_low(cbuf + d, diff);
                v_store(hsad + d, vx_load(hsad + d) + v_expand_low(diff));
                d += v_uint16::nlanes;
            }
            for( ; d < ndisp; d++ )
            {
                int diff = abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] += (ushort)diff;
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
            v_uint8 lv = vx_setall_u8((uchar)lval);
            for( d = 0; d <= ndisp - v_uint8::nlanes; d += v_uint8::nlanes )
            {
                v_uint8 diff = v_absdiff(lv, vx_load(rptr + d));
                v_int8 cbs = v_reinterpret_as_s8(vx_load(cbuf_sub + d));
                v_store(cbuf + d, diff);
                v_store(hsad + d, v_reinterpret_as_u16(v_reinterpret_as_s16(vx_load(hsad + d) + v_expand_low(diff)) - v_expand_low(cbs)));
                v_store(hsad + d + v_uint16::nlanes, v_reinterpret_as_u16(v_reinterpret_as_s16(vx_load(hsad + d + v_uint16::nlanes) + v_expand_high(diff)) - v_expand_high(cbs)));
            }
            if( d <= ndisp - v_uint16::nlanes)
            {
                v_uint8 diff = v_absdiff(lv, vx_load_low(rptr + d));
                v_store_low(cbuf + d, diff);
                v_store(hsad + d, v_reinterpret_as_u16(v_reinterpret_as_s16(vx_load(hsad + d) + v_expand_low(diff)) - vx_load_expand((schar*)cbuf_sub + d)));
                d += v_uint16::nlanes;
            }
            for( ; d < ndisp; d++ )
            {
                int diff = abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] = hsad[d] + (ushort)diff - cbuf_sub[d];
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
        {
            for( d = 0; d <= ndisp-2*v_uint16::nlanes; d += 2*v_uint16::nlanes )
            {
                v_store(sad + d, vx_load(sad + d) + vx_load(hsad + d));
                v_store(sad + d + v_uint16::nlanes, vx_load(sad + d + v_uint16::nlanes) + vx_load(hsad + d + v_uint16::nlanes));
            }
            if( d <= ndisp-v_uint16::nlanes )
            {
                v_store(sad + d, vx_load(sad + d) + vx_load(hsad + d));
                d += v_uint16::nlanes;
            }
            if( d <= ndisp-v_uint16::nlanes/2 )
            {
                v_store_low(sad + d, vx_load_low(sad + d) + vx_load_low(hsad + d));
                d += v_uint16::nlanes/2;
            }
            for( ; d < ndisp; d++ )
                sad[d] = sad[d] + hsad[d];
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
            v_int16 minsad8 = vx_setall_s16(SHRT_MAX);
            v_int16 mind8 = vx_setall_s16(0);

            for( d = 0; d <= ndisp - 2*v_int16::nlanes; d += 2*v_int16::nlanes )
            {
                v_int16 sad8 = v_reinterpret_as_s16(vx_load(hsad + d)) - v_reinterpret_as_s16(vx_load(hsad_sub + d)) + v_reinterpret_as_s16(vx_load(sad + d));
                v_store(sad + d, v_reinterpret_as_u16(sad8));
                mind8 = v_max(mind8, (minsad8 > sad8) & vx_setall_s16((short)d));
                minsad8 = v_min(minsad8, sad8);

                sad8 = v_reinterpret_as_s16(vx_load(hsad + d + v_int16::nlanes)) - v_reinterpret_as_s16(vx_load(hsad_sub + d + v_int16::nlanes)) + v_reinterpret_as_s16(vx_load(sad + d + v_int16::nlanes));
                v_store(sad + d + v_int16::nlanes, v_reinterpret_as_u16(sad8));
                mind8 = v_max(mind8, (minsad8 > sad8) & vx_setall_s16((short)d+v_int16::nlanes));
                minsad8 = v_min(minsad8, sad8);
            }
            if( d <= ndisp - v_int16::nlanes )
            {
                v_int16 sad8 = v_reinterpret_as_s16(vx_load(hsad + d)) - v_reinterpret_as_s16(vx_load(hsad_sub + d)) + v_reinterpret_as_s16(vx_load(sad + d));
                v_store(sad + d, v_reinterpret_as_u16(sad8));
                mind8 = v_max(mind8, (minsad8 > sad8) & vx_setall_s16((short)d));
                minsad8 = v_min(minsad8, sad8);
                d += v_int16::nlanes;
            }
            minsad = v_reduce_min(minsad8);
            v_int16 v_mask = (vx_setall_s16((short)minsad) == minsad8);
            mind = v_reduce_min(((mind8+vx_load(v_seq)) & v_mask) | (vx_setall_s16(SHRT_MAX) & ~v_mask));
            for( ; d < ndisp; d++ )
            {
                int sad8 = (int)(hsad[d]) - hsad_sub[d] + sad[d];
                sad[d] = (ushort)sad8;
                if(minsad > sad8)
                {
                    mind = d;
                    minsad = sad8;
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
                v_int32 thresh4 = vx_setall_s32(thresh + 1);
                v_int32 d1 = vx_setall_s32(mind-1), d2 = vx_setall_s32(mind+1);
                v_int32 dd_4 = vx_setall_s32(v_int32::nlanes);
                v_int32 d4 = vx_load_expand(v_seq);

                for( d = 0; d <= ndisp - v_int16::nlanes; d += v_int16::nlanes )
                {
                    v_int32 sad4_l, sad4_h;
                    v_expand(v_reinterpret_as_s16(vx_load(sad + d)), sad4_l, sad4_h);
                    if( v_check_any((thresh4 > sad4_l) & ((d1 > d4) | (d4 > d2))) )
                        break;
                    d4 += dd_4;
                    if( v_check_any((thresh4 > sad4_h) & ((d1 > d4) | (d4 > d2))) )
                        break;
                    d4 += dd_4;
                }
                if( d <= ndisp - v_int16::nlanes )
                {
                    dptr[y*dstep] = FILTERED;
                    continue;
                }
                if( d <= ndisp - v_int32::nlanes )
                {
                    v_int32 sad4_l = vx_load_expand((short*)sad + d);
                    if (v_check_any((thresh4 > sad4_l) & ((d1 > d4) | (d4 > d2))))
                    {
                        dptr[y*dstep] = FILTERED;
                        continue;
                    }
                    d += v_int16::nlanes;
                }
                for( ; d < ndisp; d++ )
                {
                    if( (thresh + 1) > sad[d] && ((mind - 1) > d || d > (mind + 1)) )
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
                dptr[y*dstep] = dispDescale<dType>(ndisp - mind - 1 + mindisp, p-n, d);
            }
            else
                dptr[y*dstep] = dispDescale<dType>(ndisp - mind - 1 + mindisp, 0, 0);
            costptr[y*coststep] = sad[mind];
        }
    }
}
#endif

template <typename mType>
static void
findStereoCorrespondenceBM( const Mat& left, const Mat& right,
                            Mat& disp, Mat& cost, const StereoBMParams& state,
                            int _dy0, int _dy1, const BufferBM & bufX, size_t bufNum )
{

    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);
    int ndisp = state.numDisparities;
    int mindisp = state.minDisparity;
    int lofs = MAX(ndisp - 1 + mindisp, 0);
    int rofs = -MIN(ndisp - 1 + mindisp, 0);
    int width = left.cols, height = left.rows;
    int width1 = width - rofs - ndisp + 1;
    int textureThreshold = state.textureThreshold;
    int uniquenessRatio = state.uniquenessRatio;
    const int disp_shift = dispShiftTemplate<mType>::value;
    mType FILTERED = (mType)((mindisp - 1) << disp_shift);

    int *hsad, *hsad_sub;
    uchar *cbuf;
    const uchar* lptr0 = left.ptr() + lofs;
    const uchar* rptr0 = right.ptr() + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    mType* dptr = disp.ptr<mType>();
    int sstep = (int)left.step;
    int dstep = (int)(disp.step/sizeof(dptr[0]));
    int cstep = (height+dy0+dy1)*ndisp;
    int costbuf = 0;
    int coststep = cost.data ? (int)(cost.step/sizeof(costbuf)) : 0;
    const uchar * tab = bufX.tab;

#if CV_SIMD
    int v_seq[v_int32::nlanes];
    for (int i = 0; i < v_int32::nlanes; ++i)
        v_seq[i] = i;
    v_int32 d0_4 = vx_load(v_seq), dd_4 = vx_setall_s32(v_int32::nlanes);
#endif

    int *sad = bufX.sad[bufNum] + 1;
    int *hsad0 = bufX.hsad[bufNum] + (wsz2 + 1) * ndisp;
    int *htext = bufX.htext[bufNum] + (wsz2 + 1);
    uchar *cbuf0 = bufX.cbuf0[bufNum] + (wsz2 + 1) * ndisp;

    // initialize buffers
    memset(sad - 1, 0, (ndisp + 2) * sizeof(sad[0]));
    memset(hsad0 - dy0 * ndisp, 0, (height + wsz + 2) * ndisp * sizeof(hsad[0]));
    memset(htext - dy0, 0, (height + wsz + 2) * sizeof(htext[0]));

    for( x = -wsz2-1; x < wsz2; x++ )
    {
        hsad = hsad0 - dy0*ndisp; cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0*ndisp;
        lptr = lptr0 + std::min(std::max(x, -lofs), width-lofs-1) - dy0*sstep;
        rptr = rptr0 + std::min(std::max(x, -rofs), width-rofs-ndisp) - dy0*sstep;
        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            d = 0;
#if CV_SIMD
            {
                v_uint8 lv = vx_setall_u8((uchar)lval);

                for( ; d <= ndisp - v_uint8::nlanes; d += v_uint8::nlanes )
                {
                    v_uint8 rv = vx_load(rptr + d);
                    v_int32 hsad_0 = vx_load(hsad + d);
                    v_int32 hsad_1 = vx_load(hsad + d + v_int32::nlanes);
                    v_int32 hsad_2 = vx_load(hsad + d + 2*v_int32::nlanes);
                    v_int32 hsad_3 = vx_load(hsad + d + 3*v_int32::nlanes);
                    v_uint8 diff = v_absdiff(lv, rv);
                    v_store(cbuf + d, diff);

                    v_uint16 diff0, diff1;
                    v_uint32 diff00, diff01, diff10, diff11;
                    v_expand(diff, diff0, diff1);
                    v_expand(diff0, diff00, diff01);
                    v_expand(diff1, diff10, diff11);

                    hsad_0 += v_reinterpret_as_s32(diff00);
                    hsad_1 += v_reinterpret_as_s32(diff01);
                    hsad_2 += v_reinterpret_as_s32(diff10);
                    hsad_3 += v_reinterpret_as_s32(diff11);

                    v_store(hsad + d, hsad_0);
                    v_store(hsad + d + v_int32::nlanes, hsad_1);
                    v_store(hsad + d + 2*v_int32::nlanes, hsad_2);
                    v_store(hsad + d + 3*v_int32::nlanes, hsad_3);
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
#if CV_SIMD
            {
                v_uint8 lv = vx_setall_u8((uchar)lval);
                for( ; d <= ndisp - v_uint8::nlanes; d += v_uint8::nlanes )
                {
                    v_uint8 rv = vx_load(rptr + d);
                    v_int32 hsad_0 = vx_load(hsad + d);
                    v_int32 hsad_1 = vx_load(hsad + d + v_int32::nlanes);
                    v_int32 hsad_2 = vx_load(hsad + d + 2*v_int32::nlanes);
                    v_int32 hsad_3 = vx_load(hsad + d + 3*v_int32::nlanes);
                    v_uint8 cbs = vx_load(cbuf_sub + d);
                    v_uint8 diff = v_absdiff(lv, rv);
                    v_store(cbuf + d, diff);

                    v_uint16 diff0, diff1, cbs0, cbs1;
                    v_int32 diff00, diff01, diff10, diff11, cbs00, cbs01, cbs10, cbs11;
                    v_expand(diff, diff0, diff1);
                    v_expand(cbs, cbs0, cbs1);
                    v_expand(v_reinterpret_as_s16(diff0), diff00, diff01);
                    v_expand(v_reinterpret_as_s16(diff1), diff10, diff11);
                    v_expand(v_reinterpret_as_s16(cbs0), cbs00, cbs01);
                    v_expand(v_reinterpret_as_s16(cbs1), cbs10, cbs11);

                    v_int32 diff_0 = diff00 - cbs00;
                    v_int32 diff_1 = diff01 - cbs01;
                    v_int32 diff_2 = diff10 - cbs10;
                    v_int32 diff_3 = diff11 - cbs11;
                    hsad_0 += diff_0;
                    hsad_1 += diff_1;
                    hsad_2 += diff_2;
                    hsad_3 += diff_3;

                    v_store(hsad + d, hsad_0);
                    v_store(hsad + d + v_int32::nlanes, hsad_1);
                    v_store(hsad + d + 2*v_int32::nlanes, hsad_2);
                    v_store(hsad + d + 3*v_int32::nlanes, hsad_3);
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
#if CV_SIMD
            {
                for( d = 0; d <= ndisp-2*v_int32::nlanes; d += 2*v_int32::nlanes )
                {
                    v_int32 s0 = vx_load(sad + d);
                    v_int32 s1 = vx_load(sad + d + v_int32::nlanes);
                    v_int32 t0 = vx_load(hsad + d);
                    v_int32 t1 = vx_load(hsad + d + v_int32::nlanes);
                    s0 += t0;
                    s1 += t1;
                    v_store(sad + d, s0);
                    v_store(sad + d + v_int32::nlanes, s1);
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
#if CV_SIMD
            {
                v_int32 minsad4 = vx_setall_s32(INT_MAX);
                v_int32 mind4 = vx_setall_s32(0), d4 = d0_4;

                for( ; d <= ndisp - 2*v_int32::nlanes; d += 2*v_int32::nlanes )
                {
                    v_int32 sad4 = vx_load(sad + d) + vx_load(hsad + d) - vx_load(hsad_sub + d);
                    v_store(sad + d, sad4);
                    mind4 = v_select(minsad4 > sad4, d4, mind4);
                    minsad4 = v_min(minsad4, sad4);
                    d4 += dd_4;

                    sad4 = vx_load(sad + d + v_int32::nlanes) + vx_load(hsad + d + v_int32::nlanes) - vx_load(hsad_sub + d + v_int32::nlanes);
                    v_store(sad + d + v_int32::nlanes, sad4);
                    mind4 = v_select(minsad4 > sad4, d4, mind4);
                    minsad4 = v_min(minsad4, sad4);
                    d4 += dd_4;
                }

                int CV_DECL_ALIGNED(CV_SIMD_WIDTH) minsad_buf[v_int32::nlanes], mind_buf[v_int32::nlanes];
                v_store(minsad_buf, minsad4);
                v_store(mind_buf, mind4);
                for (int i = 0; i < v_int32::nlanes; ++i)
                    if(minsad_buf[i] < minsad || (minsad == minsad_buf[i] && mind_buf[i] < mind)) { minsad = minsad_buf[i]; mind = mind_buf[i]; }
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
                dptr[y*dstep] = dispDescale<mType>(ndisp - mind - 1 + mindisp, p-n, d);

                costptr[y*coststep] = sad[mind];
            }
        }
    }
}

#ifdef HAVE_OPENCL
static bool ocl_prefiltering(InputArray left0, InputArray right0, OutputArray left, OutputArray right, StereoBMParams* state)
{
    if (state->useNormPrefilter())
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
                     const BufferBM &bufX_, const StereoBMParams &state_)
        : bufX(bufX_), state(state_)
    {
        imgs0[0] = &left0; imgs0[1] = &right0;
        imgs[0] = &left; imgs[1] = &right;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        for( int i = range.start; i < range.end; i++ )
        {
            if (state.useNormPrefilter())
                prefilterNorm( *imgs0[i], *imgs[i], state.preFilterSize, state.preFilterCap, bufX.prefilter[i] );
            else
                prefilterXSobel( *imgs0[i], *imgs[i], state.preFilterCap );
        }
    }

    const Mat* imgs0[2];
    Mat* imgs[2];
    const BufferBM &bufX;
    const StereoBMParams &state;
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
                             Mat& _disp, const StereoBMParams &_state,
                             int _nstripes,
                             Rect _validDisparityRect,
                             Mat& _cost, const BufferBM & buf_ )
        : state(_state), buf(buf_)
    {
        CV_Assert( _disp.type() == CV_16S || _disp.type() == CV_32S );
        left = &_left; right = &_right;
        disp = &_disp;
        nstripes = _nstripes;
        validDisparityRect = _validDisparityRect;
        cost = &_cost;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int cols = left->cols, rows = left->rows;
        int _row0 = std::min(cvRound(range.start * rows / nstripes), rows);
        int _row1 = std::min(cvRound(range.end * rows / nstripes), rows);

        int dispShift = disp->type() == CV_16S ? DISPARITY_SHIFT_16S :
                                                 DISPARITY_SHIFT_32S;
        int FILTERED = (state.minDisparity - 1) << dispShift;

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
        Mat cost_i = state.disp12MaxDiff >= 0 ? cost->rowRange(row0, row1) : Mat();

#if CV_SIMD
        if (state.useShorts())
        {
            if( disp_i.type() == CV_16S)
                findStereoCorrespondenceBM_SIMD<short>( left_i, right_i, disp_i, cost_i, state, row0, rows - row1, buf, range.start );
            else
                findStereoCorrespondenceBM_SIMD<int>( left_i, right_i, disp_i, cost_i, state, row0, rows - row1, buf, range.start);
        }
        else
#endif
        {
            if( disp_i.type() == CV_16S )
                findStereoCorrespondenceBM<short>( left_i, right_i, disp_i, cost_i, state, row0, rows - row1, buf, range.start );
            else
                findStereoCorrespondenceBM<int>( left_i, right_i, disp_i, cost_i, state, row0, rows - row1, buf, range.start );
        }

        if( state.disp12MaxDiff >= 0 )
            validateDisparity( disp_i, cost_i, state.minDisparity, state.numDisparities, state.disp12MaxDiff );

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
    Mat* disp, *cost;
    const StereoBMParams &state;

    int nstripes;
    Rect validDisparityRect;
    const BufferBM & buf;
};

class StereoBMImpl CV_FINAL : public StereoBM
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

    void compute( InputArray leftarr, InputArray rightarr, OutputArray disparr ) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

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
            CV_Error( Error::StsOutOfRange, "numDisparities must be positive and divisible by 16" );

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
        if(ocl::isOpenCLActivated() && disparr.isUMat() && params.textureThreshold == 0)
        {
            UMat left, right;
            if(ocl_prefiltering(leftarr, rightarr, left, right, &params))
            {
                if(ocl_stereobm(left, right, disparr, &params))
                {
                    disp_shift = DISPARITY_SHIFT_16S;
                    FILTERED = (params.minDisparity - 1) << disp_shift;

                    if (params.useFilterSpeckles())
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

        {
            const double SAD_overhead_coeff = 10.0;
            const double N0 = 8000000 / (params.useShorts() ? 1 : 4);  // approx tbb's min number instructions reasonable for one thread
            const double maxStripeSize = std::min(
                std::max(
                    N0 / (width * ndisp),
                    (params.SADWindowSize-1) * SAD_overhead_coeff
                ),
                (double)height
            );
            const int nstripes = cvCeil(height / maxStripeSize);
            BufferBM localBuf(nstripes, width, height, params);

            // Prefiltering
            parallel_for_(Range(0, 2), PrefilterInvoker(left0, right0, left, right, localBuf, params), 1);


            Rect validDisparityRect(0, 0, width, height), R1 = params.roi1, R2 = params.roi2;
            validDisparityRect = getValidDisparityROI(!R1.empty() ? R1 : validDisparityRect,
                                                      !R2.empty() ? R2 : validDisparityRect,
                                                      params.minDisparity, params.numDisparities,
                                                      params.SADWindowSize);

            FindStereoCorrespInvoker invoker(left, right, disp, params, nstripes, validDisparityRect, cost, localBuf);
            parallel_for_(Range(0, nstripes), invoker);

            if (params.useFilterSpeckles())
            {
                slidingSumBuf.create( 1, width * height * (sizeof(Point_<short>) + sizeof(int) + sizeof(uchar)), CV_8U );
                filterSpeckles(disp, FILTERED, params.speckleWindowSize, params.speckleRange, slidingSumBuf);
            }

        }

        if (disp0.data != disp.data)
            disp.convertTo(disp0, disp0.type(), 1./(1 << disp_shift), 0);
    }

    int getMinDisparity() const CV_OVERRIDE { return params.minDisparity; }
    void setMinDisparity(int minDisparity) CV_OVERRIDE { params.minDisparity = minDisparity; }

    int getNumDisparities() const CV_OVERRIDE { return params.numDisparities; }
    void setNumDisparities(int numDisparities) CV_OVERRIDE { params.numDisparities = numDisparities; }

    int getBlockSize() const CV_OVERRIDE { return params.SADWindowSize; }
    void setBlockSize(int blockSize) CV_OVERRIDE { params.SADWindowSize = blockSize; }

    int getSpeckleWindowSize() const CV_OVERRIDE { return params.speckleWindowSize; }
    void setSpeckleWindowSize(int speckleWindowSize) CV_OVERRIDE { params.speckleWindowSize = speckleWindowSize; }

    int getSpeckleRange() const CV_OVERRIDE { return params.speckleRange; }
    void setSpeckleRange(int speckleRange) CV_OVERRIDE { params.speckleRange = speckleRange; }

    int getDisp12MaxDiff() const CV_OVERRIDE { return params.disp12MaxDiff; }
    void setDisp12MaxDiff(int disp12MaxDiff) CV_OVERRIDE { params.disp12MaxDiff = disp12MaxDiff; }

    int getPreFilterType() const CV_OVERRIDE { return params.preFilterType; }
    void setPreFilterType(int preFilterType) CV_OVERRIDE { params.preFilterType = preFilterType; }

    int getPreFilterSize() const CV_OVERRIDE { return params.preFilterSize; }
    void setPreFilterSize(int preFilterSize) CV_OVERRIDE { params.preFilterSize = preFilterSize; }

    int getPreFilterCap() const CV_OVERRIDE { return params.preFilterCap; }
    void setPreFilterCap(int preFilterCap) CV_OVERRIDE { params.preFilterCap = preFilterCap; }

    int getTextureThreshold() const CV_OVERRIDE { return params.textureThreshold; }
    void setTextureThreshold(int textureThreshold) CV_OVERRIDE { params.textureThreshold = textureThreshold; }

    int getUniquenessRatio() const CV_OVERRIDE { return params.uniquenessRatio; }
    void setUniquenessRatio(int uniquenessRatio) CV_OVERRIDE { params.uniquenessRatio = uniquenessRatio; }

    int getSmallerBlockSize() const CV_OVERRIDE { return 0; }
    void setSmallerBlockSize(int) CV_OVERRIDE {}

    Rect getROI1() const CV_OVERRIDE { return params.roi1; }
    void setROI1(Rect roi1) CV_OVERRIDE { params.roi1 = roi1; }

    Rect getROI2() const CV_OVERRIDE { return params.roi2; }
    void setROI2(Rect roi2) CV_OVERRIDE { params.roi2 = roi2; }

    void write(FileStorage& fs) const CV_OVERRIDE
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

    void read(const FileNode& fn) CV_OVERRIDE
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
