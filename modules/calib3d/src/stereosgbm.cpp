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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

/*
 This is a variation of
 "Stereo Processing by Semiglobal Matching and Mutual Information"
 by Heiko Hirschmuller.

 We match blocks rather than individual pixels, thus the algorithm is called
 SGBM (Semi-global block matching)
 */

#include "precomp.hpp"
#include <limits.h>
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

namespace cv
{

typedef uchar PixType;
typedef short CostType;
typedef short DispType;

// NR - the number of directions. the loop on x that computes Lr assumes that NR == 8.
// if you change NR, please, modify the loop as well.
enum { NR = 8, NR2 = NR/2 };


struct StereoSGBMParams
{
    StereoSGBMParams()
    {
        minDisparity = numDisparities = 0;
        SADWindowSize = 0;
        P1 = P2 = 0;
        disp12MaxDiff = 0;
        preFilterCap = 0;
        uniquenessRatio = 0;
        speckleWindowSize = 0;
        speckleRange = 0;
        mode = StereoSGBM::MODE_SGBM;
    }

    StereoSGBMParams( int _minDisparity, int _numDisparities, int _SADWindowSize,
                      int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                      int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                      int _mode )
    {
        minDisparity = _minDisparity;
        numDisparities = _numDisparities;
        SADWindowSize = _SADWindowSize;
        P1 = _P1;
        P2 = _P2;
        disp12MaxDiff = _disp12MaxDiff;
        preFilterCap = _preFilterCap;
        uniquenessRatio = _uniquenessRatio;
        speckleWindowSize = _speckleWindowSize;
        speckleRange = _speckleRange;
        mode = _mode;
    }

    inline bool isFullDP() const
    {
        return mode == StereoSGBM::MODE_HH || mode == StereoSGBM::MODE_HH4;
    }
    inline Size calcSADWindowSize() const
    {
        const int dim = SADWindowSize > 0 ? SADWindowSize : 5;
        return Size(dim, dim);
    }

    int minDisparity;
    int numDisparities;
    int SADWindowSize;
    int preFilterCap;
    int uniquenessRatio;
    int P1;
    int P2;
    int speckleWindowSize;
    int speckleRange;
    int disp12MaxDiff;
    int mode;
};

#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_SIMD_WIDTH == 16
static inline v_int16 vx_setseq_s16()
{ return v_int16(0, 1, 2, 3, 4, 5, 6, 7); }
#elif CV_SIMD_WIDTH == 32
static inline v_int16 vx_setseq_s16()
{ return v_int16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }
#elif CV_SIMD_WIDTH == 64
static inline v_int16 vx_setseq_s16()
{ return v_int16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31); }
#else
struct vseq_s16
{
    short data[VTraits<v_int16>::max_nlanes];
    vseq_s16()
    {
        for (int i = 0; i < VTraits<v_int16>::vlanes(); i++)
            data[i] = i;
    }
};
static inline v_int16 vx_setseq_s16()
{
    static vseq_s16 vseq;
    return vx_load(vseq.data);
}
#endif
// define some additional reduce operations:
static inline void min_pos(const v_int16& val, const v_int16& pos, short &min_val, short &min_pos)
{
    min_val = v_reduce_min(val);
    v_int16 v_mask = (v_eq(vx_setall_s16(min_val), val));
    min_pos = v_reduce_min(v_or(v_and(v_add(pos, vx_setseq_s16()), v_mask), v_and(vx_setall_s16(SHRT_MAX), v_not(v_mask))));
}
#endif

static const int DEFAULT_RIGHT_BORDER = -1;

/*
 For each pixel row1[x], max(maxD, 0) <= minX <= x < maxX <= width - max(0, -minD),
 and for each disparity minD<=d<maxD the function
 computes the cost (cost[(x-minX)*(maxD - minD) + (d - minD)]), depending on the difference between
 row1[x] and row2[x-d]. The subpixel algorithm from
 "Depth Discontinuities by Pixel-to-Pixel Stereo" by Stan Birchfield and C. Tomasi
 is used, hence the suffix BT.

 the temporary buffer should contain width2*2 elements
 */
static void calcPixelCostBT( const Mat& img1, const Mat& img2, int y,
                            int minD, int maxD, CostType* cost,
                            PixType* buffer, const PixType* tab,
                            int xrange_min = 0, int xrange_max = DEFAULT_RIGHT_BORDER )
{
    int x, c, width = img1.cols, cn = img1.channels();
    int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
    int D = (int)alignSize(maxD - minD, VTraits<v_int16>::vlanes()), width1 = maxX1 - minX1;
    //This minX1 & maxX2 correction is defining which part of calculatable line must be calculated
    //That is needs of parallel algorithm
    xrange_min = (xrange_min < 0) ? 0: xrange_min;
    xrange_max = (xrange_max == DEFAULT_RIGHT_BORDER) || (xrange_max > width1) ? width1 : xrange_max;
    maxX1 = minX1 + xrange_max;
    minX1 += xrange_min;
    width1 = maxX1 - minX1;
    int minX2 = std::max(minX1 - maxD, 0), maxX2 = std::min(maxX1 - minD, width);
    int width2 = maxX2 - minX2;
    const PixType *row1 = img1.ptr<PixType>(y), *row2 = img2.ptr<PixType>(y);
    PixType *prow1 = buffer + width2*2, *prow2 = prow1 + width*cn*2;

    for( c = 0; c < cn*2; c++ )
    {
        prow1[width*c] = prow1[width*c + width-1] =
        prow2[width*c] = prow2[width*c + width-1] = tab[0];
    }

    int n1 = y > 0 ? -(int)img1.step : 0, s1 = y < img1.rows-1 ? (int)img1.step : 0;
    int n2 = y > 0 ? -(int)img2.step : 0, s2 = y < img2.rows-1 ? (int)img2.step : 0;

    int minX_cmn = std::min(minX1,minX2)-1;
    int maxX_cmn = std::max(maxX1,maxX2)+1;
    minX_cmn = std::max(minX_cmn, 1);
    maxX_cmn = std::min(maxX_cmn, width - 1);
    if( cn == 1 )
    {
        for( x = minX_cmn; x < maxX_cmn; x++ )
        {
            prow1[x] = tab[(row1[x+1] - row1[x-1])*2 + row1[x+n1+1] - row1[x+n1-1] + row1[x+s1+1] - row1[x+s1-1]];
            prow2[width-1-x] = tab[(row2[x+1] - row2[x-1])*2 + row2[x+n2+1] - row2[x+n2-1] + row2[x+s2+1] - row2[x+s2-1]];

            prow1[x+width] = row1[x];
            prow2[width-1-x+width] = row2[x];
        }
    }
    else
    {
        for( x = minX_cmn; x < maxX_cmn; x++ )
        {
            prow1[x] = tab[(row1[x*3+3] - row1[x*3-3])*2 + row1[x*3+n1+3] - row1[x*3+n1-3] + row1[x*3+s1+3] - row1[x*3+s1-3]];
            prow1[x+width] = tab[(row1[x*3+4] - row1[x*3-2])*2 + row1[x*3+n1+4] - row1[x*3+n1-2] + row1[x*3+s1+4] - row1[x*3+s1-2]];
            prow1[x+width*2] = tab[(row1[x*3+5] - row1[x*3-1])*2 + row1[x*3+n1+5] - row1[x*3+n1-1] + row1[x*3+s1+5] - row1[x*3+s1-1]];

            prow2[width-1-x] = tab[(row2[x*3+3] - row2[x*3-3])*2 + row2[x*3+n2+3] - row2[x*3+n2-3] + row2[x*3+s2+3] - row2[x*3+s2-3]];
            prow2[width-1-x+width] = tab[(row2[x*3+4] - row2[x*3-2])*2 + row2[x*3+n2+4] - row2[x*3+n2-2] + row2[x*3+s2+4] - row2[x*3+s2-2]];
            prow2[width-1-x+width*2] = tab[(row2[x*3+5] - row2[x*3-1])*2 + row2[x*3+n2+5] - row2[x*3+n2-1] + row2[x*3+s2+5] - row2[x*3+s2-1]];

            prow1[x+width*3] = row1[x*3];
            prow1[x+width*4] = row1[x*3+1];
            prow1[x+width*5] = row1[x*3+2];

            prow2[width-1-x+width*3] = row2[x*3];
            prow2[width-1-x+width*4] = row2[x*3+1];
            prow2[width-1-x+width*5] = row2[x*3+2];
        }
    }

    memset( cost + xrange_min*D, 0, width1*D*sizeof(cost[0]) );

    buffer -= width-maxX2;
    cost -= (minX1-xrange_min)*D + minD; // simplify the cost indices inside the loop

    for( c = 0; c < cn*2; c++, prow1 += width, prow2 += width )
    {
        int diff_scale = c < cn ? 0 : 2;

        // precompute
        //   v0 = min(row2[x-1/2], row2[x], row2[x+1/2]) and
        //   v1 = max(row2[x-1/2], row2[x], row2[x+1/2]) and
        //   to process values from [minX2, maxX2) we should check memory location (width - 1 - maxX2, width - 1 - minX2]
        //   so iterate through [width - maxX2, width - minX2)
        for( x = width-maxX2; x < width-minX2; x++ )
        {
            int v = prow2[x];
            int vl = x > 0 ? (v + prow2[x-1])/2 : v;
            int vr = x < width-1 ? (v + prow2[x+1])/2 : v;
            int v0 = std::min(vl, vr); v0 = std::min(v0, v);
            int v1 = std::max(vl, vr); v1 = std::max(v1, v);
            buffer[x] = (PixType)v0;
            buffer[x + width2] = (PixType)v1;
        }

        for( x = minX1; x < maxX1; x++ )
        {
            int u = prow1[x];
            int ul = x > 0 ? (u + prow1[x-1])/2 : u;
            int ur = x < width-1 ? (u + prow1[x+1])/2 : u;
            int u0 = std::min(ul, ur); u0 = std::min(u0, u);
            int u1 = std::max(ul, ur); u1 = std::max(u1, u);

            int d = minD;
        #if (CV_SIMD || CV_SIMD_SCALABLE)
            v_uint8 _u  = vx_setall_u8((uchar)u), _u0 = vx_setall_u8((uchar)u0);
            v_uint8 _u1 = vx_setall_u8((uchar)u1);

            for( ; d <= maxD - 2*VTraits<v_int16>::vlanes(); d += 2*VTraits<v_int16>::vlanes() )
            {
                v_uint8 _v  = vx_load(prow2  + width-x-1 + d);
                v_uint8 _v0 = vx_load(buffer + width-x-1 + d);
                v_uint8 _v1 = vx_load(buffer + width-x-1 + d + width2);
                v_uint8 c0 = v_max(v_sub(_u, _v1), v_sub(_v0, _u));
                v_uint8 c1 = v_max(v_sub(_v, _u1), v_sub(_u0, _v));
                v_uint8 diff = v_min(c0, c1);

                v_int16 _c0 = vx_load_aligned(cost + x*D + d);
                v_int16 _c1 = vx_load_aligned(cost + x*D + d + VTraits<v_int16>::vlanes());

                v_uint16 diff1,diff2;
                v_expand(diff,diff1,diff2);
                v_store_aligned(cost + x*D + d,                   v_add(_c0, v_reinterpret_as_s16(v_shr(diff1, diff_scale))));
                v_store_aligned(cost + x*D + d + VTraits<v_int16>::vlanes(), v_add(_c1, v_reinterpret_as_s16(v_shr(diff2, diff_scale))));
            }
        #endif
            for( ; d < maxD; d++ )
            {
                int v = prow2[width-x-1 + d];
                int v0 = buffer[width-x-1 + d];
                int v1 = buffer[width-x-1 + d + width2];
                int c0 = std::max(0, u - v1); c0 = std::max(c0, v0 - u);
                int c1 = std::max(0, v - u1); c1 = std::max(c1, u0 - v);

                cost[x*D + d] = (CostType)(cost[x*D+d] + (std::min(c0, c1) >> diff_scale));
            }
        }
    }
}



class BufferSGBM
{
private:
    size_t width1;
    size_t Da;
    size_t Dlra;
    size_t costWidth;
    size_t costHeight;
    size_t hsumRows;
    bool fullDP;
    uchar dirs;
    uchar dirs2;
    static const size_t TAB_OFS = 256*4;

public:
    CostType* Cbuf;
    CostType* Sbuf;
    CostType* hsumBuf;
    CostType* pixDiff;
    CostType* disp2cost;
    DispType* disp2ptr;
    PixType* tempBuf;
    std::vector<CostType*> Lr;
    std::vector<CostType*> minLr;
    PixType * clipTab;

private:
    utils::BufferArea area;

public:
    BufferSGBM(size_t width1_,
               size_t Da_,
               size_t Dlra_,
               size_t cn,
               size_t width,
               size_t height,
               const StereoSGBMParams &params)
        : width1(width1_),
        Da(Da_),
        Dlra(Dlra_),
        Cbuf(NULL),
        Sbuf(NULL),
        hsumBuf(NULL),
        pixDiff(NULL),
        disp2cost(NULL),
        disp2ptr(NULL),
        tempBuf(NULL),
        Lr(2, (CostType*)NULL),
        minLr(2, (CostType*)NULL),
        clipTab(NULL)
    {
        const size_t TAB_SIZE = 256 + TAB_OFS*2;
        fullDP = params.isFullDP();
        costWidth = width1 * Da;
        costHeight = fullDP ? height : 1;
        hsumRows = params.calcSADWindowSize().height + 2;
        dirs = params.mode == StereoSGBM::MODE_HH4 ? 1 : NR;
        dirs2 = params.mode == StereoSGBM::MODE_HH4 ? 1 : NR2;
        // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
        // we keep pixel difference cost (C) and the summary cost over NR directions (S).
        // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
        area.allocate(Cbuf, costWidth * costHeight, CV_SIMD_WIDTH); // summary cost over different (nDirs) directions
        area.allocate(Sbuf, costWidth * costHeight, CV_SIMD_WIDTH);
        area.allocate(hsumBuf, costWidth * hsumRows, CV_SIMD_WIDTH);
        area.allocate(pixDiff, costWidth, CV_SIMD_WIDTH);
        area.allocate(disp2cost,    width, CV_SIMD_WIDTH);
        area.allocate(disp2ptr,     width, CV_SIMD_WIDTH);
        area.allocate(tempBuf,      width * (4 * cn + 2), CV_SIMD_WIDTH);
        // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
        // for 8-way dynamic programming we need the current row and
        // the previous row, i.e. 2 rows in total
        for (size_t i = 0; i < 2; ++i)
        {
            // 2D: [ NR ][ w1 * NR2 ][ NR ] * [ Dlra ]
            area.allocate(Lr[i], calcLrCount() * Dlra, CV_SIMD_WIDTH);
            // 1D: [ NR ][ w1 * NR2 ][ NR ]
            area.allocate(minLr[i], calcLrCount(), CV_SIMD_WIDTH);
        }
        area.allocate(clipTab, TAB_SIZE, CV_SIMD_WIDTH);
        area.commit();

        // init clipTab
        const int ftzero = std::max(params.preFilterCap, 15) | 1;
        for(int i = 0; i < (int)TAB_SIZE; i++ )
            clipTab[i] = (PixType)(std::min(std::max(i - (int)TAB_OFS, -ftzero), ftzero) + ftzero);
    }
    inline const PixType * getClipTab() const
    {
        return clipTab + TAB_OFS;
    }
    inline void initCBuf(CostType val) const
    {
        for (size_t i = 0; i < costWidth * costHeight; ++i)
            Cbuf[i] = val;
    }
    inline void clearLr(const Range & range = Range::all()) const
    {
            for (uchar i = 0; i < 2; ++i)
            {
                if (range == Range::all())
                {
                    memset(Lr[i],    0, calcLrCount() * Dlra * sizeof(CostType));
                    memset(minLr[i], 0, calcLrCount()        * sizeof(CostType));
                }
                else
                {
                    memset(getLr(i, range.start), 0, range.size() * sizeof(CostType) * Dlra);
                    memset(getMinLr(i, range.start), 0, range.size() * sizeof(CostType));
                }
            }
    }
    inline size_t calcLrCount() const
    {
        return width1 * dirs2 + 2 * dirs;
    }
    inline void swapLr()
    {
        std::swap(Lr[0], Lr[1]);
        std::swap(minLr[0], minLr[1]);
    }
    inline CostType * getHSumBuf(int row) const
    {
        return hsumBuf + (row % hsumRows) * costWidth;
    }
    inline CostType * getCBuf(int row) const
    {
        CV_Assert(row >= 0);
        return Cbuf + (!fullDP ? 0 : (row * costWidth));
    }
    inline CostType * getSBuf(int row) const
    {
        CV_Assert(row >= 0);
        return Sbuf + (!fullDP ? 0 : (row * costWidth));
    }
    inline void clearSBuf(int row, const Range & range = Range::all()) const
    {
        if (range == Range::all())
            memset(getSBuf(row), 0, costWidth * sizeof(CostType));
        else
            memset(getSBuf(row) + range.start * Da, 0, range.size() * Da * sizeof(CostType));
    }

    // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
    // and will occasionally use negative indices with the arrays
    // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
    inline CostType * getLr(uchar id, int idx, uchar shift = 0) const
    {
        CV_Assert(id < 2);
        const size_t fixed_offset = dirs * Dlra;
        return Lr[id] + fixed_offset + (idx * (int)dirs2 + (int)shift) * (int)Dlra;
    }
    inline CostType * getMinLr(uchar id, int idx, uchar shift = 0) const
    {
        CV_Assert(id < 2);
        const size_t fixed_offset = dirs;
        return minLr[id] + fixed_offset + (idx * dirs2 + shift);
    }
};

/*
 computes disparity for "roi" in img1 w.r.t. img2 and write it to disp1buf.
 that is, disp1buf(x, y)=d means that img1(x+roi.x, y+roi.y) ~ img2(x+roi.x-d, y+roi.y).
 minD <= d < maxD.
 disp2full is the reverse disparity map, that is:
 disp2full(x+roi.x,y+roi.y)=d means that img2(x+roi.x, y+roi.y) ~ img1(x+roi.x+d, y+roi.y)

 note that disp1buf will have the same size as the roi and
 disp2full will have the same size as img1 (or img2).
 On exit disp2buf is not the final disparity, it is an intermediate result that becomes
 final after all the tiles are processed.

 the disparity in disp1buf is written with sub-pixel accuracy
 (4 fractional bits, see StereoSGBM::DISP_SCALE),
 using quadratic interpolation, while the disparity in disp2buf
 is written as is, without interpolation.

 disp2cost also has the same size as img1 (or img2).
 It contains the minimum current cost, used to find the best disparity, corresponding to the minimal cost.
 */
static void computeDisparitySGBM( const Mat& img1, const Mat& img2,
                                 Mat& disp1, const StereoSGBMParams& params )
{
    const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
    const int DISP_SCALE = (1 << DISP_SHIFT);
    const CostType MAX_COST = SHRT_MAX;

    int minD = params.minDisparity, maxD = minD + params.numDisparities;
    int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
    int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
    int P1 = params.P1 > 0 ? params.P1 : 2, P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
    int k, width = disp1.cols, height = disp1.rows;
    int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
    const int D = params.numDisparities;
    int width1 = maxX1 - minX1;
    int Da = (int)alignSize(D,VTraits<v_int16>::vlanes());
    int Dlra = Da + VTraits<v_int16>::vlanes();//Additional memory is necessary to store disparity values(MAX_COST) for d=-1 and d=D
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
    int SW2 = params.calcSADWindowSize().width/2, SH2 = params.calcSADWindowSize().height/2;
    int npasses = params.isFullDP() ? 2 : 1;

    if( minX1 >= maxX1 )
    {
        disp1 = Scalar::all(INVALID_DISP_SCALED);
        return;
    }

    BufferSGBM mem(width1, Da, Dlra, img1.channels(), width, height, params);
    mem.initCBuf((CostType)P2); // add P2 to every C(x,y). it saves a few operations in the inner loops

    for( int pass = 1; pass <= npasses; pass++ )
    {
        int x1, y1, x2, y2, dx, dy;

        if( pass == 1 )
        {
            y1 = 0; y2 = height; dy = 1;
            x1 = 0; x2 = width1; dx = 1;
        }
        else
        {
            y1 = height-1; y2 = -1; dy = -1;
            x1 = width1-1; x2 = -1; dx = -1;
        }

        uchar lrID = 0;
        mem.clearLr();

        for( int y = y1; y != y2; y += dy )
        {
            int x, d;
            DispType* disp1ptr = disp1.ptr<DispType>(y);
            CostType* const C = mem.getCBuf(y);
            CostType* const S = mem.getSBuf(y);

            if( pass == 1 ) // compute C on the first pass, and reuse it on the second pass, if any.
            {
                int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

                for( k = dy1; k <= dy2; k++ )
                {
                    CostType* hsumAdd = mem.getHSumBuf(std::min(k, height-1));

                    if( k < height )
                    {
                        calcPixelCostBT( img1, img2, k, minD, maxD, mem.pixDiff, mem.tempBuf, mem.getClipTab() );

                        memset(hsumAdd, 0, Da*sizeof(CostType));
#if (CV_SIMD || CV_SIMD_SCALABLE)
                        v_int16 h_scale = vx_setall_s16((short)SW2 + 1);
                        for( d = 0; d < Da; d += VTraits<v_int16>::vlanes() )
                        {
                            v_int16 v_hsumAdd = v_mul(vx_load_aligned(mem.pixDiff + d), h_scale);
                            for( x = Da; x <= SW2*Da; x += Da )
                                v_hsumAdd = v_add(v_hsumAdd, vx_load_aligned(mem.pixDiff + x + d));
                            v_store_aligned(hsumAdd + d, v_hsumAdd);
                        }
#else
                        for (d = 0; d < D; d++)
                        {
                            hsumAdd[d] = (CostType)(mem.pixDiff[d] * (SW2 + 1));
                            for( x = Da; x <= SW2*Da; x += Da )
                                hsumAdd[d] = (CostType)(hsumAdd[d] + mem.pixDiff[x + d]);
                        }
#endif

                        if( y > 0 )
                        {
                            const CostType* hsumSub = mem.getHSumBuf(std::max(y - SH2 - 1, 0));
                            const CostType* Cprev =  mem.getCBuf(y - 1);

#if (CV_SIMD || CV_SIMD_SCALABLE)
                            for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                                v_store_aligned(C + d, v_sub(v_add(vx_load_aligned(Cprev + d), vx_load_aligned(hsumAdd + d)), vx_load_aligned(hsumSub + d)));
#else
                            for (d = 0; d < D; d++)
                                C[d] = (CostType)(Cprev[d] + hsumAdd[d] - hsumSub[d]);
#endif

                            for( x = Da; x < width1*Da; x += Da )
                            {
                                const CostType* pixAdd = mem.pixDiff + std::min(x + SW2*Da, (width1-1)*Da);
                                const CostType* pixSub = mem.pixDiff + std::max(x - (SW2+1)*Da, 0);
#if (CV_SIMD || CV_SIMD_SCALABLE)
                                for( d = 0; d < Da; d += VTraits<v_int16>::vlanes() )
                                {
                                    v_int16 hv = v_add(v_sub(vx_load_aligned(hsumAdd + x - Da + d), vx_load_aligned(pixSub + d)), vx_load_aligned(pixAdd + d));
                                    v_store_aligned(hsumAdd + x + d, hv);
                                    v_store_aligned(C + x + d, v_add(v_sub(vx_load_aligned(Cprev + x + d), vx_load_aligned(hsumSub + x + d)), hv));
                                }
#else
                                for( d = 0; d < D; d++ )
                                {
                                    int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - Da + d] + pixAdd[d] - pixSub[d]);
                                    C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
                                }
#endif
                            }
                        }
                        else
                        {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                            v_int16 v_scale = vx_setall_s16(k == 0 ? (short)SH2 + 1 : 1);
                            for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                                v_store_aligned(C + d, v_add(vx_load_aligned(C + d), v_mul(vx_load_aligned(hsumAdd + d), v_scale)));
#else
                            int scale = k == 0 ? SH2 + 1 : 1;
                            for (d = 0; d < D; d++)
                                C[d] = (CostType)(C[d] + hsumAdd[d] * scale);
#endif
                            for( x = Da; x < width1*Da; x += Da )
                            {
                                const CostType* pixAdd = mem.pixDiff + std::min(x + SW2*Da, (width1-1)*Da);
                                const CostType* pixSub = mem.pixDiff + std::max(x - (SW2+1)*Da, 0);

#if (CV_SIMD || CV_SIMD_SCALABLE)
                                for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                                {
                                    v_int16 hv = v_sub(v_add(vx_load_aligned(hsumAdd + x - Da + d), vx_load_aligned(pixAdd + d)), vx_load_aligned(pixSub + d));
                                    v_store_aligned(hsumAdd + x + d, hv);
                                    v_store_aligned(C + x + d, v_add(vx_load_aligned(C + x + d), v_mul(hv, v_scale)));
                                }
#else
                                for( d = 0; d < D; d++ )
                                {
                                    CostType hv = (CostType)(hsumAdd[x - Da + d] + pixAdd[d] - pixSub[d]);
                                    hsumAdd[x + d] = hv;
                                    C[x + d] = (CostType)(C[x + d] + hv * scale);
                                }
#endif
                            }
                        }
                    }
                    else
                    {
                        if( y > 0 )
                        {
                            const CostType* hsumSub = mem.getHSumBuf(std::max(y - SH2 - 1, 0));
                            const CostType* Cprev = mem.getCBuf(y - 1);
#if (CV_SIMD || CV_SIMD_SCALABLE)
                            for (x = 0; x < width1*Da; x += VTraits<v_int16>::vlanes())
                                v_store_aligned(C + x, v_add(v_sub(vx_load_aligned(Cprev + x), vx_load_aligned(hsumSub + x)), vx_load_aligned(hsumAdd + x)));
#else
                            for (x = 0; x < width1*Da; x++)
                                C[x] = (CostType)(Cprev[x] + hsumAdd[x] - hsumSub[x]);
#endif
                        }
                        else
                        {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                            for (x = 0; x < width1*Da; x += VTraits<v_int16>::vlanes())
                                v_store_aligned(C + x, v_add(vx_load_aligned(C + x), vx_load_aligned(hsumAdd + x)));
#else
                            for (x = 0; x < width1*Da; x++)
                                C[x] = (CostType)(C[x] + hsumAdd[x]);
#endif
                        }
                    }

                }

                // also, clear the S buffer
                mem.clearSBuf(y);
            }

            /*
             [formula 13 in the paper]
             compute L_r(p, d) = C(p, d) +
             min(L_r(p-r, d),
             L_r(p-r, d-1) + P1,
             L_r(p-r, d+1) + P1,
             min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
             where p = (x,y), r is one of the directions.
             we process all the directions at once:
             0: r=(-dx, 0)
             1: r=(-1, -dy)
             2: r=(0, -dy)
             3: r=(1, -dy)   !!!Note that only directions 0 to 3 are processed
             4: r=(-2, -dy)
             5: r=(-1, -dy*2)
             6: r=(1, -dy*2)
             7: r=(2, -dy)
             */

            for( x = x1; x != x2; x += dx )
            {
                int delta0 = P2 + *mem.getMinLr(lrID, x - dx);
                int delta1 = P2 + *mem.getMinLr(1 - lrID, x - 1, 1);
                int delta2 = P2 + *mem.getMinLr(1 - lrID, x,     2);
                int delta3 = P2 + *mem.getMinLr(1 - lrID, x + 1, 3);

                CostType* Lr_p0 = mem.getLr(lrID, x - dx);
                CostType* Lr_p1 = mem.getLr(1 - lrID, x - 1, 1);
                CostType* Lr_p2 = mem.getLr(1 - lrID, x,     2);
                CostType* Lr_p3 = mem.getLr(1 - lrID, x + 1, 3);

                Lr_p0[-1] = Lr_p0[D] = MAX_COST;
                Lr_p1[-1] = Lr_p1[D] = MAX_COST;
                Lr_p2[-1] = Lr_p2[D] = MAX_COST;
                Lr_p3[-1] = Lr_p3[D] = MAX_COST;

                CostType* Lr_p = mem.getLr(lrID, x);
                const CostType* Cp = C + x*Da;
                CostType* Sp = S + x*Da;

                CostType* minL = mem.getMinLr(lrID, x);
                d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_int16 _P1 = vx_setall_s16((short)P1);

                v_int16 _delta0 = vx_setall_s16((short)delta0);
                v_int16 _delta1 = vx_setall_s16((short)delta1);
                v_int16 _delta2 = vx_setall_s16((short)delta2);
                v_int16 _delta3 = vx_setall_s16((short)delta3);
                v_int16 _minL0 = vx_setall_s16((short)MAX_COST);
                v_int16 _minL1 = vx_setall_s16((short)MAX_COST);
                v_int16 _minL2 = vx_setall_s16((short)MAX_COST);
                v_int16 _minL3 = vx_setall_s16((short)MAX_COST);

                for( ; d <= D - VTraits<v_int16>::vlanes(); d += VTraits<v_int16>::vlanes() )
                {
                    v_int16 Cpd = vx_load_aligned(Cp + d);
                    v_int16 Spd = vx_load_aligned(Sp + d);
                    v_int16 L;

                    L = v_add(v_sub(v_min(v_min(v_min(vx_load_aligned(Lr_p0 + d), v_add(vx_load(Lr_p0 + d - 1), _P1)), v_add(vx_load(Lr_p0 + d + 1), _P1)), _delta0), _delta0), Cpd);
                    v_store_aligned(Lr_p + d, L);
                    _minL0 = v_min(_minL0, L);
                    Spd = v_add(Spd, L);

                    L = v_add(v_sub(v_min(v_min(v_min(vx_load_aligned(Lr_p1 + d), v_add(vx_load(Lr_p1 + d - 1), _P1)), v_add(vx_load(Lr_p1 + d + 1), _P1)), _delta1), _delta1), Cpd);
                    v_store_aligned(Lr_p + d + Dlra, L);
                    _minL1 = v_min(_minL1, L);
                    Spd = v_add(Spd, L);

                    L = v_add(v_sub(v_min(v_min(v_min(vx_load_aligned(Lr_p2 + d), v_add(vx_load(Lr_p2 + d - 1), _P1)), v_add(vx_load(Lr_p2 + d + 1), _P1)), _delta2), _delta2), Cpd);
                    v_store_aligned(Lr_p + d + Dlra*2, L);
                    _minL2 = v_min(_minL2, L);
                    Spd = v_add(Spd, L);

                    L = v_add(v_sub(v_min(v_min(v_min(vx_load_aligned(Lr_p3 + d), v_add(vx_load(Lr_p3 + d - 1), _P1)), v_add(vx_load(Lr_p3 + d + 1), _P1)), _delta3), _delta3), Cpd);
                    v_store_aligned(Lr_p + d + Dlra*3, L);
                    _minL3 = v_min(_minL3, L);
                    Spd = v_add(Spd, L);

                    v_store_aligned(Sp + d, Spd);
                }

#if CV_SIMD_WIDTH > 32
                minL[0] = v_reduce_min(_minL0);
                minL[1] = v_reduce_min(_minL1);
                minL[2] = v_reduce_min(_minL2);
                minL[3] = v_reduce_min(_minL3);
#else
                // Get minimum for L0-L3
                v_int16 t0, t1, t2, t3;
                v_zip(_minL0, _minL2, t0, t2);
                v_zip(_minL1, _minL3, t1, t3);
                v_zip(v_min(t0, t2), v_min(t1, t3), t0, t1);
                t0 = v_min(t0, t1);
                t0 = v_min(t0, v_rotate_right<4>(t0));
#if CV_SIMD_WIDTH == 32
                CostType buf[VTraits<v_int16>::max_nlanes];
                v_store_low(buf, v_min(t0, v_rotate_right<8>(t0)));
                minL[0] = buf[0];
                minL[1] = buf[1];
                minL[2] = buf[2];
                minL[3] = buf[3];
#else
                v_store_low(minL, t0);
#endif
#endif
#else
                minL[0] = MAX_COST;
                minL[1] = MAX_COST;
                minL[2] = MAX_COST;
                minL[3] = MAX_COST;
#endif
                for( ; d < D; d++ )
                {
                    int Cpd = Cp[d], L;
                    int Spd = Sp[d];

                    L = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0))) - delta0;
                    Lr_p[d] = (CostType)L;
                    minL[0] = std::min(minL[0], (CostType)L);
                    Spd += L;

                    L = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d - 1] + P1, std::min(Lr_p1[d + 1] + P1, delta1))) - delta1;
                    Lr_p[d + Dlra] = (CostType)L;
                    minL[1] = std::min(minL[1], (CostType)L);
                    Spd += L;

                    L = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d - 1] + P1, std::min(Lr_p2[d + 1] + P1, delta2))) - delta2;
                    Lr_p[d + Dlra*2] = (CostType)L;
                    minL[2] = std::min(minL[2], (CostType)L);
                    Spd += L;

                    L = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d - 1] + P1, std::min(Lr_p3[d + 1] + P1, delta3))) - delta3;
                    Lr_p[d + Dlra*3] = (CostType)L;
                    minL[3] = std::min(minL[3], (CostType)L);
                    Spd += L;

                    Sp[d] = saturate_cast<CostType>(Spd);
                }
            }

            if( pass == npasses )
            {
                x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_int16 v_inv_dist = vx_setall_s16((DispType)INVALID_DISP_SCALED);
                v_int16 v_max_cost = vx_setall_s16(MAX_COST);
                for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes() )
                {
                    v_store(disp1ptr + x, v_inv_dist);
                    v_store(mem.disp2ptr + x, v_inv_dist);
                    v_store(mem.disp2cost + x, v_max_cost);
                }
#endif
                for( ; x < width; x++ )
                {
                    disp1ptr[x] = mem.disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                    mem.disp2cost[x] = MAX_COST;
                }

                for( x = width1 - 1; x >= 0; x-- )
                {
                    CostType* Sp = S + x*Da;
                    CostType minS = MAX_COST;
                    short bestDisp = -1;

                    if( npasses == 1 )
                    {
                        CostType* Lr_p0 = mem.getLr(lrID, x + 1);
                        Lr_p0[-1] = Lr_p0[D] = MAX_COST;
                        CostType* Lr_p = mem.getLr(lrID, x);

                        const CostType* Cp = C + x*Da;

                        d = 0;
                        int delta0 = P2 + *mem.getMinLr(lrID, x + 1);
                        int minL0 = MAX_COST;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                        v_int16 _P1 = vx_setall_s16((short)P1);
                        v_int16 _delta0 = vx_setall_s16((short)delta0);

                        v_int16 _minL0 = vx_setall_s16((short)MAX_COST);
                        v_int16 _minS = vx_setall_s16(MAX_COST), _bestDisp = vx_setall_s16(-1);
                        for( ; d <= D - VTraits<v_int16>::vlanes(); d += VTraits<v_int16>::vlanes() )
                        {
                            v_int16 Cpd = vx_load_aligned(Cp + d);
                            v_int16 L0 = v_add(v_sub(v_min(v_min(v_min(vx_load_aligned(Lr_p0 + d), v_add(vx_load(Lr_p0 + d - 1), _P1)), v_add(vx_load(Lr_p0 + d + 1), _P1)), _delta0), _delta0), Cpd);

                            v_store_aligned(Lr_p + d, L0);
                            _minL0 = v_min(_minL0, L0);
                            L0 = v_add(L0, vx_load_aligned(Sp + d));
                            v_store_aligned(Sp + d, L0);

                            _bestDisp = v_select(v_gt(_minS, L0), vx_setall_s16((short)d), _bestDisp);
                            _minS = v_min(_minS, L0);
                        }
                        minL0 = (CostType)v_reduce_min(_minL0);
                        min_pos(_minS, _bestDisp, minS, bestDisp);
#endif
                        for( ; d < D; d++ )
                        {
                            int L0 = Cp[d] + std::min((int)Lr_p0[d], std::min(Lr_p0[d-1] + P1, std::min(Lr_p0[d+1] + P1, delta0))) - delta0;

                            Lr_p[d] = (CostType)L0;
                            minL0 = std::min(minL0, L0);

                            CostType Sval = Sp[d] = saturate_cast<CostType>(Sp[d] + L0);
                            if( Sval < minS )
                            {
                                minS = Sval;
                                bestDisp = (short)d;
                            }
                        }
                        *mem.getMinLr(lrID, x) = (CostType)minL0;
                    }
                    else
                    {
                        d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                        v_int16 _minS = vx_setall_s16(MAX_COST), _bestDisp = vx_setall_s16(-1);
                        for( ; d <= D - VTraits<v_int16>::vlanes(); d+= VTraits<v_int16>::vlanes() )
                        {
                            v_int16 L0 = vx_load_aligned(Sp + d);
                            _bestDisp = v_select(v_gt(_minS, L0), vx_setall_s16((short)d), _bestDisp);
                            _minS = v_min( L0, _minS );
                        }
                        min_pos(_minS, _bestDisp, minS, bestDisp);
#endif
                        for( ; d < D; d++ )
                        {
                            int Sval = Sp[d];
                            if( Sval < minS )
                            {
                                minS = (CostType)Sval;
                                bestDisp = (short)d;
                            }
                        }
                    }

                    for( d = 0; d < D; d++ )
                    {
                        if( Sp[d]*(100 - uniquenessRatio) < minS*100 && std::abs(bestDisp - d) > 1 )
                            break;
                    }
                    if( d < D )
                        continue;
                    d = bestDisp;
                    int _x2 = x + minX1 - d - minD;
                    if( mem.disp2cost[_x2] > minS )
                    {
                        mem.disp2cost[_x2] = (CostType)minS;
                        mem.disp2ptr[_x2] = (DispType)(d + minD);
                    }

                    if( 0 < d && d < D-1 )
                    {
                        // do subpixel quadratic interpolation:
                        //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
                        //   then find minimum of the parabola.
                        int denom2 = std::max(Sp[d-1] + Sp[d+1] - 2*Sp[d], 1);
                        d = d*DISP_SCALE + ((Sp[d-1] - Sp[d+1])*DISP_SCALE + denom2)/(denom2*2);
                    }
                    else
                        d *= DISP_SCALE;
                    disp1ptr[x + minX1] = (DispType)(d + minD*DISP_SCALE);
                }

                for( x = minX1; x < maxX1; x++ )
                {
                    // we round the computed disparity both towards -inf and +inf and check
                    // if either of the corresponding disparities in disp2 is consistent.
                    // This is to give the computed disparity a chance to look valid if it is.
                    int d1 = disp1ptr[x];
                    if( d1 == INVALID_DISP_SCALED )
                        continue;
                    int _d = d1 >> DISP_SHIFT;
                    int d_ = (d1 + DISP_SCALE-1) >> DISP_SHIFT;
                    int _x = x - _d, x_ = x - d_;
                    if( 0 <= _x && _x < width && mem.disp2ptr[_x] >= minD && std::abs(mem.disp2ptr[_x] - _d) > disp12MaxDiff &&
                       0 <= x_ && x_ < width && mem.disp2ptr[x_] >= minD && std::abs(mem.disp2ptr[x_] - d_) > disp12MaxDiff )
                        disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
                }
            }

            lrID = 1 - lrID; // now shift the cyclic buffers
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
struct CalcVerticalSums: public ParallelLoopBody
{
    CalcVerticalSums(const Mat& _img1, const Mat& _img2, const StereoSGBMParams& params, const BufferSGBM &mem_)
        : img1(_img1), img2(_img2), mem(mem_)
    {
        minD = params.minDisparity;
        maxD = minD + params.numDisparities;
        SW2 = SH2 = params.calcSADWindowSize().height/2;
        P1 = params.P1 > 0 ? params.P1 : 2;
        P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
        height = img1.rows;
        width = img1.cols;
        int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
        D = maxD - minD;
        Da = (int)alignSize(D, VTraits<v_int16>::vlanes());
        Dlra = Da + VTraits<v_int16>::vlanes();//Additional memory is necessary to store disparity values(MAX_COST) for d=-1 and d=D
        width1 = maxX1 - minX1;
        D = params.numDisparities;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        const CostType MAX_COST = SHRT_MAX;
        const int npasses = 2;
        const int x1 = range.start, x2 = range.end;
        int k;

        CostType* pixDiff = 0;
        PixType* tempBuf = 0;
        utils::BufferArea aux_area;
        aux_area.allocate(pixDiff, ((x2 - x1) + 2 * SW2) * Da, CV_SIMD_WIDTH);
        aux_area.allocate(tempBuf, width * (4 * img1.channels() + 2) * sizeof(PixType), CV_SIMD_WIDTH);
        aux_area.commit();

        // Simplification of index calculation
        if (x1 > SW2)
            pixDiff -= (x1 - SW2) * Da;

        for( int pass = 1; pass <= npasses; pass++ )
        {
            int y1, y2, dy;

            if( pass == 1 )
            {
                y1 = 0; y2 = height; dy = 1;
            }
            else
            {
                y1 = height-1; y2 = -1; dy = -1;
            }

            uchar lrID = 0;
            mem.clearLr(range);

            for( int y = y1; y != y2; y += dy )
            {
                int x, d;
                CostType* C = mem.getCBuf(y);
                CostType* S = mem.getSBuf(y);

                if( pass == 1 ) // compute C on the first pass, and reuse it on the second pass, if any.
                {
                    int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

                    for( k = dy1; k <= dy2; k++ )
                    {
                        CostType* hsumAdd = mem.getHSumBuf(std::min(k, height-1));

                        if( k < height )
                        {
                            calcPixelCostBT( img1, img2, k, minD, maxD, pixDiff, tempBuf, mem.getClipTab(), x1 - SW2, x2 + SW2);

                            memset(hsumAdd + x1*Da, 0, Da*sizeof(CostType));
                            for( x = (x1 - SW2)*Da; x <= (x1 + SW2)*Da; x += Da )
                            {
                                int xbord = x <= 0 ? 0 : (x > (width1 - 1)*Da ? (width1 - 1)*Da : x);
#if (CV_SIMD || CV_SIMD_SCALABLE)
                                for( d = 0; d < Da; d += VTraits<v_int16>::vlanes() )
                                    v_store_aligned(hsumAdd + x1*Da + d, v_add(vx_load_aligned(hsumAdd + x1 * this->Da + d), vx_load_aligned(pixDiff + xbord + d)));
#else
                                for( d = 0; d < D; d++ )
                                    hsumAdd[x1*Da + d] = (CostType)(hsumAdd[x1*Da + d] + pixDiff[xbord + d]);
#endif
                            }

                            if( y > 0 )
                            {
                                const CostType* hsumSub =  mem.getHSumBuf(std::max(y - SH2 - 1, 0));
                                const CostType* Cprev = mem.getCBuf(y - 1);
#if (CV_SIMD || CV_SIMD_SCALABLE)
                                for( d = 0; d < Da; d += VTraits<v_int16>::vlanes() )
                                    v_store_aligned(C + x1*Da + d, v_sub(v_add(vx_load_aligned(Cprev + x1 * this->Da + d), vx_load_aligned(hsumAdd + x1 * this->Da + d)), vx_load_aligned(hsumSub + x1 * this->Da + d)));
#else
                                for( d = 0; d < D; d++ )
                                    C[x1*Da + d] = (CostType)(Cprev[x1*Da + d] + hsumAdd[x1*Da + d] - hsumSub[x1*Da + d]);
#endif
                                for( x = (x1+1)*Da; x < x2*Da; x += Da )
                                {
                                    const CostType* pixAdd = pixDiff + std::min(x + SW2*Da, (width1-1)*Da);
                                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*Da, 0);

#if (CV_SIMD || CV_SIMD_SCALABLE)
                                    for( d = 0; d < Da; d += VTraits<v_int16>::vlanes() )
                                    {
                                        v_int16 hv = v_add(v_sub(vx_load_aligned(hsumAdd + x - this->Da + d), vx_load_aligned(pixSub + d)), vx_load_aligned(pixAdd + d));
                                        v_store_aligned(hsumAdd + x + d, hv);
                                        v_store_aligned(C + x + d, v_add(v_sub(vx_load_aligned(Cprev + x + d), vx_load_aligned(hsumSub + x + d)), hv));
                                    }
#else
                                    for( d = 0; d < D; d++ )
                                    {
                                        int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - Da + d] + pixAdd[d] - pixSub[d]);
                                        C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
                                    }
#endif
                                }
                            }
                            else
                            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                                v_int16 v_scale = vx_setall_s16(k == 0 ? (short)SH2 + 1 : 1);
                                for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                                    v_store_aligned(C + x1*Da + d, v_add(vx_load_aligned(C + x1 * this->Da + d), v_mul(vx_load_aligned(hsumAdd + x1 * this->Da + d), v_scale)));
#else
                                int scale = k == 0 ? SH2 + 1 : 1;
                                for (d = 0; d < D; d++)
                                    C[x1*Da + d] = (CostType)(C[x1*Da + d] + hsumAdd[x1*Da + d] * scale);
#endif
                                for( x = (x1+1)*Da; x < x2*Da; x += Da )
                                {
                                    const CostType* pixAdd = pixDiff + std::min(x + SW2*Da, (width1-1)*Da);
                                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*Da, 0);
#if (CV_SIMD || CV_SIMD_SCALABLE)
                                    for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                                    {
                                        v_int16 hv = v_sub(v_add(vx_load_aligned(hsumAdd + x - this->Da + d), vx_load_aligned(pixAdd + d)), vx_load_aligned(pixSub + d));
                                        v_store_aligned(hsumAdd + x + d, hv);
                                        v_store_aligned(C + x + d, v_add(vx_load_aligned(C + x + d), v_mul(hv, v_scale)));
                                    }
#else
                                    for( d = 0; d < D; d++ )
                                    {
                                        CostType hv = (CostType)(hsumAdd[x - Da + d] + pixAdd[d] - pixSub[d]);
                                        hsumAdd[x + d] = hv;
                                        C[x + d] = (CostType)(C[x + d] + hv * scale);
                                    }
#endif
                                }
                            }
                        }
                        else
                        {
/*                            if (y > 0)
                            {
                                const CostType* hsumSub = mem.getHSumBuf(std::max(y - SH2 - 1, 0));
                                const CostType* Cprev = mem.getCBuf(y - 1);

#if (CV_SIMD || CV_SIMD_SCALABLE)
                                for( x = x1*Da; x < x2*Da; x += VTraits<v_int16>::vlanes() )
                                    v_store_aligned(C + x, v_add(v_sub(vx_load_aligned(Cprev + x), vx_load_aligned(hsumSub + x)), vx_load_aligned(hsumAdd + x)));
#else
                                for( x = x1*Da; x < x2*Da; x++ )
                                    C[x] = (CostType)(Cprev[x] + hsumAdd[x] - hsumSub[x]);
#endif
                            }
                            else*/
                            if(y == 0)
                            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                                for( x = x1*Da; x < x2*Da; x += VTraits<v_int16>::vlanes() )
                                    v_store_aligned(C + x, v_add(vx_load_aligned(C + x), vx_load_aligned(hsumAdd + x)));
#else
                                for( x = x1*Da; x < x2*Da; x++ )
                                    C[x] = (CostType)(C[x] + hsumAdd[x]);
#endif
                            }
                        }
                    }
                    mem.clearSBuf(y, range);
                }

//              [formula 13 in the paper]
//              compute L_r(p, d) = C(p, d) +
//              min(L_r(p-r, d),
//              L_r(p-r, d-1) + P1,
//              L_r(p-r, d+1) + P1,
//              min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
//              where p = (x,y), r is one of the directions.
//              we process one directions on first pass and other on second:
//              r=(0, dy), where dy=1 on first pass and dy=-1 on second

                for( x = x1; x != x2; x++ )
                {
                    int delta = P2 + *mem.getMinLr(1 - lrID, x);
                    CostType* Lr_ppr = mem.getLr(1 - lrID, x);

                    Lr_ppr[-1] = Lr_ppr[D] = MAX_COST;

                    CostType* Lr_p = mem.getLr(lrID, x);
                    const CostType* Cp = C + x*Da;
                    CostType* Sp = S + x*Da;

                    CostType& minL = *(mem.getMinLr(lrID, x));
                    d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    v_int16 _P1 = vx_setall_s16((short)P1);

                    v_int16 _delta = vx_setall_s16((short)delta);
                    v_int16 _minL = vx_setall_s16((short)MAX_COST);

                    for( ; d <= D - VTraits<v_int16>::vlanes(); d += VTraits<v_int16>::vlanes() )
                    {
                        v_int16 Cpd = vx_load_aligned(Cp + d);
                        v_int16 L = v_add(v_sub(v_min(v_min(v_min(vx_load_aligned(Lr_ppr + d), v_add(vx_load(Lr_ppr + d - 1), _P1)), v_add(vx_load(Lr_ppr + d + 1), _P1)), _delta), _delta), Cpd);
                        v_store_aligned(Lr_p + d, L);
                        _minL = v_min(_minL, L);
                        v_store_aligned(Sp + d, v_add(vx_load_aligned(Sp + d), L));
                    }
                    minL = v_reduce_min(_minL);
#else
                    minL = MAX_COST;
#endif
                    for( ; d < D; d++ )
                    {
                        int Cpd = Cp[d], L;

                        L = Cpd + std::min((int)Lr_ppr[d], std::min(Lr_ppr[d-1] + P1, std::min(Lr_ppr[d+1] + P1, delta))) - delta;

                        Lr_p[d] = (CostType)L;
                        minL = std::min(minL, (CostType)L);

                        Sp[d] = saturate_cast<CostType>(Sp[d] + L);
                    }
                }
                lrID = 1 - lrID; // now shift the cyclic buffers
            }
        }
    }
    const Mat& img1;
    const Mat& img2;
    const BufferSGBM & mem;
    int minD;
    int maxD;
    int D, Da, Dlra;
    int SH2;
    int SW2;
    int width;
    int width1;
    int height;
    int P1;
    int P2;
};

struct CalcHorizontalSums: public ParallelLoopBody
{
    CalcHorizontalSums(const Mat& _img1, const Mat& _img2, Mat& _disp1, const StereoSGBMParams& params, const BufferSGBM &mem_)
        : img1(_img1), img2(_img2), disp1(_disp1), mem(mem_)
    {
        minD = params.minDisparity;
        maxD = minD + params.numDisparities;
        P1 = params.P1 > 0 ? params.P1 : 2;
        P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
        uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
        disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
        height = img1.rows;
        width = img1.cols;
        minX1 = std::max(maxD, 0);
        maxX1 = width + std::min(minD, 0);
        INVALID_DISP = minD - 1;
        INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
        D = maxD - minD;
        Da = (int)alignSize(D, VTraits<v_int16>::vlanes());
        Dlra = Da + VTraits<v_int16>::vlanes();//Additional memory is necessary to store disparity values(MAX_COST) for d=-1 and d=D
        width1 = maxX1 - minX1;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int y1 = range.start, y2 = range.end;

        const size_t LrSize = 2 * (1 + Dlra + 1);

        CostType * Lr = 0;
        CostType * disp2cost = 0;
        DispType * disp2ptr = 0;
        utils::BufferArea aux_area;
        aux_area.allocate(Lr, LrSize);
        aux_area.allocate(disp2cost, width, CV_SIMD_WIDTH);
        aux_area.allocate(disp2ptr, width, CV_SIMD_WIDTH);
        aux_area.commit();

        CostType minLr;

        for( int y = y1; y != y2; y++)
        {
            int x, d;
            DispType* disp1ptr = disp1.ptr<DispType>(y);
            CostType* C = mem.getCBuf(y);
            CostType* S = mem.getSBuf(y);

            x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_int16 v_inv_dist = vx_setall_s16((DispType)INVALID_DISP_SCALED);
            v_int16 v_max_cost = vx_setall_s16(MAX_COST);
            for (; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
            {
                v_store(disp1ptr + x, v_inv_dist);
                v_store(disp2ptr + x, v_inv_dist);
                v_store(disp2cost + x, v_max_cost);
            }
#endif
            for( ; x < width; x++ )
            {
                disp1ptr[x] = disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                disp2cost[x] = MAX_COST;
            }

            // clear buffers
            aux_area.zeroFill(Lr);
            Lr[0] = Lr[1 + D] = Lr[3 + Dlra - 1] = Lr[3 + Dlra + D] = MAX_COST;

            minLr = 0;
//          [formula 13 in the paper]
//          compute L_r(p, d) = C(p, d) +
//          min(L_r(p-r, d),
//          L_r(p-r, d-1) + P1,
//          L_r(p-r, d+1) + P1,
//          min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
//          where p = (x,y), r is one of the directions.
//          we process all the directions at once:
//          we process one directions on first pass and other on second:
//          r=(dx, 0), where dx=1 on first pass and dx=-1 on second
            for( x = 0; x != width1; x++)
            {
                int delta = minLr + P2;
                CostType* Lr_ppr = Lr + ((x&1)? 1 : 3 + Dlra);
                CostType* Lr_p = Lr + ((x&1)? 3 + Dlra : 1);
                const CostType* Cp = C + x*Da;
                CostType* Sp = S + x*Da;

                d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_int16 _P1 = vx_setall_s16((short)P1);

                v_int16 _delta = vx_setall_s16((short)delta);
                v_int16 _minL = vx_setall_s16((short)MAX_COST);

                for( ; d <= D - VTraits<v_int16>::vlanes(); d += VTraits<v_int16>::vlanes())
                {
                    v_int16 Cpd = vx_load_aligned(Cp + d);
                    v_int16 L = v_add(v_sub(v_min(v_min(v_min(vx_load(Lr_ppr + d), v_add(vx_load(Lr_ppr + d - 1), _P1)), v_add(vx_load(Lr_ppr + d + 1), _P1)), _delta), _delta), Cpd);
                    v_store(Lr_p + d, L);
                    _minL = v_min(_minL, L);
                    v_store_aligned(Sp + d, v_add(vx_load_aligned(Sp + d), L));
                }
                minLr = v_reduce_min(_minL);
#else
                minLr = MAX_COST;
#endif
                for( ; d < D; d++ )
                {
                    int Cpd = Cp[d], L;
                    L = Cpd + std::min((int)Lr_ppr[d], std::min(Lr_ppr[d-1] + P1, std::min(Lr_ppr[d+1] + P1, delta))) - delta;
                    Lr_p[d] = (CostType)L;
                    minLr = std::min(minLr, (CostType)L);
                    Sp[d] = saturate_cast<CostType>(Sp[d] + L);
                }
            }

            aux_area.zeroFill(Lr);
            Lr[0] = Lr[1 + D] = Lr[3 + Dlra - 1] = Lr[3 + Dlra + D] = MAX_COST;

            minLr = 0;

            for( x = width1-1; x != -1; x--)
            {
                int delta = minLr + P2;
                CostType* Lr_ppr = Lr + ((x&1)? 1 : 3 + Dlra);
                CostType* Lr_p = Lr + ((x&1)? 3 + Dlra : 1);
                const CostType* Cp = C + x*Da;
                CostType* Sp = S + x*Da;
                CostType minS = MAX_COST;
                short bestDisp = -1;
                minLr = MAX_COST;

                d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_int16 _P1 = vx_setall_s16((short)P1);
                v_int16 _delta = vx_setall_s16((short)delta);

                v_int16 _minL = vx_setall_s16((short)MAX_COST);
                v_int16 _minS = vx_setall_s16(MAX_COST), _bestDisp = vx_setall_s16(-1);
                for( ; d <= D - VTraits<v_int16>::vlanes(); d += VTraits<v_int16>::vlanes() )
                {
                    v_int16 Cpd = vx_load_aligned(Cp + d);
                    v_int16 L = v_add(v_sub(v_min(v_min(v_min(vx_load(Lr_ppr + d), v_add(vx_load(Lr_ppr + d - 1), _P1)), v_add(vx_load(Lr_ppr + d + 1), _P1)), _delta), _delta), Cpd);
                    v_store(Lr_p + d, L);
                    _minL = v_min(_minL, L);
                    L = v_add(L, vx_load_aligned(Sp + d));
                    v_store_aligned(Sp + d, L);

                    _bestDisp = v_select(v_gt(_minS, L), vx_setall_s16((short)d), _bestDisp);
                    _minS = v_min( L, _minS );
                }
                minLr = v_reduce_min(_minL);

                min_pos(_minS, _bestDisp, minS, bestDisp);
#endif
                for( ; d < D; d++ )
                {
                    int Cpd = Cp[d], L;

                    L = Cpd + std::min((int)Lr_ppr[d], std::min(Lr_ppr[d-1] + P1, std::min(Lr_ppr[d+1] + P1, delta))) - delta;

                    Lr_p[d] = (CostType)L;
                    minLr = std::min(minLr, (CostType)L);

                    Sp[d] = saturate_cast<CostType>(Sp[d] + L);
                    if( Sp[d] < minS )
                    {
                        minS = Sp[d];
                        bestDisp = (short)d;
                    }
                }

                //Some postprocessing procedures and saving
                for( d = 0; d < D; d++ )
                {
                    if( Sp[d]*(100 - uniquenessRatio) < minS*100 && std::abs(bestDisp - d) > 1 )
                        break;
                }
                if( d < D )
                    continue;
                d = bestDisp;
                int _x2 = x + minX1 - d - minD;
                if( disp2cost[_x2] > minS )
                {
                    disp2cost[_x2] = (CostType)minS;
                    disp2ptr[_x2] = (DispType)(d + minD);
                }

                if( 0 < d && d < D-1 )
                {
                    // do subpixel quadratic interpolation:
                    //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
                    //   then find minimum of the parabola.
                    int denom2 = std::max(Sp[d-1] + Sp[d+1] - 2*Sp[d], 1);
                    d = d*DISP_SCALE + ((Sp[d-1] - Sp[d+1])*DISP_SCALE + denom2)/(denom2*2);
                }
                else
                    d *= DISP_SCALE;
                disp1ptr[x + minX1] = (DispType)(d + minD*DISP_SCALE);
            }
            //Left-right check sanity procedure
            for( x = minX1; x < maxX1; x++ )
            {
                // we round the computed disparity both towards -inf and +inf and check
                // if either of the corresponding disparities in disp2 is consistent.
                // This is to give the computed disparity a chance to look valid if it is.
                int d1 = disp1ptr[x];
                if( d1 == INVALID_DISP_SCALED )
                    continue;
                int _d = d1 >> DISP_SHIFT;
                int d_ = (d1 + DISP_SCALE-1) >> DISP_SHIFT;
                int _x = x - _d, x_ = x - d_;
                if( 0 <= _x && _x < width && disp2ptr[_x] >= minD && std::abs(disp2ptr[_x] - _d) > disp12MaxDiff &&
                   0 <= x_ && x_ < width && disp2ptr[x_] >= minD && std::abs(disp2ptr[x_] - d_) > disp12MaxDiff )
                    disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
            }
        }
    }

    static const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
    static const int DISP_SCALE = (1 << DISP_SHIFT);
    static const CostType MAX_COST = SHRT_MAX;
    const Mat& img1;
    const Mat& img2;
    Mat& disp1;
    const BufferSGBM & mem;
    int minD;
    int maxD;
    int D, Da, Dlra;
    int width;
    int width1;
    int height;
    int P1;
    int P2;
    int minX1;
    int maxX1;
    int INVALID_DISP;
    int INVALID_DISP_SCALED;
    int uniquenessRatio;
    int disp12MaxDiff;
};
/*
 computes disparity for "roi" in img1 w.r.t. img2 and write it to disp1buf.
 that is, disp1buf(x, y)=d means that img1(x+roi.x, y+roi.y) ~ img2(x+roi.x-d, y+roi.y).
 minD <= d < maxD.

 note that disp1buf will have the same size as the roi and
 On exit disp1buf is not the final disparity, it is an intermediate result that becomes
 final after all the tiles are processed.

 the disparity in disp1buf is written with sub-pixel accuracy
 (4 fractional bits, see StereoSGBM::DISP_SCALE),
 using quadratic interpolation, while the disparity in disp2buf
 is written as is, without interpolation.
 */
static void computeDisparitySGBM_HH4( const Mat& img1, const Mat& img2,
                                 Mat& disp1, const StereoSGBMParams& params)
{
    const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
    const int DISP_SCALE = (1 << DISP_SHIFT);
    int minD = params.minDisparity, maxD = minD + params.numDisparities;
    Size SADWindowSize;
    SADWindowSize.width = SADWindowSize.height = params.SADWindowSize > 0 ? params.SADWindowSize : 5;
    int P1 = params.P1 > 0 ? params.P1 : 2, P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
    int width = disp1.cols, height = disp1.rows;
    int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
    int width1 = maxX1 - minX1;
    int Da = (int)alignSize(params.numDisparities, VTraits<v_int16>::vlanes());
    int Dlra = Da + VTraits<v_int16>::vlanes();//Additional memory is necessary to store disparity values(MAX_COST) for d=-1 and d=D
    int INVALID_DISP = minD - 1;
    int INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;

    if( minX1 >= maxX1 )
    {
        disp1 = Scalar::all(INVALID_DISP_SCALED);
        return;
    }

    BufferSGBM mem(width1, Da, Dlra, img1.channels(), width, height, params);
    mem.initCBuf((CostType)P2); // add P2 to every C(x,y). it saves a few operations in the inner loops

    parallel_for_(Range(0,width1),CalcVerticalSums(img1, img2, params, mem),8);
    parallel_for_(Range(0,height),CalcHorizontalSums(img1, img2, disp1, params, mem),8);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

class BufferSGBM3Way
{
private:
    size_t hsumCols;
    size_t hsumRows;
public:
    CostType *curCostVolumeLine;
    CostType *hsumBuf;
    CostType *pixDiff;
    PixType *tmpBuf;
    CostType *horPassCostVolume;
    CostType *vertPassCostVolume;
    CostType *vertPassMin;
    CostType *rightPassBuf;
    CostType *disp2CostBuf;
    short *disp2Buf;
private:
    utils::BufferArea area;
public:
    BufferSGBM3Way(int width1, int width, int num_ch, int Da, int SH2, int P2) :
        curCostVolumeLine(0),
        hsumBuf(0),
        pixDiff(0),
        tmpBuf(0),
        horPassCostVolume(0),
        vertPassCostVolume(0),
        vertPassMin(0),
        rightPassBuf(0),
        disp2CostBuf(0),
        disp2Buf(0)
    {
        hsumCols = width1 * Da;
        hsumRows = SH2*2 + 2;
        area.allocate(curCostVolumeLine, hsumCols, CV_SIMD_WIDTH);
        area.allocate(hsumBuf, hsumCols * hsumRows, CV_SIMD_WIDTH);
        area.allocate(pixDiff,hsumCols, CV_SIMD_WIDTH);
        area.allocate(tmpBuf, width * (4 * num_ch + 2), CV_SIMD_WIDTH);
        area.allocate(horPassCostVolume, (width1 + 2) * Da, CV_SIMD_WIDTH);
        area.allocate(vertPassCostVolume, (width1 + 2) * Da, CV_SIMD_WIDTH);
        area.allocate(vertPassMin, width1 + 2, CV_SIMD_WIDTH);
        area.allocate(rightPassBuf, Da, CV_SIMD_WIDTH);
        area.allocate(disp2CostBuf, width, CV_SIMD_WIDTH);
        area.allocate(disp2Buf, width, CV_SIMD_WIDTH);
        area.commit();
        area.zeroFill();
        for(size_t i = 0; i < hsumCols; i++)
            curCostVolumeLine[i] = (CostType)P2;
    }
    inline void clearRightPassBuf()
    {
        area.zeroFill(rightPassBuf);
    }
    CostType *getHSumBuf(int x) const
    {
        return hsumBuf + (x % hsumRows) * hsumCols;
    }
};

struct SGBM3WayMainLoop : public ParallelLoopBody
{
    const Mat *img1, *img2;
    Mat* dst_disp;

    int stripe_sz;
    int stripe_overlap;

    int width,height;
    int minD, maxD, D, Da;
    int minX1, maxX1, width1;

    int SW2, SH2;
    int P1, P2;
    int uniquenessRatio, disp12MaxDiff;

    int TAB_OFS;

    utils::BufferArea aux_area;
    PixType* clipTab;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    short idx_row[VTraits<v_int16>::max_nlanes];
#endif
    SGBM3WayMainLoop(const Mat& _img1, const Mat& _img2, Mat* _dst_disp, const StereoSGBMParams& params, int stripe_size, int _stripe_overlap);
    void operator () (const Range& range) const CV_OVERRIDE;
    template<bool x_nlanes> void impl(const Range& range) const;

private:
    void getRawMatchingCost(const BufferSGBM3Way &mem, int y, int src_start_idx) const;

    template<bool x_nlanes>
    void accumulateCostsLeftTop(const BufferSGBM3Way &mem,
                                int x,
                                CostType &leftMinCost) const;

    template<bool x_nlanes>
    void accumulateCostsRight(const BufferSGBM3Way &mem,
                              int x,
                              CostType &rightMinCost,
                              short &optimal_disp,
                              CostType &min_cost) const;
};

SGBM3WayMainLoop::SGBM3WayMainLoop(const Mat& _img1,
                                   const Mat& _img2,
                                   Mat* _dst_disp,
                                   const StereoSGBMParams& params,
                                   int _stripe_sz,
                                   int _stripe_overlap)
    : img1(&_img1),
    img2(&_img2),
    dst_disp(_dst_disp),
    stripe_sz(_stripe_sz),
    stripe_overlap(_stripe_overlap),
    clipTab(0)
{
    // precompute a lookup table for the raw matching cost computation:
    TAB_OFS = 256*4;
    const int TAB_SIZE = 256 + TAB_OFS*2;
    aux_area.allocate(clipTab, TAB_SIZE, CV_SIMD_WIDTH);
    aux_area.commit();
    const int ftzero = std::max(params.preFilterCap, 15) | 1;
    for(int k = 0; k < TAB_SIZE; k++ )
        clipTab[k] = (PixType)(std::min(std::max(k - TAB_OFS, -ftzero), ftzero) + ftzero);

    width = img1->cols; height = img1->rows;
    minD = params.minDisparity; maxD = minD + params.numDisparities; D = maxD - minD;
    minX1 = std::max(maxD, 0); maxX1 = width + std::min(minD, 0); width1 = maxX1 - minX1;
    Da = (int)alignSize(D, VTraits<v_int16>::vlanes());

    SW2 = SH2 = params.SADWindowSize > 0 ? params.SADWindowSize/2 : 1;

    P1 = params.P1 > 0 ? params.P1 : 2; P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
    uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
    disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    for(short i = 0; i < VTraits<v_int16>::vlanes(); ++i)
        idx_row[i] = i;
#endif
}

// performing block matching and building raw cost-volume for the current row
void SGBM3WayMainLoop::getRawMatchingCost(const BufferSGBM3Way &mem, int y, int src_start_idx) const
{
    CostType* C = mem.curCostVolumeLine;
    CostType* pixDiff = mem.pixDiff;
    PixType* tmpBuf = mem.tmpBuf;
    int x, d;
    int dy1 = (y == src_start_idx) ? src_start_idx : y + SH2, dy2 = (y == src_start_idx) ? src_start_idx+SH2 : dy1;

    for(int k = dy1; k <= dy2; k++ )
    {
        CostType* hsumAdd = mem.getHSumBuf(std::min(k, height-1));
        if( k < height )
        {
            calcPixelCostBT( *img1, *img2, k, minD, maxD, pixDiff, tmpBuf, clipTab + TAB_OFS );

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_int16 sw2_1 = vx_setall_s16((short)SW2 + 1);
            for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
            {
                v_int16 hsA = v_mul(vx_load_aligned(pixDiff + d), sw2_1);
                for (x = Da; x <= SW2 * Da; x += Da)
                    hsA = v_add(hsA, vx_load_aligned(pixDiff + x + d));
                v_store_aligned(hsumAdd + d, hsA);
            }
#else
            for (d = 0; d < D; d++)
            {
                CostType hsA = (CostType)(pixDiff[d] * (SW2 + 1));
                for (x = Da; x <= SW2 * Da; x += Da)
                    hsA += pixDiff[x + d];
                hsumAdd[d] = hsA;
            }
#endif
            if( y > src_start_idx )
            {
                const CostType* hsumSub = mem.getHSumBuf(std::max(y - SH2 - 1, src_start_idx));

#if (CV_SIMD || CV_SIMD_SCALABLE)
                for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                    v_store_aligned(C + d, v_sub(v_add(vx_load_aligned(C + d), vx_load_aligned(hsumAdd + d)), vx_load_aligned(hsumSub + d)));
#else
                for (d = 0; d < D; d++)
                    C[d] = (CostType)(C[d] + hsumAdd[d] - hsumSub[d]);
#endif

                for( x = Da; x < width1*Da; x += Da )
                {
                    const CostType* pixAdd = pixDiff + std::min(x + SW2*Da, (width1-1)*Da);
                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*Da, 0);
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    v_int16 hv_reg;
                    for( d = 0; d < Da; d+=VTraits<v_int16>::vlanes() )
                    {
                        hv_reg = v_sub(v_add(vx_load_aligned(hsumAdd + x - this->Da + d), vx_load_aligned(pixAdd + d)), vx_load_aligned(pixSub + d));
                        v_store_aligned(hsumAdd+x+d,hv_reg);
                        v_store_aligned(C+x+d,v_sub(v_add(vx_load_aligned(C + x + d), hv_reg), vx_load_aligned(hsumSub + x + d)));
                    }
#else
                    for( d = 0; d < D; d++ )
                    {
                        int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                        C[x + d] = (CostType)(C[x + d] + hv - hsumSub[x + d]);
                    }
#endif
                }
            }
            else
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_int16 v_scale = vx_setall_s16(k == src_start_idx ? (short)SH2 + 1 : 1);
                for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                    v_store_aligned(C + d, v_add(vx_load_aligned(C + d), v_mul(vx_load_aligned(hsumAdd + d), v_scale)));
#else
                int scale = k == src_start_idx ? SH2 + 1 : 1;
                for (d = 0; d < D; d++)
                    C[d] = (CostType)(C[d] + hsumAdd[d] * scale);
#endif
                for( x = Da; x < width1*Da; x += Da )
                {
                    const CostType* pixAdd = pixDiff + std::min(x + SW2*Da, (width1-1)*Da);
                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*Da, 0);
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    for (d = 0; d < Da; d += VTraits<v_int16>::vlanes())
                    {
                        v_int16 hv = v_sub(v_add(vx_load_aligned(hsumAdd + x - this->Da + d), vx_load_aligned(pixAdd + d)), vx_load_aligned(pixSub + d));
                        v_store_aligned(hsumAdd + x + d, hv);
                        v_store_aligned(C + x + d, v_add(vx_load_aligned(C + x + d), v_mul(hv, v_scale)));
                    }
#else
                    for (d = 0; d < D; d++)
                    {
                        CostType hv = (CostType)(hsumAdd[x - Da + d] + pixAdd[d] - pixSub[d]);
                        hsumAdd[x + d] = hv;
                        C[x + d] = (CostType)(C[x + d] + hv * scale);
                    }
#endif
                }
            }
        }
        else
        {
            if( y > src_start_idx )
            {
                const CostType* hsumSub = mem.getHSumBuf(std::max(y - SH2 - 1, src_start_idx));
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( x = 0; x < width1*Da; x += VTraits<v_int16>::vlanes())
                    v_store_aligned(C + x, v_sub(v_add(vx_load_aligned(C + x), vx_load_aligned(hsumAdd + x)), vx_load_aligned(hsumSub + x)));
#else
                for( x = 0; x < width1*Da; x++ )
                    C[x] = (CostType)(C[x] + hsumAdd[x] - hsumSub[x]);
#endif
            }
            else
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( x = 0; x < width1*Da; x += VTraits<v_int16>::vlanes())
                    v_store_aligned(C + x, v_add(vx_load_aligned(C + x), vx_load_aligned(hsumAdd + x)));
#else
                for( x = 0; x < width1*Da; x++ )
                    C[x] = (CostType)(C[x] + hsumAdd[x]);
#endif
            }
        }
    }
}

// performing SGM cost accumulation from left to right (result is stored in leftBuf) and
// in-place cost accumulation from top to bottom (result is stored in topBuf)
template<bool x_nlanes>
void SGBM3WayMainLoop::accumulateCostsLeftTop(const BufferSGBM3Way &mem, int x, CostType& leftMinCost) const
{
    CostType *leftBuf = mem.horPassCostVolume + x;
    CostType *leftBuf_prev = mem.horPassCostVolume + x - Da;
    CostType *topBuf = mem.vertPassCostVolume + x;
    CostType *costs = mem.curCostVolumeLine - Da + x;
    CostType& topMinCost = mem.vertPassMin[x/Da];
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_int16 P1_reg = vx_setall_s16(cv::saturate_cast<CostType>(P1));

    v_int16 leftMinCostP2_reg   = vx_setall_s16(cv::saturate_cast<CostType>(leftMinCost+P2));
    v_int16 leftMinCost_new_reg = vx_setall_s16(SHRT_MAX);
    v_int16 src0_leftBuf        = vx_setall_s16(SHRT_MAX);
    v_int16 src1_leftBuf        = vx_load_aligned(leftBuf_prev);

    v_int16 topMinCostP2_reg   = vx_setall_s16(cv::saturate_cast<CostType>(topMinCost+P2));
    v_int16 topMinCost_new_reg = vx_setall_s16(SHRT_MAX);
    v_int16 src0_topBuf        = vx_setall_s16(SHRT_MAX);
    v_int16 src1_topBuf        = vx_load_aligned(topBuf);

    v_int16 src2;
    v_int16 src_shifted_left,src_shifted_right;
    v_int16 res;

    for(;i<Da-VTraits<v_int16>::vlanes();i+= VTraits<v_int16>::vlanes())
    {
        //process leftBuf:
        //lookahead load:
        src2 = vx_load_aligned(leftBuf_prev+i+VTraits<v_int16>::vlanes());

        //get shifted versions of the current block and add P1:
        src_shifted_left  = v_rotate_left<1>  (src1_leftBuf,src0_leftBuf);
        src_shifted_right = v_rotate_right<1> (src1_leftBuf,src2        );

        // process and save current block:
        res = v_add(vx_load_aligned(costs + i), v_sub(v_min(v_add(v_min(src_shifted_left, src_shifted_right), P1_reg), v_min(src1_leftBuf, leftMinCostP2_reg)), leftMinCostP2_reg));
        leftMinCost_new_reg = v_min(leftMinCost_new_reg,res);
        v_store_aligned(leftBuf+i, res);

        //update src buffers:
        src0_leftBuf = src1_leftBuf;
        src1_leftBuf = src2;

        //process topBuf:
        //lookahead load:
        src2 = vx_load_aligned(topBuf+i+VTraits<v_int16>::vlanes());

        //get shifted versions of the current block and add P1:
        src_shifted_left  = v_rotate_left<1>  (src1_topBuf,src0_topBuf);
        src_shifted_right = v_rotate_right<1> (src1_topBuf,src2       );

        // process and save current block:
        res = v_add(vx_load_aligned(costs + i), v_sub(v_min(v_add(v_min(src_shifted_left, src_shifted_right), P1_reg), v_min(src1_topBuf, topMinCostP2_reg)), topMinCostP2_reg));
        topMinCost_new_reg = v_min(topMinCost_new_reg,res);
        v_store_aligned(topBuf+i, res);

        //update src buffers:
        src0_topBuf = src1_topBuf;
        src1_topBuf = src2;
    }

    // a bit different processing for the last cycle of the loop:
    if(x_nlanes)
    {
        src2 = vx_setall_s16(SHRT_MAX);
        //process leftBuf:
        src_shifted_left  = v_rotate_left<1>  (src1_leftBuf,src0_leftBuf);
        src_shifted_right = v_rotate_right<1> (src1_leftBuf,src2        );

        res = v_add(vx_load_aligned(costs + this->Da - VTraits<v_int16>::vlanes()), v_sub(v_min(v_add(v_min(src_shifted_left, src_shifted_right), P1_reg), v_min(src1_leftBuf, leftMinCostP2_reg)), leftMinCostP2_reg));
        leftMinCost = v_reduce_min(v_min(leftMinCost_new_reg,res));
        v_store_aligned(leftBuf+Da-VTraits<v_int16>::vlanes(), res);

        //process topBuf:
        src_shifted_left  = v_rotate_left<1>  (src1_topBuf,src0_topBuf);
        src_shifted_right = v_rotate_right<1> (src1_topBuf,src2       );

        res = v_add(vx_load_aligned(costs + this->Da - VTraits<v_int16>::vlanes()), v_sub(v_min(v_add(v_min(src_shifted_left, src_shifted_right), P1_reg), v_min(src1_topBuf, topMinCostP2_reg)), topMinCostP2_reg));
        topMinCost = v_reduce_min(v_min(topMinCost_new_reg,res));
        v_store_aligned(topBuf+Da-VTraits<v_int16>::vlanes(), res);
    }
    else
    {
        CostType leftMinCost_new = v_reduce_min(leftMinCost_new_reg);
        CostType topMinCost_new  = v_reduce_min(topMinCost_new_reg);
        CostType leftBuf_prev_i_minus_1 = i > 0 ? leftBuf_prev[i-1] : SHRT_MAX;
        CostType topBuf_i_minus_1       = i > 0 ? topBuf[i-1] : SHRT_MAX;
#else
    {
        CostType leftMinCost_new = SHRT_MAX;
        CostType topMinCost_new  = SHRT_MAX;
        CostType leftBuf_prev_i_minus_1 = SHRT_MAX;
        CostType topBuf_i_minus_1       = SHRT_MAX;
#endif
        int leftMinCost_P2  = leftMinCost + P2;
        int topMinCost_P2   = topMinCost  + P2;
        CostType tmp;
        for(;i<D-1;i++)
        {
            leftBuf[i] = cv::saturate_cast<CostType>(costs[i] + std::min(std::min(leftBuf_prev_i_minus_1+P1,leftBuf_prev[i+1]+P1),std::min((int)leftBuf_prev[i],leftMinCost_P2))-leftMinCost_P2);
            leftBuf_prev_i_minus_1 = leftBuf_prev[i];
            leftMinCost_new = std::min(leftMinCost_new,leftBuf[i]);

            tmp = topBuf[i];
            topBuf[i]  = cv::saturate_cast<CostType>(costs[i] + std::min(std::min(topBuf_i_minus_1+P1,topBuf[i+1]+P1),std::min((int)topBuf[i],topMinCost_P2))-topMinCost_P2);
            topBuf_i_minus_1 = tmp;
            topMinCost_new  = std::min(topMinCost_new,topBuf[i]);
        }

        leftBuf[D-1] = cv::saturate_cast<CostType>(costs[D-1] + std::min(leftBuf_prev_i_minus_1+P1,std::min((int)leftBuf_prev[D-1],leftMinCost_P2))-leftMinCost_P2);
        leftMinCost = std::min(leftMinCost_new,leftBuf[D-1]);

        topBuf[D-1]  = cv::saturate_cast<CostType>(costs[D-1] + std::min(topBuf_i_minus_1+P1,std::min((int)topBuf[D-1],topMinCost_P2))-topMinCost_P2);
        topMinCost  = std::min(topMinCost_new,topBuf[D-1]);
    }
}

// performing in-place SGM cost accumulation from right to left (the result is stored in rightBuf) and
// summing rightBuf, topBuf, leftBuf together (the result is stored in leftBuf), as well as finding the
// optimal disparity value with minimum accumulated cost
template<bool x_nlanes>
void SGBM3WayMainLoop::accumulateCostsRight(const BufferSGBM3Way &mem, int x,
                                            CostType& rightMinCost, short& optimal_disp, CostType& min_cost) const
{
    CostType* costs = mem.curCostVolumeLine - Da + x;
    CostType* rightBuf = mem.rightPassBuf;
    CostType* topBuf = mem.vertPassCostVolume + x;
    CostType* leftBuf = mem.horPassCostVolume + x;

    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_int16 P1_reg = vx_setall_s16(cv::saturate_cast<CostType>(P1));

    v_int16 rightMinCostP2_reg   = vx_setall_s16(cv::saturate_cast<CostType>(rightMinCost+P2));
    v_int16 rightMinCost_new_reg = vx_setall_s16(SHRT_MAX);
    v_int16 src0_rightBuf        = vx_setall_s16(SHRT_MAX);
    v_int16 src1_rightBuf        = vx_load(rightBuf);

    v_int16 src2;
    v_int16 src_shifted_left,src_shifted_right;
    v_int16 res;

    v_int16 min_sum_cost_reg = vx_setall_s16(SHRT_MAX);
    v_int16 min_sum_pos_reg  = vx_setall_s16(0);

    for(;i<Da-VTraits<v_int16>::vlanes();i+=VTraits<v_int16>::vlanes())
    {
        //lookahead load:
        src2 = vx_load_aligned(rightBuf+i+VTraits<v_int16>::vlanes());

        //get shifted versions of the current block and add P1:
        src_shifted_left  = v_rotate_left<1>  (src1_rightBuf,src0_rightBuf);
        src_shifted_right = v_rotate_right<1> (src1_rightBuf,src2         );

        // process and save current block:
        res = v_add(vx_load_aligned(costs + i), v_sub(v_min(v_add(v_min(src_shifted_left, src_shifted_right), P1_reg), v_min(src1_rightBuf, rightMinCostP2_reg)), rightMinCostP2_reg));
        rightMinCost_new_reg = v_min(rightMinCost_new_reg,res);
        v_store_aligned(rightBuf+i, res);

        // compute and save total cost:
        res = v_add(v_add(res, vx_load_aligned(leftBuf + i)), vx_load_aligned(topBuf + i));
        v_store_aligned(leftBuf+i, res);

        // track disparity value with the minimum cost:
        min_sum_cost_reg = v_min(min_sum_cost_reg,res);
        min_sum_pos_reg = v_add(min_sum_pos_reg, v_and(v_eq(min_sum_cost_reg, res), v_sub(vx_setall_s16((short)i), min_sum_pos_reg)));

        //update src:
        src0_rightBuf    = src1_rightBuf;
        src1_rightBuf    = src2;
    }

    // a bit different processing for the last cycle of the loop:
    if(x_nlanes)
    {
        src2 = vx_setall_s16(SHRT_MAX);
        src_shifted_left  = v_rotate_left<1>  (src1_rightBuf,src0_rightBuf);
        src_shifted_right = v_rotate_right<1> (src1_rightBuf,src2         );

        res = v_add(vx_load_aligned(costs + this->D - VTraits<v_int16>::vlanes()), v_sub(v_min(v_add(v_min(src_shifted_left, src_shifted_right), P1_reg), v_min(src1_rightBuf, rightMinCostP2_reg)), rightMinCostP2_reg));
        rightMinCost = v_reduce_min(v_min(rightMinCost_new_reg,res));
        v_store_aligned(rightBuf+D-VTraits<v_int16>::vlanes(), res);

        res = v_add(v_add(res, vx_load_aligned(leftBuf + this->D - VTraits<v_int16>::vlanes())), vx_load_aligned(topBuf + this->D - VTraits<v_int16>::vlanes()));
        v_store_aligned(leftBuf+D-VTraits<v_int16>::vlanes(), res);

        min_sum_cost_reg = v_min(min_sum_cost_reg,res);
        min_sum_pos_reg = v_add(min_sum_pos_reg, v_and(v_eq(min_sum_cost_reg, res), v_sub(vx_setall_s16((short)(this->D - VTraits<v_int16>::vlanes())), min_sum_pos_reg)));
        min_pos(min_sum_cost_reg,min_sum_pos_reg, min_cost, optimal_disp);
    }
    else
    {
        CostType rightMinCost_new = v_reduce_min(rightMinCost_new_reg);
        CostType rightBuf_i_minus_1 = i > 0 ? rightBuf[i] : SHRT_MAX;
        min_pos(min_sum_cost_reg,min_sum_pos_reg, min_cost, optimal_disp);
#else
    {
        CostType rightMinCost_new = SHRT_MAX;
        CostType rightBuf_i_minus_1 = SHRT_MAX;
        min_cost = SHRT_MAX;
#endif
        int rightMinCost_P2  = rightMinCost + P2;
        CostType tmp;
        for(;i<D-1;i++)
        {
            tmp = rightBuf[i];
            rightBuf[i]  = cv::saturate_cast<CostType>(costs[i] + std::min(std::min(rightBuf_i_minus_1+P1,rightBuf[i+1]+P1),std::min((int)rightBuf[i],rightMinCost_P2))-rightMinCost_P2);
            rightBuf_i_minus_1 = tmp;
            rightMinCost_new  = std::min(rightMinCost_new,rightBuf[i]);
            leftBuf[i] = cv::saturate_cast<CostType>((int)leftBuf[i]+rightBuf[i]+topBuf[i]);
            if(leftBuf[i]<min_cost)
            {
                optimal_disp = (short)i;
                min_cost = leftBuf[i];
            }
        }

        rightBuf[D-1]  = cv::saturate_cast<CostType>(costs[D-1] + std::min(rightBuf_i_minus_1+P1,std::min((int)rightBuf[D-1],rightMinCost_P2))-rightMinCost_P2);
        rightMinCost  = std::min(rightMinCost_new,rightBuf[D-1]);
        leftBuf[D-1] = cv::saturate_cast<CostType>((int)leftBuf[D-1]+rightBuf[D-1]+topBuf[D-1]);
        if(leftBuf[D-1]<min_cost)
        {
            optimal_disp = (short)D-1;
            min_cost = leftBuf[D-1];
        }
    }
}

void SGBM3WayMainLoop::operator () (const Range& range) const
{
    if (D == Da) impl<true>(range);
    else impl<false>(range);
}

template<bool x_nlanes>
void SGBM3WayMainLoop::impl(const Range& range) const
{
    // force separate processing of stripes:
    if(range.end>range.start+1)
    {
        for(int n=range.start;n<range.end;n++)
            (*this)(Range(n,n+1));
        return;
    }

    const int DISP_SCALE = (1 << StereoMatcher::DISP_SHIFT);
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;

    // setting up the ranges:
    int src_start_idx = std::max(std::min(range.start * stripe_sz - stripe_overlap, height),0);
    int src_end_idx   = std::min(range.end   * stripe_sz, height);

    int dst_offset;
    if(range.start==0)
        dst_offset=stripe_overlap;
    else
        dst_offset=0;

    Mat cur_disp   = dst_disp[range.start];
    cur_disp = Scalar(INVALID_DISP_SCALED);

    BufferSGBM3Way mem(width1, width, img1->channels(), Da, SH2, P2);
    CostType *horPassCostVolume = mem.horPassCostVolume;
    // start real processing:
    for(int y=src_start_idx;y<src_end_idx;y++)
    {
        getRawMatchingCost(mem, y, src_start_idx);

        short* disp_row = (short*)cur_disp.ptr(dst_offset+(y-src_start_idx));

        // initialize the auxiliary buffers for the pseudo left-right consistency check:
        for(int x=0;x<width;x++)
        {
            mem.disp2Buf[x] = (short)INVALID_DISP_SCALED;
            mem.disp2CostBuf[x] = SHRT_MAX;
        }
        CostType prev_min, min_cost;
        int d;
        short best_d;
        d = best_d = 0;

        // forward pass
        prev_min=0;
        for (int x=Da;x<(1+width1)*Da;x+=Da)
            accumulateCostsLeftTop<x_nlanes>(mem, x, prev_min);

        //backward pass
        mem.clearRightPassBuf();
        prev_min=0;
        for (int x=width1*Da;x>=Da;x-=Da)
        {
            accumulateCostsRight<x_nlanes>(mem, x, prev_min, best_d, min_cost);

            if(uniquenessRatio>0)
            {
                d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                horPassCostVolume+=x;
                int thresh = (100*min_cost)/(100-uniquenessRatio);
                v_int16 thresh_reg = vx_setall_s16((short)(thresh+1));
                v_int16 d1 = vx_setall_s16((short)(best_d-1));
                v_int16 d2 = vx_setall_s16((short)(best_d+1));
                v_int16 eight_reg = vx_setall_s16((short)VTraits<v_int16>::vlanes());
                v_int16 cur_d = vx_load(idx_row);
                v_int16 mask;

                for( ; d <= D - 2*VTraits<v_int16>::vlanes(); d+=2*VTraits<v_int16>::vlanes() )
                {
                    mask = v_and(v_lt(vx_load_aligned(horPassCostVolume + d), thresh_reg), v_or(v_lt(cur_d, d1), v_gt(cur_d, d2)));
                    cur_d = v_add(cur_d, eight_reg);
                    if( v_check_any(mask) )
                        break;
                    mask = v_and(v_lt(vx_load_aligned(horPassCostVolume + d + VTraits<v_int16>::vlanes()), thresh_reg), v_or(v_lt(cur_d, d1), v_gt(cur_d, d2)));
                    cur_d = v_add(cur_d, eight_reg);
                    if( v_check_any(mask) )
                        break;
                }
                if( d <= D - 2*VTraits<v_int16>::vlanes() )
                {
                    horPassCostVolume-=x;
                    continue;
                }
                if( d <= D - VTraits<v_int16>::vlanes() )
                {
                    if( v_check_any(v_and(v_lt(vx_load_aligned(horPassCostVolume + d), thresh_reg), v_or(v_lt(cur_d, d1), v_gt(cur_d, d2)))) )
                    {
                        horPassCostVolume-=x;
                        continue;
                    }
                    d+=VTraits<v_int16>::vlanes();
                }
                horPassCostVolume-=x;
#endif
                for( ; d < D; d++ )
                {
                    if( horPassCostVolume[x+d]*(100 - uniquenessRatio) < min_cost*100 && std::abs(d - best_d) > 1 )
                        break;
                }
                if( d < D )
                    continue;
            }
            d = best_d;

            int _x2 = x/Da - 1 + minX1 - d - minD;
            if( _x2>=0 && _x2<width && mem.disp2CostBuf[_x2] > min_cost )
            {
                mem.disp2CostBuf[_x2] = min_cost;
                mem.disp2Buf[_x2] = (short)(d + minD);
            }

            if( 0 < d && d < D-1 )
            {
                // do subpixel quadratic interpolation:
                //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
                //   then find minimum of the parabola.
                int denom2 = std::max(horPassCostVolume[x+d-1] + horPassCostVolume[x+d+1] - 2*horPassCostVolume[x+d], 1);
                d = d*DISP_SCALE + ((horPassCostVolume[x+d-1] - horPassCostVolume[x+d+1])*DISP_SCALE + denom2)/(denom2*2);
            }
            else
                d *= DISP_SCALE;

            disp_row[(x/Da)-1 + minX1] = (DispType)(d + minD*DISP_SCALE);
        }

        for(int x = minX1; x < maxX1; x++ )
        {
            // pseudo LRC consistency check using only one disparity map;
            // pixels with difference more than disp12MaxDiff are invalidated
            int d1 = disp_row[x];
            if( d1 == INVALID_DISP_SCALED )
                continue;
            int _d = d1 >> StereoMatcher::DISP_SHIFT;
            int d_ = (d1 + DISP_SCALE-1) >> StereoMatcher::DISP_SHIFT;
            int _x = x - _d, x_ = x - d_;
            if( 0 <= _x && _x < width && mem.disp2Buf[_x] >= minD && std::abs(mem.disp2Buf[_x] - _d) > disp12MaxDiff &&
                0 <= x_ && x_ < width && mem.disp2Buf[x_] >= minD && std::abs(mem.disp2Buf[x_] - d_) > disp12MaxDiff )
                disp_row[x] = (short)INVALID_DISP_SCALED;
        }
    }
}

template <uchar nstripes>
static void computeDisparity3WaySGBM(const Mat& img1, const Mat& img2, Mat& disp1, const StereoSGBMParams& params)
{
    // allocate separate dst_disp arrays to avoid conflicts due to stripe overlap:
    int stripe_sz = (int)ceil(img1.rows/(double)nstripes);
    int stripe_overlap = (params.SADWindowSize/2+1) + (int)ceil(0.1*stripe_sz);
    Mat dst_disp[nstripes];
    for(int i=0;i<nstripes;i++)
        dst_disp[i].create(stripe_sz+stripe_overlap,img1.cols,CV_16S);

    parallel_for_(
        Range(0,nstripes),
        SGBM3WayMainLoop(img1,img2,dst_disp,params,stripe_sz,stripe_overlap)
    );

    //assemble disp1 from dst_disp:
    short* dst_row;
    short* src_row;
    for(int i=0;i<disp1.rows;i++)
    {
        dst_row = (short*)disp1.ptr(i);
        src_row = (short*)dst_disp[i/stripe_sz].ptr(stripe_overlap+i%stripe_sz);
        memcpy(dst_row,src_row,disp1.cols*sizeof(short));
    }
}

class StereoSGBMImpl CV_FINAL : public StereoSGBM
{
public:
    StereoSGBMImpl()
        : params()
    {
        // nothing
    }

    StereoSGBMImpl( int _minDisparity, int _numDisparities, int _SADWindowSize,
                    int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                    int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                    int _mode )
        : params(_minDisparity, _numDisparities, _SADWindowSize,
                 _P1, _P2, _disp12MaxDiff, _preFilterCap,
                 _uniquenessRatio, _speckleWindowSize, _speckleRange,
                 _mode)
    {
        // nothing
    }

    void compute( InputArray leftarr, InputArray rightarr, OutputArray disparr ) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        Mat left = leftarr.getMat(), right = rightarr.getMat();
        CV_Assert( left.size() == right.size() && left.type() == right.type() &&
                   left.depth() == CV_8U );

        disparr.create( left.size(), CV_16S );
        Mat disp = disparr.getMat();

        if(params.mode==MODE_SGBM_3WAY)
            // the number of stripes is fixed, disregarding the number of threads/processors
            // to make the results fully reproducible
            computeDisparity3WaySGBM<4>( left, right, disp, params );
        else if(params.mode==MODE_HH4)
            computeDisparitySGBM_HH4( left, right, disp, params );
        else
            computeDisparitySGBM( left, right, disp, params );

        medianBlur(disp, disp, 3);

        if( params.speckleWindowSize > 0 )
            filterSpeckles(disp, (params.minDisparity - 1)*StereoMatcher::DISP_SCALE, params.speckleWindowSize,
                           StereoMatcher::DISP_SCALE*params.speckleRange, buffer);
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

    int getPreFilterCap() const CV_OVERRIDE { return params.preFilterCap; }
    void setPreFilterCap(int preFilterCap) CV_OVERRIDE { params.preFilterCap = preFilterCap; }

    int getUniquenessRatio() const CV_OVERRIDE { return params.uniquenessRatio; }
    void setUniquenessRatio(int uniquenessRatio) CV_OVERRIDE { params.uniquenessRatio = uniquenessRatio; }

    int getP1() const CV_OVERRIDE { return params.P1; }
    void setP1(int P1) CV_OVERRIDE { params.P1 = P1; }

    int getP2() const CV_OVERRIDE { return params.P2; }
    void setP2(int P2) CV_OVERRIDE { params.P2 = P2; }

    int getMode() const CV_OVERRIDE { return params.mode; }
    void setMode(int mode) CV_OVERRIDE { params.mode = mode; }

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
        << "preFilterCap" << params.preFilterCap
        << "uniquenessRatio" << params.uniquenessRatio
        << "P1" << params.P1
        << "P2" << params.P2
        << "mode" << params.mode;
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
        params.preFilterCap = (int)fn["preFilterCap"];
        params.uniquenessRatio = (int)fn["uniquenessRatio"];
        params.P1 = (int)fn["P1"];
        params.P2 = (int)fn["P2"];
        params.mode = (int)fn["mode"];
    }

    StereoSGBMParams params;
    Mat buffer;

    static const char* name_;
};

const char* StereoSGBMImpl::name_ = "StereoMatcher.SGBM";


Ptr<StereoSGBM> StereoSGBM::create(int minDisparity, int numDisparities, int SADWindowSize,
                                 int P1, int P2, int disp12MaxDiff,
                                 int preFilterCap, int uniquenessRatio,
                                 int speckleWindowSize, int speckleRange,
                                 int mode)
{
    return Ptr<StereoSGBM>(
        new StereoSGBMImpl(minDisparity, numDisparities, SADWindowSize,
                           P1, P2, disp12MaxDiff,
                           preFilterCap, uniquenessRatio,
                           speckleWindowSize, speckleRange,
                           mode));
}

Rect getValidDisparityROI( Rect roi1, Rect roi2,
                          int minDisparity,
                          int numberOfDisparities,
                          int SADWindowSize )
{
    int SW2 = SADWindowSize/2;
    int maxD = minDisparity + numberOfDisparities - 1;

    int xmin = std::max(roi1.x, roi2.x + maxD) + SW2;
    int xmax = std::min(roi1.x + roi1.width, roi2.x + roi2.width) - SW2;
    int ymin = std::max(roi1.y, roi2.y) + SW2;
    int ymax = std::min(roi1.y + roi1.height, roi2.y + roi2.height) - SW2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);

    return r.width > 0 && r.height > 0 ? r : Rect();
}

typedef cv::Point_<short> Point2s;

template <typename T>
void filterSpecklesImpl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf)
{
    using namespace cv;

    int width = img.cols, height = img.rows, npixels = width*height;
    // space allocation for:
    // labels : npixels * sizeof(int)
    // wbuf   : npixels * sizeof(Point2s)
    // rtype  : (npixels + 1) * sizeof(uchar)
    size_t bufSize = npixels * sizeof(int)            // for labels
                     + npixels * sizeof(Point2s)      // for wavefront buffer (wbuf)
                     + (npixels + 1) * sizeof(uchar); // for region type (rtype)
    if( !_buf.isContinuous() || _buf.empty() || _buf.cols*_buf.rows*_buf.elemSize() < bufSize )
        _buf.reserveBuffer(bufSize);

    uchar* buf = _buf.ptr();
    int i, j, dstep = (int)(img.step/sizeof(T));

    // store labels and their corresponding region types
    int* labels = (int*)buf;
    buf += npixels*sizeof(labels[0]);
    // store wavefront
    Point2s* wbuf = (Point2s*)buf;
    buf += npixels*sizeof(wbuf[0]);
    // store region type
    uchar* rtype = (uchar*)buf;
    buf += (npixels + 1) * sizeof(rtype[0]);
    int curlabel = 0;

    // clear out label and rtype buffers
    memset(labels, 0, npixels*sizeof(labels[0]));
    memset(rtype, 0, (npixels + 1)*sizeof(rtype[0]));

    for( i = 0; i < height; i++ )
    {
        T* ds = img.ptr<T>(i);
        int* ls = labels + width*i;

        for( j = 0; j < width; j++ )
        {
            if( ds[j] != newVal )   // not a bad disparity
            {
                if( ls[j] )     // has a label, check for bad label
                {
                    if( rtype[ls[j]] ) // small region, zero out disparity
                        ds[j] = (T)newVal;
                }
                // no label, assign and propagate
                else
                {
                    Point2s* ws = wbuf; // initialize wavefront
                    Point2s p((short)j, (short)i);  // current pixel
                    curlabel++; // next label
                    int count = 0;  // current region size
                    ls[j] = curlabel;

                    // wavefront propagation
                    while( ws >= wbuf ) // wavefront not empty
                    {
                        count++;
                        // put neighbors onto wavefront
                        T* dpp = &img.at<T>(p.y, p.x);
                        T dp = *dpp;
                        int* lpp = labels + width*p.y + p.x;

                        if( p.y < height-1 && !lpp[+width] && dpp[+dstep] != newVal && std::abs(dp - dpp[+dstep]) <= maxDiff )
                        {
                            lpp[+width] = curlabel;
                            *ws++ = Point2s(p.x, p.y+1);
                        }

                        if( p.y > 0 && !lpp[-width] && dpp[-dstep] != newVal && std::abs(dp - dpp[-dstep]) <= maxDiff )
                        {
                            lpp[-width] = curlabel;
                            *ws++ = Point2s(p.x, p.y-1);
                        }

                        if( p.x < width-1 && !lpp[+1] && dpp[+1] != newVal && std::abs(dp - dpp[+1]) <= maxDiff )
                        {
                            lpp[+1] = curlabel;
                            *ws++ = Point2s(p.x+1, p.y);
                        }

                        if( p.x > 0 && !lpp[-1] && dpp[-1] != newVal && std::abs(dp - dpp[-1]) <= maxDiff )
                        {
                            lpp[-1] = curlabel;
                            *ws++ = Point2s(p.x-1, p.y);
                        }

                        // pop most recent and propagate
                        // NB: could try least recent, maybe better convergence
                        p = *--ws;
                    }

                    // assign label type
                    if( count <= maxSpeckleSize )   // speckle region
                    {
                        rtype[ls[j]] = 1;   // small region label
                        ds[j] = (T)newVal;
                    }
                    else
                        rtype[ls[j]] = 0;   // large region label
                }
            }
        }
    }
}

#ifdef HAVE_IPP
static bool ipp_filterSpeckles(Mat &img, int maxSpeckleSize, int newVal, int maxDiff, Mat &buffer)
{
#if IPP_VERSION_X100 >= 810
    CV_INSTRUMENT_REGION_IPP();

    IppDataType dataType = ippiGetDataType(img.depth());
    IppiSize    size     = ippiSize(img.size());
    int         bufferSize;

    if(img.channels() != 1)
        return false;

    if(dataType != ipp8u && dataType != ipp16s)
        return false;

    if(ippiMarkSpecklesGetBufferSize(size, dataType, 1, &bufferSize) < 0)
        return false;

    if(bufferSize && (buffer.empty() || (int)(buffer.step*buffer.rows) < bufferSize))
        buffer.create(1, (int)bufferSize, CV_8U);

    switch(dataType)
    {
    case ipp8u:  return CV_INSTRUMENT_FUN_IPP(ippiMarkSpeckles_8u_C1IR, img.ptr<Ipp8u>(), (int)img.step, size, (Ipp8u)newVal, maxSpeckleSize, (Ipp8u)maxDiff, ippiNormL1, buffer.ptr<Ipp8u>()) >= 0;
    case ipp16s: return CV_INSTRUMENT_FUN_IPP(ippiMarkSpeckles_16s_C1IR, img.ptr<Ipp16s>(), (int)img.step, size, (Ipp16s)newVal, maxSpeckleSize, (Ipp16s)maxDiff, ippiNormL1, buffer.ptr<Ipp8u>()) >= 0;
    default:     return false;
    }
#else
    CV_UNUSED(img); CV_UNUSED(maxSpeckleSize); CV_UNUSED(newVal); CV_UNUSED(maxDiff); CV_UNUSED(buffer);
    return false;
#endif
}
#endif

}

void cv::filterSpeckles( InputOutputArray _img, double _newval, int maxSpeckleSize,
                         double _maxDiff, InputOutputArray __buf )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();
    int type = img.type();
    Mat temp, &_buf = __buf.needed() ? __buf.getMatRef() : temp;
    CV_Assert( type == CV_8UC1 || type == CV_16SC1 );

    int newVal = cvRound(_newval), maxDiff = cvRound(_maxDiff);

    CV_IPP_RUN_FAST(ipp_filterSpeckles(img, maxSpeckleSize, newVal, maxDiff, _buf));

    if (type == CV_8UC1)
        filterSpecklesImpl<uchar>(img, newVal, maxSpeckleSize, maxDiff, _buf);
    else
        filterSpecklesImpl<short>(img, newVal, maxSpeckleSize, maxDiff, _buf);
}

void cv::validateDisparity( InputOutputArray _disp, InputArray _cost, int minDisparity,
                            int numberOfDisparities, int disp12MaxDiff )
{
    CV_INSTRUMENT_REGION();

    Mat disp = _disp.getMat(), cost = _cost.getMat();
    int cols = disp.cols, rows = disp.rows;
    int minD = minDisparity, maxD = minDisparity + numberOfDisparities;
    int x, minX1 = std::max(maxD, 0), maxX1 = cols + std::min(minD, 0);
    AutoBuffer<int> _disp2buf(cols*2);
    int* disp2buf = _disp2buf.data();
    int* disp2cost = disp2buf + cols;
    const int DISP_SHIFT = 4, DISP_SCALE = 1 << DISP_SHIFT;
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
    int costType = cost.type();

    disp12MaxDiff *= DISP_SCALE;

    CV_Assert( numberOfDisparities > 0 && disp.type() == CV_16S &&
              (costType == CV_16S || costType == CV_32S) &&
              disp.size() == cost.size() );

    for( int y = 0; y < rows; y++ )
    {
        short* dptr = disp.ptr<short>(y);

        for( x = 0; x < cols; x++ )
        {
            disp2buf[x] = INVALID_DISP_SCALED;
            disp2cost[x] = INT_MAX;
        }

        if( costType == CV_16S )
        {
            const short* cptr = cost.ptr<short>(y);

            for( x = minX1; x < maxX1; x++ )
            {
                int d = dptr[x], c = cptr[x];

                if( d == INVALID_DISP_SCALED )
                    continue;

                int x2 = x - ((d + DISP_SCALE/2) >> DISP_SHIFT);

                if( disp2cost[x2] > c )
                {
                    disp2cost[x2] = c;
                    disp2buf[x2] = d;
                }
            }
        }
        else
        {
            const int* cptr = cost.ptr<int>(y);

            for( x = minX1; x < maxX1; x++ )
            {
                int d = dptr[x], c = cptr[x];

                if( d == INVALID_DISP_SCALED )
                    continue;

                int x2 = x - ((d + DISP_SCALE/2) >> DISP_SHIFT);

                if( disp2cost[x2] > c )
                {
                    disp2cost[x2] = c;
                    disp2buf[x2] = d;
                }
            }
        }

        for( x = minX1; x < maxX1; x++ )
        {
            // we round the computed disparity both towards -inf and +inf and check
            // if either of the corresponding disparities in disp2 is consistent.
            // This is to give the computed disparity a chance to look valid if it is.
            int d = dptr[x];
            if( d == INVALID_DISP_SCALED )
                continue;
            int d0 = d >> DISP_SHIFT;
            int d1 = (d + DISP_SCALE-1) >> DISP_SHIFT;
            int x0 = x - d0, x1 = x - d1;
            if( (0 <= x0 && x0 < cols && disp2buf[x0] > INVALID_DISP_SCALED && std::abs(disp2buf[x0] - d) > disp12MaxDiff) &&
                (0 <= x1 && x1 < cols && disp2buf[x1] > INVALID_DISP_SCALED && std::abs(disp2buf[x1] - d) > disp12MaxDiff) )
                dptr[x] = (short)INVALID_DISP_SCALED;
        }
    }
}
