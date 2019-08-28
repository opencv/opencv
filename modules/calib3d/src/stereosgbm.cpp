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

namespace cv
{

typedef uchar PixType;
typedef short CostType;
typedef short DispType;

enum { NR = 16, NR2 = NR/2 };


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
                            int tabOfs, int , int xrange_min = 0, int xrange_max = DEFAULT_RIGHT_BORDER )
{
    int x, c, width = img1.cols, cn = img1.channels();
    int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
    int D = maxD - minD, width1 = maxX1 - minX1;
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

    tab += tabOfs;

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

    buffer -= width-1-maxX2;
    cost -= (minX1-xrange_min)*D + minD; // simplify the cost indices inside the loop

    for( c = 0; c < cn*2; c++, prow1 += width, prow2 += width )
    {
        int diff_scale = c < cn ? 0 : 2;

        // precompute
        //   v0 = min(row2[x-1/2], row2[x], row2[x+1/2]) and
        //   v1 = max(row2[x-1/2], row2[x], row2[x+1/2]) and
        for( x = width-1-maxX2; x < width-1- minX2; x++ )
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

        #if CV_SIMD128
            if (true)
            {
                v_uint8x16 _u  = v_setall_u8((uchar)u), _u0 = v_setall_u8((uchar)u0);
                v_uint8x16 _u1 = v_setall_u8((uchar)u1);

                for( int d = minD; d < maxD; d += 16 )
                {
                    v_uint8x16 _v  = v_load(prow2  + width-x-1 + d);
                    v_uint8x16 _v0 = v_load(buffer + width-x-1 + d);
                    v_uint8x16 _v1 = v_load(buffer + width-x-1 + d + width2);
                    v_uint8x16 c0 = v_max(_u - _v1, _v0 - _u);
                    v_uint8x16 c1 = v_max(_v - _u1, _u0 - _v);
                    v_uint8x16 diff = v_min(c0, c1);

                    v_int16x8 _c0 = v_load_aligned(cost + x*D + d);
                    v_int16x8 _c1 = v_load_aligned(cost + x*D + d + 8);

                    v_uint16x8 diff1,diff2;
                    v_expand(diff,diff1,diff2);
                    v_store_aligned(cost + x*D + d,     _c0 + v_reinterpret_as_s16(diff1 >> diff_scale));
                    v_store_aligned(cost + x*D + d + 8, _c1 + v_reinterpret_as_s16(diff2 >> diff_scale));
                }
            }
            else
        #endif
            {
                for( int d = minD; d < maxD; d++ )
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
}


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
                                 Mat& disp1, const StereoSGBMParams& params,
                                 Mat& buffer )
{
#if CV_SIMD128
    // maxDisparity is supposed to multiple of 16, so we can forget doing else
    static const uchar LSBTab[] =
    {
        0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
    };
    static const v_uint16x8 v_LSB = v_uint16x8(0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);
#endif

    const int ALIGN = 16;
    const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
    const int DISP_SCALE = (1 << DISP_SHIFT);
    const CostType MAX_COST = SHRT_MAX;

    int minD = params.minDisparity, maxD = minD + params.numDisparities;
    Size SADWindowSize;
    SADWindowSize.width = SADWindowSize.height = params.SADWindowSize > 0 ? params.SADWindowSize : 5;
    int ftzero = std::max(params.preFilterCap, 15) | 1;
    int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
    int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
    int P1 = params.P1 > 0 ? params.P1 : 2, P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
    int k, width = disp1.cols, height = disp1.rows;
    int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
    int D = maxD - minD, width1 = maxX1 - minX1;
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
    int SW2 = SADWindowSize.width/2, SH2 = SADWindowSize.height/2;
    bool fullDP = params.mode == StereoSGBM::MODE_HH;
    int npasses = fullDP ? 2 : 1;
    const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2;
    PixType clipTab[TAB_SIZE];

    for( k = 0; k < TAB_SIZE; k++ )
        clipTab[k] = (PixType)(std::min(std::max(k - TAB_OFS, -ftzero), ftzero) + ftzero);

    if( minX1 >= maxX1 )
    {
        disp1 = Scalar::all(INVALID_DISP_SCALED);
        return;
    }

    CV_Assert( D % 16 == 0 );

    // NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
    // if you change NR, please, modify the loop as well.
    int D2 = D+16, NRD2 = NR2*D2;

    // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
    // for 8-way dynamic programming we need the current row and
    // the previous row, i.e. 2 rows in total
    const int NLR = 2;
    const int LrBorder = NLR - 1;

    // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
    // we keep pixel difference cost (C) and the summary cost over NR directions (S).
    // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
    size_t costBufSize = width1*D;
    size_t CSBufSize = costBufSize*(fullDP ? height : 1);
    size_t minLrSize = (width1 + LrBorder*2)*NR2, LrSize = minLrSize*D2;
    int hsumBufNRows = SH2*2 + 2;
    size_t totalBufSize = (LrSize + minLrSize)*NLR*sizeof(CostType) + // minLr[] and Lr[]
    costBufSize*(hsumBufNRows + 1)*sizeof(CostType) + // hsumBuf, pixdiff
    CSBufSize*2*sizeof(CostType) + // C, S
    width*16*img1.channels()*sizeof(PixType) + // temp buffer for computing per-pixel cost
    width*(sizeof(CostType) + sizeof(DispType)) + 1024; // disp2cost + disp2

    if( buffer.empty() || !buffer.isContinuous() ||
        buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize )
        buffer.reserveBuffer(totalBufSize);

    // summary cost over different (nDirs) directions
    CostType* Cbuf = (CostType*)alignPtr(buffer.ptr(), ALIGN);
    CostType* Sbuf = Cbuf + CSBufSize;
    CostType* hsumBuf = Sbuf + CSBufSize;
    CostType* pixDiff = hsumBuf + costBufSize*hsumBufNRows;

    CostType* disp2cost = pixDiff + costBufSize + (LrSize + minLrSize)*NLR;
    DispType* disp2ptr = (DispType*)(disp2cost + width);
    PixType* tempBuf = (PixType*)(disp2ptr + width);

    // add P2 to every C(x,y). it saves a few operations in the inner loops
    for(k = 0; k < (int)CSBufSize; k++ )
        Cbuf[k] = (CostType)P2;

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

        CostType *Lr[NLR]={0}, *minLr[NLR]={0};

        for( k = 0; k < NLR; k++ )
        {
            // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
            // and will occasionally use negative indices with the arrays
            // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
            // however, then the alignment will be imperfect, i.e. bad for SSE,
            // thus we shift the pointers by 8 (8*sizeof(short) == 16 - ideal alignment)
            Lr[k] = pixDiff + costBufSize + LrSize*k + NRD2*LrBorder + 8;
            memset( Lr[k] - LrBorder*NRD2 - 8, 0, LrSize*sizeof(CostType) );
            minLr[k] = pixDiff + costBufSize + LrSize*NLR + minLrSize*k + NR2*LrBorder;
            memset( minLr[k] - LrBorder*NR2, 0, minLrSize*sizeof(CostType) );
        }

        for( int y = y1; y != y2; y += dy )
        {
            int x, d;
            DispType* disp1ptr = disp1.ptr<DispType>(y);
            CostType* C = Cbuf + (!fullDP ? 0 : y*costBufSize);
            CostType* S = Sbuf + (!fullDP ? 0 : y*costBufSize);

            if( pass == 1 ) // compute C on the first pass, and reuse it on the second pass, if any.
            {
                int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

                for( k = dy1; k <= dy2; k++ )
                {
                    CostType* hsumAdd = hsumBuf + (std::min(k, height-1) % hsumBufNRows)*costBufSize;

                    if( k < height )
                    {
                        calcPixelCostBT( img1, img2, k, minD, maxD, pixDiff, tempBuf, clipTab, TAB_OFS, ftzero );

                        memset(hsumAdd, 0, D*sizeof(CostType));
                        for( x = 0; x <= SW2*D; x += D )
                        {
                            int scale = x == 0 ? SW2 + 1 : 1;
                            for( d = 0; d < D; d++ )
                                hsumAdd[d] = (CostType)(hsumAdd[d] + pixDiff[x + d]*scale);
                        }

                        if( y > 0 )
                        {
                            const CostType* hsumSub = hsumBuf + (std::max(y - SH2 - 1, 0) % hsumBufNRows)*costBufSize;
                            const CostType* Cprev = !fullDP || y == 0 ? C : C - costBufSize;

                            for( x = D; x < width1*D; x += D )
                            {
                                const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

                            #if CV_SIMD128
                                if (true)
                                {
                                    for( d = 0; d < D; d += 8 )
                                    {
                                        v_int16x8 hv = v_load(hsumAdd + x - D + d);
                                        v_int16x8 Cx = v_load(Cprev + x + d);
                                        v_int16x8 psub = v_load(pixSub + d);
                                        v_int16x8 padd = v_load(pixAdd + d);
                                        hv = (hv - psub + padd);
                                        psub = v_load(hsumSub + x + d);
                                        Cx = Cx - psub + hv;
                                        v_store(hsumAdd + x + d, hv);
                                        v_store(C + x + d, Cx);
                                    }
                                }
                                else
                            #endif
                                {
                                    for( d = 0; d < D; d++ )
                                    {
                                        int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                                        C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
                                    }
                                }
                            }
                        }
                        else
                        {
                            for( x = D; x < width1*D; x += D )
                            {
                                const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

                                for( d = 0; d < D; d++ )
                                    hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                            }
                        }
                    }

                    if( y == 0 )
                    {
                        int scale = k == 0 ? SH2 + 1 : 1;
                        for( x = 0; x < width1*D; x++ )
                            C[x] = (CostType)(C[x] + hsumAdd[x]*scale);
                    }
                }

                // also, clear the S buffer
                for( k = 0; k < width1*D; k++ )
                    S[k] = 0;
            }

            // clear the left and the right borders
            memset( Lr[0] - NRD2*LrBorder - 8, 0, NRD2*LrBorder*sizeof(CostType) );
            memset( Lr[0] + width1*NRD2 - 8, 0, NRD2*LrBorder*sizeof(CostType) );
            memset( minLr[0] - NR2*LrBorder, 0, NR2*LrBorder*sizeof(CostType) );
            memset( minLr[0] + width1*NR2, 0, NR2*LrBorder*sizeof(CostType) );

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
             3: r=(1, -dy)
             4: r=(-2, -dy)
             5: r=(-1, -dy*2)
             6: r=(1, -dy*2)
             7: r=(2, -dy)
             */

            for( x = x1; x != x2; x += dx )
            {
                int xm = x*NR2, xd = xm*D2;

                int delta0 = minLr[0][xm - dx*NR2] + P2, delta1 = minLr[1][xm - NR2 + 1] + P2;
                int delta2 = minLr[1][xm + 2] + P2, delta3 = minLr[1][xm + NR2 + 3] + P2;

                CostType* Lr_p0 = Lr[0] + xd - dx*NRD2;
                CostType* Lr_p1 = Lr[1] + xd - NRD2 + D2;
                CostType* Lr_p2 = Lr[1] + xd + D2*2;
                CostType* Lr_p3 = Lr[1] + xd + NRD2 + D2*3;

                Lr_p0[-1] = Lr_p0[D] = Lr_p1[-1] = Lr_p1[D] =
                Lr_p2[-1] = Lr_p2[D] = Lr_p3[-1] = Lr_p3[D] = MAX_COST;

                CostType* Lr_p = Lr[0] + xd;
                const CostType* Cp = C + x*D;
                CostType* Sp = S + x*D;

            #if CV_SIMD128
                if (true)
                {
                    v_int16x8 _P1 = v_setall_s16((short)P1);

                    v_int16x8 _delta0 = v_setall_s16((short)delta0);
                    v_int16x8 _delta1 = v_setall_s16((short)delta1);
                    v_int16x8 _delta2 = v_setall_s16((short)delta2);
                    v_int16x8 _delta3 = v_setall_s16((short)delta3);
                    v_int16x8 _minL0 = v_setall_s16((short)MAX_COST);

                    for( d = 0; d < D; d += 8 )
                    {
                        v_int16x8 Cpd = v_load(Cp + d);
                        v_int16x8 L0, L1, L2, L3;

                        L0 = v_load(Lr_p0 + d);
                        L1 = v_load(Lr_p1 + d);
                        L2 = v_load(Lr_p2 + d);
                        L3 = v_load(Lr_p3 + d);

                        L0 = v_min(L0, (v_load(Lr_p0 + d - 1) + _P1));
                        L0 = v_min(L0, (v_load(Lr_p0 + d + 1) + _P1));

                        L1 = v_min(L1, (v_load(Lr_p1 + d - 1) + _P1));
                        L1 = v_min(L1, (v_load(Lr_p1 + d + 1) + _P1));

                        L2 = v_min(L2, (v_load(Lr_p2 + d - 1) + _P1));
                        L2 = v_min(L2, (v_load(Lr_p2 + d + 1) + _P1));

                        L3 = v_min(L3, (v_load(Lr_p3 + d - 1) + _P1));
                        L3 = v_min(L3, (v_load(Lr_p3 + d + 1) + _P1));

                        L0 = v_min(L0, _delta0);
                        L0 = ((L0 - _delta0) + Cpd);

                        L1 = v_min(L1, _delta1);
                        L1 = ((L1 - _delta1) + Cpd);

                        L2 = v_min(L2, _delta2);
                        L2 = ((L2 - _delta2) + Cpd);

                        L3 = v_min(L3, _delta3);
                        L3 = ((L3 - _delta3) + Cpd);

                        v_store(Lr_p + d, L0);
                        v_store(Lr_p + d + D2, L1);
                        v_store(Lr_p + d + D2*2, L2);
                        v_store(Lr_p + d + D2*3, L3);

                        // Get minimum from in L0-L3
                        v_int16x8 t02L, t02H, t13L, t13H, t0123L, t0123H;
                        v_zip(L0, L2, t02L, t02H);            // L0[0] L2[0] L0[1] L2[1]...
                        v_zip(L1, L3, t13L, t13H);            // L1[0] L3[0] L1[1] L3[1]...
                        v_int16x8 t02 = v_min(t02L, t02H);    // L0[i] L2[i] L0[i] L2[i]...
                        v_int16x8 t13 = v_min(t13L, t13H);    // L1[i] L3[i] L1[i] L3[i]...
                        v_zip(t02, t13, t0123L, t0123H);      // L0[i] L1[i] L2[i] L3[i]...
                        v_int16x8 t0 = v_min(t0123L, t0123H);
                        _minL0 = v_min(_minL0, t0);

                        v_int16x8 Sval = v_load(Sp + d);

                        L0 = L0 + L1;
                        L2 = L2 + L3;
                        Sval = Sval + L0;
                        Sval = Sval + L2;

                        v_store(Sp + d, Sval);
                    }

                    v_int32x4 minL, minH;
                    v_expand(_minL0, minL, minH);
                    v_pack_store(&minLr[0][xm], v_min(minL, minH));
                }
                else
            #endif
                {
                    int minL0 = MAX_COST, minL1 = MAX_COST, minL2 = MAX_COST, minL3 = MAX_COST;

                    for( d = 0; d < D; d++ )
                    {
                        int Cpd = Cp[d], L0, L1, L2, L3;

                        L0 = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d-1] + P1, std::min(Lr_p0[d+1] + P1, delta0))) - delta0;
                        L1 = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d-1] + P1, std::min(Lr_p1[d+1] + P1, delta1))) - delta1;
                        L2 = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d-1] + P1, std::min(Lr_p2[d+1] + P1, delta2))) - delta2;
                        L3 = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d-1] + P1, std::min(Lr_p3[d+1] + P1, delta3))) - delta3;

                        Lr_p[d] = (CostType)L0;
                        minL0 = std::min(minL0, L0);

                        Lr_p[d + D2] = (CostType)L1;
                        minL1 = std::min(minL1, L1);

                        Lr_p[d + D2*2] = (CostType)L2;
                        minL2 = std::min(minL2, L2);

                        Lr_p[d + D2*3] = (CostType)L3;
                        minL3 = std::min(minL3, L3);

                        Sp[d] = saturate_cast<CostType>(Sp[d] + L0 + L1 + L2 + L3);
                    }
                    minLr[0][xm] = (CostType)minL0;
                    minLr[0][xm+1] = (CostType)minL1;
                    minLr[0][xm+2] = (CostType)minL2;
                    minLr[0][xm+3] = (CostType)minL3;
                }
            }

            if( pass == npasses )
            {
                for( x = 0; x < width; x++ )
                {
                    disp1ptr[x] = disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                    disp2cost[x] = MAX_COST;
                }

                for( x = width1 - 1; x >= 0; x-- )
                {
                    CostType* Sp = S + x*D;
                    int minS = MAX_COST, bestDisp = -1;

                    if( npasses == 1 )
                    {
                        int xm = x*NR2, xd = xm*D2;

                        int minL0 = MAX_COST;
                        int delta0 = minLr[0][xm + NR2] + P2;
                        CostType* Lr_p0 = Lr[0] + xd + NRD2;
                        Lr_p0[-1] = Lr_p0[D] = MAX_COST;
                        CostType* Lr_p = Lr[0] + xd;

                        const CostType* Cp = C + x*D;

                    #if CV_SIMD128
                        if (true)
                        {
                            v_int16x8 _P1 = v_setall_s16((short)P1);
                            v_int16x8 _delta0 = v_setall_s16((short)delta0);

                            v_int16x8 _minL0 = v_setall_s16((short)minL0);
                            v_int16x8 _minS = v_setall_s16(MAX_COST), _bestDisp = v_setall_s16(-1);
                            v_int16x8 _d8 = v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = v_setall_s16(8);

                            for( d = 0; d < D; d += 8 )
                            {
                                v_int16x8 Cpd = v_load(Cp + d);
                                v_int16x8 L0 = v_load(Lr_p0 + d);

                                L0 = v_min(L0, v_load(Lr_p0 + d - 1) + _P1);
                                L0 = v_min(L0, v_load(Lr_p0 + d + 1) + _P1);
                                L0 = v_min(L0, _delta0);
                                L0 = L0 - _delta0 + Cpd;

                                v_store(Lr_p + d, L0);
                                _minL0 = v_min(_minL0, L0);
                                L0 = L0 + v_load(Sp + d);
                                v_store(Sp + d, L0);

                                v_int16x8 mask = _minS > L0;
                                _minS = v_min(_minS, L0);
                                _bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
                                _d8 += _8;
                            }
                            short bestDispBuf[8];
                            v_store(bestDispBuf, _bestDisp);

                            v_int32x4 min32L, min32H;
                            v_expand(_minL0, min32L, min32H);
                            minLr[0][xm] = (CostType)std::min(v_reduce_min(min32L), v_reduce_min(min32H));

                            v_expand(_minS, min32L, min32H);
                            minS = std::min(v_reduce_min(min32L), v_reduce_min(min32H));

                            v_int16x8 ss = v_setall_s16((short)minS);
                            v_uint16x8 minMask = v_reinterpret_as_u16(ss == _minS);
                            v_uint16x8 minBit = minMask & v_LSB;

                            v_uint32x4 minBitL, minBitH;
                            v_expand(minBit, minBitL, minBitH);

                            int idx = v_reduce_sum(minBitL) + v_reduce_sum(minBitH);
                            bestDisp = bestDispBuf[LSBTab[idx]];
                        }
                        else
                    #endif
                        {
                            for( d = 0; d < D; d++ )
                            {
                                int L0 = Cp[d] + std::min((int)Lr_p0[d], std::min(Lr_p0[d-1] + P1, std::min(Lr_p0[d+1] + P1, delta0))) - delta0;

                                Lr_p[d] = (CostType)L0;
                                minL0 = std::min(minL0, L0);

                                int Sval = Sp[d] = saturate_cast<CostType>(Sp[d] + L0);
                                if( Sval < minS )
                                {
                                    minS = Sval;
                                    bestDisp = d;
                                }
                            }
                            minLr[0][xm] = (CostType)minL0;
                        }
                    }
                    else
                    {
                    #if CV_SIMD128
                        if (true)
                        {
                            v_int16x8 _minS = v_setall_s16(MAX_COST), _bestDisp = v_setall_s16(-1);
                            v_int16x8 _d8 = v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = v_setall_s16(8);

                            for( d = 0; d < D; d+= 8 )
                            {
                                v_int16x8 L0 = v_load(Sp + d);
                                v_int16x8 mask = L0 < _minS;
                                _minS = v_min( L0, _minS );
                                _bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
                                _d8 = _d8 + _8;
                            }
                            v_int32x4 _d0, _d1;
                            v_expand(_minS, _d0, _d1);
                            minS = (int)std::min(v_reduce_min(_d0), v_reduce_min(_d1));
                            v_int16x8 v_mask = v_setall_s16((short)minS) == _minS;

                            _bestDisp = (_bestDisp & v_mask) | (v_setall_s16(SHRT_MAX) & ~v_mask);
                            v_expand(_bestDisp, _d0, _d1);
                            bestDisp = (int)std::min(v_reduce_min(_d0), v_reduce_min(_d1));
                        }
                        else
                    #endif
                        {
                            for( d = 0; d < D; d++ )
                            {
                                int Sval = Sp[d];
                                if( Sval < minS )
                                {
                                    minS = Sval;
                                    bestDisp = d;
                                }
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

            // now shift the cyclic buffers
            std::swap( Lr[0], Lr[1] );
            std::swap( minLr[0], minLr[1] );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
struct CalcVerticalSums: public ParallelLoopBody
{
    CalcVerticalSums(const Mat& _img1, const Mat& _img2, const StereoSGBMParams& params,
                     CostType* alignedBuf, PixType* _clipTab): img1(_img1), img2(_img2), clipTab(_clipTab)
    {
        minD = params.minDisparity;
        maxD = minD + params.numDisparities;
        SW2 = SH2 = (params.SADWindowSize > 0 ? params.SADWindowSize : 5)/2;
        ftzero = std::max(params.preFilterCap, 15) | 1;
        P1 = params.P1 > 0 ? params.P1 : 2;
        P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
        height = img1.rows;
        width = img1.cols;
        int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
        D = maxD - minD;
        width1 = maxX1 - minX1;
        D2 = D + 16;
        costBufSize = width1*D;
        CSBufSize = costBufSize*height;
        minLrSize = width1;
        LrSize = minLrSize*D2;
        hsumBufNRows = SH2*2 + 2;
        Cbuf = alignedBuf;
        Sbuf = Cbuf + CSBufSize;
        hsumBuf = Sbuf + CSBufSize;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        static const CostType MAX_COST = SHRT_MAX;
        static const int ALIGN = 16;
        static const int TAB_OFS = 256*4;
        static const int npasses = 2;
        int x1 = range.start, x2 = range.end, k;
        size_t pixDiffSize = ((x2 - x1) + 2*SW2)*D;
        size_t auxBufsSize = pixDiffSize*sizeof(CostType) +                 //pixdiff size
                             width*16*img1.channels()*sizeof(PixType) + 32; //tempBuf
        Mat auxBuff;
        auxBuff.create(1, (int)auxBufsSize, CV_8U);
        CostType* pixDiff = (CostType*)alignPtr(auxBuff.ptr(), ALIGN);
        PixType* tempBuf = (PixType*)(pixDiff + pixDiffSize);

        // Simplification of index calculation
        pixDiff -= (x1>SW2 ? (x1 - SW2): 0)*D;

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

            CostType *Lr[NLR]={0}, *minLr[NLR]={0};

            for( k = 0; k < NLR; k++ )
            {
                // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
                // and will occasionally use negative indices with the arrays
                // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
                // however, then the alignment will be imperfect, i.e. bad for SSE,
                // thus we shift the pointers by 8 (8*sizeof(short) == 16 - ideal alignment)
                Lr[k] = hsumBuf + costBufSize*hsumBufNRows + LrSize*k + 8;
                memset( Lr[k] + x1*D2 - 8, 0, (x2-x1)*D2*sizeof(CostType) );
                minLr[k] = hsumBuf + costBufSize*hsumBufNRows + LrSize*NLR + minLrSize*k;
                memset( minLr[k] + x1, 0, (x2-x1)*sizeof(CostType) );
            }

            for( int y = y1; y != y2; y += dy )
            {
                int x, d;
                CostType* C = Cbuf + y*costBufSize;
                CostType* S = Sbuf + y*costBufSize;

                if( pass == 1 ) // compute C on the first pass, and reuse it on the second pass, if any.
                {
                    int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

                    for( k = dy1; k <= dy2; k++ )
                    {
                        CostType* hsumAdd = hsumBuf + (std::min(k, height-1) % hsumBufNRows)*costBufSize;

                        if( k < height )
                        {
                            calcPixelCostBT( img1, img2, k, minD, maxD, pixDiff, tempBuf, clipTab, TAB_OFS, ftzero, x1 - SW2, x2 + SW2);

                            memset(hsumAdd + x1*D, 0, D*sizeof(CostType));
                            for( x = (x1 - SW2)*D; x <= (x1 + SW2)*D; x += D )
                            {
                                int xbord = x <= 0 ? 0 : (x > (width1 - 1)*D? (width1 - 1)*D : x);
                                for( d = 0; d < D; d++ )
                                    hsumAdd[x1*D + d] = (CostType)(hsumAdd[x1*D + d] + pixDiff[xbord + d]);
                            }

                            if( y > 0 )
                            {
                                const CostType* hsumSub = hsumBuf + (std::max(y - SH2 - 1, 0) % hsumBufNRows)*costBufSize;
                                const CostType* Cprev = C - costBufSize;

                                for( d = 0; d < D; d++ )
                                    C[x1*D + d] = (CostType)(Cprev[x1*D + d] + hsumAdd[x1*D + d] - hsumSub[x1*D + d]);

                                for( x = (x1+1)*D; x < x2*D; x += D )
                                {
                                    const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

                                #if CV_SIMD128
                                    if (true)
                                    {
                                        for( d = 0; d < D; d += 8 )
                                        {
                                            v_int16x8 hv = v_load(hsumAdd + x - D + d);
                                            v_int16x8 Cx = v_load(Cprev + x + d);
                                            v_int16x8 psub = v_load(pixSub + d);
                                            v_int16x8 padd = v_load(pixAdd + d);
                                            hv = (hv - psub + padd);
                                            psub = v_load(hsumSub + x + d);
                                            Cx = Cx - psub + hv;
                                            v_store(hsumAdd + x + d, hv);
                                            v_store(C + x + d, Cx);
                                        }
                                    }
                                    else
                                #endif
                                    {
                                        for( d = 0; d < D; d++ )
                                        {
                                            int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                                            C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for( x = (x1+1)*D; x < x2*D; x += D )
                                {
                                    const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

                                    for( d = 0; d < D; d++ )
                                        hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                                }
                            }
                        }

                        if( y == 0 )
                        {
                            int scale = k == 0 ? SH2 + 1 : 1;
                            for( x = x1*D; x < x2*D; x++ )
                                C[x] = (CostType)(C[x] + hsumAdd[x]*scale);
                        }
                    }

                    // also, clear the S buffer
                    for( k = x1*D; k < x2*D; k++ )
                        S[k] = 0;
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
                    int xd = x*D2;

                    int delta = minLr[1][x] + P2;

                    CostType* Lr_ppr = Lr[1] + xd;

                    Lr_ppr[-1] = Lr_ppr[D] = MAX_COST;

                    CostType* Lr_p = Lr[0] + xd;
                    const CostType* Cp = C + x*D;
                    CostType* Sp = S + x*D;

                #if CV_SIMD128
                    if (true)
                    {
                        v_int16x8 _P1 = v_setall_s16((short)P1);

                        v_int16x8 _delta = v_setall_s16((short)delta);
                        v_int16x8 _minL = v_setall_s16((short)MAX_COST);

                        for( d = 0; d < D; d += 8 )
                        {
                            v_int16x8 Cpd = v_load(Cp + d);
                            v_int16x8 L;

                            L = v_load(Lr_ppr + d);

                            L = v_min(L, (v_load(Lr_ppr + d - 1) + _P1));
                            L = v_min(L, (v_load(Lr_ppr + d + 1) + _P1));

                            L = v_min(L, _delta);
                            L = ((L - _delta) + Cpd);

                            v_store(Lr_p + d, L);

                            // Get minimum from in L-L3
                            _minL = v_min(_minL, L);

                            v_int16x8 Sval = v_load(Sp + d);

                            Sval = Sval + L;

                            v_store(Sp + d, Sval);
                        }

                        v_int32x4 min1, min2, min12;
                        v_expand(_minL, min1, min2);
                        min12 = v_min(min1,min2);
                        minLr[0][x] = (CostType)v_reduce_min(min12);
                    }
                    else
                #endif
                    {
                        int minL = MAX_COST;

                        for( d = 0; d < D; d++ )
                        {
                            int Cpd = Cp[d], L;

                            L = Cpd + std::min((int)Lr_ppr[d], std::min(Lr_ppr[d-1] + P1, std::min(Lr_ppr[d+1] + P1, delta))) - delta;

                            Lr_p[d] = (CostType)L;
                            minL = std::min(minL, L);

                            Sp[d] = saturate_cast<CostType>(Sp[d] + L);
                        }
                        minLr[0][x] = (CostType)minL;
                    }
                }

                // now shift the cyclic buffers
                std::swap( Lr[0], Lr[1] );
                std::swap( minLr[0], minLr[1] );
            }
        }
    }
    static const int NLR = 2;
    const Mat& img1;
    const Mat& img2;
    CostType* Cbuf;
    CostType* Sbuf;
    CostType* hsumBuf;
    PixType* clipTab;
    int minD;
    int maxD;
    int D;
    int D2;
    int SH2;
    int SW2;
    int width;
    int width1;
    int height;
    int P1;
    int P2;
    size_t costBufSize;
    size_t CSBufSize;
    size_t minLrSize;
    size_t LrSize;
    size_t hsumBufNRows;
    int ftzero;
};

struct CalcHorizontalSums: public ParallelLoopBody
{
    CalcHorizontalSums(const Mat& _img1, const Mat& _img2, Mat& _disp1, const StereoSGBMParams& params,
                     CostType* alignedBuf): img1(_img1), img2(_img2), disp1(_disp1)
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
        width1 = maxX1 - minX1;
        costBufSize = width1*D;
        CSBufSize = costBufSize*height;
        D2 = D + 16;
        LrSize = 2 * D2;
        Cbuf = alignedBuf;
        Sbuf = Cbuf + CSBufSize;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int y1 = range.start, y2 = range.end;
        size_t auxBufsSize = LrSize * sizeof(CostType) + width*(sizeof(CostType) + sizeof(DispType)) + 32;

        Mat auxBuff;
        auxBuff.create(1, (int)auxBufsSize, CV_8U);
        CostType *Lr = ((CostType*)alignPtr(auxBuff.ptr(), ALIGN)) + 8;
        CostType* disp2cost = Lr + LrSize;
        DispType* disp2ptr = (DispType*)(disp2cost + width);

        CostType minLr;

        for( int y = y1; y != y2; y++)
        {
            int x, d;
            DispType* disp1ptr = disp1.ptr<DispType>(y);
            CostType* C = Cbuf + y*costBufSize;
            CostType* S = Sbuf + y*costBufSize;

            for( x = 0; x < width; x++ )
            {
                disp1ptr[x] = disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                disp2cost[x] = MAX_COST;
            }

            // clear buffers
            memset( Lr - 8, 0, LrSize*sizeof(CostType) );
            Lr[-1] = Lr[D] = Lr[D2 - 1] = Lr[D2 + D] = MAX_COST;

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

                CostType* Lr_ppr = Lr + ((x&1)? 0 : D2);

                CostType* Lr_p = Lr + ((x&1)? D2 :0);
                const CostType* Cp = C + x*D;
                CostType* Sp = S + x*D;

            #if CV_SIMD128
                if (true)
                {
                    v_int16x8 _P1 = v_setall_s16((short)P1);

                    v_int16x8 _delta = v_setall_s16((short)delta);
                    v_int16x8 _minL = v_setall_s16((short)MAX_COST);

                    for( d = 0; d < D; d += 8 )
                    {
                        v_int16x8 Cpd = v_load(Cp + d);
                        v_int16x8 L;

                        L = v_load(Lr_ppr + d);

                        L = v_min(L, (v_load(Lr_ppr + d - 1) + _P1));
                        L = v_min(L, (v_load(Lr_ppr + d + 1) + _P1));

                        L = v_min(L, _delta);
                        L = ((L - _delta) + Cpd);

                        v_store(Lr_p + d, L);

                        // Get minimum from in L-L3
                        _minL = v_min(_minL, L);

                        v_int16x8 Sval = v_load(Sp + d);

                        Sval = Sval + L;

                        v_store(Sp + d, Sval);
                    }

                    v_int32x4 min1, min2, min12;
                    v_expand(_minL, min1, min2);
                    min12 = v_min(min1,min2);
                    minLr = (CostType)v_reduce_min(min12);
                }
                else
            #endif
                {
                    minLr = MAX_COST;
                    for( d = 0; d < D; d++ )
                    {
                        int Cpd = Cp[d], L;

                        L = Cpd + std::min((int)Lr_ppr[d], std::min(Lr_ppr[d-1] + P1, std::min(Lr_ppr[d+1] + P1, delta))) - delta;

                        Lr_p[d] = (CostType)L;
                        minLr = (CostType)std::min((int)minLr, L);

                        Sp[d] = saturate_cast<CostType>(Sp[d] + L);
                    }
                }
            }

            memset( Lr - 8, 0, LrSize*sizeof(CostType) );
            Lr[-1] = Lr[D] = Lr[D2 - 1] = Lr[D2 + D] = MAX_COST;

            minLr = 0;

            for( x = width1-1; x != -1; x--)
            {
                int delta = minLr + P2;

                CostType* Lr_ppr = Lr + ((x&1)? 0 :D2);

                CostType* Lr_p = Lr + ((x&1)? D2 :0);
                const CostType* Cp = C + x*D;
                CostType* Sp = S + x*D;
                int minS = MAX_COST, bestDisp = -1;
                minLr = MAX_COST;

            #if CV_SIMD128
                if (true)
                {
                    v_int16x8 _P1 = v_setall_s16((short)P1);

                    v_int16x8 _delta = v_setall_s16((short)delta);
                    v_int16x8 _minL = v_setall_s16((short)MAX_COST);

                    v_int16x8 _minS = v_setall_s16(MAX_COST), _bestDisp = v_setall_s16(-1);
                    v_int16x8 _d8 = v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = v_setall_s16(8);

                    for( d = 0; d < D; d+= 8 )
                    {
                        v_int16x8 Cpd = v_load(Cp + d);
                        v_int16x8 L;

                        L = v_load(Lr_ppr + d);

                        L = v_min(L, (v_load(Lr_ppr + d - 1) + _P1));
                        L = v_min(L, (v_load(Lr_ppr + d + 1) + _P1));

                        L = v_min(L, _delta);
                        L = ((L - _delta) + Cpd);

                        v_store(Lr_p + d, L);

                        // Get minimum from in L-L3
                        _minL = v_min(_minL, L);

                        v_int16x8 Sval = v_load(Sp + d);

                        Sval = Sval + L;

                        v_int16x8 mask = Sval < _minS;
                        _minS = v_min( Sval, _minS );
                        _bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
                        _d8 = _d8 + _8;

                        v_store(Sp + d, Sval);
                    }
                    v_int32x4 min1, min2, min12;
                    v_expand(_minL, min1, min2);
                    min12 = v_min(min1,min2);
                    minLr = (CostType)v_reduce_min(min12);

                    v_int32x4 _d0, _d1;
                    v_expand(_minS, _d0, _d1);
                    minS = (int)std::min(v_reduce_min(_d0), v_reduce_min(_d1));
                    v_int16x8 v_mask = v_setall_s16((short)minS) == _minS;

                    _bestDisp = (_bestDisp & v_mask) | (v_setall_s16(SHRT_MAX) & ~v_mask);
                    v_expand(_bestDisp, _d0, _d1);
                    bestDisp = (int)std::min(v_reduce_min(_d0), v_reduce_min(_d1));
                }
                else
            #endif
                {
                    for( d = 0; d < D; d++ )
                    {
                        int Cpd = Cp[d], L;

                        L = Cpd + std::min((int)Lr_ppr[d], std::min(Lr_ppr[d-1] + P1, std::min(Lr_ppr[d+1] + P1, delta))) - delta;

                        Lr_p[d] = (CostType)L;
                        minLr = (CostType)std::min((int)minLr, L);

                        Sp[d] = saturate_cast<CostType>(Sp[d] + L);
                        if( Sp[d] < minS )
                        {
                            minS = Sp[d];
                            bestDisp = d;
                        }
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
    static const int ALIGN = 16;
    const Mat& img1;
    const Mat& img2;
    Mat& disp1;
    CostType* Cbuf;
    CostType* Sbuf;
    int minD;
    int maxD;
    int D;
    int D2;
    int width;
    int width1;
    int height;
    int P1;
    int P2;
    int minX1;
    int maxX1;
    size_t costBufSize;
    size_t CSBufSize;
    size_t LrSize;
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
                                 Mat& disp1, const StereoSGBMParams& params,
                                 Mat& buffer )
{
    const int ALIGN = 16;
    const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
    const int DISP_SCALE = (1 << DISP_SHIFT);
    int minD = params.minDisparity, maxD = minD + params.numDisparities;
    Size SADWindowSize;
    SADWindowSize.width = SADWindowSize.height = params.SADWindowSize > 0 ? params.SADWindowSize : 5;
    int ftzero = std::max(params.preFilterCap, 15) | 1;
    int P1 = params.P1 > 0 ? params.P1 : 2, P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
    int k, width = disp1.cols, height = disp1.rows;
    int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
    int D = maxD - minD, width1 = maxX1 - minX1;
    int SH2 = SADWindowSize.height/2;
    int INVALID_DISP = minD - 1;
    int INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
    const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2;
    PixType clipTab[TAB_SIZE];

    for( k = 0; k < TAB_SIZE; k++ )
        clipTab[k] = (PixType)(std::min(std::max(k - TAB_OFS, -ftzero), ftzero) + ftzero);

    if( minX1 >= maxX1 )
    {
        disp1 = Scalar::all(INVALID_DISP_SCALED);
        return;
    }

    CV_Assert( D % 16 == 0 );

    int D2 = D+16;

    // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
    // for dynamic programming we need the current row and
    // the previous row, i.e. 2 rows in total
    const int NLR = 2;

    // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
    // we keep pixel difference cost (C) and the summary cost over 4 directions (S).
    // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
    size_t costBufSize = width1*D;
    size_t CSBufSize = costBufSize*height;
    size_t minLrSize = width1 , LrSize = minLrSize*D2;
    int hsumBufNRows = SH2*2 + 2;
    size_t totalBufSize = (LrSize + minLrSize)*NLR*sizeof(CostType) + // minLr[] and Lr[]
                          costBufSize*hsumBufNRows*sizeof(CostType) + // hsumBuf
                          CSBufSize*2*sizeof(CostType) + 1024;        // C, S

    if( buffer.empty() || !buffer.isContinuous() ||
        buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize )
    {
        buffer.reserveBuffer(totalBufSize);
    }

    // summary cost over different (nDirs) directions
    CostType* Cbuf = (CostType*)alignPtr(buffer.ptr(), ALIGN);

    // add P2 to every C(x,y). it saves a few operations in the inner loops
    for(k = 0; k < (int)CSBufSize; k++ )
        Cbuf[k] = (CostType)P2;

    parallel_for_(Range(0,width1),CalcVerticalSums(img1, img2, params, Cbuf, clipTab),8);
    parallel_for_(Range(0,height),CalcHorizontalSums(img1, img2, disp1, params, Cbuf),8);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////

void getBufferPointers(Mat& buffer, int width, int width1, int D, int num_ch, int SH2, int P2,
                       CostType*& curCostVolumeLine, CostType*& hsumBuf, CostType*& pixDiff,
                       PixType*& tmpBuf, CostType*& horPassCostVolume,
                       CostType*& vertPassCostVolume, CostType*& vertPassMin, CostType*& rightPassBuf,
                       CostType*& disp2CostBuf, short*& disp2Buf);

struct SGBM3WayMainLoop : public ParallelLoopBody
{
    Mat* buffers;
    const Mat *img1, *img2;
    Mat* dst_disp;

    int nstripes, stripe_sz;
    int stripe_overlap;

    int width,height;
    int minD, maxD, D;
    int minX1, maxX1, width1;

    int SW2, SH2;
    int P1, P2;
    int uniquenessRatio, disp12MaxDiff;

    int costBufSize, hsumBufNRows;
    int TAB_OFS, ftzero;

    PixType* clipTab;

    SGBM3WayMainLoop(Mat *_buffers, const Mat& _img1, const Mat& _img2, Mat* _dst_disp, const StereoSGBMParams& params, PixType* _clipTab, int _nstripes, int _stripe_overlap);
    void getRawMatchingCost(CostType* C, CostType* hsumBuf, CostType* pixDiff, PixType* tmpBuf, int y, int src_start_idx) const;
    void operator () (const Range& range) const CV_OVERRIDE;
};

SGBM3WayMainLoop::SGBM3WayMainLoop(Mat *_buffers, const Mat& _img1, const Mat& _img2, Mat* _dst_disp, const StereoSGBMParams& params, PixType* _clipTab, int _nstripes, int _stripe_overlap):
buffers(_buffers), img1(&_img1), img2(&_img2), dst_disp(_dst_disp), clipTab(_clipTab)
{
    nstripes = _nstripes;
    stripe_overlap = _stripe_overlap;
    stripe_sz = (int)ceil(img1->rows/(double)nstripes);

    width = img1->cols; height = img1->rows;
    minD = params.minDisparity; maxD = minD + params.numDisparities; D = maxD - minD;
    minX1 = std::max(maxD, 0); maxX1 = width + std::min(minD, 0); width1 = maxX1 - minX1;
    CV_Assert( D % 16 == 0 );

    SW2 = SH2 = params.SADWindowSize > 0 ? params.SADWindowSize/2 : 1;

    P1 = params.P1 > 0 ? params.P1 : 2; P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1+1);
    uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
    disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;

    costBufSize = width1*D;
    hsumBufNRows = SH2*2 + 2;
    TAB_OFS = 256*4;
    ftzero = std::max(params.preFilterCap, 15) | 1;
}

void getBufferPointers(Mat& buffer, int width, int width1, int D, int num_ch, int SH2, int P2,
                       CostType*& curCostVolumeLine, CostType*& hsumBuf, CostType*& pixDiff,
                       PixType*& tmpBuf, CostType*& horPassCostVolume,
                       CostType*& vertPassCostVolume, CostType*& vertPassMin, CostType*& rightPassBuf,
                       CostType*& disp2CostBuf, short*& disp2Buf)
{
    // allocating all the required memory:
    int costVolumeLineSize = width1*D;
    int width1_ext = width1+2;
    int costVolumeLineSize_ext = width1_ext*D;
    int hsumBufNRows = SH2*2 + 2;

    // main buffer to store matching costs for the current line:
    int curCostVolumeLineSize = costVolumeLineSize*sizeof(CostType);

    // auxiliary buffers for the raw matching cost computation:
    int hsumBufSize  = costVolumeLineSize*hsumBufNRows*sizeof(CostType);
    int pixDiffSize  = costVolumeLineSize*sizeof(CostType);
    int tmpBufSize   = width*16*num_ch*sizeof(PixType);

    // auxiliary buffers for the matching cost aggregation:
    int horPassCostVolumeSize  = costVolumeLineSize_ext*sizeof(CostType); // buffer for the 2-pass horizontal cost aggregation
    int vertPassCostVolumeSize = costVolumeLineSize_ext*sizeof(CostType); // buffer for the vertical cost aggregation
    int vertPassMinSize        = width1_ext*sizeof(CostType);             // buffer for storing minimum costs from the previous line
    int rightPassBufSize       = D*sizeof(CostType);                      // additional small buffer for the right-to-left pass

    // buffers for the pseudo-LRC check:
    int disp2CostBufSize = width*sizeof(CostType);
    int disp2BufSize     = width*sizeof(short);

    // sum up the sizes of all the buffers:
    size_t totalBufSize = curCostVolumeLineSize +
                          hsumBufSize +
                          pixDiffSize +
                          tmpBufSize  +
                          horPassCostVolumeSize +
                          vertPassCostVolumeSize +
                          vertPassMinSize +
                          rightPassBufSize +
                          disp2CostBufSize +
                          disp2BufSize +
                          16;  //to compensate for the alignPtr shifts

    if( buffer.empty() || !buffer.isContinuous() || buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize )
        buffer.reserveBuffer(totalBufSize);

    // set up all the pointers:
    curCostVolumeLine  = (CostType*)alignPtr(buffer.ptr(), 16);
    hsumBuf            = curCostVolumeLine + costVolumeLineSize;
    pixDiff            = hsumBuf + costVolumeLineSize*hsumBufNRows;
    tmpBuf             = (PixType*)(pixDiff + costVolumeLineSize);
    horPassCostVolume  = (CostType*)(tmpBuf + width*16*num_ch);
    vertPassCostVolume = horPassCostVolume + costVolumeLineSize_ext;
    rightPassBuf       = vertPassCostVolume + costVolumeLineSize_ext;
    vertPassMin        = rightPassBuf + D;
    disp2CostBuf       = vertPassMin + width1_ext;
    disp2Buf           = disp2CostBuf + width;

    // initialize memory:
    memset(buffer.ptr(),0,totalBufSize);
    for(int i=0;i<costVolumeLineSize;i++)
        curCostVolumeLine[i] = (CostType)P2; //such initialization simplifies the cost aggregation loops a bit
}

// performing block matching and building raw cost-volume for the current row
void SGBM3WayMainLoop::getRawMatchingCost(CostType* C, // target cost-volume row
                                          CostType* hsumBuf, CostType* pixDiff, PixType* tmpBuf, //buffers
                                          int y, int src_start_idx) const
{
    int x, d;
    int dy1 = (y == src_start_idx) ? src_start_idx : y + SH2, dy2 = (y == src_start_idx) ? src_start_idx+SH2 : dy1;

    for(int k = dy1; k <= dy2; k++ )
    {
        CostType* hsumAdd = hsumBuf + (std::min(k, height-1) % hsumBufNRows)*costBufSize;
        if( k < height )
        {
            calcPixelCostBT( *img1, *img2, k, minD, maxD, pixDiff, tmpBuf, clipTab, TAB_OFS, ftzero );

            memset(hsumAdd, 0, D*sizeof(CostType));
            for(x = 0; x <= SW2*D; x += D )
            {
                int scale = x == 0 ? SW2 + 1 : 1;

                for( d = 0; d < D; d++ )
                    hsumAdd[d] = (CostType)(hsumAdd[d] + pixDiff[x + d]*scale);
            }

            if( y > src_start_idx )
            {
                const CostType* hsumSub = hsumBuf + (std::max(y - SH2 - 1, src_start_idx) % hsumBufNRows)*costBufSize;

                for( x = D; x < width1*D; x += D )
                {
                    const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

#if CV_SIMD128
                    if (true)
                    {
                        v_int16x8 hv_reg;
                        for( d = 0; d < D; d+=8 )
                        {
                            hv_reg = v_load_aligned(hsumAdd+x-D+d) + (v_load_aligned(pixAdd+d) - v_load_aligned(pixSub+d));
                            v_store_aligned(hsumAdd+x+d,hv_reg);
                            v_store_aligned(C+x+d,v_load_aligned(C+x+d)+(hv_reg-v_load_aligned(hsumSub+x+d)));
                        }
                    }
                    else
#endif
                    {
                        for( d = 0; d < D; d++ )
                        {
                            int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                            C[x + d] = (CostType)(C[x + d] + hv - hsumSub[x + d]);
                        }
                    }
                }
            }
            else
            {
                for( x = D; x < width1*D; x += D )
                {
                    const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                    const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

                    for( d = 0; d < D; d++ )
                        hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                }
            }
        }

        if( y == src_start_idx )
        {
            int scale = k == src_start_idx ? SH2 + 1 : 1;
            for( x = 0; x < width1*D; x++ )
                C[x] = (CostType)(C[x] + hsumAdd[x]*scale);
        }
    }
}

#if CV_SIMD128
// define some additional reduce operations:
inline short min_pos(const v_int16x8& val, const v_int16x8& pos, const short min_val)
{
    v_int16x8 v_min = v_setall_s16(min_val);
    v_int16x8 v_mask = v_min == val;
    v_int16x8 v_pos = (pos & v_mask) | (v_setall_s16(SHRT_MAX) & ~v_mask);

    return v_reduce_min(v_pos);
}
#endif

// performing SGM cost accumulation from left to right (result is stored in leftBuf) and
// in-place cost accumulation from top to bottom (result is stored in topBuf)
inline void accumulateCostsLeftTop(CostType* leftBuf, CostType* leftBuf_prev, CostType* topBuf, CostType* costs,
                                   CostType& leftMinCost, CostType& topMinCost, int D, int P1, int P2)
{
#if CV_SIMD128
    if (true)
    {
        v_int16x8 P1_reg = v_setall_s16(cv::saturate_cast<CostType>(P1));

        v_int16x8 leftMinCostP2_reg   = v_setall_s16(cv::saturate_cast<CostType>(leftMinCost+P2));
        v_int16x8 leftMinCost_new_reg = v_setall_s16(SHRT_MAX);
        v_int16x8 src0_leftBuf        = v_setall_s16(SHRT_MAX);
        v_int16x8 src1_leftBuf        = v_load_aligned(leftBuf_prev);

        v_int16x8 topMinCostP2_reg   = v_setall_s16(cv::saturate_cast<CostType>(topMinCost+P2));
        v_int16x8 topMinCost_new_reg = v_setall_s16(SHRT_MAX);
        v_int16x8 src0_topBuf        = v_setall_s16(SHRT_MAX);
        v_int16x8 src1_topBuf        = v_load_aligned(topBuf);

        v_int16x8 src2;
        v_int16x8 src_shifted_left,src_shifted_right;
        v_int16x8 res;

        for(int i=0;i<D-8;i+=8)
        {
            //process leftBuf:
            //lookahead load:
            src2 = v_load_aligned(leftBuf_prev+i+8);

            //get shifted versions of the current block and add P1:
            src_shifted_left  = v_extract<7> (src0_leftBuf,src1_leftBuf) + P1_reg;
            src_shifted_right = v_extract<1> (src1_leftBuf,src2        ) + P1_reg;

            // process and save current block:
            res = v_load_aligned(costs+i) + (v_min(v_min(src_shifted_left,src_shifted_right),v_min(src1_leftBuf,leftMinCostP2_reg))-leftMinCostP2_reg);
            leftMinCost_new_reg = v_min(leftMinCost_new_reg,res);
            v_store_aligned(leftBuf+i, res);

            //update src buffers:
            src0_leftBuf = src1_leftBuf;
            src1_leftBuf = src2;

            //process topBuf:
            //lookahead load:
            src2 = v_load_aligned(topBuf+i+8);

            //get shifted versions of the current block and add P1:
            src_shifted_left  = v_extract<7> (src0_topBuf,src1_topBuf) + P1_reg;
            src_shifted_right = v_extract<1> (src1_topBuf,src2       ) + P1_reg;

            // process and save current block:
            res = v_load_aligned(costs+i) + (v_min(v_min(src_shifted_left,src_shifted_right),v_min(src1_topBuf,topMinCostP2_reg))-topMinCostP2_reg);
            topMinCost_new_reg = v_min(topMinCost_new_reg,res);
            v_store_aligned(topBuf+i, res);

            //update src buffers:
            src0_topBuf = src1_topBuf;
            src1_topBuf = src2;
        }

        // a bit different processing for the last cycle of the loop:
        //process leftBuf:
        src2 = v_setall_s16(SHRT_MAX);
        src_shifted_left  = v_extract<7> (src0_leftBuf,src1_leftBuf) + P1_reg;
        src_shifted_right = v_extract<1> (src1_leftBuf,src2        ) + P1_reg;

        res = v_load_aligned(costs+D-8) + (v_min(v_min(src_shifted_left,src_shifted_right),v_min(src1_leftBuf,leftMinCostP2_reg))-leftMinCostP2_reg);
        leftMinCost = v_reduce_min(v_min(leftMinCost_new_reg,res));
        v_store_aligned(leftBuf+D-8, res);

        //process topBuf:
        src2 = v_setall_s16(SHRT_MAX);
        src_shifted_left  = v_extract<7> (src0_topBuf,src1_topBuf) + P1_reg;
        src_shifted_right = v_extract<1> (src1_topBuf,src2       ) + P1_reg;

        res = v_load_aligned(costs+D-8) + (v_min(v_min(src_shifted_left,src_shifted_right),v_min(src1_topBuf,topMinCostP2_reg))-topMinCostP2_reg);
        topMinCost = v_reduce_min(v_min(topMinCost_new_reg,res));
        v_store_aligned(topBuf+D-8, res);
    }
    else
#endif
    {
        CostType leftMinCost_new = SHRT_MAX;
        CostType topMinCost_new  = SHRT_MAX;
        int leftMinCost_P2  = leftMinCost + P2;
        int topMinCost_P2   = topMinCost  + P2;
        CostType leftBuf_prev_i_minus_1 = SHRT_MAX;
        CostType topBuf_i_minus_1       = SHRT_MAX;
        CostType tmp;

        for(int i=0;i<D-1;i++)
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
inline void accumulateCostsRight(CostType* rightBuf, CostType* topBuf, CostType* leftBuf, CostType* costs,
                                 CostType& rightMinCost, int D, int P1, int P2, int& optimal_disp, CostType& min_cost)
{
#if CV_SIMD128
    if (true)
    {
        v_int16x8 P1_reg = v_setall_s16(cv::saturate_cast<CostType>(P1));

        v_int16x8 rightMinCostP2_reg   = v_setall_s16(cv::saturate_cast<CostType>(rightMinCost+P2));
        v_int16x8 rightMinCost_new_reg = v_setall_s16(SHRT_MAX);
        v_int16x8 src0_rightBuf        = v_setall_s16(SHRT_MAX);
        v_int16x8 src1_rightBuf        = v_load(rightBuf);

        v_int16x8 src2;
        v_int16x8 src_shifted_left,src_shifted_right;
        v_int16x8 res;

        v_int16x8 min_sum_cost_reg = v_setall_s16(SHRT_MAX);
        v_int16x8 min_sum_pos_reg  = v_setall_s16(0);
        v_int16x8 loop_idx(0,1,2,3,4,5,6,7);
        v_int16x8 eight_reg = v_setall_s16(8);

        for(int i=0;i<D-8;i+=8)
        {
            //lookahead load:
            src2 = v_load_aligned(rightBuf+i+8);

            //get shifted versions of the current block and add P1:
            src_shifted_left  = v_extract<7> (src0_rightBuf,src1_rightBuf) + P1_reg;
            src_shifted_right = v_extract<1> (src1_rightBuf,src2         ) + P1_reg;

            // process and save current block:
            res = v_load_aligned(costs+i) + (v_min(v_min(src_shifted_left,src_shifted_right),v_min(src1_rightBuf,rightMinCostP2_reg))-rightMinCostP2_reg);
            rightMinCost_new_reg = v_min(rightMinCost_new_reg,res);
            v_store_aligned(rightBuf+i, res);

            // compute and save total cost:
            res = res + v_load_aligned(leftBuf+i) + v_load_aligned(topBuf+i);
            v_store_aligned(leftBuf+i, res);

            // track disparity value with the minimum cost:
            min_sum_cost_reg = v_min(min_sum_cost_reg,res);
            min_sum_pos_reg = min_sum_pos_reg + ((min_sum_cost_reg == res) & (loop_idx - min_sum_pos_reg));
            loop_idx = loop_idx+eight_reg;

            //update src:
            src0_rightBuf    = src1_rightBuf;
            src1_rightBuf    = src2;
        }

        // a bit different processing for the last cycle of the loop:
        src2 = v_setall_s16(SHRT_MAX);
        src_shifted_left  = v_extract<7> (src0_rightBuf,src1_rightBuf) + P1_reg;
        src_shifted_right = v_extract<1> (src1_rightBuf,src2         ) + P1_reg;

        res = v_load_aligned(costs+D-8) + (v_min(v_min(src_shifted_left,src_shifted_right),v_min(src1_rightBuf,rightMinCostP2_reg))-rightMinCostP2_reg);
        rightMinCost = v_reduce_min(v_min(rightMinCost_new_reg,res));
        v_store_aligned(rightBuf+D-8, res);

        res = res + v_load_aligned(leftBuf+D-8) + v_load_aligned(topBuf+D-8);
        v_store_aligned(leftBuf+D-8, res);

        min_sum_cost_reg = v_min(min_sum_cost_reg,res);
        min_cost = v_reduce_min(min_sum_cost_reg);
        min_sum_pos_reg = min_sum_pos_reg + ((min_sum_cost_reg == res) & (loop_idx - min_sum_pos_reg));
        optimal_disp = min_pos(min_sum_cost_reg,min_sum_pos_reg, min_cost);
    }
    else
#endif
    {
        CostType rightMinCost_new = SHRT_MAX;
        int rightMinCost_P2  = rightMinCost + P2;
        CostType rightBuf_i_minus_1 = SHRT_MAX;
        CostType tmp;
        min_cost = SHRT_MAX;

        for(int i=0;i<D-1;i++)
        {
            tmp = rightBuf[i];
            rightBuf[i]  = cv::saturate_cast<CostType>(costs[i] + std::min(std::min(rightBuf_i_minus_1+P1,rightBuf[i+1]+P1),std::min((int)rightBuf[i],rightMinCost_P2))-rightMinCost_P2);
            rightBuf_i_minus_1 = tmp;
            rightMinCost_new  = std::min(rightMinCost_new,rightBuf[i]);
            leftBuf[i] = cv::saturate_cast<CostType>((int)leftBuf[i]+rightBuf[i]+topBuf[i]);
            if(leftBuf[i]<min_cost)
            {
                optimal_disp = i;
                min_cost = leftBuf[i];
            }
        }

        rightBuf[D-1]  = cv::saturate_cast<CostType>(costs[D-1] + std::min(rightBuf_i_minus_1+P1,std::min((int)rightBuf[D-1],rightMinCost_P2))-rightMinCost_P2);
        rightMinCost  = std::min(rightMinCost_new,rightBuf[D-1]);
        leftBuf[D-1] = cv::saturate_cast<CostType>((int)leftBuf[D-1]+rightBuf[D-1]+topBuf[D-1]);
        if(leftBuf[D-1]<min_cost)
        {
            optimal_disp = D-1;
            min_cost = leftBuf[D-1];
        }
    }
}

void SGBM3WayMainLoop::operator () (const Range& range) const
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

    Mat cur_buffer = buffers [range.start];
    Mat cur_disp   = dst_disp[range.start];
    cur_disp = Scalar(INVALID_DISP_SCALED);

    // prepare buffers:
    CostType *curCostVolumeLine, *hsumBuf, *pixDiff;
    PixType* tmpBuf;
    CostType *horPassCostVolume, *vertPassCostVolume, *vertPassMin, *rightPassBuf, *disp2CostBuf;
    short* disp2Buf;
    getBufferPointers(cur_buffer,width,width1,D,img1->channels(),SH2,P2,
                      curCostVolumeLine,hsumBuf,pixDiff,tmpBuf,horPassCostVolume,
                      vertPassCostVolume,vertPassMin,rightPassBuf,disp2CostBuf,disp2Buf);

    // start real processing:
    for(int y=src_start_idx;y<src_end_idx;y++)
    {
        getRawMatchingCost(curCostVolumeLine,hsumBuf,pixDiff,tmpBuf,y,src_start_idx);

        short* disp_row = (short*)cur_disp.ptr(dst_offset+(y-src_start_idx));

        // initialize the auxiliary buffers for the pseudo left-right consistency check:
        for(int x=0;x<width;x++)
        {
            disp2Buf[x] = (short)INVALID_DISP_SCALED;
            disp2CostBuf[x] = SHRT_MAX;
        }
        CostType* C = curCostVolumeLine - D;
        CostType prev_min, min_cost;
        int d, best_d;
        d = best_d = 0;

        // forward pass
        prev_min=0;
        for (int x=D;x<(1+width1)*D;x+=D)
            accumulateCostsLeftTop(horPassCostVolume+x,horPassCostVolume+x-D,vertPassCostVolume+x,C+x,prev_min,vertPassMin[x/D],D,P1,P2);

        //backward pass
        memset(rightPassBuf,0,D*sizeof(CostType));
        prev_min=0;
        for (int x=width1*D;x>=D;x-=D)
        {
            accumulateCostsRight(rightPassBuf,vertPassCostVolume+x,horPassCostVolume+x,C+x,prev_min,D,P1,P2,best_d,min_cost);

            if(uniquenessRatio>0)
            {
#if CV_SIMD128
                if (true)
                {
                    horPassCostVolume+=x;
                    int thresh = (100*min_cost)/(100-uniquenessRatio);
                    v_int16x8 thresh_reg = v_setall_s16((short)(thresh+1));
                    v_int16x8 d1 = v_setall_s16((short)(best_d-1));
                    v_int16x8 d2 = v_setall_s16((short)(best_d+1));
                    v_int16x8 eight_reg = v_setall_s16(8);
                    v_int16x8 cur_d(0,1,2,3,4,5,6,7);
                    v_int16x8 mask,cost1,cost2;

                    for( d = 0; d < D; d+=16 )
                    {
                        cost1 = v_load_aligned(horPassCostVolume+d);
                        cost2 = v_load_aligned(horPassCostVolume+d+8);

                        mask = cost1 < thresh_reg;
                        mask = mask & ( (cur_d<d1) | (cur_d>d2) );
                        if( v_check_any(mask) )
                            break;

                        cur_d = cur_d+eight_reg;

                        mask = cost2 < thresh_reg;
                        mask = mask & ( (cur_d<d1) | (cur_d>d2) );
                        if( v_check_any(mask) )
                            break;

                        cur_d = cur_d+eight_reg;
                    }
                    horPassCostVolume-=x;
                }
                else
#endif
                {
                    for( d = 0; d < D; d++ )
                    {
                        if( horPassCostVolume[x+d]*(100 - uniquenessRatio) < min_cost*100 && std::abs(d - best_d) > 1 )
                            break;
                    }
                }
                if( d < D )
                    continue;
            }
            d = best_d;

            int _x2 = x/D - 1 + minX1 - d - minD;
            if( _x2>=0 && _x2<width && disp2CostBuf[_x2] > min_cost )
            {
                disp2CostBuf[_x2] = min_cost;
                disp2Buf[_x2] = (short)(d + minD);
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

            disp_row[(x/D)-1 + minX1] = (DispType)(d + minD*DISP_SCALE);
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
            if( 0 <= _x && _x < width && disp2Buf[_x] >= minD && std::abs(disp2Buf[_x] - _d) > disp12MaxDiff &&
                0 <= x_ && x_ < width && disp2Buf[x_] >= minD && std::abs(disp2Buf[x_] - d_) > disp12MaxDiff )
                disp_row[x] = (short)INVALID_DISP_SCALED;
        }
    }
}

static void computeDisparity3WaySGBM( const Mat& img1, const Mat& img2,
                                      Mat& disp1, const StereoSGBMParams& params,
                                      Mat* buffers, int nstripes )
{
    // precompute a lookup table for the raw matching cost computation:
    const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2;
    PixType* clipTab = new PixType[TAB_SIZE];
    int ftzero = std::max(params.preFilterCap, 15) | 1;
    for(int k = 0; k < TAB_SIZE; k++ )
        clipTab[k] = (PixType)(std::min(std::max(k - TAB_OFS, -ftzero), ftzero) + ftzero);

    // allocate separate dst_disp arrays to avoid conflicts due to stripe overlap:
    int stripe_sz = (int)ceil(img1.rows/(double)nstripes);
    int stripe_overlap = (params.SADWindowSize/2+1) + (int)ceil(0.1*stripe_sz);
    Mat* dst_disp = new Mat[nstripes];
    for(int i=0;i<nstripes;i++)
        dst_disp[i].create(stripe_sz+stripe_overlap,img1.cols,CV_16S);

    parallel_for_(Range(0,nstripes),SGBM3WayMainLoop(buffers,img1,img2,dst_disp,params,clipTab,nstripes,stripe_overlap));

    //assemble disp1 from dst_disp:
    short* dst_row;
    short* src_row;
    for(int i=0;i<disp1.rows;i++)
    {
        dst_row = (short*)disp1.ptr(i);
        src_row = (short*)dst_disp[i/stripe_sz].ptr(stripe_overlap+i%stripe_sz);
        memcpy(dst_row,src_row,disp1.cols*sizeof(short));
    }

    delete[] clipTab;
    delete[] dst_disp;
}

class StereoSGBMImpl CV_FINAL : public StereoSGBM
{
public:
    StereoSGBMImpl()
    {
        params = StereoSGBMParams();
    }

    StereoSGBMImpl( int _minDisparity, int _numDisparities, int _SADWindowSize,
                    int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                    int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                    int _mode )
    {
        params = StereoSGBMParams( _minDisparity, _numDisparities, _SADWindowSize,
                                   _P1, _P2, _disp12MaxDiff, _preFilterCap,
                                   _uniquenessRatio, _speckleWindowSize, _speckleRange,
                                   _mode );
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
            computeDisparity3WaySGBM( left, right, disp, params, buffers, num_stripes );
        else if(params.mode==MODE_HH4)
            computeDisparitySGBM_HH4( left, right, disp, params, buffer );
        else
            computeDisparitySGBM( left, right, disp, params, buffer );

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

    // the number of stripes is fixed, disregarding the number of threads/processors
    // to make the results fully reproducible:
    static const int num_stripes = 4;
    Mat buffers[num_stripes];

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
    size_t bufSize = npixels*(int)(sizeof(Point2s) + sizeof(int) + sizeof(uchar));
    if( !_buf.isContinuous() || _buf.empty() || _buf.cols*_buf.rows*_buf.elemSize() < bufSize )
        _buf.reserveBuffer(bufSize);

    uchar* buf = _buf.ptr();
    int i, j, dstep = (int)(img.step/sizeof(T));
    int* labels = (int*)buf;
    buf += npixels*sizeof(labels[0]);
    Point2s* wbuf = (Point2s*)buf;
    buf += npixels*sizeof(wbuf[0]);
    uchar* rtype = (uchar*)buf;
    int curlabel = 0;

    // clear out label assignments
    memset(labels, 0, npixels*sizeof(labels[0]));

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
