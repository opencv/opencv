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


static inline int cmpBlocks(const uchar* A, const uchar* B, int Bstep, CvSize blockSize )
{
    int x, s = 0;
    for( ; blockSize.height--; A += blockSize.width, B += Bstep )
    {
        for( x = 0; x <= blockSize.width - 4; x += 4 )
            s += std::abs(A[x] - B[x]) + std::abs(A[x+1] - B[x+1]) +
                std::abs(A[x+2] - B[x+2]) + std::abs(A[x+3] - B[x+3]);
        for( ; x < blockSize.width; x++ )
            s += std::abs(A[x] - B[x]);
    }
    return s;
}


CV_IMPL void
cvCalcOpticalFlowBM( const void* srcarrA, const void* srcarrB,
                     CvSize blockSize, CvSize shiftSize,
                     CvSize maxRange, int usePrevious,
                     void* velarrx, void* velarry )
{
    CvMat stubA, *srcA = cvGetMat( srcarrA, &stubA );
    CvMat stubB, *srcB = cvGetMat( srcarrB, &stubB );

    CvMat stubx, *velx = cvGetMat( velarrx, &stubx );
    CvMat stuby, *vely = cvGetMat( velarry, &stuby );

    if( !CV_ARE_TYPES_EQ( srcA, srcB ))
        CV_Error( CV_StsUnmatchedFormats, "Source images have different formats" );

    if( !CV_ARE_TYPES_EQ( velx, vely ))
        CV_Error( CV_StsUnmatchedFormats, "Destination images have different formats" );

    CvSize velSize(
        (srcA->width - blockSize.width + shiftSize.width)/shiftSize.width,
        (srcA->height - blockSize.height + shiftSize.height)/shiftSize.height
    );

    if( !CV_ARE_SIZES_EQ( srcA, srcB ) ||
        !CV_ARE_SIZES_EQ( velx, vely ) ||
        velx->width != velSize.width ||
        vely->height != velSize.height )
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( CV_MAT_TYPE( srcA->type ) != CV_8UC1 ||
        CV_MAT_TYPE( velx->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat, "Source images must have 8uC1 type and "
                                           "destination images must have 32fC1 type" );

    if( srcA->step != srcB->step || velx->step != vely->step )
        CV_Error( CV_BadStep, "two source or two destination images have different steps" );

    const int SMALL_DIFF=2;
    const int BIG_DIFF=128;

    // scanning scheme coordinates
    std::vector<CvPoint> _ss((2 * maxRange.width + 1) * (2 * maxRange.height + 1));
    CvPoint* ss = &_ss[0];
    int ss_count = 0;

    int blWidth = blockSize.width, blHeight = blockSize.height;
    int blSize = blWidth*blHeight;
    int acceptLevel = blSize * SMALL_DIFF;
    int escapeLevel = blSize * BIG_DIFF;

    int i, j;

    std::vector<uchar> _blockA(cvAlign(blSize + 16, 16));
    uchar* blockA = (uchar*)cvAlignPtr(&_blockA[0], 16);

    // Calculate scanning scheme
    int min_count = MIN( maxRange.width, maxRange.height );

    // use spiral search pattern
    //
    //     9 10 11 12
    //     8  1  2 13
    //     7  *  3 14
    //     6  5  4 15
    //... 20 19 18 17
    //

    for( i = 0; i < min_count; i++ )
    {
        // four cycles along sides
        int x = -i-1, y = x;

        // upper side
        for( j = -i; j <= i + 1; j++, ss_count++ )
        {
            ss[ss_count].x = ++x;
            ss[ss_count].y = y;
        }

        // right side
        for( j = -i; j <= i + 1; j++, ss_count++ )
        {
            ss[ss_count].x = x;
            ss[ss_count].y = ++y;
        }

        // bottom side
        for( j = -i; j <= i + 1; j++, ss_count++ )
        {
            ss[ss_count].x = --x;
            ss[ss_count].y = y;
        }

        // left side
        for( j = -i; j <= i + 1; j++, ss_count++ )
        {
            ss[ss_count].x = x;
            ss[ss_count].y = --y;
        }
    }

    // the rest part
    if( maxRange.width < maxRange.height )
    {
        int xleft = -min_count;

        // cycle by neighbor rings
        for( i = min_count; i < maxRange.height; i++ )
        {
            // two cycles by x
            int y = -(i + 1);
            int x = xleft;

            // upper side
            for( j = -maxRange.width; j <= maxRange.width; j++, ss_count++, x++ )
            {
                ss[ss_count].x = x;
                ss[ss_count].y = y;
            }

            x = xleft;
            y = -y;
            // bottom side
            for( j = -maxRange.width; j <= maxRange.width; j++, ss_count++, x++ )
            {
                ss[ss_count].x = x;
                ss[ss_count].y = y;
            }
        }
    }
    else if( maxRange.width > maxRange.height )
    {
        int yupper = -min_count;

        // cycle by neighbor rings
        for( i = min_count; i < maxRange.width; i++ )
        {
            // two cycles by y
            int x = -(i + 1);
            int y = yupper;

            // left side
            for( j = -maxRange.height; j <= maxRange.height; j++, ss_count++, y++ )
            {
                ss[ss_count].x = x;
                ss[ss_count].y = y;
            }

            y = yupper;
            x = -x;
            // right side
            for( j = -maxRange.height; j <= maxRange.height; j++, ss_count++, y++ )
            {
                ss[ss_count].x = x;
                ss[ss_count].y = y;
            }
        }
    }

    int maxX = srcB->cols - blockSize.width, maxY = srcB->rows - blockSize.height;
    const uchar* Adata = srcA->data.ptr;
    const uchar* Bdata = srcB->data.ptr;
    int Astep = srcA->step, Bstep = srcB->step;

    // compute the flow
    for( i = 0; i < velx->rows; i++ )
    {
        float* vx = (float*)(velx->data.ptr + velx->step*i);
        float* vy = (float*)(vely->data.ptr + vely->step*i);

        for( j = 0; j < velx->cols; j++ )
        {
            int X1 = j*shiftSize.width, Y1 = i*shiftSize.height, X2, Y2;
            int offX = 0, offY = 0;

            if( usePrevious )
            {
                offX = cvRound(vx[j]);
                offY = cvRound(vy[j]);
            }

            int k;
            for( k = 0; k < blHeight; k++ )
                memcpy( blockA + k*blWidth, Adata + Astep*(Y1 + k) + X1, blWidth );

            X2 = X1 + offX;
            Y2 = Y1 + offY;
            int dist = INT_MAX;
            if( 0 <= X2 && X2 <= maxX && 0 <= Y2 && Y2 <= maxY )
                dist = cmpBlocks( blockA, Bdata + Bstep*Y2 + X2, Bstep, blockSize );

            int countMin = 1;
            int sumx = offX, sumy = offY;

            if( dist > acceptLevel )
            {
                // do brute-force search
                for( k = 0; k < ss_count; k++ )
                {
                    int dx = offX + ss[k].x;
                    int dy = offY + ss[k].y;
                    X2 = X1 + dx;
                    Y2 = Y1 + dy;

                    if( !(0 <= X2 && X2 <= maxX && 0 <= Y2 && Y2 <= maxY) )
                        continue;

                    int tmpDist = cmpBlocks( blockA, Bdata + Bstep*Y2 + X2, Bstep, blockSize );
                    if( tmpDist < acceptLevel )
                    {
                        sumx = dx; sumy = dy;
                        countMin = 1;
                        break;
                    }

                    if( tmpDist < dist )
                    {
                        dist = tmpDist;
                        sumx = dx; sumy = dy;
                        countMin = 1;
                    }
                    else if( tmpDist == dist )
                    {
                        sumx += dx; sumy += dy;
                        countMin++;
                    }
                }

                if( dist > escapeLevel )
                {
                    sumx = offX;
                    sumy = offY;
                    countMin = 1;
                }
            }

            vx[j] = (float)sumx/countMin;
            vy[j] = (float)sumy/countMin;
        }
    }
}

/* End of file. */
