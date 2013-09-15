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

/****************************************************************************************\
    The code below is some modification of Stan Birchfield's algorithm described in:

    Depth Discontinuities by Pixel-to-Pixel Stereo
    Stan Birchfield and Carlo Tomasi
    International Journal of Computer Vision,
    35(3): 269-293, December 1999.

    This implementation uses different cost function that results in
    O(pixPerRow*maxDisparity) complexity of dynamic programming stage versus
    O(pixPerRow*log(pixPerRow)*maxDisparity) in the above paper.
\****************************************************************************************/

/****************************************************************************************\
*       Find stereo correspondence by dynamic programming algorithm                      *
\****************************************************************************************/
#define ICV_DP_STEP_LEFT  0
#define ICV_DP_STEP_UP    1
#define ICV_DP_STEP_DIAG  2

#define ICV_BIRCH_DIFF_LUM 5

#define ICV_MAX_DP_SUM_VAL (INT_MAX/4)

typedef struct _CvDPCell
{
    uchar  step; //local-optimal step
    int    sum;  //current sum
}_CvDPCell;

typedef struct _CvRightImData
{
    uchar min_val, max_val;
} _CvRightImData;

#define CV_IMAX3(a,b,c) ((temp2 = (a) >= (b) ? (a) : (b)),(temp2 >= (c) ? temp2 : (c)))
#define CV_IMIN3(a,b,c) ((temp3 = (a) <= (b) ? (a) : (b)),(temp3 <= (c) ? temp3 : (c)))

static void icvFindStereoCorrespondenceByBirchfieldDP( uchar* src1, uchar* src2,
                                                uchar* disparities,
                                                CvSize size, int widthStep,
                                                int    maxDisparity,
                                                float  _param1, float _param2,
                                                float  _param3, float _param4,
                                                float  _param5 )
{
    int     x, y, i, j, temp2, temp3;
    int     d, s;
    int     dispH =  maxDisparity + 3;
    uchar  *dispdata;
    int     imgW = size.width;
    int     imgH = size.height;
    uchar   val, prevval, prev, curr;
    int     min_val;
    uchar*  dest = disparities;
    int param1 = cvRound(_param1);
    int param2 = cvRound(_param2);
    int param3 = cvRound(_param3);
    int param4 = cvRound(_param4);
    int param5 = cvRound(_param5);

    #define CELL(d,x)   cells[(d)+(x)*dispH]

    uchar*              dsi = (uchar*)cvAlloc(sizeof(uchar)*imgW*dispH);
    uchar*              edges = (uchar*)cvAlloc(sizeof(uchar)*imgW*imgH);
    _CvDPCell*          cells = (_CvDPCell*)cvAlloc(sizeof(_CvDPCell)*imgW*MAX(dispH,(imgH+1)/2));
    _CvRightImData*     rData = (_CvRightImData*)cvAlloc(sizeof(_CvRightImData)*imgW);
    int*                reliabilities = (int*)cells;

    for( y = 0; y < imgH; y++ )
    {
        uchar* srcdata1 = src1 + widthStep * y;
        uchar* srcdata2 = src2 + widthStep * y;

        //init rData
        prevval = prev = srcdata2[0];
        for( j = 1; j < imgW; j++ )
        {
            curr = srcdata2[j];
            val = (uchar)((curr + prev)>>1);
            rData[j-1].max_val = (uchar)CV_IMAX3( val, prevval, prev );
            rData[j-1].min_val = (uchar)CV_IMIN3( val, prevval, prev );
            prevval = val;
            prev = curr;
        }
        rData[j-1] = rData[j-2];//last elem

        // fill dissimularity space image
        for( i = 1; i <= maxDisparity + 1; i++ )
        {
            dsi += imgW;
            rData--;
            for( j = i - 1; j < imgW - 1; j++ )
            {
                int t;
                if( (t = srcdata1[j] - rData[j+1].max_val) >= 0 )
                {
                    dsi[j] = (uchar)t;
                }
                else if( (t = rData[j+1].min_val - srcdata1[j]) >= 0 )
                {
                    dsi[j] = (uchar)t;
                }
                else
                {
                    dsi[j] = 0;
                }
            }
        }
        dsi -= (maxDisparity+1)*imgW;
        rData += maxDisparity+1;

        //intensity gradients image construction
        //left row
        edges[y*imgW] = edges[y*imgW+1] = edges[y*imgW+2] = 2;
        edges[y*imgW+imgW-1] = edges[y*imgW+imgW-2] = edges[y*imgW+imgW-3] = 1;
        for( j = 3; j < imgW-4; j++ )
        {
            edges[y*imgW+j] = 0;

            if( ( CV_IMAX3( srcdata1[j-3], srcdata1[j-2], srcdata1[j-1] ) -
                  CV_IMIN3( srcdata1[j-3], srcdata1[j-2], srcdata1[j-1] ) ) >= ICV_BIRCH_DIFF_LUM )
            {
                edges[y*imgW+j] |= 1;
            }
            if( ( CV_IMAX3( srcdata2[j+3], srcdata2[j+2], srcdata2[j+1] ) -
                  CV_IMIN3( srcdata2[j+3], srcdata2[j+2], srcdata2[j+1] ) ) >= ICV_BIRCH_DIFF_LUM )
            {
                edges[y*imgW+j] |= 2;
            }
        }

        //find correspondence using dynamical programming
        //init DP table
        for( x = 0; x < imgW; x++ )
        {
            CELL(0,x).sum = CELL(dispH-1,x).sum = ICV_MAX_DP_SUM_VAL;
            CELL(0,x).step = CELL(dispH-1,x).step = ICV_DP_STEP_LEFT;
        }
        for( d = 2; d < dispH; d++ )
        {
            CELL(d,d-2).sum = ICV_MAX_DP_SUM_VAL;
            CELL(d,d-2).step = ICV_DP_STEP_UP;
        }
        CELL(1,0).sum  = 0;
        CELL(1,0).step = ICV_DP_STEP_LEFT;

        for( x = 1; x < imgW; x++ )
        {
            int dp = MIN( x + 1, maxDisparity + 1);
            uchar* _edges = edges + y*imgW + x;
            int e0 = _edges[0] & 1;
            _CvDPCell* _cell = cells + x*dispH;

            do
            {
                int _s = dsi[dp*imgW+x];
                int sum[3];

                //check left step
                sum[0] = _cell[dp-dispH].sum - param2;

                //check up step
                if( _cell[dp+1].step != ICV_DP_STEP_DIAG && e0 )
                {
                    sum[1] = _cell[dp+1].sum + param1;

                    if( _cell[dp-1-dispH].step != ICV_DP_STEP_UP && (_edges[1-dp] & 2) )
                    {
                        int t;

                        sum[2] = _cell[dp-1-dispH].sum + param1;

                        t = sum[1] < sum[0];

                        //choose local-optimal pass
                        if( sum[t] <= sum[2] )
                        {
                            _cell[dp].step = (uchar)t;
                            _cell[dp].sum = sum[t] + _s;
                        }
                        else
                        {
                            _cell[dp].step = ICV_DP_STEP_DIAG;
                            _cell[dp].sum = sum[2] + _s;
                        }
                    }
                    else
                    {
                        if( sum[0] <= sum[1] )
                        {
                            _cell[dp].step = ICV_DP_STEP_LEFT;
                            _cell[dp].sum = sum[0] + _s;
                        }
                        else
                        {
                            _cell[dp].step = ICV_DP_STEP_UP;
                            _cell[dp].sum = sum[1] + _s;
                        }
                    }
                }
                else if( _cell[dp-1-dispH].step != ICV_DP_STEP_UP && (_edges[1-dp] & 2) )
                {
                    sum[2] = _cell[dp-1-dispH].sum + param1;
                    if( sum[0] <= sum[2] )
                    {
                        _cell[dp].step = ICV_DP_STEP_LEFT;
                        _cell[dp].sum = sum[0] + _s;
                    }
                    else
                    {
                        _cell[dp].step = ICV_DP_STEP_DIAG;
                        _cell[dp].sum = sum[2] + _s;
                    }
                }
                else
                {
                    _cell[dp].step = ICV_DP_STEP_LEFT;
                    _cell[dp].sum = sum[0] + _s;
                }
            }
            while( --dp );
        }// for x

        //extract optimal way and fill disparity image
        dispdata = dest + widthStep * y;

        //find min_val
        min_val = ICV_MAX_DP_SUM_VAL;
        for( i = 1; i <= maxDisparity + 1; i++ )
        {
            if( min_val > CELL(i,imgW-1).sum )
            {
                d = i;
                min_val = CELL(i,imgW-1).sum;
            }
        }

        //track optimal pass
        for( x = imgW - 1; x > 0; x-- )
        {
            dispdata[x] = (uchar)(d - 1);
            while( CELL(d,x).step == ICV_DP_STEP_UP ) d++;
            if ( CELL(d,x).step == ICV_DP_STEP_DIAG )
            {
                s = x;
                while( CELL(d,x).step == ICV_DP_STEP_DIAG )
                {
                    d--;
                    x--;
                }
                for( i = x; i < s; i++ )
                {
                    dispdata[i] = (uchar)(d-1);
                }
            }
        }//for x
    }// for y

    //Postprocessing the Disparity Map

    //remove obvious errors in the disparity map
    for( x = 0; x < imgW; x++ )
    {
        for( y = 1; y < imgH - 1; y++ )
        {
            if( dest[(y-1)*widthStep+x] == dest[(y+1)*widthStep+x] )
            {
                dest[y*widthStep+x] = dest[(y-1)*widthStep+x];
            }
        }
    }

    //compute intensity Y-gradients
    for( x = 0; x < imgW; x++ )
    {
        for( y = 1; y < imgH - 1; y++ )
        {
            if( ( CV_IMAX3( src1[(y-1)*widthStep+x], src1[y*widthStep+x],
                        src1[(y+1)*widthStep+x] ) -
                  CV_IMIN3( src1[(y-1)*widthStep+x], src1[y*widthStep+x],
                        src1[(y+1)*widthStep+x] ) ) >= ICV_BIRCH_DIFF_LUM )
            {
                edges[y*imgW+x] |= 4;
                edges[(y+1)*imgW+x] |= 4;
                edges[(y-1)*imgW+x] |= 4;
                y++;
            }
        }
    }

    //remove along any particular row, every gradient
    //for which two adjacent columns do not agree.
    for( y = 0; y < imgH; y++ )
    {
        prev = edges[y*imgW];
        for( x = 1; x < imgW - 1; x++ )
        {
            curr = edges[y*imgW+x];
            if( (curr & 4) &&
                ( !( prev & 4 ) ||
                  !( edges[y*imgW+x+1] & 4 ) ) )
            {
                edges[y*imgW+x] -= 4;
            }
            prev = curr;
        }
    }

    // define reliability
    for( x = 0; x < imgW; x++ )
    {
        for( y = 1; y < imgH; y++ )
        {
            i = y - 1;
            for( ; y < imgH && dest[y*widthStep+x] == dest[(y-1)*widthStep+x]; y++ )
                ;
            s = y - i;
            for( ; i < y; i++ )
            {
                reliabilities[i*imgW+x] = s;
            }
        }
    }

    //Y - propagate reliable regions
    for( x = 0; x < imgW; x++ )
    {
        for( y = 0; y < imgH; y++ )
        {
            d = dest[y*widthStep+x];
            if( reliabilities[y*imgW+x] >= param4 && !(edges[y*imgW+x] & 4) &&
                d > 0 )//highly || moderately
            {
                disparities[y*widthStep+x] = (uchar)d;
                //up propagation
                for( i = y - 1; i >= 0; i-- )
                {
                    if(  ( edges[i*imgW+x] & 4 ) ||
                         ( dest[i*widthStep+x] < d &&
                           reliabilities[i*imgW+x] >= param3 ) ||
                         ( reliabilities[y*imgW+x] < param5 &&
                           dest[i*widthStep+x] - 1 == d ) ) break;

                    disparities[i*widthStep+x] = (uchar)d;
                }

                //down propagation
                for( i = y + 1; i < imgH; i++ )
                {
                    if(  ( edges[i*imgW+x] & 4 ) ||
                         ( dest[i*widthStep+x] < d &&
                           reliabilities[i*imgW+x] >= param3 ) ||
                         ( reliabilities[y*imgW+x] < param5 &&
                           dest[i*widthStep+x] - 1 == d ) ) break;

                    disparities[i*widthStep+x] = (uchar)d;
                }
                y = i - 1;
            }
            else
            {
                disparities[y*widthStep+x] = (uchar)d;
            }
        }
    }

    // define reliability along X
    for( y = 0; y < imgH; y++ )
    {
        for( x = 1; x < imgW; x++ )
        {
            i = x - 1;
            for( ; x < imgW && dest[y*widthStep+x] == dest[y*widthStep+x-1]; x++ ) {}
            s = x - i;
            for( ; i < x; i++ )
            {
                reliabilities[y*imgW+i] = s;
            }
        }
    }

    //X - propagate reliable regions
    for( y = 0; y < imgH; y++ )
    {
        for( x = 0; x < imgW; x++ )
        {
            d = dest[y*widthStep+x];
            if( reliabilities[y*imgW+x] >= param4 && !(edges[y*imgW+x] & 1) &&
                d > 0 )//highly || moderately
            {
                disparities[y*widthStep+x] = (uchar)d;
                //up propagation
                for( i = x - 1; i >= 0; i-- )
                {
                    if(  (edges[y*imgW+i] & 1) ||
                         ( dest[y*widthStep+i] < d &&
                           reliabilities[y*imgW+i] >= param3 ) ||
                         ( reliabilities[y*imgW+x] < param5 &&
                           dest[y*widthStep+i] - 1 == d ) ) break;

                    disparities[y*widthStep+i] = (uchar)d;
                }

                //down propagation
                for( i = x + 1; i < imgW; i++ )
                {
                    if(  (edges[y*imgW+i] & 1) ||
                         ( dest[y*widthStep+i] < d &&
                           reliabilities[y*imgW+i] >= param3 ) ||
                         ( reliabilities[y*imgW+x] < param5 &&
                           dest[y*widthStep+i] - 1 == d ) ) break;

                    disparities[y*widthStep+i] = (uchar)d;
                }
                x = i - 1;
            }
            else
            {
                disparities[y*widthStep+x] = (uchar)d;
            }
        }
    }

    //release resources
    cvFree( &dsi );
    cvFree( &edges );
    cvFree( &cells );
    cvFree( &rData );
}


/*F///////////////////////////////////////////////////////////////////////////
//
//    Name:    cvFindStereoCorrespondence
//    Purpose: find stereo correspondence on stereo-pair
//    Context:
//    Parameters:
//      leftImage - left image of stereo-pair (format 8uC1).
//      rightImage - right image of stereo-pair (format 8uC1).
//      mode -mode of correspondance retrieval (now CV_RETR_DP_BIRCHFIELD only)
//      dispImage - destination disparity image
//      maxDisparity - maximal disparity
//      param1, param2, param3, param4, param5 - parameters of algorithm
//    Returns:
//    Notes:
//      Images must be rectified.
//      All images must have format 8uC1.
//F*/
CV_IMPL void
cvFindStereoCorrespondence(
                   const  CvArr* leftImage, const  CvArr* rightImage,
                   int     mode,
                   CvArr*  depthImage,
                   int     maxDisparity,
                   double  param1, double  param2, double  param3,
                   double  param4, double  param5  )
{
    CV_FUNCNAME( "cvFindStereoCorrespondence" );

    __BEGIN__;

    CvMat  *src1, *src2;
    CvMat  *dst;
    CvMat  src1_stub, src2_stub, dst_stub;
    int    coi;

    CV_CALL( src1 = cvGetMat( leftImage, &src1_stub, &coi ));
    if( coi ) CV_ERROR( CV_BadCOI, "COI is not supported by the function" );
    CV_CALL( src2 = cvGetMat( rightImage, &src2_stub, &coi ));
    if( coi ) CV_ERROR( CV_BadCOI, "COI is not supported by the function" );
    CV_CALL( dst = cvGetMat( depthImage, &dst_stub, &coi ));
    if( coi ) CV_ERROR( CV_BadCOI, "COI is not supported by the function" );

    // check args
    if( CV_MAT_TYPE( src1->type ) != CV_8UC1 ||
        CV_MAT_TYPE( src2->type ) != CV_8UC1 ||
        CV_MAT_TYPE( dst->type ) != CV_8UC1) CV_ERROR(CV_StsUnsupportedFormat,
                        "All images must be single-channel and have 8u" );

    if( !CV_ARE_SIZES_EQ( src1, src2 ) || !CV_ARE_SIZES_EQ( src1, dst ) )
            CV_ERROR( CV_StsUnmatchedSizes, "" );

    if( maxDisparity <= 0 || maxDisparity >= src1->width || maxDisparity > 255 )
        CV_ERROR(CV_StsOutOfRange,
                 "parameter /maxDisparity/ is out of range");

    if( mode == CV_DISPARITY_BIRCHFIELD )
    {
        if( param1 == CV_UNDEF_SC_PARAM ) param1 = CV_IDP_BIRCHFIELD_PARAM1;
        if( param2 == CV_UNDEF_SC_PARAM ) param2 = CV_IDP_BIRCHFIELD_PARAM2;
        if( param3 == CV_UNDEF_SC_PARAM ) param3 = CV_IDP_BIRCHFIELD_PARAM3;
        if( param4 == CV_UNDEF_SC_PARAM ) param4 = CV_IDP_BIRCHFIELD_PARAM4;
        if( param5 == CV_UNDEF_SC_PARAM ) param5 = CV_IDP_BIRCHFIELD_PARAM5;

        CV_CALL( icvFindStereoCorrespondenceByBirchfieldDP( src1->data.ptr,
            src2->data.ptr, dst->data.ptr,
            cvGetMatSize( src1 ), src1->step,
            maxDisparity, (float)param1, (float)param2, (float)param3,
            (float)param4, (float)param5 ) );
    }
    else
    {
        CV_ERROR( CV_StsBadArg, "Unsupported mode of function" );
    }

    __END__;
}

/* End of file. */
