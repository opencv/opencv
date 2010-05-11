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

typedef struct _PointInfo
{
    CvPoint pt;
    int left_neigh;
    int right_neigh;

}
icvPointInfo;


static CvStatus
icvFindDominantPointsIPAN( CvSeq * contour,
                           CvMemStorage * storage,
                           CvSeq ** corners, int dmin2, int dmax2, int dneigh2, float amax )
{
    CvStatus status = CV_OK;

    /* variables */
    int n = contour->total;

    float *sharpness;
    float *distance;
    icvPointInfo *ptInf;

    int i, j, k;

    CvSeqWriter writer;

    float mincos = (float) cos( 3.14159265359 * amax / 180 );

    /* check bad arguments */
    if( contour == NULL )
        return CV_NULLPTR_ERR;
    if( storage == NULL )
        return CV_NULLPTR_ERR;
    if( corners == NULL )
        return CV_NULLPTR_ERR;
    if( dmin2 < 0 )
        return CV_BADSIZE_ERR;
    if( dmax2 < dmin2 )
        return CV_BADSIZE_ERR;
    if( (dneigh2 > dmax2) || (dneigh2 < 0) )
        return CV_BADSIZE_ERR;
    if( (amax < 0) || (amax > 180) )
        return CV_BADSIZE_ERR;

    sharpness = (float *) cvAlloc( n * sizeof( float ));
    distance = (float *) cvAlloc( n * sizeof( float ));

    ptInf = (icvPointInfo *) cvAlloc( n * sizeof( icvPointInfo ));

/*****************************************************************************************/
/*                                 First pass                                            */
/*****************************************************************************************/

    if( CV_IS_SEQ_CHAIN_CONTOUR( contour ))
    {
        CvChainPtReader reader;

        cvStartReadChainPoints( (CvChain *) contour, &reader );

        for( i = 0; i < n; i++ )
        {
            CV_READ_CHAIN_POINT( ptInf[i].pt, reader );
        }
    }
    else if( CV_IS_SEQ_POINT_SET( contour ))
    {
        CvSeqReader reader;

        cvStartReadSeq( contour, &reader, 0 );

        for( i = 0; i < n; i++ )
        {
            CV_READ_SEQ_ELEM( ptInf[i].pt, reader );
        }
    }
    else
    {
        return CV_BADFLAG_ERR;
    }

    for( i = 0; i < n; i++ )
    {
        /* find nearest suitable points
           which satisfy distance constraint >dmin */
        int left_near = 0;
        int right_near = 0;
        int left_far, right_far;

        float dist_l = 0;
        float dist_r = 0;

        int i_plus = 0;
        int i_minus = 0;

        float max_cos_alpha;

        /* find  right minimum */
        while( dist_r < dmin2 )
        {
            float dx, dy;
            int ind;

            if( i_plus >= n )
                goto error;

            right_near = i_plus;

            if( dist_r < dneigh2 )
                ptInf[i].right_neigh = i_plus;

            i_plus++;

            ind = (i + i_plus) % n;
            dx = (float) (ptInf[i].pt.x - ptInf[ind].pt.x);
            dy = (float) (ptInf[i].pt.y - ptInf[ind].pt.y);
            dist_r = dx * dx + dy * dy;
        }
        /* find right maximum */
        while( dist_r <= dmax2 )
        {
            float dx, dy;
            int ind;

            if( i_plus >= n )
                goto error;

            distance[(i + i_plus) % n] = cvSqrt( dist_r );

            if( dist_r < dneigh2 )
                ptInf[i].right_neigh = i_plus;

            i_plus++;

            right_far = i_plus;

            ind = (i + i_plus) % n;

            dx = (float) (ptInf[i].pt.x - ptInf[ind].pt.x);
            dy = (float) (ptInf[i].pt.y - ptInf[ind].pt.y);
            dist_r = dx * dx + dy * dy;
        }
        right_far = i_plus;

        /* left minimum */
        while( dist_l < dmin2 )
        {
            float dx, dy;
            int ind;

            if( i_minus <= -n )
                goto error;

            left_near = i_minus;

            if( dist_l < dneigh2 )
                ptInf[i].left_neigh = i_minus;

            i_minus--;

            ind = i + i_minus;
            ind = (ind < 0) ? (n + ind) : ind;

            dx = (float) (ptInf[i].pt.x - ptInf[ind].pt.x);
            dy = (float) (ptInf[i].pt.y - ptInf[ind].pt.y);
            dist_l = dx * dx + dy * dy;
        }

        /* find left maximum */
        while( dist_l <= dmax2 )
        {
            float dx, dy;
            int ind;

            if( i_minus <= -n )
                goto error;

            ind = i + i_minus;
            ind = (ind < 0) ? (n + ind) : ind;

            distance[ind] = cvSqrt( dist_l );

            if( dist_l < dneigh2 )
                ptInf[i].left_neigh = i_minus;

            i_minus--;

            left_far = i_minus;

            ind = i + i_minus;
            ind = (ind < 0) ? (n + ind) : ind;

            dx = (float) (ptInf[i].pt.x - ptInf[ind].pt.x);
            dy = (float) (ptInf[i].pt.y - ptInf[ind].pt.y);
            dist_l = dx * dx + dy * dy;
        }
        left_far = i_minus;

        if( (i_plus - i_minus) > n + 2 )
            goto error;

        max_cos_alpha = -1;
        for( j = left_far + 1; j < left_near; j++ )
        {
            float dx, dy;
            float a, a2;
            int leftind = i + j;

            leftind = (leftind < 0) ? (n + leftind) : leftind;

            a = distance[leftind];
            a2 = a * a;

            for( k = right_near + 1; k < right_far; k++ )
            {
                int ind = (i + k) % n;
                float c2, cosalpha;
                float b = distance[ind];
                float b2 = b * b;

                /* compute cosinus */
                dx = (float) (ptInf[leftind].pt.x - ptInf[ind].pt.x);
                dy = (float) (ptInf[leftind].pt.y - ptInf[ind].pt.y);

                c2 = dx * dx + dy * dy;
                cosalpha = (a2 + b2 - c2) / (2 * a * b);

                max_cos_alpha = MAX( max_cos_alpha, cosalpha );

                if( max_cos_alpha < mincos )
                    max_cos_alpha = -1;

                sharpness[i] = max_cos_alpha;
            }
        }
    }
/*****************************************************************************************/
/*                                 Second pass                                           */
/*****************************************************************************************/

    cvStartWriteSeq( (contour->flags & ~CV_SEQ_ELTYPE_MASK) | CV_SEQ_ELTYPE_INDEX,
                     sizeof( CvSeq ), sizeof( int ), storage, &writer );

    /* second pass - nonmaxima suppression */
    /* neighborhood of point < dneigh2 */
    for( i = 0; i < n; i++ )
    {
        int suppressed = 0;
        if( sharpness[i] == -1 )
            continue;

        for( j = 1; (j <= ptInf[i].right_neigh) && (suppressed == 0); j++ )
        {
            if( sharpness[i] < sharpness[(i + j) % n] )
                suppressed = 1;
        }

        for( j = -1; (j >= ptInf[i].left_neigh) && (suppressed == 0); j-- )
        {
            int ind = i + j;

            ind = (ind < 0) ? (n + ind) : ind;
            if( sharpness[i] < sharpness[ind] )
                suppressed = 1;
        }

        if( !suppressed )
            CV_WRITE_SEQ_ELEM( i, writer );
    }

    *corners = cvEndWriteSeq( &writer );

    cvFree( &sharpness );
    cvFree( &distance );
    cvFree( &ptInf );

    return status;

  error:
    /* dmax is so big (more than contour diameter)
       that algorithm could become infinite cycle */
    cvFree( &sharpness );
    cvFree( &distance );
    cvFree( &ptInf );

    return CV_BADRANGE_ERR;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvFindDominantPoints
//    Purpose:
//      Applies some algorithm to find dominant points ( corners ) of contour
//     
//    Context:
//    Parameters:
//      contours - pointer to input contour object.
//      out_numbers - array of dominant points indices
//      count - length of out_numbers array on input
//              and numbers of founded dominant points on output   
//                  
//      method - only CV_DOMINANT_IPAN now
//      parameters - array of parameters
//                   for IPAN algorithm
//                   [0] - minimal distance
//                   [1] - maximal distance
//                   [2] - neighborhood distance (must be not greater than dmaximal distance)
//                   [3] - maximal possible angle of curvature
//    Returns:
//      CV_OK or error code
//    Notes:
//      User must allocate out_numbers array. If it is small - function fills array 
//      with part of points and returns  error
//F*/
CV_IMPL CvSeq*
cvFindDominantPoints( CvSeq * contour, CvMemStorage * storage, int method,
                      double parameter1, double parameter2, double parameter3, double parameter4 )
{
    CvSeq* corners = 0;

    if( !contour )
        CV_Error( CV_StsNullPtr, "" );

    if( !storage )
        storage = contour->storage;

    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    switch (method)
    {
    case CV_DOMINANT_IPAN:
        {
            int dmin = cvRound(parameter1);
            int dmax = cvRound(parameter2);
            int dneigh = cvRound(parameter3);
            int amax = cvRound(parameter4);

            if( amax == 0 )
                amax = 150;
            if( dmin == 0 )
                dmin = 7;
            if( dmax == 0 )
                dmax = dmin + 2;
            if( dneigh == 0 )
                dneigh = dmin;

            IPPI_CALL( icvFindDominantPointsIPAN( contour, storage, &corners,
                                                  dmin*dmin, dmax*dmax, dneigh*dneigh, (float)amax ));
        }
        break;
    default:
        CV_Error( CV_StsBadArg, "" );
    }

    return corners;
}

/* End of file. */
