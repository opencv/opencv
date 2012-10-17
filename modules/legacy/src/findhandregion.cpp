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

#define _CV_NORM_L2(a) (float)(icvSqrt32f(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]))
#define _CV_NORM_L22(a) (float)(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

/****************************************************************************************\

   find region where hand is   (for gesture recognition)
   flag = 0 (use left bucket)  flag = 1 (use right bucket)

\****************************************************************************************/

static CvStatus CV_STDCALL
icvFindHandRegion( CvPoint3D32f * points, int count,
                   CvSeq * indexs,
                   float *line, CvSize2D32f size, int flag,
                   CvPoint3D32f * center,
                   CvMemStorage * storage, CvSeq ** numbers )
{

/*    IppmVect32f sub, cros;   */
    float *sub, *cros;
    CvSeqWriter writer;
    CvSeqReader reader;

    CvStatus status;
    int nbins = 20, i, l, i_point, left, right;
    int *bin_counts = 0;        //  pointer to the point's counter in the bickets
    int low_count;              //  low threshold

    CvPoint *tmp_number = 0, *pt;
    float value, vmin, vmax, vl, bsize, vc;
    float hand_length, hand_length2, hand_left, hand_right;
    float threshold, threshold2;
    float *vv = 0;
    float a[3];

    status = CV_OK;

    hand_length = size.width;
    hand_length2 = hand_length / 2;

    threshold = (float) (size.height * 3 / 5.);
    threshold2 = threshold * threshold;

/*    low_count = count/nbins;     */
    low_count = (int) (count / 60.);

    assert( points != NULL && line != NULL );
    if( points == NULL || line == NULL )
        return CV_NULLPTR_ERR;

    assert( count > 5 );
    if( count < 5 )
        return CV_BADFLAG_ERR;

    assert( flag == 0 || flag == 1 );
    if( flag != 0 && flag != 1 )
        return CV_BADFLAG_ERR;

/*  create vectors         */
    sub = icvCreateVector_32f( 3 );
    cros = icvCreateVector_32f( 3 );
    if( sub == NULL || cros == NULL )
        return CV_OUTOFMEM_ERR;

/*  alloc memory for the point's projections on the line    */
    vv = (float *) cvAlloc( count * sizeof( float ));

    if( vv == NULL )
        return CV_OUTOFMEM_ERR;

/*  alloc memory for the point's counter in the bickets     */
    bin_counts = (int *) cvAlloc( nbins * sizeof( int ));

    if( bin_counts == NULL )
    {
        status = CV_OUTOFMEM_ERR;
        goto M_END;
    }
    memset( bin_counts, 0, nbins * sizeof( int ));

    cvStartReadSeq( indexs, &reader, 0 );

/*  alloc memory for the temporale point's numbers      */
    tmp_number = (CvPoint *) cvAlloc( count * sizeof( CvPoint ));
    if( tmp_number == NULL )
    {
        status = CV_OUTOFMEM_ERR;
        goto M_END;
    }

/*  find min and max point's projection on the line     */
    vmin = 1000;
    vmax = -1000;
    i_point = 0;
    for( i = 0; i < count; i++ )
    {
/*
        icvSubVector_32f ((IppmVect32f )&points[i], (IppmVect32f )&line[3], sub, 3);

        icvCrossProduct2L_32f ((IppmVect32f )&line[0], sub, cros);
*/

        sub[0] = points[i].x - line[3];
        sub[1] = points[i].y - line[4];
        sub[2] = points[i].z - line[5];
        a[0] = sub[0] * line[1] - sub[1] * line[0];
        a[1] = sub[1] * line[2] - sub[2] * line[1];
        a[2] = sub[2] * line[0] - sub[0] * line[2];

/*      if(IPPI_NORM_L22 ( cros ) < threshold2)    */
        if( _CV_NORM_L22( a ) < threshold2 )
        {
            value = (float)icvDotProduct_32f( sub, &line[0], 3 );
            if( value > vmax )
                vmax = value;
            if( value < vmin )
                vmin = value;

            vv[i_point] = value;

            pt = (CvPoint*)cvGetSeqElem( indexs, i );
            tmp_number[i_point] = *pt;
            i_point++;
        }
    }

/*  compute the length of one bucket             */
    vl = vmax - vmin;
    bsize = vl / nbins;

/*  compute the number of points in each bucket   */
    for( i = 0; i < i_point; i++ )
    {
        l = cvRound( (vv[i] - vmin) / bsize );
        bin_counts[l]++;
    }

    *numbers = cvCreateSeq( CV_SEQ_POINT_SET, sizeof( CvSeq ), sizeof( CvPoint ), storage );
    assert( numbers != 0 );
    if( numbers == NULL )
    {
        status = CV_OUTOFMEM_ERR;
        goto M_END;
    }

    cvStartAppendToSeq( *numbers, &writer );

    if( flag == 0 )
    {
/*  find the leftmost bucket           */
        for( l = 0; l < nbins; l++ )
        {
            if( bin_counts[l] > low_count )
                break;
        }
        left = l;

/*  compute center point of the left hand     */
        hand_left = vmin + left * bsize;
        vc = hand_left + hand_length2;
        hand_right = hand_left + hand_length;
    }
    else
    {
/*  find the rightmost bucket                */
        for( l = nbins - 1; l >= 0; l-- )
        {
            if( bin_counts[l] > low_count )
                break;
        }
        right = l;

/*  compute center point of the right hand    */
        hand_right = vmax - (nbins - right - 1) * bsize;
        vc = hand_right - hand_length2;
        hand_left = hand_right - hand_length;
    }

    icvScaleVector_32f( &line[0], sub, 3, vc );
    icvAddVector_32f( &line[3], sub, (float *) center, 3 );

/*  select hand's points and calculate mean value     */

    //ss.x = ss.y = ss.z = 0;
    for( l = 0; l < i_point; l++ )
    {
        if( vv[l] >= hand_left && vv[l] <= hand_right )
        {
            CV_WRITE_SEQ_ELEM( tmp_number[l], writer );

        }
    }

    cvEndWriteSeq( &writer );

  M_END:
    if( tmp_number != NULL )
        cvFree( &tmp_number );
    if( bin_counts != NULL )
        cvFree( &bin_counts );
    if( vv != NULL )
        cvFree( &vv );
    if( sub != NULL ) icvDeleteVector (sub);
    if( cros != NULL ) icvDeleteVector (cros);

    return status;

}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


#define _CV_NORM_L31(a) (float)(icvSqrt32f(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]))
#define _CV_NORM_L32(a) (float)(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

/****************************************************************************************\

   find region where hand is   (for gesture recognition)
   flag = 0 (use left bucket)  flag = 1 (use right bucket)

\****************************************************************************************/

static CvStatus CV_STDCALL
icvFindHandRegionA( CvPoint3D32f * points, int count,
                    CvSeq * indexs,
                    float *line, CvSize2D32f size, int jc,
                    CvPoint3D32f * center,
                    CvMemStorage * storage, CvSeq ** numbers )
{

/*    IppmVect32f sub, cros;   */
    float *sub, *cros;
    float eps = (float) 0.01;
    CvSeqWriter writer;
    CvSeqReader reader;

    CvStatus status;
    float gor[3] = { 1, 0, 0 };
    float ver[3] = { 0, 1, 0 };

    int nbins = 20, i, l, i_point, left, right, jmin, jmax, jl;
    int j_left, j_right;
    int *bin_counts = 0;        //  pointer to the point's counter in the bickets

//    int *bin_countsj = 0;   //  pointer to the index's counter in the bickets
    int low_count;              //  low threshold

    CvPoint *tmp_number = 0, *pt;
    float value, vmin, vmax, vl, bsize, bsizej, vc, vcl, vcr;
    double v_ver, v_gor;
    float hand_length, hand_length2, hand_left, hand_right;
    float threshold, threshold2;
    float *vv = 0;
    float a[3];
    char log;

    status = CV_OK;

    hand_length = size.width;
    hand_length2 = hand_length / 2;

    threshold = (float) (size.height * 3 / 5.);
    threshold2 = threshold * threshold;

/*    low_count = count/nbins;     */
    low_count = (int) (count / 60.);

    assert( points != NULL && line != NULL );
    if( points == NULL || line == NULL )
        return CV_NULLPTR_ERR;

    assert( count > 5 );
    if( count < 5 )
        return CV_BADFLAG_ERR;

/*  create vectors         */
    sub = icvCreateVector_32f( 3 );
    cros = icvCreateVector_32f( 3 );
    if( sub == NULL || cros == NULL )
        return CV_OUTOFMEM_ERR;

/*  alloc memory for the point's projections on the line    */
    vv = (float *) cvAlloc( count * sizeof( float ));

    if( vv == NULL )
        return CV_OUTOFMEM_ERR;

/*  alloc memory for the point's counter in the bickets     */
    bin_counts = (int *) cvAlloc( nbins * sizeof( int ));

    if( bin_counts == NULL )
    {
        status = CV_OUTOFMEM_ERR;
        goto M_END;
    }
    memset( bin_counts, 0, nbins * sizeof( int ));

/*  alloc memory for the point's counter in the bickets     */
//    bin_countsj = (int*) icvAlloc(nbins*sizeof(int));
//    if(bin_countsj == NULL) {status = CV_OUTOFMEM_ERR; goto M_END;}
//    memset(bin_countsj,0,nbins*sizeof(int));

    cvStartReadSeq( indexs, &reader, 0 );

/*  alloc memory for the temporale point's numbers      */
    tmp_number = (CvPoint *) cvAlloc( count * sizeof( CvPoint ));
    if( tmp_number == NULL )
    {
        status = CV_OUTOFMEM_ERR;
        goto M_END;
    }

/*  find min and max point's projection on the line     */
    vmin = 1000;
    vmax = -1000;
    jmin = 1000;
    jmax = -1000;
    i_point = 0;
    for( i = 0; i < count; i++ )
    {
/*
        icvSubVector_32f ((IppmVect32f )&points[i], (IppmVect32f )&line[3], sub, 3);

        icvCrossProduct2L_32f ((IppmVect32f )&line[0], sub, cros);
*/

        sub[0] = points[i].x - line[3];
        sub[1] = points[i].y - line[4];
        sub[2] = points[i].z - line[5];

//      if(fabs(sub[0])<eps||fabs(sub[1])<eps||fabs(sub[2])<eps) continue;

        a[0] = sub[0] * line[1] - sub[1] * line[0];
        a[1] = sub[1] * line[2] - sub[2] * line[1];
        a[2] = sub[2] * line[0] - sub[0] * line[2];

        v_gor = icvDotProduct_32f( gor, &line[0], 3 );
        v_ver = icvDotProduct_32f( ver, &line[0], 3 );

        if( v_ver > v_gor )
            log = true;
        else
            log = false;


/*      if(IPPI_NORM_L22 ( cros ) < threshold2)    */
/*
        if(fabs(a[0])<eps && fabs(a[1])<eps && fabs(a[2])<eps)
        {
            icvDotProduct_32f( sub, &line[0], 3, &value);
            if(value > vmax) vmax = value;
            if(value < vmin) vmin = value;

            vv[i_point] = value;

            pt = (CvPoint* )icvGetSeqElem ( indexs, i, 0);

            if(pt->x > jmax) jmax = pt->x;
            if(pt->x < jmin) jmin = pt->x;

            tmp_number[i_point] = *pt;
            i_point++;
        }
        else
*/
        {
            if( _CV_NORM_L32( a ) < threshold2 )
            {
                value = (float)icvDotProduct_32f( sub, &line[0], 3 );
                if( value > vmax )
                    vmax = value;
                if( value < vmin )
                    vmin = value;

                vv[i_point] = value;

                pt = (CvPoint*)cvGetSeqElem( indexs, i );

                if( !log )
                {
                    if( pt->x > jmax )
                        jmax = pt->x;
                    if( pt->x < jmin )
                        jmin = pt->x;
                }
                else
                {
                    if( pt->y > jmax )
                        jmax = pt->y;
                    if( pt->y < jmin )
                        jmin = pt->y;
                }


                tmp_number[i_point] = *pt;
                i_point++;
            }
        }
    }

/*  compute the length of one bucket along the line        */
    vl = vmax - vmin;

/*  examining on the arm's existence  */
    if( vl < eps )
    {
        *numbers = NULL;
        status = CV_OK;
        goto M_END;
    }

    bsize = vl / nbins;

/*  compute the number of points in each bucket along the line  */
    for( i = 0; i < i_point; i++ )
    {
        l = cvRound( (vv[i] - vmin) / bsize );
        bin_counts[l]++;
    }

    /*  compute the length of one bucket along the X axe        */
    jl = jmax - jmin;
    if( jl <= 1 )
    {
        *numbers = NULL;
        status = CV_OK;
        goto M_END;
    }

    bsizej = (float) (jl / (nbins + 0.));

/*  compute the number of points in each bucket along the X axe */
//    for(i=0;i<i_point;i++)
//    {
//        l = cvRound((tmp_number[i].x - jmin)/bsizej);
//        bin_countsj[l]++;
//    }


    left = right = -1;

/*  find the leftmost and the rightmost buckets           */
    for( l = 0; l < nbins; l++ )
    {
        if( bin_counts[l] > low_count && left == -1 )
            left = l;
        else if( bin_counts[l] > low_count && left >= 0 )
            right = l;

    }

/*  compute center point of the left hand     */
    if( left == -1 && right == -1 )
    {
        *numbers = NULL;
        status = CV_OK;
        goto M_END;
    }

    hand_left = vmin + left * bsize;
    j_left = (int) (jmin + left * bsizej);

    vcl = hand_left + hand_length2;

/*  compute center point of the right hand    */
    hand_right = vmax - (nbins - right - 1) * bsize;
    vcr = hand_right - hand_length2;

    j_right = (int) (jmax - (nbins - right - 1) * bsizej);

    j_left = abs( j_left - jc );
    j_right = abs( j_right - jc );

    if( j_left <= j_right )
    {
        hand_right = hand_left + hand_length;
        vc = vcl;
    }
    else
    {
        hand_left = hand_right - hand_length;
        vc = vcr;
    }

    icvScaleVector_32f( &line[0], sub, 3, vc );
    icvAddVector_32f( &line[3], sub, (float *) center, 3 );

/*  select hand's points and calculate mean value     */
    *numbers = cvCreateSeq( CV_SEQ_POINT_SET, sizeof( CvSeq ), sizeof( CvPoint ), storage );
    assert( *numbers != 0 );
    if( *numbers == NULL )
    {
        status = CV_OUTOFMEM_ERR;
        goto M_END;
    }

    cvStartAppendToSeq( *numbers, &writer );

    for( l = 0; l < i_point; l++ )
    {
        if( vv[l] >= hand_left && vv[l] <= hand_right )
        {
            CV_WRITE_SEQ_ELEM( tmp_number[l], writer );

        }
    }

    cvEndWriteSeq( &writer );

  M_END:
    if( tmp_number != NULL )
        cvFree( &tmp_number );
//    if(bin_countsj != NULL) cvFree( &bin_countsj );
    if( bin_counts != NULL )
        cvFree( &bin_counts );

    if( vv != NULL )
        cvFree( &vv );

    if( sub != NULL ) icvDeleteVector (sub);
    if( cros != NULL ) icvDeleteVector (cros);

    return status;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:     cvFindHandRegion
//    Purpose:  finds hand region in range image data
//    Context:
//    Parameters:
//      points - pointer to the input point's set.
//      count  - the number of the input points.
//      indexs - pointer to the input sequence of the point's indexes
//      line   - pointer to the 3D-line
//      size   - size of the hand in meters
//      flag   - hand direction's flag (0 - left, -1 - right,
//               otherwise j-index of the initial image center)
//      center - pointer to the output hand center
//      storage - pointer to the memory storage
//      numbers - pointer to the output sequence of the point's indexes inside
//                hand region
//
//    Notes:
//F*/
CV_IMPL void
cvFindHandRegion( CvPoint3D32f * points, int count,
                  CvSeq * indexs,
                  float *line, CvSize2D32f size, int flag,
                  CvPoint3D32f * center, CvMemStorage * storage, CvSeq ** numbers )
{
    if(flag == 0 || flag == -1)
    {
        IPPI_CALL( icvFindHandRegion( points, count, indexs, line, size, -flag,
                                       center, storage, numbers ));
    }
    else
        IPPI_CALL( icvFindHandRegionA( points, count, indexs, line, size, flag,
                                        center, storage, numbers ));
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:     cvFindHandRegionA
//    Purpose:  finds hand region in range image data
//    Context:
//    Parameters:
//      points - pointer to the input point's set.
//      count  - the number of the input points.
//      indexs - pointer to the input sequence of the point's indexes
//      line   - pointer to the 3D-line
//      size   - size of the hand in meters
//      jc     - j-index of the initial image center
//      center - pointer to the output hand center
//      storage - pointer to the memory storage
//      numbers - pointer to the output sequence of the point's indexes inside
//                hand region
//
//    Notes:
//F*/
CV_IMPL void
cvFindHandRegionA( CvPoint3D32f * points, int count,
                   CvSeq * indexs,
                   float *line, CvSize2D32f size, int jc,
                   CvPoint3D32f * center, CvMemStorage * storage, CvSeq ** numbers )
{
    IPPI_CALL( icvFindHandRegionA( points, count, indexs, line, size, jc,
                                    center, storage, numbers ));
}

