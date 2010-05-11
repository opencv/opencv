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
#include "_vm.h"
#include <stdlib.h>
#include <assert.h>


/*======================================================================================*/

CvStatus
icvDynamicCorrespond( int *first,       /* first sequence of runs */
                      /* s0|w0|s1|w1|...|s(n-1)|w(n-1)|sn */
                      int first_runs,   /* number of runs */
                      int *second,      /* second sequence of runs */
                      int second_runs, int *first_corr, /* s0'|e0'|s1'|e1'|... */
                      int *second_corr )
{

    float Pd, Fi, S;
    float Occlusion;
    float *costTable;
    uchar *matchEdges;
    int prev;
    int curr;
    int baseIndex;
    int i, j;
    int i_1, j_1;
    int n;
    int l_beg, r_beg, l_end, r_end, l_len, r_len;
    int first_curr;
    int second_curr;
    int l_color, r_color;
    int len_color;
    float cost, cost1;
    float min1, min2, min3;
    float cmin;
    uchar cpath;
    int row_size;

    /* Test arguments for errors */

    if( (first == 0) ||
        (first_runs < 1) ||
        (second == 0) || (second_runs < 1) || (first_corr == 0) || (second_corr == 0) )

        return CV_BADFACTOR_ERR;


    Pd = 0.95f;
    Fi = (float) CV_PI;
    S = 1;

    Occlusion = (float) log( Pd * Fi / ((1 - Pd) * sqrt( fabs( (CV_PI * 2) * (1. / S) ))));

    costTable = (float *)cvAlloc( (first_runs + 1) * (second_runs + 1) * sizeof( float ));

    if( costTable == 0 )
        return CV_OUTOFMEM_ERR;

    matchEdges = (uchar *)cvAlloc( (first_runs + 1) * (second_runs + 1) * sizeof( uchar ));

    if( matchEdges == 0 )
    {
        cvFree( &costTable );
        return CV_OUTOFMEM_ERR;
    }

    row_size = first_runs + 1;

    /* ============= Fill costTable ============= */

    costTable[0] = 0.0f;

    /* Fill upper line in the cost Table */

    prev = first[0];
    curr = 2;

    for( n = 0; n < first_runs; n++ )
    {

        l_end = first[curr];
        curr += 2;
        costTable[n + 1] = costTable[n] + Occlusion * (l_end - prev);
        prev = l_end;

    }                           /* for */

    /* Fill lefter line in the cost Table */

    prev = second[0];
    curr = 2;
    baseIndex = 0;

    for( n = 0; n < second_runs; n++ )
    {

        l_end = second[curr];
        curr += 2;
        costTable[baseIndex + row_size] = costTable[baseIndex] + Occlusion * (l_end - prev);
        baseIndex += row_size;
        prev = l_end;

    }                           /* for */

    /* Count costs in the all rest cells */

    first_curr = 0;
    second_curr = 0;

    for( i = 1; i <= first_runs; i++ )
    {
        for( j = 1; j <= second_runs; j++ )
        {

            first_curr = (i - 1) * 2;
            second_curr = (j - 1) * 2;

            l_beg = first[first_curr];
            first_curr++;
            l_color = first[first_curr];
            first_curr++;
            l_end = first[first_curr];
            l_len = l_end - l_beg + 1;

            r_beg = second[second_curr];
            second_curr++;
            r_color = second[second_curr];
            second_curr++;
            r_end = second[second_curr];
            r_len = r_end - r_beg + 1;

            i_1 = i - 1;
            j_1 = j - 1;

            if( r_len == l_len )
            {
                cost = 0;
            }
            else
            {

                if( r_len > l_len )
                {
                    cost = (float) (r_len * r_len - l_len * l_len) * (1 / (r_len * l_len));
                }
                else
                {
                    cost = (float) (l_len * l_len - r_len * r_len) * (1 / (r_len * l_len));
                }
            }                   /* if */

            len_color = r_color - l_color;

            cost1 = (float) ((len_color * len_color) >> 2);

            min2 = costTable[i_1 + j * row_size] + Occlusion * l_len;

            min3 = costTable[i + j_1 * row_size] + Occlusion * r_len;

            min1 = costTable[i_1 + j_1 * row_size] + cost + (float) cost1;

            if( min1 < min2 )
            {

                if( min1 < min3 )
                {
                    cmin = min1;
                    cpath = 1;
                }
                else
                {
                    cmin = min3;
                    cpath = 3;
                }               /* if */

            }
            else
            {

                if( min2 < min3 )
                {
                    cmin = min2;
                    cpath = 2;
                }
                else
                {
                    cmin = min3;
                    cpath = 3;
                }               /* if */

            }                   /* if */

            costTable[i + j * row_size] = cmin;
            matchEdges[i + j * row_size] = cpath;
        }                       /* for */
    }                           /* for */

    /* =========== Reconstruct the Path =========== */

    i = first_runs;
    j = second_runs;

    first_curr = i * 2 - 2;
    second_curr = j * 2 - 2;


    while( i > 0 && j > 0 )
    {

        /* Connect begins */
        switch (matchEdges[i + j * row_size])
        {

        case 1:                /* to diagonal */

            first_corr[first_curr] = second[second_curr];
            first_corr[first_curr + 1] = second[second_curr + 2];
            second_corr[second_curr] = first[first_curr];
            second_corr[second_curr + 1] = first[first_curr + 2];

            first_curr -= 2;
            second_curr -= 2;
            i--;
            j--;

            break;

        case 2:                /* to left */

            first_corr[first_curr] = second[second_curr + 2];
            first_corr[first_curr + 1] = second[second_curr + 2];

            first_curr -= 2;
            i--;

            break;

        case 3:                /* to up */

            second_corr[second_curr] = first[first_curr + 2];
            second_corr[second_curr + 1] = first[first_curr + 2];

            second_curr -= 2;
            j--;

            break;
        }                       /* switch */
    }                           /* while */

    /* construct rest of horisontal path if its need */
    while( i > 0 )
    {

        first_corr[first_curr] = second[second_curr + 2];       /* connect to begin */
        first_corr[first_curr + 1] = second[second_curr + 2];   /* connect to begin */

        first_curr -= 2;
        i--;

    }                           /* while */

    /* construct rest of vertical path if its need */
    while( j > 0 )
    {

        second_corr[second_curr] = first[first_curr + 2];
        second_corr[second_curr + 1] = first[first_curr + 2];

        second_curr -= 2;
        j--;

    }                           /* while */

    cvFree( &costTable );
    cvFree( &matchEdges );

    return CV_NO_ERR;
}                               /* icvDynamicCorrespond */


/*======================================================================================*/

static CvStatus
icvDynamicCorrespondMulti( int lines,   /* number of scanlines */
                           int *first,  /* s0|w0|s1|w1|...s(n-1)|w(n-1)|sn */
                           int *first_runs,     /* numbers of runs */
                           int *second, int *second_runs, int *first_corr,      /* s0'|e0'|s1'|e1'|... */
                           int *second_corr )
{
    CvStatus error;

    int currFirst;
    int currSecond;
    int currFirstCorr;
    int currSecondCorr;
    int n;

    /* Test errors */

    if( (lines < 1) ||
        (first == 0) ||
        (first_runs == 0) ||
        (second == 0) || (second_runs == 0) || (first_corr == 0) || (second_corr == 0) )
        return CV_BADFACTOR_ERR;

    currFirst = 0;
    currSecond = 0;
    currFirstCorr = 0;
    currSecondCorr = 0;

    for( n = 0; n < lines; n++ )
    {

        error = icvDynamicCorrespond( &(first[currFirst]),
                                      first_runs[n],
                                      &(second[currSecond]),
                                      second_runs[n],
                                      &(first_corr[currFirstCorr]),
                                      &(second_corr[currSecondCorr]) );

        if( error != CV_NO_ERR )
            return error;

        currFirst += first_runs[n] * 2 + 1;
        currSecond += second_runs[n] * 2 + 1;
        currFirstCorr += first_runs[n] * 2;
        currSecondCorr += second_runs[n] * 2;

    }

    return CV_NO_ERR;

}                               /* icvDynamicCorrespondMulti */


/*======================================================================================*/

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvDynamicCorrespondMulti
//    Purpose: The functions 
//    Context:
//    Parameters:  
//
//    Notes:  
//F*/
CV_IMPL void
cvDynamicCorrespondMulti( int lines,    /* number of scanlines */
                          int *first,   /* s0|w0|s1|w1|...s(n-1)|w(n-1)|sn */
                          int *first_runs,      /* numbers of runs */
                          int *second, int *second_runs, int *first_corr,       /* s0'|e0'|s1'|e1'|... */
                          int *second_corr )
{
    IPPI_CALL( icvDynamicCorrespondMulti( lines,        /* number of scanlines */
                                          first,        /* s0|w0|s1|w1|...s(n-1)|w(n-1)|sn */
                                          first_runs,   /* numbers of runs */
                                          second, second_runs, first_corr,      /* s0'|e0'|s1'|e1'|... */
                                          second_corr ));
}
