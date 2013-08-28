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
#include <assert.h>


static CvStatus
icvMorphEpilines8uC3( uchar * first_pix,        /* raster epiline from image 1      */
                      uchar * second_pix,       /* raster epiline from image 2      */
                      uchar * dst_pix,  /* raster epiline from dest image   */
                      /* (it's an output parameter)       */
                      float alpha,      /* relative position of camera      */
                      int *first,       /* first sequence of runs           */
                      int first_runs,   /* it's length                      */
                      int *second,      /* second sequence of runs          */
                      int second_runs, int *first_corr, /* corr data for the 1st seq        */
                      int *second_corr, /* corr data for the 2nd seq        */
                      int dst_len )
{

    float alpha1;               /* alpha - 1.0 */
    int s, s1;                  /* integer variant of alpha and alpha1 ( 0 <= s,s1 <= 256 ) */
    int curr;                   /* current index in run's array */

    float begLine;              /* begin of current run */
    float endLine;              /* end   of current run */

    float begCorr;              /* begin of correspondence destination of run */
    float endCorr;              /* end   of correspondence destination of run */

    int begDestLine;            /* begin of current destanation of run */
    int endDestLine;            /* end   of current destanation of run */
    int begLineIndex;
    int endLineIndex;
    int indexImg1;
    float step = 0;
    int n;

    memset( dst_pix, 0, dst_len );
    alpha1 = (float) (1.0 - alpha);

    s = (int) (alpha * 256);
    s1 = 256 - s;

    /* --------------Create first line------------- */

    begLineIndex = first[0];
    begLine = (float) begLineIndex;

    curr = 0;

    for( n = 0; n < first_runs; n++ )
    {                           /* for each run */

        begCorr = (float) first_corr[curr];
        curr++;
        endCorr = (float) first_corr[curr];
        curr++;
        endLineIndex = first[curr];
        endLine = (float) endLineIndex;

        begDestLine = (int) (alpha * begLine + alpha1 * begCorr);
        endDestLine = (int) (alpha * endLine + alpha1 * endCorr);

        indexImg1 = begDestLine * 3;

        step = 0;
        if( endDestLine != begDestLine )
            step = (endLine - begLine) / ((float) (endDestLine - begDestLine));

        if( begCorr != endCorr )
        {

            for( ; begDestLine < endDestLine; begDestLine++ )
            {
                /* for each pixel */

                begLineIndex = (int) begLine;
                begLineIndex *= 3;

                /* Blend R */
                dst_pix[indexImg1] = (uchar) (((int) (first_pix[begLineIndex]) * s) >> 8);

                indexImg1++;

                /* Blend G */
                dst_pix[indexImg1] = (uchar) (((int) (first_pix[begLineIndex + 1]) * s) >> 8);

                indexImg1++;

                /* Blend B */
                dst_pix[indexImg1] = (uchar) (((int) (first_pix[begLineIndex + 2]) * s) >> 8);

                indexImg1++;

                begLine += step;

            }                   /* for */
        }
        else
        {

            for( ; begDestLine < endDestLine; begDestLine++ )
            {
                /* for each pixel */

                begLineIndex = (int) begLine;
                begLineIndex *= 3;

                /* Blend R */
                dst_pix[indexImg1] = first_pix[begLineIndex];
                indexImg1++;

                /* Blend G */
                dst_pix[indexImg1] = first_pix[begLineIndex + 1];
                indexImg1++;

                /* Blend B */
                dst_pix[indexImg1] = first_pix[begLineIndex + 2];

                indexImg1++;

                begLine += step;

            }                   /* for */
        }                       /* if */

        begLineIndex = endLineIndex;
        begLine = endLine;


    }                           /* for each runs in first line */

    begLineIndex = second[0];
    begLine = (float) begLineIndex;

    curr = 0;

    /* --------------Create second line------------- */
    curr = 0;;
    for( n = 0; n < second_runs; n++ )
    {                           /* for each run */

        begCorr = (float) second_corr[curr];
        curr++;
        endCorr = (float) second_corr[curr];
        curr++;
        endLineIndex = second[curr];
        endLine = (float) endLineIndex;

        begDestLine = (int) (alpha1 * begLine + alpha * begCorr);
        endDestLine = (int) (alpha1 * endLine + alpha * endCorr);

        indexImg1 = begDestLine * 3;

        step = 0;
        if (endDestLine != begDestLine)
            step = (endLine - begLine) / ((float) (endDestLine - begDestLine));

        if( begCorr != endCorr )
        {

            for( ; begDestLine < endDestLine; begDestLine++ )
            {
                /* for each pixel */

                begLineIndex = (int) begLine;
                begLineIndex *= 3;

                /* Blend R */
                dst_pix[indexImg1] =
                    (uchar) (dst_pix[indexImg1] +
                             (uchar) (((unsigned int) (second_pix[begLineIndex]) * s1) >> 8));

                indexImg1++;

                /* Blend G */
                dst_pix[indexImg1] =
                    (uchar) (dst_pix[indexImg1] +
                             (uchar) (((unsigned int) (second_pix[begLineIndex + 1]) * s1) >>
                                      8));

                indexImg1++;

                /* Blend B */
                dst_pix[indexImg1] =
                    (uchar) (dst_pix[indexImg1] +
                             (uchar) (((unsigned int) (second_pix[begLineIndex + 2]) * s1) >>
                                      8));

                indexImg1++;

                begLine += step;

            }                   /* for */
        }
        else
        {

            for( ; begDestLine < endDestLine; begDestLine++ )
            {
                /* for each pixel */

                begLineIndex = (int) begLine;
                begLineIndex *= 3;

                /* Blend R */
                dst_pix[indexImg1] = (uchar) (dst_pix[indexImg1] + second_pix[begLineIndex]);
                indexImg1++;

                /* Blend G */
                dst_pix[indexImg1] =
                    (uchar) (dst_pix[indexImg1] + second_pix[begLineIndex + 1]);
                indexImg1++;

                /* Blend B */
                dst_pix[indexImg1] =
                    (uchar) (dst_pix[indexImg1] + second_pix[begLineIndex + 2]);
                /*assert(indexImg1 < dst_len); */

                indexImg1++;

                begLine += step;

            }                   /* for */
        }                       /* if */

        begLineIndex = endLineIndex;
        begLine = endLine;

    }                           /* for each runs in second line */

    return CV_NO_ERR;

}                               /* icvMorphEpilines8uC3 */


/*======================================================================================*/

static CvStatus
icvMorphEpilines8uC3Multi( int lines,   /* number of lines                              */
                           uchar * first_pix,   /* raster epilines from the first image         */
                           int *first_num,      /* numbers of pixel in first line               */
                           uchar * second_pix,  /* raster epilines from the second image        */
                           int *second_num,     /* numbers of pixel in second line              */
                           uchar * dst_pix,     /* raster epiline from the destination image    */
                           /* (it's an output parameter)                   */
                           int *dst_num,        /* numbers of pixel in output line              */
                           float alpha, /* relative position of camera                  */
                           int *first,  /* first sequence of runs                       */
                           int *first_runs,     /* it's length                                  */
                           int *second, /* second sequence of runs                      */
                           int *second_runs, int *first_corr,   /* correspond information for the 1st seq       */
                           int *second_corr )   /* correspond information for the 2nd seq       */
{
    CvStatus error;
    int currLine;
    int currFirstPix = 0;
    //int currFirstNum = 0;
    int currSecondPix = 0;
    //int currSecondNum = 0;
    int currDstPix = 0;
    int currFirst = 0;
    //int currFirstRuns = 0;
    int currSecond = 0;
    //int currSecondRuns = 0;
    int currFirstCorr = 0;
    int currSecondCorr = 0;

    if( lines < 1 ||
        first_pix == 0 ||
        first_num == 0 ||
        second_pix == 0 ||
        second_num == 0 ||
        dst_pix == 0 ||
        dst_num == 0 ||
        alpha < 0 ||
        alpha > 1 ||
        first == 0 ||
        first_runs == 0 ||
        second == 0 || second_runs == 0 || first_corr == 0 || second_corr == 0 )
        return CV_BADFACTOR_ERR;

    for( currLine = 0; currLine < lines; currLine++ )
    {

        error = icvMorphEpilines8uC3( &(first_pix[currFirstPix]),
                                      &(second_pix[currSecondPix]),
                                      &(dst_pix[currDstPix]),
                                      alpha,
                                      &(first[currFirst]),
                                      first_runs[currLine],
                                      &(second[currSecond]),
                                      second_runs[currLine],
                                      &(first_corr[currFirstCorr]),
                                      &(second_corr[currSecondCorr]), dst_num[currLine] * 3 );


        if( error != CV_NO_ERR )
            return CV_NO_ERR;

        currFirstPix += first_num[currLine] * 3;
        currSecondPix += second_num[currLine] * 3;
        currDstPix += dst_num[currLine] * 3;
        currFirst += (first_runs[currLine] * 2) + 1;
        currSecond += (second_runs[currLine] * 2) + 1;
        currFirstCorr += first_runs[currLine] * 2;
        currSecondCorr += second_runs[currLine] * 2;

    }                           /* for */

    return CV_NO_ERR;

}                               /* icvMorphEpilines8uC3Multi */




/*======================================================================================*/

CV_IMPL void
cvMorphEpilinesMulti( int lines,        /* number of lines             */
                      uchar * first_pix,        /* raster epilines from the first image      */
                      int *first_num,   /* numbers of pixel in first line            */
                      uchar * second_pix,       /* raster epilines from the second image     */
                      int *second_num,  /* numbers of pixel in second line           */
                      uchar * dst_pix,  /* raster epiline from the destination image */
                      /* (it's an output parameter)                */
                      int *dst_num,     /* numbers of pixel in output line           */
                      float alpha,      /* relative position of camera               */
                      int *first,       /* first sequence of runs                    */
                      int *first_runs,  /* it's length                               */
                      int *second,      /* second sequence of runs                   */
                      int *second_runs, int *first_corr,        /* correspond information for the 1st seq    */
                      int *second_corr  /* correspond information for the 2nd seq     */
     )
{
    IPPI_CALL( icvMorphEpilines8uC3Multi( lines,        /* number of lines                           */
                                          first_pix,    /* raster epilines from the first image      */
                                          first_num,    /* numbers of pixel in first line            */
                                          second_pix,   /* raster epilines from the second image     */
                                          second_num,   /* numbers of pixel in second line           */
                                          dst_pix,      /* raster epiline from the destination image */
                                          /* (it's an output parameter)                   */
                                          dst_num,      /* numbers of pixel in output line           */
                                          alpha,        /* relative position of camera               */
                                          first,        /* first sequence of runs                    */
                                          first_runs,   /* it's length                               */
                                          second,       /* second sequence of runs                   */
                                          second_runs, first_corr,      /* correspond information for the 1st seq    */
                                          second_corr   /* correspond information for the 2nd seq     */
                ));
}
