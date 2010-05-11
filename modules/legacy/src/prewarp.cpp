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

/* Valery Mosyagin */

static CvStatus
icvFindRuns( int numLines,      /* number of scanlines      */
             uchar * prewarp_1, /* prewarp image 1          */
             uchar * prewarp_2, /* prewarp image 2          */
             int *line_lens_1,  /* line lengths 1           */
             int *line_lens_2,  /* line lengths 2           */
             int *runs_1,       /* result runs  1           */
             int *runs_2,       /* result runs  2           */
             int *num_runs_1,   /* numbers of first runs    */
             int *num_runs_2 )
{
    CvStatus err;

    err = icvFindRunsInOneImage( numLines, prewarp_1, line_lens_1, runs_1, num_runs_1 );

    if( err != CV_NO_ERR )
        return err;

    err = icvFindRunsInOneImage( numLines, prewarp_2, line_lens_2, runs_2, num_runs_2 );

    return err;

}


/*======================================================================================*/

CV_INLINE int
icvGetColor( uchar * valueRGB )
{
    int R = *valueRGB;
    int G = *(valueRGB + 1);
    int B = *(valueRGB + 2);

    return ( ((R + G + B) >> 3) & 0xFFFC );
}                               /* vm_GetColor */


/*======================================================================================*/

CvStatus
icvFindRunsInOneImage( int numLines,    /* number of scanlines      */
                       uchar * prewarp, /* prewarp image            */
                       int *line_lens,  /* line lengths in pixels   */
                       int *runs,       /* result runs              */
                       int *num_runs )
{
    int epiLine;
    int run_index;
    int curr_color;
    int index;
    int color;
    uchar *curr_point;
    int num;


    run_index = 0;

    curr_point = prewarp;

    for( epiLine = 0; epiLine < numLines; epiLine++ )
    {

        curr_color = icvGetColor( curr_point );

        runs[run_index++] = 0;
        runs[run_index++] = curr_color;

        curr_point += 3;

        num = 1;
        for( index = 1; index < line_lens[epiLine]; index++ )
        {

            color = icvGetColor( curr_point );

            if( color != curr_color )
            {
                runs[run_index++] = index;
                runs[run_index++] = color;
                curr_color = color;
                num++;
            }

            curr_point += 3;
        }

        runs[run_index++] = index;
        num_runs[epiLine] = num;
    }

    return CV_NO_ERR;
}


/*======================================================================================*/

CV_IMPL void
cvFindRuns( int numLines,       /* number of scanlines   */
            uchar * prewarp_1,  /* prewarp image 1       */
            uchar * prewarp_2,  /* prewarp image 2       */
            int *line_lens_1,   /* line lengths 1        */
            int *line_lens_2,   /* line lengths 2        */
            int *runs_1,        /* result runs  1        */
            int *runs_2,        /* result runs  2        */
            int *num_runs_1,    /* numbers of first runs */
            int *num_runs_2 )
{
    IPPI_CALL( icvFindRuns( numLines,   /* number of scanlines   */
                            prewarp_1,  /* prewarp image 1       */
                            prewarp_2,  /* prewarp image 2       */
                            line_lens_1,        /* line lengths 1        */
                            line_lens_2,        /* line lengths 2        */
                            runs_1,     /* result runs  1        */
                            runs_2,     /* result runs  2        */
                            num_runs_1, /* numbers of first runs */
                            num_runs_2 ));
}
