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

#include "cvtest.h"

#if 0

#include <stdlib.h>
#include <math.h>
#include <float.h>

#define HT_STANDARD (size_t)1
#define HT_PP (size_t)2
#define HT_MD (size_t)3

static char* func_names[] = {"cvHoughTransform", "cvHoughTransformP", "cvHoughTransformSDiv"};
static char* test_desc[] = { "Run the hough transform function"};

int test_dt(void* arg);
int read_image_dims(void);
int read_gen_type(void);

int test_ht(void* arg)
{
    int nlines = 10;
    int* lines = new int[4*nlines];
    float* flines = new float[2*nlines];
    float rho = 10.0f, theta = 0.1f;
    int srn = 10, stn = 10;
    int threshold = 10;
    int lineLength = 10, lineGap = 2;
    int w = 100; /* width and height of the rect */
    int h = 100;
    int type = (int)(size_t)arg;
    IplImage* image; /* Source and destination images */
    image = cvCreateImage( cvSize(w, h), 8, 1 );
    cvZero(image);
    
    if( image == NULL )
    {
        delete lines;
        delete flines;
        return trsResult(TRS_FAIL, "Not enough memory to perform the test");
    }

    switch(type)
    {
    case HT_STANDARD:
        /* Run the distance transformation function */
        cvHoughLines(image, rho, theta, threshold, flines, nlines);
        break;

    case HT_PP:
        cvHoughLinesP(image, rho, theta, threshold, lineLength, lineGap, lines, nlines);
        break;

    case HT_MD:
        cvHoughLinesSDiv(image, rho, srn, theta, stn, threshold, flines, nlines);
        break;

    default:
        cvReleaseImage(& image);
        delete lines;
        delete flines;
        trsResult(TRS_FAIL, "No such function");
    }
    cvReleaseImage( &image );
    delete lines;
    delete flines;
    if(cvGetErrStatus() < 0)
    {
        return trsResult(TRS_FAIL, "Function returned 'bad argument'");
    }
    else
    {
        return trsResult(TRS_OK, "No errors");
    }
}

void InitAHoughLines(void)
{
    /* Registering test functions */
    trsRegArg(func_names[0], test_desc[0], atsAlgoClass, test_ht, HT_STANDARD);
    trsRegArg(func_names[1], test_desc[0], atsAlgoClass, test_ht, HT_PP);
    trsRegArg(func_names[2], test_desc[0], atsAlgoClass, test_ht, HT_MD);

} /* InitADistanceTransform*/

#endif

