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

/*////////////////////// emd_test /////////////////////////*/

typedef struct matr_info
{
    float*  x_origin;
    float*  y_origin;
    float*  matr;
    int     rows;
    int     cols;
}
matr_info;

static float matr_dist( const float* x, const float* y, void* param )
{
    matr_info* mi = (matr_info*)param;
    int i = (int)(x - mi->x_origin) - 1;
    int j = (int)(y - mi->y_origin) - 1;
    assert( 0 <= i && i < mi->rows &&
            0 <= j && j < mi->cols );
    return  mi->matr[i*mi->cols + j];
}

class CV_EMDTest : public CvTest
{
public:
    CV_EMDTest();
protected:
    void run(int);
};


CV_EMDTest::CV_EMDTest():
    CvTest( "emd", "cvCalcEMD" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}

void CV_EMDTest::run( int )
{
    int code = CvTS::OK;
    const double success_error_level = 1e-6;
    #define M 10000
    double emd0 = 2460./210;
    static float cost[] = 
    {
        16, 16, 13, 22, 17,
        14, 14, 13, 19, 15,
        19, 19, 20, 23,  M,
        M ,  0,  M,  0,  0
    };
    static float  w1[] = { 50, 60, 50, 50 },
                  w2[] = { 30, 20, 70, 30, 60 };
    matr_info mi;
    float emd;

    mi.x_origin = w1;
    mi.y_origin = w2;

    mi.rows = sizeof(w1)/sizeof(w1[0]);
    mi.cols = sizeof(w2)/sizeof(w2[0]);
    mi.matr = cost;

    emd = cvCalcEMD( w1, mi.rows, w2, mi.cols, 0, (CvDisType)-1,
                     matr_dist, 0, &mi );

    if( fabs( emd - emd0 ) > success_error_level*emd0 )
    {
        ts->printf( CvTS::LOG,
            "The computed distance is %.2f, while it should be %.2f\n", emd, emd0 );
        code = CvTS::FAIL_BAD_ACCURACY;
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
}

CV_EMDTest emd_test;

/* End of file. */
