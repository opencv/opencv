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

/*  This is regression test for Snakes functions of OpenCV.
//  This test will generate fixed figure, read initial position
//  of snake from file, run OpenCV function and compare result
//  position of snake with position(from file) which must be resulting.
//
//  Test is considered to be succesfull if resultant positions
//  are identical.
*/

class CV_SnakeTest : public CvTest
{
public:
    CV_SnakeTest();
protected:
    void run(int);
};

#define SCAN  0

CV_SnakeTest::CV_SnakeTest():
    CvTest( "snakes", "cvSnakeImage" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}

void CV_SnakeTest::run( int /*start_from*/ )
{
    int code = CvTS::OK;
    static const char* file_name[] =
    {
        "ring",
        "square"
    };

    const int numfig_image = 1;
    const int numfig_grad  = 1;

    FILE* file = 0;
#ifndef _MAX_PATH
#define _MAX_PATH 1024
#endif
    char abs_file_name[_MAX_PATH];
    char rel_path[_MAX_PATH];

    int i,j;

    /* source image */
    IplImage* iplSrc = NULL;
    CvSize win;
    int length;

    float alpha,beta,gamma;
    CvTermCriteria criteria;
    long lErrors = 0;
    int progress = 0, test_case_count = numfig_image + numfig_grad;
    CvPoint* Pts = 0;
    CvPoint* resPts = 0;

    sprintf( rel_path, "%ssnakes/", ts->get_data_path() );

    criteria.type = CV_TERMCRIT_ITER;
    win.height = win.width = 3;

    for( i = 0; i < test_case_count; i++ )
    {
        progress = update_progress( progress, i, test_case_count, 0 );
        int num_pos;
        int k;

        char tmp[_MAX_PATH];

        ts->update_context( this, i, false );

        /* create full name of bitmap file */
        strcpy(tmp, rel_path);
        strcat(tmp, file_name[i]);
        strcpy( abs_file_name, tmp );
        strcat( abs_file_name, ".bmp" );

        /* read bitmap with 8u image */
        iplSrc = cvLoadImage( abs_file_name, -1 );

        if (!iplSrc)
        {
            ts->printf( CvTS::LOG, "can not load %s\n", abs_file_name );
            code = CvTS::FAIL_MISSING_TEST_DATA;
            goto _exit_;
        }

        /* init snake reading file with snake */
        strcpy(tmp, rel_path);
        strcat(tmp, file_name[i]);
        strcpy( abs_file_name, tmp );
        strcat( abs_file_name, ".txt" );

#if !SCAN
        file = fopen( abs_file_name, "r" );
#else
        file = fopen( abs_file_name, "r+" );
#endif

        if (!file)
        {
            ts->printf( CvTS::LOG, "can not load %s\n", abs_file_name );
            code = CvTS::FAIL_MISSING_TEST_DATA;
            goto _exit_;
        }

        /* read snake parameters */
        fscanf(file, "%d", &length );
        fscanf(file, "%f", &alpha );
        fscanf(file, "%f", &beta );
        fscanf(file, "%f", &gamma );

        /* allocate memory for snakes */
        Pts = (CvPoint*)cvAlloc( length * sizeof(Pts[0]) );
        resPts = (CvPoint*)cvAlloc( length * sizeof(resPts[0]) );

        /* get number of snake positions */
        fscanf(file, "%d", &num_pos );

        /* get number iterations between two positions */
        fscanf(file, "%d", &criteria.max_iter );

        /* read initial snake position */
        for ( j = 0; j < length; j++ )
        {
            fscanf(file, "%d%d", &Pts[j].x, &Pts[j].y );
        }

        for ( k = 0; k < num_pos; k++ )
        {
            /* Run CVL function to check it */
            if(i<numfig_image)
            {
                 cvSnakeImage( iplSrc, Pts, length,
                           &alpha, &beta, &gamma, CV_VALUE, win, criteria, 0 );
            }
            else
            {
                cvSnakeImage( iplSrc, Pts, length,
                           &alpha, &beta, &gamma, CV_VALUE, win, criteria, 1 /*usegrad*/ );
            }

#if !SCAN
            for ( j = 0; j < length; j++ )
            {
                fscanf(file, "%d%d", &resPts[j].x, &resPts[j].y );

                lErrors += (Pts[j].x != resPts[j].x);
                lErrors += (Pts[j].y != resPts[j].y);
            }
#else
            fseek( file, 0, SEEK_CUR );
            fprintf(file, "\n");
            for ( j = 0; j < length; j++ )
            {
                fprintf(file, "\n%d %d", Pts[j].x, Pts[j].y );
            }
#endif
        }
        fclose(file);
        file = 0;
        cvFree(&Pts);
        cvFree(&resPts);
        cvReleaseImage(&iplSrc);
    }

    if( lErrors > 0 )
    {
        ts->printf( CvTS::LOG, "Total fixed %d errors", lErrors );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

    if( file )
        fclose(file);
    cvFree(&Pts);
    cvFree(&resPts);
    cvReleaseImage(&iplSrc);

    if( code < 0 )
        ts->set_failed_test_info( code );
}

CV_SnakeTest snake_test;

/* End of file. */
