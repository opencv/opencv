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

class CV_PyrSegmentationTest : public CvTest
{
public:
    CV_PyrSegmentationTest();
protected:
    void run(int);
};

#define SCAN  0

CV_PyrSegmentationTest::CV_PyrSegmentationTest():
    CvTest( "segmentation-pyramid", "cvPyrSegmentation" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}

void CV_PyrSegmentationTest::run( int /*start_from*/ )
{
    const int level = 5;
    const double range = 15;

    int code = CvTS::OK;

    CvPoint _cp[] ={{33,33}, {43,33}, {43,43}, {33,43}};
    CvPoint _cp2[] ={{50,50}, {70,50}, {70,70}, {50,70}};
    CvPoint* cp = _cp;
    CvPoint* cp2 = _cp2;
    CvConnectedComp *dst_comp[3];
    CvRect rect[3] = {{50,50,21,21}, {0,0,128,128}, {33,33,11,11}};
    double a[3] = {441.0, 15822.0, 121.0};

/*    ippiPoint cp3[] ={130,130, 150,130, 150,150, 130,150};  */
/*	CvPoint cp[] ={0,0, 5,5, 5,0, 10,5, 10,0, 15,5, 15,0};  */
    int nPoints = 4;
    int block_size = 1000;

    CvMemStorage *storage;   /*   storage for connected component writing  */
    CvSeq *comp;

    CvRNG* rng = ts->get_rng();
    int i, j, iter;

    IplImage *image, *image_f, *image_s;
    CvSize size = {128, 128};
    const int threshold1 = 50, threshold2 = 50;

    rect[1].width = size.width;
    rect[1].height = size.height;
    a[1] = size.width*size.height - a[0] - a[2];

    OPENCV_CALL( storage = cvCreateMemStorage( block_size ) );

    for( iter = 0; iter < 2; iter++ )
    {
        int channels = iter == 0 ? 1 : 3;
        int mask[] = {0,0,0};

        image = cvCreateImage(size, 8, channels );
        image_s = cvCloneImage( image );
        image_f = cvCloneImage( image );

        if( channels == 1 )
        {
            int color1 = 30, color2 = 110, color3 = 190;

            cvSet( image, cvScalarAll(color1));
            cvFillPoly( image, &cp, &nPoints, 1, cvScalar(color2));
            cvFillPoly( image, &cp2, &nPoints, 1, cvScalar(color3));
        }
        else
        {
            CvScalar color1 = CV_RGB(30,30,30), color2 = CV_RGB(255,0,0), color3 = CV_RGB(0,255,0);

            assert( channels == 3 );
            cvSet( image, color1 );
            cvFillPoly( image, &cp, &nPoints, 1, color2);
            cvFillPoly( image, &cp2, &nPoints, 1, color3);
        }

        cvRandArr( rng, image_f, CV_RAND_UNI, cvScalarAll(0), cvScalarAll(range*2) );
        cvAddWeighted( image, 1, image_f, 1, -range, image_f );

        cvPyrSegmentation( image_f, image_s,
                           storage, &comp,
                           level, threshold1, threshold2 );

        if(comp->total != 3)
        {
            ts->printf( CvTS::LOG,
                "The segmentation function returned %d (not 3) components\n", comp->total );
            code = CvTS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }
        /*  read the connected components     */
        dst_comp[0] = (CvConnectedComp*)CV_GET_SEQ_ELEM( CvConnectedComp, comp, 0 );
        dst_comp[1] = (CvConnectedComp*)CV_GET_SEQ_ELEM( CvConnectedComp, comp, 1 );
        dst_comp[2] = (CvConnectedComp*)CV_GET_SEQ_ELEM( CvConnectedComp, comp, 2 );

        /*{
            for( i = 0; i < 3; i++ )
            {
                CvRect r = dst_comp[i]->rect;
                cvRectangle( image_s, cvPoint(r.x,r.y), cvPoint(r.x+r.width,r.y+r.height),
                    CV_RGB(255,255,255), 3, 8, 0 );
            }

            cvNamedWindow( "test", 1 );
            cvShowImage( "test", image_s );
            cvWaitKey(0);
        }*/

        code = cvTsCmpEps2( ts, image, image_s, 10, false, "the output image" );
        if( code < 0 )
            goto _exit_;

        for( i = 0; i < 3; i++)
        {
            for( j = 0; j < 3; j++ )
            {
                if( !mask[j] && dst_comp[i]->area == a[j] &&
                    dst_comp[i]->rect.x == rect[j].x &&
                    dst_comp[i]->rect.y == rect[j].y &&
                    dst_comp[i]->rect.width == rect[j].width &&
                    dst_comp[i]->rect.height == rect[j].height )
                {
                    mask[j] = 1;
                    break;
                }
            }
            if( j == 3 )
            {
                ts->printf( CvTS::LOG, "The component #%d is incorrect\n", i );
                code = CvTS::FAIL_BAD_ACCURACY;
                goto _exit_;
            }
        }

        cvReleaseImage(&image_f);
        cvReleaseImage(&image);
        cvReleaseImage(&image_s);
    }

_exit_:

    cvReleaseMemStorage( &storage );
    cvReleaseImage(&image_f);
    cvReleaseImage(&image);
    cvReleaseImage(&image_s);

    if( code < 0 )
        ts->set_failed_test_info( code );
}

CV_PyrSegmentationTest pyr_segmentation_test;

/* End of file. */
