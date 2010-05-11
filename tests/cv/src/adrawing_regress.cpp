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

#include <string.h>
#include "highgui.h"

static char* funcs = "cvLine cvLineAA cvRectangle cvCircle cvCircleAA "
                     "cvEllipse cvEllipseAA cvFillConvexPoly cvFillPoly"
                     "cvPolyLine cvPolyLineAA cvPutText";
static char* test_desc = "Drawing to image (regression test)";

static char* filedir_ = "drawing";

#ifndef RGB
#define RGB(r,g,b) CV_RGB(b,g,r)
#endif

static int my_tolower( int c )
{
	return 'A' <= c && c <= 'Z' ? c + ('a' - 'A') : c;
}

static int CheckImage(IplImage* image, char* file, char* /*funcname*/)
{
    //printf("loading %s\n", file );
    IplImage* read = cvLoadImage( file, 1 );

    if( !read )
    {
        trsWrite( ATS_CON | ATS_LST, "can't read image\n" );
        return 1;
    }

    int err = 0;

#if 0
    {
        IplImage* temp = cvCloneImage( read );
        cvAbsDiff( image, read, temp );
        cvThreshold( temp, temp, 0, 255, CV_THRESH_BINARY );
        cvvNamedWindow( "Original", 0 );
        cvvNamedWindow( "Diff", 0 );
        cvvShowImage( "Original", read );
        cvvShowImage( "Diff", temp );
        cvvWaitKey(0);
        cvvDestroyWindow( "Original" );
        cvvDestroyWindow( "Diff" );
    }
#endif

    cvAbsDiff( image, read, read );
    cvThreshold( read, read, 0, 1, CV_THRESH_BINARY );
    err = cvRound( cvNorm( read, 0, CV_L1 ))/3;

    cvReleaseImage( &read );
    return err;
}

static int ProcessImage( IplImage* image, char* funcname, int read )
{
    char name[1000];
    char lowername[1000];
    int i = 0, err;
    
    do
    {
	lowername[i] = (char)my_tolower(funcname[i]);
    }
    while( funcname[i++] != '\0' );

    if( read )
    {
        err = CheckImage( image,
                          atsGetTestDataPath(name, filedir_, lowername, "bmp"),
                          funcname );
        if( err )
	{
            trsWrite( ATS_CON | ATS_LST, "Error in %s: %d\n", funcname, err );
	}    
        return 0; //err;
    }
    else
    {
        cvvSaveImage( atsGetTestDataPath(name, filedir_, lowername, "bmp"), image );
        return 0;
    }
}

static int drawing_test()
{
    static int read_params = 0;
    static int read = 0;
    const int channel = 3;
    CvSize size = cvSize(600, 300);

    int i, j;
    int Errors = 0;

    if( !read_params )
    {
        read_params = 1;

        trsCaseRead( &read, "/n/y", "y", "Read from file ?" );
    }
    // Create image
    IplImage* image = cvCreateImage( size, IPL_DEPTH_8U, channel );

    // cvLine
    cvZero( image );

    for( i = 0; i < 100; i++ )
    {
        CvPoint p1 = cvPoint( i - 30, i * 4 + 10 );
        CvPoint p2 = cvPoint( size.width + 30 - i, size.height - 10 - i * 4 );

        cvLine( image, p1, p2, CV_RGB(178+i, 255-i, i), i % 10 );
    }
    Errors += ProcessImage( image, "cvLine", read );

    // cvLineAA
    cvZero( image );

    for( i = 0; i < 100; i++ )
    {
        CvPoint p1 = cvPoint( i - 30, i * 4 + 10 );
        CvPoint p2 = cvPoint( size.width + 30 - i, size.height - 10 - i * 4 );

        cvLine( image, p1, p2, CV_RGB(178+i, 255-i, i), 1, CV_AA, 0 );
    }
    //Errors += ProcessImage( image, "cvLineAA", read );

    // cvRectangle
    cvZero( image );

    for( i = 0; i < 100; i++ )
    {
        CvPoint p1 = cvPoint( i - 30, i * 4 + 10 );
        CvPoint p2 = cvPoint( size.width + 30 - i, size.height - 10 - i * 4 );

        cvRectangle( image, p1, p2, CV_RGB(178+i, 255-i, i), i % 10 );
    }
    Errors += ProcessImage( image, "cvRectangle", read );

#if 0
        named_window( "Diff", 0 );
#endif

    // cvCircle
    cvZero( image );

    for( i = 0; i < 100; i++ )
    {
        CvPoint p1 = cvPoint( i * 3, i * 2 );
        CvPoint p2 = cvPoint( size.width - i * 3, size.height - i * 2 );

        cvCircle( image, p1, i, CV_RGB(178+i, 255-i, i), i % 10 );
        cvCircle( image, p2, i, CV_RGB(178+i, 255-i, i), i % 10 );

#if 0
        show_iplimage( "Diff", image );
        wait_key(0);
#endif
    }
    Errors += ProcessImage( image, "cvCircle", read );

    // cvCircleAA
    cvZero( image );

    for( i = 0; i < 100; i++ )
    {
        CvPoint p1 = cvPoint( i * 3, i * 2 );
        CvPoint p2 = cvPoint( size.width - i * 3, size.height - i * 2 );

        cvCircleAA( image, p1, i, RGB(i, 255 - i, 178 + i), 0 );
        cvCircleAA( image, p2, i, RGB(i, 255 - i, 178 + i), 0 );
    }
    Errors += ProcessImage( image, "cvCircleAA", read );

    // cvEllipse
    cvZero( image );

    for( i = 10; i < 100; i += 10 )
    {
        CvPoint p1 = cvPoint( i * 6, i * 3 );
        CvSize axes = cvSize( i * 3, i * 2 );

        cvEllipse( image, p1, axes,
                   180 * i / 100, 90 * i / 100, 90 * (i - 100) / 100,
                   CV_RGB(178+i, 255-i, i), i % 10 );
    }
    Errors += ProcessImage( image, "cvEllipse", read );

    // cvEllipseAA
    cvZero( image );

    for( i = 10; i < 100; i += 10 )
    {
        CvPoint p1 = cvPoint( i * 6, i * 3 );
        CvSize axes = cvSize( i * 3, i * 2 );

        cvEllipseAA( image, p1, axes,
                   180 * i / 100, 90 * i / 100, 90 * (i - 100) / 100,
                   RGB(i, 255 - i, 178 + i), i % 10 );
    }
    Errors += ProcessImage( image, "cvEllipseAA", read );

    // cvFillConvexPoly
    cvZero( image );

    for( j = 0; j < 5; j++ )
        for( i = 0; i < 100; i += 10 )
        {
            CvPoint p[4] = {{ j * 100 - 10, i }, { j * 100 + 10, i },
                            { j * 100 + 30, i * 2 }, { j * 100 + 170, i * 3 }};
            cvFillConvexPoly( image, p, 4, CV_RGB(178+i, 255-i, i) );

        }
    Errors += ProcessImage( image, "cvFillConvexPoly", read );

    // cvFillPoly
    cvZero( image );

    for( i = 0; i < 100; i += 10 )
    {
        CvPoint p0[] = {{-10, i}, { 10, i}, { 30, i * 2}, {170, i * 3}};
        CvPoint p1[] = {{ 90, i}, {110, i}, {130, i * 2}, {270, i * 3}};
        CvPoint p2[] = {{190, i}, {210, i}, {230, i * 2}, {370, i * 3}};
        CvPoint p3[] = {{290, i}, {310, i}, {330, i * 2}, {470, i * 3}};
        CvPoint p4[] = {{390, i}, {410, i}, {430, i * 2}, {570, i * 3}};

        CvPoint* p[] = {p0, p1, p2, p3, p4};

        int n[] = {4, 4, 4, 4, 4};
        cvFillPoly( image, p, n, 5, CV_RGB(178+i, 255-i, i) );
    }
    Errors += ProcessImage( image, "cvFillPoly", read );

    // cvPolyLine
    cvZero( image );

    for( i = 0; i < 100; i += 10 )
    {
        CvPoint p0[] = {{-10, i}, { 10, i}, { 30, i * 2}, {170, i * 3}};
        CvPoint p1[] = {{ 90, i}, {110, i}, {130, i * 2}, {270, i * 3}};
        CvPoint p2[] = {{190, i}, {210, i}, {230, i * 2}, {370, i * 3}};
        CvPoint p3[] = {{290, i}, {310, i}, {330, i * 2}, {470, i * 3}};
        CvPoint p4[] = {{390, i}, {410, i}, {430, i * 2}, {570, i * 3}};

        CvPoint* p[] = {p0, p1, p2, p3, p4};

        int n[] = {4, 4, 4, 4, 4};
        cvPolyLine( image, p, n, 5, 1, CV_RGB(178+i, 255-i, i), i % 10 );
    }
    Errors += ProcessImage( image, "cvPolyLine", read );

    // cvPolyLineAA
    cvZero( image );

    for( i = 0; i < 100; i += 10 )
    {
        CvPoint p0[] = {{-10, i}, { 10, i}, { 30, i * 2}, {170, i * 3}};
        CvPoint p1[] = {{ 90, i}, {110, i}, {130, i * 2}, {270, i * 3}};
        CvPoint p2[] = {{190, i}, {210, i}, {230, i * 2}, {370, i * 3}};
        CvPoint p3[] = {{290, i}, {310, i}, {330, i * 2}, {470, i * 3}};
        CvPoint p4[] = {{390, i}, {410, i}, {430, i * 2}, {570, i * 3}};

        CvPoint* p[] = {p0, p1, p2, p3, p4};

        int n[] = {4, 4, 4, 4, 4};
        cvPolyLineAA( image, p, n, 5, 1, RGB(i, 255 - i, 178 + i), 0 );
    }
    Errors += ProcessImage( image, "cvPolyLineAA", read );

    // cvPolyLineAA
    cvZero( image );

    for( i = 1; i < 10; i++ )
    {
        CvFont font;
        cvInitFont( &font, CV_FONT_VECTOR0,
                    (double)i / 5, (double)i / 5, (double)i / 10, i );
        cvPutText( image, "privet. this is test. :)", cvPoint(0, i * 20), &font, CV_RGB(178+i, 255-i, i) );
    }
    Errors += ProcessImage( image, "cvPutText", read );

    cvReleaseImage( &image );

    return Errors ? trsResult( TRS_FAIL, "errors" ) : trsResult( TRS_OK, "ok" );
}

#if 0
static int resize_test()
{
    IplImage* src = load_iplimage( "d:/user/vp/archive/art/greatwave.jpg" );
    IplImage* image = cvCreateImage( cvGetSize( src ), 8, 1 );

    cvCvtColor( src, image, CV_BGR2GRAY );

    named_window( "image", 1 );
    show_iplimage( "image", image );
    wait_key( 0 );

    named_window( "result", 1 );

    for( int i = 0; i < 30; i++ )
    {
        IplImage* dst = cvCreateImage(
            cvSize( (rand() % 1000) + 1, (rand() % 1000) + 1),
            8, image->nChannels );

        cvResize( image, dst, CV_INTER_LINEAR );
        show_iplimage( "result", dst );

        wait_key_ex( 0, 1000 );
        cvReleaseImage( &dst );
    }

    cvReleaseImage( &image );
    destroy_window( "image" );
    destroy_window( "result" );

    return  CV_OK;
}
#endif

void InitADrawingRegress()
{
    /* Register test functions */
    trsReg( funcs, test_desc, atsAlgoClass, drawing_test );
    //trsReg( "cvResize", "", atsAlgoClass, resize_test );
}

#endif
