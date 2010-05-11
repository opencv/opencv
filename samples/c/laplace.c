#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <ctype.h>
#include <stdio.h>
#endif

int sigma = 3;
int smoothType = CV_GAUSSIAN;

int main( int argc, char** argv )
{
    IplImage* laplace = 0;
    IplImage* colorlaplace = 0;
    IplImage* planes[3] = { 0, 0, 0 };
    CvCapture* capture = 0;

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );
    else if( argc == 2 )
        capture = cvCaptureFromAVI( argv[1] );

    if( !capture )
    {
        fprintf(stderr,"Could not initialize capturing...\n");
        return -1;
    }

    cvNamedWindow( "Laplacian", 0 );
    cvCreateTrackbar( "Sigma", "Laplacian", &sigma, 15, 0 );

    for(;;)
    {
        IplImage* frame = 0;
        int i, c, ksize;

        frame = cvQueryFrame( capture );
        if( !frame )
            break;

        if( !laplace )
        {
            for( i = 0; i < 3; i++ )
                planes[i] = cvCreateImage( cvGetSize(frame), 8, 1 );
            laplace = cvCreateImage( cvGetSize(frame), IPL_DEPTH_16S, 1 );
            colorlaplace = cvCreateImage( cvGetSize(frame), 8, 3 );
        }

        ksize = (sigma*5)|1;
        cvSmooth( frame, colorlaplace, smoothType, ksize, ksize, sigma, sigma );
        cvSplit( colorlaplace, planes[0], planes[1], planes[2], 0 );
        for( i = 0; i < 3; i++ )
        {
            cvLaplace( planes[i], laplace, 5 );
            cvConvertScaleAbs( laplace, planes[i], (sigma+1)*0.25, 0 );
        }
        cvMerge( planes[0], planes[1], planes[2], 0, colorlaplace );
        colorlaplace->origin = frame->origin;

        cvShowImage("Laplacian", colorlaplace );

        c = cvWaitKey(30);
        if( c == ' ' )
            smoothType = smoothType == CV_GAUSSIAN ? CV_BLUR : smoothType == CV_BLUR ? CV_MEDIAN : CV_GAUSSIAN;
        if( c == 'q' || c == 'Q' || (c & 255) == 27 )
            break;
    }

    cvReleaseCapture( &capture );
    cvDestroyWindow("Laplacian");

    return 0;
}

#ifdef _EiC
main(1,"laplace.c");
#endif
