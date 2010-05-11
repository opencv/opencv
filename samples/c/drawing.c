#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>
#endif

#define NUMBER 100
#define DELAY 5
char wndname[] = "Drawing Demo";

CvScalar random_color(CvRNG* rng)
{
    int icolor = cvRandInt(rng);
    return CV_RGB(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}

int main( int argc, char** argv )
{
    int line_type = CV_AA; // change it to 8 to see non-antialiased graphics
    int i;
    CvPoint pt1,pt2;
    double angle;
    CvSize sz;
    CvPoint  ptt[6];
    CvPoint* pt[2];
    int  arr[2];
    CvFont font;
    CvRNG rng;
    int width = 1000, height = 700;
    int width3 = width*3, height3 = height*3;
    CvSize text_size;
    int ymin = 0;
    // Load the source image
    IplImage* image = cvCreateImage( cvSize(width,height), 8, 3 );
    IplImage* image2;

    // Create a window
    cvNamedWindow(wndname, 1 );
    cvZero( image );
    cvShowImage(wndname,image);
    cvWaitKey(DELAY);

    rng = cvRNG((unsigned)-1);
    pt[0] = &(ptt[0]);
    pt[1] = &(ptt[3]);

    arr[0] = 3;
    arr[1] = 3;

    for (i = 0; i< NUMBER; i++)
    {
        pt1.x=cvRandInt(&rng) % width3 - width;
        pt1.y=cvRandInt(&rng) % height3 - height;
        pt2.x=cvRandInt(&rng) % width3 - width;
        pt2.y=cvRandInt(&rng) % height3 - height;

        cvLine( image, pt1, pt2, random_color(&rng), cvRandInt(&rng)%10, line_type, 0 );
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    for (i = 0; i< NUMBER; i++)
    {
        pt1.x=cvRandInt(&rng) % width3 - width;
        pt1.y=cvRandInt(&rng) % height3 - height;
        pt2.x=cvRandInt(&rng) % width3 - width;
        pt2.y=cvRandInt(&rng) % height3 - height;

        cvRectangle( image,pt1, pt2, random_color(&rng), cvRandInt(&rng)%10-1, line_type, 0 );
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    for (i = 0; i< NUMBER; i++)
    {
        pt1.x=cvRandInt(&rng) % width3 - width;
        pt1.y=cvRandInt(&rng) % height3 - height;
        sz.width =cvRandInt(&rng)%200;
        sz.height=cvRandInt(&rng)%200;
        angle = (cvRandInt(&rng)%1000)*0.180;

        cvEllipse( image, pt1, sz, angle, angle - 100, angle + 200,
                   random_color(&rng), cvRandInt(&rng)%10-1, line_type, 0 );
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    for (i = 0; i< NUMBER; i++)
    {
        pt[0][0].x=cvRandInt(&rng) % width3 - width;
        pt[0][0].y=cvRandInt(&rng) % height3 - height;
        pt[0][1].x=cvRandInt(&rng) % width3 - width;
        pt[0][1].y=cvRandInt(&rng) % height3 - height;
        pt[0][2].x=cvRandInt(&rng) % width3 - width;
        pt[0][2].y=cvRandInt(&rng) % height3 - height;
        pt[1][0].x=cvRandInt(&rng) % width3 - width;
        pt[1][0].y=cvRandInt(&rng) % height3 - height;
        pt[1][1].x=cvRandInt(&rng) % width3 - width;
        pt[1][1].y=cvRandInt(&rng) % height3 - height;
        pt[1][2].x=cvRandInt(&rng) % width3 - width;
        pt[1][2].y=cvRandInt(&rng) % height3 - height;

        cvPolyLine( image, pt, arr, 2, 1, random_color(&rng), cvRandInt(&rng)%10, line_type, 0 );
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    for (i = 0; i< NUMBER; i++)
    {
        pt[0][0].x=cvRandInt(&rng) % width3 - width;
        pt[0][0].y=cvRandInt(&rng) % height3 - height;
        pt[0][1].x=cvRandInt(&rng) % width3 - width;
        pt[0][1].y=cvRandInt(&rng) % height3 - height;
        pt[0][2].x=cvRandInt(&rng) % width3 - width;
        pt[0][2].y=cvRandInt(&rng) % height3 - height;
        pt[1][0].x=cvRandInt(&rng) % width3 - width;
        pt[1][0].y=cvRandInt(&rng) % height3 - height;
        pt[1][1].x=cvRandInt(&rng) % width3 - width;
        pt[1][1].y=cvRandInt(&rng) % height3 - height;
        pt[1][2].x=cvRandInt(&rng) % width3 - width;
        pt[1][2].y=cvRandInt(&rng) % height3 - height;

        cvFillPoly( image, pt, arr, 2, random_color(&rng), line_type, 0 );
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    for (i = 0; i< NUMBER; i++)
    {
        pt1.x=cvRandInt(&rng) % width3 - width;
        pt1.y=cvRandInt(&rng) % height3 - height;

        cvCircle( image, pt1, cvRandInt(&rng)%300, random_color(&rng),
                  cvRandInt(&rng)%10-1, line_type, 0 );
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    for (i = 1; i< NUMBER; i++)
    {
        pt1.x=cvRandInt(&rng) % width3 - width;
        pt1.y=cvRandInt(&rng) % height3 - height;

        cvInitFont( &font, cvRandInt(&rng) % 8,
                    (cvRandInt(&rng)%100)*0.05+0.1, (cvRandInt(&rng)%100)*0.05+0.1,
                    (cvRandInt(&rng)%5)*0.1, cvRound(cvRandInt(&rng)%10), line_type );

        cvPutText( image, "Testing text rendering!", pt1, &font, random_color(&rng));
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    cvInitFont( &font, CV_FONT_HERSHEY_COMPLEX, 3, 3, 0.0, 5, line_type );

    cvGetTextSize( "OpenCV forever!", &font, &text_size, &ymin );

    pt1.x = (width - text_size.width)/2;
    pt1.y = (height + text_size.height)/2;
    image2 = cvCloneImage(image);

    for( i = 0; i < 255; i++ )
    {
        cvSubS( image2, cvScalarAll(i), image, 0 );
        cvPutText( image, "OpenCV forever!", pt1, &font, CV_RGB(255,i,i));
        cvShowImage(wndname,image);
        if(cvWaitKey(DELAY) >= 0) return 0;
    }

    // Wait for a key stroke; the same function arranges events processing
    cvWaitKey(0);
    cvReleaseImage(&image);
    cvReleaseImage(&image2);
    cvDestroyWindow(wndname);

    return 0;
}

#ifdef _EiC
main(1,"drawing.c");
#endif
