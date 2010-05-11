#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#endif

char file_name[] = "baboon.jpg";

int _brightness = 100;
int _contrast = 100;

int hist_size = 64;
float range_0[]={0,256};
float* ranges[] = { range_0 };
IplImage *src_image = 0, *dst_image = 0, *hist_image = 0;
CvHistogram *hist;
uchar lut[256];
CvMat* lut_mat;

/* brightness/contrast callback function */
void update_brightcont( int arg )
{
    int brightness = _brightness - 100;
    int contrast = _contrast - 100;
    int i, bin_w;
    float max_value = 0;

    /*
     * The algorithm is by Werner D. Streidt
     * (http://visca.com/ffactory/archives/5-99/msg00021.html)
     */
    if( contrast > 0 )
    {
        double delta = 127.*contrast/100;
        double a = 255./(255. - delta*2);
        double b = a*(brightness - delta);
        for( i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
            lut[i] = (uchar)v;
        }
    }
    else
    {
        double delta = -128.*contrast/100;
        double a = (256.-delta*2)/255.;
        double b = a*brightness + delta;
        for( i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
            lut[i] = (uchar)v;
        }
    }

    cvLUT( src_image, dst_image, lut_mat );
    cvShowImage( "image", dst_image );

    cvCalcHist( &dst_image, hist, 0, NULL );
    cvZero( dst_image );
    cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
    cvScale( hist->bins, hist->bins, ((double)hist_image->height)/max_value, 0 );
    /*cvNormalizeHist( hist, 1000 );*/

    cvSet( hist_image, cvScalarAll(255), 0 );
    bin_w = cvRound((double)hist_image->width/hist_size);

    for( i = 0; i < hist_size; i++ )
        cvRectangle( hist_image, cvPoint(i*bin_w, hist_image->height),
                     cvPoint((i+1)*bin_w, hist_image->height - cvRound(cvGetReal1D(hist->bins,i))),
                     cvScalarAll(0), -1, 8, 0 );

    cvShowImage( "histogram", hist_image );
}


int main( int argc, char** argv )
{
    // Load the source image. HighGUI use.
    src_image = cvLoadImage( argc == 2 ? argv[1] : file_name, 0 );

    if( !src_image )
    {
        printf("Image was not loaded.\n");
        return -1;
    }

    dst_image = cvCloneImage(src_image);
    hist_image = cvCreateImage(cvSize(320,200), 8, 1);
    hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
    lut_mat = cvCreateMatHeader( 1, 256, CV_8UC1 );
    cvSetData( lut_mat, lut, 0 );

    cvNamedWindow("image", 0);
    cvNamedWindow("histogram", 0);

    cvCreateTrackbar("brightness", "image", &_brightness, 200, update_brightcont);
    cvCreateTrackbar("contrast", "image", &_contrast, 200, update_brightcont);

    update_brightcont(0);
    cvWaitKey(0);

    cvReleaseImage(&src_image);
    cvReleaseImage(&dst_image);

    cvReleaseHist(&hist);

    return 0;
}

#ifdef _EiC
main(1,"demhist.c");
#endif

