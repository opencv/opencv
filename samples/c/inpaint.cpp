#ifdef _CH_
#pragma package <opencv>
#endif

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>

IplImage* inpaint_mask = 0;
IplImage* img0 = 0, *img = 0, *inpainted = 0;
CvPoint prev_pt = {-1,-1};

void on_mouse( int event, int x, int y, int flags, void* )
{
    if( !img )
        return;

    if( event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON) )
        prev_pt = cvPoint(-1,-1);
    else if( event == CV_EVENT_LBUTTONDOWN )
        prev_pt = cvPoint(x,y);
    else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) )
    {
        CvPoint pt = cvPoint(x,y);
        if( prev_pt.x < 0 )
            prev_pt = pt;
        cvLine( inpaint_mask, prev_pt, pt, cvScalarAll(255), 5, 8, 0 );
        cvLine( img, prev_pt, pt, cvScalarAll(255), 5, 8, 0 );
        prev_pt = pt;
        cvShowImage( "image", img );
    }
}


int main( int argc, char** argv )
{
    char* filename = argc >= 2 ? argv[1] : (char*)"fruits.jpg";

    if( (img0 = cvLoadImage(filename,-1)) == 0 )
        return 0;

    printf( "Hot keys: \n"
            "\tESC - quit the program\n"
            "\tr - restore the original image\n"
            "\ti or SPACE - run inpainting algorithm\n"
            "\t\t(before running it, paint something on the image)\n" );
    
    cvNamedWindow( "image", 1 );

    img = cvCloneImage( img0 );
    inpainted = cvCloneImage( img0 );
    inpaint_mask = cvCreateImage( cvGetSize(img), 8, 1 );

    cvZero( inpaint_mask );
    cvZero( inpainted );
    cvShowImage( "image", img );
    cvShowImage( "inpainted image", inpainted );
    cvSetMouseCallback( "image", on_mouse, 0 );

    for(;;)
    {
        int c = cvWaitKey(0);

        if( (char)c == 27 )
            break;

        if( (char)c == 'r' )
        {
            cvZero( inpaint_mask );
            cvCopy( img0, img );
            cvShowImage( "image", img );
        }

        if( (char)c == 'i' || (char)c == ' ' )
        {
            cvNamedWindow( "inpainted image", 1 );
            cvInpaint( img, inpaint_mask, inpainted, 3, CV_INPAINT_TELEA );
            cvShowImage( "inpainted image", inpainted );
        }
    }

    return 1;
}
