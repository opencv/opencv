#include "cvaux.h"
#include "highgui.h"
#include <stdio.h>

//this is a sample for foreground detection functions
int main(int argc, char** argv)
{
    IplImage*       tmp_frame = NULL;
    CvCapture*      cap = NULL;
    bool update_bg_model = true;

    if( argc < 2 )
        cap = cvCaptureFromCAM(0);
    else
        cap = cvCaptureFromFile(argv[1]);
    
    if( !cap )
    {
        printf("can not open camera or video file\n");
        return -1;
    }
    
    tmp_frame = cvQueryFrame(cap);
    if(!tmp_frame)
    {
        printf("can not read data from the video source\n");
        return -1;
    }

    cvNamedWindow("BG", 1);
    cvNamedWindow("FG", 1);

    CvBGStatModel* bg_model = 0;
    
    for( int fr = 1;tmp_frame; tmp_frame = cvQueryFrame(cap), fr++ )
    {
        if(!bg_model)
        {
            //create BG model
            bg_model = cvCreateGaussianBGModel( tmp_frame );
            //bg_model = cvCreateFGDStatModel( temp );
            continue;
        }
        
        double t = (double)cvGetTickCount();
        cvUpdateBGStatModel( tmp_frame, bg_model, update_bg_model ? -1 : 0 );
        t = (double)cvGetTickCount() - t;
        printf( "%d. %.1f\n", fr, t/(cvGetTickFrequency()*1000.) );
        cvShowImage("BG", bg_model->background);
        cvShowImage("FG", bg_model->foreground);
        char k = cvWaitKey(5);
        if( k == 27 ) break;
        if( k == ' ' )
            update_bg_model = !update_bg_model;
    }


    cvReleaseBGStatModel( &bg_model );
    cvReleaseCapture(&cap);

    return 0;
}
