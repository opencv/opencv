#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/utility.hpp"

#include <ctype.h>
#include <stdio.h>

static void help( void )
{
    printf("\nThis program illustrates Linear-Polar and Log-Polar image transforms\n"
            "Usage :\n"
            "./polar_transforms [[camera number -- Default 0],[AVI path_filename]]\n\n"
            );
}
int main( int argc, char** argv )
{
    CvCapture* capture = 0;
    IplImage*  log_polar_img = 0;
    IplImage*  lin_polar_img = 0;
    IplImage*  recovered_img = 0;

    help();
    cv::CommandLineParser parser(argc, argv, "{help h||}{@input|0|}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    std::string arg = parser.get<std::string>("@input");
    if( arg.size() == 1 && isdigit(arg[0]) )
        capture = cvCaptureFromCAM( arg[0] - '0' );
    else
        capture = cvCaptureFromAVI( arg.c_str() );
    if( !capture )
    {
        const char* name = argv[0];
        fprintf(stderr,"Could not initialize capturing...\n");
        fprintf(stderr,"Usage: %s <CAMERA_NUMBER>    , or \n       %s <VIDEO_FILE>\n", name, name);
        help();
        return -1;
    }

    cvNamedWindow( "Linear-Polar", 0 );
    cvNamedWindow( "Log-Polar", 0 );
    cvNamedWindow( "Recovered image", 0 );

    cvMoveWindow( "Linear-Polar", 20,20 );
    cvMoveWindow( "Log-Polar", 700,20 );
    cvMoveWindow( "Recovered image", 20,700 );

    for(;;)
    {
        IplImage* frame = 0;

        frame = cvQueryFrame( capture );
        if( !frame )
            break;

        if( !log_polar_img )
        {
            log_polar_img = cvCreateImage( cvSize(frame->width,frame->height), IPL_DEPTH_8U, frame->nChannels );
            lin_polar_img = cvCreateImage( cvSize(frame->width,frame->height), IPL_DEPTH_8U, frame->nChannels );
            recovered_img = cvCreateImage( cvSize(frame->width,frame->height), IPL_DEPTH_8U, frame->nChannels );
        }

        cvLogPolar(frame,log_polar_img,cvPoint2D32f(frame->width >> 1,frame->height >> 1),70, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
        cvLinearPolar(frame,lin_polar_img,cvPoint2D32f(frame->width >> 1,frame->height >> 1),70, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);

#if 0
        cvLogPolar(log_polar_img,recovered_img,cvPoint2D32f(frame->width >> 1,frame->height >> 1),70, CV_WARP_INVERSE_MAP+CV_INTER_LINEAR);
#else
        cvLinearPolar(lin_polar_img,recovered_img,cvPoint2D32f(frame->width >> 1,frame->height >> 1),70, CV_WARP_INVERSE_MAP+CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
#endif

        cvShowImage("Log-Polar", log_polar_img );
        cvShowImage("Linear-Polar", lin_polar_img );
        cvShowImage("Recovered image", recovered_img );

        if( cvWaitKey(10) >= 0 )
            break;
    }

    cvReleaseCapture( &capture );
    cvDestroyWindow("Linear-Polar");
    cvDestroyWindow("Log-Polar");
    cvDestroyWindow("Recovered image");

    return 0;
}

#ifdef _EiC
main(1,"laplace.c");
#endif
