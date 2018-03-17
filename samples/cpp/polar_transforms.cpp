#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

static void help( void )
{
    printf("\nThis program illustrates Linear-Polar and Log-Polar image transforms\n"
            "Usage :\n"
            "./polar_transforms [[camera number -- Default 0],[path_to_filename]]\n\n");
}

int main( int argc, char** argv )
{
    VideoCapture capture;
    Mat log_polar_img, lin_polar_img, recovered_log_polar, recovered_lin_polar_img;

    help();

    CommandLineParser parser(argc, argv, "{@input|0|}");
    std::string arg = parser.get<std::string>("@input");

    if( arg.size() == 1 && isdigit(arg[0]) )
        capture.open( arg[0] - '0' );
    else
        capture.open( arg.c_str() );

    if( !capture.isOpened() )
    {
        const char* name = argv[0];
        fprintf(stderr,"Could not initialize capturing...\n");
        fprintf(stderr,"Usage: %s <CAMERA_NUMBER>    , or \n       %s <VIDEO_FILE>\n", name, name);
        return -1;
    }

    namedWindow( "Linear-Polar", WINDOW_AUTOSIZE );
    namedWindow( "Log-Polar", WINDOW_AUTOSIZE);
    namedWindow( "Recovered Linear-Polar", WINDOW_AUTOSIZE);
    namedWindow( "Recovered Log-Polar", WINDOW_AUTOSIZE);

    moveWindow( "Linear-Polar", 20,20 );
    moveWindow( "Log-Polar", 700,20 );
    moveWindow( "Recovered Linear-Polar", 20, 350 );
    moveWindow( "Recovered Log-Polar", 700, 350 );

    for(;;)
    {
        Mat frame;
        capture >> frame;

        if( frame.empty() )
            break;

        Point2f center( (float)frame.cols / 2, (float)frame.rows / 2 );
        double M = 70;

        logPolar(frame,log_polar_img, center, M, INTER_LINEAR + WARP_FILL_OUTLIERS);
        linearPolar(frame,lin_polar_img, center, M, INTER_LINEAR + WARP_FILL_OUTLIERS);

        logPolar(log_polar_img, recovered_log_polar, center, M, WARP_INVERSE_MAP + INTER_LINEAR);
        linearPolar(lin_polar_img, recovered_lin_polar_img, center, M, WARP_INVERSE_MAP + INTER_LINEAR + WARP_FILL_OUTLIERS);

        imshow("Log-Polar", log_polar_img );
        imshow("Linear-Polar", lin_polar_img );
        imshow("Recovered Linear-Polar", recovered_lin_polar_img );
        imshow("Recovered Log-Polar", recovered_log_polar );

        if( waitKey(10) >= 0 )
            break;
    }

    waitKey(0);
    return 0;
}
