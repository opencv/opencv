#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
            "\nThis program demonstrates Laplace point/edge detection using OpenCV function Laplacian()\n"
            "It captures from the camera of your choice: 0, 1, ... default 0\n"
            "Call:\n"
            "./laplace [camera #, default 0]\n" << endl;
}

int sigma = 3;
int smoothType = CV_GAUSSIAN;

int main( int argc, char** argv )
{
    VideoCapture cap;
    help();

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if( argc >= 2 )
    {
        cap.open(argv[1]);
        if( cap.isOpened() )
            cout << "Video " << argv[1] <<
                ": width=" << cap.get(CV_CAP_PROP_FRAME_WIDTH) <<
                ", height=" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) <<
                ", nframes=" << cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;
        if( argc > 2 && isdigit(argv[2][0]) )
        {
            int pos;
            sscanf(argv[2], "%d", &pos);
            cout << "seeking to frame #" << pos << endl;
            cap.set(CV_CAP_PROP_POS_FRAMES, pos);
        }
    }

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return -1;
    }

    namedWindow( "Laplacian", 0 );
    createTrackbar( "Sigma", "Laplacian", &sigma, 15, 0 );

    Mat smoothed, laplace, result;

    for(;;)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        int ksize = (sigma*5)|1;
        if(smoothType == CV_GAUSSIAN)
            GaussianBlur(frame, smoothed, Size(ksize, ksize), sigma, sigma);
        else if(smoothType == CV_BLUR)
            blur(frame, smoothed, Size(ksize, ksize));
        else
            medianBlur(frame, smoothed, ksize);

        Laplacian(smoothed, laplace, CV_16S, 5);
        convertScaleAbs(laplace, result, (sigma+1)*0.25);
        imshow("Laplacian", result);

        int c = waitKey(30);
        if( c == ' ' )
            smoothType = smoothType == CV_GAUSSIAN ? CV_BLUR : smoothType == CV_BLUR ? CV_MEDIAN : CV_GAUSSIAN;
        if( c == 'q' || c == 'Q' || (c & 255) == 27 )
            break;
    }

    return 0;
}
