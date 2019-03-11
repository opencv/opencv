#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

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
            "./laplace -c=<camera #, default 0> -p=<index of the frame to be decoded/captured next>\n" << endl;
}

enum {GAUSSIAN, BLUR, MEDIAN};

int sigma = 3;
int smoothType = GAUSSIAN;

int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, "{ c | 0 | }{ p | | }");
    help();

    VideoCapture cap;
    string camera = parser.get<string>("c");
    if (camera.size() == 1 && isdigit(camera[0]))
        cap.open(parser.get<int>("c"));
    else
        cap.open(samples::findFileOrKeep(camera));
    if (!cap.isOpened())
    {
        cerr << "Can't open camera/video stream: " << camera << endl;
        return 1;
    }
    cout << "Video " << parser.get<string>("c") <<
        ": width=" << cap.get(CAP_PROP_FRAME_WIDTH) <<
        ", height=" << cap.get(CAP_PROP_FRAME_HEIGHT) <<
        ", nframes=" << cap.get(CAP_PROP_FRAME_COUNT) << endl;
    int pos = 0;
    if (parser.has("p"))
    {
        pos = parser.get<int>("p");
    }
    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    if (pos != 0)
    {
        cout << "seeking to frame #" << pos << endl;
        if (!cap.set(CAP_PROP_POS_FRAMES, pos))
        {
            cerr << "ERROR: seekeing is not supported" << endl;
        }
    }

    namedWindow("Laplacian", WINDOW_AUTOSIZE);
    createTrackbar("Sigma", "Laplacian", &sigma, 15, 0);

    Mat smoothed, laplace, result;

    for(;;)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        int ksize = (sigma*5)|1;
        if(smoothType == GAUSSIAN)
            GaussianBlur(frame, smoothed, Size(ksize, ksize), sigma, sigma);
        else if(smoothType == BLUR)
            blur(frame, smoothed, Size(ksize, ksize));
        else
            medianBlur(frame, smoothed, ksize);

        Laplacian(smoothed, laplace, CV_16S, 5);
        convertScaleAbs(laplace, result, (sigma+1)*0.25);
        imshow("Laplacian", result);

        char c = (char)waitKey(30);
        if( c == ' ' )
            smoothType = smoothType == GAUSSIAN ? BLUR : smoothType == BLUR ? MEDIAN : GAUSSIAN;
        if( c == 'q' || c == 'Q' || c == 27 )
            break;
    }

    return 0;
}
