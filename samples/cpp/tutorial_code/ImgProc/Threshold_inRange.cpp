#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

/** Function Headers */
void on_low_r_thresh_trackbar(int, void *);
void on_high_r_thresh_trackbar(int, void *);
void on_low_g_thresh_trackbar(int, void *);
void on_high_g_thresh_trackbar(int, void *);
void on_low_b_thresh_trackbar(int, void *);
void on_high_b_thresh_trackbar(int, void *);

/** Global Variables */
int low_r=30, low_g=30, low_b=30;
int high_r=100, high_g=100, high_b=100;

/** @function main */
int main()
{
    //! [mat]
    Mat frame, frame_threshold;
    //! [mat]
    //! [cap]
    VideoCapture cap(0);
    //! [cap]
    //! [window]
    namedWindow("Video Capture", WINDOW_NORMAL);
    namedWindow("Object Detection", WINDOW_NORMAL);
    //! [window]
    //! [trackbar]
    //-- Trackbars to set thresholds for RGB values
    createTrackbar("Low R","Object Detection", &low_r, 255, on_low_r_thresh_trackbar);
    createTrackbar("High R","Object Detection", &high_r, 255, on_high_r_thresh_trackbar);
    createTrackbar("Low G","Object Detection", &low_g, 255, on_low_g_thresh_trackbar);
    createTrackbar("High G","Object Detection", &high_g, 255, on_high_g_thresh_trackbar);
    createTrackbar("Low B","Object Detection", &low_b, 255, on_low_b_thresh_trackbar);
    createTrackbar("High B","Object Detection", &high_b, 255, on_high_b_thresh_trackbar);
    //! [trackbar]
    while((char)waitKey(1)!='q'){
        //! [while]
        cap>>frame;
        if(frame.empty())
            break;
        //-- Detect the object based on RGB Range Values
        inRange(frame,Scalar(low_b,low_g,low_r), Scalar(high_b,high_g,high_r),frame_threshold);
        //! [while]
        //! [show]
        //-- Show the frames
        imshow("Video Capture",frame);
        imshow("Object Detection",frame_threshold);
        //! [show]
    }
    return 0;
}
//! [low]
/** @function on_low_r_thresh_trackbar */
void on_low_r_thresh_trackbar(int, void *)
{
    low_r = min(high_r-1, low_r);
    setTrackbarPos("Low R","Object Detection", low_r);
}
//! [low]
//! [high]
/** @function on_high_r_thresh_trackbar */
void on_high_r_thresh_trackbar(int, void *)
{
    high_r = max(high_r, low_r+1);
    setTrackbarPos("High R", "Object Detection", high_r);
}
//![high]
/** @function on_low_g_thresh_trackbar */
void on_low_g_thresh_trackbar(int, void *)
{
    low_g = min(high_g-1, low_g);
    setTrackbarPos("Low G","Object Detection", low_g);
}

/** @function on_high_g_thresh_trackbar */
void on_high_g_thresh_trackbar(int, void *)
{
    high_g = max(high_g, low_g+1);
    setTrackbarPos("High G", "Object Detection", high_g);
}

/** @function on_low_b_thresh_trackbar */
void on_low_b_thresh_trackbar(int, void *)
{
    low_b= min(high_b-1, low_b);
    setTrackbarPos("Low B","Object Detection", low_b);
}

/** @function on_high_b_thresh_trackbar */
void on_high_b_thresh_trackbar(int, void *)
{
    high_b = max(high_b, low_b+1);
    setTrackbarPos("High B", "Object Detection", high_b);
}
