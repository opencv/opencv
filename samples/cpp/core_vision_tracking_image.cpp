#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/core_vision_api/tracker.hpp>

#include <stdio.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

const string WindowName = "Face Detection example";
const Scalar RectColor = CV_RGB(0,255,0);

int main()
{
    namedWindow(WindowName);
    cv::moveWindow(WindowName, 100, 100);

    Mat Viewport;
    Mat ReferenceFrame = imread("board.jpg");
    if (ReferenceFrame.empty())
    {
        printf("Error: Cannot load input image\n");
        return 1;
    }

    cv::Ptr<nv::Tracker> tracker = nv::Algorithm::create<nv::Tracker>("nv::Tracker::OpticalFlow");

    tracker->initialize();

    // First frame for initialization
    tracker->feed(ReferenceFrame);

    nv::Tracker::TrackedObjectHandler obj = tracker->addObject(cv::Rect(100,100, 200, 200));

    while(true)
    {
        tracker->feed(ReferenceFrame);

        if (obj->getStatus() == nv::Tracker::LOST_STATUS)
            break;

        cv::Rect currentLocation = obj->getLocation();

        ReferenceFrame.copyTo(Viewport);
        rectangle(Viewport, currentLocation, RectColor);

        imshow(WindowName, Viewport);

        if (cvWaitKey(30) >= 0) break;
    }

    return 0;
}
