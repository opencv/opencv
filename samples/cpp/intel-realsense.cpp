/**
  @file ela.cpp
  @author Alessandro de Oliveira Faria (A.K.A. CABELO)
  @brief Example for capture in openCV data from the legacy Intel® RealSense™ F200, SR300, R200, LR200 and the ZR300 cameras.
  @date Sep 23, 2018
*/


//Compile: g++ -std=c++11 realsense.cpp -lrealsense -o realsense `pkg-config --libs --cflags opencv`

// include the librealsense C++ header file
#include <librealsense/rs.hpp>
// include OpenCV header file
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    int c;
    // Create a context object 
    rs::context ctx;
   
    // Open first available RealSense device
    rs::device * dev = ctx.get_device(0);

    // Configure all stream in VGA resolution at 30 frames per second
    dev->enable_stream(rs::stream::color, 640, 480, rs::format::bgr8, 30);
    dev->enable_stream(rs::stream::infrared, 640, 480, rs::format::y8, 30);
    dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);

    // Start realsense streaming
    dev->start();

    //  Dropped several first  40 frames to let auto-exposure stabilize
    for(int i = 0; i < 40; i++)
       dev->wait_for_frames();

    while(true)
    {
        // Capture frame    
        dev->wait_for_frames();

        // Creating OpenCV Matrix from a color image
        Mat color(Size(640, 480), CV_8UC3, (void*)dev->get_frame_data(rs::stream::color), Mat::AUTO_STEP);
        Mat ir(Size(640, 480), CV_8UC1, (void*)dev->get_frame_data(rs::stream::infrared), Mat::AUTO_STEP);
        Mat depth(Size(640, 480), CV_16U, (void*)dev->get_frame_data(rs::stream::depth), Mat::AUTO_STEP);

	//Show IR, Depth and Color Frame
        imshow("Display Image Color", color);
        imshow("Display Image IR", ir);
        imshow("Display Image Depth", depth);

	c = waitKey(5);
        if( c == 27 || c == 'q' || c == 'Q' )
        break;
    }

    return 0;
}

