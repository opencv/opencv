//TickMeter sample program
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    VideoCapture cap;
    if(argc>1)
    {
        cap.open(argv[1]);
    }
    else
    {
        cap.open(0);
    }
    Mat frame;
    TickMeter t;
    namedWindow("TickMeter",1);
    while(true)
    {
        t.start();
        cap>>frame;
        imshow("TickMeter",frame);
        waitKey(1);
        t.stop();
        cout<<"Current FPS = "<<t.getFps()<<endl;
    }
    return 0;
}