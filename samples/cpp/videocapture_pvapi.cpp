//////////////////////////////////////////////////////
// Sample illustrating the use of the VideoCapture  //
// interface in combination with PvAPI interface    //
//                                                  //
// Succesfully tested on Prosilica and Manta series //
//////////////////////////////////////////////////////

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
    // Capturing multiple AVT cameras can be done by simply initiating
    // two VideoCaptures after eachother.
    VideoCapture camera1(0 + CV_CAP_PVAPI);
    VideoCapture camera2(0 + CV_CAP_PVAPI);
    Mat frame1, frame2;

    for(;;){
        camera1 >> frame1;
        camera2 >> frame2;

        imshow("camera 1 frame", frame1);
        imshow("camera 2 frame", frame2);

        int key = waitKey(10);
        if(key == 27){
            break;
        }
    }

    return 0;
}
