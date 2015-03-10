//////////////////////////////////////////////////////
// Sample illustrating the use of the VideoCapture  //
// interface in combination with PvAPI interface    //
//                                                  //
// Succesfully tested on Prosilica and Manta series //
//////////////////////////////////////////////////////

// --------------------------------------------------------------------------------
// Some remarks for ensuring the correct working of the interface between camera
// and the pc from which you will capture data - Linux based settings. The settings
// for Windows are the same, but edited in the graphical interface of the
// network card.
//
// You have to be sure that OpenCV is built with the PvAPI interface enabled.
//
// FIRST CONFIGURE IP SETTINGS
// - Change the IP address of your pc to 169.254.1.1
// - Change the subnet mask of your pc to 255.255.0.0
// - Change the gateway of your pc to 169.254.1.2
//
// CHANGE SOME NETWORK CARD SETTINGS
// - sudo ifconfig eth0 mtu 9000 - or 9016 ideally if your card supports that
// --------------------------------------------------------------------------------

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
