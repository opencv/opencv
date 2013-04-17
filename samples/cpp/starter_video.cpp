/*
* starter_video.cpp
*
*  Created on: Nov 23, 2010
*      Author: Ethan Rublee
*
*  Modified on: April 17, 2013
*      Author: Kevin Hughes
*
* A starter sample for using OpenCV VideoCapture with capture devices, video files or image sequences
* easy as CV_PI right?
*/

#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace {
    void help(char** av) {
        cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
             << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
             << "q,Q,esc -- quit" << endl
             << "space   -- save frame" << endl << endl
             << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
             << "\texample: " << av[0] << " 0" << endl
             << "\tYou may also pass a video file instead of a device number" << endl
             << "\texample: " << av[0] << " video.avi" << endl
             << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
             << "\texample: " << av[0] << " right%%02d.jpg" << endl;
    }

    int process(VideoCapture& capture) {
        int n = 0;
        char filename[200];
        string window_name = "video | q or esc to quit";
        cout << "press space to save a picture. q or esc to quit" << endl;
        namedWindow(window_name, WINDOW_NORMAL); //resizable window;
        Mat frame;

        for (;;) {
            capture >> frame;
            if (frame.empty())
                break;

            imshow(window_name, frame);
            char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

            switch (key) {
            case 'q':
            case 'Q':
            case 27: //escape key
                return 0;
            case ' ': //Save an image
                sprintf(filename,"filename%.3d.jpg",n++);
                imwrite(filename,frame);
                cout << "Saved " << filename << endl;
                break;
            default:
                break;
            }
        }
        return 0;
    }
}

int main(int ac, char** av) {

    if (ac != 2) {
        help(av);
        return 1;
    }
    std::string arg = av[1];
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(arg.c_str()));
    if (!capture.isOpened()) {
        cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
        help(av);
        return 1;
    }
    return process(capture);
}
