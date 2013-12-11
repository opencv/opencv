/*
 * starter_video.cpp
 *
 *  Created on: Nov 23, 2010
 *      Author: Ethan Rublee
 *
 * A starter sample for using opencv, get a video stream and display the images
 * Use http://datamatrix.kaywa.com/  to generate datamatrix images using strings of length 3 or less.
 * easy as CV_PI right?
 */
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace
{
    void help(char** av)
    {
        cout << "\nThis program justs gets you started reading images from video\n"
        "Usage:\n./" << av[0] << " <video device number>\n" << "q,Q,esc -- quit\n"
        << "space   -- save frame\n\n"
        << "\tThis is a starter sample, to get you up and going in a copy pasta fashion\n"
        << "\tThe program captures frames from a camera connected to your computer.\n"
        << "\tTo find the video device number, try ls /dev/video* \n"
        << "\tYou may also pass a video file, like my_vide.avi instead of a device number"
        << "\n"
        << "DATA:\n"
        << "Generate a datamatrix from  from http://datamatrix.kaywa.com/  \n"
        << "  NOTE: This only handles strings of len 3 or less\n"
        << "  Resize the screen to be large enough for your camera to see, and it should find an read it.\n\n"
        << endl;
    }

    int process(VideoCapture& capture)
    {
        int n = 0;
        char filename[200];
        string window_name = "video | q or esc to quit";
        cout << "press space to save a picture. q or esc to quit" << endl;
        namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
        Mat frame;
        for (;;)
        {
            capture >> frame;
            if (frame.empty())
                break;
            cv::Mat gray;
            cv::cvtColor(frame,gray,COLOR_RGB2GRAY);
            vector<String> codes;
            Mat corners;
            findDataMatrix(gray, codes, corners);
            drawDataMatrixCodes(frame, codes, corners);
            imshow(window_name, frame);
            char key = (char) waitKey(5); //delay N millis, usually long enough to display and capture input
            switch (key)
            {
                case 'q':
                case 'Q':
                case 27: //escape key
                    return 0;
                case ' ': //Save an image
                    sprintf(filename, "filename%.3d.jpg", n++);
                    imwrite(filename, frame);
                    cout << "Saved " << filename << endl;
                    break;
                default:
                    break;
            }
        }
        return 0;
    }

}

int main(int ac, char** av)
{

    if (ac != 2)
    {
        help(av);
        return 1;
    }
    std::string arg = av[1];
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(arg.c_str()));
    if (!capture.isOpened())
    {
        cerr << "Failed to open a video device or video file!\n" << endl;
        help(av);
        return 1;
    }
    return process(capture);
}
