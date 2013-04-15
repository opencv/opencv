/*
* starter_image_sequence.cpp
*
*  Created on: July 23, 2012
*      Author: Kevin Hughes
*
* A simple example of how to use the built in functionality of cv::VideoCapture to handle
* sequences of images. Image sequences are a common way to distribute data sets for various
* computer vision problems, for example the change detection data set from CVPR 2012
* http://www.changedetection.net/
*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void help(char** argv) 
{
    cout << "\nThis program gets you started reading a sequence of images using cv::VideoCapture.\n"
         << "Image sequences are a common way to distribute video data sets for computer vision.\n"
         << "Usage: " << argv[0] << " <path to the first image in the sequence>\n"
         << "example: " << argv[0] << " right%%02d.jpg\n"
         << "q,Q,esc -- quit\n"
         << "\tThis is a starter sample, to get you up and going in a copy pasta fashion\n"
         << endl;
}

int main(int argc, char** argv)
{
    if(argc != 2) 
    {
        help(argv);
        return 1;
    }

    string arg = argv[1];
    VideoCapture sequence(arg);
    if (!sequence.isOpened())
    {
        cerr << "Failed to open Image Sequence!\n" << endl;
        return 1;
    }
    
    Mat image;
    namedWindow("Image | q or esc to quit", CV_WINDOW_NORMAL);
    
    for(;;)
    {
        sequence >> image;
        if(image.empty())
        {
            cout << "End of Sequence" << endl;
            break;
        }
        
        imshow("image | q or esc to quit", image);

        char key = (char)waitKey(500);
        if(key == 'q' || key == 'Q' || key == 27)
            break;
    }

    return 0;
}
