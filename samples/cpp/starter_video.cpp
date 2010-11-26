/*
 * starter_video.cpp
 *
 *  Created on: Nov 23, 2010
 *      Author: Ethan Rublee
 *
 * A starter sample for using opencv, get a video stream and display the images
 * easy as CV_PI right?
 */
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace {
void help(char** av) {
	cout << "usage:" << av[0] << " <video device number>\n"
			<< "\tThis is a starter sample, to get you up and going in a copy pasta fashion\n"
			<< "\tThe program captures frames from a camera connected to your computer.\n"
			<< "\tTo find the video device number, try ls /dev/video* \n"
			<< "\tYou may also pass a video file, like my_vide.avi instead of a device number"
			<< endl;
}

int process(VideoCapture& capture) {
	string window_name = "video | q or esc to quit";
	cout << "press q or esc to quit" << endl;
	namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window;
	Mat frame;
	for (;;) {
		capture >> frame;
		if (frame.empty())
			continue;
		imshow(window_name, frame);
		char key = (char)waitKey(5); //delay N millis, usually long enough to display and capture input
		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			return 0;
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
	VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file
	if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
		capture.open(atoi(arg.c_str()));
	if (!capture.isOpened()) {
		cerr << "Failed to open a video device or video file!" << endl;
		return 1;
	}
	return process(capture);
}
