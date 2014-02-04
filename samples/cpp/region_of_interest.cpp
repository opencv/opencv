#include <iostream>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


void help(void);
Mat getROI(Mat &, int, int, int, int);

int main()
{
	help();

	Mat img, roi;
	VideoCapture cap(0);

	int len=30, wid=40, x=30, y=30;
	char ch;

	if(!cap.isOpened())
	{
		help();
		cout<<"\n\nCannot Open Camera....\n";
	}

	namedWindow("Main", WINDOW_NORMAL);
	namedWindow("Region of interest", WINDOW_AUTOSIZE);

	createTrackbar("Length", "Main", &len, 300, NULL);
	createTrackbar("Width", "Main", &wid, 300, NULL);
	createTrackbar("X", "Main", &x, 300, NULL);
	createTrackbar("Y", "Main", &y, 300, NULL);

	bool gray = true;
	Mat grayroi;
	while(true)
	{
		cap>>img;
		roi = getROI(img, len, wid, x, y);

		if(gray == true)
		{
			cvtColor(roi, grayroi, COLOR_BGR2GRAY);
			imshow("Region of interest", grayroi);
		}
		else if(gray == false)
			imshow("Region of interest", roi);

		imshow("Main", img);
		ch = waitKey(1);
		if(ch == 'g' || ch == 'G')
		{
			if(gray == true)
				gray = false;
			else if(gray == false)
				gray = true;
		}
		else if(ch==27)
			break;
	}
	return 0;
}

void help(void)
{
	cout<<"This program demonstrates how to set the Region of Interest in an image from a live-feed in OpenCV.\n\n";
	cout<<"1.\t Usage ./roi (Uses Default Webcam). \n";
	cout<<"2.\t ESC --> Quits the program. \n";
	cout<<"3.\t 'g' --> Switch between grayscale and RGB outputs.\n";
	cout<<"4.\t Adjust trackbars to position and resize the rectangle.\n\n\n";
}

Mat getROI(Mat &img, int l, int w, int x, int y)
{
	rectangle(img, Point(x,y), Point(x+l, y+w), Scalar(0), 2, 8, 0);
	Rect ROI = Rect(x, y, l, w);
	Mat roi = img(ROI);

	return roi;
}
