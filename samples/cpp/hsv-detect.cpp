/*
  @file hsv-detect.cpp
  @author Alessandro de Oliveira Faria (A.K.A. CABELO)
  @brief Example of how to identify object with the HSV color space. With the example in /samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp, we can use the values lowH, lowS, lowV, highH, highS, highV in the inRange function. See the example in this video: https://www.youtube.com/watch?v=oPOv4P1EqTI . Questions and suggestions email to: Alessandro de Oliveira Faria cabelo[at]opensuse[dot]org or OpenCV Team.
  @date Aug 20, 2020
*/
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ obj1Lower   |41 94 0|  HSV low value of the object1. }"
    "{ obj1Upper   |91 255 255| HSV high value of the object1. }"
    "{ obj2Lower   |134 116 82| HSV low value of the object2. }"
    "{ obj2Upper   |179 255 255| HSV high value of the object2. }"
    "{ minArea     | 1000 | Minimum area value. }"
    "{ label1      | Apple | Name of object1. }"
    "{ label2      | Lemon | Name of object2. }";

using namespace cv;
using namespace std;

Scalar calcLargestArea(Mat mask, vector<vector<Point>> &contours)
{
	int largest_contour_index = 0;
	int largest_area = 0;
	double a;
	dilate(mask,mask, Mat(), Point(-1,-1),1);
	erode(mask,mask, Mat(), Point(-1,-1), 3);
	findContours(mask, contours, RETR_LIST, CHAIN_APPROX_NONE);
   for( int i = 0; i< contours.size(); i++ )
   {
		a=contourArea( contours[i],false); 
		if(a>largest_area)
		{
 			largest_area=a;
			largest_contour_index=i;               
		}
	}
	return Scalar(largest_contour_index, largest_area,countNonZero(mask));
}

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);
	parser = CommandLineParser(argc, argv, keys);
	parser.about("Use this example o run object detection with HSV using OpenCV.");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String labels[] = { parser.get<String>("label1"), parser.get<String>("label2") };

	Scalar obj1Lower = parser.get<Scalar>("obj1Lower");
	Scalar obj1Upper = parser.get<Scalar>("obj1Upper");

	Scalar obj2Lower = parser.get<Scalar>("obj2Lower");
	Scalar obj2Upper = parser.get<Scalar>("obj2Upper");
	int minArea      = parser.get<int>("minArea");

	bool testObj; 
	Mat frame,image,OutputImageMask1,OutputImageMask2;
	vector<vector<Point>>  contours;
	float _x,_y,radius;
	Scalar result;
	Point2f center,_center;

	VideoCapture cap;
	cap.open(0);
	while(true)
	{
		cap>>frame;
		cvtColor(frame, image,  COLOR_BGR2HSV);

		inRange(image,obj1Lower,obj1Upper,OutputImageMask1);
		inRange(image,obj2Lower,obj2Upper,OutputImageMask2);

		testObj = (countNonZero(OutputImageMask1)> countNonZero(OutputImageMask2));
		result = calcLargestArea((testObj == 1?OutputImageMask1:OutputImageMask2),contours);
	
		if(result[1]>minArea)
		{      
			minEnclosingCircle( contours[result[0]], center, radius);
			cv::Moments m=cv::moments(contours[result[0]]);
			_x = (m.m10/m.m00);
			_y = (m.m01/m.m00);
			_center = Point2f(_x,_y);
			if(radius>10)
			{
				circle(frame, center, cvRound(radius), Scalar(0, 255, 255), 2, LINE_AA);
				circle(frame, _center, 5, Scalar(255,0,0), -1, LINE_AA);
			}
			imshow("image",frame);
			cv::displayOverlay("image",labels[testObj],2000); 
		}
		else
		imshow("image",frame);
		if (waitKey(5) >= 0) break;
	}
	cap.release();
}
