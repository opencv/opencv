/* License:
   July 20, 2011
   Standard BSD

   BOOK: It would be nice if you cited it:
   Learning OpenCV 2: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   Main OpenCV site
   http://opencv.willowgarage.com/wiki/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;


int main( int argc, char** argv )
{
	cv::Mat img = imread(argv[1],-1);
	if(img.empty())
	{
		std::cerr << "Couldn't open the image " << argv[1] << std::endl;
		std::cerr << "Call: ./ch2_ex2_1 faceScene.jpg" << std::endl;
		return -1;
	}
	cv::namedWindow("Exmple1", CV_WINDOW_AUTOSIZE );
	cv::imshow("Example1", img );
	cv::waitKey(0);
	cv::destroyWindow("Example1");
}
