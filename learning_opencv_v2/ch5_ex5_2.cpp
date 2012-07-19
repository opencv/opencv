//
// Example 5-2. Example code making use of addWeighted and cv::threshold()
//
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
//#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void sum_rgb( const Mat& src, Mat& dst ) {
  
  // Split image onto the color planes.	
  vector<Mat> planes;	
  split(src, planes);
  
  Mat b = planes[0], g = planes[1], r = planes[2], s;
  // Add equally weighted rgb values.
  addWeighted( r, 1./3., g, 1./3., 0.0, s );
  addWeighted( s, 2./3., b, 1./3., 0.0, s );
  // Truncate values above 100.
  threshold( s, dst, 100, 100, CV_THRESH_TRUNC );
}

void help()
{
	cout << "Call: ./ch5_ex5_2 faceScene.jpg" << endl;
	cout << "Shows use of alpha blending (addWeighted) and threshold" << endl;
}

int main(int argc, char** argv)
{
	help();
	if(argc < 2) { cout << "specify input image" << endl; return -1; }

	// Load the image from the given file name.
	Mat src = imread( argv[1] ), dst;
	if( src.empty() ) { cout << "can not load " << argv[1] << endl; return -1; }
	sum_rgb( src, dst);
	// Create a named window with a the name of the file and
	// show the image in the window
	imshow( argv[1], dst );

	// Idle until the user hits any key.
	waitKey(0);

	return 0;
}
