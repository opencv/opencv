//
// example 8-2
// Contours example using trackbar to threshold
//
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

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat g_gray, g_binary;
int g_thresh = 100;

void on_trackbar(int, void*) {
  threshold( g_gray, g_binary, g_thresh, 255, CV_THRESH_BINARY );
  vector<vector<Point> > contours;
  findContours(g_binary, contours, noArray(), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  g_binary = Scalar::all(0);
    
  drawContours(g_binary, contours, -1, Scalar::all(255));
  imshow( "Contours", g_binary );
}

int main( int argc, char** argv )
{
  if( argc != 2 || (g_gray = imread(argv[1], 0)).empty() )
  {
	  cout << "Find threshold dependent contours\nUsage: ./ch8_ex8_2 fruits.jpg" << endl;
      return -1;
  }
  namedWindow( "Contours", 1 );
  createTrackbar( 
    "Threshold", 
    "Contours", 
    &g_thresh, 
    255, 
    on_trackbar
  );
  on_trackbar(0, 0);
  waitKey();
  return 0; 
}
