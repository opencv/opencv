//
// example 6-1 Hough circles
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
//
/*
You'll have to tune to detect the circles you expect
*/



#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  if(argc != 2) { cout << "Hough Circle detect\nUsage: ch6_ex6_1 <imagename>\n" << endl; return -1; }
    
  Mat src = imread(argv[1], 1), image;
  if( src.empty() ) { cout << "Can not load " << argv[1] << endl; return -1; }
  cvtColor(src, image, CV_BGR2GRAY);  
  
  GaussianBlur(image, image, Size(5,5), 0, 0);
  
  vector<Vec3f> circles;
  HoughCircles(image, circles, CV_HOUGH_GRADIENT, 2, image.cols/10);
    
  for( size_t i = 0; i < circles.size(); i++ ) {
    circle(src, Point(cvRound(circles[i][0]), cvRound(circles[i][1])),
           cvRound(circles[i][2]), Scalar(0,0,255), 2, CV_AA);
  }
  imshow( "Hough Circles", src);
  waitKey(0);
  return 0;  
}

