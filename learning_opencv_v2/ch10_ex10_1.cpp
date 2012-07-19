// Example 10-1. Pyramid Lucas-Kanade optical flow code
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

void help()
{
	cout << "Call: ./ch10_ex10_1" << endl;
	cout << "Demonstrates Pyramid Lucas-Kanade optical flow." << endl;
	cout << "Must have OpticalFlow0.jpg and OpticalFlow1.jpg in this directory" << endl;
}
int main(int argc, char** argv) {
   // Initialize, load two images from the file system, and
   // allocate the images and other structures we will need for
   // results.
	//
	Mat imgA = imread("OpticalFlow0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgB = imread("OpticalFlow1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Size img_sz = imgA.size();
	int win_size = 10;
	Mat imgC = imread("OpticalFlow1.jpg",CV_LOAD_IMAGE_UNCHANGED);
	
	// The first thing we need to do is get the features
	// we want to track.
	//
    vector<Point2f> cornersA, cornersB;
    const int MAX_CORNERS = 500;
	goodFeaturesToTrack(imgA, cornersA, MAX_CORNERS, 0.01, 5, noArray(), 3, false, 0.04);
    
	cornerSubPix(imgA, cornersA, Size(win_size, win_size), Size(-1,-1),
                 TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));

	// Call the Lucas Kanade algorithm
	//
    vector<uchar> features_found;
	calcOpticalFlowPyrLK(imgA, imgB, cornersA, cornersB, features_found, noArray(),
                         Size(win_size*4+1,win_size*4+1), 5,
                         TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ));
    // Now make some image of what we are looking at:
    //
    for( int i = 0; i < (int)cornersA.size(); i++ ) {
     if( !features_found[i] )
         continue;
     line(imgC, cornersA[i], cornersB[i], Scalar(0,255,0), 2, CV_AA);   
  }
  imshow("ImageA",imgA);
  imshow("ImageB",imgB);
  imshow("LKpyr_OpticalFlow",imgC);
  waitKey(0);
  return 0;
}

