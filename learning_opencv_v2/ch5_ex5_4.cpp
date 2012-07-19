//      ./ch5_ex5_4 100 1 0 15 10 fruits.jpg
// Example 5-4. Threshold versus adaptive threshold
// Compare thresholding with adaptive thresholding
// CALL:
// ./adaptThreshold Threshold 1binary 1adaptivemean \
//                    blocksize offset filename
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

int main( int argc, char** argv )
{
  if(argc != 7) { cout <<
   "Usage: ch5_ex5_4 fixed_threshold invert(0=off|1=on) adaptive_type(0=mean|1=gaussian) block_size offset image\n"
   "Example: ch5_ex5_4 100 1 0 15 10 fruits.jpg\n"; return -1; }
  //Command line
  double fixed_threshold = (double)atof(argv[1]);
  int threshold_type = atoi(argv[2]) ?
          CV_THRESH_BINARY : CV_THRESH_BINARY_INV;
  int adaptive_method = atoi(argv[3]) ?
          CV_ADAPTIVE_THRESH_MEAN_C : CV_ADAPTIVE_THRESH_GAUSSIAN_C;
  int block_size = atoi(argv[4]);
  double offset = (double)atof(argv[5]);
  Mat Igray = imread(argv[6], CV_LOAD_IMAGE_GRAYSCALE);
  //Read in gray image
  if( Igray.empty() ){ cout << "Can not load " << argv[6] << endl; return -1; }
  // Declare the output images
  Mat It, Iat;
  //Threshold
  threshold(Igray,It,fixed_threshold,255,threshold_type);
  adaptiveThreshold(Igray, Iat, 255, adaptive_method,
                    threshold_type, block_size, offset);
  //Show the results
  imshow("Raw",Igray);
  imshow("Threshold",It);
  imshow("Adaptive Threshold",Iat);
  waitKey(0);
  return 0;
}

