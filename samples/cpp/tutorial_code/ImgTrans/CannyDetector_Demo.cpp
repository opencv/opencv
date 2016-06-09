/**
 * @file CannyDetector_Demo.cpp
 * @brief Sample code showing how to detect edges using the Canny Detector
 * @author OpenCV team
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThresholdSobel;
int lowThresholdScharr;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name1 = "Edge Map sobel";
const char* window_name2 = "Edge Map Scharr";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
static void CannyThreshold(int, void*)
{
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );

    /// Canny detector with scharr
    Mat dx,dy;

    Scharr(detected_edges,dx,CV_16S,1,0);
    Scharr(detected_edges,dy,CV_16S,0,1);
    Canny( dx,dy, detected_edges, lowThresholdScharr, lowThresholdScharr*ratio );
    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);
    imshow(window_name2, dst);
    blur(src_gray, detected_edges, Size(3, 3));
    Canny( detected_edges, detected_edges, lowThresholdSobel, lowThresholdSobel*ratio, kernel_size );
    dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);
    imshow(window_name1, dst);

}


/**
 * @function main
 */
int main( int, char** argv )
{
  /// Load an image
  src = imread( argv[1] );

  if( src.empty() )
    { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  /// Create a window
  namedWindow( window_name1, WINDOW_AUTOSIZE );
  namedWindow( window_name2, WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold Sobel:", window_name1, &lowThresholdSobel, max_lowThreshold, CannyThreshold );
  createTrackbar( "Min Threshold Scharr:", window_name2, &lowThresholdScharr, max_lowThreshold*4, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
}
