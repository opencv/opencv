/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to use the function calcHist
 * @author
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/**
 * @function main
 */
int main( int argc, char** argv )
{
  Mat src, dst;

  /// Load image
  src = imread( argv[1], 1 );

  if( !src.data )
    { return -1; }

  /// Separate the image in 3 places ( R, G and B )
  vector<Mat> rgb_planes;
  split( src, rgb_planes ); 

  /// Establish the number of bins 
  int histSize = 255;

  /// Set the ranges ( for R,G,B) )
  float range[] = { 0, 255 } ; 
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat r_hist, g_hist, b_hist;

  /// Compute the histograms:
  calcHist( &rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for R, G and B
  int hist_w = 400; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_w, hist_h, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel 
  for( int i = 1; i < histSize; i++ )
    { 
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) , 
		       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ), 
	               Scalar( 0, 0, 255), 2, 8, 0  ); 
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) , 
		       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ), 
	               Scalar( 0, 255, 0), 2, 8, 0  ); 
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) , 
		       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ), 
	               Scalar( 255, 0, 0), 2, 8, 0  );       
    }

 /// Display 
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );

  waitKey(0);

  return 0;
 
}
