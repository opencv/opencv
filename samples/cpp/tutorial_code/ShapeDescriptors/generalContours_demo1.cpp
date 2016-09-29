/**
 * @function generalContours_demo1.cpp
 * @brief Demo code to find contours in an image
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );

/**
 * @function main
 */
int main( int, char** argv )
{
  //![setup]
  /// Load source image
  src = imread( argv[1], IMREAD_COLOR );

  /// Convert image to gray and blur it
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );
  //![setup]

  //![createWindow]
  /// Create Window
  const char* source_window = "Source";
  namedWindow( source_window, WINDOW_AUTOSIZE );
  imshow( source_window, src );
  //![createWindow]

  //![taskbar]
  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  //![taskbar]

  //![callback00]
  thresh_callback( 0, 0 );
  //![callback00]

  //![waitForIt]
  waitKey(0);
  //![waitForIt]

  return(0);
}

/**
 * @function thresh_callback
 */
void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  //![threshold]
  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  //![threshold]

  //![findContours]
  /// Find contours
  findContours( threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  //![findContours]

  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );

  //![allthework]
  for( size_t i = 0; i < contours.size(); i++ )
  {
    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    minEnclosingCircle( contours_poly[i], center[i], radius[i] );
  }
  //![allthework]

  //![zeroMat]
  /// Draw polygonal contour + bonding rects + circles
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  //![zeroMat]

  //![forContour]
  for( size_t i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
    rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
    circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
  }
  //![forContour]

  //![showDrawings]
  /// Show in a window
  namedWindow( "Contours", WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
  //![showDrawings]
}
