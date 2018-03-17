/**
 * @function pointPolygonTest_demo.cpp
 * @brief Demo code to use the pointPolygonTest function...fairly easy
 * @author OpenCV team
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function main
 */
int main( void )
{
  /// Create an image
  const int r = 100;
  Mat src = Mat::zeros( Size( 4*r, 4*r ), CV_8UC1 );

  /// Create a sequence of points to make a contour:
  vector<Point2f> vert(6);

  vert[0] = Point( 3*r/2, static_cast<int>(1.34*r) );
  vert[1] = Point( 1*r, 2*r );
  vert[2] = Point( 3*r/2, static_cast<int>(2.866*r) );
  vert[3] = Point( 5*r/2, static_cast<int>(2.866*r) );
  vert[4] = Point( 3*r, 2*r );
  vert[5] = Point( 5*r/2, static_cast<int>(1.34*r) );

  /// Draw it in src
  for( int j = 0; j < 6; j++ )
     { line( src, vert[j],  vert[(j+1)%6], Scalar( 255 ), 3, 8 ); }

  /// Get the contours
  vector<vector<Point> > contours;

  findContours( src, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

  /// Calculate the distances to the contour
  Mat raw_dist( src.size(), CV_32FC1 );

  for( int j = 0; j < src.rows; j++ )
     { for( int i = 0; i < src.cols; i++ )
          { raw_dist.at<float>(j,i) = (float)pointPolygonTest( contours[0], Point2f((float)i,(float)j), true ); }
     }

  double minVal; double maxVal;
  minMaxLoc( raw_dist, &minVal, &maxVal, 0, 0, Mat() );
  minVal = abs(minVal); maxVal = abs(maxVal);

  /// Depicting the  distances graphically
  Mat drawing = Mat::zeros( src.size(), CV_8UC3 );

  for( int j = 0; j < src.rows; j++ )
     { for( int i = 0; i < src.cols; i++ )
          {
            if( raw_dist.at<float>(j,i) < 0 )
              { drawing.at<Vec3b>(j,i)[0] = (uchar)(255 - abs(raw_dist.at<float>(j,i))*255/minVal); }
            else if( raw_dist.at<float>(j,i) > 0 )
              { drawing.at<Vec3b>(j,i)[2] = (uchar)(255 - raw_dist.at<float>(j,i)*255/maxVal); }
            else
              { drawing.at<Vec3b>(j,i)[0] = 255; drawing.at<Vec3b>(j,i)[1] = 255; drawing.at<Vec3b>(j,i)[2] = 255; }
          }
     }

  /// Show your results
  imshow( "Source", src );
  imshow( "Distance", drawing );

  waitKey(0);
  return(0);
}
