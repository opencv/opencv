/**
 * @file HoughCircle_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/// Global variables
Mat src, src_gray;
Mat result;
int min_threshold = 30;
int max_threshold = 100;
int slider = max_threshold;
const char* title = "Hough Circle Transform Demo";

/// Function Headers
void help();
void Gradient_Hough( int, void* );

/**
 * @function main
 */
int main(int, char** argv)
{
    /// Read the image
    src = imread( argv[1], 1 );

    if( !src.data )
    {
        help();
        return -1;
    }

    /// Convert it to gray
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    /// Reduce the noise so we avoid false circle detection
    GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

    /// Create Trackbar
    namedWindow( title, WINDOW_AUTOSIZE );
    createTrackbar( "Thres:", title, &slider, max_threshold, Gradient_Hough );

    /// Initialize
    Gradient_Hough( 0, 0 );

    waitKey(0);
    return 0;
}

/**
 * @function help
 */
void help()
{
    printf("\t Hough Transform to detect circles \n ");
    printf("\t---------------------------------\n ");
    printf(" Usage: ./HoughCircle_Demo <image_name> \n");
}

/**
 * @function Gradient_Hough
 */
void Gradient_Hough( int, void* )
{
    vector<Vec3f> circles;
    cvtColor( src_gray, result, COLOR_GRAY2BGR );

    /// Accumulator Threshold must be positive
    if ( slider < 1 ) {
      printf("Threshold must be positive\n");
      return;
    }

    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows/8, 200, slider, 0, 0 );

    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // circle center
         circle( result, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // circle outline
         circle( result, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }

    imshow( title, result );
}
