/**
 * @file BackProject_Demo2.cpp
 * @brief Sample code for backproject function usage ( a bit more elaborated )
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

/// Global Variables
Mat src, hsv, mask;

int low = 20, up = 20;
const char* window_image = "Source image";

/// Function Headers
void Hist_and_Backproj( );
void pickPoint (int event, int x, int y, int, void* );

/**
 * @function main
 */
int main( int, char** argv )
{
    /// Read the image
    src = imread( argv[1] );

    /// Transform it to HSV
    cvtColor( src, hsv, COLOR_BGR2HSV );

    /// Show the image
    namedWindow( window_image );
    imshow( window_image, src );

    /// Set Trackbars for floodfill thresholds
    createTrackbar( "Low thresh", window_image, &low, 255, 0 );
    createTrackbar( "High thresh", window_image, &up, 255, 0 );
    /// Set a Mouse Callback
    setMouseCallback( window_image, pickPoint, 0 );

    waitKey();
    return 0;
}

/**
 * @function pickPoint
 */
void pickPoint (int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
    {
        return;
    }

    // Fill and get the mask
    Point seed = Point( x, y );

    int newMaskVal = 255;
    Scalar newVal = Scalar( 120, 120, 120 );

    int connectivity = 8;
    int flags = connectivity + (newMaskVal << 8 ) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

    Mat mask2 = Mat::zeros( src.rows + 2, src.cols + 2, CV_8U );
    floodFill( src, mask2, seed, newVal, 0, Scalar( low, low, low ), Scalar( up, up, up), flags );
    mask = mask2( Range( 1, mask2.rows - 1 ), Range( 1, mask2.cols - 1 ) );

    imshow( "Mask", mask );

    Hist_and_Backproj( );
}

/**
 * @function Hist_and_Backproj
 */
void Hist_and_Backproj( )
{
    Mat hist;
    int h_bins = 30; int s_bins = 32;
    int histSize[] = { h_bins, s_bins };

    float h_range[] = { 0, 180 };
    float s_range[] = { 0, 256 };
    const float* ranges[] = { h_range, s_range };

    int channels[] = { 0, 1 };

    /// Get the Histogram and normalize it
    calcHist( &hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false );

    normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

    /// Get Backprojection
    Mat backproj;
    calcBackProject( &hsv, 1, channels, hist, backproj, ranges, 1, true );

    /// Draw the backproj
    imshow( "BackProj", backproj );
}
