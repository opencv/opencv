/**
 * @function cornerHarris_Demo.cpp
 * @brief Demo code for detecting corners using Harris-Stephens method
 * @author OpenCV team
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo( int, void* );

/**
 * @function main
 */
int main( int argc, char** argv )
{
    /// Load source image and convert it to gray
    CommandLineParser parser( argc, argv, "{@input | building.jpg | input image}" );
    src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    /// Create a window and a trackbar
    namedWindow( source_window );
    createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );
    imshow( source_window, src );

    cornerHarris_demo( 0, 0 );

    waitKey();
    return 0;
}

/**
 * @function cornerHarris_demo
 * @brief Executes the corner detection and draw a circle around the possible corners
 */
void cornerHarris_demo( int, void* )
{
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    Mat dst = Mat::zeros( src.size(), CV_32FC1 );
    cornerHarris( src_gray, dst, blockSize, apertureSize, k );

    /// Normalizing
    Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    /// Drawing a circle around corners
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }

    /// Showing the result
    namedWindow( corners_window );
    imshow( corners_window, dst_norm_scaled );
}
