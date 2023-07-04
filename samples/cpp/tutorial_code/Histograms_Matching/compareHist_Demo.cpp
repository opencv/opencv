/**
 * @file compareHist_Demo.cpp
 * @brief Sample code to use the function compareHist
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

const char* keys =
    "{ help  h| | Print help message. }"
    "{ @input1 |Histogram_Comparison_Source_0.jpg | Path to input image 1. }"
    "{ @input2 |Histogram_Comparison_Source_1.jpg | Path to input image 2. }"
    "{ @input3 |Histogram_Comparison_Source_2.jpg | Path to input image 3. }";

/**
 * @function main
 */
int main( int argc, char** argv )
{
    //! [Load three images with different environment settings]
    CommandLineParser parser( argc, argv, keys );
    samples::addSamplesDataSearchSubDirectory( "doc/tutorials/imgproc/histograms/histogram_comparison/images" );
    Mat src_base = imread(samples::findFile( parser.get<String>( "@input1" ) ) );
    Mat src_test1 = imread(samples::findFile( parser.get<String>( "@input2" ) ) );
    Mat src_test2 = imread(samples::findFile( parser.get<String>( "@input3" ) ) );
    if( src_base.empty() || src_test1.empty() || src_test2.empty() )
    {
        cout << "Could not open or find the images!\n" << endl;
        parser.printMessage();
        return -1;
    }
    //! [Load three images with different environment settings]

    //! [Convert to HSV]
    Mat hsv_base, hsv_test1, hsv_test2;
    cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
    cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );
    cvtColor( src_test2, hsv_test2, COLOR_BGR2HSV );
    //! [Convert to HSV]

    //! [Convert to HSV half]
    Mat hsv_half_down = hsv_base( Range( hsv_base.rows/2, hsv_base.rows ), Range( 0, hsv_base.cols ) );
    //! [Convert to HSV half]

    //! [Using 50 bins for hue and 60 for saturation]
    int h_bins = 50, s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };
    //! [Using 50 bins for hue and 60 for saturation]

    //! [Calculate the histograms for the HSV images]
    Mat hist_base, hist_half_down, hist_test1, hist_test2;

    calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
    normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false );
    normalize( hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
    normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false );
    normalize( hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat() );
    //! [Calculate the histograms for the HSV images]

    //! [Apply the histogram comparison methods]
    for( int compare_method = 0; compare_method < 4; compare_method++ )
    {
        double base_base = compareHist( hist_base, hist_base, compare_method );
        double base_half = compareHist( hist_base, hist_half_down, compare_method );
        double base_test1 = compareHist( hist_base, hist_test1, compare_method );
        double base_test2 = compareHist( hist_base, hist_test2, compare_method );

        cout << "Method " << compare_method << " Perfect, Base-Half, Base-Test(1), Base-Test(2) : "
             <<  base_base << " / " << base_half << " / " << base_test1 << " / " << base_test2 << endl;
    }
    //! [Apply the histogram comparison methods]

    cout << "Done \n";
    return 0;
}
