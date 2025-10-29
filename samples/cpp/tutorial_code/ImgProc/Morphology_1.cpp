/**
 * @file Morphology_1.cpp
 * @brief Erosion and Dilation sample code
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/// Global variables
Mat src, erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 3;
int const max_kernel_size = 21;

/** Function Headers */
void Erosion( int, void* );
void Dilation( int, void* );

//![main]
/**
 * @function main
 */
int main( int argc, char** argv )
{
  /// Load an image
  CommandLineParser parser( argc, argv, "{@input | LinuxLogo.jpg | input image}" );
  src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
  if( src.empty() )
  {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  /// Create windows
  namedWindow( "Erosion Demo", WINDOW_AUTOSIZE );
  namedWindow( "Dilation Demo", WINDOW_AUTOSIZE );
  moveWindow( "Dilation Demo", src.cols, 0 );

  /// Create Erosion Trackbar
  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse \n 3: Diamond", "Erosion Demo",
          &erosion_elem, max_elem,
          Erosion );

  createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
          &erosion_size, max_kernel_size,
          Erosion );

  /// Create Dilation Trackbar
  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse \n 3: Diamond", "Dilation Demo",
          &dilation_elem, max_elem,
          Dilation );

  createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
          &dilation_size, max_kernel_size,
          Dilation );

  /// Default start
  Erosion( 0, 0 );
  Dilation( 0, 0 );

  waitKey(0);
  return 0;
}
//![main]

//![erosion]
/**
 * @function Erosion
 */
void Erosion( int, void* )
{
  int erosion_type = 0;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  else if( erosion_elem == 3) { erosion_type = MORPH_DIAMOND; }

  //![kernel]
  Mat element = getStructuringElement( erosion_type,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
  //![kernel]

  /// Apply the erosion operation
  erode( src, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
}
//![erosion]

//![dilation]
/**
 * @function Dilation
 */
void Dilation( int, void* )
{
  int dilation_type = 0;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
  else if( dilation_elem == 3) { dilation_type = MORPH_DIAMOND; }

  Mat element = getStructuringElement( dilation_type,
                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       Point( dilation_size, dilation_size ) );

  /// Apply the dilation operation
  dilate( src, dilation_dst, element );
  imshow( "Dilation Demo", dilation_dst );
}
//![dilation]
