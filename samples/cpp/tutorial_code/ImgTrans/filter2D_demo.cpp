/**
 * @file filter2D_demo.cpp
 * @brief Sample code that shows how to implement your own linear filters by using filter2D function
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

/**
 * @function main
 */
int main ( int argc, char** argv )
{
  /// Declare variables
  Mat src, dst;

  Mat kernel;
  Point anchor;
  double delta;
  int ddepth;
  int kernel_size;
  const char* window_name = "filter2D Demo";

  //![load]
  String imageName("../data/lena.jpg"); // by default
  if (argc > 1)
  {
    imageName = argv[1];
  }
  src = imread( imageName, IMREAD_COLOR ); // Load an image

  if( src.empty() )
    { return -1; }
  //![load]

  //![init_arguments]
  /// Initialize arguments for the filter
  anchor = Point( -1, -1 );
  delta = 0;
  ddepth = -1;
  //![init_arguments]

  /// Loop - Will filter the image with different kernel sizes each 0.5 seconds
  int ind = 0;
  for(;;)
       {
         char c = (char)waitKey(500);
         /// Press 'ESC' to exit the program
         if( c == 27 )
           { break; }

         //![update_kernel]
         /// Update kernel size for a normalized box filter
         kernel_size = 3 + 2*( ind%5 );
         kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
         //![update_kernel]

         //![apply_filter]
         filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
         //![apply_filter]
         imshow( window_name, dst );
         ind++;
       }

  return 0;
}
