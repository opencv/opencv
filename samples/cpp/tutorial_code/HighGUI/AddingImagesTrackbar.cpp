/**
 * @file LinearBlend.cpp
 * @brief Simple linear blender ( dst = alpha*src1 + beta*src2 )
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace cv;

/** Global Variables */
const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

/** Matrices to store images */
Mat src1;
Mat src2;
Mat dst;

/**
 * @function on_trackbar
 * @brief Callback for trackbar
 */
static void on_trackbar( int, void* )
{
   alpha = (double) alpha_slider/alpha_slider_max ;

   beta = ( 1.0 - alpha );

   addWeighted( src1, alpha, src2, beta, 0.0, dst);

   imshow( "Linear Blend", dst );
}


/**
 * @function main
 * @brief Main function
 */
int main( void )
{
   /// Read image ( same size, same type )
   src1 = imread("../images/LinuxLogo.jpg");
   src2 = imread("../images/WindowsLogo.jpg");

   if( src1.empty() ) { printf("Error loading src1 \n"); return -1; }
   if( src2.empty() ) { printf("Error loading src2 \n"); return -1; }

   /// Initialize values
   alpha_slider = 0;

   /// Create Windows
   namedWindow("Linear Blend", 1);

   /// Create Trackbars
   char TrackbarName[50];
   sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );
   createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );

   /// Show some stuff
   on_trackbar( alpha_slider, 0 );

   /// Wait until user press some key
   waitKey(0);
   return 0;
}
