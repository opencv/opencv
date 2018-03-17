/**
 * @file LinearBlend.cpp
 * @brief Simple linear blender ( dst = alpha*src1 + beta*src2 )
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
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

//![on_trackbar]
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
//![on_trackbar]

/**
 * @function main
 * @brief Main function
 */
int main( void )
{
   //![load]
   /// Read images ( both have to be of the same size and type )
   src1 = imread("../data/LinuxLogo.jpg");
   src2 = imread("../data/WindowsLogo.jpg");
   //![load]

   if( src1.empty() ) { printf("Error loading src1 \n"); return -1; }
   if( src2.empty() ) { printf("Error loading src2 \n"); return -1; }

   /// Initialize values
   alpha_slider = 0;

   //![window]
   namedWindow("Linear Blend", WINDOW_AUTOSIZE); // Create Window
   //![window]

   //![create_trackbar]
   char TrackbarName[50];
   sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );
   createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );
   //![create_trackbar]

   /// Show some stuff
   on_trackbar( alpha_slider, 0 );

   /// Wait until user press some key
   waitKey(0);
   return 0;
}
