/**
 * @file LinearTransforms.cpp
 * @brief Simple program to change contrast and brightness
 * @date Mon, June 6, 2011
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

/** Global Variables */
const int alpha_max = 5;
const int beta_max = 125;
int alpha; /**< Simple contrast control */
int beta;  /**< Simple brightness control*/

/** Matrices to store images */
Mat image;

/**
 * @function on_trackbar
 * @brief Called whenever any of alpha or beta changes
 */
static void on_trackbar( int, void* )
{
    Mat new_image = Mat::zeros( image.size(), image.type() );

    for( int y = 0; y < image.rows; y++ )
        for( int x = 0; x < image.cols; x++ )
            for( int c = 0; c < 3; c++ )
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );

    imshow("New Image", new_image);
}


/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
   /// Read image given by user
   String imageName("../data/lena.jpg"); // by default
   if (argc > 1)
   {
      imageName = argv[1];
   }
   image = imread( imageName );

   /// Initialize values
   alpha = 1;
   beta = 0;

   /// Create Windows
   namedWindow("Original Image", 1);
   namedWindow("New Image", 1);

   /// Create Trackbars
   createTrackbar( "Contrast", "New Image", &alpha, alpha_max, on_trackbar );
   createTrackbar( "Brightness", "New Image", &beta, beta_max, on_trackbar );

   /// Show some stuff
   imshow("Original Image", image);
   imshow("New Image", image);

   /// Wait until user press some key
   waitKey();
   return 0;
}
