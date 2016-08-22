/**
 * @file AddingImages.cpp
 * @brief Simple linear blender ( dst = alpha*src1 + beta*src2 )
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

/**
 * @function main
 * @brief Main function
 */
int main( void )
{
   double alpha = 0.5; double beta; double input;

   Mat src1, src2, dst;

   /// Ask the user enter alpha
   std::cout<<" Simple Linear Blender "<<std::endl;
   std::cout<<"-----------------------"<<std::endl;
   std::cout<<"* Enter alpha [0-1]: ";
   std::cin>>input;

   // We use the alpha provided by the user if it is between 0 and 1
   if( alpha >= 0 && alpha <= 1 )
     { alpha = input; }

   //![load]
   /// Read images ( both have to be of the same size and type )
   src1 = imread("../data/LinuxLogo.jpg");
   src2 = imread("../data/WindowsLogo.jpg");
   //![load]

   if( src1.empty() ) { std::cout<< "Error loading src1"<<std::endl; return -1; }
   if( src2.empty() ) { std::cout<< "Error loading src2"<<std::endl; return -1; }

   //![blend_images]
   beta = ( 1.0 - alpha );
   addWeighted( src1, alpha, src2, beta, 0.0, dst);
   //![blend_images]

   //![display]
   imshow( "Linear Blend", dst );
   waitKey(0);
   //![display]

   return 0;
}
