/**
 * @file BasicLinearTransforms.cpp
 * @brief Simple program to change contrast and brightness
 * @author OpenCV team
 */

#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace cv;

double alpha; /**< Simple contrast control */
int beta;  /**< Simple brightness control */

/**
 * @function main
 * @brief Main function 
 */
int main( int argc, char** argv )
{
   /// Read image given by user
   Mat image = imread( argv[1] );
   Mat new_image = Mat::zeros( image.size(), image.type() );

   /// Initialize values 
   std::cout<<" Basic Linear Transforms "<<std::endl;
   std::cout<<"-------------------------"<<std::endl;
   std::cout<<"* Enter the alpha value [1.0-3.0]: ";std::cin>>alpha;
   std::cout<<"* Enter the beta value [0-100]: "; std::cin>>beta;


   /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
   /// Instead of these 'for' loops we could have used simply:
   /// image.convertTo(new_image, -1, alpha, beta);
   /// but we wanted to show you how to access the pixels :)
   for( int y = 0; y < image.rows; y++ )
      { for( int x = 0; x < image.cols; x++ )
           { for( int c = 0; c < 3; c++ )
                {
		  new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
                }
	   }
      }

   /// Create Windows
   namedWindow("Original Image", 1);
   namedWindow("New Image", 1);

   /// Show stuff
   imshow("Original Image", image);
   imshow("New Image", new_image);

   
   /// Wait until user press some key
   waitKey();
   return 0;
}
