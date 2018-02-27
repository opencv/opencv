/**
 * @function Remap_Demo.cpp
 * @brief Demo code for Remap
 * @author Ana Huaman
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;

/// Global variables
Mat src, dst;
Mat map_x, map_y;
const char* remap_window = "Remap demo";
int ind = 0;

/// Function Headers
void update_map( void );

/**
 * @function main
 */
int main(int argc, const char** argv)
{
  /// Load the image
  CommandLineParser parser(argc, argv, "{@image |../data/chicky_512.png|input image name}");
  std::string filename = parser.get<std::string>(0);
  src = imread( filename, IMREAD_COLOR );

  /// Create dst, map_x and map_y with the same size as src:
  dst.create( src.size(), src.type() );
  map_x.create( src.size(), CV_32FC1 );
  map_y.create( src.size(), CV_32FC1 );

  /// Create window
  namedWindow( remap_window, WINDOW_AUTOSIZE );

  /// Loop
  for(;;)
  {
    /// Each 1 sec. Press ESC to exit the program
    char c = (char)waitKey( 1000 );

    if( c == 27 )
      { break; }

    /// Update map_x & map_y. Then apply remap
    update_map();
    remap( src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0) );

    // Display results
    imshow( remap_window, dst );
  }
  return 0;
}

/**
 * @function update_map
 * @brief Fill the map_x and map_y matrices with 4 types of mappings
 */
void update_map( void )
{
  ind = ind%4;

  for( int j = 0; j < src.rows; j++ )
    { for( int i = 0; i < src.cols; i++ )
     {
           switch( ind )
         {
         case 0:
           if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 )
                 {
               map_x.at<float>(j,i) = 2*( i - src.cols*0.25f ) + 0.5f ;
               map_y.at<float>(j,i) = 2*( j - src.rows*0.25f ) + 0.5f ;
              }
           else
         { map_x.at<float>(j,i) = 0 ;
               map_y.at<float>(j,i) = 0 ;
                 }
                   break;
         case 1:
               map_x.at<float>(j,i) = (float)i ;
               map_y.at<float>(j,i) = (float)(src.rows - j) ;
           break;
             case 2:
               map_x.at<float>(j,i) = (float)(src.cols - i) ;
               map_y.at<float>(j,i) = (float)j ;
           break;
             case 3:
               map_x.at<float>(j,i) = (float)(src.cols - i) ;
               map_y.at<float>(j,i) = (float)(src.rows - j) ;
           break;
             } // end of switch
     }
    }
  ind++;
}
