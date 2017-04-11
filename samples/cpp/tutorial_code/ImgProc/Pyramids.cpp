/**
 * @file Pyramids.cpp
 * @brief Sample code of image pyramids (pyrDown and pyrUp)
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

/// Global variables
Mat src, dst, tmp;

const char* window_name = "Pyramids Demo";


/**
 * @function main
 */
int main( void )
{
  /// General instructions
  printf( "\n Zoom In-Out demo  \n " );
  printf( "------------------ \n" );
  printf( " * [u] -> Zoom in  \n" );
  printf( " * [d] -> Zoom out \n" );
  printf( " * [ESC] -> Close program \n \n" );

  //![load]
  src = imread( "../data/chicky_512.png" ); // Loads the test image
  if( src.empty() )
    { printf(" No data! -- Exiting the program \n");
      return -1; }
  //![load]

  tmp = src;
  dst = tmp;

  //![create_window]
  imshow( window_name, dst );
  //![create_window]

  //![infinite_loop]
  for(;;)
  {
    char c = (char)waitKey(0);

    if( c == 27 )
      { break; }
    //![pyrup]
    if( c == 'u' )
      { pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) );
        printf( "** Zoom In: Image x 2 \n" );
      }
    //![pyrup]
    //![pyrdown]
    else if( c == 'd' )
      { pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 ) );
        printf( "** Zoom Out: Image / 2 \n" );
      }
    //![pyrdown]
    imshow( window_name, dst );

    //![update_tmp]
    tmp = dst;
    //![update_tmp]
   }
   //![infinite_loop]

   return 0;
}
