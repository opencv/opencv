/**
 * @file LoadSaveImage.cpp
 * @brief Sample code that load an image, modify it and save the new image.  
 * @author OpenCV team
 */

#include <cv.h>
#include <highgui.h>

using namespace cv;

/**
 * @function main
 * @brief Self-explanatory
 */
int main( int argc, char** argv )
{
  /// Get the name of the file to be loaded
  char* imageName = argv[1];

  /// Create a Mat object
  Mat image;
  
  /// Load the image using imread
  image = imread( imageName, 1 );
  
  /// Verify that the image was loaded
  if( argc != 2 || !image.data )
    {
      printf( " No image data \n " );
      return -1;
    }

  /// Change the image to Grayscale
  Mat gray_image;
  cvtColor( image, gray_image, CV_RGB2GRAY );

  /// Save our gray image
  imwrite( "../../images/Gray_Image.png", gray_image );

  /// Create a couple of windows and show our images
  namedWindow( imageName, CV_WINDOW_AUTOSIZE );
  namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );

  imshow( imageName, image );
  imshow( "Gray image", gray_image ); 

  /// Wait until user finish the application   
  waitKey(0);

  return 0;
}
