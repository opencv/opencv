/**
 * @file SBM_Sample
 * @brief Get a disparity map of two images
 * @author A. Huaman
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

const char *windowDisparity = "Disparity";

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  //-- 1. Read the images
  Mat imgLeft = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat imgRight = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
  //-- And create the image in which we will save our disparities
  Mat imgDisparity16S = Mat( imgLeft.rows, imgLeft.cols, CV_16S );
  Mat imgDisparity8U = Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );

  if( !imgLeft.data || !imgRight.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- 2. Call the constructor for StereoBM
  int ndisparities = 16*5;   /**< Range of disparity */
  int SADWindowSize = 21; /**< Size of the block window. Must be odd */

  StereoBM sbm( StereoBM::BASIC_PRESET,
                                ndisparities,
                SADWindowSize );

  //-- 3. Calculate the disparity image
  sbm( imgLeft, imgRight, imgDisparity16S, CV_16S );

  //-- Check its extreme values
  double minVal; double maxVal;

  minMaxLoc( imgDisparity16S, &minVal, &maxVal );

  printf("Min disp: %f Max value: %f \n", minVal, maxVal);

  //-- 4. Display it as a CV_8UC1 image
  imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));

  namedWindow( windowDisparity, WINDOW_NORMAL );
  imshow( windowDisparity, imgDisparity8U );

  //-- 5. Save the image
  imwrite("SBM_sample.png", imgDisparity16S);

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SBMSample <imgLeft> <imgRight>" << std::endl; }
