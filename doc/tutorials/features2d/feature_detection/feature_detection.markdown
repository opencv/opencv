Feature Detection {#tutorial_feature_detection}
=================

Goal
----

In this tutorial you will learn how to:

-   Use the @ref cv::FeatureDetector interface in order to find interest points. Specifically:
    -   Use the cv::xfeatures2d::SURF and its function cv::xfeatures2d::SURF::detect to perform the
        detection process
    -   Use the function @ref cv::drawKeypoints to draw the detected keypoints

Theory
------

Code
----

This tutorial code's is shown lines below.
@code{.cpp}
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

/* @function main */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  Ptr<SURF> detector = SURF::create( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );

  //-- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
  imshow("Keypoints 2", img_keypoints_2 );

  waitKey(0);

  return 0;
  }

  /* @function readme */
  void readme()
  { std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl; }
@endcode

Explanation
-----------

Result
------

-#  Here is the result of the feature detection applied to the first image:

    ![](images/Feature_Detection_Result_a.jpg)

-#  And here is the result for the second image:

    ![](images/Feature_Detection_Result_b.jpg)
