.. _feature_detection:

Feature Detection
******************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Use the :feature_detector:`FeatureDetector<>` interface in order to find interest points. Specifically:

     * Use the :surf_feature_detector:`SurfFeatureDetector<>` and its function :feature_detector_detect:`detect<>` to perform the detection process
     * Use the function :draw_keypoints:`drawKeypoints<>` to draw the detected keypoints
     

Theory
======

Code
====

This tutorial code's is shown lines below. You can also download it from `here <https://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/tutorial_code/features2D/SURF_detector.cpp>`_

.. code-block:: cpp 

   #include <stdio.h>
   #include <iostream>
   #include "opencv2/core/core.hpp"
   #include "opencv2/features2d/features2d.hpp"	
   #include "opencv2/highgui/highgui.hpp"

   using namespace cv;

   void readme();

   /** @function main */
   int main( int argc, char** argv )
   {
     if( argc != 3 )
     { readme(); return -1; }

     Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
     Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
  
     if( !img_1.data || !img_2.data )
     { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

     //-- Step 1: Detect the keypoints using SURF Detector
     int minHessian = 400;

     SurfFeatureDetector detector( minHessian );

     std::vector<KeyPoint> keypoints_1, keypoints_2;

     detector.detect( img_1, keypoints_1 );
     detector.detect( img_2, keypoints_2 );

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

     /** @function readme */
     void readme()
     { std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl; }

Explanation
============

Result
======
 
#. Here is the result of the feature detection applied to the first image:
 
   .. image:: images/Feature_Detection_Result_a.jpg
      :align: center
      :height: 125pt

#. And here is the result for the second image:

   .. image:: images/Feature_Detection_Result_b.jpg
      :align: center  
      :height: 200pt 

