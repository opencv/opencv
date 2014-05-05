.. _feature_description:

Feature Description
*******************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Use the :descriptor_extractor:`DescriptorExtractor<>` interface in order to find the feature vector correspondent to the keypoints. Specifically:

     * Use :surf_descriptor_extractor:`SurfDescriptorExtractor<>` and its function :descriptor_extractor:`compute<>` to perform the required calculations.
     * Use a :brute_force_matcher:`BFMatcher<>`	to match the features vector
     * Use the function :draw_matches:`drawMatches<>` to draw the detected matches.


Theory
======

Code
====

This tutorial code's is shown lines below. You can also download it from `here <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/features2D/SURF_descriptor.cpp>`_

.. code-block:: cpp

   #include <stdio.h>
   #include <iostream>
   #include "opencv2/core.hpp"
   #include "opencv2/features2d.hpp"
   #include "opencv2/highgui.hpp"
   #include "opencv2/nonfree.hpp"

   using namespace cv;

   void readme();

   /** @function main */
   int main( int argc, char** argv )
   {
     if( argc != 3 )
      { return -1; }

     Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
     Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

     if( !img_1.data || !img_2.data )
      { return -1; }

     //-- Step 1: Detect the keypoints using SURF Detector
     int minHessian = 400;

     SurfFeatureDetector detector( minHessian );

     std::vector<KeyPoint> keypoints_1, keypoints_2;

     detector.detect( img_1, keypoints_1 );
     detector.detect( img_2, keypoints_2 );

     //-- Step 2: Calculate descriptors (feature vectors)
     SurfDescriptorExtractor extractor;

     Mat descriptors_1, descriptors_2;

     extractor.compute( img_1, keypoints_1, descriptors_1 );
     extractor.compute( img_2, keypoints_2, descriptors_2 );

     //-- Step 3: Matching descriptor vectors with a brute force matcher
     BFMatcher matcher(NORM_L2);
     std::vector< DMatch > matches;
     matcher.match( descriptors_1, descriptors_2, matches );

     //-- Draw matches
     Mat img_matches;
     drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

     //-- Show detected matches
     imshow("Matches", img_matches );

     waitKey(0);

     return 0;
     }

    /** @function readme */
    void readme()
    { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }

Explanation
============

Result
======

#. Here is the result after applying the BruteForce matcher between the two original images:

   .. image:: images/Feature_Description_BruteForce_Result.jpg
      :align: center
      :height: 200pt
