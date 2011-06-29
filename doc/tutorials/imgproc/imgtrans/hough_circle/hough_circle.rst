.. _hough_circle:

Hough Circle Transform
***********************

Goal
=====
In this tutorial you will learn how to:

* Use the OpenCV functions :hough_circles:`HoughCircles <>` to detect circles in an image.

Code
======

#. **What does this program do?**
 
   * Loads an image and blur it to reduce the noise
   * Applies the *Hough Circle Transform* to the blurred image . 
   * Display the detected circle in a window.

#. The sample code that we will explain can be downloaded from `here <https://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/houghlines.cpp>`_. A slightly fancier version (which shows both Hough standard and probabilistic with trackbars for changing the threshold values) can be found  `here <https://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/tutorial_code/ImgTrans/HoughCircle_Demo.cpp>`_

.. code-block:: cpp 

   #include "opencv2/highgui/highgui.hpp"
   #include "opencv2/imgproc/imgproc.hpp"
   #include <iostream>
   #include <stdio.h>

   using namespace cv;

   /** @function main */
   int main(int argc, char** argv)
   {
     Mat src, src_gray;

     /// Read the image
     src = imread( argv[1], 1 );

     if( !src.data )
       { return -1; }

     /// Convert it to gray 
     cvtColor( src, src_gray, CV_BGR2GRAY );

     /// Reduce the noise so we avoid false circle detection
     GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

     vector<Vec3f> circles;

     /// Apply the Hough Transform to find the circles
     HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );

     /// Draw the circles detected
     for( size_t i = 0; i < circles.size(); i++ )
     {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // circle center
         circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // circle outline
         circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
      }

     /// Show your results 
     namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
     imshow( "Hough Circle Transform Demo", src );

     waitKey(0);
     return 0;
   }

Result
=======
 
.. image:: images/Hough_Circle_Tutorial_Result.jpg
   :alt: Result of detecting circles with Hough Transform
   :align: center 
