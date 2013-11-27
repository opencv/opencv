.. _bounding_rotated_ellipses:


Creating Bounding rotated boxes and ellipses for contours
**********************************************************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Use the OpenCV function :min_area_rect:`minAreaRect <>`
   * Use the OpenCV function :fit_ellipse:`fitEllipse <>`


Theory
======

Code
====

This tutorial code's is shown lines below. You can also download it from `here <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo2.cpp>`_

.. code-block:: cpp

   #include "opencv2/highgui/highgui.hpp"
   #include "opencv2/imgproc/imgproc.hpp"
   #include <iostream>
   #include <stdio.h>
   #include <stdlib.h>

   using namespace cv;
   using namespace std;

   Mat src; Mat src_gray;
   int thresh = 100;
   int max_thresh = 255;
   RNG rng(12345);

   /// Function header
   void thresh_callback(int, void* );

   /** @function main */
   int main( int argc, char** argv )
   {
     /// Load source image and convert it to gray
     src = imread( argv[1], 1 );

     /// Convert image to gray and blur it
     cvtColor( src, src_gray, CV_BGR2GRAY );
     blur( src_gray, src_gray, Size(3,3) );

     /// Create Window
     char* source_window = "Source";
     namedWindow( source_window, CV_WINDOW_AUTOSIZE );
     imshow( source_window, src );

     createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
     thresh_callback( 0, 0 );

     waitKey(0);
     return(0);
   }

   /** @function thresh_callback */
   void thresh_callback(int, void* )
   {
     Mat threshold_output;
     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     /// Detect edges using Threshold
     threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
     /// Find contours
     findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

     /// Find the rotated rectangles and ellipses for each contour
     vector<RotatedRect> minRect( contours.size() );
     vector<RotatedRect> minEllipse( contours.size() );

     for( int i = 0; i < contours.size(); i++ )
        { minRect[i] = minAreaRect( Mat(contours[i]) );
          if( contours[i].size() > 5 )
            { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
        }

     /// Draw contours + rotated rects + ellipses
     Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
     for( int i = 0; i< contours.size(); i++ )
        {
          Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
          // contour
          drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
          // ellipse
          ellipse( drawing, minEllipse[i], color, 2, 8 );
          // rotated rectangle
          Point2f rect_points[4]; minRect[i].points( rect_points );
          for( int j = 0; j < 4; j++ )
             line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
        }

     /// Show in a window
     namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
     imshow( "Contours", drawing );
   }

Explanation
============

Result
======

#. Here it is:

   ========== ==========
    |BRE_0|   |BRE_1|
   ========== ==========

   .. |BRE_0|  image:: images/Bounding_Rotated_Ellipses_Source_Image.jpg
                    :align: middle

   .. |BRE_1|  image:: images/Bounding_Rotated_Ellipses_Result.jpg
                    :align: middle
