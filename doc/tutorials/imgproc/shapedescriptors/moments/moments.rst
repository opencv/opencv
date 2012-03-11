.. _moments:


Image Moments
**************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Use the OpenCV function :moments:`moments <>` 
   * Use the OpenCV function :contour_area:`contourArea <>`
   * Use the OpenCV function :arc_length:`arcLength <>`           

Theory
======

Code
====

This tutorial code's is shown lines below. You can also download it from `here <http://code.opencv.org/svn/opencv/trunk/opencv/samples/cpp/tutorial_code/ShapeDescriptors/moments_demo.cpp>`_

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

     createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
     thresh_callback( 0, 0 );

     waitKey(0);
     return(0);
   }

   /** @function thresh_callback */
   void thresh_callback(int, void* )
   {
     Mat canny_output;
     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     /// Detect edges using canny
     Canny( src_gray, canny_output, thresh, thresh*2, 3 );
     /// Find contours  
     findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

     /// Get the moments
     vector<Moments> mu(contours.size() );
     for( int i = 0; i < contours.size(); i++ )
        { mu[i] = moments( contours[i], false ); }

     ///  Get the mass centers: 
     vector<Point2f> mc( contours.size() );
     for( int i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

     /// Draw contours
     Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
     for( int i = 0; i< contours.size(); i++ )
        { 
          Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
          drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() ); 
          circle( drawing, mc[i], 4, color, -1, 8, 0 );
        }

     /// Show in a window
     namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
     imshow( "Contours", drawing );

     /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
     printf("\t Info: Area and Contour Length \n");
     for( int i = 0; i< contours.size(); i++ )
        {
          printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );  
          Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
          drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() ); 
          circle( drawing, mc[i], 4, color, -1, 8, 0 );
        }
   }

Explanation
============

Result
======

#. Here it is:

   ========== ==========  ==========  
    |MU_0|     |MU_1|      |MU_2|   
   ========== ==========  ========== 

   .. |MU_0|  image:: images/Moments_Source_Image.jpg
                    :width: 250pt
                    :align: middle

   .. |MU_1|  image:: images/Moments_Result1.jpg
                    :width: 250pt
                    :align: middle   

   .. |MU_2|  image:: images/Moments_Result2.jpg
                    :width: 250pt
                    :align: middle   

