.. _harris_detector:

Harris corner detector
**********************

Goal
=====

In this tutorial you will learn:

.. container:: enumeratevisibleitemswithsquare

   * What features are and why they are important
   * Use the function :corner_harris:`cornerHarris <>` to detect corners using the Harris-Stephens method.

Theory
======

What is a feature?
-------------------

.. container:: enumeratevisibleitemswithsquare

   * In computer vision, usually we need to find matching points between different frames of an environment. Why? If we know how two images relate to each other, we can use *both* images to extract information of them.

   * When we say **matching points** we are referring, in a general sense, to *characteristics* in the scene that we can recognize easily. We call these characteristics **features**.

   * **So, what characteristics should a feature have?**

     * It must be *uniquely recognizable*


Types of Image Features
------------------------

To mention a few:

.. container:: enumeratevisibleitemswithsquare

   * Edges
   * **Corners** (also known as interest points)
   * Blobs (also known as regions of interest )

In this tutorial we will study the *corner* features, specifically.

Why is a corner so special?
----------------------------

.. container:: enumeratevisibleitemswithsquare

   * Because, since it is the intersection of two edges, it represents a point in which the directions of these two edges *change*. Hence, the gradient of the image (in both directions) have a high variation, which can be used to detect it.


How does it work?
-----------------

.. container:: enumeratevisibleitemswithsquare

   * Let's look for corners. Since corners represents a variation in the gradient in the image, we will look for this "variation".

   * Consider a grayscale image :math:`I`. We are going to sweep a window :math:`w(x,y)` (with displacements :math:`u` in the x direction and :math:`v` in the right direction) :math:`I` and will calculate the variation of intensity.

     .. math::

        E(u,v) = \sum _{x,y} w(x,y)[ I(x+u,y+v) - I(x,y)]^{2}

     where:

     * :math:`w(x,y)` is the window at position :math:`(x,y)`
     * :math:`I(x,y)` is the intensity at :math:`(x,y)`
     * :math:`I(x+u,y+v)` is the intensity at the moved window :math:`(x+u,y+v)`

   * Since we are looking for windows with corners, we are looking for windows with a large variation in intensity. Hence, we have to maximize the equation above, specifically the term:

     .. math::

        \sum _{x,y}[ I(x+u,y+v) - I(x,y)]^{2}


   * Using *Taylor expansion*:

     .. math::

        E(u,v) \approx \sum _{x,y}[ I(x,y) + u I_{x} + vI_{y} - I(x,y)]^{2}


   * Expanding the equation and cancelling properly:

     .. math::

        E(u,v) \approx \sum _{x,y} u^{2}I_{x}^{2} + 2uvI_{x}I_{y} + v^{2}I_{y}^{2}

   * Which can be expressed in a matrix form as:

     .. math::

        E(u,v) \approx \begin{bmatrix}
                        u & v
                       \end{bmatrix}
                       \left (
               \displaystyle \sum_{x,y}
                       w(x,y)
                       \begin{bmatrix}
                        I_x^{2} & I_{x}I_{y} \\
                        I_xI_{y} & I_{y}^{2}
               \end{bmatrix}
               \right )
               \begin{bmatrix}
                        u \\
            v
                       \end{bmatrix}

   * Let's denote:

     .. math::

        M = \displaystyle \sum_{x,y}
                  w(x,y)
                  \begin{bmatrix}
                            I_x^{2} & I_{x}I_{y} \\
                            I_xI_{y} & I_{y}^{2}
                       \end{bmatrix}

   * So, our equation now is:

     .. math::

        E(u,v) \approx \begin{bmatrix}
                        u & v
                       \end{bmatrix}
               M
               \begin{bmatrix}
                        u \\
            v
                       \end{bmatrix}


   * A score is calculated for each window, to determine if it can possibly contain a corner:

     .. math::

        R = det(M) - k(trace(M))^{2}

     where:

     * det(M) = :math:`\lambda_{1}\lambda_{2}`
     * trace(M) = :math:`\lambda_{1}+\lambda_{2}`

     a window with a score :math:`R` greater than a certain value is considered a "corner"

Code
====

This tutorial code's is shown lines below. You can also download it from `here <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/TrackingMotion/cornerHarris_Demo.cpp>`_

.. code-block:: cpp

   #include "opencv2/highgui.hpp"
   #include "opencv2/imgproc.hpp"
   #include <iostream>
   #include <stdio.h>
   #include <stdlib.h>

   using namespace cv;
   using namespace std;

   /// Global variables
   Mat src, src_gray;
   int thresh = 200;
   int max_thresh = 255;

   char* source_window = "Source image";
   char* corners_window = "Corners detected";

   /// Function header
   void cornerHarris_demo( int, void* );

   /** @function main */
   int main( int argc, char** argv )
   {
     /// Load source image and convert it to gray
     src = imread( argv[1], 1 );
     cvtColor( src, src_gray, CV_BGR2GRAY );

     /// Create a window and a trackbar
     namedWindow( source_window, CV_WINDOW_AUTOSIZE );
     createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );
     imshow( source_window, src );

     cornerHarris_demo( 0, 0 );

     waitKey(0);
     return(0);
   }

   /** @function cornerHarris_demo */
   void cornerHarris_demo( int, void* )
   {

     Mat dst, dst_norm, dst_norm_scaled;
     dst = Mat::zeros( src.size(), CV_32FC1 );

     /// Detector parameters
     int blockSize = 2;
     int apertureSize = 3;
     double k = 0.04;

     /// Detecting corners
     cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

     /// Normalizing
     normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
     convertScaleAbs( dst_norm, dst_norm_scaled );

     /// Drawing a circle around corners
     for( int j = 0; j < dst_norm.rows ; j++ )
        { for( int i = 0; i < dst_norm.cols; i++ )
             {
               if( (int) dst_norm.at<float>(j,i) > thresh )
                 {
                  circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                 }
             }
        }
     /// Showing the result
     namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
     imshow( corners_window, dst_norm_scaled );
   }


Explanation
============

Result
======

The original image:

.. image:: images/Harris_Detector_Original_Image.jpg
              :align: center

The detected corners are surrounded by a small black circle

.. image:: images/Harris_Detector_Result.jpg
              :align: center
