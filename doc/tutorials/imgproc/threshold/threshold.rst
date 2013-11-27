.. _Basic_Threshold:

Basic Thresholding Operations
*******************************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Perform basic thresholding operations using OpenCV function :threshold:`threshold <>`


Cool Theory
============

.. note::
   The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

What is Thresholding?
-----------------------

* The simplest segmentation method

* Application example: Separate out regions of an image corresponding to objects which we want to analyze. This separation is based on the variation of intensity between the object pixels and the background pixels.

* To differentiate the pixels we are interested in from the rest (which will eventually be rejected), we perform a comparison of  each pixel intensity value with respect to a *threshold* (determined according to the problem to solve).

* Once we have separated properly the important pixels, we can set them with a determined value to identify them (i.e. we can assign them a value of :math:`0` (black), :math:`255` (white) or any value  that suits your needs).

  .. image:: images/Threshold_Tutorial_Theory_Example.jpg
     :alt: Threshold simple example
     :align: center

Types of Thresholding
-----------------------

* OpenCV offers the function :threshold:`threshold <>` to perform thresholding operations.

* We can effectuate :math:`5` types of Thresholding operations with this function. We will explain them in the following subsections.

* To illustrate how these thresholding processes work, let's consider that we have a source image with pixels with intensity values :math:`src(x,y)`. The plot below depicts this. The horizontal blue line represents the threshold :math:`thresh` (fixed).

  .. image:: images/Threshold_Tutorial_Theory_Base_Figure.png
     :alt: Threshold Binary
     :align: center

Threshold Binary
^^^^^^^^^^^^^^^^^

* This thresholding operation can be expressed as:

  .. math::

     \texttt{dst} (x,y) =  \fork{\texttt{maxVal}}{if $\texttt{src}(x,y) > \texttt{thresh}$}{0}{otherwise}

* So, if the intensity of the pixel :math:`src(x,y)` is higher than :math:`thresh`, then the new pixel intensity is set to a :math:`MaxVal`. Otherwise, the pixels are set to :math:`0`.

  .. image:: images/Threshold_Tutorial_Theory_Binary.png
     :alt: Threshold Binary
     :align: center


Threshold Binary, Inverted
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* This thresholding operation can be expressed as:

  .. math::

     \texttt{dst} (x,y) =  \fork{0}{if $\texttt{src}(x,y) > \texttt{thresh}$}{\texttt{maxVal}}{otherwise}

* If the intensity of the pixel :math:`src(x,y)` is higher than :math:`thresh`, then the new pixel intensity is set to a :math:`0`. Otherwise, it is set to :math:`MaxVal`.

  .. image:: images/Threshold_Tutorial_Theory_Binary_Inverted.png
     :alt: Threshold Binary Inverted
     :align: center

Truncate
^^^^^^^^^

* This thresholding operation can be expressed as:

  .. math::

     \texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if $\texttt{src}(x,y) > \texttt{thresh}$}{\texttt{src}(x,y)}{otherwise}

* The maximum intensity value for the pixels is :math:`thresh`, if :math:`src(x,y)` is greater, then its value is *truncated*. See figure below:

  .. image:: images/Threshold_Tutorial_Theory_Truncate.png
     :alt: Threshold Truncate
     :align: center



Threshold to Zero
^^^^^^^^^^^^^^^^^^

* This operation can be expressed as:

   .. math::

      \texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if $\texttt{src}(x,y) > \texttt{thresh}$}{0}{otherwise}

* If :math:`src(x,y)` is lower than :math:`thresh`, the new pixel value will be set to :math:`0`.

  .. image:: images/Threshold_Tutorial_Theory_Zero.png
     :alt: Threshold Zero
     :align: center


Threshold to Zero, Inverted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* This operation can be expressed as:

   .. math::

      \texttt{dst} (x,y) =  \fork{0}{if $\texttt{src}(x,y) > \texttt{thresh}$}{\texttt{src}(x,y)}{otherwise}

* If  :math:`src(x,y)` is greater than :math:`thresh`, the new pixel value will be set to :math:`0`.

  .. image:: images/Threshold_Tutorial_Theory_Zero_Inverted.png
     :alt: Threshold Zero Inverted
     :align: center


Code
======

The tutorial code's is shown lines below. You can also download it from `here <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/ImgProc/Threshold.cpp>`_

.. code-block:: cpp

   #include "opencv2/imgproc/imgproc.hpp"
   #include "opencv2/highgui/highgui.hpp"
   #include <stdlib.h>
   #include <stdio.h>

   using namespace cv;

   /// Global variables

   int threshold_value = 0;
   int threshold_type = 3;;
   int const max_value = 255;
   int const max_type = 4;
   int const max_BINARY_value = 255;

   Mat src, src_gray, dst;
   char* window_name = "Threshold Demo";

   char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
   char* trackbar_value = "Value";

   /// Function headers
   void Threshold_Demo( int, void* );

   /**
    * @function main
    */
   int main( int argc, char** argv )
   {
     /// Load an image
     src = imread( argv[1], 1 );

     /// Convert the image to Gray
     cvtColor( src, src_gray, CV_RGB2GRAY );

     /// Create a window to display results
     namedWindow( window_name, CV_WINDOW_AUTOSIZE );

     /// Create Trackbar to choose type of Threshold
     createTrackbar( trackbar_type,
                     window_name, &threshold_type,
                     max_type, Threshold_Demo );

     createTrackbar( trackbar_value,
                     window_name, &threshold_value,
                     max_value, Threshold_Demo );

     /// Call the function to initialize
     Threshold_Demo( 0, 0 );

     /// Wait until user finishes program
     while(true)
     {
       int c;
       c = waitKey( 20 );
       if( (char)c == 27 )
         { break; }
      }

   }


   /**
    * @function Threshold_Demo
    */
   void Threshold_Demo( int, void* )
   {
     /* 0: Binary
        1: Binary Inverted
        2: Threshold Truncated
        3: Threshold to Zero
        4: Threshold to Zero Inverted
      */

     threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

     imshow( window_name, dst );
   }



Explanation
=============


#. Let's check the general structure of the program:

   * Load an image. If it is RGB we convert it to Grayscale. For this, remember that we can use the function :cvt_color:`cvtColor <>`:

     .. code-block:: cpp

        src = imread( argv[1], 1 );

        /// Convert the image to Gray
        cvtColor( src, src_gray, CV_RGB2GRAY );


   * Create a window to display the result

     .. code-block:: cpp

        namedWindow( window_name, CV_WINDOW_AUTOSIZE );

   * Create :math:`2` trackbars for the user to enter user input:

     * 	**Type of thresholding**: Binary, To Zero, etc...
     *  **Threshold value**

     .. code-block:: cpp

        createTrackbar( trackbar_type,
                     window_name, &threshold_type,
                     max_type, Threshold_Demo );

        createTrackbar( trackbar_value,
                     window_name, &threshold_value,
                     max_value, Threshold_Demo );

   * Wait until the user enters the threshold value, the type of thresholding (or until the program exits)

   * Whenever the user changes the value of any of the Trackbars, the function *Threshold_Demo* is called:

     .. code-block:: cpp

        /**
         * @function Threshold_Demo
         */
        void Threshold_Demo( int, void* )
        {
          /* 0: Binary
             1: Binary Inverted
             2: Threshold Truncated
             3: Threshold to Zero
             4: Threshold to Zero Inverted
           */

          threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

          imshow( window_name, dst );
        }

     As you can see, the function :threshold:`threshold <>` is invoked. We give :math:`5` parameters:

     * *src_gray*: Our input image
     * *dst*: Destination (output) image
     * *threshold_value*: The :math:`thresh` value with respect to which the thresholding operation is made
     * *max_BINARY_value*: The value  used with the Binary thresholding operations (to set the chosen pixels)
     * *threshold_type*: One of the :math:`5` thresholding operations. They are listed in the comment section of the function above.



Results
========

#. After compiling this program, run it giving a path to an image as argument. For instance, for an input image as:


   .. image:: images/Threshold_Tutorial_Original_Image.jpg
      :alt: Threshold Original Image
      :align: center

#. First, we try to threshold our image with a *binary threhold inverted*. We expect that the pixels brighter than the :math:`thresh` will turn dark, which is what actually happens, as we can see in the snapshot below (notice from the original image, that the doggie's tongue and eyes are particularly bright in comparison with the image, this is reflected in the output image).


   .. image:: images/Threshold_Tutorial_Result_Binary_Inverted.jpg
      :alt: Threshold Result Binary Inverted
      :align: center


#. Now we try with the *threshold to zero*. With this, we expect that the darkest pixels (below the threshold) will become completely black, whereas the pixels with value greater than the threshold will keep its original value. This is verified by the following snapshot of the output image:

   .. image:: images/Threshold_Tutorial_Result_Zero.jpg
      :alt: Threshold Result Zero
      :align: center
