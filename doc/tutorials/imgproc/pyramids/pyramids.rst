.. _Pyramids:

Image Pyramids
***************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Use the OpenCV functions :pyr_up:`pyrUp <>` and :pyr_down:`pyrDown <>` to downsample  or upsample a given image.

Theory
=======

.. note::
   The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

.. container:: enumeratevisibleitemswithsquare

   * Usually we need to convert an image to a size different than its original. For this, there are two possible options:

     #. *Upsize* the image (zoom in) or
     #. *Downsize* it (zoom out).

   * Although there is a *geometric transformation* function in OpenCV that -literally- resize an image (:resize:`resize <>`, which we will show in a future tutorial), in this section we analyze first the use of **Image Pyramids**, which are widely applied in a huge range of vision applications.


Image Pyramid
--------------

.. container:: enumeratevisibleitemswithsquare

   * An image pyramid is a collection of images - all arising from a single original image - that are successively downsampled until some desired stopping point is reached.

   * There are two common kinds of image pyramids:

     * **Gaussian pyramid:** Used to downsample images

     * **Laplacian pyramid:** Used to  reconstruct an upsampled image from an image lower in the pyramid (with less resolution)

   * In this tutorial we'll use the *Gaussian pyramid*.

Gaussian Pyramid
^^^^^^^^^^^^^^^^^

* Imagine the pyramid as a set of layers in which the higher the layer, the smaller the size.

  .. image:: images/Pyramids_Tutorial_Pyramid_Theory.png
     :alt: Pyramid figure
     :align: center

* Every layer is numbered from bottom to top, so layer :math:`(i+1)` (denoted as :math:`G_{i+1}` is smaller than layer :math:`i` (:math:`G_{i}`).

* To produce layer :math:`(i+1)` in the Gaussian pyramid, we do the following:

  * Convolve :math:`G_{i}` with a Gaussian kernel:

    .. math::

       \frac{1}{16} \begin{bmatrix} 1 & 4 & 6 & 4 & 1  \\ 4 & 16 & 24 & 16 & 4  \\ 6 & 24 & 36 & 24 & 6  \\ 4 & 16 & 24 & 16 & 4  \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}

  * Remove every even-numbered row and column.

* You can easily notice that the resulting image will be exactly one-quarter the area of its predecessor. Iterating this process on the input image :math:`G_{0}` (original image) produces the entire pyramid.

* The procedure above was useful to downsample an image. What if we want to make it bigger?:

  * First, upsize the image to twice the original in each dimension, wit the new even rows and columns filled with zeros (:math:`0`)

  * Perform a convolution with the same kernel shown above (multiplied by 4) to approximate the values of the "missing pixels"

* These two procedures (downsampling and upsampling as explained above) are implemented by the OpenCV functions :pyr_up:`pyrUp <>` and :pyr_down:`pyrDown <>`, as we will see in an example with the code below:

.. note::
   When we reduce the size of an image, we are actually *losing* information of the image.

Code
======

This tutorial code's is shown lines below. You can also download it from `here <http://code.opencv.org/projects/opencv/repository/revisions/master/raw/samples/cpp/tutorial_code/ImgProc/Pyramids.cpp>`_

.. code-block:: cpp

   #include "opencv2/imgproc.hpp"
   #include "opencv2/highgui.hpp"
   #include <math.h>
   #include <stdlib.h>
   #include <stdio.h>

   using namespace cv;

   /// Global variables
   Mat src, dst, tmp;
   char* window_name = "Pyramids Demo";


   /**
    * @function main
    */
   int main( int argc, char** argv )
   {
     /// General instructions
     printf( "\n Zoom In-Out demo  \n " );
     printf( "------------------ \n" );
     printf( " * [u] -> Zoom in  \n" );
     printf( " * [d] -> Zoom out \n" );
     printf( " * [ESC] -> Close program \n \n" );

     /// Test image - Make sure it s divisible by 2^{n}
     src = imread( "../images/chicky_512.jpg" );
     if( !src.data )
       { printf(" No data! -- Exiting the program \n");
         return -1; }

     tmp = src;
     dst = tmp;

     /// Create window
     namedWindow( window_name, CV_WINDOW_AUTOSIZE );
     imshow( window_name, dst );

     /// Loop
     while( true )
     {
       int c;
       c = waitKey(10);

       if( (char)c == 27 )
         { break; }
       if( (char)c == 'u' )
         { pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) );
           printf( "** Zoom In: Image x 2 \n" );
         }
       else if( (char)c == 'd' )
        { pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 ) );
          printf( "** Zoom Out: Image / 2 \n" );
        }

       imshow( window_name, dst );
       tmp = dst;
     }
     return 0;
   }

Explanation
=============

#. Let's check the general structure of the program:

   * Load an image (in this case it is defined in the program, the user does not have to enter it as an argument)

     .. code-block:: cpp

        /// Test image - Make sure it s divisible by 2^{n}
        src = imread( "../images/chicky_512.jpg" );
        if( !src.data )
          { printf(" No data! -- Exiting the program \n");
            return -1; }

   * Create a Mat object to store the result of the operations (*dst*) and one to save temporal results (*tmp*).

     .. code-block:: cpp

        Mat src, dst, tmp;
        /* ... */
        tmp = src;
        dst = tmp;



   * Create a window to display the result

     .. code-block:: cpp

        namedWindow( window_name, CV_WINDOW_AUTOSIZE );
        imshow( window_name, dst );

   * Perform an infinite loop waiting for user input.

     .. code-block:: cpp

        while( true )
        {
          int c;
          c = waitKey(10);

          if( (char)c == 27 )
            { break; }
          if( (char)c == 'u' )
            { pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) );
              printf( "** Zoom In: Image x 2 \n" );
            }
          else if( (char)c == 'd' )
           { pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 ) );
             printf( "** Zoom Out: Image / 2 \n" );
           }

          imshow( window_name, dst );
          tmp = dst;
        }


     Our program exits if the user presses *ESC*. Besides, it has two options:

     * **Perform upsampling (after pressing 'u')**

       .. code-block:: cpp

          pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 )

       We use the function :pyr_up:`pyrUp <>` with 03 arguments:

       * *tmp*: The current image, it is initialized with the *src* original image.
       * *dst*: The destination image (to be shown on screen, supposedly the double of the input image)
       * *Size( tmp.cols*2, tmp.rows*2 )* : The destination size. Since we are upsampling, :pyr_up:`pyrUp <>` expects a size double than the input image (in this case *tmp*).

     * **Perform downsampling (after pressing 'd')**

       .. code-block:: cpp

          pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 )

       Similarly as with :pyr_up:`pyrUp <>`, we use the function :pyr_down:`pyrDown <>` with 03 arguments:

       * *tmp*: The current image, it is initialized with the *src* original image.
       * *dst*: The destination image (to be shown on screen, supposedly half the input image)
       * *Size( tmp.cols/2, tmp.rows/2 )* : The destination size. Since we are upsampling, :pyr_down:`pyrDown <>` expects half the size the input image (in this case *tmp*).

     * Notice that it is important that the input image can be divided by a factor of two (in both dimensions). Otherwise, an error will be shown.

     * Finally, we update the input image **tmp** with the current image displayed, so the subsequent operations are performed on it.

       .. code-block:: cpp

          tmp = dst;



Results
========

* After compiling the code above we can test it. The program calls an image **chicky_512.jpg** that comes in the *tutorial_code/image* folder. Notice that this image is :math:`512 \times 512`, hence a downsample won't generate any error (:math:`512 = 2^{9}`). The original image is shown below:

  .. image:: images/Pyramids_Tutorial_Original_Image.jpg
     :alt: Pyramids: Original image
     :align: center

* First we apply two successive :pyr_down:`pyrDown <>` operations by pressing 'd'. Our output is:

  .. image:: images/Pyramids_Tutorial_PyrDown_Result.jpg
     :alt: Pyramids: PyrDown Result
     :align: center

* Note that we should have lost some resolution due to the fact that we are diminishing the size of the image. This is evident after we apply :pyr_up:`pyrUp <>` twice (by pressing 'u'). Our output is now:

  .. image:: images/Pyramids_Tutorial_PyrUp_Result.jpg
     :alt: Pyramids: PyrUp Result
     :align: center
