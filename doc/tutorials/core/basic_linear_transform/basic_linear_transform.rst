.. _Basic_Linear_Transform:

Changing the contrast and brightness of an image!
***************************************************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   + Access pixel values
   + Initialize a matrix with zeros
   + Learn what :saturate_cast:`saturate_cast <>` does and why it is useful
   + Get some cool info about pixel transformations

Theory
=======

.. note::

   The explanation below belongs to the book `Computer Vision: Algorithms and Applications <http://szeliski.org/Book/>`_  by Richard Szeliski

Image Processing
--------------------

.. container:: enumeratevisibleitemswithsquare

   * A general image processing operator is a function that takes one or more input images and produces an output image.

   * Image transforms can be seen as:

     + Point operators (pixel transforms)
     + Neighborhood (area-based) operators


Pixel Transforms
-----------------

.. container:: enumeratevisibleitemswithsquare

   * In this kind of image processing transform, each output pixel's value depends on only the corresponding input pixel value (plus, potentially, some globally collected information or parameters).

   * Examples of such operators include *brightness and contrast adjustments* as well as color correction and transformations.

Brightness and contrast adjustments
------------------------------------

.. container:: enumeratevisibleitemswithsquare

   * Two commonly used point processes are *multiplication* and *addition* with a constant:

     .. math::

        g(x) = \alpha f(x) + \beta

   * The parameters :math:`\alpha > 0` and :math:`\beta` are often called the *gain* and *bias* parameters; sometimes these parameters are said to control *contrast* and *brightness* respectively.

   * You can think of :math:`f(x)` as the source image pixels and :math:`g(x)` as the output image pixels. Then, more conveniently we can write the expression as:

     .. math::

        g(i,j) = \alpha \cdot f(i,j) + \beta

     where :math:`i` and :math:`j` indicates that the pixel is located in the *i-th* row and *j-th* column.

Code
=====

* The following code performs the operation :math:`g(i,j) = \alpha \cdot f(i,j) + \beta` :

.. code-block:: cpp

   #include <opencv2/opencv.hpp>
   #include <iostream>

   using namespace cv;

   double alpha; /**< Simple contrast control */
   int beta;  /**< Simple brightness control */

   int main( int argc, char** argv )
   {
       /// Read image given by user
       Mat image = imread( argv[1] );
       Mat new_image = Mat::zeros( image.size(), image.type() );

       /// Initialize values
       std::cout<<" Basic Linear Transforms "<<std::endl;
       std::cout<<"-------------------------"<<std::endl;
       std::cout<<"* Enter the alpha value [1.0-3.0]: ";std::cin>>alpha;
       std::cout<<"* Enter the beta value [0-100]: "; std::cin>>beta;

       /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
       for( int y = 0; y < image.rows; y++ ) {
           for( int x = 0; x < image.cols; x++ ) {
               for( int c = 0; c < 3; c++ ) {
                   new_image.at<Vec3b>(y,x)[c] =
                   saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
               }
           }
       }

       /// Create Windows
       namedWindow("Original Image", 1);
       namedWindow("New Image", 1);

       /// Show stuff
       imshow("Original Image", image);
       imshow("New Image", new_image);

       /// Wait until user press some key
       waitKey();
       return 0;
   }

Explanation
============

#. We begin by creating parameters to save :math:`\alpha` and :math:`\beta` to be entered by the user:

   .. code-block:: cpp

      double alpha;
      int beta;


#. We load an image using :imread:`imread <>` and save it in a Mat object:

   .. code-block:: cpp

      Mat image = imread( argv[1] );

#. Now, since we will make some transformations to this image, we need a new Mat object to store it. Also, we want this to have the following features:

   .. container:: enumeratevisibleitemswithsquare

      * Initial pixel values equal to zero
      * Same size and type as the original image

   .. code-block:: cpp

      Mat new_image = Mat::zeros( image.size(), image.type() );

   We observe that :mat_zeros:`Mat::zeros <>` returns a Matlab-style zero initializer based on *image.size()* and *image.type()*

#. Now, to perform the operation :math:`g(i,j) = \alpha \cdot f(i,j) + \beta` we will access to each pixel in image. Since we are operating with RGB images, we will have three values per pixel (R, G and B), so we will also access them separately. Here is the piece of code:

   .. code-block:: cpp

      for( int y = 0; y < image.rows; y++ ) {
          for( int x = 0; x < image.cols; x++ ) {
              for( int c = 0; c < 3; c++ ) {
                  new_image.at<Vec3b>(y,x)[c] =
                    saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
              }
          }
      }

   Notice the following:

   .. container:: enumeratevisibleitemswithsquare

      * To access each pixel in the images we are using this syntax: *image.at<Vec3b>(y,x)[c]* where *y* is the row, *x* is the column and *c* is R, G or B (0, 1 or 2).

      * Since the operation :math:`\alpha \cdot p(i,j) + \beta` can give values out of range or not integers (if :math:`\alpha` is float), we use :saturate_cast:`saturate_cast <>` to make sure the values are valid.


#. Finally, we create windows and show the images, the usual way.

   .. code-block:: cpp

      namedWindow("Original Image", 1);
      namedWindow("New Image", 1);

      imshow("Original Image", image);
      imshow("New Image", new_image);

      waitKey(0);

.. note::

   Instead of using the **for** loops to access each pixel, we could have simply used this command:

   .. code-block:: cpp

      image.convertTo(new_image, -1, alpha, beta);

   where :convert_to:`convertTo <>` would effectively perform *new_image = a*image + beta*. However, we wanted to show you how to access each pixel. In any case, both methods give the same result.

Result
=======

* Running our code and using :math:`\alpha = 2.2` and :math:`\beta = 50`

  .. code-block:: bash

     $ ./BasicLinearTransforms lena.jpg
     Basic Linear Transforms
     -------------------------
     * Enter the alpha value [1.0-3.0]: 2.2
     * Enter the beta value [0-100]: 50

* We get this:

  .. image:: images/Basic_Linear_Transform_Tutorial_Result_0.jpg
     :alt: Basic Linear Transform - Final Result
     :align: center
