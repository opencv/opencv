.. _sobel_derivatives:

Sobel Derivatives
******************


Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Use the OpenCV function :sobel:`Sobel <>` to calculate the derivatives from an image.
   * Use the OpenCV function :scharr:`Scharr <>` to calculate a more accurate derivative for a kernel of size :math:`3 \cdot 3`  
  
Theory
========

.. note::
   The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.


#. In the last two tutorials we have seen applicative examples of convolutions. One of the most important convolutions is the computation of derivatives in an image (or an approximation to them).

#. Why may be important the calculus of the derivatives in an image? Let's imagine we want to detect the *edges* present in the image. For instance:


   .. image:: images/Sobel_Derivatives_Tutorial_Theory_0.jpg
           :alt: How intensity changes in an edge
           :align: center
 
   You can easily notice that in an *edge*, the pixel intensity *changes* in a notorious way. A good way to express *changes* is by using *derivatives*. A high change in gradient indicates a major change in the image. 

#. To be more graphical, let's assume we have a 1D-image. An edge is shown by the "jump" in intensity in the plot below:

   .. image:: images/Sobel_Derivatives_Tutorial_Theory_Intensity_Function.jpg
           :alt:  Intensity Plot for an edge
           :align: center

#. The edge "jump" can be seen more easily if we take the first derivative (actually, here appears as a maximum)

   .. image:: images/Sobel_Derivatives_Tutorial_Theory_dIntensity_Function.jpg
           :alt:  First derivative of Intensity - Plot for an edge
           :align: center

#. So, from the explanation above, we can deduce that a method to detect edges in an image can be performed by locating pixel locations where the gradient is higher than its neighbors (or to generalize, higher than a threshold).

#. More detailed explanation, please refer to **Learning OpenCV** by Bradski and Kaehler

Sobel Operator
---------------

#. The Sobel Operator is a discrete differentiation operator. It computes an approximation of the gradient of an image intensity function. 

#. The Sobel Operator combines Gaussian smoothing and differentiation.  

Formulation
^^^^^^^^^^^^
Assuming that the image to  be operated is :math:`I`:

#. We calculate two derivatives:

   a. **Horizontal changes**: This is computed by convolving :math:`I` with a kernel :math:`G_{x}` with odd size. For example for a kernel size of 3, :math:`G_{x}` would be computed as:

      .. math::
   
         G_{x} = \begin{bmatrix}
         -1 & 0 & +1  \\
         -2 & 0 & +2  \\
         -1 & 0 & +1 
         \end{bmatrix} * I

   b. **Vertical changes**: This is computed by convolving :math:`I` with a kernel :math:`G_{y}` with odd size. For example for a kernel size of 3, :math:`G_{y}` would be computed as:

      .. math::
   
         G_{y} = \begin{bmatrix}
         -1 & -2 & -1  \\
         0 & 0 & 0  \\
         +1 & +2 & +1 
         \end{bmatrix} * I

#. At each point of the image we calculate an approximation of the *gradient* in that point by combining both results above:

    .. math::

       G = \sqrt{ G_{x}^{2} + G_{y}^{2} }

   Although sometimes the following simpler equation is used:

   .. math::
      
      G = |G_{x}| + |G_{y}|


.. note::

   When the size of the kernel is :math:`3`, the Sobel kernel shown above may produce noticeable inaccuracies (after all, Sobel is only an approximation of the derivative). OpenCV addresses this inaccuracy for kernels of size 3 by using the :scharr:`Scharr <>` function. This is as fast but more accurate than the standar Sobel function. It implements the following kernels:

      .. math::

         G_{x} = \begin{bmatrix}
         -3 & 0 & +3  \\
         -10 & 0 & +10  \\
         -3 & 0 & +3 
         \end{bmatrix} 
   
         G_{y} = \begin{bmatrix}
         -3 & -10 & -3  \\
         0 & 0 & 0  \\
         +3 & +10 & +3 
         \end{bmatrix} 

  You can check out more information of this function in the OpenCV reference (:scharr:`Scharr <>`). Also, in the sample code below, you will notice that above the code for :sobel:`Sobel <>` function there is also code for the :scharr:`Scharr <>` function commented. Uncommenting it (and obviously commenting the Sobel stuff) should give you an idea of how this function works.

Code
=====

#. **What does this program do?**
 
   * Applies the *Sobel Operator* and generates as output an image with the detected *edges* bright on a darker background.
 
#. The tutorial code's is shown lines below. You can also download it from `here <https://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp>`_

.. code-block:: cpp 

   #include "opencv2/imgproc/imgproc.hpp"
   #include "opencv2/highgui/highgui.hpp"
   #include <stdlib.h>
   #include <stdio.h>

   using namespace cv;

   /** @function main */
   int main( int argc, char** argv )
   {

     Mat src, src_gray;
     Mat grad; 
     char* window_name = "Sobel Demo - Simple Edge Detector";
     int scale = 1;
     int delta = 0;
     int ddepth = CV_16S;

     int c;

     /// Load an image
     src = imread( argv[1] );

     if( !src.data )
     { return -1; }

     GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

     /// Convert it to gray
     cvtColor( src, src_gray, CV_RGB2GRAY );

     /// Create window
     namedWindow( window_name, CV_WINDOW_AUTOSIZE );

     /// Generate grad_x and grad_y
     Mat grad_x, grad_y;
     Mat abs_grad_x, abs_grad_y;
 
     /// Gradient X
     //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
     Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );   
     convertScaleAbs( grad_x, abs_grad_x );

     /// Gradient Y  
     //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
     Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );   
     convertScaleAbs( grad_y, abs_grad_y );

     /// Total Gradient (approximate)
     addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

     imshow( window_name, grad );

     waitKey(0);

     return 0;
     }


Explanation
=============

#. First we declare the variables we are going to use:

   ..  code-block:: cpp

       Mat src, src_gray;
       Mat grad; 
       char* window_name = "Sobel Demo - Simple Edge Detector";
       int scale = 1;
       int delta = 0;
       int ddepth = CV_16S;

#. As usual we load our source image *src*:

   .. code-block:: cpp

     src = imread( argv[1] );
  
     if( !src.data )
     { return -1; }

#. First, we apply a :gaussian_blur:`GaussianBlur <>` to our image to reduce the noise ( kernel size = 3 )
 
   .. code-block:: cpp

      GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

#. Now we convert our filtered image to grayscale:

   .. code-block:: cpp

      cvtColor( src, src_gray, CV_RGB2GRAY );

#. Second, we calculate the "*derivatives*" in *x* and *y* directions. For this, we use the function :sobel:`Sobel <>` as shown below:
 
   .. code-block:: cpp

      Mat grad_x, grad_y;
      Mat abs_grad_x, abs_grad_y;
 
      /// Gradient X
      Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );   
      /// Gradient Y  
      Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );   

   The function takes the following arguments:

   * *src_gray*: In our example, the input image. Here it is *CV_8U* 
   * *grad_x*/*grad_y*: The output image. 
   * *ddepth*: The depth of the output image. We set it to *CV_16S* to avoid overflow.
   * *x_order*: The order of the derivative in **x** direction. 
   * *y_order*: The order of the derivative in **y** direction. 
   * *scale*, *delta* and *BORDER_DEFAULT*: We use default values.

   Notice that to calculate the gradient in *x* direction we use: :math:`x_{order}= 1` and :math:`y_{order} = 0`. We do analogously for the *y* direction. 

#. We convert our partial results back to *CV_8U*:

   .. code-block:: cpp

      convertScaleAbs( grad_x, abs_grad_x );
      convertScaleAbs( grad_y, abs_grad_y );
 

#. Finally, we try to approximate the *gradient* by adding both directional gradients (note that this is not an exact calculation at all! but it is good for our purposes).

   .. code-block:: cpp

     addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

#. Finally, we show our result:

   .. code-block:: cpp

      imshow( window_name, grad );



Results
========

#. Here is the output of applying our basic detector to *lena.jpg*:
   

   .. image:: images/Sobel_Derivatives_Tutorial_Result.jpg
           :alt: Result of applying Sobel operator to lena.jpg
           :width: 300pt
           :align: center
