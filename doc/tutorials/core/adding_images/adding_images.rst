.. _Adding_Images:

Adding (blending) two images using OpenCV
*******************************************

Goal
=====

In this tutorial you will learn:

.. container:: enumeratevisibleitemswithsquare

   * what is *linear blending* and why it is useful;
   * how to add two images using :add_weighted:`addWeighted <>`

Theory
=======

.. note::

   The explanation below belongs to the book `Computer Vision: Algorithms and Applications <http://szeliski.org/Book/>`_  by Richard Szeliski

From our previous tutorial, we know already a bit of *Pixel operators*. An interesting dyadic (two-input) operator is the *linear blend operator*:

.. math::

   g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)

By varying :math:`\alpha` from :math:`0 \rightarrow 1` this operator can be used to perform a temporal *cross-dissolve* between two images or videos, as seen in slide shows and film productions (cool, eh?)

Code
=====

As usual, after the not-so-lengthy explanation, let's go to the code:

.. code-block:: cpp

   #include <cv.h>
   #include <highgui.h>
   #include <iostream>

   using namespace cv;

   int main( int argc, char** argv )
   {
    double alpha = 0.5; double beta; double input;

    Mat src1, src2, dst;

    /// Ask the user enter alpha
    std::cout<<" Simple Linear Blender "<<std::endl;
    std::cout<<"-----------------------"<<std::endl;
    std::cout<<"* Enter alpha [0-1]: ";
    std::cin>>input;

    /// We use the alpha provided by the user if it is between 0 and 1
    if( input >= 0.0 && input <= 1.0 )
      { alpha = input; }

    /// Read image ( same size, same type )
    src1 = imread("../../images/LinuxLogo.jpg");
    src2 = imread("../../images/WindowsLogo.jpg");

    if( !src1.data ) { printf("Error loading src1 \n"); return -1; }
    if( !src2.data ) { printf("Error loading src2 \n"); return -1; }

    /// Create Windows
    namedWindow("Linear Blend", 1);

    beta = ( 1.0 - alpha );
    addWeighted( src1, alpha, src2, beta, 0.0, dst);

    imshow( "Linear Blend", dst );

    waitKey(0);
    return 0;
   }

Explanation
============

#. Since we are going to perform:

   .. math::

      g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)

   We need two source images (:math:`f_{0}(x)` and :math:`f_{1}(x)`). So, we load them in the usual way:

   .. code-block:: cpp

      src1 = imread("../../images/LinuxLogo.jpg");
      src2 = imread("../../images/WindowsLogo.jpg");

   .. warning::

      Since we are *adding* *src1* and *src2*, they both have to be of the same size (width and height) and type.

#. Now we need to generate the :math:`g(x)` image. For this, the function :add_weighted:`addWeighted <>` comes quite handy:

   .. code-block:: cpp

      beta = ( 1.0 - alpha );
      addWeighted( src1, alpha, src2, beta, 0.0, dst);

   since :add_weighted:`addWeighted <>` produces:

   .. math::

      dst = \alpha \cdot src1 + \beta \cdot src2 + \gamma

   In this case, :math:`\gamma` is the argument :math:`0.0` in the code above.

#. Create windows, show the images and wait for the user to end the program.

Result
=======

.. image:: images/Adding_Images_Tutorial_Result_0.jpg
   :alt: Blending Images Tutorial - Final Result
   :align: center
