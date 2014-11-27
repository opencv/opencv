Sobel Derivatives {#tutorial_sobel_derivatives}
=================

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::Sobel to calculate the derivatives from an image.
-   Use the OpenCV function @ref cv::Scharr to calculate a more accurate derivative for a kernel of
    size \f$3 \cdot 3\f$

Theory
------

@note The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

1.  In the last two tutorials we have seen applicative examples of convolutions. One of the most
    important convolutions is the computation of derivatives in an image (or an approximation to
    them).
2.  Why may be important the calculus of the derivatives in an image? Let's imagine we want to
    detect the *edges* present in the image. For instance:

    ![image](images/Sobel_Derivatives_Tutorial_Theory_0.jpg)

    You can easily notice that in an *edge*, the pixel intensity *changes* in a notorious way. A
    good way to express *changes* is by using *derivatives*. A high change in gradient indicates a
    major change in the image.

3.  To be more graphical, let's assume we have a 1D-image. An edge is shown by the "jump" in
    intensity in the plot below:

    ![image](images/Sobel_Derivatives_Tutorial_Theory_Intensity_Function.jpg)

4.  The edge "jump" can be seen more easily if we take the first derivative (actually, here appears
    as a maximum)

    ![image](images/Sobel_Derivatives_Tutorial_Theory_dIntensity_Function.jpg)

5.  So, from the explanation above, we can deduce that a method to detect edges in an image can be
    performed by locating pixel locations where the gradient is higher than its neighbors (or to
    generalize, higher than a threshold).
6.  More detailed explanation, please refer to **Learning OpenCV** by Bradski and Kaehler

### Sobel Operator

1.  The Sobel Operator is a discrete differentiation operator. It computes an approximation of the
    gradient of an image intensity function.
2.  The Sobel Operator combines Gaussian smoothing and differentiation.

#### Formulation

Assuming that the image to be operated is \f$I\f$:

1.  We calculate two derivatives:
    a.  **Horizontal changes**: This is computed by convolving \f$I\f$ with a kernel \f$G_{x}\f$ with odd
        size. For example for a kernel size of 3, \f$G_{x}\f$ would be computed as:

        \f[G_{x} = \begin{bmatrix}
        -1 & 0 & +1  \\
        -2 & 0 & +2  \\
        -1 & 0 & +1
        \end{bmatrix} * I\f]

    b.  **Vertical changes**: This is computed by convolving \f$I\f$ with a kernel \f$G_{y}\f$ with odd
        size. For example for a kernel size of 3, \f$G_{y}\f$ would be computed as:

        \f[G_{y} = \begin{bmatrix}
        -1 & -2 & -1  \\
        0 & 0 & 0  \\
        +1 & +2 & +1
        \end{bmatrix} * I\f]

2.  At each point of the image we calculate an approximation of the *gradient* in that point by
    combining both results above:

    \f[G = \sqrt{ G_{x}^{2} + G_{y}^{2} }\f]

    Although sometimes the following simpler equation is used:

    \f[G = |G_{x}| + |G_{y}|\f]

@note
   When the size of the kernel is @ref cv::3\`, the Sobel kernel shown above may produce noticeable
    inaccuracies (after all, Sobel is only an approximation of the derivative). OpenCV addresses
    this inaccuracy for kernels of size 3 by using the :scharr:\`Scharr function. This is as fast
    but more accurate than the standar Sobel function. It implements the following kernels:

    \f[G_{x} = \begin{bmatrix}
    -3 & 0 & +3  \\
    -10 & 0 & +10  \\
    -3 & 0 & +3
    \end{bmatrix}\f]\f[G_{y} = \begin{bmatrix}
    -3 & -10 & -3  \\
    0 & 0 & 0  \\
    +3 & +10 & +3
    \end{bmatrix}\f]

You can check out more information of this function in the OpenCV reference (@ref cv::Scharr ).
Also, in the sample code below, you will notice that above the code for @ref cv::Sobel function
there is also code for the @ref cv::Scharr function commented. Uncommenting it (and obviously
commenting the Sobel stuff) should give you an idea of how this function works.

Code
----

1.  **What does this program do?**
    -   Applies the *Sobel Operator* and generates as output an image with the detected *edges*
        bright on a darker background.

2.  The tutorial code's is shown lines below. You can also download it from
    [here](https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp)
@code{.cpp}
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/* @function main */
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
  cvtColor( src, src_gray, COLOR_RGB2GRAY );

  /// Create window
  namedWindow( window_name, WINDOW_AUTOSIZE );

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
@endcode
Explanation
-----------

1.  First we declare the variables we are going to use:
    @code{.cpp}
    Mat src, src_gray;
    Mat grad;
    char* window_name = "Sobel Demo - Simple Edge Detector";
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    @endcode
2.  As usual we load our source image *src*:
    @code{.cpp}
    src = imread( argv[1] );

    if( !src.data )
    { return -1; }
    @endcode
3.  First, we apply a @ref cv::GaussianBlur to our image to reduce the noise ( kernel size = 3 )
    @code{.cpp}
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
    @endcode
4.  Now we convert our filtered image to grayscale:
    @code{.cpp}
    cvtColor( src, src_gray, COLOR_RGB2GRAY );
    @endcode
5.  Second, we calculate the "*derivatives*" in *x* and *y* directions. For this, we use the
    function @ref cv::Sobel as shown below:
    @code{.cpp}
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    @endcode
    The function takes the following arguments:

    -   *src_gray*: In our example, the input image. Here it is *CV_8U*
    -   *grad_x*/*grad_y*: The output image.
    -   *ddepth*: The depth of the output image. We set it to *CV_16S* to avoid overflow.
    -   *x_order*: The order of the derivative in **x** direction.
    -   *y_order*: The order of the derivative in **y** direction.
    -   *scale*, *delta* and *BORDER_DEFAULT*: We use default values.

    Notice that to calculate the gradient in *x* direction we use: \f$x_{order}= 1\f$ and
    \f$y_{order} = 0\f$. We do analogously for the *y* direction.

6.  We convert our partial results back to *CV_8U*:
    @code{.cpp}
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    @endcode
7.  Finally, we try to approximate the *gradient* by adding both directional gradients (note that
    this is not an exact calculation at all! but it is good for our purposes).
    @code{.cpp}
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    @endcode
8.  Finally, we show our result:
    @code{.cpp}
    imshow( window_name, grad );
    @endcode
Results
-------

1.  Here is the output of applying our basic detector to *lena.jpg*:

    ![image](images/Sobel_Derivatives_Tutorial_Result.jpg)


