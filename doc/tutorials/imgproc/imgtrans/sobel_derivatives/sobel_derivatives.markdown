Sobel Derivatives {#tutorial_sobel_derivatives}
=================

@tableofcontents

@prev_tutorial{tutorial_copyMakeBorder}
@next_tutorial{tutorial_laplace_operator}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function **Sobel()** to calculate the derivatives from an image.
-   Use the OpenCV function **Scharr()** to calculate a more accurate derivative for a kernel of
    size \f$3 \cdot 3\f$

Theory
------

@note The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

-#  In the last two tutorials we have seen applicative examples of convolutions. One of the most
    important convolutions is the computation of derivatives in an image (or an approximation to
    them).
-#  Why may be important the calculus of the derivatives in an image? Let's imagine we want to
    detect the *edges* present in the image. For instance:

    ![](images/Sobel_Derivatives_Tutorial_Theory_0.jpg)

    You can easily notice that in an *edge*, the pixel intensity *changes* in a notorious way. A
    good way to express *changes* is by using *derivatives*. A high change in gradient indicates a
    major change in the image.

-#  To be more graphical, let's assume we have a 1D-image. An edge is shown by the "jump" in
    intensity in the plot below:

    ![](images/Sobel_Derivatives_Tutorial_Theory_Intensity_Function.jpg)

-#  The edge "jump" can be seen more easily if we take the first derivative (actually, here appears
    as a maximum)

    ![](images/Sobel_Derivatives_Tutorial_Theory_dIntensity_Function.jpg)

-#  So, from the explanation above, we can deduce that a method to detect edges in an image can be
    performed by locating pixel locations where the gradient is higher than its neighbors (or to
    generalize, higher than a threshold).
-#  More detailed explanation, please refer to **Learning OpenCV** by Bradski and Kaehler

### Sobel Operator

-#  The Sobel Operator is a discrete differentiation operator. It computes an approximation of the
    gradient of an image intensity function.
-#  The Sobel Operator combines Gaussian smoothing and differentiation.

### Formulation

Assuming that the image to be operated is \f$I\f$:

-#  We calculate two derivatives:
    -#  **Horizontal changes**: This is computed by convolving \f$I\f$ with a kernel \f$G_{x}\f$ with odd
        size. For example for a kernel size of 3, \f$G_{x}\f$ would be computed as:

        \f[G_{x} = \begin{bmatrix}
        -1 & 0 & +1  \\
        -2 & 0 & +2  \\
        -1 & 0 & +1
        \end{bmatrix} * I\f]

    -#  **Vertical changes**: This is computed by convolving \f$I\f$ with a kernel \f$G_{y}\f$ with odd
        size. For example for a kernel size of 3, \f$G_{y}\f$ would be computed as:

        \f[G_{y} = \begin{bmatrix}
        -1 & -2 & -1  \\
        0 & 0 & 0  \\
        +1 & +2 & +1
        \end{bmatrix} * I\f]

-#  At each point of the image we calculate an approximation of the *gradient* in that point by
    combining both results above:

    \f[G = \sqrt{ G_{x}^{2} + G_{y}^{2} }\f]

    Although sometimes the following simpler equation is used:

    \f[G = |G_{x}| + |G_{y}|\f]

@note
    When the size of the kernel is `3`, the Sobel kernel shown above may produce noticeable
    inaccuracies (after all, Sobel is only an approximation of the derivative). OpenCV addresses
    this inaccuracy for kernels of size 3 by using the **Scharr()** function. This is as fast
    but more accurate than the standard Sobel function. It implements the following kernels:
    \f[G_{x} = \begin{bmatrix}
    -3 & 0 & +3  \\
    -10 & 0 & +10  \\
    -3 & 0 & +3
    \end{bmatrix}\f]\f[G_{y} = \begin{bmatrix}
    -3 & -10 & -3  \\
    0 & 0 & 0  \\
    +3 & +10 & +3
    \end{bmatrix}\f]
@note
    You can check out more information of this function in the OpenCV reference - **Scharr()** .
    Also, in the sample code below, you will notice that above the code for **Sobel()** function
    there is also code for the **Scharr()** function commented. Uncommenting it (and obviously
    commenting the Sobel stuff) should give you an idea of how this function works.

Code
----

-#  **What does this program do?**
    -   Applies the *Sobel Operator* and generates as output an image with the detected *edges*
        bright on a darker background.

-#  The tutorial code's is shown lines below.

@add_toggle_cpp
You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp)
@include samples/cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp
@end_toggle

@add_toggle_java
You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/java/tutorial_code/ImgTrans/SobelDemo/SobelDemo.java)
@include samples/java/tutorial_code/ImgTrans/SobelDemo/SobelDemo.java
@end_toggle

@add_toggle_python
You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/python/tutorial_code/ImgTrans/SobelDemo/sobel_demo.py)
@include samples/python/tutorial_code/ImgTrans/SobelDemo/sobel_demo.py
@end_toggle

Explanation
-----------

### Declare variables

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp variables

### Load source image

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp load

### Reduce noise

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp reduce_noise

### Grayscale

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp convert_to_gray

### Sobel Operator

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp sobel

-   We calculate the "derivatives" in *x* and *y* directions. For this, we use the
    function **Sobel()** as shown below:
    The function takes the following arguments:

    -   *src_gray*: In our example, the input image. Here it is *CV_8U*
    -   *grad_x* / *grad_y* : The output image.
    -   *ddepth*: The depth of the output image. We set it to *CV_16S* to avoid overflow.
    -   *x_order*: The order of the derivative in **x** direction.
    -   *y_order*: The order of the derivative in **y** direction.
    -   *scale*, *delta* and *BORDER_DEFAULT*: We use default values.

    Notice that to calculate the gradient in *x* direction we use: \f$x_{order}= 1\f$ and
    \f$y_{order} = 0\f$. We do analogously for the *y* direction.

### Convert output to a CV_8U image

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp convert

### Gradient

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp blend

We try to approximate the *gradient* by adding both directional gradients (note that
this is not an exact calculation at all! but it is good for our purposes).

### Show results

@snippet cpp/tutorial_code/ImgTrans/Sobel_Demo.cpp display

Results
-------

-#  Here is the output of applying our basic detector to *lena.jpg*:

    ![](images/Sobel_Derivatives_Tutorial_Result.jpg)
