Laplace Operator {#tutorial_laplace_operator}
================

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::Laplacian to implement a discrete analog of the *Laplacian
    operator*.

Theory
------

-#  In the previous tutorial we learned how to use the *Sobel Operator*. It was based on the fact
    that in the edge area, the pixel intensity shows a "jump" or a high variation of intensity.
    Getting the first derivative of the intensity, we observed that an edge is characterized by a
    maximum, as it can be seen in the figure:

    ![](images/Laplace_Operator_Tutorial_Theory_Previous.jpg)

-#  And...what happens if we take the second derivative?

    ![](images/Laplace_Operator_Tutorial_Theory_ddIntensity.jpg)

    You can observe that the second derivative is zero! So, we can also use this criterion to
    attempt to detect edges in an image. However, note that zeros will not only appear in edges
    (they can actually appear in other meaningless locations); this can be solved by applying
    filtering where needed.

### Laplacian Operator

-#  From the explanation above, we deduce that the second derivative can be used to *detect edges*.
    Since images are "*2D*", we would need to take the derivative in both dimensions. Here, the
    Laplacian operator comes handy.
-#  The *Laplacian operator* is defined by:

\f[Laplace(f) = \dfrac{\partial^{2} f}{\partial x^{2}} + \dfrac{\partial^{2} f}{\partial y^{2}}\f]

-#  The Laplacian operator is implemented in OpenCV by the function @ref cv::Laplacian . In fact,
    since the Laplacian uses the gradient of images, it calls internally the *Sobel* operator to
    perform its computation.

Code
----

-#  **What does this program do?**
    -   Loads an image
    -   Remove noise by applying a Gaussian blur and then convert the original image to grayscale
    -   Applies a Laplacian operator to the grayscale image and stores the output image
    -   Display the result in a window

-#  The tutorial code's is shown lines below. You can also download it from
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp)
    @include samples/cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp

Explanation
-----------

-#  Create some needed variables:
    @code{.cpp}
    Mat src, src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    char* window_name = "Laplace Demo";
    @endcode
-#  Loads the source image:
    @code{.cpp}
    src = imread( argv[1] );

    if( !src.data )
      { return -1; }
    @endcode
-#  Apply a Gaussian blur to reduce noise:
    @code{.cpp}
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
    @endcode
-#  Convert the image to grayscale using @ref cv::cvtColor
    @code{.cpp}
    cvtColor( src, src_gray, COLOR_RGB2GRAY );
    @endcode
-#  Apply the Laplacian operator to the grayscale image:
    @code{.cpp}
    Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    @endcode
    where the arguments are:

    -   *src_gray*: The input image.
    -   *dst*: Destination (output) image
    -   *ddepth*: Depth of the destination image. Since our input is *CV_8U* we define *ddepth* =
        *CV_16S* to avoid overflow
    -   *kernel_size*: The kernel size of the Sobel operator to be applied internally. We use 3 in
        this example.
    -   *scale*, *delta* and *BORDER_DEFAULT*: We leave them as default values.

-#  Convert the output from the Laplacian operator to a *CV_8U* image:
    @code{.cpp}
    convertScaleAbs( dst, abs_dst );
    @endcode
-#  Display the result in a window:
    @code{.cpp}
    imshow( window_name, abs_dst );
    @endcode

Results
-------

-#  After compiling the code above, we can run it giving as argument the path to an image. For
    example, using as an input:

    ![](images/Laplace_Operator_Tutorial_Original_Image.jpg)

-#  We obtain the following result. Notice how the trees and the silhouette of the cow are
    approximately well defined (except in areas in which the intensity are very similar, i.e. around
    the cow's head). Also, note that the roof of the house behind the trees (right side) is
    notoriously marked. This is due to the fact that the contrast is higher in that region.

    ![](images/Laplace_Operator_Tutorial_Result.jpg)
