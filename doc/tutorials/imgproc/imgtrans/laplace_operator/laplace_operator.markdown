Laplace Operator {#tutorial_laplace_operator}
================

@prev_tutorial{tutorial_sobel_derivatives}
@next_tutorial{tutorial_canny_detector}

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function **Laplacian()** to implement a discrete analog of the *Laplacian
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

-#  The Laplacian operator is implemented in OpenCV by the function **Laplacian()** . In fact,
    since the Laplacian uses the gradient of images, it calls internally the *Sobel* operator to
    perform its computation.

Code
----

-#  **What does this program do?**
    -   Loads an image
    -   Remove noise by applying a Gaussian blur and then convert the original image to grayscale
    -   Applies a Laplacian operator to the grayscale image and stores the output image
    -   Display the result in a window

@add_toggle_cpp
-#  The tutorial code's is shown lines below. You can also download it from
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp)
    @include samples/cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp
@end_toggle

@add_toggle_java
-#  The tutorial code's is shown lines below. You can also download it from
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java)
    @include samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java
@end_toggle

@add_toggle_python
-#  The tutorial code's is shown lines below. You can also download it from
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py)
    @include samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py
@end_toggle

Explanation
-----------

#### Declare variables

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp variables
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java variables
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py variables
@end_toggle

#### Load source image

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp load
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java load
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py load
@end_toggle

#### Reduce noise

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp reduce_noise
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java reduce_noise
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py reduce_noise
@end_toggle

#### Grayscale

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp convert_to_gray
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java convert_to_gray
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py convert_to_gray
@end_toggle

#### Laplacian operator

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp laplacian
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java laplacian
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py laplacian
@end_toggle

-   The arguments are:
    -   *src_gray*: The input image.
    -   *dst*: Destination (output) image
    -   *ddepth*: Depth of the destination image. Since our input is *CV_8U* we define *ddepth* =
        *CV_16S* to avoid overflow
    -   *kernel_size*: The kernel size of the Sobel operator to be applied internally. We use 3 in
        this example.
    -   *scale*, *delta* and *BORDER_DEFAULT*: We leave them as default values.

#### Convert output to a *CV_8U* image

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp convert
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java convert
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py convert
@end_toggle

#### Display the result

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/Laplace_Demo.cpp display
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/LaPlace/LaplaceDemo.java display
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/LaPlace/laplace_demo.py display
@end_toggle

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
