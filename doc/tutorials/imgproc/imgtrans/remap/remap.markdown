Remapping {#tutorial_remap}
=========

@tableofcontents

@prev_tutorial{tutorial_generalized_hough_ballard_guil}
@next_tutorial{tutorial_warp_affine}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

a.  Use the OpenCV function @ref cv::remap to implement simple remapping routines.

Theory
------

### What is remapping?

-   It is the process of taking pixels from one place in the image and locating them in another
    position in a new image.
-   To accomplish the mapping process, it might be necessary to do some interpolation for
    non-integer pixel locations, since there will not always be a one-to-one-pixel correspondence
    between source and destination images.
-   We can express the remap for every pixel location \f$(x,y)\f$ as:

    \f[g(x,y) = f ( h(x,y) )\f]

    where \f$g()\f$ is the remapped image, \f$f()\f$ the source image and \f$h(x,y)\f$ is the mapping function
    that operates on \f$(x,y)\f$.

-   Let's think in a quick example. Imagine that we have an image \f$I\f$ and, say, we want to do a
    remap such that:

    \f[h(x,y) = (I.cols - x, y )\f]

    What would happen? It is easily seen that the image would flip in the \f$x\f$ direction. For
    instance, consider the input image:

    ![](images/Remap_Tutorial_Theory_0.jpg)

    observe how the red circle changes positions with respect to \f$x\f$ (considering \f$x\f$ the horizontal
    direction):

    ![](images/Remap_Tutorial_Theory_1.jpg)

-   In OpenCV, the function @ref cv::remap offers a simple remapping implementation.

Code
----

-   **What does this program do?**
    -   Loads an image
    -   Each second, apply 1 of 4 different remapping processes to the image and display them
        indefinitely in a window.
    -   Wait for the user to exit the program

@add_toggle_cpp
-   The tutorial code is shown lines below. You can also download it from
    [here](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/ImgTrans/Remap_Demo.cpp)
    @include samples/cpp/tutorial_code/ImgTrans/Remap_Demo.cpp
@end_toggle

@add_toggle_java
-   The tutorial code is shown lines below. You can also download it from
    [here](https://github.com/opencv/opencv/tree/5.x/samples/java/tutorial_code/ImgTrans/remap/RemapDemo.java)
    @include samples/java/tutorial_code/ImgTrans/remap/RemapDemo.java
@end_toggle

@add_toggle_python
-   The tutorial code is shown lines below. You can also download it from
    [here](https://github.com/opencv/opencv/tree/5.x/samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py)
    @include samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py
@end_toggle

Explanation
-----------

-   Load an image:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Remap_Demo.cpp Load
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/remap/RemapDemo.java Load
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py Load
    @end_toggle

-   Create the destination image and the two mapping matrices (for x and y )

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Remap_Demo.cpp Create
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/remap/RemapDemo.java Create
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py Create
    @end_toggle

-   Create a window to display results

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Remap_Demo.cpp Window
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/remap/RemapDemo.java Window
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py Window
    @end_toggle

-   Establish a loop. Each 1000 ms we update our mapping matrices (*mat_x* and *mat_y*) and apply
    them to our source image:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Remap_Demo.cpp Loop
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/remap/RemapDemo.java Loop
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py Loop
    @end_toggle

-   The function that applies the remapping is @ref cv::remap . We give the following arguments:
    -   **src**: Source image
    -   **dst**: Destination image of same size as *src*
    -   **map_x**: The mapping function in the x direction. It is equivalent to the first component
        of \f$h(i,j)\f$
    -   **map_y**: Same as above, but in y direction. Note that *map_y* and *map_x* are both of
        the same size as *src*
    -   **INTER_LINEAR**: The type of interpolation to use for non-integer pixels. This is by
        default.
    -   **BORDER_CONSTANT**: Default

    How do we update our mapping matrices *mat_x* and *mat_y*? Go on reading:

-   **Updating the mapping matrices:** We are going to perform 4 different mappings:
    -#  Reduce the picture to half its size and will display it in the middle:
        \f[h(i,j) = ( 2 \times i - src.cols/2  + 0.5, 2 \times j - src.rows/2  + 0.5)\f]
        for all pairs \f$(i,j)\f$ such that: \f$\dfrac{src.cols}{4}<i<\dfrac{3 \cdot src.cols}{4}\f$ and
        \f$\dfrac{src.rows}{4}<j<\dfrac{3 \cdot src.rows}{4}\f$
    -#  Turn the image upside down: \f$h( i, j ) = (i, src.rows - j)\f$
    -#  Reflect the image from left to right: \f$h(i,j) = ( src.cols - i, j )\f$
    -#  Combination of b and c: \f$h(i,j) = ( src.cols - i, src.rows - j )\f$

This is expressed in the following snippet. Here, *map_x* represents the first coordinate of
*h(i,j)* and *map_y* the second coordinate.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/Remap_Demo.cpp Update
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/remap/RemapDemo.java Update
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py Update
@end_toggle

Result
------

-#  After compiling the code above, you can execute it giving as argument an image path. For
    instance, by using the following image:

    ![](images/Remap_Tutorial_Original_Image.jpg)

-#  This is the result of reducing it to half the size and centering it:

    ![](images/Remap_Tutorial_Result_0.jpg)

-#  Turning it upside down:

    ![](images/Remap_Tutorial_Result_1.jpg)

-#  Reflecting it in the x direction:

    ![](images/Remap_Tutorial_Result_2.jpg)

-#  Reflecting it in both directions:

    ![](images/Remap_Tutorial_Result_3.jpg)
