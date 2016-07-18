Adding (blending) two images using OpenCV {#tutorial_adding_images}
=========================================

Goal
----

In this tutorial you will learn:

-   what is *linear blending* and why it is useful;
-   how to add two images using @ref cv::addWeighted

Theory
------

@note
   The explanation below belongs to the book [Computer Vision: Algorithms and
    Applications](http://szeliski.org/Book/) by Richard Szeliski

From our previous tutorial, we know already a bit of *Pixel operators*. An interesting dyadic
(two-input) operator is the *linear blend operator*:

\f[g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)\f]

By varying \f$\alpha\f$ from \f$0 \rightarrow 1\f$ this operator can be used to perform a temporal
*cross-dissolve* between two images or videos, as seen in slide shows and film productions (cool,
eh?)

Source Code
-----------

Download the source code from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/core/AddingImages/AddingImages.cpp).
@include cpp/tutorial_code/core/AddingImages/AddingImages.cpp

Explanation
-----------

-#  Since we are going to perform:

    \f[g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)\f]

    We need two source images (\f$f_{0}(x)\f$ and \f$f_{1}(x)\f$). So, we load them in the usual way:
    @snippet cpp/tutorial_code/core/AddingImages/AddingImages.cpp load

    **warning**

    Since we are *adding* *src1* and *src2*, they both have to be of the same size (width and
    height) and type.

-#  Now we need to generate the `g(x)` image. For this, the function @ref cv::addWeighted comes quite handy:
    @snippet cpp/tutorial_code/core/AddingImages/AddingImages.cpp blend_images
    since @ref cv::addWeighted  produces:
    \f[dst = \alpha \cdot src1 + \beta \cdot src2 + \gamma\f]
    In this case, `gamma` is the argument \f$0.0\f$ in the code above.

-#  Create windows, show the images and wait for the user to end the program.
    @snippet cpp/tutorial_code/core/AddingImages/AddingImages.cpp display

Result
------

![](images/Adding_Images_Tutorial_Result_Big.jpg)
