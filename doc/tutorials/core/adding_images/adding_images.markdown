Adding (blending) two images using OpenCV {#tutorial_adding_images}
=========================================

@prev_tutorial{tutorial_mat_operations}
@next_tutorial{tutorial_basic_linear_transform}

Goal
----

In this tutorial you will learn:

-   what is *linear blending* and why it is useful;
-   how to add two images using **addWeighted()**

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

@add_toggle_cpp
Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/core/AddingImages/AddingImages.cpp).
@include cpp/tutorial_code/core/AddingImages/AddingImages.cpp
@end_toggle

@add_toggle_java
Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/master/samples/java/tutorial_code/core/AddingImages/AddingImages.java).
@include java/tutorial_code/core/AddingImages/AddingImages.java
@end_toggle

@add_toggle_python
Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/master/samples/python/tutorial_code/core/AddingImages/adding_images.py).
@include python/tutorial_code/core/AddingImages/adding_images.py
@end_toggle

Explanation
-----------

Since we are going to perform:

\f[g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)\f]

We need two source images (\f$f_{0}(x)\f$ and \f$f_{1}(x)\f$). So, we load them in the usual way:
@add_toggle_cpp
@snippet cpp/tutorial_code/core/AddingImages/AddingImages.cpp load
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/AddingImages/AddingImages.java load
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/AddingImages/adding_images.py load
@end_toggle

We used the following images: [LinuxLogo.jpg](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/LinuxLogo.jpg) and [WindowsLogo.jpg](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/WindowsLogo.jpg)

@warning Since we are *adding* *src1* and *src2*, they both have to be of the same size
(width and height) and type.

Now we need to generate the `g(x)` image. For this, the function **addWeighted()** comes quite handy:

@add_toggle_cpp
@snippet cpp/tutorial_code/core/AddingImages/AddingImages.cpp blend_images
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/AddingImages/AddingImages.java blend_images
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/AddingImages/adding_images.py blend_images
Numpy version of above line (but cv function is around 2x faster):
\code{.py}
    dst = np.uint8(alpha*(img1)+beta*(img2))
\endcode
@end_toggle

since **addWeighted()**  produces:
\f[dst = \alpha \cdot src1 + \beta \cdot src2 + \gamma\f]
In this case, `gamma` is the argument \f$0.0\f$ in the code above.

Create windows, show the images and wait for the user to end the program.
@add_toggle_cpp
@snippet cpp/tutorial_code/core/AddingImages/AddingImages.cpp display
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/AddingImages/AddingImages.java display
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/AddingImages/adding_images.py display
@end_toggle

Result
------

![](images/Adding_Images_Tutorial_Result_Big.jpg)
