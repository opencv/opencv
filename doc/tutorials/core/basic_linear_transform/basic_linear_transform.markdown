Changing the contrast and brightness of an image! {#tutorial_basic_linear_transform}
=================================================

@prev_tutorial{tutorial_adding_images}
@next_tutorial{tutorial_discrete_fourier_transform}

Goal
----

In this tutorial you will learn how to:

-   Access pixel values
-   Initialize a matrix with zeros
-   Learn what @ref cv::saturate_cast does and why it is useful
-   Get some cool info about pixel transformations
-   Improve the brightness of an image on a practical example

Theory
------

@note
   The explanation below belongs to the book [Computer Vision: Algorithms and
    Applications](http://szeliski.org/Book/) by Richard Szeliski

### Image Processing

-   A general image processing operator is a function that takes one or more input images and
    produces an output image.
-   Image transforms can be seen as:
    -   Point operators (pixel transforms)
    -   Neighborhood (area-based) operators

### Pixel Transforms

-   In this kind of image processing transform, each output pixel's value depends on only the
    corresponding input pixel value (plus, potentially, some globally collected information or
    parameters).
-   Examples of such operators include *brightness and contrast adjustments* as well as color
    correction and transformations.

### Brightness and contrast adjustments

-   Two commonly used point processes are *multiplication* and *addition* with a constant:

    \f[g(x) = \alpha f(x) + \beta\f]

-   The parameters \f$\alpha > 0\f$ and \f$\beta\f$ are often called the *gain* and *bias* parameters;
    sometimes these parameters are said to control *contrast* and *brightness* respectively.
-   You can think of \f$f(x)\f$ as the source image pixels and \f$g(x)\f$ as the output image pixels. Then,
    more conveniently we can write the expression as:

    \f[g(i,j) = \alpha \cdot f(i,j) + \beta\f]

    where \f$i\f$ and \f$j\f$ indicates that the pixel is located in the *i-th* row and *j-th* column.

Code
----

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgProc/BasicLinearTransforms.cpp)

-   The following code performs the operation \f$g(i,j) = \alpha \cdot f(i,j) + \beta\f$ :
    @include samples/cpp/tutorial_code/ImgProc/BasicLinearTransforms.cpp
@end_toggle

@add_toggle_java
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/BasicLinearTransformsDemo.java)

-   The following code performs the operation \f$g(i,j) = \alpha \cdot f(i,j) + \beta\f$ :
    @include samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/BasicLinearTransformsDemo.java
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/BasicLinearTransforms.py)

-   The following code performs the operation \f$g(i,j) = \alpha \cdot f(i,j) + \beta\f$ :
    @include samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/BasicLinearTransforms.py
@end_toggle

Explanation
-----------

-   We load an image using @ref cv::imread and save it in a Mat object:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgProc/BasicLinearTransforms.cpp basic-linear-transform-load
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/BasicLinearTransformsDemo.java basic-linear-transform-load
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/BasicLinearTransforms.py basic-linear-transform-load
@end_toggle

-   Now, since we will make some transformations to this image, we need a new Mat object to store
    it. Also, we want this to have the following features:

    -   Initial pixel values equal to zero
    -   Same size and type as the original image

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgProc/BasicLinearTransforms.cpp basic-linear-transform-output
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/BasicLinearTransformsDemo.java basic-linear-transform-output
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/BasicLinearTransforms.py basic-linear-transform-output
@end_toggle

We observe that @ref cv::Mat::zeros returns a Matlab-style zero initializer based on
*image.size()* and *image.type()*

-   We ask now the values of \f$\alpha\f$ and \f$\beta\f$ to be entered by the user:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgProc/BasicLinearTransforms.cpp basic-linear-transform-parameters
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/BasicLinearTransformsDemo.java basic-linear-transform-parameters
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/BasicLinearTransforms.py basic-linear-transform-parameters
@end_toggle

-   Now, to perform the operation \f$g(i,j) = \alpha \cdot f(i,j) + \beta\f$ we will access to each
    pixel in image. Since we are operating with BGR images, we will have three values per pixel (B,
    G and R), so we will also access them separately. Here is the piece of code:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgProc/BasicLinearTransforms.cpp basic-linear-transform-operation
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/BasicLinearTransformsDemo.java basic-linear-transform-operation
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/BasicLinearTransforms.py basic-linear-transform-operation
@end_toggle

Notice the following (**C++ code only**):
-   To access each pixel in the images we are using this syntax: *image.at\<Vec3b\>(y,x)[c]*
    where *y* is the row, *x* is the column and *c* is B, G or R (0, 1 or 2).
-   Since the operation \f$\alpha \cdot p(i,j) + \beta\f$ can give values out of range or not
    integers (if \f$\alpha\f$ is float), we use cv::saturate_cast to make sure the
    values are valid.

-   Finally, we create windows and show the images, the usual way.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgProc/BasicLinearTransforms.cpp basic-linear-transform-display
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/BasicLinearTransformsDemo.java basic-linear-transform-display
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/BasicLinearTransforms.py basic-linear-transform-display
@end_toggle

@note
    Instead of using the **for** loops to access each pixel, we could have simply used this command:

@add_toggle_cpp
@code{.cpp}
image.convertTo(new_image, -1, alpha, beta);
@endcode
@end_toggle

@add_toggle_java
@code{.java}
image.convertTo(newImage, -1, alpha, beta);
@endcode
@end_toggle

@add_toggle_python
@code{.py}
new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
@endcode
@end_toggle

where @ref cv::Mat::convertTo would effectively perform *new_image = a*image + beta\*. However, we
wanted to show you how to access each pixel. In any case, both methods give the same result but
convertTo is more optimized and works a lot faster.

Result
------

-   Running our code and using \f$\alpha = 2.2\f$ and \f$\beta = 50\f$
    @code{.bash}
    $ ./BasicLinearTransforms lena.jpg
    Basic Linear Transforms
    -------------------------
    * Enter the alpha value [1.0-3.0]: 2.2
    * Enter the beta value [0-100]: 50
    @endcode

-   We get this:

    ![](images/Basic_Linear_Transform_Tutorial_Result_big.jpg)

Practical example
----

In this paragraph, we will put into practice what we have learned to correct an underexposed image by adjusting the brightness
and the contrast of the image. We will also see another technique to correct the brightness of an image called
gamma correction.

### Brightness and contrast adjustments

Increasing (/ decreasing) the \f$\beta\f$ value will add (/ subtract) a constant value to every pixel. Pixel values outside of the [0 ; 255]
range will be saturated (i.e. a pixel value higher (/ lesser) than 255 (/ 0) will be clamped to 255 (/ 0)).

![In light gray, histogram of the original image, in dark gray when brightness = 80 in Gimp](images/Basic_Linear_Transform_Tutorial_hist_beta.png)

The histogram represents for each color level the number of pixels with that color level. A dark image will have many pixels with
low color value and thus the histogram will present a peak in its left part. When adding a constant bias, the histogram is shifted to the
right as we have added a constant bias to all the pixels.

The \f$\alpha\f$ parameter will modify how the levels spread. If \f$ \alpha < 1 \f$, the color levels will be compressed and the result
will be an image with less contrast.

![In light gray, histogram of the original image, in dark gray when contrast < 0 in Gimp](images/Basic_Linear_Transform_Tutorial_hist_alpha.png)

Note that these histograms have been obtained using the Brightness-Contrast tool in the Gimp software. The brightness tool should be
identical to the \f$\beta\f$ bias parameters but the contrast tool seems to differ to the \f$\alpha\f$ gain where the output range
seems to be centered with Gimp (as you can notice in the previous histogram).

It can occur that playing with the \f$\beta\f$ bias will improve the brightness but in the same time the image will appear with a
slight veil as the contrast will be reduced. The \f$\alpha\f$ gain can be used to diminue this effect but due to the saturation,
we will lose some details in the original bright regions.

### Gamma correction

[Gamma correction](https://en.wikipedia.org/wiki/Gamma_correction) can be used to correct the brightness of an image by using a non
linear transformation between the input values and the mapped output values:

\f[O = \left( \frac{I}{255} \right)^{\gamma} \times 255\f]

As this relation is non linear, the effect will not be the same for all the pixels and will depend to their original value.

![Plot for different values of gamma](images/Basic_Linear_Transform_Tutorial_gamma.png)

When \f$ \gamma < 1 \f$, the original dark regions will be brighter and the histogram will be shifted to the right whereas it will
be the opposite with \f$ \gamma > 1 \f$.

### Correct an underexposed image

The following image has been corrected with: \f$ \alpha = 1.3 \f$ and \f$ \beta = 40 \f$.

![By Visem (Own work) [CC BY-SA 3.0], via Wikimedia Commons](images/Basic_Linear_Transform_Tutorial_linear_transform_correction.jpg)

The overall brightness has been improved but you can notice that the clouds are now greatly saturated due to the numerical saturation
of the implementation used ([highlight clipping](https://en.wikipedia.org/wiki/Clipping_(photography)) in photography).

The following image has been corrected with: \f$ \gamma = 0.4 \f$.

![By Visem (Own work) [CC BY-SA 3.0], via Wikimedia Commons](images/Basic_Linear_Transform_Tutorial_gamma_correction.jpg)

The gamma correction should tend to add less saturation effect as the mapping is non linear and there is no numerical saturation possible as in the previous method.

![Left: histogram after alpha, beta correction ; Center: histogram of the original image ; Right: histogram after the gamma correction](images/Basic_Linear_Transform_Tutorial_histogram_compare.png)

The previous figure compares the histograms for the three images (the y-ranges are not the same between the three histograms).
You can notice that most of the pixel values are in the lower part of the histogram for the original image. After \f$ \alpha \f$,
\f$ \beta \f$ correction, we can observe a big peak at 255 due to the saturation as well as a shift in the right.
After gamma correction, the histogram is shifted to the right but the pixels in the dark regions are more shifted
(see the gamma curves [figure](Basic_Linear_Transform_Tutorial_gamma.png)) than those in the bright regions.

In this tutorial, you have seen two simple methods to adjust the contrast and the brightness of an image. **They are basic techniques
and are not intended to be used as a replacement of a raster graphics editor!**

### Code

@add_toggle_cpp
Code for the tutorial is [here](https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/ImgProc/changing_contrast_brightness_image/changing_contrast_brightness_image.cpp).
@end_toggle

@add_toggle_java
Code for the tutorial is [here](https://github.com/opencv/opencv/blob/master/samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/ChangingContrastBrightnessImageDemo.java).
@end_toggle

@add_toggle_python
Code for the tutorial is [here](https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/changing_contrast_brightness_image.py).
@end_toggle

Code for the gamma correction:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgProc/changing_contrast_brightness_image/changing_contrast_brightness_image.cpp changing-contrast-brightness-gamma-correction
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgProc/changing_contrast_brightness_image/ChangingContrastBrightnessImageDemo.java changing-contrast-brightness-gamma-correction
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/changing_contrast_brightness_image.py changing-contrast-brightness-gamma-correction
@end_toggle

A look-up table is used to improve the performance of the computation as only 256 values needs to be calculated once.

### Additional resources

-   [Gamma correction in graphics rendering](https://learnopengl.com/#!Advanced-Lighting/Gamma-Correction)
-   [Gamma correction and images displayed on CRT monitors](http://www.graphics.cornell.edu/~westin/gamma/gamma.html)
-   [Digital exposure techniques](http://www.cambridgeincolour.com/tutorials/digital-exposure-techniques.htm)
