Discrete Fourier Transform {#tutorial_discrete_fourier_transform}
==========================

@tableofcontents

@prev_tutorial{tutorial_basic_linear_transform}
@next_tutorial{tutorial_file_input_output_with_xml_yml}

|    |    |
| -: | :- |
| Original author | Bernát Gábor |
| Compatibility | OpenCV >= 3.0 |

Goal
----

We'll seek answers for the following questions:

-   What is a Fourier transform and why use it?
-   How to do it in OpenCV?
-   Usage of functions such as: **copyMakeBorder()** , **merge()** , **dft()** ,
    **getOptimalDFTSize()** , **log()** and **normalize()** .

Source code
-----------

@add_toggle_cpp
You can [download this from here
](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp) or
find it in the
`samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp` of the
OpenCV source code library.
@end_toggle

@add_toggle_java
You can [download this from here
](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java) or
find it in the
`samples/java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java` of the
OpenCV source code library.
@end_toggle

@add_toggle_python
You can [download this from here
](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py) or
find it in the
`samples/python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py` of the
OpenCV source code library.
@end_toggle

Here's a sample usage of **dft()** :

@add_toggle_cpp
@include cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp
@end_toggle

@add_toggle_java
@include java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java
@end_toggle

@add_toggle_python
@include python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py
@end_toggle

Explanation
-----------

The Fourier Transform will decompose an image into its sinus and cosines components. In other words,
it will transform an image from its spatial domain to its frequency domain. The idea is that any
function may be approximated exactly with the sum of infinite sinus and cosines functions. The
Fourier Transform is a way how to do this. Mathematically a two dimensional images Fourier transform
is:

\f[F(k,l) = \displaystyle\sum\limits_{i=0}^{N-1}\sum\limits_{j=0}^{N-1} f(i,j)e^{-i2\pi(\frac{ki}{N}+\frac{lj}{N})}\f]\f[e^{ix} = \cos{x} + i\sin {x}\f]

Here f is the image value in its spatial domain and F in its frequency domain. The result of the
transformation is complex numbers. Displaying this is possible either via a *real* image and a
*complex* image or via a *magnitude* and a *phase* image. However, throughout the image processing
algorithms only the *magnitude* image is interesting as this contains all the information we need
about the images geometric structure. Nevertheless, if you intend to make some modifications of the
image in these forms and then you need to retransform it you'll need to preserve both of these.

In this sample I'll show how to calculate and show the *magnitude* image of a Fourier Transform. In
case of digital images are discrete. This means they may take up a value from a given domain value.
For example in a basic gray scale image values usually are between zero and 255. Therefore the
Fourier Transform too needs to be of a discrete type resulting in a Discrete Fourier Transform
(*DFT*). You'll want to use this whenever you need to determine the structure of an image from a
geometrical point of view. Here are the steps to follow (in case of a gray scale input image *I*):

### Expand the image to an optimal size

The performance of a DFT is dependent of the image
size. It tends to be the fastest for image sizes that are multiple of the numbers two, three and
five. Therefore, to achieve maximal performance it is generally a good idea to pad border values
to the image to get a size with such traits. The **getOptimalDFTSize()** returns this
optimal size and we can use the **copyMakeBorder()** function to expand the borders of an
image (the appended pixels are initialized with zero):

@add_toggle_cpp
@snippet cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp expand
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java expand
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py expand
@end_toggle

### Make place for both the complex and the real values

The result of a Fourier Transform is
complex. This implies that for each image value the result is two image values (one per
component). Moreover, the frequency domains range is much larger than its spatial counterpart.
Therefore, we store these usually at least in a *float* format. Therefore we'll convert our
input image to this type and expand it with another channel to hold the complex values:

@add_toggle_cpp
@snippet cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp complex_and_real
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java complex_and_real
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py complex_and_real
@end_toggle

### Make the Discrete Fourier Transform
It's possible an in-place calculation (same input as
output):

@add_toggle_cpp
@snippet cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp dft
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java dft
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py dft
@end_toggle

### Transform the real and complex values to magnitude
A complex number has a real (*Re*) and a
complex (imaginary - *Im*) part. The results of a DFT are complex numbers. The magnitude of a
DFT is:

\f[M = \sqrt[2]{ {Re(DFT(I))}^2 + {Im(DFT(I))}^2}\f]

Translated to OpenCV code:

@add_toggle_cpp
@snippet cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp magnitude
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java magnitude
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py magnitude
@end_toggle

### Switch to a logarithmic scale
It turns out that the dynamic range of the Fourier
coefficients is too large to be displayed on the screen. We have some small and some high
changing values that we can't observe like this. Therefore the high values will all turn out as
white points, while the small ones as black. To use the gray scale values to for visualization
we can transform our linear scale to a logarithmic one:

\f[M_1 = \log{(1 + M)}\f]

Translated to OpenCV code:

@add_toggle_cpp
@snippet cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp log
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java log
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py log
@end_toggle

### Crop and rearrange
Remember, that at the first step, we expanded the image? Well, it's time
to throw away the newly introduced values. For visualization purposes we may also rearrange the
quadrants of the result, so that the origin (zero, zero) corresponds with the image center.

@add_toggle_cpp
@snippet cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp crop_rearrange
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java crop_rearrange
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py crop_rearrange
@end_toggle

### Normalize
This is done again for visualization purposes. We now have the magnitudes,
however this are still out of our image display range of zero to one. We normalize our values to
this range using the @ref cv::normalize() function.

@add_toggle_cpp
@snippet cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp normalize
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/core/discrete_fourier_transform/DiscreteFourierTransform.java normalize
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.py normalize
@end_toggle

Result
------

An application idea would be to determine the geometrical orientation present in the image. For
example, let us find out if a text is horizontal or not? Looking at some text you'll notice that the
text lines sort of form also horizontal lines and the letters form sort of vertical lines. These two
main components of a text snippet may be also seen in case of the Fourier transform. Let us use
[this horizontal ](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/imageTextN.png) and [this rotated](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/imageTextR.png)
image about a text.

In case of the horizontal text:

![](images/result_normal.jpg)

In case of a rotated text:

![](images/result_rotated.jpg)

You can see that the most influential components of the frequency domain (brightest dots on the
magnitude image) follow the geometric rotation of objects on the image. From this we may calculate
the offset and perform an image rotation to correct eventual miss alignments.
