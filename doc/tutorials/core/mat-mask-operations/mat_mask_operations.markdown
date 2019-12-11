Mask operations on matrices {#tutorial_mat_mask_operations}
===========================

@prev_tutorial{tutorial_how_to_scan_images}
@next_tutorial{tutorial_mat_operations}

Mask operations on matrices are quite simple. The idea is that we recalculate each pixel's value in
an image according to a mask matrix (also known as kernel). This mask holds values that will adjust
how much influence neighboring pixels (and the current pixel) have on the new pixel value. From a
mathematical point of view we make a weighted average, with our specified values.

Our test case
-------------

Let's consider the issue of an image contrast enhancement method. Basically we want to apply for
every pixel of the image the following formula:

\f[I(i,j) = 5*I(i,j) - [ I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1)]\f]\f[\iff I(i,j)*M, \text{where }
M = \bordermatrix{ _i\backslash ^j  & -1 &  0 & +1 \cr
                     -1 &  0 & -1 &  0 \cr
                      0 & -1 &  5 & -1 \cr
                     +1 &  0 & -1 &  0 \cr
                 }\f]

The first notation is by using a formula, while the second is a compacted version of the first by
using a mask. You use the mask by putting the center of the mask matrix (in the upper case noted by
the zero-zero index) on the pixel you want to calculate and sum up the pixel values multiplied with
the overlapped matrix values. It's the same thing, however in case of large matrices the latter
notation is a lot easier to look over.

Code
----

@add_toggle_cpp
You can download this source code from [here
](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp) or look in the
OpenCV source code libraries sample directory at
`samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp`.
@include samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp
@end_toggle

@add_toggle_java
You can download this source code from [here
](https://raw.githubusercontent.com/opencv/opencv/master/samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java) or look in the
OpenCV source code libraries sample directory at
`samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java`.
@include samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java
@end_toggle

@add_toggle_python
You can download this source code from [here
](https://raw.githubusercontent.com/opencv/opencv/master/samples/python/tutorial_code/core/mat_mask_operations/mat_mask_operations.py) or look in the
OpenCV source code libraries sample directory at
`samples/python/tutorial_code/core/mat_mask_operations/mat_mask_operations.py`.
@include samples/python/tutorial_code/core/mat_mask_operations/mat_mask_operations.py
@end_toggle

The Basic Method
----------------

Now let us see how we can make this happen by using the basic pixel access method or by using the
**filter2D()** function.

Here's a function that will do this:
@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp basic_method

At first we make sure that the input images data is in unsigned char format. For this we use the
@ref cv::CV_Assert function that throws an error when the expression inside it is false.
@snippet samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp 8_bit
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java basic_method

At first we make sure that the input images data in unsigned 8 bit format.
@snippet samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java 8_bit
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_mask_operations/mat_mask_operations.py basic_method

At first we make sure that the input images data in unsigned 8 bit format.
@code{.py}
my_image = cv.cvtColor(my_image, cv.CV_8U)
@endcode

@end_toggle

We create an output image with the same size and the same type as our input. As you can see in the
@ref tutorial_how_to_scan_images_storing "storing" section, depending on the number of channels we may have one or more
subcolumns.

@add_toggle_cpp
We will iterate through them via pointers so the total number of elements depends on
this number.
@snippet samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp create_channels
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java create_channels
@end_toggle

@add_toggle_python
@code{.py}
height, width, n_channels = my_image.shape
result = np.zeros(my_image.shape, my_image.dtype)
@endcode
@end_toggle

@add_toggle_cpp
We'll use the plain C [] operator to access pixels. Because we need to access multiple rows at the
same time we'll acquire the pointers for each of them (a previous, a current and a next line). We
need another pointer to where we're going to save the calculation. Then simply access the right
items with the [] operator. For moving the output pointer ahead we simply increase this (with one
byte) after each operation:
@snippet samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp basic_method_loop

On the borders of the image the upper notation results inexistent pixel locations (like minus one -
minus one). In these points our formula is undefined. A simple solution is to not apply the kernel
in these points and, for example, set the pixels on the borders to zeros:

@snippet samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp borders
@end_toggle

@add_toggle_java
We need to access multiple rows and columns which can be done by adding or subtracting 1 to the current center (i,j).
Then we apply the sum and put the new value in the Result matrix.
@snippet samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java basic_method_loop

On the borders of the image the upper notation results in inexistent pixel locations (like (-1,-1)).
In these points our formula is undefined. A simple solution is to not apply the kernel
in these points and, for example, set the pixels on the borders to zeros:

@snippet samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java borders
@end_toggle

@add_toggle_python
We need to access multiple rows and columns which can be done by adding or subtracting 1 to the current center (i,j).
Then we apply the sum and put the new value in the Result matrix.
@snippet samples/python/tutorial_code/core/mat_mask_operations/mat_mask_operations.py basic_method_loop
@end_toggle

The filter2D function
---------------------

Applying such filters are so common in image processing that in OpenCV there is a function that
will take care of applying the mask (also called a kernel in some places). For this you first need
to define an object that holds the mask:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp kern
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java kern
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_mask_operations/mat_mask_operations.py kern
@end_toggle

Then call the **filter2D()** function specifying the input, the output image and the kernel to
use:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp filter2D
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_mask_operations/MatMaskOperations.java filter2D
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_mask_operations/mat_mask_operations.py filter2D
@end_toggle

The function even has a fifth optional argument to specify the center of the kernel, a sixth
for adding an optional value to the filtered pixels before storing them in K and a seventh one
for determining what to do in the regions where the operation is undefined (borders).

This function is shorter, less verbose and, because there are some optimizations, it is usually faster
than the *hand-coded method*. For example in my test while the second one took only 13
milliseconds the first took around 31 milliseconds. Quite some difference.

For example:

![](images/resultMatMaskFilter2D.png)

@add_toggle_cpp
Check out an instance of running the program on our [YouTube
channel](http://www.youtube.com/watch?v=7PF1tAU9se4) .
@youtube{7PF1tAU9se4}
@end_toggle
