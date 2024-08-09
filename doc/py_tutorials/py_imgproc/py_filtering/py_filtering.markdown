Smoothing Images {#tutorial_py_filtering}
================

Goals
-----

Learn to:
    -   Blur images with various low pass filters
    -   Apply custom-made filters to images (2D convolution)

2D Convolution ( Image Filtering )
----------------------------------

As in one-dimensional signals, images also can be filtered with various low-pass filters (LPF),
high-pass filters (HPF), etc. LPF helps in removing noise, blurring images, etc. HPF filters help
in finding edges in images.

OpenCV provides a function **cv.filter2D()** to convolve a kernel with an image. As an example, we
will try an averaging filter on an image. A 5x5 averaging filter kernel will look like the below:

\f[K =  \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1 & 1  \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}\f]

The operation works like this: keep this kernel above a pixel, add all the 25 pixels below this kernel,
take the average, and replace the central pixel with the new average value. This operation is continued
for all the pixels in the image. Try this code and check the result:
@code{.py}
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('opencv_logo.png')
assert img is not None, "file could not be read, check with os.path.exists()"

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
@endcode
Result:

![image](images/filter.jpg)

Image Blurring (Image Smoothing)
--------------------------------

Image blurring is achieved by convolving the image with a low-pass filter kernel. It is useful for
removing noise. It actually removes high frequency content (eg: noise, edges) from the image. So
edges are blurred a little bit in this operation (there are also blurring techniques which don't
blur the edges). OpenCV provides four main types of blurring techniques.

### 1. Averaging

This is done by convolving an image with a normalized box filter. It simply takes the average of all
the pixels under the kernel area and replaces the central element. This is done by the function
**cv.blur()** or **cv.boxFilter()**. Check the docs for more details about the kernel. We should
specify the width and height of the kernel. A 3x3 normalized box filter would look like the below:

\f[K =  \frac{1}{9} \begin{bmatrix} 1 & 1 & 1  \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}\f]

@note If you don't want to use a normalized box filter, use **cv.boxFilter()**. Pass an argument
normalize=False to the function.

Check a sample demo below with a kernel of 5x5 size:
@code{.py}
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('opencv-logo-white.png')
assert img is not None, "file could not be read, check with os.path.exists()"

blur = cv.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
@endcode
Result:

![image](images/blur.jpg)

### 2. Gaussian Blurring

In this method, instead of a box filter, a Gaussian kernel is used. It is done with the function,
**cv.GaussianBlur()**. We should specify the width and height of the kernel which should be positive
and odd. We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY
respectively. If only sigmaX is specified, sigmaY is taken as the same as sigmaX. If both are given as
zeros, they are calculated from the kernel size. Gaussian blurring is highly effective in removing
Gaussian noise from an image.

If you want, you can create a Gaussian kernel with the function, **cv.getGaussianKernel()**.

The above code can be modified for Gaussian blurring:
@code{.py}
blur = cv.GaussianBlur(img,(5,5),0)
@endcode
Result:

![image](images/gaussian.jpg)

### 3. Median Blurring

Here, the function **cv.medianBlur()** takes the median of all the pixels under the kernel area and the central
element is replaced with this median value. This is highly effective against salt-and-pepper noise
in an image. Interestingly, in the above filters, the central element is a newly
calculated value which may be a pixel value in the image or a new value. But in median blurring,
the central element is always replaced by some pixel value in the image. It reduces the noise
effectively. Its kernel size should be a positive odd integer.

In this demo, I added a 50% noise to our original image and applied median blurring. Check the result:
@code{.py}
median = cv.medianBlur(img,5)
@endcode
Result:

![image](images/median.jpg)

### 4. Bilateral Filtering

**cv.bilateralFilter()** is highly effective in noise removal while keeping edges sharp. But the
operation is slower compared to other filters. We already saw that a Gaussian filter takes the
neighbourhood around the pixel and finds its Gaussian weighted average. This Gaussian filter is a
function of space alone, that is, nearby pixels are considered while filtering. It doesn't consider
whether pixels have almost the same intensity. It doesn't consider whether a pixel is an edge pixel or
not. So it blurs the edges also, which we don't want to do.

Bilateral filtering also takes a Gaussian filter in space, but one more Gaussian filter which is a
function of pixel difference. The Gaussian function of space makes sure that only nearby pixels are considered
for blurring, while the Gaussian function of intensity difference makes sure that only those pixels with
similar intensities to the central pixel are considered for blurring. So it preserves the edges since
pixels at edges will have large intensity variation.

The below sample shows use of a bilateral filter (For details on arguments, visit docs).
@code{.py}
blur = cv.bilateralFilter(img,9,75,75)
@endcode
Result:

![image](images/bilateral.jpg)

See, the texture on the surface is gone, but the edges are still preserved.

Additional Resources
--------------------

-#  Details about the [bilateral filtering](http://people.csail.mit.edu/sparis/bf_course/)
