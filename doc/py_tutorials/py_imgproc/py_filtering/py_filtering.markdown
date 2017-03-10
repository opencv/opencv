Smoothing Images {#tutorial_py_filtering}
================

Goals
-----

Learn to:
    -   Blur the images with various low pass filters
    -   Apply custom-made filters to images (2D convolution)

2D Convolution ( Image Filtering )
----------------------------------

As in one-dimensional signals, images also can be filtered with various low-pass filters(LPF),
high-pass filters(HPF) etc. LPF helps in removing noises, blurring the images etc. HPF filters helps
in finding edges in the images.

OpenCV provides a function **cv2.filter2D()** to convolve a kernel with an image. As an example, we
will try an averaging filter on an image. A 5x5 averaging filter kernel will look like below:

\f[K =  \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1 & 1  \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}\f]

Operation is like this: keep this kernel above a pixel, add all the 25 pixels below this kernel,
take its average and replace the central pixel with the new average value. It continues this
operation for all the pixels in the image. Try this code and check the result:
@code{.py}
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

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
removing noises. It actually removes high frequency content (eg: noise, edges) from the image. So
edges are blurred a little bit in this operation. (Well, there are blurring techniques which doesn't
blur the edges too). OpenCV provides mainly four types of blurring techniques.

### 1. Averaging

This is done by convolving image with a normalized box filter. It simply takes the average of all
the pixels under kernel area and replace the central element. This is done by the function
**cv2.blur()** or **cv2.boxFilter()**. Check the docs for more details about the kernel. We should
specify the width and height of kernel. A 3x3 normalized box filter would look like below:

\f[K =  \frac{1}{9} \begin{bmatrix} 1 & 1 & 1  \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}\f]

@note If you don't want to use normalized box filter, use **cv2.boxFilter()**. Pass an argument
normalize=False to the function.

Check a sample demo below with a kernel of 5x5 size:
@code{.py}
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv-logo-white.png')

blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
@endcode
Result:

![image](images/blur.jpg)

### 2. Gaussian Blurring

In this, instead of box filter, gaussian kernel is used. It is done with the function,
**cv2.GaussianBlur()**. We should specify the width and height of kernel which should be positive
and odd. We also should specify the standard deviation in X and Y direction, sigmaX and sigmaY
respectively. If only sigmaX is specified, sigmaY is taken as same as sigmaX. If both are given as
zeros, they are calculated from kernel size. Gaussian blurring is highly effective in removing
gaussian noise from the image.

If you want, you can create a Gaussian kernel with the function, **cv2.getGaussianKernel()**.

The above code can be modified for Gaussian blurring:
@code{.py}
blur = cv2.GaussianBlur(img,(5,5),0)
@endcode
Result:

![image](images/gaussian.jpg)

### 3. Median Blurring

Here, the function **cv2.medianBlur()** takes median of all the pixels under kernel area and central
element is replaced with this median value. This is highly effective against salt-and-pepper noise
in the images. Interesting thing is that, in the above filters, central element is a newly
calculated value which may be a pixel value in the image or a new value. But in median blurring,
central element is always replaced by some pixel value in the image. It reduces the noise
effectively. Its kernel size should be a positive odd integer.

In this demo, I added a 50% noise to our original image and applied median blur. Check the result:
@code{.py}
median = cv2.medianBlur(img,5)
@endcode
Result:

![image](images/median.jpg)

### 4. Bilateral Filtering

**cv2.bilateralFilter()** is highly effective in noise removal while keeping edges sharp. But the
operation is slower compared to other filters. We already saw that gaussian filter takes the a
neighbourhood around the pixel and find its gaussian weighted average. This gaussian filter is a
function of space alone, that is, nearby pixels are considered while filtering. It doesn't consider
whether pixels have almost same intensity. It doesn't consider whether pixel is an edge pixel or
not. So it blurs the edges also, which we don't want to do.

Bilateral filter also takes a gaussian filter in space, but one more gaussian filter which is a
function of pixel difference. Gaussian function of space make sure only nearby pixels are considered
for blurring while gaussian function of intensity difference make sure only those pixels with
similar intensity to central pixel is considered for blurring. So it preserves the edges since
pixels at edges will have large intensity variation.

Below samples shows use bilateral filter (For details on arguments, visit docs).
@code{.py}
blur = cv2.bilateralFilter(img,9,75,75)
@endcode
Result:

![image](images/bilateral.jpg)

See, the texture on the surface is gone, but edges are still preserved.

Additional Resources
--------------------

-#  Details about the [bilateral filtering](http://people.csail.mit.edu/sparis/bf_course/)

Exercises
---------
