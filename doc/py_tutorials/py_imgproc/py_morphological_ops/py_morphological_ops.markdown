Morphological Transformations {#tutorial_py_morphological_ops}
=============================

Goal
----

In this chapter,
    -   We will learn different morphological operations like Erosion, Dilation, Opening, Closing
        etc.
    -   We will see different functions like : **cv.erode()**, **cv.dilate()**,
        **cv.morphologyEx()** etc.

Theory
------

Morphological transformations are some simple operations based on the image shape. It is normally
performed on binary images. It needs two inputs, one is our original image, second one is called
**structuring element** or **kernel** which decides the nature of operation. Two basic morphological
operators are Erosion and Dilation. Then its variant forms like Opening, Closing, Gradient etc also
comes into play. We will see them one-by-one with help of following image:

![image](images/j.png)

### 1. Erosion

The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of
foreground object (Always try to keep foreground in white). So what it does? The kernel slides
through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) will be
considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).

So what happends is that, all the pixels near boundary will be discarded depending upon the size of
kernel. So the thickness or size of the foreground object decreases or simply white region decreases
in the image. It is useful for removing small white noises (as we have seen in colorspace chapter),
detach two connected objects etc.

Here, as an example, I would use a 5x5 kernel with full of ones. Let's see it how it works:
@code{.py}
import cv2 as cv
import numpy as np

img = cv.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
@endcode
Result:

![image](images/erosion.png)

### 2. Dilation

It is just opposite of erosion. Here, a pixel element is '1' if at least one pixel under the kernel
is '1'. So it increases the white region in the image or size of foreground object increases.
Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes
white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won't come
back, but our object area increases. It is also useful in joining broken parts of an object.
@code{.py}
dilation = cv.dilate(img,kernel,iterations = 1)
@endcode
Result:

![image](images/dilation.png)

### 3. Opening

Opening is just another name of **erosion followed by dilation**. It is useful in removing noise, as
we explained above. Here we use the function, **cv.morphologyEx()**
@code{.py}
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
@endcode
Result:

![image](images/opening.png)

### 4. Closing

Closing is reverse of Opening, **Dilation followed by Erosion**. It is useful in closing small holes
inside the foreground objects, or small black points on the object.
@code{.py}
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
@endcode
Result:

![image](images/closing.png)

### 5. Morphological Gradient

It is the difference between dilation and erosion of an image.

The result will look like the outline of the object.
@code{.py}
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
@endcode
Result:

![image](images/gradient.png)

### 6. Top Hat

It is the difference between input image and Opening of the image. Below example is done for a 9x9
kernel.
@code{.py}
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
@endcode
Result:

![image](images/tophat.png)

### 7. Black Hat

It is the difference between the closing of the input image and input image.
@code{.py}
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
@endcode
Result:

![image](images/blackhat.png)

Structuring Element
-------------------

We manually created a structuring elements in the previous examples with help of Numpy. It is
rectangular shape. But in some cases, you may need elliptical/circular shaped kernels. So for this
purpose, OpenCV has a function, **cv.getStructuringElement()**. You just pass the shape and size of
the kernel, you get the desired kernel.
@code{.py}
# Rectangular Kernel
>>> cv.getStructuringElement(cv.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)

# Elliptical Kernel
>>> cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)

# Cross-shaped Kernel
>>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
@endcode
Additional Resources
--------------------

-#  [Morphological Operations](http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm) at HIPR2

Exercises
---------
