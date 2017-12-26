Morphological Transformations {#tutorial_js_morphological_ops}
=============================

Goal
----

-   We will learn different morphological operations like Erosion, Dilation, Opening, Closing
        etc.
-   We will learn different functions like : **cv.erode()**, **cv.dilate()**,
        **cv.morphologyEx()** etc.

Theory
------

Morphological transformations are some simple operations based on the image shape. It is normally
performed on binary images. It needs two inputs, one is our original image, second one is called
**structuring element** or **kernel** which decides the nature of operation. Two basic morphological
operators are Erosion and Dilation. Then its variant forms like Opening, Closing, Gradient etc also
comes into play. We will see them one-by-one with help of following image:

![image](shape.jpg)

### 1. Erosion

The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of
foreground object (Always try to keep foreground in white). So what it does? The kernel slides
through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) will be
considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).

So what happends is that, all the pixels near boundary will be discarded depending upon the size of
kernel. So the thickness or size of the foreground object decreases or simply white region decreases
in the image. It is useful for removing small white noises (as we have seen in colorspace chapter),
detach two connected objects etc.

We use the function: **cv.erode (src, dst, kernel, anchor = new cv.Point(-1, -1), iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())**
@param src          input image; the number of channels can be arbitrary, but the depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F.
@param dst          output image of the same size and type as src.
@param kernel       structuring element used for erosion.
@param anchor       position of the anchor within the element; default value new cv.Point(-1, -1) means that the anchor is at the element center.
@param iterations   number of times erosion is applied.
@param borderType   pixel extrapolation method(see cv.BorderTypes).
@param borderValue  border value in case of a constant border

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_erode.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 2. Dilation

It is just opposite of erosion. Here, a pixel element is '1' if atleast one pixel under the kernel
is '1'. So it increases the white region in the image or size of foreground object increases.
Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes
white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won't come
back, but our object area increases. It is also useful in joining broken parts of an object.

We use the function: **cv.dilate (src, dst, kernel, anchor = new cv.Point(-1, -1), iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())**
@param src          input image; the number of channels can be arbitrary, but the depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F.
@param dst          output image of the same size and type as src.
@param kernel       structuring element used for dilation.
@param anchor       position of the anchor within the element; default value new cv.Point(-1, -1) means that the anchor is at the element center.
@param iterations   number of times dilation is applied.
@param borderType   pixel extrapolation method(see cv.BorderTypes).
@param borderValue  border value in case of a constant border

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_dilate.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 3. Opening

Opening is just another name of **erosion followed by dilation**. It is useful in removing noise.

We use the function: **cv.morphologyEx (src, dst, op, kernel, anchor = new cv.Point(-1, -1), iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())**
@param src          source image. The number of channels can be arbitrary. The depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F
@param dst          destination image of the same size and type as source image.
@param op           type of a morphological operation, (see cv.MorphTypes).
@param kernel       structuring element. It can be created using cv.getStructuringElement.
@param anchor       anchor position with the kernel. Negative values mean that the anchor is at the kernel center.
@param iterations   number of times dilation is applied.
@param borderType   pixel extrapolation method(see cv.BorderTypes).
@param borderValue  border value in case of a constant border. The default value has a special meaning.

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_opening.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 4. Closing

Closing is reverse of Opening, **Dilation followed by Erosion**. It is useful in closing small holes
inside the foreground objects, or small black points on the object.

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_closing.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 5. Morphological Gradient

It is the difference between dilation and erosion of an image.

The result will look like the outline of the object.

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_gradient.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 6. Top Hat

It is the difference between input image and Opening of the image.

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_topHat.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 7. Black Hat

It is the difference between the closing of the input image and input image.

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_blackHat.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

Structuring Element
-------------------

We manually created a structuring elements in the previous examples with help of cv.Mat.ones. It is
rectangular shape. But in some cases, you may need elliptical/circular shaped kernels. So for this
purpose, OpenCV has a function, **cv.getStructuringElement()**. You just pass the shape and size of
the kernel, you get the desired kernel.

We use the function: **cv.getStructuringElement (shape, ksize, anchor = new cv.Point(-1, -1))**
@param shape          element shape that could be one of cv.MorphShapes
@param ksize          size of the structuring element.
@param anchor         anchor position within the element. The default value [−1,−1] means that the anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor position. In other cases the anchor just regulates how much the result of the morphological operation is shifted.

Try it
------

\htmlonly
<iframe src="../../js_morphological_ops_getStructuringElement.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly