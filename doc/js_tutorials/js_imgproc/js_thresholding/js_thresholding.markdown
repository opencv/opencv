Image Thresholding {#tutorial_js_thresholding}
==================

Goal
----

-   In this tutorial, you will learn Simple thresholding, Adaptive thresholding, Otsu's thresholding
    etc.
-   You will learn these functions : **cv.threshold**, **cv.adaptiveThreshold** etc.

Simple Thresholding
-------------------

Here, the matter is straight forward. If pixel value is greater than a threshold value, it is
assigned one value (may be white), else it is assigned another value (may be black).

We use the function: **cv.threshold (src, dst, thresh, maxval, type)**
@param src    input array.
@param dst    output array of the same size and type and the same number of channels as src.
@param thresh threshold value.
@param maxval maximum value to use with the cv.THRESH_BINARY and cv.THRESH_BINARY_INV thresholding types.
@param type   thresholding type(see cv.ThresholdTypes).

**thresholding type** - OpenCV provides different styles of thresholding and it is decided
by the fourth parameter of the function. Different types are:

-   cv.THRESH_BINARY
-   cv.THRESH_BINARY_INV
-   cv.THRESH_TRUNC
-   cv.THRESH_TOZERO
-   cv.THRESH_OTSU
-   cv.THRESH_TRIANGLE

@note Input image should be single channel only in case of cv.THRESH_OTSU or cv.THRESH_TRIANGLE flags

Try it
------

\htmlonly
<iframe src="../../js_thresholding_threshold.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

Adaptive Thresholding
---------------------

In the previous section, we used a global value as threshold value. But it may not be good in all
the conditions where image has different lighting conditions in different areas. In that case, we go
for adaptive thresholding. In this, the algorithm calculate the threshold for a small regions of the
image. So we get different thresholds for different regions of the same image and it gives us better
results for images with varying illumination.

We use the function: **cv.adaptiveThreshold (src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)**
@param src             source 8-bit single-channel image.
@param dst             dstination image of the same size and the same type as src.
@param maxValue        non-zero value assigned to the pixels for which the condition is satisfied
@param adaptiveMethod  adaptive thresholding algorithm to use.
@param thresholdType   thresholding type that must be either cv.THRESH_BINARY or cv.THRESH_BINARY_INV.
@param blockSize       size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
@param C               constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.

**adaptiveMethod** - It decides how thresholding value is calculated:
    -   cv.ADAPTIVE_THRESH_MEAN_C
    -   cv.ADAPTIVE_THRESH_GAUSSIAN_C

Try it
------

\htmlonly
<iframe src="../../js_thresholding_adaptiveThreshold.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly