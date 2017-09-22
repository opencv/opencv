Canny Edge Detection {#tutorial_js_canny}
====================

Goal
----

-   Concept of Canny edge detection
-   OpenCV functions for that : **cv.Canny()**

Theory
------

Canny Edge Detection is a popular edge detection algorithm. It was developed by John F. Canny in 1986. It is a multi-stage algorithm and we will go through each stages.

-#  **Noise Reduction**

    Since edge detection is susceptible to noise in the image, first step is to remove the noise in the
    image with a 5x5 Gaussian filter. We have already seen this in previous chapters.

-#  **Finding Intensity Gradient of the Image**

    Smoothened image is then filtered with a Sobel kernel in both horizontal and vertical direction to
    get first derivative in horizontal direction (\f$G_x\f$) and vertical direction (\f$G_y\f$). From these two
    images, we can find edge gradient and direction for each pixel as follows:

    \f[
    Edge\_Gradient \; (G) = \sqrt{G_x^2 + G_y^2} \\
    Angle \; (\theta) = \tan^{-1} \bigg(\frac{G_y}{G_x}\bigg)
    \f]

    Gradient direction is always perpendicular to edges. It is rounded to one of four angles
    representing vertical, horizontal and two diagonal directions.

-#  **Non-maximum Suppression**

    After getting gradient magnitude and direction, a full scan of image is done to remove any unwanted
    pixels which may not constitute the edge. For this, at every pixel, pixel is checked if it is a
    local maximum in its neighborhood in the direction of gradient. Check the image below:

    ![image](images/nms.jpg)

    Point A is on the edge ( in vertical direction). Gradient direction is normal to the edge. Point B
    and C are in gradient directions. So point A is checked with point B and C to see if it forms a
    local maximum. If so, it is considered for next stage, otherwise, it is suppressed ( put to zero).

    In short, the result you get is a binary image with "thin edges".

-#  **Hysteresis Thresholding**

    This stage decides which are all edges are really edges and which are not. For this, we need two
    threshold values, minVal and maxVal. Any edges with intensity gradient more than maxVal are sure to
    be edges and those below minVal are sure to be non-edges, so discarded. Those who lie between these
    two thresholds are classified edges or non-edges based on their connectivity. If they are connected
    to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded.
    See the image below:

    ![image](images/hysteresis.jpg)

    The edge A is above the maxVal, so considered as "sure-edge". Although edge C is below maxVal, it is
    connected to edge A, so that also considered as valid edge and we get that full curve. But edge B,
    although it is above minVal and is in same region as that of edge C, it is not connected to any
    "sure-edge", so that is discarded. So it is very important that we have to select minVal and maxVal
    accordingly to get the correct result.

    This stage also removes small pixels noises on the assumption that edges are long lines.

So what we finally get is strong edges in the image.

Canny Edge Detection in OpenCV
------------------------------

We use the function: **cv.Canny(image, edges, threshold1, threshold2, apertureSize = 3, L2gradient = false)**
@param image         8-bit input image.
@param edges         output edge map; single channels 8-bit image, which has the same size as image.
@param threshold1    first threshold for the hysteresis procedure.
@param threshold2    second threshold for the hysteresis procedure..
@param apertureSize  aperture size for the Sobel operator.
@param L2gradient    specifies the equation for finding gradient
magnitude. If it is True, it uses the equation mentioned above which is more accurate, otherwise it uses this function: \f$Edge\_Gradient \; (G) = |G_x| + |G_y|\f$.

Try it
------

\htmlonly
<iframe src="../../js_canny.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly