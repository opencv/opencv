Histogram - 3 : Histogram Backprojection {#tutorial_js_histogram_backprojection}
========================================

Goal
----

-   We will learn about histogram backprojection.

Theory
------

It was proposed by **Michael J. Swain , Dana H. Ballard** in their paper **Indexing via color
histograms**.

**What is it actually in simple words?** It is used for image segmentation or finding objects of
interest in an image. In simple words, it creates an image of the same size (but single channel) as
that of our input image, where each pixel corresponds to the probability of that pixel belonging to
our object. In more simpler worlds, the output image will have our object of interest in more white
compared to remaining part. Well, that is an intuitive explanation. (I can't make it more simpler).
Histogram Backprojection is used with camshift algorithm etc.

**How do we do it ?** We create a histogram of an image containing our object of interest (in our
case, the ground, leaving player and other things). The object should fill the image as far as
possible for better results. And a color histogram is preferred over grayscale histogram, because
color of the object is a better way to define the object than its grayscale intensity. We then
"back-project" this histogram over our test image where we need to find the object, ie in other
words, we calculate the probability of every pixel belonging to the ground and show it. The
resulting output on proper thresholding gives us the ground alone.

Backprojection in OpenCV
------------------------

We use the functions: **cv.calcBackProject (images, channels, hist, dst, ranges, scale)**

@param images       source arrays. They all should have the same depth, cv.CV_8U, cv.CV_16U or cv.CV_32F , and the same size. Each of them can have an arbitrary number of channels.
@param channels     the list of channels used to compute the back projection. The number of channels must match the histogram dimensionality.
@param hist         input histogram that can be dense or sparse.
@param dst          destination back projection array that is a single-channel array of the same size and depth as images[0].
@param ranges       array of arrays of the histogram bin boundaries in each dimension(see cv.calcHist).
@param scale        optional scale factor for the output back projection.

**cv.normalize (src, dst, alpha = 1, beta = 0, norm_type = cv.NORM_L2, dtype = -1, mask = new cv.Mat())**

@param src        input array.
@param dst        output array of the same size as src .
@param alpha      norm value to normalize to or the lower range boundary in case of the range normalization.
@param beta       upper range boundary in case of the range normalization; it is not used for the norm normalization.
@param norm_type  normalization type (see cv.NormTypes).
@param dtype      when negative, the output array has the same type as src; otherwise, it has the same number of channels as src and the depth = CV_MAT_DEPTH(dtype).
@param mask       optional operation mask.

Try it
------

\htmlonly
<iframe src="../../js_histogram_backprojection_calcBackProject.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly