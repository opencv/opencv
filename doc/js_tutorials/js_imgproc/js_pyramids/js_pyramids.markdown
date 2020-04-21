Image Pyramids {#tutorial_js_pyramids}
==============

Goal
----

-   We will learn about Image Pyramids
-   We will learn these functions: **cv.pyrUp()**, **cv.pyrDown()**

Theory
------

Normally, we used to work with an image of constant size. But on some occasions, we need to work
with (the same) images in different resolution. For example, while searching for something in
an image, like face, we are not sure at what size the object will be present in said image. In that
case, we will need to create a set of the same image with different resolutions and search for object
in all of them. These set of images with different resolutions are called **Image Pyramids** (because
when they are kept in a stack with the highest resolution image at the bottom and the lowest resolution
image at top, it looks like a pyramid).

There are two kinds of Image Pyramids. 1) **Gaussian Pyramid** and 2) **Laplacian Pyramids**

Higher level (Low resolution) in a Gaussian Pyramid is formed by removing consecutive rows and
columns in Lower level (higher resolution) image. Then each pixel in higher level is formed by the
contribution from 5 pixels in underlying level with gaussian weights. By doing so, a \f$M \times N\f$
image becomes \f$M/2 \times N/2\f$ image. So area reduces to one-fourth of original area. It is called
an Octave. The same pattern continues as we go upper in pyramid (ie, resolution decreases).
Similarly while expanding, area becomes 4 times in each level. We can find Gaussian pyramids using
**cv.pyrDown()** and **cv.pyrUp()** functions.

Laplacian Pyramids are formed from the Gaussian Pyramids. There is no exclusive function for that.
Laplacian pyramid images are like edge images only. Most of its elements are zeros. They are used in
image compression. A level in Laplacian Pyramid is formed by the difference between that level in
Gaussian Pyramid and expanded version of its upper level in Gaussian Pyramid.

Downsample
------

We use the function: **cv.pyrDown (src, dst, dstsize = new cv.Size(0, 0), borderType  = cv.BORDER_DEFAULT)**
@param src         input image.
@param dst         output image; it has the specified size and the same type as src.
@param dstsize     size of the output image.
@param borderType  pixel extrapolation method(see cv.BorderTypes, cv.BORDER_CONSTANT isn't supported).

Try it
------

\htmlonly
<iframe src="../../js_pyramids_pyrDown.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

Upsample
------

We use the function: **cv.pyrUp (src, dst, dstsize = new cv.Size(0, 0), borderType  = cv.BORDER_DEFAULT)**
@param src         input image.
@param dst         output image; it has the specified size and the same type as src.
@param dstsize     size of the output image.
@param borderType  pixel extrapolation method(see cv.BorderTypes, only cv.BORDER_DEFAULT is supported).

Try it
------

\htmlonly
<iframe src="../../js_pyramids_pyrUp.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly