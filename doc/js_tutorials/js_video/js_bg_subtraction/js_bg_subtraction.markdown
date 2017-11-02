Background Subtraction {#tutorial_js_bg_subtraction}
======================

Goal
----

-   We will familiarize with the background subtraction methods available in OpenCV.js.

Basics
------

Background subtraction is a major preprocessing steps in many vision based applications. For
example, consider the cases like visitor counter where a static camera takes the number of visitors
entering or leaving the room, or a traffic camera extracting information about the vehicles etc. In
all these cases, first you need to extract the person or vehicles alone. Technically, you need to
extract the moving foreground from static background.

If you have an image of background alone, like image of the room without visitors, image of the road
without vehicles etc, it is an easy job. Just subtract the new image from the background. You get
the foreground objects alone. But in most of the cases, you may not have such an image, so we need
to extract the background from whatever images we have. It become more complicated when there is
shadow of the vehicles. Since shadow is also moving, simple subtraction will mark that also as
foreground. It complicates things.

OpenCV.js has implemented one algorithm for this purpose, which is very easy to use.

BackgroundSubtractorMOG2
------------------------

It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It is based on two
papers by Z.Zivkovic, "Improved adaptive Gausian mixture model for background subtraction" in 2004
and "Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction"
in 2006. One important feature of this algorithm is that it selects the appropriate number of
gaussian distribution for each pixel. It provides better adaptibility to varying scenes due illumination
changes etc.

While coding, we use the constructor: **cv.BackgroundSubtractorMOG2 (history = 500, varThreshold = 16,
detectShadows = true)**
@param history         Length of the history.
@param varThreshold    Threshold on the squared distance between the pixel and the sample to decide
whether a pixel is close to that sample. This parameter does not affect the background update.
@param detectShadows   If true, the algorithm will detect shadows and mark them. It decreases the
speed a bit, so if you do not need this feature, set the parameter to false.
@return                instance of cv.BackgroundSubtractorMOG2

Use **apply (image, fgmask, learningRate = -1)** method to get the foreground mask
@param image         Next video frame. Floating point frame will be used without scaling and should
be in range [0,255].
@param fgmask        The output foreground mask as an 8-bit binary image.
@param learningRate  The value between 0 and 1 that indicates how fast the background model is learnt.
Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means
that the background model is not updated at all, 1 means that the background model is completely
reinitialized from the last frame.

@note The instance of cv.BackgroundSubtractorMOG2 should be deleted manually.

Try it
------

\htmlonly
<iframe src="../../js_bg_subtraction.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly
