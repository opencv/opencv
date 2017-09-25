Template Matching {#tutorial_js_template_matching}
=================

Goals
-----

-   To find objects in an image using Template Matching
-   You will learn these functions : **cv.matchTemplate()**, **cv.minMaxLoc()**

Theory
------

Template Matching is a method for searching and finding the location of a template image in a larger
image. OpenCV comes with a function **cv.matchTemplate()** for this purpose. It simply slides the
template image over the input image (as in 2D convolution) and compares the template and patch of
input image under the template image. Several comparison methods are implemented in OpenCV. (You can
check docs for more details). It returns a grayscale image, where each pixel denotes how much does
the neighbourhood of that pixel match with template.

If input image is of size (WxH) and template image is of size (wxh), output image will have a size
of (W-w+1, H-h+1). Once you got the result, you can use **cv.minMaxLoc()** function to find where
is the maximum/minimum value. Take it as the top-left corner of rectangle and take (w,h) as width
and height of the rectangle. That rectangle is your region of template.

@note If you are using cv.TM_SQDIFF as comparison method, minimum value gives the best match.

Template Matching in OpenCV
---------------------------

We use the function: **cv.matchTemplate (image, templ, result, method, mask = new cv.Mat())**

@param image      image where the search is running. It must be 8-bit or 32-bit floating-point.
@param templ      searched template. It must be not greater than the source image and have the same data type.
@param result     map of comparison results. It must be single-channel 32-bit floating-point.
@param method     parameter specifying the comparison method(see cv.TemplateMatchModes).
@param mask       mask of searched template. It must have the same datatype and size with templ. It is not set by default.

Try it
------

\htmlonly
<iframe src="../../js_template_matching_matchTemplate.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly