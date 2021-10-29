Contours : Getting Started {#tutorial_js_contours_begin}
==========================

@next_tutorial{tutorial_js_contour_features}

Goal
----

-   Understand what contours are.
-   Learn to find contours, draw contours etc
-   You will learn these functions : **cv.findContours()**, **cv.drawContours()**

What are contours?
------------------

Contours can be explained simply as a curve joining all the continuous points (along the boundary),
having same color or intensity. The contours are a useful tool for shape analysis and object
detection and recognition.

-   For better accuracy, use binary images. So before finding contours, apply threshold or canny
    edge detection.
-   Since opencv 3.2 source image is not modified by this function.
-   In OpenCV, finding contours is like finding white object from black background. So remember,
    object to be found should be white and background should be black.

How to draw the contours?
-------------------------

To draw the contours, cv.drawContours function is used. It can also be used to draw any shape
provided you have its boundary points.

We use the functions: **cv.findContours (image, contours, hierarchy, mode, method, offset = new cv.Point(0, 0))**
@param image         source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary.
@param contours      detected contours.
@param hierarchy     containing information about the image topology. It has as many elements as the number of contours.
@param mode          contour retrieval mode(see cv.RetrievalModes).
@param method        contour approximation method(see cv.ContourApproximationModes).
@param offset        optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.

**cv.drawContours (image, contours, contourIdx, color, thickness = 1, lineType = cv.LINE_8, hierarchy = new cv.Mat(), maxLevel = INT_MAX, offset = new cv.Point(0, 0))**
@param image         destination image.
@param contours      all the input contours.
@param contourIdx    parameter indicating a contour to draw. If it is negative, all the contours are drawn.
@param color         color of the contours.
@param thickness     thickness of lines the contours are drawn with. If it is negative, the contour interiors are drawn.
@param lineType      line connectivity(see cv.LineTypes).
@param hierarchy     optional information about hierarchy. It is only needed if you want to draw only some of the contours(see maxLevel).

@param maxLevel      maximal level for drawn contours. If it is 0, only the specified contour is drawn. If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This parameter is only taken into account when there is hierarchy available.
@param offset        optional contour shift parameter.

Try it
------

\htmlonly
<iframe src="../../js_contours_begin_contours.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

Contour Approximation Method
============================

This is the fifth argument in cv.findContours function. What does it denote actually?

Above, we told that contours are the boundaries of a shape with same intensity. It stores the (x,y)
coordinates of the boundary of a shape. But does it store all the coordinates ? That is specified by
this contour approximation method.

If you pass cv.ContourApproximationModes.CHAIN_APPROX_NONE.value, all the boundary points are stored. But actually do we need all
the points? For eg, you found the contour of a straight line. Do you need all the points on the line
to represent that line? No, we need just two end points of that line. This is what
cv.CHAIN_APPROX_SIMPLE does. It removes all redundant points and compresses the contour, thereby
saving memory.
