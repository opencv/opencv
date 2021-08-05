Contours : More Functions {#tutorial_js_contours_more_functions}
=========================

@prev_tutorial{tutorial_js_contour_properties}
@next_tutorial{tutorial_js_contours_hierarchy}

Goal
----

-   Convexity defects and how to find them.
-   Finding shortest distance from a point to a polygon
-   Matching different shapes

Theory and Code
---------------

### 1. Convexity Defects

We saw what is convex hull in second chapter about contours. Any deviation of the object from this
hull can be considered as convexity defect.We can visualize it using an image. We draw a
line joining start point and end point, then draw a circle at the farthest point.

@note Remember we have to pass returnPoints = False while finding convex hull, in order to find
convexity defects.

We use the function: **cv.convexityDefects (contour, convexhull, convexityDefect)**
@param contour              input contour.
@param convexhull           convex hull obtained using convexHull that should contain indices of the contour points that make the hull
@param convexityDefect      the output vector of convexity defects. Each convexity defect is represented as 4-element(start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices in the original contour of the convexity defect beginning, end and the farthest point, and fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the farthest contour point and the hull. That is, to get the floating-point value of the depth will be fixpt_depth/256.0.

Try it
------

\htmlonly
<iframe src="../../js_contours_more_functions_convexityDefects.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 2. Point Polygon Test

This function finds the shortest distance between a point in the image and a contour. It returns the
distance which is negative when point is outside the contour, positive when point is inside and zero
if point is on the contour.

We use the function: **cv.pointPolygonTest (contour, pt, measureDist)**
@param contour      input contour.
@param pt           point tested against the contour.
@param measureDist  if true, the function estimates the signed distance from the point to the nearest contour edge. Otherwise, the function only checks if the point is inside a contour or not.

@code{.js}
let dist = cv.pointPolygonTest(cnt, new cv.Point(50, 50), true);
@endcode

### 3. Match Shapes

OpenCV comes with a function **cv.matchShapes()** which enables us to compare two shapes, or two
contours and returns a metric showing the similarity. The lower the result, the better match it is.
It is calculated based on the hu-moment values. Different measurement methods are explained in the
docs.

We use the function: **cv.matchShapes (contour1, contour2, method, parameter)**
@param contour1      first contour or grayscale image.
@param contour2      second contour or grayscale image.
@param method        comparison method, see cv::ShapeMatchModes
@param parameter     method-specific parameter(not supported now).

Try it
------

\htmlonly
<iframe src="../../js_contours_more_functions_shape.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly