Contour Features {#tutorial_js_contour_features}
================

Goal
----

-   To find the different features of contours, like area, perimeter, centroid, bounding box etc
-   You will learn plenty of functions related to contours.

1. Moments
----------

Image moments help you to calculate some features like center of mass of the object, area of the
object etc. Check out the wikipedia page on [Image
Moments](http://en.wikipedia.org/wiki/Image_moment)

We use the function: **cv.moments (array, binaryImage = false)**
@param array         raster image (single-channel, 8-bit or floating-point 2D array) or an array ( 1×N or N×1 ) of 2D points.
@param binaryImage   if it is true, all non-zero image pixels are treated as 1's. The parameter is used for images only.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_moments.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

From this moments, you can extract useful data like area, centroid etc. Centroid is given by the
relations, \f$C_x = \frac{M_{10}}{M_{00}}\f$ and \f$C_y = \frac{M_{01}}{M_{00}}\f$. This can be done as
follows:
@code{.js}
let cx = M.m10/M.m00
let cy = M.m01/M.m00
@endcode

2. Contour Area
---------------

Contour area is given by the function **cv.contourArea()** or from moments, **M['m00']**.

We use the function: **cv.contourArea (contour, oriented = false)**
@param contour    input vector of 2D points (contour vertices)
@param oriented   oriented area flag. If it is true, the function returns a signed area value, depending on the contour orientation (clockwise or counter-clockwise). Using this feature you can determine orientation of a contour by taking the sign of an area. By default, the parameter is false, which means that the absolute value is returned.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_area.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

3. Contour Perimeter
--------------------

It is also called arc length. It can be found out using **cv.arcLength()** function.

We use the function: **cv.arcLength (curve, closed)**
@param curve    input vector of 2D points.
@param closed   flag indicating whether the curve is closed or not.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_perimeter.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

4. Contour Approximation
------------------------

It approximates a contour shape to another shape with less number of vertices depending upon the
precision we specify. It is an implementation of [Douglas-Peucker
algorithm](http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm). Check the wikipedia page
for algorithm and demonstration.

We use the function: **cv.approxPolyDP (curve, approxCurve, epsilon, closed)**
@param curve        input vector of 2D points stored in cv.Mat.
@param approxCurve  result of the approximation. The type should match the type of the input curve.
@param epsilon      parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation.
@param closed       If true, the approximated curve is closed (its first and last vertices are connected). Otherwise, it is not closed.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_approxPolyDP.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

5. Convex Hull
--------------

Convex Hull will look similar to contour approximation, but it is not (Both may provide same results
in some cases). Here, **cv.convexHull()** function checks a curve for convexity defects and
corrects it. Generally speaking, convex curves are the curves which are always bulged out, or
at-least flat. And if it is bulged inside, it is called convexity defects. For example, check the
below image of hand. Red line shows the convex hull of hand. The double-sided arrow marks shows the
convexity defects, which are the local maximum deviations of hull from contours.

![image](images/convexitydefects.jpg)

We use the function: **cv.convexHull (points, hull, clockwise = false, returnPoints = true)**
@param points        input 2D point set.
@param hull          output convex hull.
@param clockwise     orientation flag. If it is true, the output convex hull is oriented clockwise. Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing to the right, and its Y axis pointing upwards.
@param returnPoints  operation flag. In case of a matrix, when the flag is true, the function returns convex hull points. Otherwise, it returns indices of the convex hull points.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_convexHull.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

6. Checking Convexity
---------------------

There is a function to check if a curve is convex or not, **cv.isContourConvex()**. It just return
whether True or False. Not a big deal.

@code{.js}
cv.isContourConvex(cnt);
@endcode

7. Bounding Rectangle
---------------------

There are two types of bounding rectangles.

### 7.a. Straight Bounding Rectangle

It is a straight rectangle, it doesn't consider the rotation of the object. So area of the bounding
rectangle won't be minimum.

We use the function: **cv.boundingRect (points)**
@param points        input 2D point set.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_boundingRect.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 7.b. Rotated Rectangle

Here, bounding rectangle is drawn with minimum area, so it considers the rotation also.

We use the function: **cv.minAreaRect (points)**
@param points        input 2D point set.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_minAreaRect.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

8. Minimum Enclosing Circle
---------------------------

Next we find the circumcircle of an object using the function **cv.minEnclosingCircle()**. It is a
circle which completely covers the object with minimum area.

We use the functions: **cv.minEnclosingCircle (points)**
@param points        input 2D point set.

**cv.circle (img, center, radius, color, thickness = 1, lineType = cv.LINE_8, shift = 0)**
@param img          image where the circle is drawn.
@param center       center of the circle.
@param radius       radius of the circle.
@param color        circle color.
@param thickness    thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn.
@param lineType     type of the circle boundary.
@param shift        number of fractional bits in the coordinates of the center and in the radius value.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_minEnclosingCircle.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

9. Fitting an Ellipse
---------------------

Next one is to fit an ellipse to an object. It returns the rotated rectangle in which the ellipse is
inscribed.
We use the functions: **cv.fitEllipse (points)**
@param points        input 2D point set.

**cv.ellipse1 (img, box, color, thickness = 1, lineType = cv.LINE_8)**
@param img        image.
@param box        alternative ellipse representation via RotatedRect. This means that the function draws an ellipse inscribed in the rotated rectangle.
@param color      ellipse color.
@param thickness  thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a filled ellipse sector is to be drawn.
@param lineType   type of the ellipse boundary.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_fitEllipse.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

10. Fitting a Line
------------------

Similarly we can fit a line to a set of points. We can approximate a straight line to it.

We use the functions: **cv.fitLine (points, line, distType, param, reps, aeps)**
@param points     input 2D point set.
@param line       output line parameters. It should be a Mat of 4 elements[vx, vy, x0, y0], where [vx, vy] is a normalized vector collinear to the line and [x0, y0] is a point on the line.
@param distType   distance used by the M-estimator(see cv.DistanceTypes).
@param param      numerical parameter ( C ) for some types of distances. If it is 0, an optimal value is chosen.
@param reps       sufficient accuracy for the radius (distance between the coordinate origin and the line).
@param aeps       sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.

**cv.line (img, pt1, pt2, color, thickness = 1, lineType = cv.LINE_8, shift = 0)**
@param img          image.
@param pt1          first point of the line segment.
@param pt2          second point of the line segment.
@param color        line color.
@param thickness    line thickness.
@param lineType     type of the line,.
@param shift        number of fractional bits in the point coordinates.

Try it
------

\htmlonly
<iframe src="../../js_contour_features_fitLine.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly