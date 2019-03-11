Contour Properties {#tutorial_py_contour_properties}
==================

Here we will learn to extract some frequently used properties of objects like Solidity, Equivalent
Diameter, Mask image, Mean Intensity etc. More features can be found at [Matlab regionprops
documentation](http://www.mathworks.in/help/images/ref/regionprops.html).

*(NB : Centroid, Area, Perimeter etc also belong to this category, but we have seen it in last
chapter)*

1. Aspect Ratio
---------------

It is the ratio of width to height of bounding rect of the object.

\f[Aspect \; Ratio = \frac{Width}{Height}\f]
@code{.py}
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
@endcode

2. Extent
---------

Extent is the ratio of contour area to bounding rectangle area.

\f[Extent = \frac{Object \; Area}{Bounding \; Rectangle \; Area}\f]
@code{.py}
area = cv.contourArea(cnt)
x,y,w,h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area
@endcode

3. Solidity
-----------

Solidity is the ratio of contour area to its convex hull area.

\f[Solidity = \frac{Contour \; Area}{Convex \; Hull \; Area}\f]
@code{.py}
area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area
@endcode

4. Equivalent Diameter
----------------------

Equivalent Diameter is the diameter of the circle whose area is same as the contour area.

\f[Equivalent \; Diameter = \sqrt{\frac{4 \times Contour \; Area}{\pi}}\f]
@code{.py}
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
@endcode

5. Orientation
--------------

Orientation is the angle at which object is directed. Following method also gives the Major Axis and
Minor Axis lengths.
@code{.py}
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)
@endcode

6. Mask and Pixel Points
------------------------

In some cases, we may need all the points which comprises that object. It can be done as follows:
@code{.py}
mask = np.zeros(imgray.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)
@endcode
Here, two methods, one using Numpy functions, next one using OpenCV function (last commented line)
are given to do the same. Results are also same, but with a slight difference. Numpy gives
coordinates in **(row, column)** format, while OpenCV gives coordinates in **(x,y)** format. So
basically the answers will be interchanged. Note that, **row = x** and **column = y**.

7. Maximum Value, Minimum Value and their locations
---------------------------------------------------

We can find these parameters using a mask image.
@code{.py}
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)
@endcode

8. Mean Color or Mean Intensity
-------------------------------

Here, we can find the average color of an object. Or it can be average intensity of the object in
grayscale mode. We again use the same mask to do it.
@code{.py}
mean_val = cv.mean(im,mask = mask)
@endcode

9. Extreme Points
-----------------

Extreme Points means topmost, bottommost, rightmost and leftmost points of the object.
@code{.py}
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
@endcode
For eg, if I apply it to an Indian map, I get the following result :

![image](images/extremepoints.jpg)

Additional Resources
--------------------

Exercises
---------

-#  There are still some features left in matlab regionprops doc. Try to implement them.
