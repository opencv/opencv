Meanshift and Camshift {#tutorial_js_meanshift}
======================

Goal
----

-   We will learn about Meanshift and Camshift algorithms to find and track objects in videos.

Meanshift
---------

The intuition behind the meanshift is simple. Consider you have a set of points. (It can be a pixel
distribution like histogram backprojection). You are given a small window ( may be a circle) and you
have to move that window to the area of maximum pixel density (or maximum number of points). It is
illustrated in the simple image given below:

![image](images/meanshift_basics.jpg)

The initial window is shown in blue circle with the name "C1". Its original center is marked in blue
rectangle, named "C1_o". But if you find the centroid of the points inside that window, you will
get the point "C1_r" (marked in small blue circle) which is the real centroid of window. Surely
they don't match. So move your window such that circle of the new window matches with previous
centroid. Again find the new centroid. Most probably, it won't match. So move it again, and continue
the iterations such that center of window and its centroid falls on the same location (or with a
small desired error). So finally what you obtain is a window with maximum pixel distribution. It is
marked with green circle, named "C2". As you can see in image, it has maximum number of points. The
whole process is demonstrated on a static image below:

![image](images/meanshift_face.gif)

So we normally pass the histogram backprojected image and initial target location. When the object
moves, obviously the movement is reflected in histogram backprojected image. As a result, meanshift
algorithm moves our window to the new location with maximum density.

### Meanshift in OpenCV.js

To use meanshift in OpenCV.js, first we need to setup the target, find its histogram so that we can
backproject the target on each frame for calculation of meanshift. We also need to provide initial
location of window. For histogram, only Hue is considered here. Also, to avoid false values due to
low light, low light values are discarded using **cv.inRange()** function.

We use the function: **cv.meanShift (probImage, window, criteria)**
@param probImage     Back projection of the object histogram. See cv.calcBackProject for details.
@param window        Initial search window.
@param criteria      Stop criteria for the iterative search algorithm.
@return              number of iterations meanShift took to converge and the new location

### Try it

\htmlonly
<iframe src="../../js_meanshift.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

Camshift
--------

Did you closely watch the last result? There is a problem. Our window always has the same size when
the object is farther away and it is very close to camera. That is not good. We need to adapt the window
size with size and rotation of the target. Once again, the solution came from "OpenCV Labs" and it
is called CAMshift (Continuously Adaptive Meanshift) published by Gary Bradsky in his paper
"Computer Vision Face Tracking for Use in a Perceptual User Interface" in 1988.

It applies meanshift first. Once meanshift converges, it updates the size of the window as,
\f$s = 2 \times \sqrt{\frac{M_{00}}{256}}\f$. It also calculates the orientation of best fitting ellipse
to it. Again it applies the meanshift with new scaled search window and previous window location.
The process is continued until required accuracy is met.

![image](images/camshift_face.gif)

### Camshift in OpenCV.js

It is almost same as meanshift, but it returns a rotated rectangle (that is our result) and box
parameters (used to be passed as search window in next iteration).

We use the function: **cv.CamShift (probImage, window, criteria)**
@param probImage     Back projection of the object histogram. See cv.calcBackProject for details.
@param window        Initial search window.
@param criteria      Stop criteria for the iterative search algorithm.
@return              Rotated rectangle and the new search window

### Try it

\htmlonly
<iframe src="../../js_camshift.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

Additional Resources
--------------------

-#  French Wikipedia page on [Camshift](http://fr.wikipedia.org/wiki/Camshift). (The two animations
    are taken from here)
2.  Bradski, G.R., "Real time face and object tracking as a component of a perceptual user
    interface," Applications of Computer Vision, 1998. WACV '98. Proceedings., Fourth IEEE Workshop
    on , vol., no., pp.214,219, 19-21 Oct 1998
