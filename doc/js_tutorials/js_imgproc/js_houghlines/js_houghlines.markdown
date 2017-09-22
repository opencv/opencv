Hough Line Transform {#tutorial_js_houghlines}
====================

Goal
----

-   We will understand the concept of the Hough Transform.
-   We will learn how to use it to detect lines in an image.
-   We will learn the following functions: **cv.HoughLines()**, **cv.HoughLinesP()**

Theory
------

The Hough Transform is a popular technique to detect any shape, if you can represent that shape in a
mathematical form. It can detect the shape even if it is broken or distorted a little bit. We will
see how it works for a line.

A line can be represented as \f$y = mx+c\f$ or in a parametric form, as
\f$\rho = x \cos \theta + y \sin \theta\f$ where \f$\rho\f$ is the perpendicular distance from the origin to the
line, and \f$\theta\f$ is the angle formed by this perpendicular line and the horizontal axis measured in
counter-clockwise (That direction varies on how you represent the coordinate system. This
representation is used in OpenCV). Check the image below:

![image](images/houghlines1.svg)

So if the line is passing below the origin, it will have a positive rho and an angle less than 180. If it
is going above the origin, instead of taking an angle greater than 180, the angle is taken less than 180,
and rho is taken negative. Any vertical line will have 0 degree and horizontal lines will have 90
degree.

Now let's see how the Hough Transform works for lines. Any line can be represented in these two terms,
\f$(\rho, \theta)\f$. So first it creates a 2D array or accumulator (to hold the values of the two parameters)
and it is set to 0 initially. Let rows denote the \f$\rho\f$ and columns denote the \f$\theta\f$. Size of
array depends on the accuracy you need. Suppose you want the accuracy of angles to be 1 degree, you will
need 180 columns. For \f$\rho\f$, the maximum distance possible is the diagonal length of the image. So
taking one pixel accuracy, the number of rows can be the diagonal length of the image.

Consider a 100x100 image with a horizontal line at the middle. Take the first point of the line. You
know its (x,y) values. Now in the line equation, put the values \f$\theta = 0,1,2,....,180\f$ and check
the \f$\rho\f$ you get. For every \f$(\rho, \theta)\f$ pair, you increment value by one in our accumulator
in its corresponding \f$(\rho, \theta)\f$ cells. So now in accumulator, the cell (50,90) = 1 along with
some other cells.

Now take the second point on the line. Do the same as above. Increment the values in the cells
corresponding to \f$(\rho, \theta)\f$ you got. This time, the cell (50,90) = 2. What you actually
do is voting the \f$(\rho, \theta)\f$ values. You continue this process for every point on the line. At
each point, the cell (50,90) will be incremented or voted up, while other cells may or may not be
voted up. This way, at the end, the cell (50,90) will have maximum votes. So if you search the
accumulator for maximum votes, you get the value (50,90) which says, there is a line in this image
at a distance 50 from the origin and at angle 90 degrees. It is well shown in the below animation (Image
Courtesy: [Amos Storkey](http://homepages.inf.ed.ac.uk/amos/hough.html) )

![](houghlinesdemo.gif)

This is how hough transform works for lines. It is simple. Below is an image which shows the accumulator. Bright spots at some locations
denote they are the parameters of possible lines in the image. (Image courtesy: [Wikipedia](http://en.wikipedia.org/wiki/Hough_transform) )

![](houghlines2.jpg)

Hough Transform in OpenCV
=========================

Everything explained above is encapsulated in the OpenCV function, **cv.HoughLines()**. It simply returns an array of (\f$(\rho, \theta)\f$ values. \f$\rho\f$ is measured in pixels and \f$\theta\f$ is measured in radians. First parameter,
Input image should be a binary image, so apply threshold or use canny edge detection before
applying hough transform.

We use the function: **cv.HoughLines (image, lines, rho, theta, threshold, srn = 0, stn = 0, min_theta = 0, max_theta = Math.PI)**
@param image       8-bit, single-channel binary source image. The image may be modified by the function.
@param lines       output vector of lines(cv.32FC2 type). Each line is represented by a two-element vector (ρ,θ) . ρ is the distance from the coordinate origin (0,0). θ is the line rotation angle in radians.
@param rho    	   distance resolution of the accumulator in pixels.
@param theta       angle resolution of the accumulator in radians.
@param threshold   accumulator threshold parameter. Only those lines are returned that get enough votes
@param srn         for the multi-scale Hough transform, it is a divisor for the distance resolution rho . The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these parameters should be positive.
@param stn         for the multi-scale Hough transform, it is a divisor for the distance resolution theta.
@param min_theta   for standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
@param max_theta   for standard and multi-scale Hough transform, maximum angle to check for lines. Must fall between min_theta and CV_PI.

Try it
------

\htmlonly
<iframe src="../../js_houghlines_HoughLines.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

Probabilistic Hough Transform
-----------------------------

In the hough transform, you can see that even for a line with two arguments, it takes a lot of
computation. Probabilistic Hough Transform is an optimization of the Hough Transform we saw. It doesn't
take all the points into consideration. Instead, it takes only a random subset of points which is
sufficient for line detection. Just we have to decrease the threshold. See image below which compares
Hough Transform and Probabilistic Hough Transform in Hough space. (Image Courtesy :
[Franck Bettinger's home page](http://phdfb1.free.fr/robot/mscthesis/node14.html) )

![image](images/houghlines4.png)

OpenCV implementation is based on Robust Detection of Lines Using the Progressive Probabilistic
Hough Transform by Matas, J. and Galambos, C. and Kittler, J.V. @cite Matas00.

We use the function: **cv.HoughLinesP (image, lines, rho, theta, threshold, minLineLength = 0, maxLineGap = 0)**

@param image          8-bit, single-channel binary source image. The image may be modified by the function.
@param lines          output vector of lines(cv.32SC4 type). Each line is represented by a 4-element vector (x1,y1,x2,y2) ,where (x1,y1) and (x2,y2) are the ending points of each detected line segment.
@param rho            distance resolution of the accumulator in pixels.
@param theta          angle resolution of the accumulator in radians.
@param threshold      accumulator threshold parameter. Only those lines are returned that get enough votes
@param minLineLength  minimum line length. Line segments shorter than that are rejected.
@param maxLineGap     maximum allowed gap between points on the same line to link them.

Try it
------

\htmlonly
<iframe src="../../js_houghlines_HoughLinesP.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly