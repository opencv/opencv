Hough Line Transform (cv.HoughLines() is not in the white list) {#tutorial_js_houghlines}
====================

Goal
----

In this chapter,
    -   We will understand the concept of the Hough Transform.
    -   We will see how to use it to detect lines in an image.
    -   We will see the following functions: **cv.HoughLines()**, **cv.HoughLinesP()**

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
@param lines       output vector of lines. Each line is represented by a two-element vector (ρ,θ) . ρ is the distance from the coordinate origin (0,0). θ is the line rotation angle in radians.
@param rho    	   distance resolution of the accumulator in pixels.
@param theta       angle resolution of the accumulator in radians.
@param threshold   accumulator threshold parameter. Only those lines are returned that get enough votes
@param srn         for the multi-scale Hough transform, it is a divisor for the distance resolution rho . The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these parameters should be positive.
@param stn         for the multi-scale Hough transform, it is a divisor for the distance resolution theta.
@param min_theta   for standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
@param max_theta   for standard and multi-scale Hough transform, maximum angle to check for lines. Must fall between min_theta and CV_PI.

Try it
------

Here is a demo. Canvas elements named HoughLinesCanvasInput and HoughLinesCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

@note cv.minMaxLoc() can finds the global minimum and maximum in an array.

\htmlonly
<!DOCTYPE html>
<head>
<style>
canvas {
    border: 1px solid black;
}
</style>
</head>
<body>
<div id="HoughLinesCodeArea">
<h2>Input your code</h2>
<button id="HoughLinesTryIt" disabled="true" onclick="HoughLinesExecuteCode()">Try it</button><br>
<textarea rows="17" cols="80" id="HoughLinesTestCode" spellcheck="false">
var src = cv.imread("HoughLinesCanvasInput");
cv.cvtColor(src, src, cv.COLOR_BGRA2GRAY, 0);

cv.imshow("HoughLinesCanvasOutput", dst);
src.delete(); dst.delete(); 
</textarea>
</div>
<div id="HoughLinesShowcase">
    <div>
        <canvas id="HoughLinesCanvasInput"></canvas>
        <canvas id="HoughLinesCanvasOutput"></canvas>
    </div>
    <input type="file" id="HoughLinesInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function HoughLinesExecuteCode() {
    var HoughLinesText = document.getElementById("HoughLinesTestCode").value;
    eval(HoughLinesText);
}

loadImageToCanvas("lena.jpg", "HoughLinesCanvasInput");
var HoughLinesInputElement = document.getElementById("HoughLinesInput");
HoughLinesInputElement.addEventListener("change", HoughLinesHandleFiles, false);
function HoughLinesHandleFiles(e) {
    var HoughLinesUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(HoughLinesUrl, "HoughLinesCanvasInput");
}

function onReady() {
    document.getElementById("HoughLinesTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
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
Hough Transform by Matas, J. and Galambos, C. and Kittler, J.V. @cite Matas00. The function used is
**cv2.HoughLinesP()**. It has two new arguments.
-   **minLineLength** - Minimum length of line. Line segments shorter than this are rejected.
-   **maxLineGap** - Maximum allowed gap between line segments to treat them as a single line.

Best thing is that, it directly returns the two endpoints of lines. In previous case, you got only
the parameters of lines, and you had to find all the points. Here, everything is direct and simple.
@include probabilistic_hough_line_transform.py
See the results below:

![image](images/houghlines5.jpg)

Additional Resources
--------------------

-#  [Hough Transform on Wikipedia](http://en.wikipedia.org/wiki/Hough_transform)

Exercises
---------
