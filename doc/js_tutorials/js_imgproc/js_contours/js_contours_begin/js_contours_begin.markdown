Contours : Getting Started {#tutorial_js_contours_begin}
==========================

Goal
----

-   Understand what contours are.
-   Learn to find contours, draw contours etc
-   You will see these functions : **cv.findContours()**, **cv.drawContours()**

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

We use the functions: **cv.findContours (image, contours, hierarchy, mode, method, offset = [0, 0])** 
@param image         source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary. 
@param contours      detected contours. 
@param hierarchy     containing information about the image topology. It has as many elements as the number of contours. 
@param mode          contour retrieval mode(see cv.RetrievalModes).
@param method        contour approximation method(see cv.ContourApproximationModes).
@param offset        optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.

**cv.drawContours (image, contours, contourIdx, color, thickness = 1, lineType = cv.LINE_8, hierarchy = cv.Mat(), maxLevel = INT_MAX, offset = [0, 0])** 
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

Here is a demo. Canvas elements named contoursCanvasInput and contoursCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
<style>
canvas {
    border: 1px solid black;
}
.err {
    color: red;
}
</style>
</head>
<body>
<div id="contoursCodeArea">
<h2>Input your code</h2>
<button id="contoursTryIt" disabled="true" onclick="contoursExecuteCode()">Try it</button><br>
<textarea rows="17" cols="90" id="contoursTestCode" spellcheck="false">
var src = cv.imread("contoursCanvasInput");
var dst = new cv.Mat.zeros(src.cols, src.rows, cv.CV_8UC3);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(src, src, 120, 200, cv.THRESH_BINARY);
var contours  = new cv.MatVector();
var hierarchy = new cv.Mat();
// You can try more different conversion
cv.findContours(src, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE, [0, 0]);
for (var i = 0; i < contours.size(); ++i)
{
    var color = new cv.Scalar(Math.random()*255, Math.random()*255, Math.random()*255);
    cv.drawContours(dst, contours, i, color, 1, cv.LINE_8, hierarchy, 100, [0, 0]);
    color.delete();
}
cv.imshow("contoursCanvasOutput", dst);
src.delete(); dst.delete(); contours.delete(); hierarchy.delete(); 
</textarea>
<p class="err" id="contoursErr"></p>
</div>
<div id="contoursShowcase">
    <div>
        <canvas id="contoursCanvasInput"></canvas>
        <canvas id="contoursCanvasOutput"></canvas>
    </div>
    <input type="file" id="contoursInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function contoursExecuteCode() {
    var contoursText = document.getElementById("contoursTestCode").value;
    try {
        eval(contoursText);
        document.getElementById("contoursErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("contoursErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "contoursCanvasInput");
var contoursInputElement = document.getElementById("contoursInput");
contoursInputElement.addEventListener("change", contoursHandleFiles, false);
function contoursHandleFiles(e) {
    var contoursUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(contoursUrl, "contoursCanvasInput");
}

function onReady() {
    document.getElementById("contoursTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
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
cv2.CHAIN_APPROX_SIMPLE does. It removes all redundant points and compresses the contour, thereby
saving memory.