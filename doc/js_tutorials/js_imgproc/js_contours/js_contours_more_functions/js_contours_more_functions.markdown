Contours : More Functions {#tutorial_js_contours_more_functions}
=========================

Goal
----

In this chapter, we will learn about
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

Here is a demo. Canvas elements named convexityDefectsCanvasInput and convexityDefectsCanvasOutput have been prepared. Choose an image and
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
<div id="convexityDefectsCodeArea">
<h2>Input your code</h2>
<button id="convexityDefectsTryIt" disabled="true" onclick="convexityDefectsExecuteCode()">Try it</button><br>
<textarea rows="24" cols="100" id="convexityDefectsTestCode" spellcheck="false">
var src = cv.imread("convexityDefectsCanvasInput");
var dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(src, src, 100, 200, cv.THRESH_BINARY);
var contours  = new cv.MatVector();
var hierarchy = new cv.Mat();
cv.findContours(src, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE, [0,0]);
var hierarchy = new cv.Mat();
var hull = new cv.Mat();
var defect = new cv.Mat();
var cnt = contours.get(0);
var lineColor = new cv.Scalar(255, 0, 0), circleColor = new cv.Scalar(255, 255, 255);
cv.convexHull(cnt, hull, false, false);
cv.convexityDefects (cnt, hull, defect);
for(var i = 0 ; i < defect.rows; ++i)
{
    let start = [cnt.data32s()[defect.data32s()[i * 4] * 2], cnt.data32s()[defect.data32s()[i * 4] * 2 + 1]]; 
    let end = [cnt.data32s()[defect.data32s()[i * 4 + 1] * 2], cnt.data32s()[defect.data32s()[i * 4 + 1] * 2 + 1]]; 
    let far = [cnt.data32s()[defect.data32s()[i * 4 + 2] * 2], cnt.data32s()[defect.data32s()[i * 4 + 2] * 2 + 1]];
    cv.line(dst, start, end, lineColor, 2, cv.LINE_AA, 0);
    cv.circle(dst, far, 3, circleColor, -1);
}
cv.imshow("convexityDefectsCanvasOutput", dst);
src.delete(); dst.delete(); hierarchy.delete(); contours.delete(); hull.delete(); lineColor.delete(); circleColor.delete(); defect.delete();

</textarea>
<p class="err" id="convexityDefectsErr"></p>
</div>
<div id="convexityDefectsShowcase">
    <div>
        <canvas id="convexityDefectsCanvasInput"></canvas>
        <canvas id="convexityDefectsCanvasOutput"></canvas>
    </div>
    <input type="file" id="convexityDefectsInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function convexityDefectsExecuteCode() {
    var convexityDefectsText = document.getElementById("convexityDefectsTestCode").value;
    try {
        eval(convexityDefectsText);
        document.getElementById("convexityDefectsErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("convexityDefectsErr").innerHTML = err;
    }
}
loadImageToCanvas("shape.jpg", "convexityDefectsCanvasInput");
var convexityDefectsInputElement = document.getElementById("convexityDefectsInput");
convexityDefectsInputElement.addEventListener("change", convexityDefectsHandleFiles, false);
function convexityDefectsHandleFiles(e) {
    var convexityDefectsUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(convexityDefectsUrl, "convexityDefectsCanvasInput");
}
</script>
</body>
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
var dist = cv.pointPolygonTest(contours.get(0), [50, 50], true);
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

Here is a demo. Canvas elements named matchShapesCanvasInput and matchShapesCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

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
<div id="matchShapesCodematchShapes">
<h2>Input your code</h2>
<button id="matchShapesTryIt" disabled="true" onclick="matchShapesExecuteCode()">Try it</button><br>
<textarea rows="16" cols="90" id="matchShapesTestCode" spellcheck="false">
var src = cv.imread("matchShapesCanvasInput");
var dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(src, src, 177, 200, cv.THRESH_BINARY);
var contours  = new cv.MatVector();
var hierarchy = new cv.Mat();
cv.findContours(src, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE, [0,0]);
var contourID0 = 0, contourID1 = 1;
var color0 = new cv.Scalar(255, 0, 0), color1 = new cv.Scalar(0, 0, 255);
// You can try more different conversion
var result = cv.matchShapes(contours.get(contourID0), contours.get(contourID1), 1, 0);
matchShapesOutput.innerHTML = result;
cv.drawContours(dst, contours, contourID0, color0, 1, cv.LINE_8, hierarchy, 100, [0, 0]);
cv.drawContours(dst, contours, contourID1, color1, 1, cv.LINE_8, hierarchy, 100, [0, 0]);
cv.imshow("matchShapesCanvasOutput", dst);
src.delete(); dst.delete(); contours.delete(); hierarchy.delete(); color0.delete(); color1.delete();
</textarea>
<p class="err" id="matchShapesErr"></p>
</div>
<div id="matchShapesShowcase">
    <div>
        <canvas id="matchShapesCanvasInput"></canvas>
        <canvas id="matchShapesCanvasOutput"></canvas>
    </div>
    <input type="file" id="matchShapesInput" name="file" />
    <p><strong>The result is: </strong><span id="matchShapesOutput"></span></p>
</div>
<script>
var matchShapesOutput = document.getElementById("matchShapesOutput");
function matchShapesExecuteCode() {
    var matchShapesText = document.getElementById("matchShapesTestCode").value;
    try {
        eval(matchShapesText);
        document.getElementById("matchShapesErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("matchShapesErr").innerHTML = err;
    }
}

loadImageToCanvas("LinuxLogo.jpg", "matchShapesCanvasInput");
var matchShapesInputElement = document.getElementById("matchShapesInput");
matchShapesInputElement.addEventListener("change", matchShapesHandleFiles, false);
function matchShapesHandleFiles(e) {
    var matchShapesUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(matchShapesUrl, "matchShapesCanvasInput");
}
function onReady() {
    document.getElementById("convexityDefectsTryIt").disabled = false;
    document.getElementById("matchShapesTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly