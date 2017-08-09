Hough Circle Transform {#tutorial_js_houghcircles}
======================

Goal
----

-   We will learn to use Hough Transform to find circles in an image.
-   We will learn these functions: **cv.HoughCircles()**

Theory
------

A circle is represented mathematically as \f$(x-x_{center})^2 + (y - y_{center})^2 = r^2\f$ where
\f$(x_{center},y_{center})\f$ is the center of the circle, and \f$r\f$ is the radius of the circle. From
equation, we can see we have 3 parameters, so we need a 3D accumulator for hough transform, which
would be highly ineffective. So OpenCV uses more trickier method, **Hough Gradient Method** which
uses the gradient information of edges.

We use the function: **cv.HoughCircles (image, circles, method, dp, minDist, param1 = 100, param2 = 100, minRadius = 0, maxRadius = 0)** 

@param image       8-bit, single-channel, grayscale input image.
@param circles     output vector of found circles(cv.CV_32FC3 type). Each vector is encoded as a 3-element floating-point vector (x,y,radius) .
@param method      detection method(see cv.HoughModes). Currently, the only implemented method is HOUGH_GRADIENT
@param dp      	   inverse ratio of the accumulator resolution to the image resolution. For example, if dp = 1 , the accumulator has the same resolution as the input image. If dp = 2 , the accumulator has half as big width and height.
@param minDist     minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
@param param1      first method-specific parameter. In case of HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
@param param2      second method-specific parameter. In case of HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
@param minRadius   minimum circle radius.
@param maxRadius   maximum circle radius.

Try it
------

Try this demo using the code above. Canvas elements named HoughCirclesPCanvasInput and HoughCirclesPCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

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
<div id="HoughCirclesPCodeArea">
<h2>Input your code</h2>
<button id="HoughCirclesPTryIt" disabled="true" onclick="HoughCirclesPExecuteCode()">Try it</button><br>
<textarea rows="17" cols="80" id="HoughCirclesPTestCode" spellcheck="false">
let src = cv.imread("HoughCirclesPCanvasInput");
let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8U);
let circles = new cv.Mat();
let color = new cv.Scalar(255, 0, 0);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
// You can try more different parameters
cv.HoughCircles(src, circles, cv.HOUGH_GRADIENT, 1, 45, 75, 40, 0, 0);
// draw circles
for (let i = 0; i < circles.cols; ++i) {
    let x = circles.data32F[i * 3];
    let y = circles.data32F[i * 3 + 1];
    let radius = circles.data32F[i * 3 + 2];
    let center = new cv.Point(x, y);
    cv.circle(dst, center, radius, color);
}
cv.imshow("HoughCirclesPCanvasOutput", dst);
src.delete(); dst.delete(); circles.delete();
</textarea>
<p class="err" id="HoughCirclesPErr"></p>
</div>
<div id="HoughCirclesPShowcase">
    <div>
        <canvas id="HoughCirclesPCanvasInput"></canvas>
        <canvas id="HoughCirclesPCanvasOutput"></canvas>
    </div>
    <input type="file" id="HoughCirclesPInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function HoughCirclesPExecuteCode() {
    let HoughCirclesPText = document.getElementById("HoughCirclesPTestCode").value;
    try {
        eval(HoughCirclesPText);
        document.getElementById("HoughCirclesPErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("HoughCirclesPErr").innerHTML = err;
    }
}

loadImageToCanvas("coins.jpg", "HoughCirclesPCanvasInput");
let HoughCirclesPInputElement = document.getElementById("HoughCirclesPInput");
HoughCirclesPInputElement.addEventListener("change", HoughCirclesPHandleFiles, false);
function HoughCirclesPHandleFiles(e) {
    let HoughCirclesPUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(HoughCirclesPUrl, "HoughCirclesPCanvasInput");
}

function onReady() {
    document.getElementById("HoughCirclesPTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
