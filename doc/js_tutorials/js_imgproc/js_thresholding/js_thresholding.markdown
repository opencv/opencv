Image Thresholding {#tutorial_js_thresholding}
==================

Goal
----

-   In this tutorial, you will learn Simple thresholding, Adaptive thresholding, Otsu's thresholding
    etc.
-   You will learn these functions : **cv.threshold**, **cv.adaptiveThreshold** etc.

Simple Thresholding
-------------------

Here, the matter is straight forward. If pixel value is greater than a threshold value, it is
assigned one value (may be white), else it is assigned another value (may be black). 

We use the function: **cv.threshold (src, dst, thresh, maxval, type)**
@param src    input array.
@param dst    output array of the same size and type and the same number of channels as src. 
@param thresh threshold value.
@param maxval maximum value to use with the cv.THRESH_BINARY and cv.THRESH_BINARY_INV thresholding types. 
@param type   thresholding type(see cv.ThresholdTypes).

**thresholding type** - OpenCV provides different styles of thresholding and it is decided
by the fourth parameter of the function. Different types are:

-   cv.THRESH_BINARY
-   cv.THRESH_BINARY_INV
-   cv.THRESH_TRUNC
-   cv.THRESH_TOZERO
-   cv.THRESH_OTSU
-   cv.THRESH_TRIANGLE

@note Input image should be single channel only in case of cv.THRESH_OTSU or cv.THRESH_TRIANGLE flags

Try it
------

Try this demo using the code above. Canvas elements named thresholdCanvasInput and thresholdCanvasOutput have been prepared. Choose an image and 
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
<style>
canvas {
    border: 1px solid black;
}
.err{
    color: red;
}
</style>
</head>
<body>
<div id="thresholdCodeArea">
<h2>Input your code</h2>
<button id="thresholdTryIt" disabled="true" onclick="thresholdExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="thresholdTestCode" spellcheck="false">
let src = cv.imread("thresholdCanvasInput");
let dst = new cv.Mat();
// You can try more different parameters
cv.threshold(src, dst, 177, 200, cv.THRESH_BINARY)
cv.imshow("thresholdCanvasOutput", dst);
src.delete();
dst.delete();
</textarea>
<p class="err" id="thresholdErr"></p>
</div>
<div id="thresholdShowcase">
    <div>
        <canvas id="thresholdCanvasInput"></canvas>
        <canvas id="thresholdCanvasOutput"></canvas>
    </div>
    <input type="file" id="thresholdInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function thresholdExecuteCode() {
    let thresholdText = document.getElementById("thresholdTestCode").value;
    try {
        eval(thresholdText);
        document.getElementById("thresholdErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("thresholdErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "thresholdCanvasInput");
let thresholdInputElement = document.getElementById("thresholdInput");
thresholdInputElement.addEventListener("change", thresholdHandleFiles, false);
function thresholdHandleFiles(e) {
    let thresholdUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(thresholdUrl, "thresholdCanvasInput");
}
</script>
</body>
\endhtmlonly

Adaptive Thresholding
---------------------

In the previous section, we used a global value as threshold value. But it may not be good in all
the conditions where image has different lighting conditions in different areas. In that case, we go
for adaptive thresholding. In this, the algorithm calculate the threshold for a small regions of the
image. So we get different thresholds for different regions of the same image and it gives us better
results for images with varying illumination.

We use the function: **cv.adaptiveThreshold (src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)**
@param src             source 8-bit single-channel image.
@param dst             dstination image of the same size and the same type as src. 
@param maxValue        non-zero value assigned to the pixels for which the condition is satisfied
@param adaptiveMethod  adaptive thresholding algorithm to use.
@param thresholdType   thresholding type that must be either cv.THRESH_BINARY or cv.THRESH_BINARY_INV.
@param blockSize       size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
@param C               constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.

**adaptiveMethod** - It decides how thresholding value is calculated:
    -   cv.ADAPTIVE_THRESH_MEAN_C 
    -   cv.ADAPTIVE_THRESH_GAUSSIAN_C 

Try it
------

Try this demo using the code above. Canvas elements named adaptiveThresholdCanvasInput and adaptiveThresholdCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="adaptiveThresholdCodeArea">
<h2>Input your code</h2>
<button id="adaptiveThresholdTryIt" disabled="true" onclick="adaptiveThresholdExecuteCode()">Try it</button><br>
<textarea rows="9" cols="80" id="adaptiveThresholdTestCode" spellcheck="false">
let src = cv.imread("adaptiveThresholdCanvasInput");
let dst = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
// You can try more different parameters
cv.adaptiveThreshold(src, dst, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2)
cv.imshow("adaptiveThresholdCanvasOutput", dst);
src.delete();
dst.delete();
</textarea>
<p class="err" id="adaptiveThresholdErr"></p>
</div>
<div id="adaptiveThresholdShowcase">
    <div>
        <canvas id="adaptiveThresholdCanvasInput"></canvas>
        <canvas id="adaptiveThresholdCanvasOutput"></canvas>
    </div>
    <input type="file" id="adaptiveThresholdInput" name="file" />
</div>
<script>
function adaptiveThresholdExecuteCode() {
    let adaptiveThresholdText = document.getElementById("adaptiveThresholdTestCode").value;
    try {
        eval(adaptiveThresholdText);
        document.getElementById("adaptiveThresholdErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("adaptiveThresholdErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "adaptiveThresholdCanvasInput");
let adaptiveThresholdInputElement = document.getElementById("adaptiveThresholdInput");
adaptiveThresholdInputElement.addEventListener("change", adaptiveThresholdHandleFiles, false);
function adaptiveThresholdHandleFiles(e) {
    let adaptiveThresholdUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(adaptiveThresholdUrl, "adaptiveThresholdCanvasInput");
}
function onReady() {
    document.getElementById("thresholdTryIt").disabled = false;
    document.getElementById("adaptiveThresholdTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
