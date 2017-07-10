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

We use the function: **cv.threshold(src, dst, thresh, maxval, type)**
@param src    input array.
@param dst    output array of the same size and type and the same number of channels as src. 
@param thresh threshold value.
@param maxval maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types. 
@param type   thresholding type

**Warning:** 

Input image should be single channel only in case of cv.ThresholdTypes.THRESH_OTSU.value or cv.ThresholdTypes.THRESH_TRIANGLE.value flags

OpenCV provides different styles of thresholding and it is decided
by the fourth parameter of the function. Different types are:

-   cv.ThresholdTypes.THRESH_BINARY.value
-   cv.ThresholdTypes.THRESH_BINARY_INV.value
-   cv.ThresholdTypes.THRESH_TRUNC.value
-   cv.ThresholdTypes.THRESH_TOZERO.value
-   cv.ThresholdTypes.THRESH_OTSU.value
-   cv.ThresholdTypes.THRESH_TRIANGLE.value

Documentation clearly explain what each type is meant for. Please check out the documentation.

Try it
------

Here is a demo. Canvas elements named thresholdCanvas1 and thresholdCanvas2 have been prepared. Choose an image and 
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
<div id="thresholdCodeArea">
<h2>Input your code</h2>
<button id="thresholdTryIt" disabled="true" onclick="thresholdExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="thresholdTestCode" spellcheck="false">
var src = cv.imread("thresholdCanvas1");
var dst = new cv.Mat();
// You can try more different conversion
cv.threshold(src, dst, 177, 200, cv.ThresholdTypes.THRESH_BINARY.value)
cv.imshow("thresholdCanvas2", dst);
src.delete();
dst.delete();
</textarea>
</div>
<div id="thresholdShowcase">
    <div>
        <canvas id="thresholdCanvas1"></canvas>
        <canvas id="thresholdCanvas2"></canvas>
    </div>
    <input type="file" id="thresholdInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function thresholdExecuteCode() {
    var thresholdText = document.getElementById("thresholdTestCode").value;
    eval(thresholdText);
}

loadImageToCanvas("lena.jpg", "thresholdCanvas1");
var thresholdInputElement = document.getElementById("thresholdInput");
thresholdInputElement.addEventListener("change", thresholdHandleFiles, false);
function thresholdHandleFiles(e) {
    var thresholdUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(thresholdUrl, "thresholdCanvas1");
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

We use the function: **cv.adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)**
@param src             source 8-bit single-channel image.
@param dst             dstination image of the same size and the same type as src. 
@param maxValue Non-zero value assigned to the pixels for which the condition is satisfied
@param adaptiveMethod  adaptive thresholding algorithm to use.
@param thresholdType   thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV.
@param blockSize       size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
@param C               constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.

**adaptiveMethod** - It decides how thresholding value is calculated.
    -   cv.AdaptiveThresholdTypes.ADAPTIVE_THRESH_MEAN_C.value : threshold value is the mean of neighbourhood area.
    -   cv.AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C.value : threshold value is the weighted sum of neighbourhood
        values where weights are a gaussian window.

Try it
------

Here is a demo. Canvas elements named adaptiveThresholdCanvas1 and adaptiveThresholdCanvas2 have been prepared. Choose an image and
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
<div id="adaptiveThresholdCodeArea">
<h2>Input your code</h2>
<button id="adaptiveThresholdTryIt" disabled="true" onclick="adaptiveThresholdExecuteCode()">Try it</button><br>
<textarea rows="11" cols="80" id="adaptiveThresholdTestCode" spellcheck="false">
var src = cv.imread("adaptiveThresholdCanvas1");
var dst = new cv.Mat();
cv.cvtColor(src, src, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
// You can try more different conversion
cv.adaptiveThreshold(src, dst, 200, cv.AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C.value, cv.ThresholdTypes.THRESH_BINARY.value, 3, 2)
cv.imshow("adaptiveThresholdCanvas2", dst);
src.delete();
dst.delete();
</textarea>
</div>
<div id="adaptiveThresholdShowcase">
    <div>
        <canvas id="adaptiveThresholdCanvas1"></canvas>
        <canvas id="adaptiveThresholdCanvas2"></canvas>
    </div>
    <input type="file" id="adaptiveThresholdInput" name="file" />
</div>
<script>
function adaptiveThresholdExecuteCode() {
    var adaptiveThresholdText = document.getElementById("adaptiveThresholdTestCode").value;
    eval(adaptiveThresholdText);
}

loadImageToCanvas("lena.jpg", "adaptiveThresholdCanvas1");
var adaptiveThresholdInputElement = document.getElementById("adaptiveThresholdInput");
adaptiveThresholdInputElement.addEventListener("change", adaptiveThresholdHandleFiles, false);
function adaptiveThresholdHandleFiles(e) {
    var adaptiveThresholdUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(adaptiveThresholdUrl, "adaptiveThresholdCanvas1");
}

document.getElementById("opencvjs").onload = function() {
    document.getElementById("thresholdTryIt").disabled = false;
    document.getElementById("adaptiveThresholdTryIt").disabled = false;
};
</script>
</body>
\endhtmlonly
