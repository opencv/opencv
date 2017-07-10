Changing Colorspaces {#tutorial_js_colorspaces}
====================

Goal
----

-   In this tutorial, you will learn how to convert images from one color-space to another, like
    BGR \f$\leftrightarrow\f$ Gray, BGR \f$\leftrightarrow\f$ HSV etc.
-   You will learn following functions : **cv.cvtColor()**, **cv.inRange()** etc.

cvtColor
--------------------

There are more than 150 color-space conversion methods available in OpenCV. But we will look into
the most widely used one: BGR \f$\leftrightarrow\f$ Gray.

We use the function: **cv.cvtColor(src, dst, code, dstCn)**
@param src    input image.
@param dst    output image.
@param code   color space conversion code.
@param dstCn  number of channels in the destination image; if the parameter is 0, the number of the channels is derived automatically from src and code.

For BGR \f$\rightarrow\f$ Gray conversion we use the codes cv.ColorConversionCodes.COLOR_RGBA2GRAY.value.

Try it
------

Here is a demo. Canvas elements named cvtColorCanvas1 and cvtColorCanvas2 have been prepared. Choose an image and 
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
<div id="cvtColorCodeArea">
<h2>Input your code</h2>
<button id="cvtColorTryIt" disabled="true" onclick="cvtColorExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="cvtColorTestCode" spellcheck="false">
var src = cv.imread("cvtColorCanvas1");
var dst = new cv.Mat();
// You can try more different conversion
cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
//cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_RGB2HSV.value, 0);
cv.imshow("cvtColorCanvas2", dst);
src.delete();
dst.delete();
</textarea>
</div>
<div id="cvtColorShowcase">
    <div>
        <canvas id="cvtColorCanvas1"></canvas>
        <canvas id="cvtColorCanvas2"></canvas>
    </div>
    <input type="file" id="cvtColorInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function cvtColorExecuteCode() {
    var cvtColorText = document.getElementById("cvtColorTestCode").value;
    eval(cvtColorText);
}

loadImageToCanvas("lena.jpg", "cvtColorCanvas1");
var cvtColorInputElement = document.getElementById("cvtColorInput");
cvtColorInputElement.addEventListener("change", cvtColorHandleFiles, false);
function cvtColorHandleFiles(e) {
    var cvtColorUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(cvtColorUrl, "cvtColorCanvas1");
}

</script>
</body>
\endhtmlonly

inRange
---------------

Checks if array elements lie between the elements of two other arrays.

We use the function: **cv.inRange(src, lowerb, upperb, dst)**
@param src     first input image.
@param lowerb  inclusive lower boundary Mat of the same size as src. 
@param upperb  inclusive upper boundary Mat of the same size as src. 
@param dst     output image of the same size as src and CV_8U type.

Try it
------

Here is a demo. Canvas elements named inRangeCanvas1 and inRangeCanvas2 have been prepared. Choose an image and 
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
<div id="inRangeCodeArea">
<h2>Input your code</h2>
<button id="inRangeTryIt" disabled="true" onclick="inRangeExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="inRangeTestCode" spellcheck="false">
var src = cv.imread("inRangeCanvas1");
var dst = new cv.Mat();
var src = cv.imread("inRangeCanvas1");
var dst = new cv.Mat();
function matFromScalar (cols, rows, type, scalar) {
    var scalarMat = new cv.Mat(cols, rows, type);
    for (var i=0; i<cols*rows*scalarMat.channels(); ++i) {
        scalarMat.data()[i] = scalar[i%scalarMat.channels()];
    }
    return scalarMat;
}
var low = matFromScalar(src.cols , src.rows, src.type(), [0,0,0,0]);
var high = matFromScalar(src.cols , src.rows, src.type(), [200,150,200,255]);
cv.inRange(src, low, high, dst);
console.warn(dst.data());
cv.cvtColor(dst, dst, cv.ColorConversionCodes.COLOR_GRAY2RGBA.value, 0);
cv.imshow("inRangeCanvas2", dst);
src.delete();
dst.delete();
</textarea>
</div>
<div id="inRangeShowcase">
    <div>
        <canvas id="inRangeCanvas1"></canvas>
        <canvas id="inRangeCanvas2"></canvas>
    </div>
    <input type="file" id="inRangeInput" name="file" />
</div>
<script>
function inRangeExecuteCode() {
    var inRangeText = document.getElementById("inRangeTestCode").value;
    eval(inRangeText);
}

loadImageToCanvas("lena.jpg", "inRangeCanvas1");
var inRangeInputElement = document.getElementById("inRangeInput");
inRangeInputElement.addEventListener("change", inRangeHandleFiles, false);
function inRangeHandleFiles(e) {
    var inRangeUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(inRangeUrl, "inRangeCanvas1");
}

document.getElementById("opencvjs").onload = function() {
    document.getElementById("inRangeTryIt").disabled = false;
    document.getElementById("cvtColorTryIt").disabled = false;
};

</script>
</body>
\endhtmlonly
