Getting Started with Images {#tutorial_js_image_display}
===========================

Goals
-----

-   Learn how to read an image and how to display it in a web.

Read an image
-------------

OpenCV-JavaScript saves image as cv.Mat type. We use HTML canvas element to transfer cv.Mat to the web  
or in reverse. The ImageData interface can represent or set the underlying pixel data of an area of a 
canvas element. 

@sa Please refer to canvas docs for more details.

First, creat an ImageData obj from canvas.
@code{.js}
var canvas = document.getElementById(canvas_id);
var ctx = canvas.getContext("2d");
var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
@endcode

Then use cv.matFromArray to construct a cv.Mat.
@code{.js}
var src = cv.matFromArray(imgData, cv.CV_8UC4);
@endcode

@note Cause canvas only support 8-bit RGBA image, the cv.Mat type is cv.CV_8UC4.


Display an image
----------------

First, convert the type of src to cv.CV_8UC4.
@code{.js}
var dst = new cv.Mat();
// scale and shift are used to map the data to [0, 255].
src.convertTo(dst, cv.CV_8U, scale, shift); 
// *** is GRAY, RGB, or RGBA, according to src.channels() is 1, 3 or 4.
cv.cvtColor(dst, dst, cv.ColorConversionCodes.COLOR_***2RGBA.value, 0); 
@endcode

Then new an ImageData obj from dst.
@code{.js}
var imgData = new ImageData(new Uint8ClampedArray(dst.data()), dst.cols, dst.rows);
@endcode

Finally, display it.
@code{.js}
var canvas = document.getElementById(canvas_id);
var ctx = canvas.getContext("2d");
ctx.clearRect(0, 0, canvas.width, canvas.height);
canvas.width = imgData.width;
canvas.height = imgData.height;
ctx.putImageData(imgData, 0, 0);
@endcode

In addition, OpenCV-JavaScript implements image read and show using the above method. You can use cv.imread and 
cv.imshow to read image from html canvas and display it.
@code{.js}
var img = cv.imread("canvasInput");
cv.imshow("canvasOutput", img);
img.delete();
@endcode

Try it
------

Here is the demo for above code. Canvas elements named canvasInput and canvasOutput have been prepared. Choose an image and 
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
<div id="CodeArea">
<h2>Input your code</h2>
<button id="tryIt" disabled="true" onclick="executeCode()">Try it</button><br>
<textarea rows="11" cols="80" id="TestCode" spellcheck="false">
var src = cv.imread("canvasInput");
var dst = new cv.Mat();
// To distinguish the input and output, we graying the image.
// You can try more different conversion
cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
cv.imshow("canvasOutput", dst);
src.delete();
dst.delete();
</textarea>
</div>
<div id="showcase">
    <div>
        <canvas id="canvasInput"></canvas>
        <canvas id="canvasOutput"></canvas>
    </div>
    <input type="file" id="input" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function executeCode() {
    var text = document.getElementById("TestCode").value;
    eval(text);
}

loadImageToCanvas("lena.jpg", "canvasInput");

var inputElement = document.getElementById("input");
inputElement.addEventListener("change", handleFiles, false);
function handleFiles(e) {
    var url = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(url, "canvasInput");
}

document.getElementById("opencvjs").onload = function() {
    document.getElementById("tryIt").disabled = false;
};
</script>
</body>
\endhtmlonly