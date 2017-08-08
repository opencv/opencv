Getting Started with Images {#tutorial_js_image_display}
===========================

Goals
-----

-   Learn how to read an image and how to display it in a web.

Read an image
-------------

OpenCV.js saves image as cv.Mat type. We use HTML canvas element to transfer cv.Mat to the web  
or in reverse. The ImageData interface can represent or set the underlying pixel data of an area of a 
canvas element. 

@sa Please refer to canvas docs for more details.

First, creat an ImageData obj from canvas.
@code{.js}
let canvas = document.getElementById(canvas_id);
let ctx = canvas.getContext("2d");
let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
@endcode

Then use cv.matFromArray to construct a cv.Mat.
@code{.js}
let src = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
src.data.set(imgData.data);
@endcode

@note Cause canvas only support 8-bit RGBA image with continuous storage, the cv.Mat type is cv.CV_8UC4. 
It is different from native OpenCV, cause images returned and shown by the native **imread** and 
**imshow** have the channels stored in BGR order.

Display an image
----------------

First, convert the type of src to cv.CV_8UC4.
@code{.js}
let dst = new cv.Mat();
// scale and shift are used to map the data to [0, 255].
src.convertTo(dst, cv.CV_8U, scale, shift); 
// *** is GRAY, RGB, or RGBA, according to src.channels() is 1, 3 or 4.
cv.cvtColor(dst, dst, cv.COLOR_***2RGBA); 
@endcode

Then new an ImageData obj from dst.
@code{.js}
let imgData = new ImageData(new Uint8ClampedArray(dst.data, dst.cols, dst.rows);
@endcode

Finally, display it.
@code{.js}
let canvas = document.getElementById(canvas_id);
let ctx = canvas.getContext("2d");
ctx.clearRect(0, 0, canvas.width, canvas.height);
canvas.width = imgData.width;
canvas.height = imgData.height;
ctx.putImageData(imgData, 0, 0);
@endcode

In addition, OpenCV.js implements image read and show using the above method. You can use 
cv.imread and cv.imshow to read image from html canvas and display it.
@code{.js}
// imageSource should be canvas element or id, or img element or id.
let img = cv.imread(imageSource);
// canvasOutput should be canvas element or id.
cv.imshow(canvasOutput, img);
img.delete();
@endcode

Try it
------

Here is the demo for above code. Canvas "canvasInput" and canvas "canvasOutput" have been prepared. Choose an image and click `Try it` to see the result. And you can change the code in the textbox to investigate more.

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
<div id="CodeArea">
<h2>Input your code</h2>
<button id="tryIt" disabled="true" onclick="executeCode()">Try it</button><br>
<textarea rows="11" cols="80" id="TestCode" spellcheck="false">
let src = cv.imread("canvasInput");
let dst = new cv.Mat();
// To distinguish the input and output, we graying the image.
// You can try more different conversion
cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
cv.imshow("canvasOutput", dst);
src.delete();
dst.delete();
</textarea>
<p class="err" id="imErr"></p>
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
    let text = document.getElementById("TestCode").value;
    try {
        eval(text);
        document.getElementById("imErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("imErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "canvasInput");

let inputElement = document.getElementById("input");
inputElement.addEventListener("change", handleFiles, false);
function handleFiles(e) {
    let url = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(url, "canvasInput");
}

function onReady() {
    document.getElementById("tryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly