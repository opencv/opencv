Histograms - 2: Histogram Equalization {#tutorial_js_histogram_equalization}
======================================

Goal
----

-   We will learn the concepts of histogram equalization and use it to improve the contrast of our
    images.

Theory
------

Consider an image whose pixel values are confined to some specific range of values only. For eg,
brighter image will have all pixels confined to high values. But a good image will have pixels from
all regions of the image. So you need to stretch this histogram to either ends (as given in below
image, from wikipedia) and that is what Histogram Equalization does (in simple words). This normally
improves the contrast of the image.

![image](images/histogram_equalization.png)

I would recommend you to read the wikipedia page on [Histogram
Equalization](http://en.wikipedia.org/wiki/Histogram_equalization) for more details about it. It has
a very good explanation with worked out examples, so that you would understand almost everything
after reading that.

Histograms Equalization in OpenCV
---------------------------------

We use the function: **cv.equalizeHist (src, dst)**

@param src      source 8-bit single channel image.
@param dst      destination image of the same size and type as src.

Try it
------

Try this demo using the code above. Canvas elements named equalizeHistCanvasInput, imageGrayCanvasOutput and equalizeHistCanvasOutput have been prepared. Choose an image and
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
<div id="equalizeHistCodeArea">
<h2>Input your code</h2>
<button id="equalizeHistTryIt" disabled="true" onclick="equalizeHistExecuteCode()">Try it</button><br>
<textarea rows="7" cols="80" id="equalizeHistTestCode" spellcheck="false">
let src = cv.imread("equalizeHistCanvasInput");
let dst = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.equalizeHist(src, dst);
cv.imshow("imageGrayCanvasOutput", src);
cv.imshow("equalizeHistCanvasOutput", dst);
src.delete(); dst.delete();
</textarea>
<p class="err" id="equalizeHistErr"></p>
</div>
<div id="equalizeHistShowcase">
    <div>
        <p>Original</p>
        <canvas id="equalizeHistCanvasInput"></canvas>
        <input type="file" id="equalizeHistInput" name="file" />
    </div>
    <div>
        <p>Gray Image</p>
        <canvas id="imageGrayCanvasOutput"></canvas>
    </div>
    <div>
        <p>EqualizeHist Image</p>
        <canvas id="equalizeHistCanvasOutput"></canvas>
    </div>
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function equalizeHistExecuteCode() {
    let equalizeHistText = document.getElementById("equalizeHistTestCode").value;
    try {
        eval(equalizeHistText);
        document.getElementById("equalizeHistErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("equalizeHistErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "equalizeHistCanvasInput");
let equalizeHistInputElement = document.getElementById("equalizeHistInput");
equalizeHistInputElement.addEventListener("change", equalizeHistHandleFiles, false);
function equalizeHistHandleFiles(e) {
    let equalizeHistUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(equalizeHistUrl, "equalizeHistCanvasInput");
}
</script>
</body>
\endhtmlonly

CLAHE (Contrast Limited Adaptive Histogram Equalization)
--------------------------------------------------------

In **adaptive histogram equalization**, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region
(unless there is noise). If noise is there, it will be amplified. To avoid this, **contrast limiting** is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.

We use the class: **cv.CLAHE (clipLimit = 40, tileGridSize = new cv.Size(8, 8))**

@param clipLimit      threshold for contrast limiting.
@param tileGridSize   size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.

@note Don't forget to delete CLAHE!

Try it
------

Try this demo using the code above. Canvas elements named createCLAHECanvasInput, equalCanvasOutput and createCLAHECanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="createCLAHECodeArea">
<h2>Input your code</h2>
<button id="createCLAHETryIt" disabled="true" onclick="createCLAHEExecuteCode()">Try it</button><br>
<textarea rows="9" cols="80" id="createCLAHETestCode" spellcheck="false">
let src = cv.imread("createCLAHECanvasInput");
let equalDst = new cv.Mat(), claheDst = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.equalizeHist(src, equalDst);
let tileGridSize = new cv.Size(8, 8);
// You can try more different parameters
let clahe = new cv.CLAHE(40, tileGridSize);
clahe.apply(src, claheDst);
cv.imshow("equalCanvasOutput", equalDst);
cv.imshow("createCLAHECanvasOutput", claheDst);
src.delete(); equalDst.delete(); claheDst.delete(); clahe.delete();
</textarea>
<p class="err" id="createCLAHEErr"></p>
</div>
<div id="createCLAHEShowcase">
    <div>
        <p>Original</p>
        <canvas id="createCLAHECanvasInput"></canvas>
        <input type="file" id="createCLAHEInput" name="file" />
    </div>
    <div>
        <p>EqualizeHist Image</p>
        <canvas id="equalCanvasOutput"></canvas>
    </div>
    <div>
        <p>CreateCLAHE Image</p>
        <canvas id="createCLAHECanvasOutput"></canvas>
    </div>
</div>
<script>
function createCLAHEExecuteCode() {
    let createCLAHEText = document.getElementById("createCLAHETestCode").value;
    try {
        eval(createCLAHEText);
        document.getElementById("createCLAHEErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("createCLAHEErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "createCLAHECanvasInput");
let createCLAHEInputElement = document.getElementById("createCLAHEInput");
createCLAHEInputElement.addEventListener("change", createCLAHEHandleFiles, false);
function createCLAHEHandleFiles(e) {
    let createCLAHEUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(createCLAHEUrl, "createCLAHECanvasInput");
}

function onReady() {
    document.getElementById("createCLAHETryIt").disabled = false;
    document.getElementById("equalizeHistTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
