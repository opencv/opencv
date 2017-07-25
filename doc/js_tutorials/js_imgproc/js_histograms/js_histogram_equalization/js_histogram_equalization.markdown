Histograms - 2: Histogram Equalization {#tutorial_js_histogram_equalization}
======================================

Goal
----

In this section,

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

Here is a demo. Canvas elements named equalizeHistCanvasInput, imageGrayCanvasOutput and equalizeHistCanvasOutput have been prepared. Choose an image and
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
<div id="equalizeHistCodeArea">
<h2>Input your code</h2>
<button id="equalizeHistTryIt" disabled="true" onclick="equalizeHistExecuteCode()">Try it</button><br>
<textarea rows="7" cols="80" id="equalizeHistTestCode" spellcheck="false">
var src = cv.imread("equalizeHistCanvasInput");
var dst = new cv.Mat();
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
    var equalizeHistText = document.getElementById("equalizeHistTestCode").value;
    try {
        eval(equalizeHistText);
        document.getElementById("equalizeHistErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("equalizeHistErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "equalizeHistCanvasInput");
var equalizeHistInputElement = document.getElementById("equalizeHistInput");
equalizeHistInputElement.addEventListener("change", equalizeHistHandleFiles, false);
function equalizeHistHandleFiles(e) {
    var equalizeHistUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(equalizeHistUrl, "equalizeHistCanvasInput");
}

function onReady() {
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

CLAHE (Contrast Limited Adaptive Histogram Equalization)
--------------------------------------------------------

@note cv.createCLAHE() is not in the white list.

The first histogram equalization we just saw, considers the global contrast of the image. In many
cases, it is not a good idea. For example, below image shows an input image and its result after
global histogram equalization.

![image](images/clahe_1.jpg)

It is true that the background contrast has improved after histogram equalization. But compare the
face of statue in both images. We lost most of the information there due to over-brightness. It is
because its histogram is not confined to a particular region as we saw in previous cases (Try to
plot histogram of input image, you will get more intuition).

So to solve this problem, **adaptive histogram equalization** is used. In this, image is divided
into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks
are histogram equalized as usual. So in a small area, histogram would confine to a small region
(unless there is noise). If noise is there, it will be amplified. To avoid this, **contrast
limiting** is applied. If any histogram bin is above the specified contrast limit (by default 40 in
OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram
equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is
applied.

Below code snippet shows how to apply CLAHE in OpenCV:
@code{.py}
import numpy as np
import cv2

img = cv2.imread('tsukuba_l.png',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imwrite('clahe_2.jpg',cl1)
@endcode
See the result below and compare it with results above, especially the statue region:

![image](images/clahe_2.jpg)



