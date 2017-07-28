Histograms - 1 : Find, Plot, Analyze !!! {#tutorial_js_histogram_begins}
========================================

Goal
----

Learn to
    -   Find histograms
    -   Plot histograms
    -   You will see the function: **cv.calcHist()**.

Theory
------

So what is histogram ? You can consider histogram as a graph or plot, which gives you an overall
idea about the intensity distribution of an image. It is a plot with pixel values (ranging from 0 to
255, not always) in X-axis and corresponding number of pixels in the image on Y-axis.

It is just another way of understanding the image. By looking at the histogram of an image, you get
intuition about contrast, brightness, intensity distribution etc of that image. Almost all image
processing tools today, provides features on histogram. Below is an image from [Cambridge in Color
website](http://www.cambridgeincolour.com/tutorials/histograms1.htm), and I recommend you to visit
the site for more details.

![image](histogram_sample.jpg)

You can see the image and its histogram. (Remember, this histogram is drawn for grayscale image, not
color image). Left region of histogram shows the amount of darker pixels in image and right region
shows the amount of brighter pixels. From the histogram, you can see dark region is more than
brighter region, and amount of midtones (pixel values in mid-range, say around 127) are very less.

Find Histogram
--------------

We use the function: **cv.calcHist (image, nimages, channels, mask, hist, dims, histSize, ranges, uniform = true, accumulate = false)** 

@param image        source arrays. They all should have the same depth, cv.CV_8U, cv.CV_16U or cv.CV_32F , and the same size. Each of them can have an arbitrary number of channels. 
@param nimages      number of source images.
@param channels     list of the dims channels used to compute the histogram. The first array channels are numerated from 0 to images[0].channels()-1 , the second array channels are counted from images[0].channels() to images[0].channels() + images[1].channels()-1, and so on.
@param mask         optional mask. If the matrix is not empty, it must be an 8-bit array of the same size as images[i] . The non-zero mask elements mark the array elements counted in the histogram.
@param hist        	output histogram(cv.CV_32F type), which is a dense or sparse dims -dimensional array.
@param dims         histogram dimensionality that must be positive and not greater than 32(in the current OpenCV version).
@param histSize     array of histogram sizes in each dimension.
@param ranges       array of the dims arrays of the histogram bin boundaries in each dimension.
@param uniform      flag indicating whether the histogram is uniform or not.
@param accumulate   accumulation flag. If it is set, the histogram is not cleared in the beginning when it is allocated. This feature enables you to compute a single histogram from several sets of arrays, or to update the histogram in time.

Try it
------

Here is a demo. Canvas elements named calcHistCanvasInput and calcHistCanvasOutput have been prepared. Choose an image and
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
<div id="calcHistCodeArea">
<h2>Input your code</h2>
<button id="calcHistTryIt" disabled="true" onclick="calcHistExecuteCode()">Try it</button><br>
<textarea rows="17" cols="90" id="calcHistTestCode" spellcheck="false">
var src = cv.imread("calcHistCanvasInput");
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
var srcVec = new cv.MatVector();
srcVec.push_back(src);
var accumulate = false;
var channels = [0], histSize = [256], ranges = [0,255];
var hist = new cv.Mat(), mask = new cv.Mat(), color = new cv.Scalar(255, 255, 255);
var scale = 2;
cv.calcHist(srcVec, channels, mask, hist, histSize, ranges, accumulate);
var result = new cv.MinMaxLocResult();
cv.minMaxLoc(hist, result, mask);
var max = result.maxVal;
var dst = new cv.Mat.zeros(src.rows, histSize[0] * scale, cv.CV_8UC3);
// draw histogram
for(var i = 0; i < histSize[0]; i++)
{
    var binVal = hist.data32f()[i] * src.rows / max;
    cv.rectangle(dst, [i * scale, src.rows - 1], [(i + 1) * scale - 1, src.rows - binVal], color, cv.FILLED);
}
cv.imshow("calcHistCanvasOutput", dst);
src.delete(); dst.delete(); srcVec.delete(); mask.delete(); hist.delete(); color.delete(); result.delete();
</textarea>
<p class="err" id="calcHistErr"></p>
</div>
<div id="calcHistShowcase">
    <div>
        <canvas id="calcHistCanvasInput"></canvas>
        <canvas id="calcHistCanvasOutput"></canvas>
    </div>
    <input type="file" id="calcHistInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function calcHistExecuteCode() {
    var calcHistText = document.getElementById("calcHistTestCode").value;
    try {
        eval(calcHistText);
        document.getElementById("calcHistErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("calcHistErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "calcHistCanvasInput");
var calcHistInputElement = document.getElementById("calcHistInput");
calcHistInputElement.addEventListener("change", calcHistHandleFiles, false);
function calcHistHandleFiles(e) {
    var calcHistUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(calcHistUrl, "calcHistCanvasInput");
}

function onReady() {
    document.getElementById("calcHistTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly