Histogram - 4 : Histogram Backprojection {#tutorial_js_histogram_backprojection}
========================================

Goal
----

In this chapter, we will learn about histogram backprojection.

Theory
------

It was proposed by **Michael J. Swain , Dana H. Ballard** in their paper **Indexing via color
histograms**.

**What is it actually in simple words?** It is used for image segmentation or finding objects of
interest in an image. In simple words, it creates an image of the same size (but single channel) as
that of our input image, where each pixel corresponds to the probability of that pixel belonging to
our object. In more simpler worlds, the output image will have our object of interest in more white
compared to remaining part. Well, that is an intuitive explanation. (I can't make it more simpler).
Histogram Backprojection is used with camshift algorithm etc.

**How do we do it ?** We create a histogram of an image containing our object of interest (in our
case, the ground, leaving player and other things). The object should fill the image as far as
possible for better results. And a color histogram is preferred over grayscale histogram, because
color of the object is a better way to define the object than its grayscale intensity. We then
"back-project" this histogram over our test image where we need to find the object, ie in other
words, we calculate the probability of every pixel belonging to the ground and show it. The
resulting output on proper thresholding gives us the ground alone.

Backprojection in OpenCV
------------------------

We use the functions: **cv.calcBackProject (images, nimages, channels, hist, backProject, ranges, scale = 1, uniform = true)** 

@param images       source arrays. They all should have the same depth, cv.CV_8U, cv.CV_16U or cv.CV_32F , and the same size. Each of them can have an arbitrary number of channels.
@param nimages      number of source images.
@param channels     the list of channels used to compute the back projection. The number of channels must match the histogram dimensionality. 
@param hist         input histogram that can be dense or sparse.
@param backProject  destination back projection array that is a single-channel array of the same size and depth as images[0].
@param ranges       array of arrays of the histogram bin boundaries in each dimension(see cv.calcHist).
@param scale        optional scale factor for the output back projection.
@param uniform      flag indicating whether the histogram is uniform or not.

**cv.normalize (src, dst, alpha = 1, beta = 0, norm_type = cv.NORM_L2, dtype = -1, mask = Mat())** 

@param src        input array.
@param dst        output array of the same size as src .
@param alpha      norm value to normalize to or the lower range boundary in case of the range normalization.
@param beta       upper range boundary in case of the range normalization; it is not used for the norm normalization.
@param norm_type  normalization type (see cv.NormTypes).
@param dtype      when negative, the output array has the same type as src; otherwise, it has the same number of channels as src and the depth = CV_MAT_DEPTH(dtype).
@param mask       optional operation mask.

Try it
------

Here is a demo. Canvas elements named calcBackProjectCanvasSrcInput, calcBackProjectCanvasDstInput and calcBackProjectCanvasOutput have been prepared. Choose an image and
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
<div id="calcBackProjectCodeArea">
<h2>Input your code</h2>
<button id="calcBackProjectTryIt" disabled="true" onclick="calcBackProjectExecuteCode()">Try it</button><br>
<textarea rows="15" cols="90" id="calcBackProjectTestCode" spellcheck="false">
var src = cv.imread("calcBackProjectCanvasSrcInput");
var dst = cv.imread("calcBackProjectCanvasDstInput");
cv.cvtColor(src, src, cv.COLOR_RGB2HSV, 0);
cv.cvtColor(dst, dst, cv.COLOR_RGB2HSV, 0);
var srcVec = new cv.MatVector(), dstVec = new cv.MatVector();
srcVec.push_back(src); dstVec.push_back(dst);
var backproj = new cv.Mat(), none = new cv.Mat(), mask = new cv.Mat(), hist = new cv.Mat();
var channels = [0], histSize = [50], ranges = [0, 180];
var accumulate = false;
cv.calcHist(srcVec, channels, mask, hist, histSize, ranges, accumulate);
cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX, -1 ,none);
cv.calcBackProject(dstVec, channels, hist, backproj, ranges, 1);
cv.imshow("calcBackProjectCanvasOutput", backproj);
src.delete(); dst.delete(); srcVec.delete(); dstVec.delete(); backproj.delete(); mask.delete(); hist.delete(); none.delete();
</textarea>
<p class="err" id="calcBackProjectErr"></p>
</div>
<div id="calcBackProjectShowcase">
    <div>
   	    <p>SrcInput</p>
        <canvas id="calcBackProjectCanvasSrcInput"></canvas>
        <input type="file" id="calcBackProjectSrcInput" name="file" />
    </div>
    <div>
    	<p>DstInput</p>
        <canvas id="calcBackProjectCanvasDstInput"></canvas>
        <input type="file" id="calcBackProjectDstInput" name="file" />
    </div>
    <div>
    	<p>BackProject</p>
        <canvas id="calcBackProjectCanvasOutput"></canvas>
    </div>
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function calcBackProjectExecuteCode() {
    var calcBackProjectText = document.getElementById("calcBackProjectTestCode").value;
    try {
        eval(calcBackProjectText);
        document.getElementById("calcBackProjectErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("calcBackProjectErr").innerHTML = err;
    }
}

loadImageToCanvas("handSrc.jpg", "calcBackProjectCanvasSrcInput");
loadImageToCanvas("handDst.jpg", "calcBackProjectCanvasDstInput");

var calcBackProjectSrcInputElement = document.getElementById("calcBackProjectSrcInput");
calcBackProjectSrcInputElement.addEventListener("change", calcBackProjectSrcHandleFiles, false);
function calcBackProjectSrcHandleFiles(e) {
    var calcBackProjectSrcUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(calcBackProjectSrcUrl, "calcBackProjectCanvasSrcInput");
}

var calcBackProjectDstInputElement = document.getElementById("calcBackProjectDstInput");
calcBackProjectDstInputElement.addEventListener("change", calcBackProjectDstHandleFiles, false);
function calcBackProjectDstHandleFiles(e) {
    var calcBackProjectDstUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(calcBackProjectDstUrl, "calcBackProjectCanvasDstInput");
}

function onReady() {
    document.getElementById("calcBackProjectTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly