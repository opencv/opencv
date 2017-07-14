Image Pyramids {#tutorial_js_pyramids}
==============

Goal
----

In this chapter,
    -   We will learn about Image Pyramids
    -   We will see these functions: **cv.pyrUp()**, **cv.pyrDown()**

Theory
------

Normally, we used to work with an image of constant size. But in some occassions, we need to work
with images of different resolution of the same image. For example, while searching for something in
an image, like face, we are not sure at what size the object will be present in the image. In that
case, we will need to create a set of images with different resolution and search for object in all
the images. These set of images with different resolution are called Image Pyramids (because when
they are kept in a stack with biggest image at bottom and smallest image at top look like a
pyramid).

There are two kinds of Image Pyramids. 1) Gaussian Pyramid and 2) Laplacian Pyramids

Higher level (Low resolution) in a Gaussian Pyramid is formed by removing consecutive rows and
columns in Lower level (higher resolution) image. Then each pixel in higher level is formed by the
contribution from 5 pixels in underlying level with gaussian weights. By doing so, a \f$M \times N\f$
image becomes \f$M/2 \times N/2\f$ image. So area reduces to one-fourth of original area. It is called
an Octave. The same pattern continues as we go upper in pyramid (ie, resolution decreases).
Similarly while expanding, area becomes 4 times in each level. We can find Gaussian pyramids using
**cv.pyrDown()** and **cv.pyrUp()** functions.

Laplacian Pyramids are formed from the Gaussian Pyramids. There is no exclusive function for that.
Laplacian pyramid images are like edge images only. Most of its elements are zeros. They are used in
image compression. A level in Laplacian Pyramid is formed by the difference between that level in
Gaussian Pyramid and expanded version of its upper level in Gaussian Pyramid. 

Downsample
------

We use the function: **cv.pyrDown (src, dst, dstsize = [0, 0], borderType  = cv.BORDER_DEFAULT)** 
@param src         input image.
@param dst         output image; it has the specified size and the same type as src.
@param dstsize     size of the output image.
@param borderType  pixel extrapolation method(see cv.BorderTypes, cv.BORDER_CONSTANT isn't supported).

Try it
------

Here is a demo. Canvas elements named pyrDownCanvasInput and pyrDownCanvasOutput have been prepared. Choose an image and
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
<div id="pyrDownCodeArea">
<h2>Input your code</h2>
<button id="pyrDownTryIt" disabled="true" onclick="pyrDownExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="pyrDownTestCode" spellcheck="false">
var src = cv.imread("pyrDownCanvasInput");
var dst = new cv.Mat();
// You can try more different conversion
cv.pyrDown(src, dst, [0, 0], cv.BORDER_DEFAULT);
cv.imshow("pyrDownCanvasOutput", dst);
src.delete(); 
dst.delete();
</textarea>
</div>
<div id="pyrDownShowcase">
    <div>
        <canvas id="pyrDownCanvasInput"></canvas>
        <canvas id="pyrDownCanvasOutput"></canvas>
    </div>
    <input type="file" id="pyrDownInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function pyrDownExecuteCode() {
    var pyrDownText = document.getElementById("pyrDownTestCode").value;
    eval(pyrDownText);
}

loadImageToCanvas("lena.jpg", "pyrDownCanvasInput");
var pyrDownInputElement = document.getElementById("pyrDownInput");
pyrDownInputElement.addEventListener("change", pyrDownHandleFiles, false);
function pyrDownHandleFiles(e) {
    var pyrDownUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(pyrDownUrl, "pyrDownCanvasInput");
}
</script>
</body>
\endhtmlonly

Upsample
------

We use the function: **cv.pyrUp (src, dst, dstsize = [0, 0], borderType  = cv.BORDER_DEFAULT)** 
@param src         input image.
@param dst         output image; it has the specified size and the same type as src.
@param dstsize     size of the output image.
@param borderType  pixel extrapolation method(see cv.BorderTypes, only cv.BORDER_DEFAULT is supported).

Try it
------

Here is a demo. Canvas elements named pyrUpCanvasInput and pyrUpCanvasOutput have been prepared. Choose an image and
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
<div id="pyrUpCodeArea">
<h2>Input your code</h2>
<button id="pyrUpTryIt" disabled="true" onclick="pyrUpExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="pyrUpTestCode" spellcheck="false">
var src = cv.imread("pyrUpCanvasInput");
var dst = new cv.Mat();
// You can try more different conversion
cv.pyrUp(src, dst, [0, 0], cv.BORDER_DEFAULT);
cv.imshow("pyrUpCanvasOutput", dst);
src.delete(); 
dst.delete();
</textarea>
</div>
<div id="pyrUpShowcase">
    <div>
        <canvas id="pyrUpCanvasInput"></canvas>
        <canvas id="pyrUpCanvasOutput"></canvas>
    </div>
    <input type="file" id="pyrUpInput" name="file" />
</div>
<script>
function pyrUpExecuteCode() {
    var pyrUpText = document.getElementById("pyrUpTestCode").value;
    eval(pyrUpText);
}

loadImageToCanvas("lena.jpg", "pyrUpCanvasInput");
var pyrUpInputElement = document.getElementById("pyrUpInput");
pyrUpInputElement.addEventListener("change", pyrUpHandleFiles, false);
function pyrUpHandleFiles(e) {
    var pyrUpUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(pyrUpUrl, "pyrUpCanvasInput");
}
function onReady() {
    document.getElementById("pyrDownTryIt").disabled = false;
    document.getElementById("pyrUpTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
