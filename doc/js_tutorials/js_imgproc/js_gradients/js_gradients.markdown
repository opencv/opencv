Image Gradients {#tutorial_js_gradients}
===============

Goal
----

In this chapter, we will learn to:

-   Find Image gradients, edges etc
-   We will see following functions : **cv.Sobel()**, **cv.Scharr()**, **cv.Laplacian()** etc

Theory
------

OpenCV provides three types of gradient filters or High-pass filters, Sobel, Scharr and Laplacian.
We will see each one of them.

### 1. Sobel and Scharr Derivatives

Sobel operators is a joint Gausssian smoothing plus differentiation operation, so it is more
resistant to noise. You can specify the direction of derivatives to be taken, vertical or horizontal
(by the arguments, yorder and xorder respectively). You can also specify the size of kernel by the
argument ksize. If ksize = -1, a 3x3 Scharr filter is used which gives better results than 3x3 Sobel
filter. Please see the docs for kernels used.

We use the functions: **cv.Sobel (src, dst, ddepth, dx, dy, ksize = 3, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)** 
@param src         input image.
@param dst         output image of the same size and the same number of channels as src.
@param ddepth      output image depth(see cv.combinations); in the case of 8-bit input images it will result in truncated derivatives.
@param dx          order of the derivative x.
@param dy          order of the derivative y.
@param ksize       size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
@param scale       optional scale factor for the computed derivative values.
@param delta       optional delta value that is added to the results prior to storing them in dst.
@param borderType  pixel extrapolation method(see cv.BorderTypes).

**cv.Scharr (src, dst, ddepth, dx, dy, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)** 
@param src         input image.
@param dst         output image of the same size and the same number of channels as src.
@param ddepth      output image depth(see cv.combinations).
@param dx          order of the derivative x.
@param dy          order of the derivative y.
@param scale       optional scale factor for the computed derivative values.
@param delta       optional delta value that is added to the results prior to storing them in dst.
@param borderType  pixel extrapolation method(see cv.BorderTypes).

Try it
------

Here is a demo. Canvas elements named SobelCanvasInput, SobelCanvasOutputX and SobelCanvasOutputY have been prepared. Choose an image and
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
<div id="SobelCodeArea">
<h2>Input your code</h2>
<button id="SobelTryIt" disabled="true" onclick="SobelExecuteCode()">Try it</button><br>
<textarea rows="13" cols="80" id="SobelTestCode" spellcheck="false">
let src = cv.imread("SobelCanvasInput");
let dstx = new cv.Mat();
let dsty = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0);
// You can try more different parameters
cv.Sobel(src, dstx, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
cv.Sobel(src, dsty, cv.CV_8U, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
// cv.Scharr(src, dstx, cv.CV_8U, 1, 0, 1, 0, cv.BORDER_DEFAULT);
// cv.Scharr(src, dsty, cv.CV_8U, 0, 1, 1, 0, cv.BORDER_DEFAULT);
cv.imshow("SobelCanvasOutputX", dstx);
cv.imshow("SobelCanvasOutputY", dsty); 
src.delete(); dstx.delete(); dsty.delete();
</textarea>
<p class="err" id="SobelErr"></p>
</div>
<div id="SobelShowcase">
    <div>
        <p>Original</p>
        <canvas id="SobelCanvasInput"></canvas>
        <input type="file" id="SobelInput" name="file" />
    </div>       
    <div>
        <p>Sobel X</p>
        <canvas id="SobelCanvasOutputX"></canvas>
    </div>
    <div>
        <p>Sobel Y</p>
        <canvas id="SobelCanvasOutputY"></canvas>
    </div>
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function SobelExecuteCode() {
    let SobelText = document.getElementById("SobelTestCode").value;
    try {
        eval(SobelText);
        document.getElementById("SobelErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("SobelErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "SobelCanvasInput");
let SobelInputElement = document.getElementById("SobelInput");
SobelInputElement.addEventListener("change", SobelHandleFiles, false);
function SobelHandleFiles(e) {
    let SobelUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(SobelUrl, "SobelCanvasInput");
}
</script>
</body>
\endhtmlonly

### 2. Laplacian Derivatives

It calculates the Laplacian of the image given by the relation,
\f$\Delta src = \frac{\partial ^2{src}}{\partial x^2} + \frac{\partial ^2{src}}{\partial y^2}\f$ where
each derivative is found using Sobel derivatives. If ksize = 1, then following kernel is used for
filtering:

\f[kernel = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0  \end{bmatrix}\f]

We use the function: **cv.Laplacian (src, dst, ddepth, ksize = 1, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)** 
@param src         input image.
@param dst         output image of the same size and the same number of channels as src.
@param ddepth      output image depth.
@param ksize       aperture size used to compute the second-derivative filters.
@param scale       optional scale factor for the computed Laplacian values.
@param delta       optional delta value that is added to the results prior to storing them in dst.
@param borderType  pixel extrapolation method(see cv.BorderTypes).

Try it
------

Here is a demo. Canvas elements named LaplacianCanvasInput and LaplacianCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="LaplacianCodeArea">
<h2>Input your code</h2>
<button id="LaplacianTryIt" disabled="true" onclick="LaplacianExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="LaplacianTestCode" spellcheck="false">
let src = cv.imread("LaplacianCanvasInput");
let dst = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0);
// You can try more different parameters
cv.Laplacian(src, dst, cv.CV_8U, 1, 1, 0, cv.BORDER_DEFAULT);
cv.imshow("LaplacianCanvasOutput", dst);
src.delete(); dst.delete();
</textarea>
<p class="err" id="LaplacianErr"></p>
</div>
<div id="LaplacianShowcase">
    <div>
        <canvas id="LaplacianCanvasInput"></canvas>
        <canvas id="LaplacianCanvasOutput"></canvas>
    </div>
    <input type="file" id="LaplacianInput" name="file" />
</div>
<script>
function LaplacianExecuteCode() {
    let LaplacianText = document.getElementById("LaplacianTestCode").value;
    try {
        eval(LaplacianText);
        document.getElementById("LaplacianErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("LaplacianErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "LaplacianCanvasInput");
let LaplacianInputElement = document.getElementById("LaplacianInput");
LaplacianInputElement.addEventListener("change", LaplacianHandleFiles, false);
function LaplacianHandleFiles(e) {
    let LaplacianUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(LaplacianUrl, "LaplacianCanvasInput");
}
</script>
</body>
\endhtmlonly

One Important Matter!
---------------------

In our last example, output datatype is cv.CV_8U. But there is a slight problem with
that. Black-to-White transition is taken as Positive slope (it has a positive value) while
White-to-Black transition is taken as a Negative slope (It has negative value). So when you convert
data to cv.CV_8U, all negative slopes are made zero. In simple words, you miss that edge.

If you want to detect both edges, better option is to keep the output datatype to some higher forms,
like cv.CV_16S, cv.CV_64F etc, take its absolute value and then convert back to cv.CV_8U.
Below code demonstrates this procedure for a horizontal Sobel filter and difference in results.

Try it
------

Here is a demo. Canvas elements named absSobelCanvasInput, absSobelCanvasOutput8U and absSobelCanvasOutput64F have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

We use the function: **cv.convertScaleAbs (src, dst, alpha = 1, beta = 0)** 
@param src     input array.
@param dst     output array.
@param alpha   optional scale factor.
@param beta    optional delta added to the scaled values.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="absSobelCodeArea">
<h2>Input your code</h2>
<button id="absSobelTryIt" disabled="true" onclick="absSobelExecuteCode()">Try it</button><br>
<textarea rows="11" cols="80" id="absSobelTestCode" spellcheck="false">
let src = cv.imread("absSobelCanvasInput");
let dstx = new cv.Mat();
let absDstx = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0);
// You can try more different parameters
cv.Sobel(src, dstx, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
cv.Sobel(src, absDstx, cv.CV_64F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
cv.convertScaleAbs(absDstx, absDstx, 1, 0);
cv.imshow("absSobelCanvasOutput8U", dstx);
cv.imshow("absSobelCanvasOutput64F", absDstx); 
src.delete(); dstx.delete(); absDstx.delete();
</textarea>
<p class="err" id="absSobelErr"></p>
</div>
<div id="absSobelShowcase">
    <div>
        <p>Original</p>
        <canvas id="absSobelCanvasInput"></canvas>
        <input type="file" id="absSobelInput" name="file" />
    </div>       
    <div>
        <p>Sobel X(cv.CV_8U)</p>
        <canvas id="absSobelCanvasOutput8U"></canvas>
    </div>
    <div>
        <p>Sobel X(cv.CV_64F)</p>
        <canvas id="absSobelCanvasOutput64F"></canvas>
    </div>
</div>
<script>
function absSobelExecuteCode() {
    let absSobelText = document.getElementById("absSobelTestCode").value;
    try {
        eval(absSobelText);
        document.getElementById("absSobelErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("absSobelErr").innerHTML = err;
    }
}

loadImageToCanvas("LinuxLogo.jpg", "absSobelCanvasInput");
let absSobelInputElement = document.getElementById("absSobelInput");
absSobelInputElement.addEventListener("change", absSobelHandleFiles, false);
function absSobelHandleFiles(e) {
    let absSobelUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(absSobelUrl, "absSobelCanvasInput");
}

function onReady() {
    document.getElementById("SobelTryIt").disabled = false;
    document.getElementById("LaplacianTryIt").disabled = false;
    document.getElementById("absSobelTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
