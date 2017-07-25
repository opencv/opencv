Smoothing Images {#tutorial_js_filtering}
================

Goals
-----

Learn to:
    -   Blur the images with various low pass filters
    -   Apply custom-made filters to images (2D convolution)

2D Convolution ( Image Filtering )
----------------------------------

As in one-dimensional signals, images also can be filtered with various low-pass filters(LPF),
high-pass filters(HPF) etc. LPF helps in removing noises, blurring the images etc. HPF filters helps
in finding edges in the images.

OpenCV provides a function **cv.filter2D()** to convolve a kernel with an image. As an example, we
will try an averaging filter on an image. A 5x5 averaging filter kernel will look like below:

\f[K =  \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1 & 1  \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}\f]

@note cv.filter2D() which can convolve an arbitrary kernel with an image is not in the white list.

Image Blurring (Image Smoothing)
--------------------------------

Image blurring is achieved by convolving the image with a low-pass filter kernel. It is useful for
removing noises. It actually removes high frequency content (eg: noise, edges) from the image. So
edges are blurred a little bit in this operation. (Well, there are blurring techniques which doesn't
blur the edges too). OpenCV provides mainly four types of blurring techniques.

### 1. Averaging

This is done by convolving image with a normalized box filter. It simply takes the average of all
the pixels under kernel area and replace the central element. This is done by the function
**cv.blur()** or **cv.boxFilter()**. Check the docs for more details about the kernel. We should
specify the width and height of kernel. A 3x3 normalized box filter would look like below:

\f[K =  \frac{1}{9} \begin{bmatrix} 1 & 1 & 1  \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}\f]

We use the functions: **cv.blur (src, dst, ksize, anchor = [-1,-1], borderType = cv.BORDER_DEFAULT)** 
@param src         input image; it can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
@param dst         output image of the same size and type as src.
@param ksize       blurring kernel size.
@param anchor      anchor point; anchor = [-1, -1] means that the anchor is at the kernel center. 
@param borderType  border mode used to extrapolate pixels outside of the image(see cv.BorderTypes).

**cv.boxFilter (src, dst, ddepth, ksize, anchor = [-1,-1], normalize = true, borderType = cv.BORDER_DEFAULT)**
@param src         input image.
@param dst         output image of the same size and type as src.
@param ddepth      the output image depth (-1 to use src.depth()).
@param ksize       blurring kernel size.
@param anchor      anchor point; anchor = [-1, -1] means that the anchor is at the kernel center. 
@param normalize   flag, specifying whether the kernel is normalized by its area or not.
@param borderType  border mode used to extrapolate pixels outside of the image(see cv.BorderTypes).

@note If you don't want to use normalized box filter, use **cv.boxFilter()**. Pass an argument
normalize = false to the function.

Try it
------

Here is a demo. Canvas elements named blurCanvasInput and blurCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
<style>
canvas {
    border: 1px solid black;
}
.err{
    color: red;
}
</style>
</head>
<body>
<div id="blurCodeArea">
<h2>Input your code</h2>
<button id="blurTryIt" disabled="true" onclick="blurExecuteCode()">Try it</button><br>
<textarea rows="9" cols="80" id="blurTestCode" spellcheck="false">
var src = cv.imread("blurCanvasInput");
var dst = new cv.Mat();
// You can try more different conversion
cv.blur(src, dst, [3, 3], [-1, -1], cv.BORDER_DEFAULT);
//cv.boxFilter(src, dst, -1, [3, 3], [-1, -1], true, cv.BORDER_DEFAULT)
cv.imshow("blurCanvasOutput", dst);
src.delete();
dst.delete();
</textarea>
<p class="err" id="blurErr"></p>
</div>
<div id="blurShowcase">
    <div>
        <canvas id="blurCanvasInput"></canvas>
        <canvas id="blurCanvasOutput"></canvas>
    </div>
    <input type="file" id="blurInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function blurExecuteCode() {
    var blurText = document.getElementById("blurTestCode").value;
    try {
        eval(blurText);
        document.getElementById("blurErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("blurErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "blurCanvasInput");
var blurInputElement = document.getElementById("blurInput");
blurInputElement.addEventListener("change", blurHandleFiles, false);
function blurHandleFiles(e) {
    var blurUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(blurUrl, "blurCanvasInput");
}
</script>
</body>
\endhtmlonly


### 2. Gaussian Blurring

In this, instead of box filter, gaussian kernel is used.

We use the function: **cv.GaussianBlur (src, dst, ksize, sigmaX, sigmaY = 0, borderType = cv.BORDER_DEFAULT)** 
@param src         input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
@param dst         output image of the same size and type as src.
@param ksize       blurring kernel size.
@param sigmaX      Gaussian kernel standard deviation in X direction.
@param sigmaY      Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
@param borderType  pixel extrapolation method(see cv.BorderTypes).

Try it
------

Here is a demo. Canvas elements named GaussianBlurCanvasInput and GaussianBlurCanvasOutput have been prepared. Choose an image and
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
<div id="GaussianBlurCodeArea">
<h2>Input your code</h2>
<button id="GaussianBlurTryIt" disabled="true" onclick="GaussianBlurExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="GaussianBlurTestCode" spellcheck="false">
var src = cv.imread("GaussianBlurCanvasInput");
var dst = new cv.Mat();
// You can try more different conversion
cv.GaussianBlur(src, dst, [5, 5], 0, 0, cv.BORDER_DEFAULT);
cv.imshow("GaussianBlurCanvasOutput", dst);
src.delete();
dst.delete();
</textarea>
<p class="err" id="GaussianBlurErr"></p>
</div>
<div id="GaussianBlurShowcase">
    <div>
        <canvas id="GaussianBlurCanvasInput"></canvas>
        <canvas id="GaussianBlurCanvasOutput"></canvas>
    </div>
    <input type="file" id="GaussianBlurInput" name="file" />
</div>
<script>
function GaussianBlurExecuteCode() {
    var GaussianBlurText = document.getElementById("GaussianBlurTestCode").value;
    try {
        eval(GaussianBlurText);
        document.getElementById("GaussianBlurErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("GaussianBlurErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "GaussianBlurCanvasInput");
var GaussianBlurInputElement = document.getElementById("GaussianBlurInput");
GaussianBlurInputElement.addEventListener("change", GaussianBlurHandleFiles, false);
function GaussianBlurHandleFiles(e) {
    var GaussianBlurUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(GaussianBlurUrl, "GaussianBlurCanvasInput");
}
</script>
</body>
\endhtmlonly


### 3. Median Blurring

Here, the function **cv.medianBlur()** takes median of all the pixels under kernel area and central
element is replaced with this median value. This is highly effective against salt-and-pepper noise
in the images. Interesting thing is that, in the above filters, central element is a newly
calculated value which may be a pixel value in the image or a new value. But in median blurring,
central element is always replaced by some pixel value in the image. It reduces the noise
effectively. Its kernel size should be a positive odd integer.

We use the function: **cv.medianBlur (src, dst, ksize)** 
@param src         input 1, 3, or 4 channel image; when ksize is 3 or 5, the image depth should be cv.CV_8U, cv.CV_16U, or cv.CV_32F, for larger aperture sizes, it can only be cv.CV_8U.
@param dst         destination array of the same size and type as src.
@param ksize       aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...

@note The median filter uses cv.BORDER_REPLICATE internally to cope with border pixels.

Try it
------

Here is a demo. Canvas elements named medianBlurCanvasInput and medianBlurCanvasOutput have been prepared. Choose an image and
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
<div id="medianBlurCodeArea">
<h2>Input your code</h2>
<button id="medianBlurTryIt" disabled="true" onclick="medianBlurExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="medianBlurTestCode" spellcheck="false">
var src = cv.imread("medianBlurCanvasInput");
var dst = new cv.Mat();
// You can try more different conversion
cv.medianBlur(src, dst, 5);
cv.imshow("medianBlurCanvasOutput", dst);
src.delete();
dst.delete();
</textarea>
<p class="err" id="medianBlurErr"></p>
</div>
<div id="medianBlurShowcase">
    <div>
        <canvas id="medianBlurCanvasInput"></canvas>
        <canvas id="medianBlurCanvasOutput"></canvas>
    </div>
    <input type="file" id="medianBlurInput" name="file" />
</div>
<script>
function medianBlurExecuteCode() {
    var medianBlurText = document.getElementById("medianBlurTestCode").value;
    try {
        eval(medianBlurText);
        document.getElementById("medianBlurErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("medianBlurErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "medianBlurCanvasInput");
var medianBlurInputElement = document.getElementById("medianBlurInput");
medianBlurInputElement.addEventListener("change", medianBlurHandleFiles, false);
function medianBlurHandleFiles(e) {
    var medianBlurUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(medianBlurUrl, "medianBlurCanvasInput");
}
</script>
</body>
\endhtmlonly


### 4. Bilateral Filtering

**cv.bilateralFilter()** is highly effective in noise removal while keeping edges sharp. But the
operation is slower compared to other filters. We already saw that gaussian filter takes the a
neighbourhood around the pixel and find its gaussian weighted average. This gaussian filter is a
function of space alone, that is, nearby pixels are considered while filtering. It doesn't consider
whether pixels have almost same intensity. It doesn't consider whether pixel is an edge pixel or
not. So it blurs the edges also, which we don't want to do.

Bilateral filter also takes a gaussian filter in space, but one more gaussian filter which is a
function of pixel difference. Gaussian function of space make sure only nearby pixels are considered
for blurring while gaussian function of intensity difference make sure only those pixels with
similar intensity to central pixel is considered for blurring. So it preserves the edges since
pixels at edges will have large intensity variation.

We use the function: **cv.bilateralFilter (src, dst, d, sigmaColor, sigmaSpace, borderType = cv.BORDER_DEFAULT)** 
@param src          source 8-bit or floating-point, 1-channel or 3-channel image.
@param dst          output image of the same size and type as src.
@param d            diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
@param sigmaColor   filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
@param sigmaSpace   filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough. When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
@param borderType   border mode used to extrapolate pixels outside of the image(see cv.BorderTypes).

@note For simplicity, you can set the 2 sigma values to be the same. If they are small (< 10), the filter will not have much effect, whereas if they are large (> 150), they will have a very strong effect, making the image look "cartoonish". Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering.

Try it
------

Here is a demo. Canvas elements named bilateralFilterCanvasInput and bilateralFilterCanvasOutput have been prepared. Choose an image and
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
<div id="bilateralFilterCodeArea">
<h2>Input your code</h2>
<button id="bilateralFilterTryIt" disabled="true" onclick="bilateralFilterExecuteCode()">Try it</button><br>
<textarea rows="9" cols="80" id="bilateralFilterTestCode" spellcheck="false">
var src = cv.imread("bilateralFilterCanvasInput");
var dst = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGBA2RGB, 0);
// You can try more different conversion
cv.bilateralFilter(src, dst, 9, 75, 75, cv.BORDER_DEFAULT);
cv.imshow("bilateralFilterCanvasOutput", dst);
src.delete();
dst.delete();
</textarea>
<p class="err" id="bilateralFilterErr"></p>
</div>
<div id="bilateralFilterShowcase">
    <div>
        <canvas id="bilateralFilterCanvasInput"></canvas>
        <canvas id="bilateralFilterCanvasOutput"></canvas>
    </div>
    <input type="file" id="bilateralFilterInput" name="file" />
</div>
<script>
function bilateralFilterExecuteCode() {
    var bilateralFilterText = document.getElementById("bilateralFilterTestCode").value;
    try {
        eval(bilateralFilterText);
        document.getElementById("bilateralFilterErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("bilateralFilterErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "bilateralFilterCanvasInput");
var bilateralFilterInputElement = document.getElementById("bilateralFilterInput");
bilateralFilterInputElement.addEventListener("change", bilateralFilterHandleFiles, false);
function bilateralFilterHandleFiles(e) {
    var bilateralFilterUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(bilateralFilterUrl, "bilateralFilterCanvasInput");
}

function onReady() {
    document.getElementById("blurTryIt").disabled = false;
    document.getElementById("GaussianBlurTryIt").disabled = false;
    document.getElementById("medianBlurTryIt").disabled = false;
    document.getElementById("bilateralFilterTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
