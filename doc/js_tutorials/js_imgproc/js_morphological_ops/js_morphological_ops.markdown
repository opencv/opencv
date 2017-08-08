Morphological Transformations {#tutorial_js_morphological_ops}
=============================

Goal
----

In this chapter,
    -   We will learn different morphological operations like Erosion, Dilation, Opening, Closing
        etc.
    -   We will see different functions like : **cv.erode()**, **cv.dilate()**,
        **cv.morphologyEx()** etc.

Theory
------

Morphological transformations are some simple operations based on the image shape. It is normally
performed on binary images. It needs two inputs, one is our original image, second one is called
**structuring element** or **kernel** which decides the nature of operation. Two basic morphological
operators are Erosion and Dilation. Then its variant forms like Opening, Closing, Gradient etc also
comes into play. We will see them one-by-one with help of following image:

![image](shape.jpg)

### 1. Erosion

The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of
foreground object (Always try to keep foreground in white). So what it does? The kernel slides
through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) will be
considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).

So what happends is that, all the pixels near boundary will be discarded depending upon the size of
kernel. So the thickness or size of the foreground object decreases or simply white region decreases
in the image. It is useful for removing small white noises (as we have seen in colorspace chapter),
detach two connected objects etc.

We use the function: **cv.erode (src, dst, kernel, anchor = new cv.Point(-1, -1), iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())** 
@param src          input image; the number of channels can be arbitrary, but the depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F.
@param dst          output image of the same size and type as src.
@param kernel       structuring element used for erosion.
@param anchor       position of the anchor within the element; default value new cv.Point(-1, -1) means that the anchor is at the element center.
@param iterations   number of times erosion is applied.
@param borderType   pixel extrapolation method(see cv.BorderTypes).
@param borderValue  border value in case of a constant border

Try it
------

Here is a demo. Canvas elements named erodeCanvasInput and erodeCanvasOutput have been prepared. Choose an image and
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
<div id="erodeCodeArea">
<h2>Input your code</h2>
<button id="erodeTryIt" disabled="true" onclick="erodeExecuteCode()">Try it</button><br>
<textarea rows="9" cols="90" id="erodeTestCode" spellcheck="false">
let src = cv.imread("erodeCanvasInput");
let dst = new cv.Mat();
let M = cv.Mat.ones(5, 5, cv.CV_8U);
let anchor = new cv.Point(-1, -1);
// You can try more different parameters
cv.erode(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
cv.imshow("erodeCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); 
</textarea>
<p class="err" id="erodeErr"></p>
</div>
<div id="erodeShowcase">
    <div>
        <canvas id="erodeCanvasInput"></canvas>
        <canvas id="erodeCanvasOutput"></canvas>
    </div>
    <input type="file" id="erodeInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function erodeExecuteCode() {
    let erodeText = document.getElementById("erodeTestCode").value;
    try {
        eval(erodeText);
        document.getElementById("erodeErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("erodeErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "erodeCanvasInput");
let erodeInputElement = document.getElementById("erodeInput");
erodeInputElement.addEventListener("change", erodeHandleFiles, false);
function erodeHandleFiles(e) {
    let erodeUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(erodeUrl, "erodeCanvasInput");
}
</script>
</body>
\endhtmlonly

### 2. Dilation

It is just opposite of erosion. Here, a pixel element is '1' if atleast one pixel under the kernel
is '1'. So it increases the white region in the image or size of foreground object increases.
Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes
white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won't come
back, but our object area increases. It is also useful in joining broken parts of an object.

We use the function: **cv.dilate (src, dst, kernel, anchor = new cv.Point(-1, -1), iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())** 
@param src          input image; the number of channels can be arbitrary, but the depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F.
@param dst          output image of the same size and type as src.
@param kernel       structuring element used for dilation.
@param anchor       position of the anchor within the element; default value new cv.Point(-1, -1) means that the anchor is at the element center.
@param iterations   number of times dilation is applied.
@param borderType   pixel extrapolation method(see cv.BorderTypes).
@param borderValue  border value in case of a constant border

Try it
------

Here is a demo. Canvas elements named dilateCanvasInput and dilateCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="dilateCodeArea">
<h2>Input your code</h2>
<button id="dilateTryIt" disabled="true" onclick="dilateExecuteCode()">Try it</button><br>
<textarea rows="8" cols="90" id="dilateTestCode" spellcheck="false">
let src = cv.imread("dilateCanvasInput");
let dst = new cv.Mat();
let M = cv.Mat.ones(5, 5, cv.CV_8U);
let anchor = new cv.Point(-1, -1);
// You can try more different parameters
cv.dilate(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
cv.imshow("dilateCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); 
</textarea>
<p class="err" id="dilateErr"></p>
</div>
<div id="dilateShowcase">
    <div>
        <canvas id="dilateCanvasInput"></canvas>
        <canvas id="dilateCanvasOutput"></canvas>
    </div>
    <input type="file" id="dilateInput" name="file" />
</div>
<script>
function dilateExecuteCode() {
    let dilateText = document.getElementById("dilateTestCode").value;
    try {
        eval(dilateText);
        document.getElementById("dilateErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("dilateErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "dilateCanvasInput");
let dilateInputElement = document.getElementById("dilateInput");
dilateInputElement.addEventListener("change", dilateHandleFiles, false);
function dilateHandleFiles(e) {
    let dilateUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(dilateUrl, "dilateCanvasInput");
}
</script>
</body>
\endhtmlonly

### 3. Opening

Opening is just another name of **erosion followed by dilation**. It is useful in removing noise.

We use the function: **cv.morphologyEx (src, dst, op, kernel, anchor = new cv.Point(-1, -1), iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())** 
@param src          source image. The number of channels can be arbitrary. The depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F
@param dst          destination image of the same size and type as source image.
@param op           type of a morphological operation, (see cv.MorphTypes).
@param kernel       structuring element. It can be created using cv.getStructuringElement.
@param anchor       anchor position with the kernel. Negative values mean that the anchor is at the kernel center.
@param iterations   number of times dilation is applied.
@param borderType   pixel extrapolation method(see cv.BorderTypes).
@param borderValue  border value in case of a constant border. The default value has a special meaning.

Try it
------

Here is a demo. Canvas elements named openingCanvasInput and openingCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="openingCodeArea">
<h2>Input your code</h2>
<button id="openingTryIt" disabled="true" onclick="openingExecuteCode()">Try it</button><br>
<textarea rows="8" cols="90" id="openingTestCode" spellcheck="false">
let src = cv.imread("openingCanvasInput");
let dst = new cv.Mat();
let M = cv.Mat.ones(5, 5, cv.CV_8U);
let anchor = new cv.Point(-1, -1);
// You can try more different parameters
cv.morphologyEx(src, dst, cv.MORPH_OPEN, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
cv.imshow("openingCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); 
</textarea>
<p class="err" id="openingErr"></p>
</div>
<div id="openingShowcase">
    <div>
        <canvas id="openingCanvasInput"></canvas>
        <canvas id="openingCanvasOutput"></canvas>
    </div>
    <input type="file" id="openingInput" name="file" />
</div>
<script>
function openingExecuteCode() {
    let openingText = document.getElementById("openingTestCode").value;
    try {
        eval(openingText);
        document.getElementById("openingErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("openingErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "openingCanvasInput");
let openingInputElement = document.getElementById("openingInput");
openingInputElement.addEventListener("change", openingHandleFiles, false);
function openingHandleFiles(e) {
    let openingUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(openingUrl, "openingCanvasInput");
}
</script>
</body>
\endhtmlonly

### 4. Closing

Closing is reverse of Opening, **Dilation followed by Erosion**. It is useful in closing small holes
inside the foreground objects, or small black points on the object.

Try it
------

Here is a demo. Canvas elements named closingCanvasInput and closingCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="closingCodeArea">
<h2>Input your code</h2>
<button id="closingTryIt" disabled="true" onclick="closingExecuteCode()">Try it</button><br>
<textarea rows="8" cols="90" id="closingTestCode" spellcheck="false">
let src = cv.imread("closingCanvasInput");
let dst = new cv.Mat();
let M = cv.Mat.ones(5, 5, cv.CV_8U);
// You can try more different parameters
cv.morphologyEx(src, dst, cv.MORPH_CLOSE, M);
cv.imshow("closingCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); 
</textarea>
<p class="err" id="closingErr"></p>
</div>
<div id="closingShowcase">
    <div>
        <canvas id="closingCanvasInput"></canvas>
        <canvas id="closingCanvasOutput"></canvas>
    </div>
    <input type="file" id="closingInput" name="file" />
</div>
<script>
function closingExecuteCode() {
    let closingText = document.getElementById("closingTestCode").value;
    try {
        eval(closingText);
        document.getElementById("closingErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("closingErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "closingCanvasInput");
let closingInputElement = document.getElementById("closingInput");
closingInputElement.addEventListener("change", closingHandleFiles, false);
function closingHandleFiles(e) {
    let closingUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(closingUrl, "closingCanvasInput");
}
</script>
</body>
\endhtmlonly

### 5. Morphological Gradient

It is the difference between dilation and erosion of an image.

The result will look like the outline of the object.

Try it
------

Here is a demo. Canvas elements named gradientCanvasInput and gradientCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="gradientCodeArea">
<h2>Input your code</h2>
<button id="gradientTryIt" disabled="true" onclick="gradientExecuteCode()">Try it</button><br>
<textarea rows="8" cols="90" id="gradientTestCode" spellcheck="false">
let src = cv.imread("gradientCanvasInput");
cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
let dst = new cv.Mat();
let M = cv.Mat.ones(5, 5, cv.CV_8U);
// You can try more different parameters
cv.morphologyEx(src, dst, cv.MORPH_GRADIENT, M);
cv.imshow("gradientCanvasOutput", dst);
src.delete(); dst.delete(); M.delete();
</textarea>
<p class="err" id="gradientErr"></p>
</div>
<div id="gradientShowcase">
    <div>
        <canvas id="gradientCanvasInput"></canvas>
        <canvas id="gradientCanvasOutput"></canvas>
    </div>
    <input type="file" id="gradientInput" name="file" />
</div>
<script>
function gradientExecuteCode() {
    let gradientText = document.getElementById("gradientTestCode").value;
    try {
        eval(gradientText);
        document.getElementById("gradientErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("gradientErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "gradientCanvasInput");
let gradientInputElement = document.getElementById("gradientInput");
gradientInputElement.addEventListener("change", gradientHandleFiles, false);
function gradientHandleFiles(e) {
    let gradientUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(gradientUrl, "gradientCanvasInput");
}
</script>
</body>
\endhtmlonly

### 6. Top Hat

It is the difference between input image and Opening of the image. 

Try it
------

Here is a demo. Canvas elements named topHatCanvasInput and topHatCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="topHatCodeArea">
<h2>Input your code</h2>
<button id="topHatTryIt" disabled="true" onclick="topHatExecuteCode()">Try it</button><br>
<textarea rows="8" cols="90" id="topHatTestCode" spellcheck="false">
let src = cv.imread("topHatCanvasInput");
cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
let dst = new cv.Mat();
let M = cv.Mat.ones(9, 9, cv.CV_8U);
// You can try more different parameters
cv.morphologyEx(src, dst, cv.MORPH_TOPHAT, M);
cv.imshow("topHatCanvasOutput", dst);
src.delete(); dst.delete(); M.delete();
</textarea>
<p class="err" id="topHatErr"></p>
</div>
<div id="topHatShowcase">
    <div>
        <canvas id="topHatCanvasInput"></canvas>
        <canvas id="topHatCanvasOutput"></canvas>
    </div>
    <input type="file" id="topHatInput" name="file" />
</div>
<script>
function topHatExecuteCode() {
    let topHatText = document.getElementById("topHatTestCode").value;
    try {
        eval(topHatText);
        document.getElementById("topHatErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("topHatErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "topHatCanvasInput");
let topHatInputElement = document.getElementById("topHatInput");
topHatInputElement.addEventListener("change", topHatHandleFiles, false);
function topHatHandleFiles(e) {
    let topHatUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(topHatUrl, "topHatCanvasInput");
}
</script>
</body>
\endhtmlonly

### 7. Black Hat

It is the difference between the closing of the input image and input image.

Try it
------

Here is a demo. Canvas elements named blackHatCanvasInput and blackHatCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="blackHatCodeArea">
<h2>Input your code</h2>
<button id="blackHatTryIt" disabled="true" onclick="blackHatExecuteCode()">Try it</button><br>
<textarea rows="8" cols="90" id="blackHatTestCode" spellcheck="false">
let src = cv.imread("blackHatCanvasInput");
cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
let dst = new cv.Mat();
let M = cv.Mat.ones(53, 53, cv.CV_8U);
// You can try more different parameters
cv.morphologyEx(src, dst, cv.MORPH_BLACKHAT, M);
cv.imshow("blackHatCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); 
</textarea>
<p class="err" id="blackHatErr"></p>
</div>
<div id="blackHatShowcase">
    <div>
        <canvas id="blackHatCanvasInput"></canvas>
        <canvas id="blackHatCanvasOutput"></canvas>
    </div>
    <input type="file" id="blackHatInput" name="file" />
</div>
<script>
function blackHatExecuteCode() {
    let blackHatText = document.getElementById("blackHatTestCode").value;
    try {
        eval(blackHatText);
        document.getElementById("blackHatErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("blackHatErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "blackHatCanvasInput");
let blackHatInputElement = document.getElementById("blackHatInput");
blackHatInputElement.addEventListener("change", blackHatHandleFiles, false);
function blackHatHandleFiles(e) {
    let blackHatUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(blackHatUrl, "blackHatCanvasInput");
}
</script>
</body>
\endhtmlonly

Structuring Element
-------------------

We manually created a structuring elements in the previous examples with help of cv.Mat.ones. It is
rectangular shape. But in some cases, you may need elliptical/circular shaped kernels. So for this
purpose, OpenCV has a function, **cv.getStructuringElement()**. You just pass the shape and size of
the kernel, you get the desired kernel.

We use the function: **cv.getStructuringElement (shape, ksize, anchor = new cv.Point(-1, -1))** 
@param shape          element shape that could be one of cv.MorphShapes
@param ksize          size of the structuring element.
@param anchor         anchor position within the element. The default value [−1,−1] means that the anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor position. In other cases the anchor just regulates how much the result of the morphological operation is shifted.

Try it
------

Here is a demo. Canvas elements named getStructuringElementCanvasInput and getStructuringElementCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="getStructuringElementCodeArea">
<h2>Input your code</h2>
<button id="getStructuringElementTryIt" disabled="true" onclick="getStructuringElementExecuteCode()">Try it</button><br>
<textarea rows="10" cols="90" id="getStructuringElementTestCode" spellcheck="false">
let src = cv.imread("getStructuringElementCanvasInput");
cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
let dst = new cv.Mat();
let M = new cv.Mat();
let ksize = new cv.Size(5, 5);
// You can try more different parameters
M = cv.getStructuringElement(cv.MORPH_CROSS, ksize);
cv.morphologyEx(src, dst, cv.MORPH_GRADIENT, M);
cv.imshow("getStructuringElementCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); 
</textarea>
<p class="err" id="getStructuringElementErr"></p>
</div>
<div id="getStructuringElementShowcase">
    <div>
        <canvas id="getStructuringElementCanvasInput"></canvas>
        <canvas id="getStructuringElementCanvasOutput"></canvas>
    </div>
    <input type="file" id="getStructuringElementInput" name="file" />
</div>
<script>
function getStructuringElementExecuteCode() {
    let getStructuringElementText = document.getElementById("getStructuringElementTestCode").value;
    try {
        eval(getStructuringElementText);
        document.getElementById("getStructuringElementErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("getStructuringElementErr").innerHTML = err;
    }
}

loadImageToCanvas("shape.jpg", "getStructuringElementCanvasInput");
let getStructuringElementInputElement = document.getElementById("getStructuringElementInput");
getStructuringElementInputElement.addEventListener("change", getStructuringElementHandleFiles, false);
function getStructuringElementHandleFiles(e) {
    let getStructuringElementUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(getStructuringElementUrl, "getStructuringElementCanvasInput");
}
function onReady() {
    document.getElementById("dilateTryIt").disabled = false;
    document.getElementById("erodeTryIt").disabled = false;
    document.getElementById("openingTryIt").disabled = false;
    document.getElementById("closingTryIt").disabled = false;
    document.getElementById("gradientTryIt").disabled = false;
    document.getElementById("topHatTryIt").disabled = false;
    document.getElementById("blackHatTryIt").disabled = false;
    document.getElementById("getStructuringElementTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
