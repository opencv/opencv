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

![image](LinuxLogo.jpg)

### 1. Erosion

The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of
foreground object (Always try to keep foreground in white). So what it does? The kernel slides
through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) will be
considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).

So what happends is that, all the pixels near boundary will be discarded depending upon the size of
kernel. So the thickness or size of the foreground object decreases or simply white region decreases
in the image. It is useful for removing small white noises (as we have seen in colorspace chapter),
detach two connected objects etc.

We use the function: **cv.erode (src, dst, kernel, anchor = [-1, -1], iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())** 
@param src          input image; the number of channels can be arbitrary, but the depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F.
@param dst          output image of the same size and type as src.
@param kernel       structuring element used for erosion.
@param anchor       position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
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
</style>
</head>
<body>
<div id="erodeCodeArea">
<h2>Input your code</h2>
<button id="erodeTryIt" disabled="true" onclick="erodeExecuteCode()">Try it</button><br>
<textarea rows="9" cols="80" id="erodeTestCode" spellcheck="false">
var src = cv.imread("erodeCanvasInput");
var dst = new cv.Mat();
var M = cv.Mat.ones(5, 5, cv.CV_8U);
var S = new cv.Scalar();
// You can try more different conversion
cv.erode(src, dst, M, [-1, -1], 1, cv.BORDER_CONSTANT, S)
cv.imshow("erodeCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); S.delete();
</textarea>
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
    var erodeText = document.getElementById("erodeTestCode").value;
    eval(erodeText);
}

loadImageToCanvas("LinuxLogo.jpg", "erodeCanvasInput");
var erodeInputElement = document.getElementById("erodeInput");
erodeInputElement.addEventListener("change", erodeHandleFiles, false);
function erodeHandleFiles(e) {
    var erodeUrl = URL.createObjectURL(e.target.files[0]);
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

We use the function: **cv.dilate (src, dst, kernel, anchor = [-1, -1], iterations = 1, borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())** 
@param src          input image; the number of channels can be arbitrary, but the depth should be one of cv.CV_8U, cv.CV_16U, cv.CV_16S, cv.CV_32F or cv.CV_64F.
@param dst          output image of the same size and type as src.
@param kernel       structuring element used for dilation.
@param anchor       position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
@param iterations   number of times dilation is applied.
@param borderType   pixel extrapolation method.
@param borderValue  border value in case of a constant border

Try it
------

Here is a demo. Canvas elements named dilateCanvasInput and dilateCanvasOutput have been prepared. Choose an image and
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
<div id="dilateCodeArea">
<h2>Input your code</h2>
<button id="dilateTryIt" disabled="true" onclick="dilateExecuteCode()">Try it</button><br>
<textarea rows="9" cols="80" id="dilateTestCode" spellcheck="false">
var src = cv.imread("dilateCanvasInput");
var dst = new cv.Mat();
var M = cv.Mat.ones(5, 5, cv.CV_8U);
var S = new cv.Scalar();
// You can try more different conversion
cv.dilate(src, dst, M, [-1, -1], 1, cv.BORDER_CONSTANT, S)
cv.imshow("dilateCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); S.delete();
</textarea>
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
    var dilateText = document.getElementById("dilateTestCode").value;
    eval(dilateText);
}

loadImageToCanvas("LinuxLogo.jpg", "dilateCanvasInput");
var dilateInputElement = document.getElementById("dilateInput");
dilateInputElement.addEventListener("change", dilateHandleFiles, false);
function dilateHandleFiles(e) {
    var dilateUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(dilateUrl, "dilateCanvasInput");
}
</script>
</body>
\endhtmlonly

@note cv.morphologyEx() should be in the white list to implement following functions.

### 3. Opening

Opening is just another name of **erosion followed by dilation**. It is useful in removing noise, as
we explained above. 


### 4. Closing

Closing is reverse of Opening, **Dilation followed by Erosion**. It is useful in closing small holes
inside the foreground objects, or small black points on the object.


### 5. Morphological Gradient

It is the difference between dilation and erosion of an image.

The result will look like the outline of the object.


### 6. Top Hat

It is the difference between input image and Opening of the image. 


### 7. Black Hat

It is the difference between the closing of the input image and input image.


Structuring Element
-------------------

We manually created a structuring elements in the previous examples with help of Numpy. It is
rectangular shape. But in some cases, you may need elliptical/circular shaped kernels. So for this
purpose, OpenCV has a function, **cv.getStructuringElement()**. You just pass the shape and size of
the kernel, you get the desired kernel.

Try it
------

Here is a demo. Canvas elements named getStructuringElementCanvasInput and getStructuringElementCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

@note cv.getStructuringElement() should be in the white list to implement Structuring Element.

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
<div id="getStructuringElementCodeArea">
<h2>Input your code</h2>
<button id="getStructuringElementTryIt" disabled="true" onclick="getStructuringElementExecuteCode()">Try it</button><br>
<textarea rows="11" cols="80" id="getStructuringElementTestCode" spellcheck="false">

</textarea>
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
    var getStructuringElementText = document.getElementById("getStructuringElementTestCode").value;
    eval(getStructuringElementText);
}

loadImageToCanvas("lena.jpg", "getStructuringElementCanvasInput");
var getStructuringElementInputElement = document.getElementById("getStructuringElementInput");
getStructuringElementInputElement.addEventListener("change", getStructuringElementHandleFiles, false);
function getStructuringElementHandleFiles(e) {
    var getStructuringElementUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(getStructuringElementUrl, "getStructuringElementCanvasInput");
}
function onReady() {
    document.getElementById("erodeTryIt").disabled = false;
    document.getElementById("dilateTryIt").disabled = false;
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
