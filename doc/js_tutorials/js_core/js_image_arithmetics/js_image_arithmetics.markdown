Arithmetic Operations on Images {#tutorial_js_image_arithmetics}
===============================

Goal
----

-   Learn several arithmetic operations on images like addition, subtraction, bitwise operations
    etc.
-   You will learn these functions : **cv.add()**, **cv.subtract()**  etc.

Image Addition
--------------

You can add two images by OpenCV function, cv.add(). res = img1 + img2. Both images should be of same depth and type.

For example, consider below sample:
@code{.js}
let src1 = cv.imread("canvasInput1");
let src2 = cv.imread("canvasInput2");
let dst = new cv.Mat();
let mask = new cv.Mat();
let dtype = -1;
cv.add(src1, src2, dst, mask, dtype);
src1.delete(); src2.delete(); dst.delete(); mask.delete();
@endcode

Image Subtraction
--------------

You can subtract two images by OpenCV function, cv.subtract(). res = img1 - img2. Both images should be of same depth and type.

For example, consider below sample:
@code{.js}
let src1 = cv.imread("canvasInput1");
let src2 = cv.imread("canvasInput2");
let dst = new cv.Mat();
let mask = new cv.Mat();
let dtype = -1;
cv.subtract(src1, src2, dst, mask, dtype);
src1.delete(); src2.delete(); dst.delete(); mask.delete();
@endcode

Bitwise Operations
------------------

This includes bitwise AND, OR, NOT and XOR operations. They will be highly useful while extracting
any part of the image, defining and working with non-rectangular
ROI etc. Below we will see an example on how to change a particular region of an image.

I want to put OpenCV logo above an image. If I add two images, it will change color. If I blend it,
I get an transparent effect. But I want it to be opaque. If it was a rectangular region, I could use
ROI as we did in last chapter. But OpenCV logo is a not a rectangular shape. So you can do it with
bitwise operations.

Try it
------

Here is a demo. Canvas elements named imageCanvasInput, logoCanvasInput and bitwiseCanvasOutput have been prepared. Choose an image and
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
<div id="bitwiseCodeArea">
<h2>Input your code</h2>
<button id="bitwiseTryIt" disabled="true" onclick="bitwiseExecuteCode()">Try it</button><br>
<textarea rows="13" cols="80" id="bitwiseTestCode" spellcheck="false">
let src = cv.imread("imageCanvasInput");
let logo = cv.imread("logoCanvasInput");
let dst = new cv.Mat();
let roi = new cv.Mat();
let mask = new cv.Mat();
let maskInv = new cv.Mat();
let imgBg = new cv.Mat();
let imgFg = new cv.Mat();
let sum = new cv.Mat();
let rect = new cv.Rect(0, 0, logo.cols, logo.rows);

// I want to put logo on top-left corner, So I create a ROI
roi = src.getRoiRect(rect);

// create a mask of logo and create its inverse mask also
cv.cvtColor(logo, mask, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(mask, mask, 100, 255, cv.THRESH_BINARY);
cv.bitwise_not(mask, maskInv);

// black-out the area of logo in ROI
cv.bitwise_and(roi, roi, imgBg, maskInv);

// take only region of logo from logo image
cv.bitwise_and(logo, logo, imgFg, mask);

// put logo in ROI and modify the main image
cv.add(imgBg, imgFg, sum);

dst = src.clone();
for(let i = 0; i < logo.rows; i++)
    for(let j = 0; j < logo.cols; j++) 
        dst.ucharPtr(i, j)[0] = sum.ucharPtr(i, j)[0];
cv.imshow("bitwiseCanvasOutput", dst);
src.delete(); dst.delete(); logo.delete(); roi.delete(); mask.delete(); maskInv.delete(); imgBg.delete(); imgFg.delete(); sum.delete(); 
</textarea>
<p class="err" id="bitwiseErr"></p>
</div>
<div id="bitwiseShowcase">
    <div>
        <p>Logo</p>
        <canvas id="logoCanvasInput"></canvas>
        <input type="file" id="templateInput" name="file" />
    </div>
    <div>
        <p>Image</p>
        <canvas id="imageCanvasInput"></canvas>
        <input type="file" id="imageInput" name="file" />
    </div>
    <div>
        <p>Result</p>
        <canvas id="bitwiseCanvasOutput"></canvas>
    </div>
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function bitwiseExecuteCode() {
    let bitwiseText = document.getElementById("bitwiseTestCode").value;
    try {
        eval(bitwiseText);
        document.getElementById("bitwiseErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("bitwiseErr").innerHTML = err;
    }
}

loadImageToCanvas("lenaFace.png", "logoCanvasInput");
loadImageToCanvas("lena.jpg", "imageCanvasInput");

let templateInputElement = document.getElementById("templateInput");
templateInputElement.addEventListener("change", templateHandleFiles, false);
function templateHandleFiles(e) {
    let templateUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(templateUrl, "logoCanvasInput");
}

let imageInputElement = document.getElementById("imageInput");
imageInputElement.addEventListener("change", imageHandleFiles, false);
function imageHandleFiles(e) {
    let imageUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(imageUrl, "imageCanvasInput");
}

function onReady() {
    document.getElementById("bitwiseTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly