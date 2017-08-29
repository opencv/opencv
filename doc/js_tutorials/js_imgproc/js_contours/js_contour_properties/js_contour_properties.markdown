Contour Properties {#tutorial_js_contour_properties}
==================

Goal
----

-   Here we will learn to extract some frequently used properties of objects like Solidity, Equivalent
Diameter, Mask image, Mean Intensity etc.

1. Aspect Ratio
---------------

It is the ratio of width to height of bounding rect of the object.

\f[Aspect \; Ratio = \frac{Width}{Height}\f]
@code{.js}
let rect = cv.boundingRect(cnt);
let aspectRatio = rect.width / rect.height;
@endcode

2. Extent
---------

Extent is the ratio of contour area to bounding rectangle area.

\f[Extent = \frac{Object \; Area}{Bounding \; Rectangle \; Area}\f]
@code{.js}
let area = cv.contourArea(cnt, false);
let rect = cv.boundingRect(cnt));
let rectArea = rect.width * rect.height;
let extent = area / rectArea;
@endcode

3. Solidity
-----------

Solidity is the ratio of contour area to its convex hull area.

\f[Solidity = \frac{Contour \; Area}{Convex \; Hull \; Area}\f]
@code{.js}
let area = cv.contourArea(cnt, false);
cv.convexHull(cnt, hull, false, true);
let hullArea = cv.contourArea(hull, false);
let solidity = area / hullArea;
@endcode

4. Equivalent Diameter
----------------------

Equivalent Diameter is the diameter of the circle whose area is same as the contour area.

\f[Equivalent \; Diameter = \sqrt{\frac{4 \times Contour \; Area}{\pi}}\f]
@code{.js}
let area = cv.contourArea(cnt, false);
let equiDiameter = Math.sqrt(4 * area / Math.PI);
@endcode

5. Orientation
--------------

Orientation is the angle at which object is directed. Following method also gives the Major Axis and
Minor Axis lengths.
@code{.js}
let rotatedRect = cv.fitEllipse(cnt);
let angle = rotatedRect.angle;
@endcode

6. Mask and Pixel Points
------------------------

In some cases, we may need all the points which comprises that object.

We use the function: **cv.transpose (src, dst)**
@param src   input array.
@param dst   output array of the same type as src.

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
<div id="transposeCodeArea">
<h2>Input your code</h2>
<button id="transposeTryIt" disabled="true" onclick="transposeExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="transposeTestCode" spellcheck="false">
let src = cv.imread("transposeCanvasInput");
let dst = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(src, src, 120, 200, cv.THRESH_BINARY);
cv.transpose(src, dst);
cv.imshow("transposeCanvasOutput", dst);
src.delete(); dst.delete();
</textarea>
<p class="err" id="transposeErr"></p>
</div>
<div id="transposeShowcase">
    <div>
        <canvas id="transposeCanvasInput"></canvas>
        <canvas id="transposeCanvasOutput"></canvas>
    </div>
    <input type="file" id="transposeInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function transposeExecuteCode() {
    let transposeText = document.getElementById("transposeTestCode").value;
    try {
        eval(transposeText);
        document.getElementById("transposeErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("transposeErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "transposeCanvasInput");
let transposeInputElement = document.getElementById("transposeInput");
transposeInputElement.addEventListener("change", transposeHandleFiles, false);
function transposeHandleFiles(e) {
    let transposeUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(transposeUrl, "transposeCanvasInput");
}

function onReady() {
    document.getElementById("transposeTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly

7. Maximum Value, Minimum Value and their locations
---------------------------------------------------

We use the function: **cv.minMaxLoc(src, mask)**
@param src      input single-channel array.
@param mask     optional mask used to select a sub-array.

@code{.js}
let result = cv.minMaxLoc(src, mask);
let minVal = result.minVal;
let maxVal = result.maxVal;
let minLoc = result.minLoc;
let maxLoc = result.maxLoc;
@endcode

8. Mean Color or Mean Intensity
-------------------------------

Here, we can find the average color of an object. Or it can be average intensity of the object in
grayscale mode. We again use the same mask to do it.

We use the function: **cv.mean (src, mask)**
@param src   input array that should have from 1 to 4 channels so that the result can be stored in Scalar.
@param mask  optional operation mask.

@code{.js}
let average = cv.mean(src, mask);
@endcode
