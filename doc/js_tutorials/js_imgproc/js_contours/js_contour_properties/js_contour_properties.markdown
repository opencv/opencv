Contour Properties {#tutorial_js_contour_properties}
==================

Here we will learn to extract some frequently used properties of objects like Solidity, Equivalent
Diameter, Mask image, Mean Intensity etc. More features can be found at [Matlab regionprops
documentation](http://www.mathworks.in/help/images/ref/regionprops.html).

*(NB : Centroid, Area, Perimeter etc also belong to this category, but we have seen it in last
chapter)*

1. Aspect Ratio
---------------

It is the ratio of width to height of bounding rect of the object.

\f[Aspect \; Ratio = \frac{Width}{Height}\f]
@code{.js}
var rect = cv.boundingRect(contours.get(0));
var aspect_ratio = rect.width/rect.height;
@endcode

2. Extent
---------

Extent is the ratio of contour area to bounding rectangle area.

\f[Extent = \frac{Object \; Area}{Bounding \; Rectangle \; Area}\f]
@code{.js}
var area = cv.contourArea(contours.get(0), false);
var rect = cv.boundingRect(contours.get(0));
var rect_area = rect.width*rect.height;
var extent = area/rect_area;
@endcode

3. Solidity
-----------

Solidity is the ratio of contour area to its convex hull area.

\f[Solidity = \frac{Contour \; Area}{Convex \; Hull \; Area}\f]
@code{.js}
var area = cv.contourArea(contours.get(0), false);
cv.convexHull(contours.get(0), hull, false, true);
var hull_area = cv.contourArea(hull, false);
var solidity = area/hull_area;
@endcode

4. Equivalent Diameter
----------------------

Equivalent Diameter is the diameter of the circle whose area is same as the contour area.

\f[Equivalent \; Diameter = \sqrt{\frac{4 \times Contour \; Area}{\pi}}\f]
@code{.js}
var area = cv.contourArea(contours.get(0), false);
var equi_diameter = Math.sqrt(4*area/Math.PI);
@endcode

5. Orientation
--------------

@note RotatedRect cv::fitEllipse(InputArray points)	

Orientation is the angle at which object is directed. Following method also gives the Major Axis and
Minor Axis lengths.
@code{.js}
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
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
</style>
</head>
<body>
<div id="transposeCodeArea">
<h2>Input your code</h2>
<button id="transposeTryIt" disabled="true" onclick="transposeExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="transposeTestCode" spellcheck="false">
var src = cv.imread("transposeCanvasInput");
var dst = new cv.Mat();
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(src, src, 120, 200, cv.THRESH_BINARY);
cv.transpose(src, dst)
cv.imshow("transposeCanvasOutput", dst);
src.delete(); dst.delete();
</textarea>
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
    var transposeText = document.getElementById("transposeTestCode").value;
    eval(transposeText);
}

loadImageToCanvas("lena.jpg", "transposeCanvasInput");
var transposeInputElement = document.getElementById("transposeInput");
transposeInputElement.addEventListener("change", transposeHandleFiles, false);
function transposeHandleFiles(e) {
    var transposeUrl = URL.createObjectURL(e.target.files[0]);
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

@note cv.minMaxLoc() is in ignore_list

We can find these parameters using a mask image.
@code{.js}
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)
@endcode

8. Mean Color or Mean Intensity
-------------------------------

Here, we can find the average color of an object. Or it can be average intensity of the object in
grayscale mode. We again use the same mask to do it.

We use the function: **cv.mean(src, mask)** 
@param src   input array that should have from 1 to 4 channels so that the result can be stored in Scalar.
@param mask  optional operation mask.

@code{.js}
var a = cv.mean(src, new cv.Mat);
@endcode

