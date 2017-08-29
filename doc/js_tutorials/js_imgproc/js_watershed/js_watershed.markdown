Image Segmentation with Watershed Algorithm {#tutorial_js_watershed}
===========================================

Goal
----

-   We will learn how to use marker-based image segmentation using watershed algorithm
-   We will learn: **cv.watershed()**

Theory
------

Any grayscale image can be viewed as a topographic surface where high intensity denotes peaks and
hills while low intensity denotes valleys. You start filling every isolated valleys (local minima)
with different colored water (labels). As the water rises, depending on the peaks (gradients)
nearby, water from different valleys, obviously with different colors will start to merge. To avoid
that, you build barriers in the locations where water merges. You continue the work of filling water
and building barriers until all the peaks are under water. Then the barriers you created gives you
the segmentation result. This is the "philosophy" behind the watershed. You can visit the [CMM
webpage on watershed](http://cmm.ensmp.fr/~beucher/wtshed.html) to understand it with the help of
some animations.

But this approach gives you oversegmented result due to noise or any other irregularities in the
image. So OpenCV implemented a marker-based watershed algorithm where you specify which are all
valley points are to be merged and which are not. It is an interactive image segmentation. What we
do is to give different labels for our object we know. Label the region which we are sure of being
the foreground or object with one color (or intensity), label the region which we are sure of being
background or non-object with another color and finally the region which we are not sure of
anything, label it with 0. That is our marker. Then apply watershed algorithm. Then our marker will
be updated with the labels we gave, and the boundaries of objects will have a value of -1.

Code
----

Below we will see an example on how to use the Distance Transform along with watershed to segment
mutually touching objects.

Consider the coins image below, the coins are touching each other. Even if you threshold it, it will
be touching each other.

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
<div id="imgCodeArea">
<h2>Image</h2>
</div>
<div id="imgShowcase">
    <div>
        <canvas id="imgCanvasInput"></canvas>
    </div>
    <input type="file" id="imgInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
loadImageToCanvas("coins.jpg", "imgCanvasInput");
let imgInputElement = document.getElementById("imgInput");
imgInputElement.addEventListener("change", imgHandleFiles, false);
function imgHandleFiles(e) {
    let imgUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(imgUrl, "imgCanvasInput");
}
</script>
</body>
\endhtmlonly

We start with finding an approximate estimate of the coins. For that, we can use the Otsu's
binarization.

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
<div id="thresholdCodeArea">
<p><strong>Threshold Image</strong></p>
<button id="thresholdTryIt" disabled="true" onclick="thresholdExecuteCode()">Try it</button><br>
<textarea rows="7" cols="80" id="thresholdTestCode" spellcheck="false">
let src = cv.imread("imgCanvasInput");
let dst = new cv.Mat(), gray = new cv.Mat();

// gray and threshold image
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

cv.imshow("thresholdCanvasOutput", gray);
src.delete(); dst.delete(); gray.delete();
</textarea>
<p class="err" id="thresholdErr"></p>
</div>
<div id="thresholdShowcase">
    <div>
        <canvas id="thresholdCanvasOutput"></canvas>
    </div>
</div>
<script>
function thresholdExecuteCode() {
    let thresholdText = document.getElementById("thresholdTestCode").value;
    try {
        eval(thresholdText);
        document.getElementById("thresholdErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("thresholdErr").innerHTML = err;
    }
}
</script>
</body>
\endhtmlonly

Now we need to remove any small white noises in the image. For that we can use morphological
opening. To remove any small holes in the object, we can use morphological closing. So, now we know
for sure that region near to center of objects are foreground and region much away from the object
are background. Only region we are not sure is the boundary region of coins.

So we need to extract the area which we are sure they are coins. Erosion removes the boundary
pixels. So whatever remaining, we can be sure it is coin. That would work if objects were not
touching each other. But since they are touching each other, another good option would be to find
the distance transform and apply a proper threshold. Next we need to find the area which we are sure
they are not coins. For that, we dilate the result. Dilation increases object boundary to
background. This way, we can make sure whatever region in background in result is really a
background, since boundary region is removed. See the image below.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="backgroundCodeArea">
<p><strong>Background</strong></p>
<button id="backgroundTryIt" disabled="true" onclick="backgroundExecuteCode()">Try it</button><br>
<textarea rows="13" cols="90" id="backgroundTestCode" spellcheck="false">
let src = cv.imread("imgCanvasInput");
let dst = new cv.Mat(), gray = new cv.Mat(), opening = new cv.Mat(), coinsBg = new cv.Mat();
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

// get background
let M = cv.Mat.ones(3, 3, cv.CV_8U);
cv.erode(gray, gray, M);
cv.dilate(gray, opening, M);
cv.dilate(opening, coinsBg, M, new cv.Point(-1, -1), 3);

cv.imshow("backgroundCanvasOutput", coinsBg);
src.delete(); dst.delete(); gray.delete(); opening.delete(); coinsBg.delete(); M.delete();
</textarea>
<p class="err" id="backgroundErr"></p>
</div>
<div id="backgroundShowcase">
    <div>
        <canvas id="backgroundCanvasOutput"></canvas>
    </div>
</div>
<script>
function backgroundExecuteCode() {
    let backgroundText = document.getElementById("backgroundTestCode").value;
    try {
        eval(backgroundText);
        document.getElementById("backgroundErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("backgroundErr").innerHTML = err;
    }
}
</script>
</body>
\endhtmlonly

The remaining regions are those which we don't have any idea, whether it is coins or background.
Watershed algorithm should find it. These areas are normally around the boundaries of coins where
foreground and background meet (Or even two different coins meet). We call it border. It can be
obtained from subtracting sure_fg area from sure_bg area.

We use the function: **cv.distanceTransform (src, dst, distanceType, maskSize, labelType = cv.CV_32F)**

@param src           8-bit, single-channel (binary) source image.
@param dst           output image with calculated distances. It is a 8-bit or 32-bit floating-point, single-channel image of the same size as src.
@param distanceType  type of distance(see cv.DistanceTypes).
@param maskSize      size of the distance transform mask, see (cv.DistanceTransformMasks).
@param labelType     type of output image. It can be cv.CV_8U or cv.CV_32F. Type cv.CV_8U can be used only for the first variant of the function and distanceType == DIST_L1.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="distanceTransformCodeArea">
<p><strong>Distance Transform</strong></p>
<button id="distanceTransformTryIt" disabled="true" onclick="distanceTransformExecuteCode()">Try it</button><br>
<textarea rows="14" cols="90" id="distanceTransformTestCode" spellcheck="false">
let src = cv.imread("imgCanvasInput");
let dst = new cv.Mat(), gray = new cv.Mat(), opening = new cv.Mat(), coinsBg = new cv.Mat(), coinsFg = new cv.Mat(), distTrans = new cv.Mat();
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
let M = cv.Mat.ones(3, 3, cv.CV_8U);
cv.erode(gray, gray, M);
cv.dilate(gray, opening, M);
cv.dilate(opening, coinsBg, M, new cv.Point(-1, -1), 3);

// distance transform
cv.distanceTransform(opening, distTrans, cv.DIST_L2, 5);
cv.normalize(distTrans, distTrans, 1, 0, cv.NORM_INF);

cv.imshow("distanceTransformCanvasOutput", distTrans);
src.delete(); dst.delete(); gray.delete(); opening.delete(); coinsBg.delete(); coinsFg.delete(); distTrans.delete(); M.delete();
</textarea>
<p class="err" id="distanceTransformErr"></p>
</div>
<div id="distanceTransformShowcase">
    <div>
        <canvas id="distanceTransformCanvasOutput"></canvas>
    </div>
</div>
<script>
function distanceTransformExecuteCode() {
    let distanceTransformText = document.getElementById("distanceTransformTestCode").value;
    try {
        eval(distanceTransformText);
        document.getElementById("distanceTransformErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("distanceTransformErr").innerHTML = err;
    }
}
</script>
</body>
\endhtmlonly

In the thresholded image, we get some regions of coins which we are sure of coins
and they are detached now. (In some cases, you may be interested in only foreground segmentation,
not in separating the mutually touching objects. In that case, you need not use distance transform,
just erosion is sufficient. Erosion is just another method to extract sure foreground area, that's
all.)

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="foregroundCodeArea">
<p><strong>Foreground</strong></p>
<button id="foregroundTryIt" disabled="true" onclick="foregroundExecuteCode()">Try it</button><br>
<textarea rows="15" cols="90" id="foregroundTestCode" spellcheck="false">
let src = cv.imread("imgCanvasInput");
let dst = new cv.Mat(), gray = new cv.Mat(), opening = new cv.Mat(), coinsBg = new cv.Mat(), coinsFg = new cv.Mat(), distTrans = new cv.Mat();
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
let M = cv.Mat.ones(3, 3, cv.CV_8U);
cv.erode(gray, gray, M);
cv.dilate(gray, opening, M);
cv.dilate(opening, coinsBg, M, new cv.Point(-1, -1), 3);
cv.distanceTransform(opening, distTrans, cv.DIST_L2, 5);
cv.normalize(distTrans, distTrans, 1, 0, cv.NORM_INF);

// get foreground
cv.threshold(distTrans, coinsFg, 0.7 * 1, 255, cv.THRESH_BINARY);

cv.imshow("foregroundCanvasOutput", coinsFg);
src.delete(); dst.delete(); gray.delete(); opening.delete(); coinsBg.delete(); coinsFg.delete(); distTrans.delete(); M.delete();
</textarea>
<p class="err" id="foregroundErr"></p>
</div>
<div id="foregroundShowcase">
    <div>
        <canvas id="foregroundCanvasOutput"></canvas>
    </div>
</div>
<script>
function foregroundExecuteCode() {
    let foregroundText = document.getElementById("foregroundTestCode").value;
    try {
        eval(foregroundText);
        document.getElementById("foregroundErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("foregroundErr").innerHTML = err;
    }
}
</script>
</body>
\endhtmlonly

Now we know for sure which are region of coins, which are background and all. So we create marker
(it is an array of same size as that of original image, but with int32 datatype) and label the
regions inside it. The regions we know for sure (whether foreground or background) are labelled with
any positive integers, but different integers, and the area we don't know for sure are just left as
zero. For this we use **cv.connectedComponents()**. It labels background of the image with 0, then
other objects are labelled with integers starting from 1.

But we know that if background is marked with 0, watershed will consider it as unknown area. So we
want to mark it with different integer. Instead, we will mark unknown region, defined by unknown,
with 0.

Now our marker is ready. It is time for final step, apply watershed. Then marker image will be
modified. The boundary region will be marked with -1.

We use the function: **cv.connectedComponents (image, labels, connectivity = 8, ltype = cv.CV_32S)**
@param image         the 8-bit single-channel image to be labeled.
@param labels        destination labeled image(cv.CV_32SC1 type).
@param connectivity  8 or 4 for 8-way or 4-way connectivity respectively.
@param ltype         output image label type. Currently cv.CV_32S and cv.CV_16U are supported.

We use the function: **cv.watershed (image, markers)**

@param image         input 8-bit 3-channel image.
@param markers       input/output 32-bit single-channel image (map) of markers. It should have the same size as image .

Try it
------

Try this demo using the code above. Canvas elements named watershedCanvasInput and watershedCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="watershedCodeArea">
<h2>Input your code</h2>
<button id="watershedTryIt" disabled="true" onclick="watershedExecuteCode()">Try it</button><br>
<textarea rows="25" cols="90" id="watershedTestCode" spellcheck="false">
let src = cv.imread("watershedCanvasInput");
let dst = new cv.Mat(), gray = new cv.Mat(), opening = new cv.Mat(), coinsBg = new cv.Mat(), coinsFg = new cv.Mat(), distTrans = new cv.Mat(), unknown = new cv.Mat(), markers = new cv.Mat();
// gray and threshold image
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
// get background
let M = cv.Mat.ones(3, 3, cv.CV_8U);
cv.erode(gray, gray, M);
cv.dilate(gray, opening, M);
cv.dilate(opening, coinsBg, M, new cv.Point(-1, -1), 3);
// distance transform
cv.distanceTransform(opening, distTrans, cv.DIST_L2, 5);
cv.normalize(distTrans, distTrans, 1, 0, cv.NORM_INF);
// get foreground
cv.threshold(distTrans, coinsFg, 0.7 * 1, 255, cv.THRESH_BINARY);
coinsFg.convertTo(coinsFg, cv.CV_8U, 1, 0);
cv.subtract(coinsBg, coinsFg, unknown);
// get connected components markers
cv.connectedComponents(coinsFg, markers);
for (let i = 0; i < markers.rows; i++)
    for (let j = 0; j < markers.cols; j++) {
        markers.intPtr(i, j)[0] = markers.ucharPtr(i, j)[0] + 1;
        if (unknown.ucharPtr(i, j)[0] == 255) {
            markers.intPtr(i, j)[0] = 0;
        }
    }
cv.cvtColor(src, src, cv.COLOR_RGBA2RGB, 0);
cv.watershed(src, markers);
// draw barriers
for (let i = 0; i < markers.rows; i++)
    for (let j = 0; j < markers.cols; j++) {
        if (markers.intPtr(i, j)[0] == -1) {
            src.ucharPtr(i, j)[0] = 255;  //R
            src.ucharPtr(i, j)[1] = 0;    //G
            src.ucharPtr(i, j)[2] = 0;    //B
        }
    }
cv.imshow("watershedCanvasOutput", src);
src.delete(); dst.delete(); gray.delete(); opening.delete(); coinsBg.delete(); coinsFg.delete(); distTrans.delete(); unknown.delete(); markers.delete(); M.delete();
</textarea>
<p class="err" id="watershedErr"></p>
</div>
<div id="watershedShowcase">
    <div>
        <canvas id="watershedCanvasInput"></canvas>
        <canvas id="watershedCanvasOutput"></canvas>
    </div>
    <input type="file" id="watershedInput" name="file" />
</div>
<script>
function watershedExecuteCode() {
    let watershedText = document.getElementById("watershedTestCode").value;
    try {
        eval(watershedText);
        document.getElementById("watershedErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("watershedErr").innerHTML = err;
    }
}

loadImageToCanvas("coins.jpg", "watershedCanvasInput");
let watershedInputElement = document.getElementById("watershedInput");
watershedInputElement.addEventListener("change", watershedHandleFiles, false);
function watershedHandleFiles(e) {
    let watershedUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(watershedUrl, "watershedCanvasInput");
}

function onReady() {
    document.getElementById("thresholdTryIt").disabled = false;
    document.getElementById("backgroundTryIt").disabled = false;
    document.getElementById("distanceTransformTryIt").disabled = false;
    document.getElementById("foregroundTryIt").disabled = false;
    document.getElementById("watershedTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
