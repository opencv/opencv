Geometric Transformations of Images {#tutorial_js_geometric_transformations}
===================================

Goals
-----

-   Learn how to apply different geometric transformation to images like translation, rotation, affine
    transformation etc.
-   You will learn these functions: **cv.resize**, **cv.warpAffine**, **cv.getAffineTransform** and **cv.warpPerspective** 

Transformations
---------------


### Scaling

Scaling is just resizing of the image. OpenCV comes with a function **cv.resize()** for this
purpose. The size of the image can be specified manually, or you can specify the scaling factor.
Different interpolation methods are used. Preferable interpolation methods are **cv.INTER_AREA**
for shrinking and **cv.INTER_CUBIC** (slow) & **cv.INTER_LINEAR** for zooming. 

We use the function: **cv.resize (src, dst, dsize, fx = 0, fy = 0, interpolation = cv.INTER_LINEAR)**
@param src    input image
@param dst    output image; it has the size dsize (when it is non-zero) or the size computed from src.size(), fx, and fy; the type of dst is the same as of src. 
@param dsize  output image size; if it equals zero, it is computed as:      
                 \f[𝚍𝚜𝚒𝚣𝚎 = 𝚂𝚒𝚣𝚎(𝚛𝚘𝚞𝚗𝚍(𝚏𝚡*𝚜𝚛𝚌.𝚌𝚘𝚕𝚜), 𝚛𝚘𝚞𝚗𝚍(𝚏𝚢*𝚜𝚛𝚌.𝚛𝚘𝚠𝚜))\f]
                 Either dsize or both fx and fy must be non-zero. 
@param fx     scale factor along the horizontal axis; when it equals 0, it is computed as  \f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜\f]        
                 
@param fy     scale factor along the vertical axis; when it equals 0, it is computed as \f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜\f] 
@param interpolation    interpolation method(see **cv.InterpolationFlags**)

Try it
------

Try this demo using the code above. Canvas elements named resizeCanvasInput and resizeCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

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
<div id="resizeCodeArea">
<h2>Input your code</h2>
<button id="resizeTryIt" disabled="true" onclick="resizeExecuteCode()">Try it</button><br>
<textarea rows="7" cols="80" id="resizeTestCode" spellcheck="false">
let src = cv.imread("resizeCanvasInput");
let dst = new cv.Mat();
let dsize = new cv.Size(300, 300);
// You can try more different parameters
cv.resize(src, dst, dsize, 0, 0, cv.INTER_AREA);
cv.imshow("resizeCanvasOutput", dst);
src.delete(); dst.delete();
</textarea>
<p class="err" id="resizeErr"></p>
</div>
<div id="resizeShowcase">
    <div>
        <canvas id="resizeCanvasInput"></canvas>
        <canvas id="resizeCanvasOutput"></canvas>
    </div>
    <input type="file" id="resizeInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function resizeExecuteCode() {
    let resizeText = document.getElementById("resizeTestCode").value;
    try {
        eval(resizeText);
        document.getElementById("resizeErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("resizeErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "resizeCanvasInput");
let resizeInputElement = document.getElementById("resizeInput");
resizeInputElement.addEventListener("change", resizeHandleFiles, false);
function resizeHandleFiles(e) {
    let resizeUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(resizeUrl, "resizeCanvasInput");
}
</script>
</body>
\endhtmlonly

### Translation

Translation is the shifting of object's location. If you know the shift in (x,y) direction, let it
be \f$(t_x,t_y)\f$, you can create the transformation matrix \f$\textbf{M}\f$ as follows:

\f[M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y  \end{bmatrix}\f]

We use the function: **cv.warpAffine (src, dst, M, dsize, flags = cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT, borderValue = new cv.Scalar())**
@param src          input image.
@param dst          output image that has the size dsize and the same type as src.
@param Mat          2 × 3 transformation matrix(cv.CV_64FC1 type).
@param dsize        size of the output image.
@param flags        combination of interpolation methods(see cv.InterpolationFlags) and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( 𝚍𝚜𝚝→𝚜𝚛𝚌 )        
@param borderMode   pixel extrapolation method (see cv.BorderTypes); when borderMode = BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.      
@param borderValue  value used in case of a constant border; by default, it is 0.

rows.

Try it
------

Try this demo using the code above. Canvas elements named warpAffineCanvasInput and warpAffineCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="warpAffineCodeArea">
<h2>Input your code</h2>
<button id="warpAffineTryIt" disabled="true" onclick="warpAffineExecuteCode()">Try it</button><br>
<textarea rows="8" cols="90" id="warpAffineTestCode" spellcheck="false">
let src = cv.imread("warpAffineCanvasInput");
let dst = new cv.Mat();
let M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, 50, 0, 1, 100]);
let dsize = new cv.Size(src.rows, src.cols);
// You can try more different parameters
cv.warpAffine(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("warpAffineCanvasOutput", dst);
src.delete(); dst.delete(); M.delete();
</textarea>
<p class="err" id="warpAffineErr"></p>
</div>
<div id="warpAffineShowcase">
    <div>
        <canvas id="warpAffineCanvasInput"></canvas>
        <canvas id="warpAffineCanvasOutput"></canvas>
    </div>
    <input type="file" id="warpAffineInput" name="file" />
</div>
<script>
function warpAffineExecuteCode() {
    let warpAffineText = document.getElementById("warpAffineTestCode").value;
    try {
        eval(warpAffineText);
        document.getElementById("warpAffineErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("warpAffineErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "warpAffineCanvasInput");
let warpAffineInputElement = document.getElementById("warpAffineInput");
warpAffineInputElement.addEventListener("change", warpAffineHandleFiles, false);
function warpAffineHandleFiles(e) {
    let warpAffineUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(warpAffineUrl, "warpAffineCanvasInput");
}
</script>
</body>
\endhtmlonly

### Rotation

Rotation of an image for an angle \f$\theta\f$ is achieved by the transformation matrix of the form

\f[M = \begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta   \end{bmatrix}\f]

But OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any
location you prefer. Modified transformation matrix is given by

\f[\begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot center.x -  \beta \cdot center.y \\ - \beta &  \alpha &  \beta \cdot center.x + (1- \alpha )  \cdot center.y \end{bmatrix}\f]

where:

\f[\begin{array}{l} \alpha =  scale \cdot \cos \theta , \\ \beta =  scale \cdot \sin \theta \end{array}\f]

We use the function: **cv.getRotationMatrix2D (center, angle, scale)**
@param center    center of the rotation in the source image.
@param angle     rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
@param scale     isotropic scale factor.

Try it
------

Try this demo using the code above. Canvas elements named rotateWarpAffineCanvasInput and rotateWarpAffineCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="rotateWarpAffineCodeArea">
<h2>Input your code</h2>
<button id="rotateWarpAffineTryIt" disabled="true" onclick="rotateWarpAffineExecuteCode()">Try it</button><br>
<textarea rows="9" cols="90" id="rotateWarpAffineTestCode" spellcheck="false">
let src = cv.imread("rotateWarpAffineCanvasInput");
let dst = new cv.Mat();
let dsize = new cv.Size(src.rows, src.cols);
let center = new cv.Point(src.cols / 2, src.rows / 2);
// You can try more different parameters
let M = cv.getRotationMatrix2D(center, 45, 1);
cv.warpAffine(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("rotateWarpAffineCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); 
</textarea>
<p class="err" id="rotateWarpAffineErr"></p>
</div>
<div id="rotateWarpAffineShowcase">
    <div>
        <canvas id="rotateWarpAffineCanvasInput"></canvas>
        <canvas id="rotateWarpAffineCanvasOutput"></canvas>
    </div>
    <input type="file" id="rotateWarpAffineInput" name="file" />
</div>
<script>
function rotateWarpAffineExecuteCode() {
    let rotateWarpAffineText = document.getElementById("rotateWarpAffineTestCode").value;
    try {
        eval(rotateWarpAffineText);
        document.getElementById("rotateWarpAffineErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("rotateWarpAffineErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "rotateWarpAffineCanvasInput");
let rotateWarpAffineInputElement = document.getElementById("rotateWarpAffineInput");
rotateWarpAffineInputElement.addEventListener("change", rotateWarpAffineHandleFiles, false);
function rotateWarpAffineHandleFiles(e) {
    let rotateWarpAffineUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(rotateWarpAffineUrl, "rotateWarpAffineCanvasInput");
}

</script>
</body>
\endhtmlonly

### Affine Transformation

In affine transformation, all parallel lines in the original image will still be parallel in the
output image. To find the transformation matrix, we need three points from input image and their
corresponding locations in output image. Then **cv.getAffineTransform** will create a 2x3 matrix
which is to be passed to **cv.warpAffine**.

We use the function: **cv.getAffineTransform (src, dst)**

@param src    three points([3, 1] size and cv.CV_32FC2 type) from input imag.
@param dst    three corresponding points([3, 1] size and cv.CV_32FC2 type) in output image.

Try it
------

Try this demo using the code above. Canvas elements named getAffineTransformCanvasInput and getAffineTransformCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="getAffineTransformCodeArea">
<h2>Input your code</h2>
<button id="getAffineTransformTryIt" disabled="true" onclick="getAffineTransformExecuteCode()">Try it</button><br>
<textarea rows="13" cols="90" id="getAffineTransformTestCode" spellcheck="false">
let src = cv.imread("getAffineTransformCanvasInput");
let dst = new cv.Mat();
// (data32F[0], data32F[1]) is the first point
// (data32F[2], data32F[3]) is the sescond point
// (data32F[4], data32F[5]) is the third point
let srcTri = cv.matFromArray(3, 1, cv.CV_32FC2, [0, 0, 0, 1, 1, 0]); 
let dstTri = cv.matFromArray(3, 1, cv.CV_32FC2, [0.6, 0.2, 0.1, 1.3, 1.5, 0.3]);
let dsize = new cv.Size(src.rows, src.cols);
let M = cv.getAffineTransform(srcTri, dstTri);
// You can try more different parameters
cv.warpAffine(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("getAffineTransformCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); srcTri.delete(); dstTri.delete(); 
</textarea>
<p class="err" id="getAffineTransformErr"></p>
</div>
<div id="getAffineTransformShowcase">
    <div>
        <canvas id="getAffineTransformCanvasInput"></canvas>
        <canvas id="getAffineTransformCanvasOutput"></canvas>
    </div>
    <input type="file" id="getAffineTransformInput" name="file" />
</div>
<script>
function getAffineTransformExecuteCode() {
    let getAffineTransformText = document.getElementById("getAffineTransformTestCode").value;
    try {
        eval(getAffineTransformText);
        document.getElementById("getAffineTransformErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("getAffineTransformErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "getAffineTransformCanvasInput");
let getAffineTransformInputElement = document.getElementById("getAffineTransformInput");
getAffineTransformInputElement.addEventListener("change", getAffineTransformHandleFiles, false);
function getAffineTransformHandleFiles(e) {
    let getAffineTransformUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(getAffineTransformUrl, "getAffineTransformCanvasInput");
}
</script>
</body>
\endhtmlonly

### Perspective Transformation

For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain straight even after the transformation. To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the function **cv.getPerspectiveTransform**. Then apply **cv.warpPerspective** with this 3x3 transformation matrix.

We use the functions: **cv.warpPerspective (src, dst, M, dsize, flags = cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT, borderValue = new cv.Scalar())**

@param src          input image.
@param dst          output image that has the size dsize and the same type as src.
@param Mat          3 × 3 transformation matrix(cv.CV_64FC1 type).
@param dsize        size of the output image.
@param flags        combination of interpolation methods (cv.INTER_LINEAR or cv.INTER_NEAREST) and the optional flag WARP_INVERSE_MAP, that sets M as the inverse transformation (𝚍𝚜𝚝→𝚜𝚛𝚌).    
@param borderMode   pixel extrapolation method (cv.BORDER_CONSTANT or cv.BORDER_REPLICATE).
@param borderValue  value used in case of a constant border; by default, it is 0.

**cv.getPerspectiveTransform (src, dst)**

@param src          coordinates of quadrangle vertices in the source image.
@param dst          coordinates of the corresponding quadrangle vertices in the destination image.

Try it
------

Try this demo using the code above. Canvas elements named warpPerspectiveCanvasInput and warpPerspectiveCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
</head>
<body>
<div id="warpPerspectiveCodeArea">
<h2>Input your code</h2>
<button id="warpPerspectiveTryIt" disabled="true" onclick="warpPerspectiveExecuteCode()">Try it</button><br>
<textarea rows="15" cols="90" id="warpPerspectiveTestCode" spellcheck="false">
let src = cv.imread("warpPerspectiveCanvasInput");
let dst = new cv.Mat();
let dsize = new cv.Size(src.rows, src.cols);
// (data32F[0], data32F[1]) is the first point
// (data32F[2], data32F[3]) is the sescond point
// (data32F[4], data32F[5]) is the third point
// (data32F[6], data32F[7]) is the fourth point
let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [56, 65, 368, 52, 28, 387, 389, 390]); 
let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, 300, 0, 0, 300, 300, 300]);
let M = cv.getPerspectiveTransform(srcTri, dstTri);
// You can try more different parameters
cv.warpPerspective(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("warpPerspectiveCanvasOutput", dst);
src.delete(); dst.delete(); M.delete(); srcTri.delete(); dstTri.delete();
</textarea>
<p class="err" id="warpPerspectiveErr"></p>
</div>
<div id="warpPerspectiveShowcase">
    <div>
        <canvas id="warpPerspectiveCanvasInput"></canvas>
        <canvas id="warpPerspectiveCanvasOutput"></canvas>
    </div>
    <input type="file" id="warpPerspectiveInput" name="file" />
</div>
<script>
function warpPerspectiveExecuteCode() {
    let warpPerspectiveText = document.getElementById("warpPerspectiveTestCode").value;
    try {
        eval(warpPerspectiveText);
        document.getElementById("warpPerspectiveErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("warpPerspectiveErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "warpPerspectiveCanvasInput");
let warpPerspectiveInputElement = document.getElementById("warpPerspectiveInput");
warpPerspectiveInputElement.addEventListener("change", warpPerspectiveHandleFiles, false);
function warpPerspectiveHandleFiles(e) {
    let warpPerspectiveUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(warpPerspectiveUrl, "warpPerspectiveCanvasInput");

}
function onReady() {
    document.getElementById("resizeTryIt").disabled = false;
    document.getElementById("warpAffineTryIt").disabled = false;
    document.getElementById("rotateWarpAffineTryIt").disabled = false;
    document.getElementById("getAffineTransformTryIt").disabled = false;
    document.getElementById("warpPerspectiveTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly