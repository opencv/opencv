Geometric Transformations of Images {#tutorial_js_geometric_transformations}
===================================

Goals
-----

-   Learn to apply different geometric transformation to images like translation, rotation, affine
    transformation etc.
-   You will see these functions: **resize**, **cv.warpAffine**, **cv.getAffineTransform** and **cv.warpPerspective** 

Transformations
---------------


### Scaling

Scaling is just resizing of the image. OpenCV comes with a function **cv2.resize()** for this
purpose. The size of the image can be specified manually, or you can specify the scaling factor.
Different interpolation methods are used. Preferable interpolation methods are **cv.InterpolationFlags.INTER_AREA.value**
for shrinking and **cv.InterpolationFlags.INTER_CUBIC.value** (slow) & **cv.InterpolationFlags.INTER_LINEAR.value** for zooming. 

We use the function: **cv.resize(src, dst, dsize, fx, fy, interpolation)**
@param src    input image
@param dst     output image; it has the size dsize (when it is non-zero) or the size computed from src.size(), fx, and fy; the type of dst is the same as of src. 
@param dsize  output image size; if it equals zero, it is computed as: 		
    			 \f[𝚍𝚜𝚒𝚣𝚎 = 𝚂𝚒𝚣𝚎(𝚛𝚘𝚞𝚗𝚍(𝚏𝚡*𝚜𝚛𝚌.𝚌𝚘𝚕𝚜), 𝚛𝚘𝚞𝚗𝚍(𝚏𝚢*𝚜𝚛𝚌.𝚛𝚘𝚠𝚜))\f]
    			 Either dsize or both fx and fy must be non-zero. 
@param fx     scale factor along the horizontal axis; when it equals 0, it is computed as  \f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜\f]		 
    			 
@param fy     scale factor along the vertical axis; when it equals 0, it is computed as \f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜\f] 
@param interpolation    interpolation method

Try it
------

Here is a demo. Canvas elements named resizeCanvas1 and resizeCanvas2 have been prepared. Choose an image and
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
<div id="resizeCodeArea">
<h2>Input your code</h2>
<button id="resizeTryIt" disabled="true" onclick="resizeExecuteCode()">Try it</button><br>
<textarea rows="8" cols="80" id="resizeTestCode" spellcheck="false">
var src = cv.imread("resizeCanvas1");
var dst = new cv.Mat();
// You can try more different conversion
cv.resize(src, dst, [600,600], 0, 0, cv.InterpolationFlags.INTER_LINEAR.value);
cv.imshow("resizeCanvas2", dst);
src.delete();
dst.delete();
</textarea>
</div>
<div id="resizeShowcase">
    <div>
        <canvas id="resizeCanvas1"></canvas>
        <canvas id="resizeCanvas2"></canvas>
    </div>
    <input type="file" id="resizeInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function resizeExecuteCode() {
    var resizeText = document.getElementById("resizeTestCode").value;
    eval(resizeText);
}

loadImageToCanvas("lena.jpg", "resizeCanvas1");
var resizeInputElement = document.getElementById("resizeInput");
resizeInputElement.addEventListener("change", resizeHandleFiles, false);
function resizeHandleFiles(e) {
    var resizeUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(resizeUrl, "resizeCanvas1");
}
</script>
</body>
\endhtmlonly

### Translation

Translation is the shifting of object's location. If you know the shift in (x,y) direction, let it
be \f$(t_x,t_y)\f$, you can create the transformation matrix \f$\textbf{M}\f$ as follows:

\f[M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y  \end{bmatrix}\f]

We use the function: **cv.warpAffine(src, dst, M, dsize, flags, borderMode, borderValue)**
@param src    input image.
@param dst    output image that has the size dsize and the same type as src.
@param Mat    2×3transformation matrix.
@param dsize  size of the output image.
@param flags  combination of interpolation methods and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( 𝚍𝚜𝚝→𝚜𝚛𝚌 )		 
@param borderMode	pixel extrapolation method; when borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.		 
@param borderValue   value used in case of a constant border  

**warning**

Third argument of the **cv.warpAffine()** function is the size of the output image, which should
be in the form of (width, height). Remember width = number of columns, and height = number of
rows.

Try it
------

Here is a demo. Canvas elements named warpAffineCanvas1 and warpAffineCanvas2 have been prepared. Choose an image and
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
<div id="warpAffineCodeArea">
<h2>Input your code</h2>
<button id="warpAffineTryIt" disabled="true" onclick="warpAffineExecuteCode()">Try it</button><br>
<textarea rows="10" cols="80" id="warpAffineTestCode" spellcheck="false">
var src = cv.imread("warpAffineCanvas1");
var dst = new cv.Mat();
var M = new cv.Mat([2,3], cv.CV_64FC1);
M.data64f()[0]=1; M.data64f()[1]=0; M.data64f()[2]=50;
M.data64f()[3]=0; M.data64f()[4]=1; M.data64f()[5]=100;
// You can try more different conversion
cv.warpAffine(src, dst, M, [src.cols,src.rows], cv.InterpolationFlags.INTER_LINEAR.value, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("warpAffineCanvas2", dst);
src.delete(); dst.delete(); M.delete();
</textarea>
</div>
<div id="warpAffineShowcase">
    <div>
        <canvas id="warpAffineCanvas1"></canvas>
        <canvas id="warpAffineCanvas2"></canvas>
    </div>
    <input type="file" id="warpAffineInput" name="file" />
</div>
<script>
function warpAffineExecuteCode() {
    var warpAffineText = document.getElementById("warpAffineTestCode").value;
    eval(warpAffineText);
}

loadImageToCanvas("lena.jpg", "warpAffineCanvas1");
var warpAffineInputElement = document.getElementById("warpAffineInput");
warpAffineInputElement.addEventListener("change", warpAffineHandleFiles, false);
function warpAffineHandleFiles(e) {
    var warpAffineUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(warpAffineUrl, "warpAffineCanvas1");
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

We use the function: **cv.warpAffine(src, dst, M, dsize, flags, borderMode, borderValue)**

Try it
------

Here is a demo. Canvas elements named rotateWarpAffineCanvas1 and rotateWarpAffineCanvas2 have been prepared. Choose an image and
click `Try it` to see the result. And you can change the code in the textbox to investigate more.

@note cv.getRotationMatrix2D() should be in the white list to simplify the operation.

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
<div id="rotateWarpAffineCodeArea">
<h2>Input your code</h2>
<button id="rotateWarpAffineTryIt" disabled="true" onclick="rotateWarpAffineExecuteCode()">Try it</button><br>
<textarea rows="18" cols="80" id="rotateWarpAffineTestCode" spellcheck="false">
var src = cv.imread("rotateWarpAffineCanvas1");
var dst = new cv.Mat(src.cols, src.rows, src.type());
var M = new cv.Mat([2,3], cv.CV_64FC1);
var degree = 45;
var angle = degree * Math.PI / 180.;
var alpha = Math.cos(angle);
var beta = Math.sin(angle);
M.data64f()[0] = alpha,
M.data64f()[1] = beta,
M.data64f()[2] = (1 - alpha) * src.cols / 2 - beta * src.rows / 2,
M.data64f()[3] = -beta,
M.data64f()[4] = alpha,
M.data64f()[5] = beta * src.cols / 2 + (1 - alpha) * src.rows / 2;
// You can try more different conversion
cv.warpAffine(src, dst, M, [src.cols,src.rows], cv.InterpolationFlags.INTER_LINEAR.value, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("rotateWarpAffineCanvas2", dst);
src.delete(); dst.delete(); M.delete();
</textarea>
</div>
<div id="rotateWarpAffineShowcase">
    <div>
        <canvas id="rotateWarpAffineCanvas1"></canvas>
        <canvas id="rotateWarpAffineCanvas2"></canvas>
    </div>
    <input type="file" id="rotateWarpAffineInput" name="file" />
</div>
<script>
function rotateWarpAffineExecuteCode() {
    var rotateWarpAffineText = document.getElementById("rotateWarpAffineTestCode").value;
    eval(rotateWarpAffineText);
}

loadImageToCanvas("lena.jpg", "rotateWarpAffineCanvas1");
var rotateWarpAffineInputElement = document.getElementById("rotateWarpAffineInput");
rotateWarpAffineInputElement.addEventListener("change", rotateWarpAffineHandleFiles, false);
function rotateWarpAffineHandleFiles(e) {
    var rotateWarpAffineUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(rotateWarpAffineUrl, "rotateWarpAffineCanvas1");
}

</script>
</body>
\endhtmlonly

### Affine Transformation

In affine transformation, all parallel lines in the original image will still be parallel in the
output image. To find the transformation matrix, we need three points from input image and their
corresponding locations in output image. Then **cv.getAffineTransform** will create a 2x3 matrix
which is to be passed to **cv.warpAffine**.

We use the function: **cv.getAffineTransform(src, dst)** and **cv.warpAffine(src, dst, M, dsize, flags, borderMode, borderValue)**

@param src    three points from input imag
@param dst    three corresponding points in output image

Try it
------

Here is a demo. Canvas elements named getAffineTransformCanvas1 and getAffineTransformCanvas2 have been prepared. Choose an image and
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
<div id="getAffineTransformCodeArea">
<h2>Input your code</h2>
<button id="getAffineTransformTryIt" disabled="true" onclick="getAffineTransformExecuteCode()">Try it</button><br>
<textarea rows="18" cols="80" id="getAffineTransformTestCode" spellcheck="false">
var src = cv.imread("getAffineTransformCanvas1");
var dst = new cv.Mat(src.cols, src.rows, src.type());
var srcTri = new cv.Mat(3, 2, cv.CV_32F); 
var dstTri = new cv.Mat(3, 2, cv.CV_32F);
srcTri.data32f()[0] = 0; dstTri.data32f()[0] = 0.6;//(data32f()[0],data32f()[1]) is the first point
srcTri.data32f()[1] = 0; dstTri.data32f()[1] = 0.2;
srcTri.data32f()[2] = 0; dstTri.data32f()[2] = 0.1;//(data32f()[0],data32f()[1]) is the sescond point
srcTri.data32f()[3] = 1; dstTri.data32f()[3] = 1.3;
srcTri.data32f()[4] = 1; dstTri.data32f()[4] = 1.5;//(data32f()[0],data32f()[1]) is the third point
srcTri.data32f()[5] = 0; dstTri.data32f()[5] = 0.3;
var M = new cv.Mat([2,3], cv.CV_64FC1);
M = cv.getAffineTransform(srcTri, dstTri);
// You can try more different conversion
cv.warpAffine(src, dst, M, [src.cols,src.rows], cv.InterpolationFlags.INTER_LINEAR.value, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("getAffineTransformCanvas2", dst);
src.delete(); dst.delete(); M.delete(); srcTri.delete(); dstTri.delete();
</textarea>
</div>
<div id="getAffineTransformShowcase">
    <div>
        <canvas id="getAffineTransformCanvas1"></canvas>
        <canvas id="getAffineTransformCanvas2"></canvas>
    </div>
    <input type="file" id="getAffineTransformInput" name="file" />
</div>
<script>
function getAffineTransformExecuteCode() {
    var getAffineTransformText = document.getElementById("getAffineTransformTestCode").value;
    eval(getAffineTransformText);
}

loadImageToCanvas("lena.jpg", "getAffineTransformCanvas1");
var getAffineTransformInputElement = document.getElementById("getAffineTransformInput");
getAffineTransformInputElement.addEventListener("change", getAffineTransformHandleFiles, false);
function getAffineTransformHandleFiles(e) {
    var getAffineTransformUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(getAffineTransformUrl, "getAffineTransformCanvas1");
}
</script>
</body>
\endhtmlonly

### Perspective Transformation

For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain
straight even after the transformation. Apply **cv.warpPerspective** with this 3x3 transformation
matrix.

We use the function: **cv.warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue)**

The parameters of cv.warpPerspective() are similar to the parameters of cv.warpAffine().

Try it
------

Here is a demo. Canvas elements named warpPerspectiveCanvas1 and warpPerspectiveCanvas2 have been prepared. Choose an image and
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
<div id="warpPerspectiveCodeArea">
<h2>Input your code</h2>
<button id="warpPerspectiveTryIt" disabled="true" onclick="warpPerspectiveExecuteCode()">Try it</button><br>
<textarea rows="11" cols="80" id="warpPerspectiveTestCode" spellcheck="false">
var src = cv.imread("warpPerspectiveCanvas1");
var dst = new cv.Mat();
var M = new cv.Mat([3,3], cv.CV_64FC1);
M.data64f()[0]=1;M.data64f()[1]=0.1;M.data64f()[2]=-65;
M.data64f()[3]=0;M.data64f()[4]=1.1;M.data64f()[5]=-75;
M.data64f()[6]=0;M.data64f()[7]=0;  M.data64f()[8]=1;
// You can try more different conversion
cv.warpPerspective(src, dst, M, [src.cols,src.rows], cv.InterpolationFlags.INTER_LINEAR.value, cv.BORDER_CONSTANT, new cv.Scalar());
cv.imshow("warpPerspectiveCanvas2", dst);
src.delete(); dst.delete(); M.delete();
</textarea>
</div>
<div id="warpPerspectiveShowcase">
    <div>
        <canvas id="warpPerspectiveCanvas1"></canvas>
        <canvas id="warpPerspectiveCanvas2"></canvas>
    </div>
    <input type="file" id="warpPerspectiveInput" name="file" />
</div>
<script>
function warpPerspectiveExecuteCode() {
    var warpPerspectiveText = document.getElementById("warpPerspectiveTestCode").value;
    eval(warpPerspectiveText);
}

loadImageToCanvas("lena.jpg", "warpPerspectiveCanvas1");
var warpPerspectiveInputElement = document.getElementById("warpPerspectiveInput");
warpPerspectiveInputElement.addEventListener("change", warpPerspectiveHandleFiles, false);
function warpPerspectiveHandleFiles(e) {
    var warpPerspectiveUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(warpPerspectiveUrl, "warpPerspectiveCanvas1");

}
document.getElementById("opencvjs").onload = function() {
    document.getElementById("resizeTryIt").disabled = false;
    document.getElementById("warpAffineTryIt").disabled = false;
    document.getElementById("rotateWarpAffineTryIt").disabled = false;
    document.getElementById("getAffineTransformTryIt").disabled = false;
    document.getElementById("warpPerspectiveTryIt").disabled = false;
};
</script>
</body>
\endhtmlonly

@note cv.getPerspectiveTransform() should be in the white list to find the transformation matrix from 4 points on the input image and corresponding points on the output image.
