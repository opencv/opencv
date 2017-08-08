Basic Operations on Images {#tutorial_js_basic_ops}
==========================

Goal
----

Learn to:

-   Access image properties
-   How to construct Mat
-   How to copy Mat
-   How to convert the type of Mat
-   Access pixel values and modify them
-   Setting Region of Interest (ROI)
-   Splitting and Merging images

Accessing Image Properties
--------------------------

Image properties include number of rows, columns and size, depth, channels, type of image data.

@code{.js}
let src = cv.imread("canvasInput");
let rows =  src.cols;
let columns = src.rows;
let size = src.size;
let depth = src.depth();
let channels = src.channels();
let type = src.type();
@endcode

@note src.type() is very important while debugging because a large number of errors in OpenCV-Python
code is caused by invalid data type.

How to construct Mat
------------------------------------

There are 4 constructors:

@code{.js}
let mat = new cv.Mat(size, type);                        // 2 parameters
let mat = new cv.Mat(rows, cols, type);                  // 3 parameters
let mat = new cv.Mat(rows, cols, type, new cv.Scalar()); // 4 parameters
let mat = matFromArray(rows, cols, type, array);
@endcode

There are 3 static functions:

@code{.js}
let mat = cv.Mat.zeros(rows, cols, type);           
let mat = cv.Mat.ones(rows, cols, type); 
let mat = cv.Mat.eye(rows, cols, type);
@endcode

**warning**

Don't forget to delete cv.Mat(cv.MatVector) when you don't want to use it any more.

How to copy Mat
------------------------------------

@code{.js}
dst = src.clone();
src.copyTo(dst, mask); // only entries indicated in the arry mask are copied    
@endcode
  
How to convert the type of Mat
------------------------------------

@code{.js}
src.convertTo(dst, rtype); // rtype desired dst matrix type
@endcode

Accessing and Modifying pixel values
------------------------------------

Firstly, you should know the following type relationship: 

-   data    \f$\leftrightarrow\f$ uchar
-   data8S  \f$\leftrightarrow\f$ char
-   data16U \f$\leftrightarrow\f$ ushort
-   data16S \f$\leftrightarrow\f$ short
-   data32S \f$\leftrightarrow\f$ int
-   data32F \f$\leftrightarrow\f$ float
-   data64F \f$\leftrightarrow\f$ double

We use different ways to get the first channel of a pixel from a CV_8UC4 Mat. The coordinate of this pixel is (x, y).

**1. data**

@code{.js}
let x = 3, y = 4;
let src = cv.imread("canvasInput");
let pixel = src.data[y * src.cols * src.channels() + x * src.channels()];
@endcode

**2. at**

@code{.js}
let x = 3, y = 4;
let src = cv.imread("canvasInput");
let pixel = src.ucharAt(y, x * src.channels());
@endcode

**3. ptr**

@code{.js}
let x = 3, y = 4;
let src = cv.imread("canvasInput");
let pixel = src.ucharPtr(y, x)[0];
@endcode

Image ROI
---------

Sometimes, you will have to play with certain region of images. For eye detection in images, first
face detection is done all over the image and when face is obtained, we select the face region alone
and search for eyes inside it instead of searching whole image. It improves accuracy (because eyes
are always on faces) and performance (because we search for a small area)

Try it
------

Here is a demo. Canvas elements named roiCanvasInput and roiCanvasOutput have been prepared. Choose an image and 
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
<div id="roiCodeArea">
<h2>Input your code</h2>
<button id="roiTryIt" disabled="true" onclick="roiExecuteCode()">Try it</button><br>
<textarea rows="7" cols="80" id="roiTestCode" spellcheck="false">
let src = cv.imread("roiCanvasInput");
let dst = new cv.Mat();
// You can try more different parameters
let rect = new cv.Rect(100, 100, 200, 200);
dst = src.getRoiRect(rect); 
cv.imshow("roiCanvasOutput", dst);
src.delete(); dst.delete(); 
</textarea>
<p class="err" id="roiErr"></p>
</div>
<div id="roiShowcase">
    <div>
        <canvas id="roiCanvasInput"></canvas>
        <canvas id="roiCanvasOutput"></canvas>
    </div>
    <input type="file" id="roiInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function roiExecuteCode() {
    let roiText = document.getElementById("roiTestCode").value;
    try {
        eval(roiText);
        document.getElementById("roiErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("roiErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "roiCanvasInput");
let roiInputElement = document.getElementById("roiInput");
roiInputElement.addEventListener("change", roiHandleFiles, false);
function roiHandleFiles(e) {
    let roiUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(roiUrl, "roiCanvasInput");
}
</script>
</body>
\endhtmlonly


Splitting and Merging Image Channels
------------------------------------

Sometimes you will need to work separately on R,G,B channels of image. Then you need to split the
RGB images to single planes. Or another time, you may need to join these individual channels to RGB
image. 

@code{.js}
let src = cv.imread("canvasInput");
let rgbPlanes = new cv.MatVector();
cv.split(src, rgbPlanes);
cv.merge(rgbPlanes, src);
src.delete(); rgbPlanes.delete();
@endcode


Making Borders for Images (Padding)
-----------------------------------

If you want to create a border around the image, something like a photo frame, you can use
**cv.copyMakeBorder()** function. But it has more applications for convolution operation, zero
padding etc. This function takes following arguments:

-   **src** - input image
-   **top**, **bottom**, **left**, **right** - border width in number of pixels in corresponding
    directions

-   **borderType** - Flag defining what kind of border to be added. It can be following types:
    -   **cv.BORDER_CONSTANT** - Adds a constant colored border. The value should be given
            as next argument.
        -   **cv.BORDER_REFLECT** - Border will be mirror reflection of the border elements,
            like this : *fedcba|abcdefgh|hgfedcb*
        -   **cv.BORDER_REFLECT_101** or **cv.BORDER_DEFAULT** - Same as above, but with a
            slight change, like this : *gfedcb|abcdefgh|gfedcba*
        -   **cv.BORDER_REPLICATE** - Last element is replicated throughout, like this:
            *aaaaaa|abcdefgh|hhhhhhh*
        -   **cv.BORDER_WRAP** - Can't explain, it will look like this :
            *cdefgh|abcdefgh|abcdefg*

-   **value** - Color of border if border type is cv.BORDER_CONSTANT

Try it
------

Here is a demo. Canvas elements named copyMakeBorderCanvasInput and copyMakeBorderCanvasOutput have been prepared. Choose an image and 
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
<div id="copyMakeBorderCodeArea">
<h2>Input your code</h2>
<button id="copyMakeBorderTryIt" disabled="true" onclick="copyMakeBorderExecuteCode()">Try it</button><br>
<textarea rows="7" cols="80" id="copyMakeBorderTestCode" spellcheck="false">
let src = cv.imread("copyMakeBorderCanvasInput");
let dst = new cv.Mat();
// You can try more different parameters
let s = new cv.Scalar(255, 0, 0, 255);
cv.copyMakeBorder(src, dst, 10, 10, 10, 10, cv.BORDER_CONSTANT, s);
cv.imshow("copyMakeBorderCanvasOutput", dst);
src.delete(); dst.delete();
</textarea>
<p class="err" id="copyMakeBorderErr"></p>
</div>
<div id="copyMakeBorderShowcase">
    <div>
        <canvas id="copyMakeBorderCanvasInput"></canvas>
        <canvas id="copyMakeBorderCanvasOutput"></canvas>
    </div>
    <input type="file" id="copyMakeBorderInput" name="file" />
</div>
<script>
function copyMakeBorderExecuteCode() {
    let copyMakeBorderText = document.getElementById("copyMakeBorderTestCode").value;
    try {
        eval(copyMakeBorderText);
        document.getElementById("copyMakeBorderErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("copyMakeBorderErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "copyMakeBorderCanvasInput");
let copyMakeBorderInputElement = document.getElementById("copyMakeBorderInput");
copyMakeBorderInputElement.addEventListener("change", copyMakeBorderHandleFiles, false);
function copyMakeBorderHandleFiles(e) {
    let copyMakeBorderUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(copyMakeBorderUrl, "copyMakeBorderCanvasInput");
}
function onReady() {
    document.getElementById("copyMakeBorderTryIt").disabled = false;
    document.getElementById("roiTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly