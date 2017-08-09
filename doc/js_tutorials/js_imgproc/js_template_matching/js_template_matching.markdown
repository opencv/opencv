Template Matching {#tutorial_js_template_matching}
=================

Goals
-----

-   To find objects in an image using Template Matching
-   You will learn these functions : **cv.matchTemplate()**, **cv.minMaxLoc()**

Theory
------

Template Matching is a method for searching and finding the location of a template image in a larger
image. OpenCV comes with a function **cv.matchTemplate()** for this purpose. It simply slides the
template image over the input image (as in 2D convolution) and compares the template and patch of
input image under the template image. Several comparison methods are implemented in OpenCV. (You can
check docs for more details). It returns a grayscale image, where each pixel denotes how much does
the neighbourhood of that pixel match with template.

If input image is of size (WxH) and template image is of size (wxh), output image will have a size
of (W-w+1, H-h+1). Once you got the result, you can use **cv.minMaxLoc()** function to find where
is the maximum/minimum value. Take it as the top-left corner of rectangle and take (w,h) as width
and height of the rectangle. That rectangle is your region of template.

@note If you are using cv.TM_SQDIFF as comparison method, minimum value gives the best match.

Template Matching in OpenCV
---------------------------

We use the function: **cv.matchTemplate (image, templ, result, method, mask = new cv.Mat())** 

@param image      image where the search is running. It must be 8-bit or 32-bit floating-point.
@param templ      searched template. It must be not greater than the source image and have the same data type.
@param result     map of comparison results. It must be single-channel 32-bit floating-point.
@param method     parameter specifying the comparison method(see cv.TemplateMatchModes).
@param mask       mask of searched template. It must have the same datatype and size with templ. It is not set by default.

Try it
------

Try this demo using the code above. Canvas elements named imageCanvasInput, templateCanvasInput and matchTemplateCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

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
<div id="matchTemplateCodeArea">
<h2>Input your code</h2>
<button id="matchTemplateTryIt" disabled="true" onclick="matchTemplateExecuteCode()">Try it</button><br>
<textarea rows="13" cols="80" id="matchTemplateTestCode" spellcheck="false">
let src = cv.imread("imageCanvasInput");
let templ = cv.imread("templateCanvasInput");
let dst = new cv.Mat();
let mask = new cv.Mat();
cv.matchTemplate(src, templ, dst, cv.TM_CCOEFF, mask);
let result = cv.minMaxLoc(dst, mask);
let max = result.maxVal;
let maxPoint = result.maxLoc;
let color = new cv.Scalar(255, 0, 0, 255);
let point = new cv.Point(maxPoint.x + templ.cols, maxPoint.y + templ.rows)
cv.rectangle(src, maxPoint, point, color, 2, cv.LINE_8, 0);
cv.imshow("matchTemplateCanvasOutput", src);
src.delete(); dst.delete(); mask.delete()
</textarea>
<p class="err" id="matchTemplateErr"></p>
</div>
<div id="matchTemplateShowcase">
    <div>
        <p>Template</p>
        <canvas id="templateCanvasInput"></canvas>
        <input type="file" id="templateInput" name="file" />
    </div>
    <div>
        <p>Image</p>
        <canvas id="imageCanvasInput"></canvas>
        <input type="file" id="imageInput" name="file" />
    </div>
    <div>
        <p>Result</p>
        <canvas id="matchTemplateCanvasOutput"></canvas>
    </div>
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function matchTemplateExecuteCode() {
    let matchTemplateText = document.getElementById("matchTemplateTestCode").value;
    try {
        eval(matchTemplateText);
        document.getElementById("matchTemplateErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("matchTemplateErr").innerHTML = err;
    }
}

loadImageToCanvas("lenaFace.png", "templateCanvasInput");
loadImageToCanvas("lena.jpg", "imageCanvasInput");

let templateInputElement = document.getElementById("templateInput");
templateInputElement.addEventListener("change", templateHandleFiles, false);
function templateHandleFiles(e) {
    let templateUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(templateUrl, "templateCanvasInput");
}

let imageInputElement = document.getElementById("imageInput");
imageInputElement.addEventListener("change", imageHandleFiles, false);
function imageHandleFiles(e) {
    let imageUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(imageUrl, "imageCanvasInput");
}

function onReady() {
    document.getElementById("matchTemplateTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly