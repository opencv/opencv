Using OpenCV.js {#tutorial_js_usage}
===============================

Steps
-----

In this tutorial, you will see how to include and start to use `opencv.js` inside a web page.

### Create a web page uploading image

First, let's create a simple web page which is able to upload an image.

@code{.js}
<!DOCTYPE html>
<html>
<body>

<h1>Hello OpenCV.js</h1>

<input type="file" id="fileInput" accept="image/gif, image/jpeg, image/png"/>
<div>
    <img id="srcImage"></img>
</div>

<script>
let imgElement = document.getElementById("srcImage")
let inputElement = document.getElementById("fileInput");
inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);
</script>

</body>
</html>
@endcode

You can copy above content and save to a local index.html file. To run it, just open it by the web browser.

@note It is a better practice that is hosting the index.html by a local web server.

### Include OpenCV.js

You need to set the URL of `opencv.js` to `src` attribute of of \<script\> tag.

@note For this tutorial, we host `opencv.js` at same folder of index.html.

Example for synchronous loading.
@code{.js}
<script src="opencv.js"></script>
@endcode

You may want to load `opencv.js` asynchronously by `async` attribute in \<script\> tag. To be notified by `opencv.js` is ready, you can
register a callback to `onload` attribute.

Example for asynchronous loading
@code{.js}
<script async src="opencv.js" onload="opencvIsReady();"></script>
@endcode

### Use OpenCV.js

Once `opencv.js` is ready, you can access OpenCV objects and functions through `cv` object.

For example, you can create a cv.Mat from an image by cv.imread.

@note As image loading is asynchronous, so you need to put cv.Mat creation inside the `onload` callback.

@code{.js}
imgElement.onload = function() {
  let mat = cv.imread(imgElement);
}
@endcode

Many OpenCV functions can be used to process cv.Mat. You can refer to other tutorials for details.

In this tutorial, we just show a cv.Mat on screen. To show a cv.Mat, you need a canvas element.

@code{.js}
<canvas id="outputCanvas"></canvas>
@endcode

You can use cv.imshow to show cv.Mat into the canvas.
@code{.js}
cv.imshow(mat, "outputCanvas");
@endcode

Putting them all together, the final index.html is shown below.

@code{.js}
<!DOCTYPE html>
<html>
<body>

<h1>Hello OpenCV.js</h1>

<p id="status">OpenCV.js is loading...</p>

<input type="file" id="fileInput" accept="image/gif, image/jpeg, image/png"/>
<div>
    <img id="srcImage"></img>
    <canvas id="outputCanvas"></canvas>
</div>

<script>
let imgElement = document.getElementById("srcImage")
let inputElement = document.getElementById("fileInput");

inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

imgElement.onload = function() {
  let mat = cv.imread(imgElement);
  cv.imshow("outputCanvas", mat);
}

function opencvIsReady() {
  document.getElementById("status").innerHTML = "OpenCV.js is ready.";
}
</script>

<script async src="opencv.js" onload="opencvIsReady();"></script>

</body>
</html>
@endcode

An embedded version is shown below. You can try it. After you upload an image, the web page will show two images. The first one is an \<img\> as cv.Mat input. The second one is a \<canvas\> as cv.Mat output.
- - -

\htmlonly
<!DOCTYPE html>
<html>
<body>

<h1>Hello OpenCV.js</h1>

<p id="status">OpenCV.js is loading.</p>

<input type="file" id="fileInput" accept="image/gif, image/jpeg, image/png"/>
<div>
    <img id="srcImage"></img>
    <canvas id="outputCanvas"></canvas>
<div>

<script>

function imread(image) {
    var canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, image.width, image.height);
    var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return cv.matFromArray(imgData, cv.CV_8UC4);
}

let imgElement = document.getElementById("srcImage")
let inputElement = document.getElementById("fileInput");
inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

imgElement.onload = function() {
  let mat = imread(imgElement);
  cv.imshow("outputCanvas", mat);
}

function opencvIsReady() {
  document.getElementById("status").innerHTML = "OpenCV.js is ready.";
}
</script>

<script async src="opencv.js" onload="opencvIsReady();"></script>

</body>
</html>
\endhtmlonly
- - -