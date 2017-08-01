Using OpenCV-JavaScript {#tutorial_js_usage}
===============================

@note We assume that you have successfully installed or built `opencv.js` in your workstation.

Steps
-----

### Create a web page uploading and displaying image

First, let's create a simple web page which is able to upload and display an image.
@code{.js}
<!DOCTYPE html>
<html>
<body>

<h1>My First OpenCV.js Web Page</h1>

<div>
    <img id='srcImage'></img>
</div>
<input type='file' id='fileInput' accept='image/gif, image/jpeg, image/png'/>

<script>
let imgElement = document.getElementById('srcImage')
let inputElement = document.getElementById('fileInput');
inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);
</script>
</body>
</html>
@endcode

You can copy above content and save to a local index.html file. To run it, please open it by the web browser.

@note It is a better practice that is hosting the index.html by a local web server.

An embedded version is shown below. You can upload an image from filesystem to display on the web page.

- - -

\htmlonly
<!DOCTYPE html>
<html>
<body>

<h1>My First OpenCV.js Web Page</h1>

<div>
    <img id='srcImage'></img>
</div>
<input type='file' id='fileInput' accept='image/gif, image/jpeg, image/png'/>

<script>
let imgElement = document.getElementById('srcImage')
let inputElement = document.getElementById('fileInput');
inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);
</script>
</body>
</html>
\endhtmlonly

- - -

### Add `opencv.js` and use cv.Mat

Now you need to load the `opencv.js` library by \<script\> tag.

Example for synchronous loading.
@code{.js}
<script src="opencv.js"></script>
@endcode

@note `opencv.js` should be in the same folder of `index.html`.

You may want to load `opencv.js` asynchronously by `async` attribute in \<script\> tag. To be notified by `opencv.js` is ready, you can
register a callback to `Module['onRuntimeInitialized']`. Please refer to [Emscripten FAQ](https://kripken.github.io/emscripten-site/docs/getting_started/FAQ.html#how-can-i-tell-when-the-page-is-fully-loaded-and-it-is-safe-to-call-compiled-functions) for details.

Example for asynchronous loading
@code{.js}
<script>
var Module = {
  'onRuntimeInitialized': () => {console.log('opencv.js is ready.');}
}
</script>
<script src="opencv.js" async></script>
@endcode

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
<canvas id='outputCanvas'></canvas>
@endcode

You can use cv.imshow to show cv.Mat into the canvas.
@code{.js}
cv.imshow(mat, 'outputCanvas');
@endcode

Putting them all together, the final index.html is shown below.

@code{.js}
<!DOCTYPE html>
<html>
<body>

<h1>My First OpenCV.js Web Page</h1>

<p id='status'>OpenCV.js is loading.</p>

<div>
    <img id='srcImage'></img>
</div>
<input type='file' id='fileInput' accept='image/gif, image/jpeg, image/png'/>
<div>
    <canvas id='outputCanvas'></canvas>
<div>

<script>
var Module = {
  'onRuntimeInitialized': () => {document.getElementById('status').innerHTML = 'OpenCV.js is ready';}
}
</script>
<script src="opencv.js" async></script>
<script>
let imgElement = document.getElementById('srcImage')
let inputElement = document.getElementById('fileInput');
inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);
imgElement.onload = function() {
  let mat = cv.imread(imgElement);
  cv.imshow('outputCanvas', mat);
}
</script>
</body>
</html>
@endcode

An embedded version is shown below. You can try it.
- - -

\htmlonly
<!DOCTYPE html>
<html>
<body>

<h1>My First OpenCV.js Web Page</h1>

<p id='status'>OpenCV.js is loading.</p>

<div>
    <img id='srcImage1'></img>
</div>
<input type='file' id='fileInput1' accept='image/gif, image/jpeg, image/png'/>
<div>
    <canvas id='outputCanvas'></canvas>
<div>

<script>
var Module = {
  'onRuntimeInitialized': () => {document.getElementById('status').innerHTML = 'OpenCV.js is ready';}
}
</script>
<script src="opencv.js" async></script>
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

let imgElement1 = document.getElementById('srcImage1')
let inputElement1 = document.getElementById('fileInput1');
inputElement1.addEventListener("change", (e) => {
  imgElement1.src = URL.createObjectURL(e.target.files[0]);
}, false);
imgElement1.onload = function() {
  let mat = imread(imgElement1);
  cv.imshow('outputCanvas', mat);
}
</script>
</body>
</html>
\endhtmlonly
- - -