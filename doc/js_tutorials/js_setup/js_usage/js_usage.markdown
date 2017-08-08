Using OpenCV.js {#tutorial_js_usage}
===============================

Steps
-----

In this tutorial, you will learn how to include and start to use `opencv.js` inside a web page.

### Create a web page

First, let's create a simple web page that is able to upload an image.

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

To run this web page, copy the content above and save to a local index.html file. To run it, open it using your web browser.

@note It is a better practice to use a local web server to host the index.html.

### Include OpenCV.js

Set the URL of `opencv.js` to `src` attribute of \<script\> tag.

@note For this tutorial, we host `opencv.js` at same folder as index.html.

Example for synchronous loading:
@code{.js}
<script src="opencv.js"></script>
@endcode

You may want to load `opencv.js` asynchronously by `async` attribute in \<script\> tag. To be notified when `opencv.js` is ready, you can
register a callback to `onload` attribute.

Example for asynchronous loading
@code{.js}
<script async src="opencv.js" onload="opencvIsReady();"></script>
@endcode

### Use OpenCV.js

Once `opencv.js` is ready, you can access OpenCV objects and functions through `cv` object.

For example, you can create a cv.Mat from an image by cv.imread.

@note Because image loading is asynchronous, you need to put cv.Mat creation inside the `onload` callback.

@code{.js}
imgElement.onload = function() {
  let mat = cv.imread(imgElement);
}
@endcode

Many OpenCV functions can be used to process cv.Mat. You can refer to other tutorials, such as @ref tutorial_js_table_of_contents_imgproc, for details.

In this tutorial, we just show a cv.Mat on screen. To show a cv.Mat, you need a canvas element.

@code{.js}
<canvas id="outputCanvas"></canvas>
@endcode

You can use cv.imshow to show cv.Mat on the canvas.
@code{.js}
cv.imshow(mat, "outputCanvas");
@endcode

Putting all of the steps  together, the final index.html is shown below.

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
  mat.delete();
}

function opencvIsReady() {
  document.getElementById("status").innerHTML = "OpenCV.js is ready.";
}
</script>

<script async src="opencv.js" onload="opencvIsReady();"></script>

</body>
</html>
@endcode

@note You have to call delete method of cv.Mat to free memory allocated in Emscripten's heap. Please refer to [Memeory management of Emscripten](https://kripken.github.io/emscripten-site/docs/porting/connecting_cpp_and_javascript/embind.html#memory-management) for details.

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

let imgElement = document.getElementById("srcImage")
let inputElement = document.getElementById("fileInput");
inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

imgElement.onload = function() {
  let mat = cv.imread(imgElement);
  cv.imshow("outputCanvas", mat);
  mat.delete();
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