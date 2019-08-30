Using OpenCV.js In Node.js{#tutorial_js_node}
===============================

@tableofcontents

Goals
-----

In this tutorial, you will learn:

-   Use OpenCV.js in a [Node.js](https://nodejs.org) application. 
-   Load images with [jimp](https://www.npmjs.com/package/jimp) in order to use them with OpenCV.js.
-   Use [node-canvas](https://www.npmjs.com/package/canvas) to support OpenCV.js utilities like `cv.imread()`, `cv.imshow()`, etc.
-   Introduce [emscripten](https://emscripten.org/) APIs, like [Module](https://emscripten.org/docs/api_reference/module.html) and [File System](https://emscripten.org/docs/api_reference/Filesystem-API.html) on which OpenCV.js is based.
-   Node.js basics. Although this tutorial assumes the user knows JavaScript, experience with Node.js is not required.

@note More than a recommendation this tutorial should be considered as Besides giving instructions to run OpenCV.js in Node.s, the objective of this tutorial is also to introduce [emscripten](https://emscripten.org/) APIs, like [Module](https://emscripten.org/docs/api_reference/module.html) and [File System](https://emscripten.org/docs/api_reference/Filesystem-API.html).

Minimal example
-----------------------------

@code{.js}
// Define a global variable 'Module' with a method 'onRuntimeInitialized':
Module = {
  onRuntimeInitialized() {
    // this is our application:
    console.log(cv.getBuildInformation())
  }
}
// Load 'opencv.js' assigning the value to the global variable 'cv'
cv = require('./opencv.js')
@endcode

Execute it
----

-   Save the file as `example1.js`.
-   Make sure the file `opencv.js` is in the same folder.
-   Make sure [Node.js](https://nodejs.org) is installed on your system.

The following command should print OpenCV build information:

@code{.bash}
node example1.js
@endcode

What just happened?
----

 * **In the first statement**: By defining a global variable named 'Module', emscripten will call `Module.onRuntimeInitialized()` when the library is ready to use. Our program is in that method and uses the global variable `cv` just like in the browser.
 
 * **cv = require('./opencv.js')**: In this statement, we require the file `opencv.js` and assign the return value to the global variable `cv`. This will load the library and as said previously emscripten will call `Module.onRuntimeInitialized()` when its ready.

 * See [emscripten Module API](https://emscripten.org/docs/api_reference/module.html) for more details.

Working with images
-----------------------------

OpenCV.js doesn't support image formats so we can't load png or jpeg images directly. In the browser it uses the HTML DOM (like HTMLCanvasElement and HTMLImageElement to decode and decode images. In node.js we will need to use a library for this.

In this example we use [jimp](https://www.npmjs.com/package/jimp), which supports common image formats and is pretty easy to use.

Install [jimp](https://www.npmjs.com/package/jimp)
----

Execute the following commands to create a new node.js package and install [jimp](https://www.npmjs.com/package/jimp) dependency:

@code{.bash}
mkdir project1
cd project1
npm init -y
npm install jimp
@endcode

**The example**

@code{.js}
const Jimp = require('jimp');

async function onRuntimeInitialized(){

  // load local image file with jimp. It supports jpg, png, bmp, tiff and gif:
  var jimpSrc = await Jimp.read('test/assets/lenna.jpg');

  // `jimpImage.bitmap` property has the decoded ImageData that we can use to create a cv:Mat
  var src = cv.matFromImageData(jimpSrc.bitmap);

  // following lines is copy&paste of opencv.js dilate tutorial:
  let dst = new cv.Mat();
  let M = cv.Mat.ones(5, 5, cv.CV_8U);
  let anchor = new cv.Point(-1, -1);
  cv.dilate(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

  // Now that we are finish, we want to write `dst` to file `output.png`. For this we create a `Jimp`
  // image which accepts the image data as a [`Buffer`](https://nodejs.org/docs/latest-v10.x/api/buffer.html). 
  // `write('output.png')` will write it to disk and Jimp infers the output format from given file name:
  new Jimp({
    width: dst.cols,
    height: dst.rows,
    data: Buffer.from(dst.data)
  })
  .write('output.png');

  src.delete();
  dst.delete();
}

// Finally, load the open.js as before. The function `onRuntimeInitialized` contains our program.
Module = {
  onRuntimeInitialized
};
cv = require('../static/opencv.js')
@code{.js}

Using OpenCV.js browser utilities in Node.js with [node-canvas](https://www.npmjs.com/package/canvas)
-----------------------------

TODO