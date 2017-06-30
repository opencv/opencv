Getting Started with Images {#tutorial_js_image_display}
===========================

Goals
-----

-   Learn how to read an image and how to display it in a web.

Read an image
-------------

OpenCV-JavaScript saves image as cv.Mat type. We use HTML canvas element to transfer cv.Mat to the web  
or in reverse. The ImageData interface can represent or set the underlying pixel data of an area of a 
canvas element. 

@sa Please refer to canvas docs for more details.

First, creat an ImageData obj from canvas.
@code{.js}
var canvas = document.getElementById(canvas_id);
var ctx = canvas.getContext('2d');
var imgData = ctx.getImageData(0,0,canvas.width, canvas.height);
@endcode

Then use cv.matFromArray to construct a cv.Mat.
@code{.js}
var src = cv.matFromArray(imgData, cv.CV_8UC4);
@endcode

@note Cause canvas only support 8-bit RGBA image, the cv.Mat type is cv.CV_8UC4.


Display an image
----------------

First, convert the type of src to cv.CV_8UC4.
@code{.js}
var dst = new cv.Mat();
cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_***2RGBA.value, 0); //*** is the type of src
@endcode

Then new an ImageData obj from dst.
@code{.js}
var imgData = new ImageData(new Uint8ClampedArray(dst.data()), dst.cols, dst.rows);
@endcode

Finally, display it.
@code{.js}
var canvas = document.getElementById(canvas_id);
var ctx = canvas.getContext("2d");
ctx.clearRect(0, 0, canvas.width, canvas.height);
canvas.width = imgData.width;
canvas.height = imgData.height;
ctx.putImageData(imgData, 0, 0);
@endcode

Try it
------

Open bin/img_proc.html in the browser, open an image file and open console to excute the below code.
To distinguish the input and output, we graying the image.
@code{.js}
var canvas1 = document.getElementById("canvas1");
var ctx1 = canvas1.getContext('2d');
var imgData1 = ctx1.getImageData(0,0,canvas1.width, canvas1.height);
var src = cv.matFromArray(imgData1, cv.CV_8UC4);
var dst = new cv.Mat();
cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
cv.cvtColor(dst, dst, cv.ColorConversionCodes.COLOR_GRAY2RGBA.value, 0);
var imgData2 = new ImageData(new Uint8ClampedArray(dst.data()), dst.cols, dst.rows);
var canvas2 = document.getElementById("canvas2");
var ctx2 = canvas2.getContext("2d");
ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
canvas2.width = imgData2.width;
canvas2.height = imgData2.height;
ctx2.putImageData(imgData2, 0, 0);
src.delete();
dst.delete();
@endcode
Result as below
![](images/Imread_Imshow_Tutorial_Result.png)

And there is an interactive webpage for this tutorial, [imshow](tutorial_js_interactive_imshow.html). 
You can change the code and investigate more.