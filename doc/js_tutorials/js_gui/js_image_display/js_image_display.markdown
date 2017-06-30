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
var ctx = canvas.getContext("2d");
var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
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

In addition, OpenCV-JavaScript implements imread and imshow using the above method. You can use imread and 
imshow to read image from html canvas and display it.
@code{.js}
var img = imread("canvas1");
imshow("canvas2", img);
img.delete();
@endcode

@note todo: imread => cv.imread, imshow => cv.imshow

Try it
------

Let's try the above code in the interactive webpage for this tutorial, [imshow](tutorial_js_interactive_imshow.html). 
@code{.js}
var src = imread("canvas1");
var dst = new cv.Mat();
// To distinguish the input and output, we graying the image.
// You can try more different conversion.
cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
cv.cvtColor(dst, dst, cv.ColorConversionCodes.COLOR_GRAY2RGBA.value, 0);
imshow("canvas2", dst);
src.delete();
dst.delete();
@endcode
Result as below
![](images/Imread_Imshow_Tutorial_Result.png)