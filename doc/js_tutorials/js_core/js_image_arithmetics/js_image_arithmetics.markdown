Arithmetic Operations on Images {#tutorial_js_image_arithmetics}
===============================

Goal
----

-   Learn several arithmetic operations on images like addition, subtraction, bitwise operations
    etc.
-   You will learn these functions : **cv.add()**, **cv.subtract()**  etc.

Image Addition
--------------

You can add two images by OpenCV function, cv.add(). res = img1 + img2. Both images should be of same depth and type.

For example, consider below sample:
@code{.js}
let src1 = cv.imread("canvasInput1");
let src2 = cv.imread("canvasInput2");
let dst = new cv.Mat();
let mask = new cv.Mat();
let dtype = -1;
cv.add(src1, src2, dst, mask, dtype);
src1.delete(); src2.delete(); dst.delete(); mask.delete();
@endcode

Image Subtraction
--------------

You can subtract two images by OpenCV function, cv.subtract(). res = img1 - img2. Both images should be of same depth and type.

For example, consider below sample:
@code{.js}
let src1 = cv.imread("canvasInput1");
let src2 = cv.imread("canvasInput2");
let dst = new cv.Mat();
let mask = new cv.Mat();
let dtype = -1;
cv.subtract(src1, src2, dst, mask, dtype);
src1.delete(); src2.delete(); dst.delete(); mask.delete();
@endcode

Bitwise Operations
------------------

This includes bitwise AND, OR, NOT and XOR operations. They will be highly useful while extracting
any part of the image, defining and working with non-rectangular
ROI etc. Below we will see an example on how to change a particular region of an image.

I want to put OpenCV logo above an image. If I add two images, it will change color. If I blend it,
I get an transparent effect. But I want it to be opaque. If it was a rectangular region, I could use
ROI as we did in last chapter. But OpenCV logo is a not a rectangular shape. So you can do it with
bitwise operations.

Try it
------

\htmlonly
<iframe src="../../js_image_arithmetics_bitwise.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly