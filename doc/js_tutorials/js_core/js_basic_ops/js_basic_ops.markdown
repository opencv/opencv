Basic Operations on Images {#tutorial_js_basic_ops}
==========================

Goal
----

-   Learn how to access image properties
-   Learn how to construct Mat
-   Learn how to copy Mat
-   Learn how to convert the type of Mat
-   Learn how to use MatVector
-   Learn how to access pixel values and modify them
-   Learn how to set Region of Interest (ROI)
-   Learn how to split and merge images

Accessing Image Properties
--------------------------

Image properties include number of rows, columns and size, depth, channels, type of image data.

@code{.js}
let src = cv.imread("canvasInput");
console.log('image width: ' + src.cols + '\n' +
            'image height: ' + src.rows + '\n' +
            'image size: ' + src.size().width + '*' src.size().height + '\n' +
            'image depth: ' + src.depth() + '\n' +
            'image channels ' + src.channels() + '\n' +
            'image type: ' + src.type() + '\n');
@endcode

@note src.type() is very important while debugging because a large number of errors in OpenCV.js
code are caused by invalid data type.

How to construct Mat
--------------------

There are 4 basic constructors:

@code{.js}
// 1. default constructor
let mat = new cv.Mat();
// 2. two-dimensional arrays by size and type
let mat = new cv.Mat(size, type);
// 3. two-dimensional arrays by rows, cols, and type
let mat = new cv.Mat(rows, cols, type);
// 4. two-dimensional arrays by rows, cols, and type with initialization value
let mat = new cv.Mat(rows, cols, type, new cv.Scalar());
@endcode

There are 3 static functions:

@code{.js}
// 1. Create a Mat which is full of zeros
let mat = cv.Mat.zeros(rows, cols, type);
// 2. Create a Mat which is full of ones
let mat = cv.Mat.ones(rows, cols, type);
// 3. Create a Mat which is an identity matrix
let mat = cv.Mat.eye(rows, cols, type);
@endcode

There are 2 factory functions:
@code{.js}
// 1. Use JS array to construct a mat.
// For example: let mat = cv.matFromArray(2, 2, cv.CV_8UC1, [1, 2, 3, 4]);
let mat = cv.matFromArray(rows, cols, type, array);
// 2. Use imgData to construct a mat
let ctx = canvas.getContext("2d");
let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
let mat = cv.matFromImageData(imgData);
@endcode

@note Don't forget to delete cv.Mat when you don't want to use it any more.

How to copy Mat
---------------

There are 2 ways to copy a Mat:

@code{.js}
// 1. Clone
let dst = src.clone();
// 2. CopyTo(only entries indicated in the mask are copied)
src.copyTo(dst, mask);
@endcode

How to convert the type of Mat
------------------------------

We use the function: **convertTo(m, rtype, alpha = 1, beta = 0)**
@param m        output matrix; if it does not have a proper size or type before the operation, it is reallocated.
@param rtype    desired output matrix type or, rather, the depth since the number of channels are the same as the input has; if rtype is negative, the output matrix will have the same type as the input.
@param alpha    optional scale factor.
@param beta     optional delta added to the scaled values.

@code{.js}
src.convertTo(dst, rtype);
@endcode

How use MatVector
-----------------

@code{.js}
let mat = new cv.Mat();
// Initialise a MatVector
let matVec = new cv.MatVector();
// Push a Mat back into MatVector
matVec.push_back(mat);
// Get a Mat fom MatVector
let cnt = matVec.get(0);
mat.delete(); matVec.delete(); cnt.delete();
@endcode

@note Don't forget to delete cv.Mat, cv.MatVector and cnt(the Mat you get from MatVector) when you don't want to use them any more.

Accessing and Modifying pixel values
------------------------------------

Firstly, you should know the following type relationship:

Data Properties  | C++ Type | JavaScript Typed Array | Mat Type
---------------  | -------- | ---------------------- | --------
data             | uchar    | Uint8Array             | CV_8U
data8S           | char     | Int8Array              | CV_8S
data16U          | ushort   | Uint16Array            | CV_16U
data16S          | short    | Int16Array             | CV_16S
data32S          | int      | Int32Array             | CV_32S
data32F          | float    | Float32Array           | CV_32F
data64F          | double   | Float64Array           | CV_64F

**1. data**

@code{.js}
let row = 3, col = 4;
let src = cv.imread("canvasInput");
if (src.isContinuous()) {
    let R = src.data[row * src.cols * src.channels() + col * src.channels()];
    let G = src.data[row * src.cols * src.channels() + col * src.channels() + 1];
    let B = src.data[row * src.cols * src.channels() + col * src.channels() + 2];
    let A = src.data[row * src.cols * src.channels() + col * src.channels() + 3];
}
@endcode

@note  Data manipulation is only valid for continuous Mat. You should use isContinuous() to check first.

**2. at**

Mat Type  | At Manipulation
--------- | ---------------
CV_8U     | ucharAt
CV_8S     | charAt
CV_16U    | ushortAt
CV_16S    | shortAt
CV_32S    | intAt
CV_32F    | floatAt
CV_64F    | doubleAt

@code{.js}
let row = 3, col = 4;
let src = cv.imread("canvasInput");
let R = src.ucharAt(row, col * src.channels());
let G = src.ucharAt(row, col * src.channels() + 1);
let B = src.ucharAt(row, col * src.channels() + 2);
let A = src.ucharAt(row, col * src.channels() + 3);
@endcode

@note  At manipulation is only for single channel access and the value can't be modified.

**3. ptr**

Mat Type  | Ptr Manipulation | JavaScript Typed Array
--------  | ---------------  | ----------------------
CV_8U     | ucharPtr         | Uint8Array
CV_8S     | charPtr          | Int8Array
CV_16U    | ushortPtr        | Uint16Array
CV_16S    | shortPtr         | Int16Array
CV_32S    | intPtr           | Int32Array
CV_32F    | floatPtr         | Float32Array
CV_64F    | doublePtr        | Float64Array

@code{.js}
let row = 3, col = 4;
let src = cv.imread("canvasInput");
let pixel = src.ucharPtr(row, col);
let R = pixel[0];
let G = pixel[1];
let B = pixel[2];
let A = pixel[3];
@endcode

mat.ucharPtr(k) get the k th row of the mat. mat.ucharPtr(i, j) get the i th row and the j th column of the mat.

Image ROI
---------

Sometimes, you will have to play with certain region of images. For eye detection in images, first
face detection is done all over the image and when face is obtained, we select the face region alone
and search for eyes inside it instead of searching whole image. It improves accuracy (because eyes
are always on faces) and performance (because we search for a small area)

We use the function: **roi (rect)**
@param rect    rectangle Region of Interest.

Try it
------

\htmlonly
<iframe src="../../js_basic_ops_roi.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly


Splitting and Merging Image Channels
------------------------------------

Sometimes you will need to work separately on R,G,B channels of image. Then you need to split the
RGB images to single planes. Or another time, you may need to join these individual channels to RGB
image.

@code{.js}
let src = cv.imread("canvasInput");
let rgbaPlanes = new cv.MatVector();
// Split the Mat
cv.split(src, rgbaPlanes);
// Get R channel
let R = rgbaPlanes.get(0);
// Merge all channels
cv.merge(rgbaPlanes, src);
src.delete(); rgbaPlanes.delete(); R.delete();
@endcode

@note Don't forget to delete cv.Mat, cv.MatVector and R(the Mat you get from MatVector) when you don't want to use them any more.

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

\htmlonly
<iframe src="../../js_basic_ops_copymakeborder.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly