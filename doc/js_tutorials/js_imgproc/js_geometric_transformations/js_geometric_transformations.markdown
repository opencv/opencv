Geometric Transformations of Images {#tutorial_js_geometric_transformations}
===================================

Goals
-----

-   Learn how to apply different geometric transformation to images like translation, rotation, affine
    transformation etc.
-   You will learn these functions: **cv.resize**, **cv.warpAffine**, **cv.getAffineTransform** and **cv.warpPerspective**

Transformations
---------------


### Scaling

Scaling is just resizing of the image. OpenCV comes with a function **cv.resize()** for this
purpose. The size of the image can be specified manually, or you can specify the scaling factor.
Different interpolation methods are used. Preferable interpolation methods are **cv.INTER_AREA**
for shrinking and **cv.INTER_CUBIC** (slow) & **cv.INTER_LINEAR** for zooming.

We use the function: **cv.resize (src, dst, dsize, fx = 0, fy = 0, interpolation = cv.INTER_LINEAR)**
@param src    input image
@param dst    output image; it has the size dsize (when it is non-zero) or the size computed from src.size(), fx, and fy; the type of dst is the same as of src.
@param dsize  output image size; if it equals zero, it is computed as:
                 \f[𝚍𝚜𝚒𝚣𝚎 = 𝚂𝚒𝚣𝚎(𝚛𝚘𝚞𝚗𝚍(𝚏𝚡*𝚜𝚛𝚌.𝚌𝚘𝚕𝚜), 𝚛𝚘𝚞𝚗𝚍(𝚏𝚢*𝚜𝚛𝚌.𝚛𝚘𝚠𝚜))\f]
                 Either dsize or both fx and fy must be non-zero.
@param fx     scale factor along the horizontal axis; when it equals 0, it is computed as  \f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜\f]

@param fy     scale factor along the vertical axis; when it equals 0, it is computed as \f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜\f]
@param interpolation    interpolation method(see **cv.InterpolationFlags**)

Try it
------

\htmlonly
<iframe src="../../js_geometric_transformations_resize.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### Translation

Translation is the shifting of object's location. If you know the shift in (x,y) direction, let it
be \f$(t_x,t_y)\f$, you can create the transformation matrix \f$\textbf{M}\f$ as follows:

\f[M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y  \end{bmatrix}\f]

We use the function: **cv.warpAffine (src, dst, M, dsize, flags = cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT, borderValue = new cv.Scalar())**
@param src          input image.
@param dst          output image that has the size dsize and the same type as src.
@param Mat          2 × 3 transformation matrix(cv.CV_64FC1 type).
@param dsize        size of the output image.
@param flags        combination of interpolation methods(see cv.InterpolationFlags) and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( 𝚍𝚜𝚝→𝚜𝚛𝚌 )
@param borderMode   pixel extrapolation method (see cv.BorderTypes); when borderMode = BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.
@param borderValue  value used in case of a constant border; by default, it is 0.

rows.

Try it
------

\htmlonly
<iframe src="../../js_geometric_transformations_warpAffine.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### Rotation

Rotation of an image for an angle \f$\theta\f$ is achieved by the transformation matrix of the form

\f[M = \begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta   \end{bmatrix}\f]

But OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any
location you prefer. Modified transformation matrix is given by

\f[\begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot center.x -  \beta \cdot center.y \\ - \beta &  \alpha &  \beta \cdot center.x + (1- \alpha )  \cdot center.y \end{bmatrix}\f]

where:

\f[\begin{array}{l} \alpha =  scale \cdot \cos \theta , \\ \beta =  scale \cdot \sin \theta \end{array}\f]

We use the function: **cv.getRotationMatrix2D (center, angle, scale)**
@param center    center of the rotation in the source image.
@param angle     rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
@param scale     isotropic scale factor.

Try it
------

\htmlonly
<iframe src="../../js_geometric_transformations_rotateWarpAffine.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### Affine Transformation

In affine transformation, all parallel lines in the original image will still be parallel in the
output image. To find the transformation matrix, we need three points from input image and their
corresponding locations in output image. Then **cv.getAffineTransform** will create a 2x3 matrix
which is to be passed to **cv.warpAffine**.

We use the function: **cv.getAffineTransform (src, dst)**

@param src    three points([3, 1] size and cv.CV_32FC2 type) from input imag.
@param dst    three corresponding points([3, 1] size and cv.CV_32FC2 type) in output image.

Try it
------

\htmlonly
<iframe src="../../js_geometric_transformations_getAffineTransform.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### Perspective Transformation

For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain straight even after the transformation. To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the function **cv.getPerspectiveTransform**. Then apply **cv.warpPerspective** with this 3x3 transformation matrix.

We use the functions: **cv.warpPerspective (src, dst, M, dsize, flags = cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT, borderValue = new cv.Scalar())**

@param src          input image.
@param dst          output image that has the size dsize and the same type as src.
@param Mat          3 × 3 transformation matrix(cv.CV_64FC1 type).
@param dsize        size of the output image.
@param flags        combination of interpolation methods (cv.INTER_LINEAR or cv.INTER_NEAREST) and the optional flag WARP_INVERSE_MAP, that sets M as the inverse transformation (𝚍𝚜𝚝→𝚜𝚛𝚌).
@param borderMode   pixel extrapolation method (cv.BORDER_CONSTANT or cv.BORDER_REPLICATE).
@param borderValue  value used in case of a constant border; by default, it is 0.

**cv.getPerspectiveTransform (src, dst)**

@param src          coordinates of quadrangle vertices in the source image.
@param dst          coordinates of the corresponding quadrangle vertices in the destination image.

Try it
------

\htmlonly
<iframe src="../../js_geometric_transformations_warpPerspective.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly