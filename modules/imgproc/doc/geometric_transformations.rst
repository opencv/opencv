Geometric Image Transformations
===============================
.. highlight:: cpp

The functions in this section perform various geometrical transformations of 2D images. They do not change the image content but deform the pixel grid and map this deformed grid to the destination image. In fact, to avoid sampling artifacts, the mapping is done in the reverse order, from destination to the source. That is, for each pixel :math:`(x, y)` of the destination image, the functions compute    coordinates of the corresponding "donor" pixel in the source image and copy the pixel value:

.. math::

    \texttt{dst} (x,y)= \texttt{src} (f_x(x,y), f_y(x,y))

In case when you specify the forward mapping
:math:`\left<g_x, g_y\right>: \texttt{src} \rightarrow \texttt{dst}` , the OpenCV functions first compute the corresponding inverse mapping
:math:`\left<f_x, f_y\right>: \texttt{dst} \rightarrow \texttt{src}` and then use the above formula.

The actual implementations of the geometrical transformations, from the most generic
:ocv:func:`remap` and to the simplest and the fastest
:ocv:func:`resize` , need to solve two main problems with the above formula:

*
    Extrapolation of non-existing pixels. Similarly to the filtering functions described in the previous section, for some
    :math:`(x,y)`  ,   either one of
    :math:`f_x(x,y)`   ,  or
    :math:`f_y(x,y)`     , or both of them may fall outside of the image. In this case, an extrapolation method needs to be used. OpenCV provides the same selection of extrapolation methods as in the filtering functions. In addition, it provides the method ``BORDER_TRANSPARENT``   . This means that the corresponding pixels in the destination image will not be modified at all.

*
    Interpolation of pixel values. Usually
    :math:`f_x(x,y)`     and
    :math:`f_y(x,y)`     are floating-point numbers. This means that
    :math:`\left<f_x, f_y\right>`     can be either an affine or perspective transformation, or radial lens distortion correction, and so on. So, a pixel value at fractional coordinates needs to be retrieved. In the simplest case, the coordinates can be just rounded to the nearest integer coordinates and the corresponding pixel can be used. This is called a nearest-neighbor interpolation. However, a better result can be achieved by using more sophisticated `interpolation methods <http://en.wikipedia.org/wiki/Multivariate_interpolation>`_
    , where a polynomial function is fit into some neighborhood of the computed pixel
    :math:`(f_x(x,y), f_y(x,y))`   ,  and then the value of the polynomial at
    :math:`(f_x(x,y), f_y(x,y))`     is taken as the interpolated pixel value. In OpenCV, you can choose between several interpolation methods. See
    :ocv:func:`resize`   for details.

convertMaps
-----------
Converts image transformation maps from one representation to another.

.. ocv:function:: void convertMaps( InputArray map1, InputArray map2, OutputArray dstmap1, OutputArray dstmap2, int dstmap1type, bool nninterpolation=false )

.. ocv:pyfunction:: cv2.convertMaps(map1, map2, dstmap1type[, dstmap1[, dstmap2[, nninterpolation]]]) -> dstmap1, dstmap2

    :param map1: The first input map of type  ``CV_16SC2``  ,  ``CV_32FC1`` , or  ``CV_32FC2`` .

    :param map2: The second input map of type  ``CV_16UC1``  , ``CV_32FC1``  , or none (empty matrix), respectively.

    :param dstmap1: The first output map that has the type  ``dstmap1type``  and the same size as  ``src`` .

    :param dstmap2: The second output map.

    :param dstmap1type: Type of the first output map that should be  ``CV_16SC2`` , ``CV_32FC1`` , or  ``CV_32FC2`` .

    :param nninterpolation: Flag indicating whether the fixed-point maps are used for the nearest-neighbor or for a more complex interpolation.

The function converts a pair of maps for
:ocv:func:`remap` from one representation to another. The following options ( ``(map1.type(), map2.type())`` :math:`\rightarrow` ``(dstmap1.type(), dstmap2.type())`` ) are supported:

*
    :math:`\texttt{(CV\_32FC1, CV\_32FC1)} \rightarrow \texttt{(CV\_16SC2, CV\_16UC1)}`     . This is the most frequently used conversion operation, in which the original floating-point maps (see
    :ocv:func:`remap`     ) are converted to a more compact and much faster fixed-point representation. The first output array contains the rounded coordinates and the second array (created only when ``nninterpolation=false``     ) contains indices in the interpolation tables.

*
    :math:`\texttt{(CV\_32FC2)} \rightarrow \texttt{(CV\_16SC2, CV\_16UC1)}`     . The same as above but the original maps are stored in one 2-channel matrix.

*
    Reverse conversion. Obviously, the reconstructed floating-point maps will not be exactly the same as the originals.

.. seealso::

    :ocv:func:`remap`,
    :ocv:func:`undistort`,
    :ocv:func:`initUndistortRectifyMap`



getAffineTransform
----------------------
Calculates an affine transform from three pairs of the corresponding points.

.. ocv:function:: Mat getAffineTransform( InputArray src, InputArray dst )

.. ocv:function:: Mat getAffineTransform( const Point2f src[], const Point2f dst[] )

.. ocv:pyfunction:: cv2.getAffineTransform(src, dst) -> retval

.. ocv:cfunction:: CvMat* cvGetAffineTransform( const CvPoint2D32f * src, const CvPoint2D32f * dst, CvMat * map_matrix )

.. ocv:pyoldfunction:: cv.GetAffineTransform(src, dst, mapMatrix)-> None

    :param src: Coordinates of triangle vertices in the source image.

    :param dst: Coordinates of the corresponding triangle vertices in the destination image.

The function calculates the :math:`2 \times 3` matrix of an affine transform so that:

.. math::

    \begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \texttt{map\_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}

where

.. math::

    dst(i)=(x'_i,y'_i),
    src(i)=(x_i, y_i),
    i=0,1,2

.. seealso::

    :ocv:func:`warpAffine`,
    :ocv:func:`transform`



getPerspectiveTransform
---------------------------
Calculates a perspective transform from four pairs of the corresponding points.

.. ocv:function:: Mat getPerspectiveTransform( InputArray src, InputArray dst )

.. ocv:function:: Mat getPerspectiveTransform( const Point2f src[], const Point2f dst[] )

.. ocv:pyfunction:: cv2.getPerspectiveTransform(src, dst) -> retval

.. ocv:cfunction:: CvMat* cvGetPerspectiveTransform( const CvPoint2D32f* src, const CvPoint2D32f* dst, CvMat* map_matrix )

.. ocv:pyoldfunction:: cv.GetPerspectiveTransform(src, dst, mapMatrix)-> None

    :param src: Coordinates of quadrangle vertices in the source image.

    :param dst: Coordinates of the corresponding quadrangle vertices in the destination image.

The function calculates the :math:`3 \times 3` matrix of a perspective transform so that:

.. math::

    \begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map\_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}

where

.. math::

    dst(i)=(x'_i,y'_i),
    src(i)=(x_i, y_i),
    i=0,1,2,3

.. seealso::

    :ocv:func:`findHomography`,
    :ocv:func:`warpPerspective`,
    :ocv:func:`perspectiveTransform`


getRectSubPix
-----------------
Retrieves a pixel rectangle from an image with sub-pixel accuracy.

.. ocv:function:: void getRectSubPix( InputArray image, Size patchSize, Point2f center, OutputArray patch, int patchType=-1 )

.. ocv:pyfunction:: cv2.getRectSubPix(image, patchSize, center[, patch[, patchType]]) -> patch

.. ocv:cfunction:: void cvGetRectSubPix( const CvArr* src, CvArr* dst, CvPoint2D32f center )
.. ocv:pyoldfunction:: cv.GetRectSubPix(src, dst, center)-> None

    :param src: Source image.

    :param patchSize: Size of the extracted patch.

    :param center: Floating point coordinates of the center of the extracted rectangle within the source image. The center must be inside the image.

    :param dst: Extracted patch that has the size  ``patchSize``  and the same number of channels as  ``src`` .

    :param patchType: Depth of the extracted pixels. By default, they have the same depth as  ``src`` .

The function ``getRectSubPix`` extracts pixels from ``src`` :

.. math::

    dst(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)

where the values of the pixels at non-integer coordinates are retrieved
using bilinear interpolation. Every channel of multi-channel
images is processed independently. While the center of the rectangle
must be inside the image, parts of the rectangle may be
outside. In this case, the replication border mode (see
:ocv:func:`borderInterpolate` ) is used to extrapolate
the pixel values outside of the image.

.. seealso::

    :ocv:func:`warpAffine`,
    :ocv:func:`warpPerspective`


getRotationMatrix2D
-----------------------
Calculates an affine matrix of 2D rotation.

.. ocv:function:: Mat getRotationMatrix2D( Point2f center, double angle, double scale )

.. ocv:pyfunction:: cv2.getRotationMatrix2D(center, angle, scale) -> retval

.. ocv:cfunction:: CvMat* cv2DRotationMatrix( CvPoint2D32f center, double angle, double scale, CvMat* map_matrix )

.. ocv:pyoldfunction:: cv.GetRotationMatrix2D(center, angle, scale, mapMatrix)-> None

    :param center: Center of the rotation in the source image.

    :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).

    :param scale: Isotropic scale factor.

    :param map_matrix: The output affine transformation, 2x3 floating-point matrix.

The function calculates the following matrix:

.. math::

    \begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &  \alpha &  \beta \cdot \texttt{center.x} + (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix}

where

.. math::

    \begin{array}{l} \alpha =  \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta =  \texttt{scale} \cdot \sin \texttt{angle} \end{array}

The transformation maps the rotation center to itself. If this is not the target, adjust the shift.

.. seealso::

    :ocv:func:`getAffineTransform`,
    :ocv:func:`warpAffine`,
    :ocv:func:`transform`



invertAffineTransform
-------------------------
Inverts an affine transformation.

.. ocv:function:: void invertAffineTransform(InputArray M, OutputArray iM)

.. ocv:pyfunction:: cv2.invertAffineTransform(M[, iM]) -> iM

    :param M: Original affine transformation.

    :param iM: Output reverse affine transformation.

The function computes an inverse affine transformation represented by
:math:`2 \times 3` matrix ``M`` :

.. math::

    \begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}

The result is also a
:math:`2 \times 3` matrix of the same type as ``M`` .



LogPolar
--------
Remaps an image to log-polar space.

.. ocv:cfunction:: void cvLogPolar( const CvArr* src, CvArr* dst, CvPoint2D32f center, double M, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS )

.. ocv:pyoldfunction:: cv.LogPolar(src, dst, center, M, flags=CV_INNER_LINEAR+CV_WARP_FILL_OUTLIERS)-> None

    :param src: Source image

    :param dst: Destination image

    :param center: The transformation center; where the output precision is maximal

    :param M: Magnitude scale parameter. See below

    :param flags: A combination of interpolation methods and the following optional flags:

            *  **CV_WARP_FILL_OUTLIERS** fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to zero

            *  **CV_WARP_INVERSE_MAP** See below

The function ``cvLogPolar`` transforms the source image using the following transformation:

  *
    Forward transformation (``CV_WARP_INVERSE_MAP`` is not set):

        .. math::

            dst( \phi , \rho ) = src(x,y)


  *
    Inverse transformation (``CV_WARP_INVERSE_MAP`` is set):

        .. math::

            dst(x,y) = src( \phi , \rho )


where

    .. math::

        \rho = M  \cdot \log{\sqrt{x^2 + y^2}} , \phi =atan(y/x)


The function emulates the human "foveal" vision and can be used for fast scale and rotation-invariant template matching, for object tracking and so forth. The function can not operate in-place.


remap
-----
Applies a generic geometrical transformation to an image.

.. ocv:function:: void remap( InputArray src, OutputArray dst, InputArray map1, InputArray map2, int interpolation, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

.. ocv:pyfunction:: cv2.remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst

.. ocv:cfunction:: void cvRemap( const CvArr* src, CvArr* dst, const CvArr* mapx, const CvArr* mapy, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0) )
.. ocv:pyoldfunction:: cv.Remap(src, dst, mapx, mapy, flags=CV_INNER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))-> None

    :param src: Source image.

    :param dst: Destination image. It has the same size as  ``map1``  and the same type as  ``src`` .
    :param map1: The first map of either  ``(x,y)``  points or just  ``x``  values having the type  ``CV_16SC2`` , ``CV_32FC1`` , or  ``CV_32FC2`` . See  :ocv:func:`convertMaps`  for details on converting a floating point representation to fixed-point for speed.

    :param map2: The second map of  ``y``  values having the type  ``CV_16UC1`` , ``CV_32FC1`` , or none (empty map if ``map1`` is  ``(x,y)``  points), respectively.

    :param interpolation: Interpolation method (see  :ocv:func:`resize` ). The method  ``INTER_AREA``  is not supported by this function.

    :param borderMode: Pixel extrapolation method (see  :ocv:func:`borderInterpolate` ). When \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function.

    :param borderValue: Value used in case of a constant border. By default, it is 0.

The function ``remap`` transforms the source image using the specified map:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))

where values of pixels with non-integer coordinates are computed using one of available interpolation methods.
:math:`map_x` and
:math:`map_y` can be encoded as separate floating-point maps in
:math:`map_1` and
:math:`map_2` respectively, or interleaved floating-point maps of
:math:`(x,y)` in
:math:`map_1` , or
fixed-point maps created by using
:ocv:func:`convertMaps` . The reason you might want to convert from floating to fixed-point
representations of a map is that they can yield much faster (~2x) remapping operations. In the converted case,
:math:`map_1` contains pairs ``(cvFloor(x), cvFloor(y))`` and
:math:`map_2` contains indices in a table of interpolation coefficients.

This function cannot operate in-place.



resize
----------
Resizes an image.

.. ocv:function:: void resize( InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )

.. ocv:pyfunction:: cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst

.. ocv:cfunction:: void cvResize( const CvArr* src, CvArr* dst, int interpolation=CV_INTER_LINEAR )
.. ocv:pyoldfunction:: cv.Resize(src, dst, interpolation=CV_INTER_LINEAR)-> None

    :param src: Source image.

    :param dst: Destination image. It has the size  ``dsize``  (when it is non-zero) or the size computed from  ``src.size()``  ,  ``fx`` ,  and  ``fy`` . The type of  ``dst``  is the same as of  ``src`` .

    :param dsize: Destination image size. If it is zero, it is computed as:

        .. math::

            \texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}


        Either  ``dsize``  or both  ``fx``  and  ``fy``  must be non-zero.

    :param fx: Scale factor along the horizontal axis. When it is 0, it is computed as

        .. math::

            \texttt{(double)dsize.width/src.cols}

    :param fy: Scale factor along the vertical axis. When it is 0, it is computed as

        .. math::

            \texttt{(double)dsize.height/src.rows}

    :param interpolation: Interpolation method:

            * **INTER_NEAREST** - a nearest-neighbor interpolation

            * **INTER_LINEAR** - a bilinear interpolation (used by default)

            * **INTER_AREA** - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the  ``INTER_NEAREST``  method.

            * **INTER_CUBIC**  - a bicubic interpolation over 4x4 pixel neighborhood

            * **INTER_LANCZOS4** - a Lanczos interpolation over 8x8 pixel neighborhood

The function ``resize`` resizes the image ``src`` down to or up to the specified size.
Note that the initial ``dst`` type or size are not taken into account. Instead, the size and type are derived from the ``src``,``dsize``,``fx`` , and ``fy`` . If you want to resize ``src`` so that it fits the pre-created ``dst`` , you may call the function as follows: ::

    // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
    resize(src, dst, dst.size(), 0, 0, interpolation);


If you want to decimate the image by factor of 2 in each direction, you can call the function this way: ::

    // specify fx and fy and let the function compute the destination image size.
    resize(src, dst, Size(), 0.5, 0.5, interpolation);

To shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK).

.. seealso::

    :ocv:func:`warpAffine`,
    :ocv:func:`warpPerspective`,
    :ocv:func:`remap`


warpAffine
--------------
Applies an affine transformation to an image.

.. ocv:function:: void warpAffine( InputArray src, OutputArray dst, InputArray M, Size dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

.. ocv:pyfunction:: cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst

.. ocv:cfunction:: void cvWarpAffine( const CvArr* src, CvArr* dst, const CvMat* map_matrix, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0) )

.. ocv:pyoldfunction:: cv.WarpAffine(src, dst, mapMatrix, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))-> None

.. ocv:cfunction:: void cvGetQuadrangleSubPix( const CvArr* src, CvArr* dst, const CvMat* map_matrix )

.. ocv:pyoldfunction:: cv.GetQuadrangleSubPix(src, dst, mapMatrix)-> None

    :param src: Source image.

    :param dst: Destination image that has the size  ``dsize``  and the same type as  ``src`` .

    :param M: :math:`2\times 3`  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize` ) and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` ).

    :param borderMode: Pixel extrapolation method (see  :ocv:func:`borderInterpolate` ). When  \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.

    :param borderValue: Value used in case of a constant border. By default, it is 0.

The function ``warpAffine`` transforms the source image using the specified matrix:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})

when the flag ``WARP_INVERSE_MAP`` is set. Otherwise, the transformation is first inverted with
:ocv:func:`invertAffineTransform` and then put in the formula above instead of ``M`` .
The function cannot operate in-place.

.. seealso::

    :ocv:func:`warpPerspective`,
    :ocv:func:`resize`,
    :ocv:func:`remap`,
    :ocv:func:`getRectSubPix`,
    :ocv:func:`transform`


.. note:: ``cvGetQuadrangleSubPix`` is similar to ``cvWarpAffine``, but the outliers are extrapolated using replication border mode.

warpPerspective
-------------------
Applies a perspective transformation to an image.

.. ocv:function:: void warpPerspective( InputArray src, OutputArray dst, InputArray M, Size dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

.. ocv:pyfunction:: cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst

.. ocv:cfunction:: void cvWarpPerspective( const CvArr* src, CvArr* dst, const CvMat* map_matrix, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0) )

.. ocv:pyoldfunction:: cv.WarpPerspective(src, dst, mapMatrix, flags=CV_INNER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))-> None

    :param src: Source image.

    :param dst: Destination image that has the size  ``dsize``  and the same type as  ``src`` .

    :param M: :math:`3\times 3`  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize` ) and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` ).

    :param borderMode: Pixel extrapolation method (see  :ocv:func:`borderInterpolate` ). When  \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function.

    :param borderValue: Value used in case of a constant border. By default, it is 0.

The function ``warpPerspective`` transforms the source image using the specified matrix:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
         \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )

when the flag ``WARP_INVERSE_MAP`` is set. Otherwise, the transformation is first inverted with
:ocv:func:`invert` and then put in the formula above instead of ``M`` .
The function cannot operate in-place.

.. seealso::

    :ocv:func:`warpAffine`,
    :ocv:func:`resize`,
    :ocv:func:`remap`,
    :ocv:func:`getRectSubPix`,
    :ocv:func:`perspectiveTransform`




initUndistortRectifyMap
---------------------------
Computes the undistortion and rectification transformation map.

.. ocv:function:: void initUndistortRectifyMap( InputArray cameraMatrix, InputArray distCoeffs, InputArray R, InputArray newCameraMatrix, Size size, int m1type, OutputArray map1, OutputArray map2 )

.. ocv:pyfunction:: cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2

.. ocv:cfunction:: void cvInitUndistortRectifyMap( const CvMat* camera_matrix, const CvMat* dist_coeffs, const CvMat * R, const CvMat* new_camera_matrix, CvArr* mapx, CvArr* mapy )
.. ocv:cfunction:: void cvInitUndistortMap( const CvMat* camera_matrix, const CvMat* distortion_coeffs, CvArr* mapx, CvArr* mapy )

.. ocv:pyoldfunction:: cv.InitUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, map1, map2)-> None
.. ocv:pyoldfunction:: cv.InitUndistortMap(cameraMatrix, distCoeffs, map1, map2)-> None

    :param cameraMatrix: Input camera matrix  :math:`A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` .

    :param distCoeffs: Input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    :param R: Optional rectification transformation in the object space (3x3 matrix).  ``R1``  or  ``R2`` , computed by  :ocv:func:`stereoRectify`  can be passed here. If the matrix is empty, the identity transformation is assumed. In ``cvInitUndistortMap`` R assumed to be an identity matrix.

    :param newCameraMatrix: New camera matrix  :math:`A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}` .

    :param size: Undistorted image size.

    :param m1type: Type of the first output map that can be  ``CV_32FC1``  or  ``CV_16SC2`` . See  :ocv:func:`convertMaps` for details.

    :param map1: The first output map.

    :param map2: The second output map.

The function computes the joint undistortion and rectification transformation and represents the result in the form of maps for
:ocv:func:`remap` . The undistorted image looks like original, as if it is captured with a camera using the camera matrix ``=newCameraMatrix`` and zero distortion. In case of a monocular camera, ``newCameraMatrix`` is usually equal to ``cameraMatrix`` , or it can be computed by
:ocv:func:`getOptimalNewCameraMatrix` for a better control over scaling. In case of a stereo camera, ``newCameraMatrix`` is normally set to ``P1`` or ``P2`` computed by
:ocv:func:`stereoRectify` .

Also, this new camera is oriented differently in the coordinate space, according to ``R`` . That, for example, helps to align two heads of a stereo camera so that the epipolar lines on both images become horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera).

The function actually builds the maps for the inverse mapping algorithm that is used by
:ocv:func:`remap` . That is, for each pixel
:math:`(u, v)` in the destination (corrected and rectified) image, the function computes the corresponding coordinates in the source image (that is, in the original image from camera). The following process is applied:

.. math::

    \begin{array}{l} x  \leftarrow (u - {c'}_x)/{f'}_x  \\ y  \leftarrow (v - {c'}_y)/{f'}_y  \\{[X\,Y\,W]} ^T  \leftarrow R^{-1}*[x \, y \, 1]^T  \\ x'  \leftarrow X/W  \\ y'  \leftarrow Y/W  \\ x"  \leftarrow x' (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 x' y' + p_2(r^2 + 2 x'^2)  \\ y"  \leftarrow y' (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y'  \\ map_x(u,v)  \leftarrow x" f_x + c_x  \\ map_y(u,v)  \leftarrow y" f_y + c_y \end{array}

where
:math:`(k_1, k_2, p_1, p_2[, k_3])` are the distortion coefficients.

In case of a stereo camera, this function is called twice: once for each camera head, after
:ocv:func:`stereoRectify` , which in its turn is called after
:ocv:func:`stereoCalibrate` . But if the stereo camera was not calibrated, it is still possible to compute the rectification transformations directly from the fundamental matrix using
:ocv:func:`stereoRectifyUncalibrated` . For each camera, the function computes homography ``H`` as the rectification transformation in a pixel domain, not a rotation matrix ``R`` in 3D space. ``R`` can be computed from ``H`` as

.. math::

    \texttt{R} =  \texttt{cameraMatrix} ^{-1}  \cdot \texttt{H} \cdot \texttt{cameraMatrix}

where ``cameraMatrix`` can be chosen arbitrarily.




getDefaultNewCameraMatrix
-------------------------
Returns the default new camera matrix.

.. ocv:function:: Mat getDefaultNewCameraMatrix(InputArray cameraMatrix, Size imgsize=Size(), bool centerPrincipalPoint=false )

.. ocv:pyfunction:: cv2.getDefaultNewCameraMatrix(cameraMatrix[, imgsize[, centerPrincipalPoint]]) -> retval

    :param cameraMatrix: Input camera matrix.

    :param imgsize: Camera view image size in pixels.

    :param centerPrincipalPoint: Location of the principal point in the new camera matrix. The parameter indicates whether this location should be at the image center or not.

The function returns the camera matrix that is either an exact copy of the input ``cameraMatrix`` (when ``centerPrinicipalPoint=false`` ), or the modified one (when ``centerPrincipalPoint=true``).

In the latter case, the new camera matrix will be:

.. math::

    \begin{bmatrix} f_x && 0 && ( \texttt{imgSize.width} -1)*0.5  \\ 0 && f_y && ( \texttt{imgSize.height} -1)*0.5  \\ 0 && 0 && 1 \end{bmatrix} ,

where
:math:`f_x` and
:math:`f_y` are
:math:`(0,0)` and
:math:`(1,1)` elements of ``cameraMatrix`` , respectively.

By default, the undistortion functions in OpenCV (see
:ocv:func:`initUndistortRectifyMap`,
:ocv:func:`undistort`) do not move the principal point. However, when you work with stereo, it is important to move the principal points in both views to the same y-coordinate (which is required by most of stereo correspondence algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for each view where the principal points are located at the center.




undistort
-------------
Transforms an image to compensate for lens distortion.

.. ocv:function:: void undistort( InputArray src, OutputArray dst, InputArray cameraMatrix, InputArray distCoeffs, InputArray newCameraMatrix=noArray() )

.. ocv:pyfunction:: cv2.undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) -> dst

.. ocv:cfunction:: void cvUndistort2( const CvArr* src, CvArr* dst, const CvMat* camera_matrix, const CvMat* distortion_coeffs, const CvMat* new_camera_matrix=0 )

.. ocv:pyoldfunction:: cv.Undistort2(src, dst, cameraMatrix, distCoeffs)-> None

    :param src: Input (distorted) image.

    :param dst: Output (corrected) image that has the same size and type as  ``src`` .

    :param cameraMatrix: Input camera matrix  :math:`A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` .

    :param distCoeffs: Input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    :param newCameraMatrix: Camera matrix of the distorted image. By default, it is the same as  ``cameraMatrix``  but you may additionally scale and shift the result by using a different matrix.

The function transforms an image to compensate radial and tangential lens distortion.

The function is simply a combination of
:ocv:func:`initUndistortRectifyMap` (with unity ``R`` ) and
:ocv:func:`remap` (with bilinear interpolation). See the former function for details of the transformation being performed.

Those pixels in the destination image, for which there is no correspondent pixels in the source image, are filled with zeros (black color).

A particular subset of the source image that will be visible in the corrected image can be regulated by ``newCameraMatrix`` . You can use
:ocv:func:`getOptimalNewCameraMatrix` to compute the appropriate ``newCameraMatrix``  depending on your requirements.

The camera matrix and the distortion parameters can be determined using
:ocv:func:`calibrateCamera` . If the resolution of images is different from the resolution used at the calibration stage,
:math:`f_x, f_y, c_x` and
:math:`c_y` need to be scaled accordingly, while the distortion coefficients remain the same.




undistortPoints
-------------------
Computes the ideal point coordinates from the observed point coordinates.

.. ocv:function:: void undistortPoints( InputArray src, OutputArray dst, InputArray cameraMatrix, InputArray distCoeffs, InputArray R=noArray(), InputArray P=noArray())

.. ocv:cfunction:: void cvUndistortPoints( const CvMat* src, CvMat* dst, const CvMat* camera_matrix, const CvMat* dist_coeffs, const CvMat* R=0, const CvMat* P=0 )
.. ocv:pyoldfunction:: cv.UndistortPoints(src, dst, cameraMatrix, distCoeffs, R=None, P=None)-> None

    :param src: Observed point coordinates, 1xN or Nx1 2-channel (CV_32FC2 or CV_64FC2).

    :param dst: Output ideal point coordinates after undistortion and reverse perspective transformation. If matrix ``P`` is identity  or omitted, ``dst`` will contain normalized point coordinates.

    :param cameraMatrix: Camera matrix  :math:`\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` .

    :param distCoeffs: Input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    :param R: Rectification transformation in the object space (3x3 matrix).  ``R1``  or  ``R2``  computed by  :ocv:func:`stereoRectify`  can be passed here. If the matrix is empty, the identity transformation is used.

    :param P: New camera matrix (3x3) or new projection matrix (3x4).  ``P1``  or  ``P2``  computed by  :ocv:func:`stereoRectify`  can be passed here. If the matrix is empty, the identity new camera matrix is used.

The function is similar to
:ocv:func:`undistort` and
:ocv:func:`initUndistortRectifyMap`  but it operates on a sparse set of points instead of a raster image. Also the function performs a reverse transformation to
:ocv:func:`projectPoints` . In case of a 3D object, it does not reconstruct its 3D coordinates, but for a planar object, it does, up to a translation vector, if the proper ``R`` is specified. ::

    // (u,v) is the input point, (u', v') is the output point
    // camera_matrix=[fx 0 cx; 0 fy cy; 0 0 1]
    // P=[fx' 0 cx' tx; 0 fy' cy' ty; 0 0 1 tz]
    x" = (u - cx)/fx
    y" = (v - cy)/fy
    (x',y') = undistort(x",y",dist_coeffs)
    [X,Y,W]T = R*[x' y' 1]T
    x = X/W, y = Y/W
    // only performed if P=[fx' 0 cx' [tx]; 0 fy' cy' [ty]; 0 0 1 [tz]] is specified
    u' = x*fx' + cx'
    v' = y*fy' + cy',

where ``undistort()`` is an approximate iterative algorithm that estimates the normalized original point coordinates out of the normalized distorted point coordinates ("normalized" means that the coordinates do not depend on the camera matrix).

The function can be used for both a stereo camera head or a monocular camera (when R is empty).


