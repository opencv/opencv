Geometric Image Transformations
===============================
.. highlight:: cpp
The functions in this section perform various geometrical transformations of 2D images. They do not change the image content but deform the pixel grid and map this deformed grid to the destination image. In fact, to avoid sampling artifacts, the mapping is done in the reverse order, from destination to the source. That is, for each pixel
:math:`(x, y)` of the destination image, the functions compute coordinates of the corresponding "donor" pixel in the source image and copy the pixel value:

.. math::

    \texttt{dst} (x,y)= \texttt{src} (f_x(x,y), f_y(x,y))

In case when you specify the forward mapping
:math:`\left<g_x, g_y\right>: \texttt{src} \rightarrow \texttt{dst}` , the OpenCV functions first compute the corresponding inverse mapping
:math:`\left<f_x, f_y\right>: \texttt{dst} \rightarrow \texttt{src}` and then use the above formula.

The actual implementations of the geometrical transformations, from the most generic
:ref:`Remap` and to the simplest and the fastest
:ref:`Resize` , need to solve two main problems with the above formula:

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
    :ref:`Resize`   for details.

.. index:: convertMaps

.. _convertMaps:

convertMaps
-----------

.. c:function:: void convertMaps( const Mat& map1, const Mat& map2, Mat& dstmap1, Mat& dstmap2, int dstmap1type, bool nninterpolation=false )

    Converts image transformation maps from one representation to another.

    :param map1: The first input map of type  ``CV_16SC2``  ,  ``CV_32FC1`` , or  ``CV_32FC2`` .
    
    :param map2: The second input map of type  ``CV_16UC1``  , ``CV_32FC1``  , or none (empty matrix), respectively.

    :param dstmap1: The first output map that has the type  ``dstmap1type``  and the same size as  ``src`` .
    
    :param dstmap2: The second output map.

    :param dstmap1type: Type of the first output map that should be  ``CV_16SC2`` , ``CV_32FC1`` , or  ``CV_32FC2`` .
    
    :param nninterpolation: Flag indicating whether the fixed-point maps are used for the nearest-neighbor or for a more complex interpolation.

The function converts a pair of maps for
:func:`remap` from one representation to another. The following options ( ``(map1.type(), map2.type())`` :math:`\rightarrow` ``(dstmap1.type(), dstmap2.type())`` ) are supported:

*
    :math:`\texttt{(CV\_32FC1, CV\_32FC1)} \rightarrow \texttt{(CV\_16SC2, CV\_16UC1)}`     . This is the most frequently used conversion operation, in which the original floating-point maps (see
    :func:`remap`     ) are converted to a more compact and much faster fixed-point representation. The first output array contains the rounded coordinates and the second array (created only when ``nninterpolation=false``     ) contains indices in the interpolation tables.

*
    :math:`\texttt{(CV\_32FC2)} \rightarrow \texttt{(CV\_16SC2, CV\_16UC1)}`     . The same as above but the original maps are stored in one 2-channel matrix.

*
    Reverse conversion. Obviously, the reconstructed floating-point maps will not be exactly the same as the originals.

See Also:
:func:`remap`,
:func:`undisort`,
:func:`initUndistortRectifyMap`

.. index:: getAffineTransform

.. _getAffineTransform:

getAffineTransform
----------------------
.. c:function:: Mat getAffineTransform( const Point2f src[], const Point2f dst[] )

    Calculates an affine transform from three pairs of the corresponding points.

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

See Also:
:func:`warpAffine`,
:func:`transform`


.. index:: getPerspectiveTransform

.. _getPerspectiveTransform:

getPerspectiveTransform
---------------------------
.. c:function:: Mat getPerspectiveTransform( const Point2f src[], const Point2f dst[] )

    Calculates a perspective transform from four pairs of the corresponding points.

    :param src: Coordinates of quadrangle vertices in the source image.

    :param dst: Coordinates of the corresponding quadrangle vertices in the destination image.

The function calculates the :math:`3 \times 3` matrix of a perspective transform so that:

.. math::

    \begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map\_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}

where

.. math::

    dst(i)=(x'_i,y'_i),
    src(i)=(x_i, y_i),
    i=0,1,2

See Also:
:func:`findHomography`,
:func:`warpPerspective`,
:func:`perspectiveTransform`

.. index:: getRectSubPix

.. getRectSubPix:

getRectSubPix
-----------------
.. c:function:: void getRectSubPix( const Mat& image, Size patchSize, Point2f center, Mat& dst, int patchType=-1 )

    Retrieves a pixel rectangle from an image with sub-pixel accuracy.

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
:func:`borderInterpolate` ) is used to extrapolate
the pixel values outside of the image.

See Also:
:func:`warpAffine`,
:func:`warpPerspective`

.. index:: getRotationMatrix2D

.. _getRotationMatrix2D:

getRotationMatrix2D
-----------------------
.. c:function:: Mat getRotationMatrix2D( Point2f center, double angle, double scale )

    Calculates an affine matrix of 2D rotation.

    :param center: Center of the rotation in the source image.

    :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).

    :param scale: Isotropic scale factor.

The function calculates the following matrix:

.. math::

    \begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &  \alpha &  \beta \cdot \texttt{center.x} - (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix}

where

.. math::

    \begin{array}{l} \alpha =  \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta =  \texttt{scale} \cdot \sin \texttt{angle} \end{array}

The transformation maps the rotation center to itself. If this is not the target, adjust the shift.

See Also:
:func:`getAffineTransform`,
:func:`warpAffine`,
:func:`transform`

.. index:: invertAffineTransform

.. _invertAffineTransform:

invertAffineTransform
-------------------------
.. c:function:: void invertAffineTransform(const Mat& M, Mat& iM)

    Inverts an affine transformation.

    :param M: Original affine transformation.

    :param iM: Output reverse affine transformation.

The function computes an inverse affine transformation represented by
:math:`2 \times 3` matrix ``M`` :

.. math::

    \begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}

The result is also a
:math:`2 \times 3` matrix of the same type as ``M`` .

.. index:: remap

.. _remap:

remap
-----

.. c:function:: void remap( const Mat& src, Mat& dst, const Mat& map1, const Mat& map2, int interpolation, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

    Applies a generic geometrical transformation to an image.

    :param src: Source image.

    :param dst: Destination image. It has the same size as  ``map1``  and the same type as  ``src`` .
    :param map1: The first map of either  ``(x,y)``  points or just  ``x``  values having the type  ``CV_16SC2`` , ``CV_32FC1`` , or  ``CV_32FC2`` . See  :func:`convertMaps`  for details on converting a floating point representation to fixed-point for speed.

    :param map2: The second map of  ``y``  values having the type  ``CV_16UC1`` , ``CV_32FC1`` , or none (empty map if ``map1`` is  ``(x,y)``  points), respectively.

    :param interpolation: Interpolation method (see  :func:`resize` ). The method  ``INTER_AREA``  is not supported by this function.

    :param borderMode: Pixel extrapolation method (see  :func:`borderInterpolate` ). When \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function.

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
:func:`convertMaps` . The reason you might want to convert from floating to fixed-point
representations of a map is that they can yield much faster (~2x) remapping operations. In the converted case,
:math:`map_1` contains pairs ``(cvFloor(x), cvFloor(y))`` and
:math:`map_2` contains indices in a table of interpolation coefficients.

This function cannot operate in-place.

.. index:: resize

.. _resize:

resize
----------

.. c:function:: void resize( const Mat& src, Mat& dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )

    Resizes an image.

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


See Also:
:func:`warpAffine`,
:func:`warpPerspective`,
:func:`remap` 

.. index:: warpAffine

.. _warpAffine:

warpAffine
--------------
.. c:function:: void warpAffine( const Mat& src, Mat& dst, const Mat& M, Size dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

    Applies an affine transformation to an image.

    :param src: Source image.

    :param dst: Destination image that has the size  ``dsize``  and the same type as  ``src`` .
    
    :param M: :math:`2\times 3`  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :func:`resize` ) and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` ).

    :param borderMode: Pixel extrapolation method (see  :func:`borderInterpolate` ). When  \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.

    :param borderValue: Value used in case of a constant border. By default, it is 0.

The function ``warpAffine`` transforms the source image using the specified matrix:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})

when the flag ``WARP_INVERSE_MAP`` is set. Otherwise, the transformation is first inverted with
:func:`invertAffineTransform` and then put in the formula above instead of ``M`` .
The function cannot operate in-place.

See Also:
:func:`warpPerspective`,
:func:`resize`,
:func:`remap`,
:func:`getRectSubPix`,
:func:`transform`

.. index:: warpPerspective

.. _warpPerspective:

warpPerspective
-------------------
.. c:function:: void warpPerspective( const Mat& src, Mat& dst, const Mat& M, Size dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

    Applies a perspective transformation to an image.

    :param src: Source image.

    :param dst: Destination image that has the size  ``dsize``  and the same type as  ``src`` .
    
	:param M: :math:`3\times 3`  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :func:`resize` ) and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` ).

    :param borderMode: Pixel extrapolation method (see  :func:`borderInterpolate` ). When  \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function.

    :param borderValue: Value used in case of a constant border. By default, it is 0.

The function ``warpPerspective`` transforms the source image using the specified matrix:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
         \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )

when the flag ``WARP_INVERSE_MAP`` is set. Otherwise, the transformation is first inverted with
:func:`invert` and then put in the formula above instead of ``M`` .
The function cannot operate in-place.

See Also:
:func:`warpAffine`,
:func:`resize`,
:func:`remap`,
:func:`getRectSubPix`,
:func:`perspectiveTransform`
 