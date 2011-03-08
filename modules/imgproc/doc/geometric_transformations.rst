Geometric Image Transformations
===============================

The functions in this section perform various geometrical transformations of 2D images. That is, they do not change the image content, but deform the pixel grid, and map this deformed grid to the destination image. In fact, to avoid sampling artifacts, the mapping is done in the reverse order, from destination to the source. That is, for each pixel
:math:`(x, y)` of the destination image, the functions compute coordinates of the corresponding "donor" pixel in the source image and copy the pixel value, that is:

.. math::

    \texttt{dst} (x,y)= \texttt{src} (f_x(x,y), f_y(x,y))

In the case when the user specifies the forward mapping:
:math:`\left<g_x, g_y\right>: \texttt{src} \rightarrow \texttt{dst}` , the OpenCV functions first compute the corresponding inverse mapping:
:math:`\left<f_x, f_y\right>: \texttt{dst} \rightarrow \texttt{src}` and then use the above formula.

The actual implementations of the geometrical transformations, from the most generic
:ref:`Remap` and to the simplest and the fastest
:ref:`Resize` , need to solve the 2 main problems with the above formula:

#.
    extrapolation of non-existing pixels. Similarly to the filtering functions, described in the previous section, for some
    :math:`(x,y)`     one of
    :math:`f_x(x,y)`     or
    :math:`f_y(x,y)`     , or they both, may fall outside of the image, in which case some extrapolation method needs to be used. OpenCV provides the same selection of the extrapolation methods as in the filtering functions, but also an additional method ``BORDER_TRANSPARENT``     , which means that the corresponding pixels in the destination image will not be modified at all.

#.
    interpolation of pixel values. Usually
    :math:`f_x(x,y)`     and
    :math:`f_y(x,y)`     are floating-point numbers (i.e.
    :math:`\left<f_x, f_y\right>`     can be an affine or perspective transformation, or radial lens distortion correction etc.), so a pixel values at fractional coordinates needs to be retrieved. In the simplest case the coordinates can be just rounded to the nearest integer coordinates and the corresponding pixel used, which is called nearest-neighbor interpolation. However, a better result can be achieved by using more sophisticated `interpolation methods <http://en.wikipedia.org/wiki/Multivariate_interpolation>`_
    , where a polynomial function is fit into some neighborhood of the computed pixel
    :math:`(f_x(x,y), f_y(x,y))`     and then the value of the polynomial at
    :math:`(f_x(x,y), f_y(x,y))`     is taken as the interpolated pixel value. In OpenCV you can choose between several interpolation methods, see
    :ref:`Resize`     .

.. index:: convertMaps

.. _convertMaps:

convertMaps
-----------

.. c:function:: void convertMaps( const Mat& map1, const Mat& map2, Mat& dstmap1, Mat& dstmap2, int dstmap1type, bool nninterpolation=false )

    Converts image transformation maps from one representation to another

    :param map1: The first input map of type  ``CV_16SC2``  or  ``CV_32FC1``  or  ``CV_32FC2``
    
    :param map2: The second input map of type  ``CV_16UC1``  or  ``CV_32FC1``  or none (empty matrix), respectively

    :param dstmap1: The first output map; will have type  ``dstmap1type``  and the same size as  ``src``
    
    :param dstmap2: The second output map

    :param dstmap1type: The type of the first output map; should be  ``CV_16SC2`` , ``CV_32FC1``  or  ``CV_32FC2``
    
    :param nninterpolation: Indicates whether the fixed-point maps will be used for nearest-neighbor or for more complex interpolation

The function converts a pair of maps for
:func:`remap` from one representation to another. The following options ( ``(map1.type(), map2.type())`` :math:`\rightarrow` ``(dstmap1.type(), dstmap2.type())`` ) are supported:

#.
    :math:`\texttt{(CV\_32FC1, CV\_32FC1)} \rightarrow \texttt{(CV\_16SC2, CV\_16UC1)}`     . This is the most frequently used conversion operation, in which the original floating-point maps (see
    :func:`remap`     ) are converted to more compact and much faster fixed-point representation. The first output array will contain the rounded coordinates and the second array (created only when ``nninterpolation=false``     ) will contain indices in the interpolation tables.

#.
    :math:`\texttt{(CV\_32FC2)} \rightarrow \texttt{(CV\_16SC2, CV\_16UC1)}`     . The same as above, but the original maps are stored in one 2-channel matrix.

#.
    the reverse conversion. Obviously, the reconstructed floating-point maps will not be exactly the same as the originals.

See also:
:func:`remap`,:func:`undisort`,:func:`initUndistortRectifyMap`

.. index:: getAffineTransform

.. _getAffineTransform:

getAffineTransform
----------------------
.. c:function:: Mat getAffineTransform( const Point2f src[], const Point2f dst[] )

    Calculates the affine transform from 3 pairs of the corresponding points

    :param src: Coordinates of a triangle vertices in the source image

    :param dst: Coordinates of the corresponding triangle vertices in the destination image

The function calculates the :math:`2 \times 3` matrix of an affine transform such that:

.. math::

    \begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \texttt{map\_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}

where

.. math::

    dst(i)=(x'_i,y'_i),
    src(i)=(x_i, y_i),
    i=0,1,2

See also:
:func:`warpAffine`,:func:`transform`


.. index:: getPerspectiveTransform

.. _getPerspectiveTransform:

getPerspectiveTransform
---------------------------
.. c:function:: Mat getPerspectiveTransform( const Point2f src[], const Point2f dst[] )

    Calculates the perspective transform from 4 pairs of the corresponding points

    :param src: Coordinates of a quadrange vertices in the source image

    :param dst: Coordinates of the corresponding quadrangle vertices in the destination image

The function calculates the :math:`3 \times 3` matrix of a perspective transform such that:

.. math::

    \begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map\_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}

where

.. math::

    dst(i)=(x'_i,y'_i),
    src(i)=(x_i, y_i),
    i=0,1,2

See also:
:func:`findHomography`,:func:`warpPerspective`,:func:`perspectiveTransform`

.. index:: getRectSubPix

.. getRectSubPix:

getRectSubPix
-----------------
.. c:function:: void getRectSubPix( const Mat& image, Size patchSize, Point2f center, Mat& dst, int patchType=-1 )

    Retrieves the pixel rectangle from an image with sub-pixel accuracy

    :param src: Source image

    :param patchSize: Size of the extracted patch

    :param center: Floating point coordinates of the extracted rectangle center within the source image. The center must be inside the image

    :param dst: The extracted patch; will have the size  ``patchSize``  and the same number of channels as  ``src``
    
    :param patchType: The depth of the extracted pixels. By default they will have the same depth as  ``src``

The function ``getRectSubPix`` extracts pixels from ``src`` :

.. math::

    dst(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)

where the values of the pixels at non-integer coordinates are retrieved
using bilinear interpolation. Every channel of multiple-channel
images is processed independently. While the rectangle center
must be inside the image, parts of the rectangle may be
outside. In this case, the replication border mode (see
:func:`borderInterpolate` ) is used to extrapolate
the pixel values outside of the image.

See also:
:func:`warpAffine`,:func:`warpPerspective`

.. index:: getRotationMatrix2D

.. _getRotationMatrix2D:

getRotationMatrix2D
-----------------------
.. c:function:: Mat getRotationMatrix2D( Point2f center, double angle, double scale )

    Calculates the affine matrix of 2d rotation.

    :param center: Center of the rotation in the source image

    :param angle: The rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner)

    :param scale: Isotropic scale factor

The function calculates the following matrix:

.. math::

    \begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &  \alpha &  \beta \cdot \texttt{center.x} - (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix}

where

.. math::

    \begin{array}{l} \alpha =  \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta =  \texttt{scale} \cdot \sin \texttt{angle} \end{array}

The transformation maps the rotation center to itself. If this is not the purpose, the shift should be adjusted.

See also:
:func:`getAffineTransform`,:func:`warpAffine`,:func:`transform`

.. index:: invertAffineTransform

.. _invertAffineTransform:

invertAffineTransform
-------------------------
.. c:function:: void invertAffineTransform(const Mat& M, Mat& iM)

    Inverts an affine transformation

    :param M: The original affine transformation

    :param iM: The output reverse affine transformation

The function computes inverse affine transformation represented by
:math:`2 \times 3` matrix ``M`` :

.. math::

    \begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}

The result will also be a
:math:`2 \times 3` matrix of the same type as ``M`` .

.. index:: remap

.. _remap:

remap
-----

.. c:function:: void remap( const Mat& src, Mat& dst, const Mat& map1, const Mat& map2, int interpolation, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

    Applies a generic geometrical transformation to an image.

    :param src: Source image

    :param dst: Destination image. It will have the same size as  ``map1``  and the same type as  ``src``
    :param map1: The first map of either  ``(x,y)``  points or just  ``x``  values having type  ``CV_16SC2`` , ``CV_32FC1``  or  ``CV_32FC2`` . See  :func:`convertMaps`  for converting floating point representation to fixed-point for speed.

    :param map2: The second map of  ``y``  values having type  ``CV_16UC1`` , ``CV_32FC1``  or none (empty map if map1 is  ``(x,y)``  points), respectively

    :param interpolation: The interpolation method, see  :func:`resize` . The method  ``INTER_AREA``  is not supported by this function

    :param borderMode: The pixel extrapolation method, see  :func:`borderInterpolate` . When the \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function

    :param borderValue: A value used in the case of a constant border. By default it is 0

The function ``remap`` transforms the source image using the specified map:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))

Where values of pixels with non-integer coordinates are computed using one of the available interpolation methods.
:math:`map_x` and
:math:`map_y` can be encoded as separate floating-point maps in
:math:`map_1` and
:math:`map_2` respectively, or interleaved floating-point maps of
:math:`(x,y)` in
:math:`map_1` , or
fixed-point maps made by using
:func:`convertMaps` . The reason you might want to convert from floating to fixed-point
representations of a map is that they can yield much faster (~2x) remapping operations. In the converted case,
:math:`map_1` contains pairs ``(cvFloor(x), cvFloor(y))`` and
:math:`map_2` contains indices in a table of interpolation coefficients.

This function can not operate in-place.

.. index:: resize

.. _resize:

resize
----------

.. c:function:: void resize( const Mat& src, Mat& dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )

    Resizes an image

    :param src: Source image

    :param dst: Destination image. It will have size  ``dsize``  (when it is non-zero) or the size computed from  ``src.size()``         and  ``fx``  and  ``fy`` . The type of  ``dst``  will be the same as of  ``src`` .

    :param dsize: The destination image size. If it is zero, then it is computed as:

        .. math::

            \texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}

        .
        Either  ``dsize``  or both  ``fx``  or  ``fy``  must be non-zero.

    :param fx: The scale factor along the horizontal axis. When 0, it is computed as

        .. math::

            \texttt{(double)dsize.width/src.cols}

    :param fy: The scale factor along the vertical axis. When 0, it is computed as

        .. math::

            \texttt{(double)dsize.height/src.rows}

    :param interpolation: The interpolation method:

            * **INTER_NEAREST** nearest-neighbor interpolation

            * **INTER_LINEAR** bilinear interpolation (used by default)

            * **INTER_AREA** resampling using pixel area relation. It may be the preferred method for image decimation, as it gives moire-free results. But when the image is zoomed, it is similar to the  ``INTER_NEAREST``  method

            * **INTER_CUBIC** bicubic interpolation over 4x4 pixel neighborhood

            * **INTER_LANCZOS4** Lanczos interpolation over 8x8 pixel neighborhood

The function ``resize`` resizes an image ``src`` down to or up to the specified size.
Note that the initial ``dst`` type or size are not taken into account. Instead the size and type are derived from the ``src``,``dsize``,``fx`` and ``fy`` . If you want to resize ``src`` so that it fits the pre-created ``dst`` , you may call the function as: ::

    // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
    resize(src, dst, dst.size(), 0, 0, interpolation);


If you want to decimate the image by factor of 2 in each direction, you can call the function this way: ::

    // specify fx and fy and let the function to compute the destination image size.
    resize(src, dst, Size(), 0.5, 0.5, interpolation);


See also:
:func:`warpAffine`,:func:`warpPerspective`,:func:`remap` .

.. index:: warpAffine

.. _warpAffine:

warpAffine
--------------
.. c:function:: void warpAffine( const Mat& src, Mat& dst, const Mat& M, Size dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

    Applies an affine transformation to an image.

    :param src: Source image

    :param dst: Destination image; will have size  ``dsize``  and the same type as  ``src``
    
    :param M: :math:`2\times 3`  transformation matrix

    :param dsize: Size of the destination image

    :param flags: A combination of interpolation methods, see  :func:`resize` , and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` )

    :param borderMode: The pixel extrapolation method, see  :func:`borderInterpolate` . When the  \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function

    :param borderValue: A value used in case of a constant border. By default it is 0

The function ``warpAffine`` transforms the source image using the specified matrix:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})

when the flag ``WARP_INVERSE_MAP`` is set. Otherwise, the transformation is first inverted with
:func:`invertAffineTransform` and then put in the formula above instead of ``M`` .
The function can not operate in-place.

See also:
:func:`warpPerspective`,:func:`resize`,:func:`remap`,:func:`getRectSubPix`,:func:`transform`

.. index:: warpPerspective

.. _warpPerspective:

warpPerspective
-------------------
.. c:function:: void warpPerspective( const Mat& src, Mat& dst, const Mat& M, Size dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

    Applies a perspective transformation to an image.

    :param src: Source image

    :param dst: Destination image; will have size  ``dsize``  and the same type as  ``src``
    :param M: :math:`3\times 3`  transformation matrix

    :param dsize: Size of the destination image

    :param flags: A combination of interpolation methods, see  :func:`resize` , and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` )

    :param borderMode: The pixel extrapolation method, see  :func:`borderInterpolate` . When the  \   ``borderMode=BORDER_TRANSPARENT`` , it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function

    :param borderValue: A value used in case of a constant border. By default it is 0

The function ``warpPerspective`` transforms the source image using the specified matrix:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
         \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )

when the flag ``WARP_INVERSE_MAP`` is set. Otherwise, the transformation is first inverted with
:func:`invert` and then put in the formula above instead of ``M`` .
The function can not operate in-place.

See also:
:func:`warpAffine`,:func:`resize`,:func:`remap`,:func:`getRectSubPix`,:func:`perspectiveTransform`
 