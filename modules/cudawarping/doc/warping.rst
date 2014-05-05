Image Warping
=============

.. highlight:: cpp



cuda::remap
-----------
Applies a generic geometrical transformation to an image.

.. ocv:function:: void cuda::remap(InputArray src, OutputArray dst, InputArray xmap, InputArray ymap, int interpolation, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar(), Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image with the size the same as  ``xmap`` and the type the same as  ``src`` .

    :param xmap: X values. Only  ``CV_32FC1`` type is supported.

    :param ymap: Y values. Only  ``CV_32FC1`` type is supported.

    :param interpolation: Interpolation method (see  :ocv:func:`resize` ). ``INTER_NEAREST`` , ``INTER_LINEAR`` and ``INTER_CUBIC`` are supported for now.

    :param borderMode: Pixel extrapolation method (see  :ocv:func:`borderInterpolate` ). ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , ``BORDER_CONSTANT`` , ``BORDER_REFLECT`` and ``BORDER_WRAP`` are supported for now.

    :param borderValue: Value used in case of a constant border. By default, it is 0.

    :param stream: Stream for the asynchronous version.

The function transforms the source image using the specified map:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} (xmap(x,y), ymap(x,y))

Values of pixels with non-integer coordinates are computed using the bilinear interpolation.

.. seealso:: :ocv:func:`remap`



cuda::resize
------------
Resizes an image.

.. ocv:function:: void cuda::resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation = INTER_LINEAR, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image  with the same type as  ``src`` . The size is ``dsize`` (when it is non-zero) or the size is computed from  ``src.size()`` , ``fx`` , and  ``fy`` .

    :param dsize: Destination image size. If it is zero, it is computed as:

        .. math::
            \texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}

        Either  ``dsize`` or both  ``fx`` and  ``fy`` must be non-zero.

    :param fx: Scale factor along the horizontal axis. If it is zero, it is computed as:

        .. math::

            \texttt{(double)dsize.width/src.cols}

    :param fy: Scale factor along the vertical axis. If it is zero, it is computed as:

        .. math::

            \texttt{(double)dsize.height/src.rows}

    :param interpolation: Interpolation method. ``INTER_NEAREST`` , ``INTER_LINEAR`` and ``INTER_CUBIC`` are supported for now.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`resize`



cuda::warpAffine
----------------
Applies an affine transformation to an image.

.. ocv:function:: void cuda::warpAffine(InputArray src, OutputArray dst, InputArray M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar(), Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8U`` , ``CV_16U`` , ``CV_32S`` , or  ``CV_32F`` depth and 1, 3, or 4 channels are supported.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` .

    :param M: *2x3*  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize`) and the optional flag  ``WARP_INVERSE_MAP`` specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ). Only ``INTER_NEAREST`` , ``INTER_LINEAR`` , and  ``INTER_CUBIC`` interpolation methods are supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`warpAffine`



cuda::buildWarpAffineMaps
-------------------------
Builds transformation maps for affine transformation.

.. ocv:function:: void cuda::buildWarpAffineMaps(InputArray M, bool inverse, Size dsize, OutputArray xmap, OutputArray ymap, Stream& stream = Stream::Null())

    :param M: *2x3*  transformation matrix.

    :param inverse: Flag  specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ).

    :param dsize: Size of the destination image.

    :param xmap: X values with  ``CV_32FC1`` type.

    :param ymap: Y values with  ``CV_32FC1`` type.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`cuda::warpAffine` , :ocv:func:`cuda::remap`



cuda::warpPerspective
---------------------
Applies a perspective transformation to an image.

.. ocv:function:: void cuda::warpPerspective(InputArray src, OutputArray dst, InputArray M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar(), Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U`` , ``CV_16U`` , ``CV_32S`` , or  ``CV_32F`` depth and 1, 3, or 4 channels are supported.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` .

    :param M: *3x3* transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize` ) and the optional flag  ``WARP_INVERSE_MAP`` specifying that  ``M`` is the inverse transformation ( ``dst => src`` ). Only  ``INTER_NEAREST`` , ``INTER_LINEAR`` , and  ``INTER_CUBIC`` interpolation methods are supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`warpPerspective`



cuda::buildWarpPerspectiveMaps
------------------------------
Builds transformation maps for perspective transformation.

.. ocv:function:: void cuda::buildWarpAffineMaps(InputArray M, bool inverse, Size dsize, OutputArray xmap, OutputArray ymap, Stream& stream = Stream::Null())

    :param M: *3x3*  transformation matrix.

    :param inverse: Flag  specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ).

    :param dsize: Size of the destination image.

    :param xmap: X values with  ``CV_32FC1`` type.

    :param ymap: Y values with  ``CV_32FC1`` type.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`cuda::warpPerspective` , :ocv:func:`cuda::remap`



cuda::buildWarpPlaneMaps
------------------------
Builds plane warping maps.

.. ocv:function:: void cuda::buildWarpPlaneMaps(Size src_size, Rect dst_roi, InputArray K, InputArray R, InputArray T, float scale, OutputArray map_x, OutputArray map_y, Stream& stream = Stream::Null())

    :param stream: Stream for the asynchronous version.



cuda::buildWarpCylindricalMaps
------------------------------
Builds cylindrical warping maps.

.. ocv:function:: void cuda::buildWarpCylindricalMaps(Size src_size, Rect dst_roi, InputArray K, InputArray R, float scale, OutputArray map_x, OutputArray map_y, Stream& stream = Stream::Null())

    :param stream: Stream for the asynchronous version.



cuda::buildWarpSphericalMaps
----------------------------
Builds spherical warping maps.

.. ocv:function:: void cuda::buildWarpSphericalMaps(Size src_size, Rect dst_roi, InputArray K, InputArray R, float scale, OutputArray map_x, OutputArray map_y, Stream& stream = Stream::Null())

    :param stream: Stream for the asynchronous version.



cuda::rotate
------------
Rotates an image around the origin (0,0) and then shifts it.

.. ocv:function:: void cuda::rotate(InputArray src, OutputArray dst, Size dsize, double angle, double xShift = 0, double yShift = 0, int interpolation = INTER_LINEAR, Stream& stream = Stream::Null())

    :param src: Source image. Supports 1, 3 or 4 channels images with ``CV_8U`` , ``CV_16U`` or ``CV_32F`` depth.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` .

    :param dsize: Size of the destination image.

    :param angle: Angle of rotation in degrees.

    :param xShift: Shift along the horizontal axis.

    :param yShift: Shift along the vertical axis.

    :param interpolation: Interpolation method. Only  ``INTER_NEAREST`` , ``INTER_LINEAR`` , and  ``INTER_CUBIC`` are supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`cuda::warpAffine`



cuda::pyrDown
-------------
Smoothes an image and downsamples it.

.. ocv:function:: void cuda::pyrDown(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image. Will have ``Size((src.cols+1)/2, (src.rows+1)/2)`` size and the same type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`pyrDown`



cuda::pyrUp
-----------
Upsamples an image and then smoothes it.

.. ocv:function:: void cuda::pyrUp(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image. Will have ``Size(src.cols*2, src.rows*2)`` size and the same type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`pyrUp`
