Image Processing
================

.. highlight:: cpp

.. index:: gpu::meanShiftFiltering

gpu::meanShiftFiltering
---------------------------
.. ocv:function:: void gpu::meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr,TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1))

    Performs mean-shift filtering for each point of the source image. It maps each point of the source image into another point. As a result, you have a new color and new position of each point.

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dst: Destination image containing the color of mapped points. It has the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.

.. index:: gpu::meanShiftProc

gpu::meanShiftProc
----------------------
.. ocv:function:: void gpu::meanShiftProc(const GpuMat& src, GpuMat& dstr, GpuMat& dstsp, int sp, int sr, TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1))

    Performs a mean-shift procedure and stores information about processed points (their colors and positions) in two images.

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dstr: Destination image containing the color of mapped points. The size and type is the same as  ``src`` .

    :param dstsp: Destination image containing the position of mapped points. The size is the same as  ``src`` size. The type is  ``CV_16SC2``.

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.

See Also:
:ocv:func:`gpu::meanShiftFiltering` 

.. index:: gpu::meanShiftSegmentation

gpu::meanShiftSegmentation
------------------------------
.. ocv:function:: void gpu::meanShiftSegmentation(const GpuMat& src, Mat& dst, int sp, int sr, int minsize, TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1))

    Performs a mean-shift segmentation of the source image and eliminates small segments.

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dst: Segmented image with the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param minsize: Minimum segment size. Smaller segements are merged.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.

.. index:: gpu::integral

gpu::integral
-----------------
.. ocv:function:: void gpu::integral(const GpuMat& src, GpuMat& sum)

.. ocv:function:: void gpu::integral(const GpuMat& src, GpuMat& sum, GpuMat& sqsum)

    Computes an integral image and a squared integral image.

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sum: Integral image containing 32-bit unsigned integer values packed into  ``CV_32SC1`` .

    :param sqsum: Squared integral image of the  ``CV_32FC1`` type.

See Also:
:ocv:func:`integral` 

.. index:: gpu::sqrIntegral

gpu::sqrIntegral
--------------------
.. ocv:function:: void gpu::sqrIntegral(const GpuMat& src, GpuMat& sqsum)

    Computes a squared integral image.

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sqsum: Squared integral image containing 64-bit unsigned integer values packed into  ``CV_64FC1`` .

.. index:: gpu::columnSum

gpu::columnSum
------------------
.. ocv:function:: void gpu::columnSum(const GpuMat& src, GpuMat& sum)

    Computes a vertical (column) sum.

    :param src: Source image. Only  ``CV_32FC1`` images are supported for now.

    :param sum: Destination image of the  ``CV_32FC1`` type.

.. index:: gpu::cornerHarris

gpu::cornerHarris
---------------------
.. ocv:function:: void gpu::cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType=BORDER_REFLECT101)

    Computes the Harris cornerness criteria at each image pixel.

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. It has the same size as ``src`` and ``CV_32FC1`` type.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101`` and  ``BORDER_REPLICATE`` are supported for now.

See Also:
:ocv:func:`cornerHarris` 

.. index:: gpu::cornerMinEigenVal

gpu::cornerMinEigenVal
--------------------------
.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType=BORDER_REFLECT101)

    Computes the minimum eigen value of 2x2 derivative covariation matrix at each pixel (the cornerness criteria).

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. The size is the same. The type is  ``CV_32FC1``.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only ``BORDER_REFLECT101`` and ``BORDER_REPLICATE`` are supported for now.

See also: :ocv:func:`cornerMinEigenVal`

.. index:: gpu::mulSpectrums

gpu::mulSpectrums
---------------------
.. ocv:function:: void gpu::mulSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, bool conjB=false)

    Performs a per-element multiplication of two Fourier spectrums.

    :param a: First spectrum.

    :param b: Second spectrum with the same size and type as  ``a`` .

    :param c: Destination spectrum.

    :param flags: Mock parameter used for CPU/GPU interfaces similarity.

    :param conjB: Optional flag to specify if the second spectrum needs to be conjugated before the multiplication.

    Only full (not packed) ``CV_32FC2`` complex spectrums in the interleaved format are supported for now.

See Also:
:ocv:func:`mulSpectrums` 

.. index:: gpu::mulAndScaleSpectrums

gpu::mulAndScaleSpectrums
-----------------------------
.. ocv:function:: void gpu::mulAndScaleSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, float scale, bool conjB=false)

    Performs a per-element multiplication of two Fourier spectrums and scales the result.

    :param a: First spectrum.

    :param b: Second spectrum with the same size and type as  ``a`` .

    :param c: Destination spectrum.

    :param flags: Mock parameter used for CPU/GPU interfaces similarity.

    :param scale: Scale constant.

    :param conjB: Optional flag to specify if the second spectrum needs to be conjugated before the multiplication.

    Only full (not packed) ``CV_32FC2`` complex spectrums in the interleaved format are supported for now.

See Also:
:ocv:func:`mulSpectrums` 

.. index:: gpu::dft

gpu::dft
------------
.. ocv:function:: void gpu::dft(const GpuMat& src, GpuMat& dst, Size dft_size, int flags=0)

    Performs a forward or inverse discrete Fourier transform (1D or 2D) of the floating point matrix. Use to handle real matrices (``CV32FC1``) and complex matrices in the interleaved format (``CV32FC2``).

    :param src: Source matrix (real or complex).

    :param dst: Destination matrix (real or complex).

    :param dft_size: Size of a discrete Fourier transform.

    :param flags: Optional flags:

            * **DFT_ROWS** Transform each individual row of the source matrix.

            * **DFT_SCALE** Scale the result: divide it by the number of elements in the transform (obtained from  ``dft_size`` ).

            * **DFT_INVERSE** Invert DFT. Use for complex-complex cases (real-complex and complex-real cases are always forward and inverse, respectively).

            * **DFT_REAL_OUTPUT** Specify the output as real. The source matrix is the result of real-complex transform, so the destination matrix must be real.
            

    The source matrix should be continuous, otherwise reallocation and data copying is performed. The function chooses an operation mode depending on the flags, size, and channel count of the source matrix:

    *
        If the source matrix is complex and the output is not specified as real, the destination matrix is complex and has the ``dft_size``    size and ``CV_32FC2``    type. The destination matrix contains a full result of the DFT (forward or inverse).

    *
        If the source matrix is complex and the output is specified as real, the function assumes that its input is the result of the forward transform (see next item). The destionation matrix has the ``dft_size``    size and ``CV_32FC1``    type. It contains the result of the inverse DFT.

    *
        If the source matrix is real (its type is ``CV_32FC1``    ), forward DFT is performed. The result of the DFT is packed into complex ( ``CV_32FC2``    ) matrix. So, the width of the destination matrix is ``dft_size.width / 2 + 1``    . But if the source is a single column, the height is reduced instead of the width.

See Also:
:ocv:func:`dft` 

.. index:: gpu::convolve

gpu::convolve
-----------------
.. ocv:function:: void gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr=false)

.. ocv:function:: void gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr, ConvolveBuf& buf)

    Computes a convolution (or cross-correlation) of two images.

    :param image: Source image. Only  ``CV_32FC1`` images are supported for now.

    :param templ: Template image. The size is not greater than the  ``image`` size. The type is the same as  ``image`` .

    :param result: Result image. The size and type is the same as  ``image`` .

    :param ccorr: Flags to evaluate cross-correlation instead of convolution.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. index:: gpu::ConvolveBuf

.. _gpu::ConvolveBuf:

gpu::ConvolveBuf
----------------
.. ocv:class:: gpu::ConvolveBuf

This class provides a memory buffer for the
    :ocv:func:`gpu::convolve` function. 
::

    struct CV_EXPORTS ConvolveBuf
    {
        ConvolveBuf() {}
        ConvolveBuf(Size image_size, Size templ_size)
            { create(image_size, templ_size); }
        void create(Size image_size, Size templ_size);

    private:
        // Hidden
    };


.. index:: gpu::ConvolveBuf::ConvolveBuf

gpu::ConvolveBuf::ConvolveBuf
---------------------------------
.. ocv:function:: ConvolveBuf::ConvolveBuf()

    Constructs an empty buffer that is properly resized after the first call of the 
    :ocv:func:`convolve` function.

.. ocv:function:: ConvolveBuf::ConvolveBuf(Size image_size, Size templ_size)

    Constructs a buffer for the 
    :ocv:func:`convolve` function with respective arguments.

.. index:: gpu::matchTemplate

gpu::matchTemplate
----------------------
.. ocv:function:: void gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method)

    Computes a proximity map for a raster template and an image where the template is searched for.

    :param image: Source image.  ``CV_32F`` and  ``CV_8U`` depth images (1..4 channels) are supported for now.

    :param templ: Template image with the size and type the same as  ``image`` .

    :param result: Map containing comparison results ( ``CV_32FC1`` ). If  ``image`` is  *W x H*  and ``templ`` is  *w x h*, then  ``result`` must be *W-w+1 x H-h+1*.

    :param method: Specifies the way to compare the template with the image.

    The following methods are supported for the ``CV_8U`` depth images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_SQDIFF_NORMED``
    * ``CV_TM_CCORR``
    * ``CV_TM_CCORR_NORMED``
    * ``CV_TM_CCOEFF``
    * ``CV_TM_CCOEFF_NORMED``

    The following methods are supported for the ``CV_32F`` images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_CCORR``

See Also:
:ocv:func:`matchTemplate` 

.. index:: gpu::remap

gpu::remap
--------------
.. ocv:function:: void gpu::remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap)

    Applies a generic geometrical transformation to an image.

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_8UC3`` source types are supported.

    :param dst: Destination image with the size the same as  ``xmap`` and the type the same as  ``src`` .

    :param xmap: X values. Only  ``CV_32FC1`` type is supported.

    :param ymap: Y values. Only  ``CV_32FC1`` type is supported.

    The function transforms the source image using the specified map:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} (xmap(x,y), ymap(x,y))

    Values of pixels with non-integer coordinates are computed using bilinear the interpolation.

See Also: :ocv:func:`remap` 

.. index:: gpu::cvtColor

gpu::cvtColor
-----------------
.. ocv:function:: void gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn = 0)

.. ocv:function:: void gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn, const Stream& stream)

    Converts an image from one color space to another.

    :param src: Source image with  ``CV_8U``, ``CV_16U``, or  ``CV_32F`` depth and 1, 3, or 4 channels.

    :param dst: Destination image with the same size and depth as  ``src`` .

    :param code: Color space conversion code. For details, see  :ocv:func:`cvtColor` . Conversion to/from Luv and Bayer color spaces is not supported.

    :param dcn: Number of channels in the destination image. If the parameter is 0, the number of the channels is derived automatically from  ``src`` and the  ``code`` .

    :param stream: Stream for the asynchronous version.

    3-channel color spaces (like ``HSV``, ``XYZ``, and so on) can be stored in a 4-channel image for better perfomance.

See Also:
:ocv:func:`cvtColor` 

.. index:: gpu::threshold

gpu::threshold
------------------
.. ocv:function:: double gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double maxval, int type)

.. ocv:function:: double gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double maxval, int type, const Stream& stream)

    Applies a fixed-level threshold to each array element.

    :param src: Source array (single-channel). ``CV_64F`` depth is not supported.

    :param dst: Destination array with the same size and type as  ``src`` .

    :param thresh: Threshold value.

    :param maxVal: Maximum value to use with  ``THRESH_BINARY`` and  ``THRESH_BINARY_INV`` threshold types.

    :param thresholdType: Threshold type. For details, see  :ocv:func:`threshold` . The ``THRESH_OTSU`` threshold type is not supported.

    :param stream: Stream for the asynchronous version.

See Also:
:ocv:func:`threshold` 

.. index:: gpu::resize

gpu::resize
---------------
.. ocv:function:: void gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx=0, double fy=0, int interpolation = INTER_LINEAR)

    Resizes an image.

    :param src: Source image.  ``CV_8UC1`` and  ``CV_8UC4`` types are supported.

    :param dst: Destination image  with the same type as  ``src`` . The size is ``dsize`` (when it is non-zero) or the size is computed from  ``src.size()``, ``fx``, and  ``fy`` .

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

    :param interpolation: Interpolation method. Only  ``INTER_NEAREST`` and  ``INTER_LINEAR`` are supported.

See Also: :ocv:func:`resize` 

.. index:: gpu::warpAffine

gpu::warpAffine
-------------------
.. ocv:function:: void gpu::warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR)

    Applies an affine transformation to an image.

    :param src: Source image.  ``CV_8U``, ``CV_16U``, ``CV_32S``, or  ``CV_32F`` depth and 1, 3, or 4 channels are supported.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` . 

    :param M: *2x3*  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize`) and the optional flag  ``WARP_INVERSE_MAP`` specifying that  ``M`` is an inverse transformation (``dst=>src``). Only ``INTER_NEAREST``, ``INTER_LINEAR``, and  ``INTER_CUBIC`` interpolation methods are supported.

See Also:
:ocv:func:`warpAffine` 

.. index:: gpu::warpPerspective

gpu::warpPerspective
------------------------
.. ocv:function:: void gpu::warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR)

    Applies a perspective transformation to an image.

    :param src: Source image. Supports  ``CV_8U``, ``CV_16U``, ``CV_32S``, or  ``CV_32F`` depth and 1, 3, or 4 channels.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` . 

    :param M: *3x3* transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize` ) and the optional flag  ``WARP_INVERSE_MAP`` specifying that  ``M`` is the inverse transformation (``dst => src``). Only  ``INTER_NEAREST``, ``INTER_LINEAR``, and  ``INTER_CUBIC`` interpolation methods are supported.

See Also:
:ocv:func:`warpPerspective` 

.. index:: gpu::rotate

gpu::rotate
---------------
.. ocv:function:: void gpu::rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift = 0, double yShift = 0, int interpolation = INTER_LINEAR)

    Rotates an image around the origin (0,0) and then shifts it.

    :param src: Source image.  ``CV_8UC1`` and  ``CV_8UC4`` types are supported.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` . 

    :param dsize: Size of the destination image.

    :param angle: Angle of rotation in degrees.

    :param xShift: Shift along the horizontal axis.

    :param yShift: Shift along the vertical axis.

    :param interpolation: Interpolation method. Only  ``INTER_NEAREST``, ``INTER_LINEAR``, and  ``INTER_CUBIC`` are supported.

See Also:
:ocv:func:`gpu::warpAffine` 

.. index:: gpu::copyMakeBorder

gpu::copyMakeBorder
-----------------------
.. ocv:function:: void gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value = Scalar())

    Copies a 2D array to a larger destination array and pads borders with the given constant.

    :param src: Source image. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1``, and  ``CV_32FC1`` types are supported.

    :param dst: Destination image with the same type as  ``src``. The size is  ``Size(src.cols+left+right, src.rows+top+bottom)`` .

    :param top, bottom, left, right: Number of pixels in each direction from the source image rectangle to extrapolate. For example:  ``top=1, bottom=1, left=1, right=1`` mean that 1 pixel-wide border needs to be built.

    :param value: Border value.

See Also:
:ocv:func:`copyMakeBorder`

.. index:: gpu::rectStdDev

gpu::rectStdDev
-------------------
.. ocv:function:: void gpu::rectStdDev(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, const Rect& rect)

    Computes a standard deviation of integral images.

    :param src: Source image. Only the ``CV_32SC1`` type is supported.

    :param sqr: Squared source image. Only  the ``CV_32FC1`` type is supported.

    :param dst: Destination image with the same type and size as  ``src`` .

    :param rect: Rectangular window.

.. index:: gpu::evenLevels

gpu::evenLevels
-------------------
.. ocv:function:: void gpu::evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel)

    Computes levels with even distribution.

    :param levels: Destination array.  ``levels`` has 1 row, ``nLevels`` columns, and the ``CV_32SC1`` type.

    :param nLevels: Number of computed levels.  ``nLevels`` must be at least 2.

    :param lowerLevel: Lower boundary value of the lowest level.

    :param upperLevel: Upper boundary value of the greatest level.

.. index:: gpu::histEven

gpu::histEven
-----------------
.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel)

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4])

    Calculates a histogram with evenly distributed bins.

    :param src: Source image. ``CV_8U``, ``CV_16U``, or ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``histSize`` columns, and the ``CV_32S`` type.

    :param histSize: Size of the histogram.

    :param lowerLevel: Lower boundary of lowest-level bin.

    :param upperLevel: Upper boundary of highest-level bin.

.. index:: gpu::histRange

gpu::histRange
------------------
.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels)

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4])

    Calculates a histogram with bins determined by the `levels` array.

    :param src: Source image. ``CV_8U``, ``CV_16U``, or  ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``(levels.cols-1)`` columns, and the  ``CV_32SC1`` type.

    :param levels: Number of levels in the histogram.

