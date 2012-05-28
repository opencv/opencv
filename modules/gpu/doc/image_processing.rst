Image Processing
================

.. highlight:: cpp



gpu::meanShiftFiltering
---------------------------
Performs mean-shift filtering for each point of the source image.

.. ocv:function:: void gpu::meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr,TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1))

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dst: Destination image containing the color of mapped points. It has the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.

It maps each point of the source image into another point. As a result, you have a new color and new position of each point.



gpu::meanShiftProc
----------------------
Performs a mean-shift procedure and stores information about processed points (their colors and positions) in two images.

.. ocv:function:: void gpu::meanShiftProc(const GpuMat& src, GpuMat& dstr, GpuMat& dstsp, int sp, int sr, TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1))

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dstr: Destination image containing the color of mapped points. The size and type is the same as  ``src`` .

    :param dstsp: Destination image containing the position of mapped points. The size is the same as  ``src`` size. The type is  ``CV_16SC2`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.

.. seealso:: :ocv:func:`gpu::meanShiftFiltering`



gpu::meanShiftSegmentation
------------------------------
Performs a mean-shift segmentation of the source image and eliminates small segments.

.. ocv:function:: void gpu::meanShiftSegmentation(const GpuMat& src, Mat& dst, int sp, int sr, int minsize, TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1))

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dst: Segmented image with the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param minsize: Minimum segment size. Smaller segments are merged.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.



gpu::integral
-----------------
Computes an integral image.

.. ocv:function:: void gpu::integral(const GpuMat& src, GpuMat& sum, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sum: Integral image containing 32-bit unsigned integer values packed into  ``CV_32SC1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`integral`



gpu::sqrIntegral
--------------------
Computes a squared integral image.

.. ocv:function:: void gpu::sqrIntegral(const GpuMat& src, GpuMat& sqsum, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sqsum: Squared integral image containing 64-bit unsigned integer values packed into  ``CV_64FC1`` .

    :param stream: Stream for the asynchronous version.



gpu::columnSum
------------------
Computes a vertical (column) sum.

.. ocv:function:: void gpu::columnSum(const GpuMat& src, GpuMat& sum)

    :param src: Source image. Only  ``CV_32FC1`` images are supported for now.

    :param sum: Destination image of the  ``CV_32FC1`` type.



gpu::cornerHarris
---------------------
Computes the Harris cornerness criteria at each image pixel.

.. ocv:function:: void gpu::cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType=BORDER_REFLECT101)

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. It has the same size as ``src`` and ``CV_32FC1`` type.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101`` and  ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerHarris`



gpu::cornerMinEigenVal
--------------------------
Computes the minimum eigen value of a 2x2 derivative covariation matrix at each pixel (the cornerness criteria).

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType=BORDER_REFLECT101)

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int borderType=BORDER_REFLECT101)

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, int borderType=BORDER_REFLECT101, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. The size is the same. The type is  ``CV_32FC1`` .

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param borderType: Pixel extrapolation method. Only ``BORDER_REFLECT101`` and ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerMinEigenVal`



gpu::mulSpectrums
---------------------
Performs a per-element multiplication of two Fourier spectrums.

.. ocv:function:: void gpu::mulSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, bool conjB=false)

    :param a: First spectrum.

    :param b: Second spectrum with the same size and type as  ``a`` .

    :param c: Destination spectrum.

    :param flags: Mock parameter used for CPU/GPU interfaces similarity.

    :param conjB: Optional flag to specify if the second spectrum needs to be conjugated before the multiplication.

    Only full (not packed) ``CV_32FC2`` complex spectrums in the interleaved format are supported for now.

.. seealso:: :ocv:func:`mulSpectrums`



gpu::mulAndScaleSpectrums
-----------------------------
Performs a per-element multiplication of two Fourier spectrums and scales the result.

.. ocv:function:: void gpu::mulAndScaleSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, float scale, bool conjB=false)

    :param a: First spectrum.

    :param b: Second spectrum with the same size and type as  ``a`` .

    :param c: Destination spectrum.

    :param flags: Mock parameter used for CPU/GPU interfaces similarity.

    :param scale: Scale constant.

    :param conjB: Optional flag to specify if the second spectrum needs to be conjugated before the multiplication.

    Only full (not packed) ``CV_32FC2`` complex spectrums in the interleaved format are supported for now.

.. seealso:: :ocv:func:`mulSpectrums`



gpu::dft
------------
Performs a forward or inverse discrete Fourier transform (1D or 2D) of the floating point matrix.

.. ocv:function:: void gpu::dft(const GpuMat& src, GpuMat& dst, Size dft_size, int flags=0)

    :param src: Source matrix (real or complex).

    :param dst: Destination matrix (real or complex).

    :param dft_size: Size of a discrete Fourier transform.

    :param flags: Optional flags:

        * **DFT_ROWS** transforms each individual row of the source matrix.

        * **DFT_SCALE** scales the result: divide it by the number of elements in the transform (obtained from  ``dft_size`` ).

        * **DFT_INVERSE** inverts DFT. Use for complex-complex cases (real-complex and complex-real cases are always forward and inverse, respectively).

        * **DFT_REAL_OUTPUT** specifies the output as real. The source matrix is the result of real-complex transform, so the destination matrix must be real.

Use to handle real matrices ( ``CV32FC1`` ) and complex matrices in the interleaved format ( ``CV32FC2`` ).

The source matrix should be continuous, otherwise reallocation and data copying is performed. The function chooses an operation mode depending on the flags, size, and channel count of the source matrix:

    * If the source matrix is complex and the output is not specified as real, the destination matrix is complex and has the ``dft_size``    size and ``CV_32FC2``    type. The destination matrix contains a full result of the DFT (forward or inverse).

    * If the source matrix is complex and the output is specified as real, the function assumes that its input is the result of the forward transform (see the next item). The destination matrix has the ``dft_size`` size and ``CV_32FC1`` type. It contains the result of the inverse DFT.

    * If the source matrix is real (its type is ``CV_32FC1`` ), forward DFT is performed. The result of the DFT is packed into complex ( ``CV_32FC2`` ) matrix. So, the width of the destination matrix is ``dft_size.width / 2 + 1`` . But if the source is a single column, the height is reduced instead of the width.

.. seealso:: :ocv:func:`dft`


gpu::ConvolveBuf
----------------
.. ocv:struct:: gpu::ConvolveBuf

Class providing a memory buffer for :ocv:func:`gpu::convolve` function, plus it allows to adjust some specific parameters. ::

    struct CV_EXPORTS ConvolveBuf
    {
        Size result_size;
        Size block_size;
        Size user_block_size;
        Size dft_size;
        int spect_len;

        GpuMat image_spect, templ_spect, result_spect;
        GpuMat image_block, templ_block, result_data;

        void create(Size image_size, Size templ_size);
        static Size estimateBlockSize(Size result_size, Size templ_size);
    };

You can use field `user_block_size` to set specific block size for :ocv:func:`gpu::convolve` function. If you leave its default value `Size(0,0)` then automatic estimation of block size will be used (which is optimized for speed). By varying `user_block_size` you can reduce memory requirements at the cost of speed.

gpu::ConvolveBuf::create
------------------------
.. ocv:function:: ConvolveBuf::create(Size image_size, Size templ_size)

Constructs a buffer for :ocv:func:`gpu::convolve` function with respective arguments.


gpu::convolve
-----------------
Computes a convolution (or cross-correlation) of two images.

.. ocv:function:: void gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr=false)

.. ocv:function:: void gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr, ConvolveBuf& buf, Stream &stream = Stream::Null())

    :param image: Source image. Only  ``CV_32FC1`` images are supported for now.

    :param templ: Template image. The size is not greater than the  ``image`` size. The type is the same as  ``image`` .

    :param result: Result image. If  ``image`` is  *W x H*  and ``templ`` is  *w x h*, then  ``result`` must be *W-w+1 x H-h+1*.

    :param ccorr: Flags to evaluate cross-correlation instead of convolution.

    :param buf: Optional buffer to avoid extra memory allocations and to adjust some specific parameters. See :ocv:class:`gpu::ConvolveBuf`.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::filter2D`

gpu::MatchTemplateBuf
---------------------
.. ocv:struct:: gpu::MatchTemplateBuf

Class providing memory buffers for :ocv:func:`gpu::matchTemplate` function, plus it allows to adjust some specific parameters. ::

    struct CV_EXPORTS MatchTemplateBuf
    {
        Size user_block_size;
        GpuMat imagef, templf;
        std::vector<GpuMat> images;
        std::vector<GpuMat> image_sums;
        std::vector<GpuMat> image_sqsums;
    };

You can use field `user_block_size` to set specific block size for :ocv:func:`gpu::matchTemplate` function. If you leave its default value `Size(0,0)` then automatic estimation of block size will be used (which is optimized for speed). By varying `user_block_size` you can reduce memory requirements at the cost of speed.

gpu::matchTemplate
----------------------
Computes a proximity map for a raster template and an image where the template is searched for.

.. ocv:function:: void gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, Stream &stream = Stream::Null())

.. ocv:function:: void gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, MatchTemplateBuf &buf, Stream& stream = Stream::Null())

    :param image: Source image.  ``CV_32F`` and  ``CV_8U`` depth images (1..4 channels) are supported for now.

    :param templ: Template image with the size and type the same as  ``image`` .

    :param result: Map containing comparison results ( ``CV_32FC1`` ). If  ``image`` is  *W x H*  and ``templ`` is  *w x h*, then  ``result`` must be *W-w+1 x H-h+1*.

    :param method: Specifies the way to compare the template with the image.

    :param buf: Optional buffer to avoid extra memory allocations and to adjust some specific parameters. See :ocv:class:`gpu::MatchTemplateBuf`.

    :param stream: Stream for the asynchronous version.

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

.. seealso:: :ocv:func:`matchTemplate`


gpu::remap
--------------
Applies a generic geometrical transformation to an image.

.. ocv:function:: void gpu::remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap, int interpolation, int borderMode = BORDER_CONSTANT, const Scalar& borderValue = Scalar(), Stream& stream = Stream::Null())

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



gpu::cvtColor
-----------------
Converts an image from one color space to another.

.. ocv:function:: void gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn = 0, Stream& stream = Stream::Null())

    :param src: Source image with  ``CV_8U`` , ``CV_16U`` , or  ``CV_32F`` depth and 1, 3, or 4 channels.

    :param dst: Destination image with the same size and depth as  ``src`` .

    :param code: Color space conversion code. For details, see  :ocv:func:`cvtColor` . Conversion to/from Luv and Bayer color spaces is not supported.

    :param dcn: Number of channels in the destination image. If the parameter is 0, the number of the channels is derived automatically from  ``src`` and the  ``code`` .

    :param stream: Stream for the asynchronous version.

3-channel color spaces (like ``HSV``, ``XYZ``, and so on) can be stored in a 4-channel image for better performance.

.. seealso:: :ocv:func:`cvtColor`



gpu::swapChannels
-----------------
Exchanges the color channels of an image in-place.

.. ocv:function:: void gpu::swapChannels(GpuMat& image, const int dstOrder[4], Stream& stream = Stream::Null())

    :param image: Source image. Supports only ``CV_8UC4`` type.

    :param dstOrder: Integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR channel order.

    :param stream: Stream for the asynchronous version.

The methods support arbitrary permutations of the original channels, including replication.



gpu::threshold
------------------
Applies a fixed-level threshold to each array element.

.. ocv:function:: double gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double maxval, int type, Stream& stream = Stream::Null())

    :param src: Source array (single-channel).

    :param dst: Destination array with the same size and type as  ``src`` .

    :param thresh: Threshold value.

    :param maxval: Maximum value to use with  ``THRESH_BINARY`` and  ``THRESH_BINARY_INV`` threshold types.

    :param type: Threshold type. For details, see  :ocv:func:`threshold` . The ``THRESH_OTSU`` threshold type is not supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`threshold`



gpu::resize
---------------
Resizes an image.

.. ocv:function:: void gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx=0, double fy=0, int interpolation = INTER_LINEAR, Stream& stream = Stream::Null())

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



gpu::warpAffine
-------------------
Applies an affine transformation to an image.

.. ocv:function:: void gpu::warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR, Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8U`` , ``CV_16U`` , ``CV_32S`` , or  ``CV_32F`` depth and 1, 3, or 4 channels are supported.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` .

    :param M: *2x3*  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize`) and the optional flag  ``WARP_INVERSE_MAP`` specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ). Only ``INTER_NEAREST`` , ``INTER_LINEAR`` , and  ``INTER_CUBIC`` interpolation methods are supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`warpAffine`



gpu::buildWarpAffineMaps
------------------------
Builds transformation maps for affine transformation.

.. ocv:function:: void buildWarpAffineMaps(const Mat& M, bool inverse, Size dsize, GpuMat& xmap, GpuMat& ymap, Stream& stream = Stream::Null())

    :param M: *2x3*  transformation matrix.

    :param inverse: Flag  specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ).

    :param dsize: Size of the destination image.

    :param xmap: X values with  ``CV_32FC1`` type.

    :param ymap: Y values with  ``CV_32FC1`` type.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::warpAffine` , :ocv:func:`gpu::remap`



gpu::warpPerspective
------------------------
Applies a perspective transformation to an image.

.. ocv:function:: void gpu::warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U`` , ``CV_16U`` , ``CV_32S`` , or  ``CV_32F`` depth and 1, 3, or 4 channels are supported.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` .

    :param M: *3x3* transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods (see  :ocv:func:`resize` ) and the optional flag  ``WARP_INVERSE_MAP`` specifying that  ``M`` is the inverse transformation ( ``dst => src`` ). Only  ``INTER_NEAREST`` , ``INTER_LINEAR`` , and  ``INTER_CUBIC`` interpolation methods are supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`warpPerspective`



gpu::buildWarpPerspectiveMaps
-----------------------------
Builds transformation maps for perspective transformation.

.. ocv:function:: void buildWarpAffineMaps(const Mat& M, bool inverse, Size dsize, GpuMat& xmap, GpuMat& ymap, Stream& stream = Stream::Null())

    :param M: *3x3*  transformation matrix.

    :param inverse: Flag  specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ).

    :param dsize: Size of the destination image.

    :param xmap: X values with  ``CV_32FC1`` type.

    :param ymap: Y values with  ``CV_32FC1`` type.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::warpPerspective` , :ocv:func:`gpu::remap`



gpu::rotate
---------------
Rotates an image around the origin (0,0) and then shifts it.

.. ocv:function:: void gpu::rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift = 0, double yShift = 0, int interpolation = INTER_LINEAR, Stream& stream = Stream::Null())

    :param src: Source image. Supports 1, 3 or 4 channels images with ``CV_8U`` , ``CV_16U`` or ``CV_32F`` depth.

    :param dst: Destination image with the same type as  ``src`` . The size is  ``dsize`` .

    :param dsize: Size of the destination image.

    :param angle: Angle of rotation in degrees.

    :param xShift: Shift along the horizontal axis.

    :param yShift: Shift along the vertical axis.

    :param interpolation: Interpolation method. Only  ``INTER_NEAREST`` , ``INTER_LINEAR`` , and  ``INTER_CUBIC`` are supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::warpAffine`



gpu::copyMakeBorder
-----------------------
Forms a border around an image.

.. ocv:function:: void gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, int borderType, const Scalar& value = Scalar(), Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` , and  ``CV_32FC1`` types are supported.

    :param dst: Destination image with the same type as  ``src``. The size is  ``Size(src.cols+left+right, src.rows+top+bottom)`` .

    :param top:

    :param bottom:

    :param left:

    :param right: Number of pixels in each direction from the source image rectangle to extrapolate. For example:  ``top=1, bottom=1, left=1, right=1`` mean that 1 pixel-wide border needs to be built.

    :param borderType: Border type. See  :ocv:func:`borderInterpolate` for details. ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , ``BORDER_CONSTANT`` , ``BORDER_REFLECT`` and ``BORDER_WRAP`` are supported for now.

    :param value: Border value.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`copyMakeBorder`



gpu::rectStdDev
-------------------
Computes a standard deviation of integral images.

.. ocv:function:: void gpu::rectStdDev(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, const Rect& rect, Stream& stream = Stream::Null())

    :param src: Source image. Only the ``CV_32SC1`` type is supported.

    :param sqr: Squared source image. Only  the ``CV_32FC1`` type is supported.

    :param dst: Destination image with the same type and size as  ``src`` .

    :param rect: Rectangular window.

    :param stream: Stream for the asynchronous version.



gpu::evenLevels
-------------------
Computes levels with even distribution.

.. ocv:function:: void gpu::evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel)

    :param levels: Destination array.  ``levels`` has 1 row, ``nLevels`` columns, and the ``CV_32SC1`` type.

    :param nLevels: Number of computed levels.  ``nLevels`` must be at least 2.

    :param lowerLevel: Lower boundary value of the lowest level.

    :param upperLevel: Upper boundary value of the greatest level.



gpu::histEven
-----------------
Calculates a histogram with evenly distributed bins.

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat& hist, GpuMat& buf, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat* hist, int* histSize, int* lowerLevel, int* upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat* hist, GpuMat& buf, int* histSize, int* lowerLevel, int* upperLevel, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U``, ``CV_16U``, or ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``histSize`` columns, and the ``CV_32S`` type.

    :param histSize: Size of the histogram.

    :param lowerLevel: Lower boundary of lowest-level bin.

    :param upperLevel: Upper boundary of highest-level bin.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



gpu::histRange
------------------
Calculates a histogram with bins determined by the ``levels`` array.

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat* hist, const GpuMat* levels, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat* hist, const GpuMat* levels, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U`` , ``CV_16U`` , or  ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``(levels.cols-1)`` columns, and the  ``CV_32SC1`` type.

    :param levels: Number of levels in the histogram.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



gpu::calcHist
------------------
Calculates histogram for one channel 8-bit image.

.. ocv:function:: void gpu::calcHist(const GpuMat& src, GpuMat& hist, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::calcHist(const GpuMat& src, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image.

    :param hist: Destination histogram with one row, 256 columns, and the  ``CV_32SC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



gpu::equalizeHist
------------------
Equalizes the histogram of a grayscale image.

.. ocv:function:: void gpu::equalizeHist(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image.

    :param hist: Destination histogram with one row, 256 columns, and the  ``CV_32SC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`equalizeHist`



gpu::buildWarpPlaneMaps
-----------------------
Builds plane warping maps.

.. ocv:function:: void gpu::buildWarpPlaneMaps(Size src_size, Rect dst_roi, const Mat& R, double f, double s, double dist, GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null())

    :param stream: Stream for the asynchronous version.



gpu::buildWarpCylindricalMaps
-----------------------------
Builds cylindrical warping maps.

.. ocv:function:: void gpu::buildWarpCylindricalMaps(Size src_size, Rect dst_roi, const Mat& R, double f, double s, GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null())

    :param stream: Stream for the asynchronous version.



gpu::buildWarpSphericalMaps
---------------------------
Builds spherical warping maps.

.. ocv:function:: void gpu::buildWarpSphericalMaps(Size src_size, Rect dst_roi, const Mat& R, double f, double s, GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null())

    :param stream: Stream for the asynchronous version.



gpu::pyrDown
-------------------
Smoothes an image and downsamples it.

.. ocv:function:: void gpu::pyrDown(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image. Will have ``Size((src.cols+1)/2, (src.rows+1)/2)`` size and the same type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`pyrDown`



gpu::pyrUp
-------------------
Upsamples an image and then smoothes it.

.. ocv:function:: void gpu::pyrUp(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image. Will have ``Size(src.cols*2, src.rows*2)`` size and the same type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`pyrUp`



gpu::blendLinear
-------------------
Performs linear blending of two images.

.. ocv:function:: void gpu::blendLinear(const GpuMat& img1, const GpuMat& img2, const GpuMat& weights1, const GpuMat& weights2, GpuMat& result, Stream& stream = Stream::Null())

    :param img1: First image. Supports only ``CV_8U`` and ``CV_32F`` depth.

    :param img2: Second image. Must have the same size and the same type as ``img1`` .

    :param weights1: Weights for first image. Must have tha same size as ``img1`` . Supports only ``CV_32F`` type.

    :param weights2: Weights for second image. Must have tha same size as ``img2`` . Supports only ``CV_32F`` type.

    :param result: Destination image.

    :param stream: Stream for the asynchronous version.



gpu::alphaComp
-------------------
Composites two images using alpha opacity values contained in each image.

.. ocv:function:: void gpu::alphaComp(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, int alpha_op, Stream& stream = Stream::Null())

    :param img1: First image. Supports ``CV_8UC4`` , ``CV_16UC4`` , ``CV_32SC4`` and ``CV_32FC4`` types.

    :param img2: Second image. Must have the same size and the same type as ``img1`` .

    :param dst: Destination image.

    :param alpha_op: Flag specifying the alpha-blending operation:

            * **ALPHA_OVER**
            * **ALPHA_IN**
            * **ALPHA_OUT**
            * **ALPHA_ATOP**
            * **ALPHA_XOR**
            * **ALPHA_PLUS**
            * **ALPHA_OVER_PREMUL**
            * **ALPHA_IN_PREMUL**
            * **ALPHA_OUT_PREMUL**
            * **ALPHA_ATOP_PREMUL**
            * **ALPHA_XOR_PREMUL**
            * **ALPHA_PLUS_PREMUL**
            * **ALPHA_PREMUL**

    :param stream: Stream for the asynchronous version.



gpu::Canny
-------------------
Finds edges in an image using the [Canny86]_ algorithm.

.. ocv:function:: void gpu::Canny(const GpuMat& image, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false)

.. ocv:function:: void gpu::Canny(const GpuMat& image, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false)

.. ocv:function:: void gpu::Canny(const GpuMat& dx, const GpuMat& dy, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false)

.. ocv:function:: void gpu::Canny(const GpuMat& dx, const GpuMat& dy, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false)

    :param image: Single-channel 8-bit input image.

    :param dx: First derivative of image in the vertical direction. Support only ``CV_32S`` type.

    :param dy: First derivative of image in the horizontal direction. Support only ``CV_32S`` type.

    :param edges: Output edge map. It has the same size and type as  ``image`` .

    :param low_thresh: First threshold for the hysteresis procedure.

    :param high_thresh: Second threshold for the hysteresis procedure.

    :param apperture_size: Aperture size for the  :ocv:func:`Sobel`  operator.

    :param L2gradient: Flag indicating whether a more accurate  :math:`L_2`  norm  :math:`=\sqrt{(dI/dx)^2 + (dI/dy)^2}`  should be used to compute the image gradient magnitude ( ``L2gradient=true`` ), or a faster default  :math:`L_1`  norm  :math:`=|dI/dx|+|dI/dy|`  is enough ( ``L2gradient=false`` ).

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. seealso:: :ocv:func:`Canny`

