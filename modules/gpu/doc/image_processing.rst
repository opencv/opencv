Image Processing
================

.. highlight:: cpp

.. index:: gpu::meanShiftFiltering

cv::gpu::meanShiftFiltering
---------------------------
.. c:function:: void gpu::meanShiftFiltering(const GpuMat\& src, GpuMat\& dst,
   int sp, int sr,
   TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER
   + TermCriteria::EPS, 5, 1))

    Performs mean-shift filtering for each point of the source image. It maps each point of the source image into another point, and as the result we have new color and new position of each point.

    :param src: Source image. Only  ``CV_8UC4``  images are supported for now.

    :param dst: Destination image, containing color of mapped points. Will have the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See  .

.. index:: gpu::meanShiftProc

cv::gpu::meanShiftProc
----------------------
.. c:function:: void gpu::meanShiftProc(const GpuMat\& src, GpuMat\& dstr, GpuMat\& dstsp,
   int sp, int sr,
   TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER
   + TermCriteria::EPS, 5, 1))

    Performs mean-shift procedure and stores information about processed points (i.e. their colors and positions) into two images.

    :param src: Source image. Only  ``CV_8UC4``  images are supported for now.

    :param dstr: Destination image, containing color of mapped points. Will have the same size and type as  ``src`` .

    :param dstsp: Destination image, containing position of mapped points. Will have the same size as  ``src``  and  ``CV_16SC2``  type.

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See  .

See also:
:func:`gpu::meanShiftFiltering` .

.. index:: gpu::meanShiftSegmentation

cv::gpu::meanShiftSegmentation
------------------------------
.. c:function:: void gpu::meanShiftSegmentation(const GpuMat\& src, Mat\& dst,
   int sp, int sr, int minsize,
   TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER
   + TermCriteria::EPS, 5, 1))

    Performs mean-shift segmentation of the source image and eleminates small segments.

    :param src: Source image. Only  ``CV_8UC4``  images are supported for now.

    :param dst: Segmented image. Will have the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param minsize: Minimum segment size. Smaller segements will be merged.

    :param criteria: Termination criteria. See  .

.. index:: gpu::integral

cv::gpu::integral
-----------------
.. c:function:: void gpu::integral(const GpuMat\& src, GpuMat\& sum)

.. c:function:: void gpu::integral(const GpuMat\& src, GpuMat\& sum, GpuMat\& sqsum)

    Computes integral image and squared integral image.

    :param src: Source image. Only  ``CV_8UC1``  images are supported for now.

    :param sum: Integral image. Will contain 32-bit unsigned integer values packed into  ``CV_32SC1`` .

    :param sqsum: Squared integral image. Will have  ``CV_32FC1``  type.

See also:
:func:`integral` .

.. index:: gpu::sqrIntegral

cv::gpu::sqrIntegral
--------------------
.. c:function:: void gpu::sqrIntegral(const GpuMat\& src, GpuMat\& sqsum)

    Computes squared integral image.

    :param src: Source image. Only  ``CV_8UC1``  images are supported for now.

    :param sqsum: Squared integral image. Will contain 64-bit unsigned integer values packed into  ``CV_64FC1`` .

.. index:: gpu::columnSum

cv::gpu::columnSum
------------------
.. c:function:: void gpu::columnSum(const GpuMat\& src, GpuMat\& sum)

    Computes vertical (column) sum.

    :param src: Source image. Only  ``CV_32FC1``  images are supported for now.

    :param sum: Destination image. Will have  ``CV_32FC1``  type.

.. index:: gpu::cornerHarris

cv::gpu::cornerHarris
---------------------
.. c:function:: void gpu::cornerHarris(const GpuMat\& src, GpuMat\& dst,
   int blockSize, int ksize, double k,
   int borderType=BORDER_REFLECT101)

    Computes Harris cornerness criteria at each image pixel.

    :param src: Source image. Only  ``CV_8UC1``  and  ``CV_32FC1``  images are supported for now.

    :param dst: Destination image. Will have the same size and  ``CV_32FC1``  type and contain cornerness values.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101``  and  ``BORDER_REPLICATE``  are supported for now.

See also:
:func:`cornerHarris` .

.. index:: gpu::cornerMinEigenVal

cv::gpu::cornerMinEigenVal
--------------------------
.. c:function:: void gpu::cornerMinEigenVal(const GpuMat\& src, GpuMat\& dst,
   int blockSize, int ksize,
   int borderType=BORDER_REFLECT101)

    Computes minimum eigen value of 2x2 derivative covariation matrix at each pixel - the cornerness criteria.

    :param src: Source image. Only  ``CV_8UC1``  and  ``CV_32FC1``  images are supported for now.

    :param dst: Destination image. Will have the same size and  ``CV_32FC1``  type and contain cornerness values.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101``  and  ``BORDER_REPLICATE``  are supported for now.

See also:
:func:`cornerMinEigenValue` .

.. index:: gpu::mulSpectrums

cv::gpu::mulSpectrums
---------------------
.. c:function:: void gpu::mulSpectrums(const GpuMat\& a, const GpuMat\& b,
   GpuMat\& c, int flags, bool conjB=false)

    Performs per-element multiplication of two Fourier spectrums.

    :param a: First spectrum.

    :param b: Second spectrum. Must have the same size and type as  ``a`` .

    :param c: Destination spectrum.

    :param flags: Mock paramter is kept for CPU/GPU interfaces similarity.

    :param conjB: Optional flag which indicates the second spectrum must be conjugated before the multiplication.

Only full (i.e. not packed) ``CV_32FC2`` complex spectrums in the interleaved format are supported for now.

See also:
:func:`mulSpectrums` .

.. index:: gpu::mulAndScaleSpectrums

cv::gpu::mulAndScaleSpectrums
-----------------------------
.. c:function:: void gpu::mulAndScaleSpectrums(const GpuMat\& a, const GpuMat\& b,
   GpuMat\& c, int flags, float scale, bool conjB=false)

    Performs per-element multiplication of two Fourier spectrums and scales the result.

    :param a: First spectrum.

    :param b: Second spectrum. Must have the same size and type as  ``a`` .

    :param c: Destination spectrum.

    :param flags: Mock paramter is kept for CPU/GPU interfaces similarity.

    :param scale: Scale constant.

    :param conjB: Optional flag which indicates the second spectrum must be conjugated before the multiplication.

Only full (i.e. not packed) ``CV_32FC2`` complex spectrums in the interleaved format are supported for now.

See also:
:func:`mulSpectrums` .

.. index:: gpu::dft

cv::gpu::dft
------------
.. c:function:: void gpu::dft(const GpuMat\& src, GpuMat\& dst, Size dft_size, int flags=0)

    Performs a forward or inverse discrete Fourier transform (1D or 2D) of floating point matrix. Can handle real matrices (CV32FC1) and complex matrices in the interleaved format (CV32FC2).

    :param src: Source matrix (real or complex).

    :param dst: Destination matrix (real or complex).

    :param dft_size: Size of discrete Fourier transform.

    :param flags: Optional flags:

            * **DFT_ROWS** Transform each individual row of the source matrix.

            * **DFT_SCALE** Scale the result: divide it by the number of elements in the transform (it's obtained from  ``dft_size`` ).

                * **DFT_INVERSE** Inverse DFT must be perfromed for complex-complex case (real-complex and complex-real cases are respectively forward and inverse always).

            * **DFT_REAL_OUTPUT** The source matrix is the result of real-complex transform, so the destination matrix must be real.
            

The source matrix should be continuous, otherwise reallocation and data copying will be performed. Function chooses the operation mode depending on the flags, size and channel count of the source matrix:

*
    If the source matrix is complex and the output isn't specified as real then the destination matrix will be complex, will have ``dft_size``     size and ``CV_32FC2``     type. It will contain full result of the DFT (forward or inverse).

*
    If the source matrix is complex and the output is specified as real then function assumes that its input is the result of the forward transform (see next item). The destionation matrix will have ``dft_size``     size and ``CV_32FC1``     type. It will contain result of the inverse DFT.

*
    If the source matrix is real (i.e. its type is ``CV_32FC1``     ) then forward DFT will be performed. The result of the DFT will be packed into complex ( ``CV_32FC2``     ) matrix so its width will be ``dft_size.width / 2 + 1``     , but if the source is a single column then height will be reduced instead of width.

See also:
:func:`dft` .

.. index:: gpu::convolve

cv::gpu::convolve
-----------------
.. c:function:: void gpu::convolve(const GpuMat\& image, const GpuMat\& templ, GpuMat\& result,
   bool ccorr=false)

.. c:function:: void gpu::convolve(const GpuMat\& image, const GpuMat\& templ, GpuMat\& result,
   bool ccorr, ConvolveBuf\& buf)

    Computes convolution (or cross-correlation) of two images.

    :param image: Source image. Only  ``CV_32FC1``  images are supported for now.

    :param templ: Template image. Must have size not greater then  ``image``  size and be the same type as  ``image`` .

    :param result: Result image. Will have the same size and type as  ``image`` .

    :param ccorr: Flags which indicates cross-correlation must be evaluated instead of convolution.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. index:: gpu::ConvolveBuf

.. _gpu::ConvolveBuf:

gpu::ConvolveBuf
----------------
.. c:type:: gpu::ConvolveBuf

Memory buffer for the
:func:`gpu::convolve` function. ::

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

cv::gpu::ConvolveBuf::ConvolveBuf
---------------------------------
.. c:function:: ConvolveBuf::ConvolveBuf()

Constructs an empty buffer which will be properly resized after first call of the convolve function.

.. c:function:: ConvolveBuf::ConvolveBuf(Size image_size, Size templ_size)

Constructs a buffer for the convolve function with respectively arguments.

.. index:: gpu::matchTemplate

cv::gpu::matchTemplate
----------------------
.. c:function:: void gpu::matchTemplate(const GpuMat\& image, const GpuMat\& templ,
   GpuMat\& result, int method)

    Computes a proximity map for a raster template and an image where the template is searched for.

    :param image: Source image.  ``CV_32F``  and  ``CV_8U``  depth images (1..4 channels) are supported for now.

    :param templ: Template image. Must have the same size and type as  ``image`` .

    :param result: Map containing comparison results ( ``CV_32FC1`` ). If  ``image``  is  :math:`W \times H`  and ``templ``  is  :math:`w \times h`  then  ``result``  must be  :math:`(W-w+1) \times (H-h+1)` .

    :param method: Specifies the way which the template must be compared with the image.

Following methods are supported for the ``CV_8U`` depth images for now:

 * CV_TM_SQDIFF
 * CV_TM_SQDIFF_NORMED
 * CV_TM_CCORR
 * CV_TM_CCORR_NORMED
 * CV_TM_CCOEFF
 * CV_TM_CCOEFF_NORMED

Following methods are supported for the ``CV_32F`` images for now:

 * CV_TM_SQDIFF
 * CV_TM_CCORR

See also:
:func:`matchTemplate` .

.. index:: gpu::remap

cv::gpu::remap
--------------
.. c:function:: void gpu::remap(const GpuMat\& src, GpuMat\& dst,  const GpuMat\& xmap, const GpuMat\& ymap)

    Applies a generic geometrical transformation to an image.

    :param src: Source image. Only  ``CV_8UC1``  and  ``CV_8UC3``  source types are supported.

    :param dst: Destination image. It will have the same size as  ``xmap``  and the same type as  ``src`` .

    :param xmap: X values. Only  ``CV_32FC1``  type is supported.

    :param ymap: Y values. Only  ``CV_32FC1``  type is supported.

The function transforms the source image using the specified map:

.. math::

    \texttt{dst} (x,y) =  \texttt{src} (xmap(x,y), ymap(x,y))

Values of pixels with non-integer coordinates are computed using bilinear interpolation.

See also:
:func:`remap` .

.. index:: gpu::cvtColor

cv::gpu::cvtColor
-----------------
.. c:function:: void gpu::cvtColor(const GpuMat\& src, GpuMat\& dst, int code, int dcn = 0)

.. c:function:: void gpu::cvtColor(const GpuMat\& src, GpuMat\& dst, int code, int dcn,  const Stream\& stream)

    Converts image from one color space to another.

    :param src: Source image with  ``CV_8U`` ,  ``CV_16U``  or  ``CV_32F``  depth and 1, 3 or 4 channels.

    :param dst: Destination image; will have the same size and the same depth as  ``src`` .

    :param code: Color space conversion code. For details see  :func:`cvtColor` . Conversion to/from Luv and Bayer color spaces doesn't supported.

    :param dcn: Number of channels in the destination image; if the parameter is 0, the number of the channels will be derived automatically from  ``src``  and the  ``code`` .

    :param stream: Stream for the asynchronous version.

3-channel color spaces (like ``HSV``,``XYZ`` , etc) can be stored to 4-channel image for better perfomance.

See also:
:func:`cvtColor` .

.. index:: gpu::threshold

cv::gpu::threshold
------------------
.. c:function:: double gpu::threshold(const GpuMat\& src, GpuMat\& dst, double thresh,  double maxval, int type)

.. c:function:: double gpu::threshold(const GpuMat\& src, GpuMat\& dst, double thresh,  double maxval, int type, const Stream\& stream)

    Applies a fixed-level threshold to each array element.

    :param src: Source array (single-channel,  ``CV_64F``  depth isn't supported).

    :param dst: Destination array; will have the same size and the same type as  ``src`` .

    :param thresh: Threshold value.

    :param maxVal: Maximum value to use with  ``THRESH_BINARY``  and  ``THRESH_BINARY_INV``  thresholding types.

    :param thresholdType: Thresholding type. For details see  :func:`threshold` .  ``THRESH_OTSU``  thresholding type doesn't supported.

    :param stream: Stream for the asynchronous version.

See also:
:func:`threshold` .

.. index:: gpu::resize

cv::gpu::resize
---------------
.. c:function:: void gpu::resize(const GpuMat\& src, GpuMat\& dst, Size dsize,  double fx=0, double fy=0,  int interpolation = INTER_LINEAR)

    Resizes an image.

    :param src: Source image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  types.

    :param dst: Destination image. It will have size  ``dsize``  (when it is non-zero) or the size computed from  ``src.size()``  and  ``fx``  and  ``fy`` . The type of  ``dst``  will be the same as of  ``src`` .

    :param dsize: Destination image size. If it is zero, then it is computed as: 

        .. math::

            
 \texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))} 

        Either  ``dsize``  or both  ``fx``  or  ``fy``  must be non-zero.

    :param fx: Scale factor along the horizontal axis. When 0, it is computed as 

        .. math::

            
 \texttt{(double)dsize.width/src.cols} 

    :param fy: Scale factor along the vertical axis. When 0, it is computed as 

        .. math::

            
 \texttt{(double)dsize.height/src.rows} 

    :param interpolation: Interpolation method. Supports only  ``INTER_NEAREST``  and  ``INTER_LINEAR`` .

See also:
:func:`resize` .

.. index:: gpu::warpAffine

cv::gpu::warpAffine
-------------------
.. c:function:: void gpu::warpAffine(const GpuMat\& src, GpuMat\& dst, const Mat\& M,  Size dsize, int flags = INTER_LINEAR)

    Applies an affine transformation to an image.

    :param src: Source image. Supports  ``CV_8U`` ,  ``CV_16U`` ,  ``CV_32S``  or  ``CV_32F``  depth and 1, 3 or 4 channels.

    :param dst: Destination image; will have size  ``dsize``  and the same type as  ``src`` .

    :param M: :math:`2\times 3`  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods, see  :func:`resize` , and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` ). Supports only  ``INTER_NEAREST`` ,  ``INTER_LINEAR``  and  ``INTER_CUBIC``  interpolation methods.

See also:
:func:`warpAffine` .

.. index:: gpu::warpPerspective

cv::gpu::warpPerspective
------------------------
.. c:function:: void gpu::warpPerspective(const GpuMat\& src, GpuMat\& dst, const Mat\& M,  Size dsize, int flags = INTER_LINEAR)

    Applies a perspective transformation to an image.

    :param src: Source image. Supports  ``CV_8U`` ,  ``CV_16U`` ,  ``CV_32S``  or  ``CV_32F``  depth and 1, 3 or 4 channels.

    :param dst: Destination image; will have size  ``dsize``  and the same type as  ``src`` .

    :param M: :math:`2
         3`  transformation matrix.

    :param dsize: Size of the destination image.

    :param flags: Combination of interpolation methods, see  :func:`resize` , and the optional flag  ``WARP_INVERSE_MAP``  that means that  ``M``  is the inverse transformation ( :math:`\texttt{dst}\rightarrow\texttt{src}` ). Supports only  ``INTER_NEAREST`` ,  ``INTER_LINEAR``  and  ``INTER_CUBIC``  interpolation methods.

See also:
:func:`warpPerspective` .

.. index:: gpu::rotate

cv::gpu::rotate
---------------
.. c:function:: void gpu::rotate(const GpuMat\& src, GpuMat\& dst, Size dsize,  double angle, double xShift = 0, double yShift = 0,  int interpolation = INTER_LINEAR)

    Rotates an image around the origin (0,0) and then shifts it.

    :param src: Source image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  types.

    :param dst: Destination image; will have size  ``dsize``  and the same type as  ``src`` .

    :param dsize: Size of the destination image.

    :param angle: Angle of rotation in degrees.

    :param xShift: Shift along horizontal axis.

    :param yShift: Shift along vertical axis.

    :param interpolation: Interpolation method. Supports only  ``INTER_NEAREST`` ,  ``INTER_LINEAR``  and  ``INTER_CUBIC`` .

See also:
:func:`gpu::warpAffine` .

.. index:: gpu::copyMakeBorder

cv::gpu::copyMakeBorder
-----------------------
.. c:function:: void gpu::copyMakeBorder(const GpuMat\& src, GpuMat\& dst,  int top, int bottom, int left, int right,  const Scalar\& value = Scalar())

    Copies 2D array to a larger destination array and pads borders with the given constant.

    :param src: Source image. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_32SC1``  and  ``CV_32FC1``  types.

    :param dst: The destination image; will have the same type as  ``src``  and the size  ``Size(src.cols+left+right, src.rows+top+bottom)`` .

    :param top, bottom, left, right: Specify how much pixels in each direction from the source image rectangle one needs to extrapolate, e.g.  ``top=1, bottom=1, left=1, right=1``  mean that 1 pixel-wide border needs to be built.

    :param value: Border value.

See also:
:func:`copyMakeBorder`
.. index:: gpu::rectStdDev

cv::gpu::rectStdDev
-------------------
.. c:function:: void gpu::rectStdDev(const GpuMat\& src, const GpuMat\& sqr, GpuMat\& dst,  const Rect\& rect)

    Computes standard deviation of integral images.

    :param src: Source image. Supports only  ``CV_32SC1``  type.

    :param sqr: Squared source image. Supports only  ``CV_32FC1``  type.

    :param dst: Destination image; will have the same type and the same size as  ``src`` .

    :param rect: Rectangular window.

.. index:: gpu::evenLevels

cv::gpu::evenLevels
-------------------
.. c:function:: void gpu::evenLevels(GpuMat\& levels, int nLevels,  int lowerLevel, int upperLevel)

    Computes levels with even distribution.

    :param levels: Destination array.  ``levels``  will have 1 row and  ``nLevels``  cols and  ``CV_32SC1``  type.

    :param nLevels: Number of levels being computed.  ``nLevels``  must be at least 2.

    :param lowerLevel: Lower boundary value of the lowest level.

    :param upperLevel: Upper boundary value of the greatest level.

.. index:: gpu::histEven

cv::gpu::histEven
-----------------
.. c:function:: void gpu::histEven(const GpuMat\& src, GpuMat\& hist,  int histSize, int lowerLevel, int upperLevel)

.. c:function:: void gpu::histEven(const GpuMat\& src, GpuMat hist[4],  int histSize[4], int lowerLevel[4], int upperLevel[4])

    Calculates histogram with evenly distributed bins.

    :param src: Source image. Supports  ``CV_8U`` ,  ``CV_16U``  or  ``CV_16S``  depth and 1 or 4 channels. For four-channel image all channels are processed separately.

    :param hist: Destination histogram. Will have one row,  ``histSize``  cols and  ``CV_32S``  type.

    :param histSize: Size of histogram.

    :param lowerLevel: Lower boundary of lowest level bin.

    :param upperLevel: Upper boundary of highest level bin.

.. index:: gpu::histRange

cv::gpu::histRange
------------------
.. c:function:: void gpu::histRange(const GpuMat\& src, GpuMat\& hist, const GpuMat\& levels)

.. c:function:: void gpu::histRange(const GpuMat\& src, GpuMat hist[4],  const GpuMat levels[4])

    Calculates histogram with bins determined by levels array.

    :param src: Source image. Supports  ``CV_8U`` ,  ``CV_16U``  or  ``CV_16S``  depth and 1 or 4 channels. For four-channel image all channels are processed separately.

    :param hist: Destination histogram. Will have one row,  ``(levels.cols-1)``  cols and  ``CV_32SC1``  type.

    :param levels: Number of levels in histogram.

