Operations on Matrices
======================

.. highlight:: cpp



gpu::gemm
------------------
Performs generalized matrix multiplication.

.. ocv:function:: void gpu::gemm(const GpuMat& src1, const GpuMat& src2, double alpha, const GpuMat& src3, double beta, GpuMat& dst, int flags = 0, Stream& stream = Stream::Null())

    :param src1: First multiplied input matrix that should have  ``CV_32FC1`` , ``CV_64FC1`` , ``CV_32FC2`` , or  ``CV_64FC2``  type.

    :param src2: Second multiplied input matrix of the same type as  ``src1`` .

    :param alpha: Weight of the matrix product.

    :param src3: Third optional delta matrix added to the matrix product. It should have the same type as  ``src1``  and  ``src2`` .

    :param beta: Weight of  ``src3`` .

    :param dst: Destination matrix. It has the proper size and the same type as input matrices.

    :param flags: Operation flags:

            * **GEMM_1_T** transpose  ``src1``
            * **GEMM_2_T** transpose  ``src2``
            * **GEMM_3_T** transpose  ``src3``

    :param stream: Stream for the asynchronous version.

The function performs generalized matrix multiplication similar to the ``gemm`` functions in BLAS level 3. For example, ``gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)`` corresponds to

.. math::

    \texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T

.. note:: Transposition operation doesn't support  ``CV_64FC2``  input type.

.. seealso:: :ocv:func:`gemm`



gpu::transpose
------------------
Transposes a matrix.

.. ocv:function:: void gpu::transpose( const GpuMat& src1, GpuMat& dst, Stream& stream=Stream::Null() )

    :param src1: Source matrix. 1-, 4-, 8-byte element sizes are supported for now (CV_8UC1, CV_8UC4, CV_16UC2, CV_32FC1, etc).

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`transpose`



gpu::flip
-------------
Flips a 2D matrix around vertical, horizontal, or both axes.

.. ocv:function:: void gpu::flip( const GpuMat& a, GpuMat& b, int flipCode, Stream& stream=Stream::Null() )

    :param a: Source matrix. Supports 1, 3 and 4 channels images with ``CV_8U``, ``CV_16U``, ``CV_32S`` or ``CV_32F`` depth.

    :param b: Destination matrix.

    :param flipCode: Flip mode for the source:

        * ``0`` Flips around x-axis.

        * ``>0`` Flips around y-axis.

        * ``<0`` Flips around both axes.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`flip`



gpu::LUT
------------
Transforms the source matrix into the destination matrix using the given look-up table: ``dst(I) = lut(src(I))``

.. ocv:function:: void gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix.  ``CV_8UC1``  and  ``CV_8UC3``  matrices are supported for now.

    :param lut: Look-up table of 256 elements. It is a continuous ``CV_8U`` matrix.

    :param dst: Destination matrix with the same depth as  ``lut``  and the same number of channels as  ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`LUT`



gpu::merge
--------------
Makes a multi-channel matrix out of several single-channel matrices.

.. ocv:function:: void gpu::merge(const GpuMat* src, size_t n, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::merge(const vector<GpuMat>& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Array/vector of source matrices.

    :param n: Number of source matrices.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`merge`



gpu::split
--------------
Copies each plane of a multi-channel matrix into an array.

.. ocv:function:: void gpu::split(const GpuMat& src, GpuMat* dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::split(const GpuMat& src, vector<GpuMat>& dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination array/vector of single-channel matrices.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`split`



gpu::magnitude
------------------
Computes magnitudes of complex matrix elements.

.. ocv:function:: void gpu::magnitude( const GpuMat& xy, GpuMat& magnitude, Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, Stream& stream = Stream::Null())

    :param xy: Source complex matrix in the interleaved format ( ``CV_32FC2`` ).

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnitudes ( ``CV_32FC1`` ).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`magnitude`



gpu::magnitudeSqr
---------------------
Computes squared magnitudes of complex matrix elements.

.. ocv:function:: void gpu::magnitudeSqr( const GpuMat& xy, GpuMat& magnitude, Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, Stream& stream = Stream::Null())

    :param xy: Source complex matrix in the interleaved format ( ``CV_32FC2`` ).

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnitude squares ( ``CV_32FC1`` ).

    :param stream: Stream for the asynchronous version.



gpu::phase
--------------
Computes polar angles of complex matrix elements.

.. ocv:function:: void gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees=false, Stream& stream = Stream::Null())

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param angle: Destination matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag for angles that must be evaluated in degrees.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`phase`



gpu::cartToPolar
--------------------
Converts Cartesian coordinates into polar.

.. ocv:function:: void gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, GpuMat& angle, bool angleInDegrees=false, Stream& stream = Stream::Null())

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnitudes ( ``CV_32FC1`` ).

    :param angle: Destination matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag for angles that must be evaluated in degrees.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`cartToPolar`



gpu::polarToCart
--------------------
Converts polar coordinates into Cartesian.

.. ocv:function:: void gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees=false, Stream& stream = Stream::Null())

    :param magnitude: Source matrix containing magnitudes ( ``CV_32FC1`` ).

    :param angle: Source matrix containing angles ( ``CV_32FC1`` ).

    :param x: Destination matrix of real components ( ``CV_32FC1`` ).

    :param y: Destination matrix of imaginary components ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag that indicates angles in degrees.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`polarToCart`



gpu::normalize
--------------
Normalizes the norm or value range of an array.

.. ocv:function:: void gpu::normalize(const GpuMat& src, GpuMat& dst, double alpha = 1, double beta = 0, int norm_type = NORM_L2, int dtype = -1, const GpuMat& mask = GpuMat())

.. ocv:function:: void gpu::normalize(const GpuMat& src, GpuMat& dst, double a, double b, int norm_type, int dtype, const GpuMat& mask, GpuMat& norm_buf, GpuMat& cvt_buf)

    :param src: input array.

    :param dst: output array of the same size as  ``src`` .

    :param alpha: norm value to normalize to or the lower range boundary in case of the range normalization.

    :param beta: upper range boundary in case of the range normalization; it is not used for the norm normalization.

    :param normType: normalization type (see the details below).

    :param dtype: when negative, the output array has the same type as ``src``; otherwise, it has the same number of channels as  ``src`` and the depth ``=CV_MAT_DEPTH(dtype)``.

    :param mask: optional operation mask.

    :param norm_buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

    :param cvt_buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`normalize`



gpu::mulSpectrums
---------------------
Performs a per-element multiplication of two Fourier spectrums.

.. ocv:function:: void gpu::mulSpectrums( const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, bool conjB=false, Stream& stream=Stream::Null() )

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

.. ocv:function:: void gpu::mulAndScaleSpectrums( const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, float scale, bool conjB=false, Stream& stream=Stream::Null() )

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

.. ocv:function:: void gpu::dft( const GpuMat& src, GpuMat& dst, Size dft_size, int flags=0, Stream& stream=Stream::Null() )

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
.. ocv:function:: gpu::ConvolveBuf::create(Size image_size, Size templ_size)

Constructs a buffer for :ocv:func:`gpu::convolve` function with respective arguments.


gpu::convolve
-----------------
Computes a convolution (or cross-correlation) of two images.

.. ocv:function:: void gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr=false)

.. ocv:function:: void gpu::convolve( const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr, ConvolveBuf& buf, Stream& stream=Stream::Null() )

    :param image: Source image. Only  ``CV_32FC1`` images are supported for now.

    :param templ: Template image. The size is not greater than the  ``image`` size. The type is the same as  ``image`` .

    :param result: Result image. If  ``image`` is  *W x H*  and ``templ`` is  *w x h*, then  ``result`` must be *W-w+1 x H-h+1*.

    :param ccorr: Flags to evaluate cross-correlation instead of convolution.

    :param buf: Optional buffer to avoid extra memory allocations and to adjust some specific parameters. See :ocv:struct:`gpu::ConvolveBuf`.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::filter2D`



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
