Arithm Operations on Matrices
=============================

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
