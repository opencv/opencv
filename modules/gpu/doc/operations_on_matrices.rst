Operations on Matrices
======================

.. highlight:: cpp



gpu::gemm
------------------
Performs generalized matrix multiplication.

.. ocv:function:: void gemm(const GpuMat& src1, const GpuMat& src2, double alpha, const GpuMat& src3, double beta, GpuMat& dst, int flags = 0, Stream& stream = Stream::Null())

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

.. seealso:: :ocv:func:`gemm`



gpu::transpose
------------------
Transposes a matrix.

.. ocv:function:: void gpu::transpose(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix. 1-, 4-, 8-byte element sizes are supported for now (CV_8UC1, CV_8UC4, CV_16UC2, CV_32FC1, etc).

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`transpose`



gpu::flip
-------------
Flips a 2D matrix around vertical, horizontal, or both axes.

.. ocv:function:: void gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports 1, 3 and 4 channels images with ``CV_8U``, ``CV_16U``, ``CV_32S`` or ``CV_32F`` depth.

    :param dst: Destination matrix.

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

.. ocv:function:: void gpu::magnitude(const GpuMat& xy, GpuMat& magnitude, Stream& stream = Stream::Null())

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

.. ocv:function:: void gpu::magnitudeSqr(const GpuMat& xy, GpuMat& magnitude, Stream& stream = Stream::Null())

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

    :param angle: Destionation matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag for angles that must be evaluated in degress.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`phase`



gpu::cartToPolar
--------------------
Converts Cartesian coordinates into polar.

.. ocv:function:: void gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, GpuMat& angle, bool angleInDegrees=false, Stream& stream = Stream::Null())

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnitudes ( ``CV_32FC1`` ).

    :param angle: Destionation matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag for angles that must be evaluated in degress.

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

    :param angleInDegrees: Flag that indicates angles in degress.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`polarToCart`
