Per-element Operations
=======================

.. highlight:: cpp



gpu::add
------------
Computes a matrix-matrix or matrix-scalar sum.

.. ocv:function:: void gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::add(const GpuMat& src1, const Scalar& src2, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` , and ``CV_32FC1`` matrices are supported for now. Version with scalar supports only ``CV_32FC1`` and ``CV_32FC2`` matrices.

    :param src2: Second source matrix or a scalar to be added to ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`add`



gpu::subtract
-----------------
Computes a matrix-matrix or matrix-scalar difference.

.. ocv:function:: void gpu::subtract(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::subtract(const GpuMat& src1, const Scalar& src2, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` , and ``CV_32FC1`` matrices are supported for now. Version with scalar supports only ``CV_32FC1`` and ``CV_32FC2`` matrices.

    :param src2: Second source matrix or a scalar to be subtracted from ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`subtract`



gpu::multiply
-----------------
Computes a matrix-matrix or matrix-scalar per-element product.

.. ocv:function:: void gpu::multiply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::multiply(const GpuMat& src1, const Scalar& src2, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` , and ``CV_32FC1`` matrices are supported for now. Version with scalar supports only ``CV_32FC1`` matrices.

    :param src2: Second source matrix or a scalar to be multiplied by ``src1`` elements.

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`multiply`



gpu::divide
---------------
Computes a matrix-matrix or matrix-scalar sum.

.. ocv:function:: void gpu::divide(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::divide(const GpuMat& src1, const Scalar& src2, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` , and ``CV_32FC1`` matrices are supported for now. Version with scalar supports only ``CV_32FC1`` matrices.

    :param src2: Second source matrix or a scalar. The ``src1`` elements are divided by it.

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param stream: Stream for the asynchronous version.

This function, in contrast to :ocv:func:`divide`, uses a round-down rounding mode.

.. seealso:: :ocv:func:`divide`



gpu::exp
------------
Computes an exponent of each matrix element.

.. ocv:function:: void gpu::exp(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix. ``CV_32FC1`` matrixes are supported for now.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`exp`



gpu::pow
------------
Raises every matrix element to a power.

.. ocv:function:: void gpu::pow(const GpuMat& src, double power, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports all type, except ``CV_64F`` depth.

    :param power: Exponent of power.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

The function ``pow`` raises every element of the input matrix to ``p`` :

.. math::

    \texttt{dst} (I) =  \fork{\texttt{src}(I)^p}{if \texttt{p} is integer}{|\texttt{src}(I)|^p}{otherwise}

.. seealso:: :ocv:func:`pow`



gpu::log
------------
Computes a natural logarithm of absolute value of each matrix element.

.. ocv:function:: void gpu::log(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix. ``CV_32FC1`` matrixes are supported for now.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`log`



gpu::absdiff
----------------
Computes per-element absolute difference of two matrices (or of a matrix and scalar).

.. ocv:function:: void gpu::absdiff(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::absdiff(const GpuMat& src1, const Scalar& src2, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` and ``CV_32FC1`` matrices are supported for now. Version with scalar supports only ``CV_32FC1`` matrices.

    :param src2: Second source matrix or a scalar to be added to ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`absdiff`



gpu::compare
----------------
Compares elements of two matrices.

.. ocv:function:: void gpu::compare(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int cmpop, Stream& stream = Stream::Null())

    :param src1: First source matrix. ``CV_8UC4`` and ``CV_32FC1`` matrices are supported for now.

    :param src2: Second source matrix with the same size and type as ``a`` .

    :param dst: Destination matrix with the same size as ``a`` and the ``CV_8UC1`` type.

    :param cmpop: Flag specifying the relation between the elements to be checked:

            * **CMP_EQ:** ``src1(.) == src2(.)``
            * **CMP_GT:** ``src1(.) < src2(.)``
            * **CMP_GE:** ``src1(.) <= src2(.)``
            * **CMP_LT:** ``src1(.) < src2(.)``
            * **CMP_LE:** ``src1(.) <= src2(.)``
            * **CMP_NE:** ``src1(.) != src2(.)``

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`compare`



gpu::bitwise_not
--------------------
Performs a per-element bitwise inversion.

.. ocv:function:: void gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



gpu::bitwise_or
-------------------
Performs a per-element bitwise disjunction of two matrices.

.. ocv:function:: void gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



gpu::bitwise_and
--------------------
Performs a per-element bitwise conjunction of two matrices.

.. ocv:function:: void gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



gpu::bitwise_xor
--------------------
Performs a per-element bitwise ``exclusive or`` operation of two matrices.

.. ocv:function:: void gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



gpu::min
------------
Computes the per-element minimum of two matrices (or a matrix and a scalar).

.. ocv:function:: void gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::min(const GpuMat& src1, double src2, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix or a scalar to compare ``src1`` elements with.

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`min`



gpu::max
------------
Computes the per-element maximum of two matrices (or a matrix and a scalar).

.. ocv:function:: void gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::max(const GpuMat& src1, double src2, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix or a scalar to compare ``src1`` elements with.

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`max`
