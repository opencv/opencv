Per-element Operations.
=======================

.. highlight:: cpp



.. index:: gpu::add

gpu::add
------------
.. ocv:function:: void gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. ocv:function:: void gpu::add(const GpuMat& src1, const Scalar& src2, GpuMat& dst)

    Computes a matrix-matrix or matrix-scalar sum.

    :param src1: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1``, and ``CV_32FC1`` matrices are supported for now.

    :param src2: Second source matrix or a scalar to be added to ``src1``.

    :param dst: Destination matrix with the same size and type as ``src1``.

See Also: :ocv:func:`add`

.. index:: gpu::subtract

gpu::subtract
-----------------
.. ocv:function:: void gpu::subtract(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. ocv:function:: void gpu::subtract(const GpuMat& src1, const Scalar& src2, GpuMat& dst)

    Computes a matrix-matrix or matrix-scalar difference.

    :param src1: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1``, and ``CV_32FC1`` matrices are supported for now.

    :param src2: Second source matrix or a scalar to be subtracted from ``src1``.

    :param dst: Destination matrix with the same size and type as ``src1``.

See Also: :ocv:func:`subtract`



.. index:: gpu::multiply

gpu::multiply
-----------------
.. ocv:function:: void gpu::multiply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. ocv:function:: void gpu::multiply(const GpuMat& src1, const Scalar& src2, GpuMat& dst)

    Computes a matrix-matrix or matrix-scalar per-element product.

    :param src1: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1``, and ``CV_32FC1`` matrices are supported for now.

    :param src2: Second source matrix or a scalar to be multiplied by ``src1`` elements.

    :param dst: Destination matrix with the same size and type as ``src1``.

See Also: :ocv:func:`multiply`


.. index:: gpu::divide

gpu::divide
---------------
.. ocv:function:: void gpu::divide(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. ocv:function:: void gpu::divide(const GpuMat& src1, const Scalar& src2, GpuMat& dst)

    Computes a matrix-matrix or matrix-scalar sum.

    :param src1: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1``, and ``CV_32FC1`` matrices are supported for now.

    :param src2: Second source matrix or a scalar. The ``src1`` elements are divided by it.

    :param dst: Destination matrix with the same size and type as ``src1``.

	This function, in contrast to :ocv:func:`divide`, uses a round-down rounding mode.

See Also: :ocv:func:`divide`



.. index:: gpu::exp

gpu::exp
------------
.. ocv:function:: void gpu::exp(const GpuMat& src, GpuMat& dst)

    Computes an exponent of each matrix element.

    :param src: Source matrix. ``CV_32FC1`` matrixes are supported for now.

    :param dst: Destination matrix with the same size and type as ``src``.

See Also: :ocv:func:`exp`



.. index:: gpu::log

gpu::log
------------
.. ocv:function:: void gpu::log(const GpuMat& src, GpuMat& dst)

    Computes a natural logarithm of absolute value of each matrix element.

    :param src: Source matrix. ``CV_32FC1`` matrixes are supported for now.

    :param dst: Destination matrix with the same size and type as ``src``.

See Also: :ocv:func:`log`



.. index:: gpu::absdiff

gpu::absdiff
----------------
.. ocv:function:: void gpu::absdiff(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. ocv:function:: void gpu::absdiff(const GpuMat& src1, const Scalar& src2, GpuMat& dst)

    Computes per-element absolute difference of two matrices (or of matrix and scalar).

    :param src1: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1`` and ``CV_32FC1`` matrices are supported for now.

    :param src2: Second source matrix or a scalar to be added to ``src1``.

    :param dst: Destination matrix with the same size and type as ``src1``.

See Also: :ocv:func:`absdiff`

.. index:: gpu::compare

gpu::compare
----------------
.. ocv:function:: void gpu::compare(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int cmpop)

    Compares elements of two matrices.

    :param src1: First source matrix. ``CV_8UC4`` and ``CV_32FC1`` matrices are supported for now.

    :param src2: Second source matrix with the same size and type as ``a``.

    :param dst: Destination matrix with the same size as ``a`` and the ``CV_8UC1`` type.

    :param cmpop: Flag specifying the relation between the elements to be checked:
        
            * **CMP_EQ:** ``src1(.) == src2(.)``
            * **CMP_GT:** ``src1(.) < src2(.)``
            * **CMP_GE:** ``src1(.) <= src2(.)``
            * **CMP_LT:** ``src1(.) < src2(.)``
            * **CMP_LE:** ``src1(.) <= src2(.)``
            * **CMP_NE:** ``src1(.) != src2(.)``

See Also: :ocv:func:`compare`


.. index:: gpu::bitwise_not

gpu::bitwise_not
--------------------
.. ocv:function:: void gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask=GpuMat())

.. ocv:function:: void gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs a per-element bitwise inversion.

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::bitwise_or

gpu::bitwise_or
-------------------
.. ocv:function:: void gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat())

.. ocv:function:: void gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs a per-element bitwise disjunction of two matrices.

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1``.

    :param dst: Destination matrix with the same size and type as ``src1``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::bitwise_and

gpu::bitwise_and
--------------------
.. ocv:function:: void gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat())

.. ocv:function:: void gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs a per-element bitwise conjunction of two matrices.

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1``.

    :param dst: Destination matrix with the same size and type as ``src1``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::bitwise_xor

gpu::bitwise_xor
--------------------
.. ocv:function:: void gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat())

.. ocv:function:: void gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs a per-element bitwise "exclusive or" operation of two matrices.

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1``.

    :param dst: Destination matrix with the same size and type as ``src1``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::min

gpu::min
------------
.. ocv:function:: void gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. ocv:function:: void gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const Stream& stream)

.. ocv:function:: void gpu::min(const GpuMat& src1, double src2, GpuMat& dst)

.. ocv:function:: void gpu::min(const GpuMat& src1, double src2, GpuMat& dst, const Stream& stream)

    Computes the per-element minimum of two matrices (or a matrix and a scalar).

    :param src1: First source matrix.

    :param src2: Second source matrix or a scalar to compare ``src1`` elements with.

    :param dst: Destination matrix with the same size and type as ``src1``.

    :param stream: Stream for the asynchronous version.

See Also: :ocv:func:`min`



.. index:: gpu::max

gpu::max
------------
.. ocv:function:: void gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. ocv:function:: void gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const Stream& stream)

.. ocv:function:: void gpu::max(const GpuMat& src1, double src2, GpuMat& dst)

.. ocv:function:: void gpu::max(const GpuMat& src1, double src2, GpuMat& dst, const Stream& stream)

    Computes the per-element maximum of two matrices (or a matrix and a scalar).

    :param src1: First source matrix.

    :param src2: Second source matrix or a scalar to compare ``src1`` elements with.

    :param dst: Destination matrix with the same size and type as ``src1``.

    :param stream: Stream for the asynchronous version.

See Also: :ocv:func:`max`
