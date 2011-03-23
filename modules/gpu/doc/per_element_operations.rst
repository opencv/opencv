Per-element Operations.
=======================

.. highlight:: cpp



.. index:: gpu::add

gpu::add
------------
.. cpp:function:: void gpu::add(const GpuMat& a, const GpuMat& b, GpuMat& c)

    Computes matrix-matrix or matrix-scalar sum.

    :param a: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1`` and ``CV_32FC1`` matrices are supported for now.

    :param b: Second source matrix. Must have the same size and type as ``a``.

    :param c: Destination matrix. Will have the same size and type as ``a``.

.. cpp:function:: void gpu::add(const GpuMat& a, const Scalar& sc, GpuMat& c)

    :param a: Source matrix. ``CV_32FC1`` and ``CV_32FC2`` matrixes are supported for now.

    :param b: Source scalar to be added to the source matrix.

    :param c: Destination matrix. Will have the same size and type as ``a``.

See also: :c:func:`add`.



.. index:: gpu::subtract

gpu::subtract
-----------------
.. cpp:function:: void gpu::subtract(const GpuMat& a, const GpuMat& b, GpuMat& c)

    Subtracts matrix from another matrix (or scalar from matrix).

    :param a: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1`` and ``CV_32FC1`` matrices are supported for now.

    :param b: Second source matrix. Must have the same size and type as ``a``.

    :param c: Destination matrix. Will have the same size and type as ``a``.

.. cpp:function:: void gpu::subtract(const GpuMat& a, const Scalar& sc, GpuMat& c)

    :param a: Source matrix. ``CV_32FC1`` and ``CV_32FC2`` matrixes are supported for now.

    :param b: Scalar to be subtracted from the source matrix elements.

    :param c: Destination matrix. Will have the same size and type as ``a``.

See also: :c:func:`subtract`.



.. index:: gpu::multiply

gpu::multiply
-----------------
.. cpp:function:: void gpu::multiply(const GpuMat& a, const GpuMat& b, GpuMat& c)

    Computes per-element product of two matrices (or of matrix and scalar).

    :param a: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1`` and ``CV_32FC1`` matrices are supported for now.

    :param b: Second source matrix. Must have the same size and type as ``a``.

    :param c: Destionation matrix. Will have the same size and type as ``a``.

.. cpp:function:: void gpu::multiply(const GpuMat& a, const Scalar& sc, GpuMat& c)

    :param a: Source matrix. ``CV_32FC1`` and ``CV_32FC2`` matrixes are supported for now.

    :param b: Scalar to be multiplied by.

    :param c: Destination matrix. Will have the same size and type as ``a``.

See also: :c:func:`multiply`.



.. index:: gpu::divide

gpu::divide
---------------
.. cpp:function:: void gpu::divide(const GpuMat& a, const GpuMat& b, GpuMat& c)

    Performs per-element division of two matrices (or division of matrix by scalar).

    :param a: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1`` and ``CV_32FC1`` matrices are supported for now.

    :param b: Second source matrix. Must have the same size and type as ``a``.

    :param c: Destionation matrix. Will have the same size and type as ``a``.

.. cpp:function:: void gpu::divide(const GpuMat& a, const Scalar& sc, GpuMat& c)

    :param a: Source matrix. ``CV_32FC1`` and ``CV_32FC2`` matrixes are supported for now.

    :param b: Scalar to be divided by.

    :param c: Destination matrix. Will have the same size and type as ``a``.

This function in contrast to :func:`divide` uses round-down rounding mode.

See also: :c:func:`divide`.



.. index:: gpu::exp

gpu::exp
------------
.. cpp:function:: void gpu::exp(const GpuMat& a, GpuMat& b)

    Computes exponent of each matrix element.

    :param a: Source matrix. ``CV_32FC1`` matrixes are supported for now.

    :param b: Destination matrix. Will have the same size and type as ``a``.

See also: :c:func:`exp`.



.. index:: gpu::log

gpu::log
------------
.. cpp:function:: void gpu::log(const GpuMat& a, GpuMat& b)

    Computes natural logarithm of absolute value of each matrix element.

    :param a: Source matrix. ``CV_32FC1`` matrixes are supported for now.

    :param b: Destination matrix. Will have the same size and type as ``a``.

See also: :c:func:`log`.



.. index:: gpu::absdiff

gpu::absdiff
----------------
.. cpp:function:: void gpu::absdiff(const GpuMat& a, const GpuMat& b, GpuMat& c)

    Computes per-element absolute difference of two matrices (or of matrix and scalar).

    :param a: First source matrix. ``CV_8UC1``, ``CV_8UC4``, ``CV_32SC1`` and ``CV_32FC1`` matrices are supported for now.

    :param b: Second source matrix. Must have the same size and type as ``a``.

    :param c: Destionation matrix. Will have the same size and type as ``a``.

.. cpp:function:: void gpu::absdiff(const GpuMat& a, const Scalar& s, GpuMat& c)

    :param a: Source matrix. ``CV_32FC1`` matrixes are supported for now.

    :param b: Scalar to be subtracted from the source matrix elements.

    :param c: Destination matrix. Will have the same size and type as ``a``.

See also: :c:func:`absdiff`.



.. index:: gpu::compare

gpu::compare
----------------
.. cpp:function:: void gpu::compare(const GpuMat& a, const GpuMat& b, GpuMat& c, int cmpop)

    Compares elements of two matrices.

    :param a: First source matrix. ``CV_8UC4`` and ``CV_32FC1`` matrices are supported for now.

    :param b: Second source matrix. Must have the same size and type as ``a``.

    :param c: Destination matrix. Will have the same size as ``a`` and be ``CV_8UC1`` type.

    :param cmpop: Flag specifying the relation between the elements to be checked:
        
            * **CMP_EQ** :math:`=`             
            * **CMP_GT** :math:`>`             
            * **CMP_GE** :math:`\ge`             
            * **CMP_LT** :math:`<`             
            * **CMP_LE** :math:`\le`             
            * **CMP_NE** :math:`\ne`             

See also: :c:func:`compare`.



.. index:: gpu::bitwise_not

gpu::bitwise_not
--------------------
.. cpp:function:: void gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask=GpuMat())

.. cpp:function:: void gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs per-element bitwise inversion.

    :param src: Source matrix.

    :param dst: Destination matrix. Will have the same size and type as ``src``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::bitwise_or

gpu::bitwise_or
-------------------
.. cpp:function:: void gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat())

.. cpp:function:: void gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs per-element bitwise disjunction of two matrices.

    :param src1: First source matrix.

    :param src2: Second source matrix. It must have the same size and type as ``src1``.

    :param dst: Destination matrix. Will have the same size and type as ``src1``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::bitwise_and

gpu::bitwise_and
--------------------
.. cpp:function:: void gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat())

.. cpp:function:: void gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs per-element bitwise conjunction of two matrices.

    :param src1: First source matrix.

    :param src2: Second source matrix. It must have the same size and type as ``src1``.

    :param dst: Destination matrix. Will have the same size and type as ``src1``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::bitwise_xor

gpu::bitwise_xor
--------------------
.. cpp:function:: void gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat())

.. cpp:function:: void gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)

    Performs per-element bitwise "exclusive or" of two matrices.

    :param src1: First source matrix.

    :param src2: Second source matrix. It must have the same size and type as ``src1``.

    :param dst: Destination matrix. Will have the same size and type as ``src1``.

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::min

gpu::min
------------
.. cpp:function:: void gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. cpp:function:: void gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const Stream& stream)

.. cpp:function:: void gpu::min(const GpuMat& src1, double value, GpuMat& dst)

.. cpp:function:: void gpu::min(const GpuMat& src1, double value, GpuMat& dst, const Stream& stream)

    Computes per-element minimum of two matrices (or a matrix and a scalar).

    :param src1: First source matrix.

    :param src2: Second source matrix.

    :param value: Scalar value to compare ``src1`` elements with.

    :param dst: Destination matrix. Will have the same size and type as ``src1``.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`min`.



.. index:: gpu::max

gpu::max
------------
.. cpp:function:: void gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)

.. cpp:function:: void gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const Stream& stream)

.. cpp:function:: void gpu::max(const GpuMat& src1, double value, GpuMat& dst)

.. cpp:function:: void gpu::max(const GpuMat& src1, double value, GpuMat& dst, const Stream& stream)

    Computes per-element maximum of two matrices (or a matrix and a scalar).

    :param src1: First source matrix.

    :param src2: Second source matrix.
    
    :param value: Scalar value to compare ``src1`` elements with.

    :param dst: Destination matrix. Will have the same size and type as ``src1``.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`max`.
