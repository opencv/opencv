Per-element Operations
=======================

.. highlight:: cpp



gpu::add
------------
Computes a matrix-matrix or matrix-scalar sum.

.. ocv:function:: void gpu::add( const GpuMat& a, const GpuMat& b, GpuMat& c, const GpuMat& mask=GpuMat(), int dtype=-1, Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::add( const GpuMat& a, const Scalar& sc, GpuMat& c, const GpuMat& mask=GpuMat(), int dtype=-1, Stream& stream=Stream::Null() )

    :param a: First source matrix.

    :param b: Second source matrix to be added to ``a`` . Matrix should have the same size and type as ``a`` .

    :param sc: A scalar to be added to ``a`` .

    :param c: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``a`` depth.

    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`add`



gpu::subtract
-----------------
Computes a matrix-matrix or matrix-scalar difference.

.. ocv:function:: void gpu::subtract( const GpuMat& a, const GpuMat& b, GpuMat& c, const GpuMat& mask=GpuMat(), int dtype=-1, Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::subtract( const GpuMat& a, const Scalar& sc, GpuMat& c, const GpuMat& mask=GpuMat(), int dtype=-1, Stream& stream=Stream::Null() )

    :param a: First source matrix.

    :param b: Second source matrix to be added to ``a`` . Matrix should have the same size and type as ``a`` .

    :param sc: A scalar to be added to ``a`` .

    :param c: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``a`` depth.

    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`subtract`



gpu::multiply
-----------------
Computes a matrix-matrix or matrix-scalar per-element product.

.. ocv:function:: void gpu::multiply( const GpuMat& a, const GpuMat& b, GpuMat& c, double scale=1, int dtype=-1, Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::multiply( const GpuMat& a, const Scalar& sc, GpuMat& c, double scale=1, int dtype=-1, Stream& stream=Stream::Null() )

    :param a: First source matrix.

    :param b: Second source matrix to be multiplied by ``a`` elements.

    :param sc: A scalar to be multiplied by ``a`` elements.

    :param c: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``a`` depth.

    :param scale: Optional scale factor.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`multiply`



gpu::divide
-----------
Computes a matrix-matrix or matrix-scalar division.

.. ocv:function:: void gpu::divide( const GpuMat& a, const GpuMat& b, GpuMat& c, double scale=1, int dtype=-1, Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::divide(const GpuMat& a, const Scalar& sc, GpuMat& c, double scale = 1, int dtype = -1, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::divide( double scale, const GpuMat& b, GpuMat& c, int dtype=-1, Stream& stream=Stream::Null() )

    :param a: First source matrix or a scalar.

    :param b: Second source matrix. The ``a`` elements are divided by it.

    :param sc: A scalar to be divided by the elements of ``a`` matrix.

    :param c: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``a`` depth.

    :param scale: Optional scale factor.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

This function, in contrast to :ocv:func:`divide`, uses a round-down rounding mode.

.. seealso:: :ocv:func:`divide`


gpu::addWeighted
----------------
Computes the weighted sum of two arrays.

.. ocv:function:: void gpu::addWeighted(const GpuMat& src1, double alpha, const GpuMat& src2, double beta, double gamma, GpuMat& dst, int dtype = -1, Stream& stream = Stream::Null())

    :param src1: First source array.

    :param alpha: Weight for the first array elements.

    :param src2: Second source array of the same size and channel number as  ``src1`` .

    :param beta: Weight for the second array elements.

    :param dst: Destination array that has the same size and number of channels as the input arrays.

    :param gamma: Scalar added to each sum.

    :param dtype: Optional depth of the destination array. When both input arrays have the same depth, ``dtype`` can be set to ``-1``, which will be equivalent to ``src1.depth()``.

    :param stream: Stream for the asynchronous version.

The function ``addWeighted`` calculates the weighted sum of two arrays as follows:

.. math::

    \texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )

where ``I`` is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently.

.. seealso:: :ocv:func:`addWeighted`



gpu::abs
------------
Computes an absolute value of each matrix element.

.. ocv:function:: void gpu::abs(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports ``CV_16S`` and ``CV_32F`` depth.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`abs`



gpu::sqr
------------
Computes a square value of each matrix element.

.. ocv:function:: void gpu::sqr(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports ``CV_8U`` , ``CV_16U`` , ``CV_16S`` and ``CV_32F`` depth.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.



gpu::sqrt
------------
Computes a square root of each matrix element.

.. ocv:function:: void gpu::sqrt(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports ``CV_8U`` , ``CV_16U`` , ``CV_16S`` and ``CV_32F`` depth.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`sqrt`



gpu::exp
------------
Computes an exponent of each matrix element.

.. ocv:function:: void gpu::exp( const GpuMat& a, GpuMat& b, Stream& stream=Stream::Null() )

    :param a: Source matrix. Supports ``CV_8U`` , ``CV_16U`` , ``CV_16S`` and ``CV_32F`` depth.

    :param b: Destination matrix with the same size and type as ``a`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`exp`



gpu::log
------------
Computes a natural logarithm of absolute value of each matrix element.

.. ocv:function:: void gpu::log( const GpuMat& a, GpuMat& b, Stream& stream=Stream::Null() )

    :param a: Source matrix. Supports ``CV_8U`` , ``CV_16U`` , ``CV_16S`` and ``CV_32F`` depth.

    :param b: Destination matrix with the same size and type as ``a`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`log`



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



gpu::absdiff
----------------
Computes per-element absolute difference of two matrices (or of a matrix and scalar).

.. ocv:function:: void gpu::absdiff( const GpuMat& a, const GpuMat& b, GpuMat& c, Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::absdiff( const GpuMat& a, const Scalar& s, GpuMat& c, Stream& stream=Stream::Null() )

    :param a: First source matrix.

    :param b: Second source matrix to be added to ``a`` .

    :param s: A scalar to be added to ``a`` .

    :param c: Destination matrix with the same size and type as ``a`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`absdiff`



gpu::compare
----------------
Compares elements of two matrices.

.. ocv:function:: void gpu::compare( const GpuMat& a, const GpuMat& b, GpuMat& c, int cmpop, Stream& stream=Stream::Null() )

    :param a: First source matrix.

    :param b: Second source matrix with the same size and type as ``a`` .

    :param sc: A scalar to be compared with ``a`` .

    :param c: Destination matrix with the same size as ``a`` and the ``CV_8UC1`` type.

    :param cmpop: Flag specifying the relation between the elements to be checked:

            * **CMP_EQ:** ``a(.) == b(.)``
            * **CMP_GT:** ``a(.) < b(.)``
            * **CMP_GE:** ``a(.) <= b(.)``
            * **CMP_LT:** ``a(.) < b(.)``
            * **CMP_LE:** ``a(.) <= b(.)``
            * **CMP_NE:** ``a(.) != b(.)``

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
Performs a per-element bitwise disjunction of two matrices or of matrix and scalar.

.. ocv:function:: void gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null())
.. ocv:function:: void gpu::bitwise_or(const GpuMat& src1, const Scalar& sc, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



gpu::bitwise_and
--------------------
Performs a per-element bitwise conjunction of two matrices or of matrix and scalar.

.. ocv:function:: void gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null())
.. ocv:function:: void gpu::bitwise_and(const GpuMat& src1, const Scalar& sc, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



gpu::bitwise_xor
--------------------
Performs a per-element bitwise ``exclusive or`` operation of two matrices of matrix and scalar.

.. ocv:function:: void gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null())
.. ocv:function:: void gpu::bitwise_xor(const GpuMat& src1, const Scalar& sc, GpuMat& dst, Stream& stream = Stream::Null())

    :param src1: First source matrix.

    :param src2: Second source matrix with the same size and type as ``src1`` .

    :param dst: Destination matrix with the same size and type as ``src1`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



gpu::rshift
--------------------
Performs pixel by pixel right shift of an image by a constant value.

.. ocv:function:: void gpu::rshift( const GpuMat& src, Scalar_<int> sc, GpuMat& dst, Stream& stream=Stream::Null() )

    :param src: Source matrix. Supports 1, 3 and 4 channels images with integers elements.

    :param sc: Constant values, one per channel.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.



gpu::lshift
--------------------
Performs pixel by pixel right left of an image by a constant value.

.. ocv:function:: void gpu::lshift( const GpuMat& src, Scalar_<int> sc, GpuMat& dst, Stream& stream=Stream::Null() )

    :param src: Source matrix. Supports 1, 3 and 4 channels images with ``CV_8U`` , ``CV_16U`` or ``CV_32S`` depth.

    :param sc: Constant values, one per channel.

    :param dst: Destination matrix with the same size and type as ``src`` .

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
