Per-element Operations
======================

.. highlight:: cpp



cuda::add
---------
Computes a matrix-matrix or matrix-scalar sum.

.. ocv:function:: void cuda::add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), int dtype = -1, Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar. Matrix should have the same size and type as ``src1`` .

    :param dst: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``src1`` depth.

    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`add`



cuda::subtract
--------------
Computes a matrix-matrix or matrix-scalar difference.

.. ocv:function:: void cuda::subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), int dtype = -1, Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar. Matrix should have the same size and type as ``src1`` .

    :param dst: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``src1`` depth.

    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`subtract`



cuda::multiply
--------------
Computes a matrix-matrix or matrix-scalar per-element product.

.. ocv:function:: void cuda::multiply(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1, Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``src1`` depth.

    :param scale: Optional scale factor.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`multiply`



cuda::divide
------------
Computes a matrix-matrix or matrix-scalar division.

.. ocv:function:: void cuda::divide(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::divide(double src1, InputArray src2, OutputArray dst, int dtype = -1, Stream& stream = Stream::Null())

    :param src1: First source matrix or a scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``src1`` depth.

    :param scale: Optional scale factor.

    :param dtype: Optional depth of the output array.

    :param stream: Stream for the asynchronous version.

This function, in contrast to :ocv:func:`divide`, uses a round-down rounding mode.

.. seealso:: :ocv:func:`divide`



cuda::absdiff
-------------
Computes per-element absolute difference of two matrices (or of a matrix and scalar).

.. ocv:function:: void cuda::absdiff(InputArray src1, InputArray src2, OutputArray dst, Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and type as the input array(s).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`absdiff`



cuda::abs
---------
Computes an absolute value of each matrix element.

.. ocv:function:: void cuda::abs(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`abs`



cuda::sqr
---------
Computes a square value of each matrix element.

.. ocv:function:: void cuda::sqr(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.



cuda::sqrt
----------
Computes a square root of each matrix element.

.. ocv:function:: void cuda::sqrt(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`sqrt`



cuda::exp
---------
Computes an exponent of each matrix element.

.. ocv:function:: void cuda::exp(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`exp`



cuda::log
---------
Computes a natural logarithm of absolute value of each matrix element.

.. ocv:function:: void cuda::log(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`log`



cuda::pow
---------
Raises every matrix element to a power.

.. ocv:function:: void cuda::pow(InputArray src, double power, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param power: Exponent of power.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.

The function ``pow`` raises every element of the input matrix to ``power`` :

.. math::

    \texttt{dst} (I) =  \fork{\texttt{src}(I)^power}{if \texttt{power} is integer}{|\texttt{src}(I)|^power}{otherwise}

.. seealso:: :ocv:func:`pow`



cuda::compare
-------------
Compares elements of two matrices (or of a matrix and scalar).

.. ocv:function:: void cuda::compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop, Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and type as the input array(s).

    :param cmpop: Flag specifying the relation between the elements to be checked:

            * **CMP_EQ:** ``a(.) == b(.)``
            * **CMP_GT:** ``a(.) < b(.)``
            * **CMP_GE:** ``a(.) <= b(.)``
            * **CMP_LT:** ``a(.) < b(.)``
            * **CMP_LE:** ``a(.) <= b(.)``
            * **CMP_NE:** ``a(.) != b(.)``

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`compare`



cuda::bitwise_not
-----------------
Performs a per-element bitwise inversion.

.. ocv:function:: void cuda::bitwise_not(InputArray src, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



cuda::bitwise_or
----------------
Performs a per-element bitwise disjunction of two matrices (or of matrix and scalar).

.. ocv:function:: void cuda::bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and type as the input array(s).

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



cuda::bitwise_and
-----------------
Performs a per-element bitwise conjunction of two matrices (or of matrix and scalar).

.. ocv:function:: void cuda::bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and type as the input array(s).

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



cuda::bitwise_xor
-----------------
Performs a per-element bitwise ``exclusive or`` operation of two matrices (or of matrix and scalar).

.. ocv:function:: void cuda::bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray(), Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and type as the input array(s).

    :param mask: Optional operation mask. 8-bit single channel image.

    :param stream: Stream for the asynchronous version.



cuda::rshift
------------
Performs pixel by pixel right shift of an image by a constant value.

.. ocv:function:: void cuda::rshift(InputArray src, Scalar_<int> val, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports 1, 3 and 4 channels images with integers elements.

    :param val: Constant values, one per channel.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.



cuda::lshift
------------
Performs pixel by pixel right left of an image by a constant value.

.. ocv:function:: void cuda::lshift(InputArray src, Scalar_<int> val, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports 1, 3 and 4 channels images with ``CV_8U`` , ``CV_16U`` or ``CV_32S`` depth.

    :param val: Constant values, one per channel.

    :param dst: Destination matrix with the same size and type as ``src`` .

    :param stream: Stream for the asynchronous version.



cuda::min
---------
Computes the per-element minimum of two matrices (or a matrix and a scalar).

.. ocv:function:: void cuda::min(InputArray src1, InputArray src2, OutputArray dst, Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and type as the input array(s).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`min`



cuda::max
---------
Computes the per-element maximum of two matrices (or a matrix and a scalar).

.. ocv:function:: void cuda::max(InputArray src1, InputArray src2, OutputArray dst, Stream& stream = Stream::Null())

    :param src1: First source matrix or scalar.

    :param src2: Second source matrix or scalar.

    :param dst: Destination matrix that has the same size and type as the input array(s).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`max`



cuda::addWeighted
-----------------
Computes the weighted sum of two arrays.

.. ocv:function:: void cuda::addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype = -1, Stream& stream = Stream::Null())

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



cuda::threshold
---------------
Applies a fixed-level threshold to each array element.

.. ocv:function:: double cuda::threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type, Stream& stream = Stream::Null())

    :param src: Source array (single-channel).

    :param dst: Destination array with the same size and type as  ``src`` .

    :param thresh: Threshold value.

    :param maxval: Maximum value to use with  ``THRESH_BINARY`` and  ``THRESH_BINARY_INV`` threshold types.

    :param type: Threshold type. For details, see  :ocv:func:`threshold` . The ``THRESH_OTSU`` threshold type is not supported.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`threshold`



cuda::magnitude
---------------
Computes magnitudes of complex matrix elements.

.. ocv:function:: void cuda::magnitude(InputArray xy, OutputArray magnitude, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::magnitude(InputArray x, InputArray y, OutputArray magnitude, Stream& stream = Stream::Null())

    :param xy: Source complex matrix in the interleaved format ( ``CV_32FC2`` ).

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnitudes ( ``CV_32FC1`` ).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`magnitude`



cuda::magnitudeSqr
------------------
Computes squared magnitudes of complex matrix elements.

.. ocv:function:: void cuda::magnitudeSqr(InputArray xy, OutputArray magnitude, Stream& stream=Stream::Null() )

.. ocv:function:: void cuda::magnitudeSqr(InputArray x, InputArray y, OutputArray magnitude, Stream& stream = Stream::Null())

    :param xy: Source complex matrix in the interleaved format ( ``CV_32FC2`` ).

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnitude squares ( ``CV_32FC1`` ).

    :param stream: Stream for the asynchronous version.



cuda::phase
-----------
Computes polar angles of complex matrix elements.

.. ocv:function:: void cuda::phase(InputArray x, InputArray y, OutputArray angle, bool angleInDegrees = false, Stream& stream = Stream::Null())

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param angle: Destination matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag for angles that must be evaluated in degrees.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`phase`



cuda::cartToPolar
-----------------
Converts Cartesian coordinates into polar.

.. ocv:function:: void cuda::cartToPolar(InputArray x, InputArray y, OutputArray magnitude, OutputArray angle, bool angleInDegrees = false, Stream& stream = Stream::Null())

    :param x: Source matrix containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnitudes ( ``CV_32FC1`` ).

    :param angle: Destination matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag for angles that must be evaluated in degrees.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`cartToPolar`



cuda::polarToCart
-----------------
Converts polar coordinates into Cartesian.

.. ocv:function:: void cuda::polarToCart(InputArray magnitude, InputArray angle, OutputArray x, OutputArray y, bool angleInDegrees = false, Stream& stream = Stream::Null())

    :param magnitude: Source matrix containing magnitudes ( ``CV_32FC1`` ).

    :param angle: Source matrix containing angles ( ``CV_32FC1`` ).

    :param x: Destination matrix of real components ( ``CV_32FC1`` ).

    :param y: Destination matrix of imaginary components ( ``CV_32FC1`` ).

    :param angleInDegrees: Flag that indicates angles in degrees.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`polarToCart`
