Matrix Reductions
=================

.. highlight:: cpp



cuda::norm
----------
Returns the norm of a matrix (or difference of two matrices).

.. ocv:function:: double cuda::norm(InputArray src1, int normType)

.. ocv:function:: double cuda::norm(InputArray src1, int normType, GpuMat& buf)

.. ocv:function:: double cuda::norm(InputArray src1, int normType, InputArray mask, GpuMat& buf)

.. ocv:function:: double cuda::norm(InputArray src1, InputArray src2, int normType=NORM_L2)

    :param src1: Source matrix. Any matrices except 64F are supported.

    :param src2: Second source matrix (if any) with the same size and type as ``src1``.

    :param normType: Norm type.  ``NORM_L1`` ,  ``NORM_L2`` , and  ``NORM_INF``  are supported for now.

    :param mask: optional operation mask; it must have the same size as ``src1`` and ``CV_8UC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`norm`



cuda::sum
---------
Returns the sum of matrix elements.

.. ocv:function:: Scalar cuda::sum(InputArray src)

.. ocv:function:: Scalar cuda::sum(InputArray src, GpuMat& buf)

.. ocv:function:: Scalar cuda::sum(InputArray src, InputArray mask, GpuMat& buf)

    :param src: Source image of any depth except for ``CV_64F`` .

    :param mask: optional operation mask; it must have the same size as ``src1`` and ``CV_8UC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`sum`



cuda::absSum
------------
Returns the sum of absolute values for matrix elements.

.. ocv:function:: Scalar cuda::absSum(InputArray src)

.. ocv:function:: Scalar cuda::absSum(InputArray src, GpuMat& buf)

.. ocv:function:: Scalar cuda::absSum(InputArray src, InputArray mask, GpuMat& buf)

    :param src: Source image of any depth except for ``CV_64F`` .

    :param mask: optional operation mask; it must have the same size as ``src1`` and ``CV_8UC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.



cuda::sqrSum
------------
Returns the squared sum of matrix elements.

.. ocv:function:: Scalar cuda::sqrSum(InputArray src)

.. ocv:function:: Scalar cuda::sqrSum(InputArray src, GpuMat& buf)

.. ocv:function:: Scalar cuda::sqrSum(InputArray src, InputArray mask, GpuMat& buf)

    :param src: Source image of any depth except for ``CV_64F`` .

    :param mask: optional operation mask; it must have the same size as ``src1`` and ``CV_8UC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.



cuda::minMax
------------
Finds global minimum and maximum matrix elements and returns their values.

.. ocv:function:: void cuda::minMax(InputArray src, double* minVal, double* maxVal=0, InputArray mask=noArray())

.. ocv:function:: void cuda::minMax(InputArray src, double* minVal, double* maxVal, InputArray mask, GpuMat& buf)

    :param src: Single-channel source image.

    :param minVal: Pointer to the returned minimum value.  Use ``NULL``  if not required.

    :param maxVal: Pointer to the returned maximum value.  Use ``NULL``  if not required.

    :param mask: Optional mask to select a sub-matrix.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

The function does not work with ``CV_64F`` images on GPUs with the compute capability < 1.3.

.. seealso:: :ocv:func:`minMaxLoc`



cuda::minMaxLoc
---------------
Finds global minimum and maximum matrix elements and returns their values with locations.

.. ocv:function:: void cuda::minMaxLoc(InputArray src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())

.. ocv:function:: void cuda::minMaxLoc(InputArray src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, InputArray mask, GpuMat& valbuf, GpuMat& locbuf)

    :param src: Single-channel source image.

    :param minVal: Pointer to the returned minimum value. Use ``NULL``  if not required.

    :param maxVal: Pointer to the returned maximum value. Use ``NULL``  if not required.

    :param minLoc: Pointer to the returned minimum location. Use ``NULL``  if not required.

    :param maxLoc: Pointer to the returned maximum location. Use ``NULL``  if not required.

    :param mask: Optional mask to select a sub-matrix.

    :param valbuf: Optional values buffer to avoid extra memory allocations. It is resized automatically.

    :param locbuf: Optional locations buffer to avoid extra memory allocations. It is resized automatically.

    The function does not work with ``CV_64F`` images on GPU with the compute capability < 1.3.

.. seealso:: :ocv:func:`minMaxLoc`



cuda::countNonZero
------------------
Counts non-zero matrix elements.

.. ocv:function:: int cuda::countNonZero(InputArray src)

.. ocv:function:: int cuda::countNonZero(InputArray src, GpuMat& buf)

    :param src: Single-channel source image.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

The function does not work with ``CV_64F`` images on GPUs with the compute capability < 1.3.

.. seealso:: :ocv:func:`countNonZero`



cuda::reduce
------------
Reduces a matrix to a vector.

.. ocv:function:: void cuda::reduce(InputArray mtx, OutputArray vec, int dim, int reduceOp, int dtype = -1, Stream& stream = Stream::Null())

    :param mtx: Source 2D matrix.

    :param vec: Destination vector. Its size and type is defined by  ``dim``  and  ``dtype``  parameters.

    :param dim: Dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row. 1 means that the matrix is reduced to a single column.

    :param reduceOp: Reduction operation that could be one of the following:

            * **CV_REDUCE_SUM** The output is the sum of all rows/columns of the matrix.

            * **CV_REDUCE_AVG** The output is the mean vector of all rows/columns of the matrix.

            * **CV_REDUCE_MAX** The output is the maximum (column/row-wise) of all rows/columns of the matrix.

            * **CV_REDUCE_MIN** The output is the minimum (column/row-wise) of all rows/columns of the matrix.

    :param dtype: When it is negative, the destination vector will have the same type as the source matrix. Otherwise, its type will be  ``CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), mtx.channels())`` .

    :param stream: Stream for the asynchronous version.

The function ``reduce`` reduces the matrix to a vector by treating the matrix rows/columns as a set of 1D vectors and performing the specified operation on the vectors until a single row/column is obtained. For example, the function can be used to compute horizontal and vertical projections of a raster image. In case of ``CV_REDUCE_SUM`` and ``CV_REDUCE_AVG`` , the output may have a larger element bit-depth to preserve accuracy. And multi-channel arrays are also supported in these two reduction modes.

.. seealso:: :ocv:func:`reduce`



cuda::meanStdDev
----------------
Computes a mean value and a standard deviation of matrix elements.

.. ocv:function:: void cuda::meanStdDev(InputArray mtx, Scalar& mean, Scalar& stddev)
.. ocv:function:: void cuda::meanStdDev(InputArray mtx, Scalar& mean, Scalar& stddev, GpuMat& buf)

    :param mtx: Source matrix.  ``CV_8UC1``  matrices are supported for now.

    :param mean: Mean value.

    :param stddev: Standard deviation value.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`meanStdDev`



cuda::rectStdDev
----------------
Computes a standard deviation of integral images.

.. ocv:function:: void cuda::rectStdDev(InputArray src, InputArray sqr, OutputArray dst, Rect rect, Stream& stream = Stream::Null())

    :param src: Source image. Only the ``CV_32SC1`` type is supported.

    :param sqr: Squared source image. Only  the ``CV_32FC1`` type is supported.

    :param dst: Destination image with the same type and size as  ``src`` .

    :param rect: Rectangular window.

    :param stream: Stream for the asynchronous version.



cuda::normalize
---------------
Normalizes the norm or value range of an array.

.. ocv:function:: void cuda::normalize(InputArray src, OutputArray dst, double alpha = 1, double beta = 0, int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray())

.. ocv:function:: void cuda::normalize(InputArray src, OutputArray dst, double alpha, double beta, int norm_type, int dtype, InputArray mask, GpuMat& norm_buf, GpuMat& cvt_buf)

    :param src: Input array.

    :param dst: Output array of the same size as  ``src`` .

    :param alpha: Norm value to normalize to or the lower range boundary in case of the range normalization.

    :param beta: Upper range boundary in case of the range normalization; it is not used for the norm normalization.

    :param normType: Normalization type ( ``NORM_MINMAX`` , ``NORM_L2`` , ``NORM_L1`` or ``NORM_INF`` ).

    :param dtype: When negative, the output array has the same type as ``src``; otherwise, it has the same number of channels as  ``src`` and the depth ``=CV_MAT_DEPTH(dtype)``.

    :param mask: Optional operation mask.

    :param norm_buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

    :param cvt_buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`normalize`



cuda::integral
--------------
Computes an integral image.

.. ocv:function:: void cuda::integral(InputArray src, OutputArray sum, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::integral(InputArray src, OutputArray sum, GpuMat& buffer, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sum: Integral image containing 32-bit unsigned integer values packed into  ``CV_32SC1`` .

    :param buffer: Optional buffer to avoid extra memory allocations. It is resized automatically.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`integral`



cuda::sqrIntegral
-----------------
Computes a squared integral image.

.. ocv:function:: void cuda::sqrIntegral(InputArray src, OutputArray sqsum, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::sqrIntegral(InputArray src, OutputArray sqsum, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sqsum: Squared integral image containing 64-bit unsigned integer values packed into  ``CV_64FC1`` .

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

    :param stream: Stream for the asynchronous version.
