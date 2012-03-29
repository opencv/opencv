Matrix Reductions
=================

.. highlight:: cpp



gpu::meanStdDev
-------------------
Computes a mean value and a standard deviation of matrix elements.

.. ocv:function:: void gpu::meanStdDev(const GpuMat& mtx, Scalar& mean, Scalar& stddev)
.. ocv:function:: void gpu::meanStdDev(const GpuMat& mtx, Scalar& mean, Scalar& stddev, GpuMat& buf)

    :param mtx: Source matrix.  ``CV_8UC1``  matrices are supported for now.

    :param mean: Mean value.

    :param stddev: Standard deviation value.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`meanStdDev`



gpu::norm
-------------
Returns the norm of a matrix (or difference of two matrices).

.. ocv:function:: double gpu::norm(const GpuMat& src1, int normType=NORM_L2)

.. ocv:function:: double gpu::norm(const GpuMat& src1, int normType, GpuMat& buf)

.. ocv:function:: double gpu::norm(const GpuMat& src1, const GpuMat& src2, int normType=NORM_L2)

    :param src1: Source matrix. Any matrices except 64F are supported.

    :param src2: Second source matrix (if any) with the same size and type as ``src1``.

    :param normType: Norm type.  ``NORM_L1`` ,  ``NORM_L2`` , and  ``NORM_INF``  are supported for now.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`norm`



gpu::sum
------------
Returns the sum of matrix elements.

.. ocv:function:: Scalar gpu::sum(const GpuMat& src)

.. ocv:function:: Scalar gpu::sum(const GpuMat& src, GpuMat& buf)

    :param src: Source image of any depth except for ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: :ocv:func:`sum`



gpu::absSum
---------------
Returns the sum of absolute values for matrix elements.

.. ocv:function:: Scalar gpu::absSum(const GpuMat& src)

.. ocv:function:: Scalar gpu::absSum(const GpuMat& src, GpuMat& buf)

    :param src: Source image of any depth except for ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.



gpu::sqrSum
---------------
Returns the squared sum of matrix elements.

.. ocv:function:: Scalar gpu::sqrSum(const GpuMat& src)

.. ocv:function:: Scalar gpu::sqrSum(const GpuMat& src, GpuMat& buf)

    :param src: Source image of any depth except for ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.



gpu::minMax
---------------
Finds global minimum and maximum matrix elements and returns their values.

.. ocv:function:: void gpu::minMax(const GpuMat& src, double* minVal, double* maxVal=0, const GpuMat& mask=GpuMat())

.. ocv:function:: void gpu::minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask, GpuMat& buf)

    :param src: Single-channel source image.

    :param minVal: Pointer to the returned minimum value.  Use ``NULL``  if not required.

    :param maxVal: Pointer to the returned maximum value.  Use ``NULL``  if not required.

    :param mask: Optional mask to select a sub-matrix.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

The function does not work with ``CV_64F`` images on GPUs with the compute capability < 1.3.

.. seealso:: :ocv:func:`minMaxLoc`



gpu::minMaxLoc
------------------
Finds global minimum and maximum matrix elements and returns their values with locations.

.. ocv:function:: void gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, const GpuMat& mask=GpuMat())

.. ocv:function:: void gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, const GpuMat& mask, GpuMat& valbuf, GpuMat& locbuf)

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



gpu::countNonZero
---------------------
Counts non-zero matrix elements.

.. ocv:function:: int gpu::countNonZero(const GpuMat& src)

.. ocv:function:: int gpu::countNonZero(const GpuMat& src, GpuMat& buf)

    :param src: Single-channel source image.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

The function does not work with ``CV_64F`` images on GPUs with the compute capability < 1.3.

.. seealso:: :ocv:func:`countNonZero`



gpu::reduce
-----------
Reduces a matrix to a vector.

.. ocv:function:: void gpu::reduce(const GpuMat& mtx, GpuMat& vec, int dim, int reduceOp, int dtype = -1, Stream& stream = Stream::Null())

    :param mtx: Source 2D matrix.

    :param vec: Destination vector. Its size and type is defined by  ``dim``  and  ``dtype``  parameters.

    :param dim: Dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row. 1 means that the matrix is reduced to a single column.

    :param reduceOp: Reduction operation that could be one of the following:

            * **CV_REDUCE_SUM** The output is the sum of all rows/columns of the matrix.

            * **CV_REDUCE_AVG** The output is the mean vector of all rows/columns of the matrix.

            * **CV_REDUCE_MAX** The output is the maximum (column/row-wise) of all rows/columns of the matrix.

            * **CV_REDUCE_MIN** The output is the minimum (column/row-wise) of all rows/columns of the matrix.

    :param dtype: When it is negative, the destination vector will have the same type as the source matrix. Otherwise, its type will be  ``CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), mtx.channels())`` .
    
The function ``reduce`` reduces the matrix to a vector by treating the matrix rows/columns as a set of 1D vectors and performing the specified operation on the vectors until a single row/column is obtained. For example, the function can be used to compute horizontal and vertical projections of a raster image. In case of ``CV_REDUCE_SUM`` and ``CV_REDUCE_AVG`` , the output may have a larger element bit-depth to preserve accuracy. And multi-channel arrays are also supported in these two reduction modes.

.. seealso:: :ocv:func:`reduce`
