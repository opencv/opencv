Matrix Reductions
=================

.. highlight:: cpp

.. index:: gpu::meanStdDev

gpu::meanStdDev
-------------------
.. cpp:function:: void gpu::meanStdDev(const GpuMat\& mtx, Scalar\& mean, Scalar\& stddev)

    Computes a mean value and a standard deviation of matrix elements.

    :param mtx: Source matrix.  ``CV_8UC1``  matrices are supported for now.

    :param mean: Mean value.

    :param stddev: Standard deviation value.

.. seealso:: 
:ocv:func:`meanStdDev` 

.. index:: gpu::norm

gpu::norm
-------------
.. ocv:function:: double gpu::norm(const GpuMat\& src1, int normType=NORM_L2)
.. ocv:function:: double gpu::norm(const GpuMat\& src1, int normType, GpuMat\& buf)
.. ocv:function:: double norm(const GpuMat\& src1, const GpuMat\& src2, int normType=NORM_L2)

    Returns the norm of a matrix (or difference of two matrices).

    :param src1: Source matrix. Any matrices except 64F are supported.

    :param src2: Second source matrix (if any) with the same size and type as ``src1``.

    :param normType: Norm type.  ``NORM_L1`` ,  ``NORM_L2`` , and  ``NORM_INF``  are supported for now.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: 
:ocv:func:`norm`

.. index:: gpu::sum

gpu::sum
------------
.. ocv:function:: Scalar gpu::sum(const GpuMat\& src)

.. ocv:function:: Scalar gpu::sum(const GpuMat\& src, GpuMat\& buf)

    Returns the sum of matrix elements.

    :param src: Source image of any depth except for ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. seealso:: 
:ocv:func:`sum` 

.. index:: gpu::absSum

gpu::absSum
---------------
.. ocv:function:: Scalar gpu::absSum(const GpuMat\& src)

.. ocv:function:: Scalar gpu::absSum(const GpuMat\& src, GpuMat\& buf)

    Returns the sum of absolute values for matrix elements.

    :param src: Source image of any depth except for ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. index:: gpu::sqrSum

gpu::sqrSum
---------------
.. ocv:function:: Scalar gpu::sqrSum(const GpuMat\& src)

.. ocv:function:: Scalar gpu::sqrSum(const GpuMat\& src, GpuMat\& buf)

    Returns the squared sum of matrix elements.

    :param src: Source image of any depth except for ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

.. index:: gpu::minMax

gpu::minMax
---------------
.. ocv:function:: void gpu::minMax(const GpuMat\& src, double* minVal, double* maxVal=0, const GpuMat\& mask=GpuMat())

.. ocv:function:: void gpu::minMax(const GpuMat\& src, double* minVal, double* maxVal, const GpuMat\& mask, GpuMat\& buf)

    Finds global minimum and maximum matrix elements and returns their values.

    :param src: Single-channel source image.

    :param minVal: Pointer to the returned minimum value.  Use ``NULL``  if not required.

    :param maxVal: Pointer to the returned maximum value.  Use ``NULL``  if not required.

    :param mask: Optional mask to select a sub-matrix.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

The function does not work with ``CV_64F`` images on GPUs with the compute capability < 1.3.
	
.. seealso:: 
:ocv:func:`minMaxLoc` 

.. index:: gpu::minMaxLoc

gpu::minMaxLoc
------------------
.. ocv:function:: void gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, const GpuMat& mask=GpuMat())

.. ocv:function:: void gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, const GpuMat& mask, GpuMat& valbuf, GpuMat& locbuf)

    Finds global minimum and maximum matrix elements and returns their values with locations.

    :param src: Single-channel source image.

    :param minVal: Pointer to the returned minimum value. Use ``NULL``  if not required.

    :param maxVal: Pointer to the returned maximum value. Use ``NULL``  if not required.

    :param minValLoc: Pointer to the returned minimum location. Use ``NULL``  if not required.

    :param maxValLoc: Pointer to the returned maximum location. Use ``NULL``  if not required.

    :param mask: Optional mask to select a sub-matrix.

    :param valbuf: Optional values buffer to avoid extra memory allocations. It is resized automatically.

    :param locbuf: Optional locations buffer to avoid extra memory allocations. It is resized automatically.

	The function does not work with ``CV_64F`` images on GPU with the compute capability < 1.3.

.. seealso:: 
:ocv:func:`minMaxLoc` 

.. index:: gpu::countNonZero

gpu::countNonZero
---------------------
.. ocv:function:: int gpu::countNonZero(const GpuMat\& src)

.. ocv:function:: int gpu::countNonZero(const GpuMat\& src, GpuMat\& buf)

    Counts non-zero matrix elements.

    :param src: Single-channel source image.

    :param buf: Optional buffer to avoid extra memory allocations. It is resized automatically.

The function does not work with ``CV_64F`` images on GPUs with the compute capability < 1.3.
	
.. seealso:: 
:ocv:func:`countNonZero` 
