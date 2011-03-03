Matrix Reductions
=================

.. highlight:: cpp

.. index:: gpu::meanStdDev

gpu::meanStdDev
-------------------
.. c:function:: void gpu::meanStdDev(const GpuMat\& mtx, Scalar\& mean, Scalar\& stddev)

    Computes mean value and standard deviation of matrix elements.

    :param mtx: Source matrix.  ``CV_8UC1``  matrices are supported for now.

    :param mean: Mean value.

    :param stddev: Standard deviation value.

See also:
:func:`meanStdDev` .

.. index:: gpu::norm

gpu::norm
-------------
.. c:function:: double gpu::norm(const GpuMat\& src, int normType=NORM_L2)

    Returns norm of matrix (or of two matrices difference).

    :param src: Source matrix. Any matrices except 64F are supported.

    :param normType: Norm type.  ``NORM_L1`` ,  ``NORM_L2``  and  ``NORM_INF``  are supported for now.

.. c:function:: double norm(const GpuMat\& src, int normType, GpuMat\& buf)

    * **src** Source matrix. Any matrices except 64F are supported.

    * **normType** Norm type.  ``NORM_L1`` ,  ``NORM_L2``  and  ``NORM_INF``  are supported for now.

    * **buf** Optional buffer to avoid extra memory allocations. It's resized automatically.

.. c:function:: double norm(const GpuMat\& src1, const GpuMat\& src2,
   int normType=NORM_L2)

    * **src1** First source matrix.  ``CV_8UC1``  matrices are supported for now.

    * **src2** Second source matrix. Must have the same size and type as  ``src1``.

    * **normType** Norm type.  ``NORM_L1`` ,  ``NORM_L2``  and  ``NORM_INF``  are supported for now.

See also:
:func:`norm` .

.. index:: gpu::sum

gpu::sum
------------
.. c:function:: Scalar gpu::sum(const GpuMat\& src)

.. c:function:: Scalar gpu::sum(const GpuMat\& src, GpuMat\& buf)

    Returns sum of matrix elements.

    :param src: Source image of any depth except  ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

See also:
:func:`sum` .

.. index:: gpu::absSum

gpu::absSum
---------------
.. c:function:: Scalar gpu::absSum(const GpuMat\& src)

.. c:function:: Scalar gpu::absSum(const GpuMat\& src, GpuMat\& buf)

    Returns sum of matrix elements absolute values.

    :param src: Source image of any depth except  ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

.. index:: gpu::sqrSum

gpu::sqrSum
---------------
.. c:function:: Scalar gpu::sqrSum(const GpuMat\& src)

.. c:function:: Scalar gpu::sqrSum(const GpuMat\& src, GpuMat\& buf)

    Returns squared sum of matrix elements.

    :param src: Source image of any depth except  ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

.. index:: gpu::minMax

gpu::minMax
---------------
.. c:function:: void gpu::minMax(const GpuMat\& src, double* minVal, double* maxVal=0, const GpuMat\& mask=GpuMat())

.. c:function:: void gpu::minMax(const GpuMat\& src, double* minVal, double* maxVal, const GpuMat\& mask, GpuMat\& buf)

    Finds global minimum and maximum matrix elements and returns their values.

    :param src: Single-channel source image.

    :param minVal: Pointer to returned minimum value.  ``NULL``  if not required.

    :param maxVal: Pointer to returned maximum value.  ``NULL``  if not required.

    :param mask: Optional mask to select a sub-matrix.

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

Function doesn't work with ``CV_64F`` images on GPU with compute capability
:math:`<` 1.3.
See also:
:func:`minMaxLoc` .

.. index:: gpu::minMaxLoc

gpu::minMaxLoc
------------------
.. c:function:: void gpu::minMaxLoc(const GpuMat\& src, double\* minVal, double* maxVal=0,
   Point* minLoc=0, Point* maxLoc=0,
   const GpuMat\& mask=GpuMat())

.. c:function:: void gpu::minMaxLoc(const GpuMat\& src, double* minVal, double* maxVal,
   Point* minLoc, Point* maxLoc, const GpuMat\& mask,
   GpuMat\& valbuf, GpuMat\& locbuf)

    Finds global minimum and maximum matrix elements and returns their values with locations.

    :param src: Single-channel source image.

    :param minVal: Pointer to returned minimum value.  ``NULL``  if not required.

    :param maxVal: Pointer to returned maximum value.  ``NULL``  if not required.

    :param minValLoc: Pointer to returned minimum location.  ``NULL``  if not required.

    :param maxValLoc: Pointer to returned maximum location.  ``NULL``  if not required.

    :param mask: Optional mask to select a sub-matrix.

    :param valbuf: Optional values buffer to avoid extra memory allocations. It's resized automatically.

    :param locbuf: Optional locations buffer to avoid extra memory allocations. It's resized automatically.

Function doesn't work with ``CV_64F`` images on GPU with compute capability
:math:`<` 1.3.
See also:
:func:`minMaxLoc` .

.. index:: gpu::countNonZero

gpu::countNonZero
---------------------
.. c:function:: int gpu::countNonZero(const GpuMat\& src)

.. c:function:: int gpu::countNonZero(const GpuMat\& src, GpuMat\& buf)

    Counts non-zero matrix elements.

    :param src: Single-channel source image.

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

Function doesn't work with ``CV_64F`` images on GPU with compute capability
:math:`<` 1.3.
See also:
:func:`countNonZero` .
