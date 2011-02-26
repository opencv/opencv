Matrix Reductions
=================

.. highlight:: cpp

.. index:: gpu::meanStdDev

cv::gpu::meanStdDev
-------------------
.. cfunction:: void meanStdDev(const GpuMat\& mtx, Scalar\& mean, Scalar\& stddev)

    Computes mean value and standard deviation of matrix elements.

    :param mtx: Source matrix.  ``CV_8UC1``  matrices are supported for now.

    :param mean: Mean value.

    :param stddev: Standard deviation value.

See also:
:func:`meanStdDev` .

.. index:: gpu::norm

cv::gpu::norm
-------------
.. cfunction:: double norm(const GpuMat\& src, int normType=NORM_L2)

    Returns norm of matrix (or of two matrices difference).

    :param src: Source matrix. Any matrices except 64F are supported.

    :param normType: Norm type.  ``NORM_L1`` ,  ``NORM_L2``  and  ``NORM_INF``  are supported for now.

.. cfunction:: double norm(const GpuMat\& src, int normType, GpuMat\& buf)

    * **src** Source matrix. Any matrices except 64F are supported.

    * **normType** Norm type.  ``NORM_L1`` ,  ``NORM_L2``  and  ``NORM_INF``  are supported for now.

    * **buf** Optional buffer to avoid extra memory allocations. It's resized automatically.

.. cfunction:: double norm(const GpuMat\& src1, const GpuMat\& src2,   int normType=NORM_L2)

    * **src1** First source matrix.  ``CV_8UC1``  matrices are supported for now.

    * **src2** Second source matrix. Must have the same size and type as  ``src1``
    .

    * **normType** Norm type.  ``NORM_L1`` ,  ``NORM_L2``  and  ``NORM_INF``  are supported for now.

See also:
:func:`norm` .

.. index:: gpu::sum

cv::gpu::sum
------------
.. cfunction:: Scalar sum(const GpuMat\& src)

.. cfunction:: Scalar sum(const GpuMat\& src, GpuMat\& buf)

    Returns sum of matrix elements.

    :param src: Source image of any depth except  ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

See also:
:func:`sum` .

.. index:: gpu::absSum

cv::gpu::absSum
---------------
.. cfunction:: Scalar absSum(const GpuMat\& src)

.. cfunction:: Scalar absSum(const GpuMat\& src, GpuMat\& buf)

    Returns sum of matrix elements absolute values.

    :param src: Source image of any depth except  ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

.. index:: gpu::sqrSum

cv::gpu::sqrSum
---------------
.. cfunction:: Scalar sqrSum(const GpuMat\& src)

.. cfunction:: Scalar sqrSum(const GpuMat\& src, GpuMat\& buf)

    Returns squared sum of matrix elements.

    :param src: Source image of any depth except  ``CV_64F`` .

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

.. index:: gpu::minMax

cv::gpu::minMax
---------------
.. cfunction:: void minMax(const GpuMat\& src, double* minVal,   double* maxVal=0, const GpuMat\& mask=GpuMat())

.. cfunction:: void minMax(const GpuMat\& src, double* minVal, double* maxVal,   const GpuMat\& mask, GpuMat\& buf)

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

cv::gpu::minMaxLoc
------------------
.. cfunction:: void minMaxLoc(const GpuMat\& src, double\* minVal, double* maxVal=0,   Point* minLoc=0, Point* maxLoc=0,   const GpuMat\& mask=GpuMat())

.. cfunction:: void minMaxLoc(const GpuMat\& src, double* minVal, double* maxVal,   Point* minLoc, Point* maxLoc, const GpuMat\& mask,   GpuMat\& valbuf, GpuMat\& locbuf)

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

cv::gpu::countNonZero
---------------------
.. cfunction:: int countNonZero(const GpuMat\& src)

.. cfunction:: int countNonZero(const GpuMat\& src, GpuMat\& buf)

    Counts non-zero matrix elements.

    :param src: Single-channel source image.

    :param buf: Optional buffer to avoid extra memory allocations. It's resized automatically.

Function doesn't work with ``CV_64F`` images on GPU with compute capability
:math:`<` 1.3.
See also:
:func:`countNonZero` .
