Matrix Reductions
=============================

.. highlight:: cpp

ocl::absSum
---------------
Returns the sum of absolute values for matrix elements.

.. ocv:function:: Scalar ocl::absSum(const oclMat &m)

    :param m: The Source image of all depth.

Counts the abs sum of matrix elements for each channel. Supports all data types.

ocl::countNonZero
---------------------
Returns the number of non-zero elements in src

.. ocv:function:: int ocl::countNonZero(const oclMat &src)

    :param src: Single-channel array

Counts non-zero array elements. Supports all data types.

ocl::min
------------------

.. ocv:function:: void ocl::min(const oclMat &src1, const oclMat &src2, oclMat &dst)

    :param src1: the first input array.

    :param src2: the second input array, must be the same size and same type as ``src1``.

    :param dst: the destination array, it will have the same size and same type as ``src1``.

Computes element-wise minima of two arrays. Supports all data types.

ocl::max
------------------

.. ocv:function:: void ocl::max(const oclMat &src1, const oclMat &src2, oclMat &dst)

    :param src1: the first input array.

    :param src2: the second input array, must be the same size and same type as ``src1``.

    :param dst: the destination array, it will have the same size and same type as ``src1``.

Computes element-wise maxima of two arrays. Supports all data types.

ocl::minMax
------------------
Returns void

.. ocv:function:: void ocl::minMax(const oclMat &src, double *minVal, double *maxVal = 0, const oclMat &mask = oclMat())

    :param src: Single-channel array

    :param minVal: Pointer to returned minimum value, should not be NULL

    :param maxVal: Pointer to returned maximum value, should not be NULL

    :param mask: The optional mask used to select a sub-array

Finds global minimum and maximum in a whole array or sub-array. Supports all data types.

ocl::minMaxLoc
------------------
Returns void

.. ocv:function:: void ocl::minMaxLoc(const oclMat &src, double *minVal, double *maxVal = 0, Point *minLoc = 0, Point *maxLoc = 0,const oclMat &mask = oclMat())

    :param src: Single-channel array

    :param minVal: Pointer to returned minimum value, should not be NULL

    :param maxVal: Pointer to returned maximum value, should not be NULL

    :param minLoc: Pointer to returned minimum location (in 2D case), should not be NULL

    :param maxLoc: Pointer to returned maximum location (in 2D case) should not be NULL

    :param mask: The optional mask used to select a sub-array

The functions minMaxLoc find minimum and maximum element values and their positions. The extremums are searched across the whole array, or, if mask is not an empty array, in the specified array region. The functions do not work with multi-channel arrays.

ocl::sqrSum
------------------
Returns the squared sum of matrix elements for each channel

.. ocv:function:: Scalar ocl::sqrSum(const oclMat &m)

    :param m: The Source image of all depth.

Counts the squared sum of matrix elements for each channel. Supports all data types.

ocl::sum
------------------
Returns the sum of matrix elements for each channel

.. ocv:function:: Scalar ocl::sum(const oclMat &m)

    :param m: The Source image of all depth.

Counts the sum of matrix elements for each channel.
