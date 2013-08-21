Operations on Matrics
=============================

.. highlight:: cpp

ocl::oclMat::convertTo
----------------------
Returns void

.. ocv:function:: void ocl::oclMat::convertTo( oclMat &m, int rtype, double alpha = 1, double beta = 0 ) const

    :param m: The destination matrix. If it does not have a proper size or type before the operation, it will be reallocated

    :param rtype: The desired destination matrix type, or rather, the depth(since the number of channels will be the same with the source one). If rtype is negative, the destination matrix will have the same type as the source.

    :param alpha: must be default now

    :param beta: must be default now

The method converts source pixel values to the target datatype. saturate cast is applied in the end to avoid possible overflows. Supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32SC4, CV_32FC1, CV_32FC4.

ocl::oclMat::copyTo
-------------------
Returns void

.. ocv:function:: void ocl::oclMat::copyTo( oclMat &m, const oclMat &mask ) const

    :param m: The destination matrix. If it does not have a proper size or type before the operation, it will be reallocated

    :param mask(optional): The operation mask. Its non-zero elements indicate, which matrix elements need to be copied

Copies the matrix to another one. Supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32SC4, CV_32FC1, CV_32FC4

ocl::oclMat::setTo
------------------
Returns oclMat

.. ocv:function:: oclMat& ocl::oclMat::setTo(const Scalar &s, const oclMat &mask = oclMat())

    :param s: Assigned scalar, which is converted to the actual array type

    :param mask: The operation mask of the same size as ``*this``

Sets all or some of the array elements to the specified value. This is the advanced variant of Mat::operator=(const Scalar s) operator. Supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32SC4, CV_32FC1, CV_32FC4.

ocl::absdiff
------------------
Returns void

.. ocv:function:: void ocl::absdiff( const oclMat& a, const oclMat& b, oclMat& c )

.. ocv:function:: void ocl::absdiff( const oclMat& a, const Scalar& s, oclMat& c )


    :param a: The first input array

    :param b: The second input array, must be the same size and same type as a

    :param s: Scalar, the second input parameter

    :param c: The destination array, it will have the same size and same type as a

Computes per-element absolute difference between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::add
------------------
Returns void

.. ocv:function:: void ocl::add( const oclMat & a, const oclMat & b, oclMat & c )

.. ocv:function:: void ocl::add( const oclMat & a, const oclMat & b, oclMat & c, const oclMat & mask )

.. ocv:function:: void ocl::add( const oclMat & a, const Scalar & sc, oclMat & c, const oclMat & mask=oclMat() )

    :param a: The first input array

    :param b: The second input array, must be the same size and same type as src1

    :param sc: Scalar, the second input parameter

    :param c: The destination array, it will have the same size and same type as src1

    :param mask: he optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

Computes per-element additon between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::subtract
------------------
Returns void

.. ocv:function:: void ocl::subtract( const oclMat& a, const oclMat& b, oclMat& c )

.. ocv:function:: void ocl::subtract( const oclMat& a, const oclMat& b, oclMat& c, const oclMat& mask )

.. ocv:function:: void ocl::subtract( const oclMat& a, const Scalar& sc, oclMat& c, const oclMat& mask=oclMat() )

.. ocv:function:: void ocl::subtract( const Scalar& sc, const oclMat& a, oclMat& c, const oclMat& mask=oclMat() )


    :param a: The first input array

    :param b: The second input array, must be the same size and same type as src1

    :param sc: Scalar, the second input parameter

    :param c: The destination array, it will have the same size and same type as src1

    :param mask: he optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

Computes per-element subtract between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::multiply
------------------
Returns void

.. ocv:function:: void ocl::multiply( const oclMat& a, const oclMat& b, oclMat& c, double scale=1 )

    :param a: The first input array

    :param b: The second input array, must be the same size and same type as src1

    :param c: The destination array, it will have the same size and same type as src1

    :param scale: must be 1 now

Computes per-element multiply between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::divide
------------------
Returns void

.. ocv:function:: void ocl::divide( const oclMat& a, const oclMat& b, oclMat& c, double scale=1 )

.. ocv:function:: void ocl::divide( double scale, const oclMat& b, oclMat& c )

    :param a: The first input array

    :param b: The second input array, must be the same size and same type as src1

    :param c: The destination array, it will have the same size and same type as src1

    :param scale: must be 1 now

Computes per-element divide between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::bitwise_and
------------------
Returns void

.. ocv:function:: void ocl::bitwise_and( const oclMat& src1, const oclMat& src2, oclMat& dst, const oclMat& mask=oclMat() )

.. ocv:function:: void ocl::bitwise_and( const oclMat& src1, const Scalar& s, oclMat& dst, const oclMat& mask=oclMat() )

    :param src1: The first input array

    :param src2: The second input array, must be the same size and same type as src1

    :param s: Scalar, the second input parameter

    :param dst: The destination array, it will have the same size and same type as src1

    :param mask: The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

Computes per-element bitwise_and between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::bitwise_or
------------------
Returns void

.. ocv:function:: void ocl::bitwise_or( const oclMat& src1, const oclMat& src2, oclMat& dst, const oclMat& mask=oclMat() )

.. ocv:function:: void ocl::bitwise_or( const oclMat& src1, const Scalar& s, oclMat& dst, const oclMat& mask=oclMat() )

    :param src1: The first input array

    :param src2: The second input array, must be the same size and same type as src1

    :param s: Scalar, the second input parameter

    :param dst: The destination array, it will have the same size and same type as src1

    :param mask: The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

Computes per-element bitwise_or between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::bitwise_xor
------------------
Returns void

.. ocv:function:: void ocl::bitwise_xor( const oclMat& src1, const oclMat& src2, oclMat& dst, const oclMat& mask=oclMat() )

.. ocv:function:: void ocl::bitwise_xor( const oclMat& src1, const Scalar& s, oclMat& dst, const oclMat& mask=oclMat() )

    :param src1: The first input array

    :param src2: The second input array, must be the same size and same type as src1

    :param sc: Scalar, the second input parameter

    :param dst: The destination array, it will have the same size and same type as src1

    :param mask: The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

Computes per-element bitwise_xor between two arrays or between array and a scalar. Supports all data types except CV_8S.

ocl::bitwise_not
------------------
Returns void

.. ocv:function:: void ocl::bitwise_not(const oclMat &src, oclMat &dst)

    :param src: The input array

    :param dst: The destination array, it will have the same size and same type as src1

The functions bitwise not compute per-element bit-wise inversion of the source array:. Supports all data types except CV_8S.

ocl::cartToPolar
------------------
Returns void

.. ocv:function:: void ocl::cartToPolar(const oclMat &x, const oclMat &y, oclMat &magnitude, oclMat &angle, bool angleInDegrees = false)

    :param x: The array of x-coordinates; must be single-precision or double-precision floating-point array

    :param y: The array of y-coordinates; it must have the same size and same type as x

    :param magnitude: The destination array of magnitudes of the same size and same type as x

    :param angle: The destination array of angles of the same size and same type as x. The angles are measured in radians (0 to 2pi ) or in degrees (0 to 360 degrees).

    :param angleInDegrees: The flag indicating whether the angles are measured in radians, which is default mode, or in degrees

Calculates the magnitude and angle of 2d vectors. Supports only CV_32F and CV_64F data types.

ocl::polarToCart
------------------
Returns void

.. ocv:function:: void ocl::polarToCart(const oclMat &magnitude, const oclMat &angle, oclMat &x, oclMat &y, bool angleInDegrees = false)

    :param magnitude: The source floating-point array of magnitudes of 2D vectors. It can be an empty matrix (=Mat()) - in this case the function assumes that all the magnitudes are =1. If it's not empty, it must have the same size and same type as angle

    :param angle: The source floating-point array of angles of the 2D vectors

    :param x: The destination array of x-coordinates of 2D vectors; will have the same size and the same type as angle

    :param y: The destination array of y-coordinates of 2D vectors; will have the same size and the same type as angle

    :param angleInDegrees: The flag indicating whether the angles are measured in radians, which is default mode, or in degrees

The function polarToCart computes the cartesian coordinates of each 2D vector represented by the corresponding elements of magnitude and angle. Supports only CV_32F and CV_64F data types.

ocl::compare
------------------
Returns void

.. ocv:function:: void ocl::compare(const oclMat &a, const oclMat &b, oclMat &c, int cmpop)

    :param a: The first source array

    :param b: The second source array; must have the same size and same type as a

    :param c: The destination array; will have the same size as a

    :param cmpop: The flag specifying the relation between the elements to be checked

Performs per-element comparison of two arrays or an array and scalar value. Supports all the 1 channel data types except CV_8S.

ocl::exp
------------------
Returns void

.. ocv:function:: void ocl::exp(const oclMat &a, oclMat &b)

    :param a: The first source array

    :param b: The dst array; must have the same size and same type as a

The function exp calculates the exponent of every element of the input array. Supports only CV_32FC1 data type.

ocl::log
------------------
Returns void

.. ocv:function:: void ocl::log(const oclMat &a, oclMat &b)

    :param a: The first source array

    :param b: The dst array; must have the same size and same type as a

The function log calculates the log of every element of the input array. Supports only CV_32FC1 data type.

ocl::LUT
------------------
Returns void

.. ocv:function:: void ocl::LUT(const oclMat &src, const oclMat &lut, oclMat &dst)

    :param src: Source array of 8-bit elements

    :param lut: Look-up table of 256 elements. In the case of multi-channel source array, the table should either have a single channel (in this case the same table is used for all channels) or the same number of channels as in the source array

    :param dst: Destination array; will have the same size and the same number of channels as src, and the same depth as lut

Performs a look-up table transform of an array. Supports only CV_8UC1 and CV_8UC4 data type.

ocl::magnitude
------------------
Returns void

.. ocv:function:: void ocl::magnitude(const oclMat &x, const oclMat &y, oclMat &magnitude)

    :param x: The floating-point array of x-coordinates of the vectors

    :param y: he floating-point array of y-coordinates of the vectors; must have the same size as x

    :param magnitude: The destination array; will have the same size and same type as x

The function magnitude calculates magnitude of 2D vectors formed from the corresponding elements of x and y arrays. Supports only CV_32F and CV_64F data type.

ocl::flip
------------------
Returns void

.. ocv:function:: void ocl::flip( const oclMat& a, oclMat& b, int flipCode )

    :param a: Source image.

    :param b: Destination image

    :param flipCode: Specifies how to flip the array: 0 means flipping around the x-axis, positive (e.g., 1) means flipping around y-axis, and negative (e.g., -1) means flipping around both axes.

The function flip flips the array in one of three different ways (row and column indices are 0-based). Supports all data types.

ocl::meanStdDev
------------------
Returns void

.. ocv:function:: void ocl::meanStdDev(const oclMat &mtx, Scalar &mean, Scalar &stddev)

    :param mtx: Source image.

    :param mean: The output parameter: computed mean value

    :param stddev: The output parameter: computed standard deviation

The functions meanStdDev compute the mean and the standard deviation M of array elements, independently for each channel, and return it via the output parameters. Supports all data types except CV_32F,CV_64F

ocl::merge
------------------
Returns void

.. ocv:function:: void ocl::merge(const vector<oclMat> &src, oclMat &dst)

    :param src: The source array or vector of the single-channel matrices to be merged. All the matrices in src must have the same size and the same type

    :param dst: The destination array; will have the same size and the same depth as src, the number of channels will match the number of source matrices

Composes a multi-channel array from several single-channel arrays. Supports all data types.

ocl::split
------------------
Returns void

.. ocv:function:: void ocl::split(const oclMat &src, vector<oclMat> &dst)

    :param src: The source multi-channel array

    :param dst: The destination array or vector of arrays; The number of arrays must match src.channels(). The arrays themselves will be reallocated if needed

The functions split split multi-channel array into separate single-channel arrays. Supports all data types.

ocl::norm
------------------
Returns the calculated norm

.. ocv:function:: double ocl::norm(const oclMat &src1, int normType = NORM_L2)

.. ocv:function:: double ocl::norm(const oclMat &src1, const oclMat &src2, int normType = NORM_L2)

    :param src1: The first source array

    :param src2: The second source array of the same size and the same type as src1

    :param normType: Type of the norm

Calculates absolute array norm, absolute difference norm, or relative difference norm. Supports only CV_8UC1 data type.

ocl::phase
------------------
Returns void

.. ocv:function:: void ocl::phase(const oclMat &x, const oclMat &y, oclMat &angle, bool angleInDegrees = false)

    :param x: The source floating-point array of x-coordinates of 2D vectors

    :param y: The source array of y-coordinates of 2D vectors; must have the same size and the same type as x

    :param angle: The destination array of vector angles; it will have the same size and same type as x

    :param angleInDegrees: When it is true, the function will compute angle in degrees, otherwise they will be measured in radians

The function phase computes the rotation angle of each 2D vector that is formed from the corresponding elements of x and y. Supports only CV_32FC1 and CV_64FC1 data type.

ocl::pow
------------------
Returns void

.. ocv:function:: void ocl::pow(const oclMat &x, double p, oclMat &y)

    :param x: The source array

    :param power: The exponent of power;The source floating-point array of angles of the 2D vectors

    :param y: The destination array, should be the same type as the source

The function pow raises every element of the input array to p. Supports only CV_32FC1 and CV_64FC1 data type.

ocl::transpose
------------------
Returns void

.. ocv:function:: void ocl::transpose(const oclMat &src, oclMat &dst)

    :param src: The source array

    :param dst: The destination array of the same type as src

Transposes a matrix. Supports 8UC1, 8UC4, 8SC4, 16UC2, 16SC2, 32SC1 and 32FC1 data types.


ocl::dft
------------
Performs a forward or inverse discrete Fourier transform (1D or 2D) of the floating point matrix.

.. ocv:function:: void ocl::dft( const oclMat& src, oclMat& dst, Size dft_size=Size(0, 0), int flags=0 )

    :param src: Source matrix (real or complex).

    :param dst: Destination matrix (real or complex).

    :param dft_size: Size of original input, which is used for transformation from complex to real.

    :param flags: Optional flags:

        * **DFT_ROWS** transforms each individual row of the source matrix.

        * **DFT_COMPLEX_OUTPUT** performs a forward transformation of 1D or 2D real array. The result, though being a complex array, has complex-conjugate symmetry (*CCS*, see the function description below for details). Such an array can be packed into a real array of the same size as input, which is the fastest option and which is what the function does by default. However, you may wish to get a full complex array (for simpler spectrum analysis, and so on). Pass the flag to enable the function to produce a full-size complex output array.

        * **DFT_INVERSE** inverts DFT. Use for complex-complex cases (real-complex and complex-real cases are always forward and inverse, respectively).

        * **DFT_REAL_OUTPUT** specifies the output as real. The source matrix is the result of real-complex transform, so the destination matrix must be real.

Use to handle real matrices ( ``CV32FC1`` ) and complex matrices in the interleaved format ( ``CV32FC2`` ).

The dft_size must be powers of 2, 3 and 5. Real to complex dft output is not the same with cpu version. real to complex and complex to real does not support DFT_ROWS

.. seealso:: :ocv:func:`dft`

ocl::gemm
------------------
Performs generalized matrix multiplication.

.. ocv:function:: void ocl::gemm(const oclMat& src1, const oclMat& src2, double alpha, const oclMat& src3, double beta, oclMat& dst, int flags = 0)

    :param src1: First multiplied input matrix that should be ``CV_32FC1`` type.

    :param src2: Second multiplied input matrix of the same type as  ``src1`` .

    :param alpha: Weight of the matrix product.

    :param src3: Third optional delta matrix added to the matrix product. It should have the same type as  ``src1``  and  ``src2`` .

    :param beta: Weight of  ``src3`` .

    :param dst: Destination matrix. It has the proper size and the same type as input matrices.

    :param flags: Operation flags:

            * **GEMM_1_T** transpose  ``src1``
            * **GEMM_2_T** transpose  ``src2``

.. seealso:: :ocv:func:`gemm`

ocl::sortByKey
------------------
Returns void

.. ocv:function:: void ocl::sortByKey(oclMat& keys, oclMat& values, int method, bool isGreaterThan = false)

    :param keys:   The keys to be used as sorting indices.

    :param values: The array of values.

    :param isGreaterThan: Determine sorting order.

    :param method: supported sorting methods:
            * **SORT_BITONIC**   bitonic sort, only support power-of-2 buffer size
            * **SORT_SELECTION** selection sort, currently cannot sort duplicate keys
            * **SORT_MERGE**     merge sort
            * **SORT_RADIX**     radix sort, only support signed int/float keys(``CV_32S``/``CV_32F``)

Returns the sorted result of all the elements in values based on equivalent keys.

The element unit in the values to be sorted is determined from the data type,
i.e., a ``CV_32FC2`` input ``{a1a2, b1b2}`` will be considered as two elements, regardless its matrix dimension.

Both keys and values will be sorted inplace.

Keys needs to be a **single** channel `oclMat`.

Example::
    input -
    keys   = {2,    3,   1}   (CV_8UC1)
    values = {10,5, 4,3, 6,2} (CV_8UC2)
    sortByKey(keys, values, SORT_SELECTION, false);
    output -
    keys   = {1,    2,   3}   (CV_8UC1)
    values = {6,2, 10,5, 4,3} (CV_8UC2)
