Operations on Arrays
====================

.. highlight:: cpp

.. index:: abs

.. _abs:

abs
-------
.. c:function:: MatExpr<...> abs(const Mat& src)

.. c:function:: MatExpr<...> abs(const MatExpr<...>& src)

    Computes an absolute value of each matrix element.

    :param src: Matrix or matrix expression.
    
``abs`` is a meta-function that is expanded to one of :func:`absdiff` forms:

    * ``C = abs(A-B)``     is equivalent to ``absdiff(A, B, C)``     

    * ``C = abs(A)``     is equivalent to ``absdiff(A, Scalar::all(0), C)``     .

    * ``C = Mat_<Vec<uchar,n> >(abs(A*alpha + beta))``     is equivalent to ``convertScaleAbs(A, C, alpha, beta)``
    
The output matrix has the same size and the same type as the input one (except for the last case, where ``C`` will be ``depth=CV_8U`` ).

See Also: :ref:`MatrixExpressions`, :func:`absdiff`

.. index:: absdiff

.. _absdiff:

absdiff
-----------

.. c:function:: void absdiff(const Mat& src1, const Mat& src2, Mat& dst)
.. c:function:: void absdiff(const Mat& src1, const Scalar& sc, Mat& dst)

    Computes the per-element absolute difference between 2 arrays or between an array and a scalar.

    :param src1: The first input array.
    :param src2: The second input array of the same size and type as  ``src1`` .
    
    :param sc: A scalar. This is the second input parameter.
    
    :param dst: The destination array of the same size and type as  ``src1`` . See  ``Mat::create`` .
    
The functions ``absdiff`` compute:

 * absolute difference between two arrays:

    .. math::
        \texttt{dst} (I) =  \texttt{saturate} (| \texttt{src1} (I) -  \texttt{src2} (I)|)

 * absolute difference between an array and a scalar:

    .. math::
        \texttt{dst} (I) =  \texttt{saturate} (| \texttt{src1} (I) -  \texttt{sc} |)

where  ``I`` is a multi-dimensional index of array elements.
In case of multi-channel arrays, each channel is processed independently.

See Also: :func:`abs`

.. index:: add

.. _add:

add
-------
.. c:function:: void add(const Mat& src1, const Mat& src2, Mat& dst)

.. c:function:: void add(const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask)

.. c:function:: void add(const Mat& src1, const Scalar& sc, Mat& dst, const Mat& mask=Mat())

    Computes the per-element sum of two arrays or an array and a scalar.

    :param src1: The first source array.

    :param src2: The second source array of the same size and type as  ``src1`` .
    
    :param sc: A scalar. This is the second input parameter.

    :param dst: The destination array of the same size and type as  ``src1`` . See  ``Mat::create`` .
    
    :param mask: An optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The functions ``add`` compute:

*
    the sum of two arrays:

    .. math::

        \texttt{dst} (I) =  \texttt{saturate} ( \texttt{src1} (I) +  \texttt{src2} (I)) \quad \texttt{if mask} (I) \ne0

*
    the sum of an array and a scalar:

    .. math::

        \texttt{dst} (I) =  \texttt{saturate} ( \texttt{src1} (I) +  \texttt{sc} ) \quad \texttt{if mask} (I) \ne0

where ``I`` is a multi-dimensional index of array elements.

The first function in the list above can be replaced with matrix expressions: ::

    dst = src1 + src2;
    dst += src1; // equivalent to add(dst, src1, dst);


In case of multi-channel arrays, each channel is processed independently.

See Also:
:func:`subtract`,:func:`addWeighted`,:func:`scaleAdd`,:func:`convertScale`,:ref:`MatrixExpressions`

.. index:: addWeighted

.. _addWeighted:

addWeighted
---------------
.. c:function:: void addWeighted(const Mat& src1, double alpha, const Mat& src2, double beta, double gamma, Mat& dst)

    Computes the weighted sum of two arrays.

    :param src1: The first source array.

    :param alpha: Weight for the first array elements.

    :param src2: The second source array of the same size and type as  ``src1`` .
    
    :param beta: Weight for the second array elements.

    :param dst: The destination array of the same size and type as  ``src1`` .
    
    :param gamma: A scalar added to each sum.

The functions ``addWeighted`` calculate the weighted sum of two arrays as follows:

.. math::

    \texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )

where ``I`` is a multi-dimensional index of array elements.

The first function can be replaced with a matrix expression: ::

    dst = src1*alpha + src2*beta + gamma;


In case of multi-channel arrays, each channel is processed independently.

See Also:
:func:`add`,:func:`subtract`,:func:`scaleAdd`,:func:`convertScale`,:ref:`MatrixExpressions`

.. index:: bitwise_and

.. _bitwise_and_:

bitwise_and
-----------
.. c:function:: void bitwise_and(const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask=Mat())

.. c:function:: void bitwise_and(const Mat& src1, const Scalar& sc, Mat& dst, const Mat& mask=Mat())

    Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar.

    :param src1: The first source array.

    :param src2: The second source array of the same size and type as  ``src1`` .
    
    :param sc: A scalar. This is the second input parameter.

    :param dst: The destination array of the same size and type as  ``src1`` . See  ``Mat::create`` .    
    
    :param mask: An optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The functions ``bitwise_and`` compute the per-element bit-wise logical conjunction:

*
    of two arrays

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0

*
    an array and a scalar:

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{sc} \quad \texttt{if mask} (I) \ne0

In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently.

See Also: ??

.. index:: bitwise_not

.. _bitwise_not_:

bitwise_not
-----------
.. c:function:: void bitwise_not(const Mat& src, Mat& dst)

    Inverts every bit of an array.

    :param src1: The source array.

    :param dst: The destination array. It is reallocated to be of the same size and type as  ``src`` . See  ``Mat::create`` .
    
    :param mask: An optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The functions ``bitwise_not`` compute per-element bit-wise inversion of the source array:

.. math::

    \texttt{dst} (I) =  \neg \texttt{src} (I)

In case of a floating-point source array, its machine-specific bit representation (usually IEEE754-compliant) is used for the operation. In case of multi-channel arrays, each channel is processed independently.

.. index:: bitwise_or

.. _bitwise_or_:

bitwise_or
----------
.. c:function:: void bitwise_or(const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask=Mat())

.. c:function:: void bitwise_or(const Mat& src1, const Scalar& sc, Mat& dst, const Mat& mask=Mat())

    Calculates the per-element bit-wise disjunction of two arrays or an array and a scalar.

    :param src1: The first source array.

    :param src2: The second source array of the same size and type as  ``src1`` .
    
    :param sc: A scalar. This is the second input parameter.

    :param dst: The destination array. It is reallocated to be of the same size and type as  ``src1`` . See  ``Mat::create`` .
    
    :param mask: An optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The functions ``bitwise_or`` compute the per-element bit-wise logical disjunction:

*
    of two arrays

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0

*
    an array and a scalar:

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{sc} \quad \texttt{if mask} (I) \ne0

In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently.

.. index:: bitwise_xor

.. _bitwise_xor_:

bitwise_xor
-----------
.. c:function:: void bitwise_xor(const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask=Mat())

.. c:function:: void bitwise_xor(const Mat& src1, const Scalar& sc, Mat& dst, const Mat& mask=Mat())

    Calculates the per-element bit-wise "exclusive or" operation on two arrays or an array and a scalar.

    :param src1: The first source array.

    :param src2: The second source array of the same size and type as  ``src1`` .
    
    :param sc: A scalar. This is the second input parameter.

    :param dst: The destination array. It is reallocated to be of the same size and type as  ``src1`` . See  ``Mat::create`` .
    
    :param mask: An optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The functions ``bitwise_xor`` compute the per-element bit-wise logical "exclusive or" operation:

 * on two arrays

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0

 * an array and a scalar:

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{sc} \quad \texttt{if mask} (I) \ne0

In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently.

.. index:: calcCovarMatrix

.. _calcCovarMatrix:

calcCovarMatrix
---------------

.. c:function:: void calcCovarMatrix( const Mat* samples, int nsamples, Mat& covar, Mat& mean, int flags, int ctype=CV_64F)

.. c:function:: void calcCovarMatrix( const Mat& samples, Mat& covar, Mat& mean, int flags, int ctype=CV_64F)

    Calculates the covariance matrix of a set of vectors.

    :param samples: Samples stored either as separate matrices or as rows/columns of a single matrix.

    :param nsamples: The number of samples when they are stored separately.

    :param covar: The output covariance matrix of the type= ``ctype``  and square size.

    :param mean: The input or output (depending on the flags) array - the mean (average) vector of the input vectors.

    :param flags: Operation flags, a combination of the following values:

            * **CV_COVAR_SCRAMBLED** The output covariance matrix is calculated as:

                .. math::

                      \texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]^T  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...],
                      
                The covariance matrix will be  :math:`\texttt{nsamples} \times \texttt{nsamples}` . Such an unusual covariance matrix is used for fast PCA of a set of very large vectors (see, for example, the EigenFaces technique for face recognition). Eigenvalues of this "scrambled" matrix match the eigenvalues of the true covariance matrix. The "true" eigenvectors can be easily calculated from the eigenvectors of the "scrambled" covariance matrix.

            * **CV_COVAR_NORMAL** The output covariance matrix is calculated as:

                .. math::

                      \texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...]^T,
                      
                ``covar``  will be a square matrix of the same size as the total number of elements in each input vector. One and only one of  ``CV_COVAR_SCRAMBLED``  and ``CV_COVAR_NORMAL``  must be specified.

            * **CV_COVAR_USE_AVG** If the flag is specified, the function does not calculate  ``mean``  from the input vectors but, instead, uses the passed  ``mean``  vector. This is useful if  ``mean``  has been pre-computed or known in advance, or if the covariance matrix is calculated by parts. In this case, ``mean``  is not a mean vector of the input sub-set of vectors but rather the mean vector of the whole set.

            * **CV_COVAR_SCALE** If the flag is specified, the covariance matrix is scaled. In the "normal" mode,  ``scale``  is  ``1./nsamples`` . In the "scrambled" mode,  ``scale``  is the reciprocal of the total number of elements in each input vector. By default (if the flag is not specified), the covariance matrix is not scaled (  ``scale=1`` ).

            * **CV_COVAR_ROWS** [Only useful in the second variant of the function] If the flag is specified, all the input vectors are stored as rows of the  ``samples``  matrix.  ``mean``  should be a single-row vector in this case.

            * **CV_COVAR_COLS** [Only useful in the second variant of the function] If the flag is specified, all the input vectors are stored as columns of the  ``samples``  matrix.  ``mean``  should be a single-column vector in this case.

The functions ``calcCovarMatrix`` calculate the covariance matrix and, optionally, the mean vector of the set of input vectors.

See Also:
:func:`PCA`,:func:`mulTransposed`,:func:`Mahalanobis`

.. index:: cartToPolar

.. _cartToPolar:

cartToPolar
-----------

.. c:function:: void cartToPolar(const Mat& x, const Mat& y, Mat& magnitude, Mat& angle, bool angleInDegrees=false)

    Calculates the magnitude and angle of 2D vectors.

    :param x: The array of x-coordinates. This must be a single-precision or double-precision floating-point array.

    :param y: The array of y-coordinates. It must have the same size and same type as  ``x`` .
    
    :param magnitude: The destination array of magnitudes of the same size and type as  ``x`` .
    
    :param angle: The destination array of angles of the same size and type as  ``x`` . The angles are measured in radians  :math:`(0`  to  :math:`2 \pi )`  or in degrees (0 to 360 degrees).

    :param angleInDegrees: The flag indicating whether the angles are measured in radians, which is a default mode, or in degrees.

The function ``cartToPolar`` calculates either the magnitude, angle, or both for every 2D vector (x(I),y(I)):

.. math::

    \begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}

The angles are calculated with
:math:`\sim\,0.3^\circ` accuracy. For the point (0,0) , the angle is set to 0.

.. index:: checkRange

.. _checkRange:

checkRange
----------

.. c:function:: bool checkRange(const Mat& src, bool quiet=true, Point* pos=0, double minVal=-DBL_MAX, double maxVal=DBL_MAX)

    Checks every element of an input array for invalid values.

    :param src: The array to check.

    :param quiet: The flag indicating whether the functions quietly return false when the array elements are out of range or they throw an exception.

    :param pos: An optional output parameter, where the position of the first outlier is stored. In the second function  ``pos`` , when not NULL, must be a pointer to array of  ``src.dims``  elements.

    :param minVal: The inclusive lower boundary of valid values range.

    :param maxVal: The exclusive upper boundary of valid values range.

The functions ``checkRange`` check that every array element is neither NaN nor
:math:`\pm \infty` . When ``minVal < -DBL_MAX`` and ``maxVal < DBL_MAX`` , the functions also check that each value is between ``minVal`` and ``maxVal`` . In case of multi-channel arrays, each channel is processed independently.
If some values are out of range, position of the first outlier is stored in ``pos`` (when
:math:`\texttt{pos}\ne0` ). Then, the functions either return false (when ``quiet=true`` ) or throw an exception.

.. index:: compare

.. _compare:

compare
-------

.. c:function:: void compare(const Mat& src1, const Mat& src2, Mat& dst, int cmpop)

.. c:function:: void compare(const Mat& src1, double value, Mat& dst, int cmpop)

    Performs the per-element comparison of two arrays or an array and scalar value.

    :param src1: The first source array.

    :param src2: The second source array of the same size and type as  ``src1`` .
    
    :param value: A scalar value to compare each array element with.

    :param dst: The destination array of the same size as  ``src1``  and type= ``CV_8UC1`` .
    
    :param cmpop: The flag specifying the relation between the elements to be checked.

            * **CMP_EQ** :math:`\texttt{src1}(I) = \texttt{src2}(I)`  or  :math:`\texttt{src1}(I) = \texttt{value}`
            * **CMP_GT** :math:`\texttt{src1}(I) > \texttt{src2}(I)`  or  :math:`\texttt{src1}(I) > \texttt{value}`
            * **CMP_GE** :math:`\texttt{src1}(I) \geq \texttt{src2}(I)`  or  :math:`\texttt{src1}(I) \geq \texttt{value}`             
            * **CMP_LT** :math:`\texttt{src1}(I) < \texttt{src2}(I)`  or  :math:`\texttt{src1}(I) < \texttt{value}`             
            * **CMP_LE** :math:`\texttt{src1}(I) \leq \texttt{src2}(I)`  or  :math:`\texttt{src1}(I) \leq \texttt{value}`             
            * **CMP_NE** :math:`\texttt{src1}(I) \ne \texttt{src2}(I)`  or  :math:`\texttt{src1}(I) \ne \texttt{value}`
            
The functions ``compare`` compare each element of ``src1`` with the corresponding element of ``src2`` or with the real scalar ``value`` . When the comparison result is true, the corresponding element of destination array is set to 255. Otherwise, it is set to 0:

    * ``dst(I) = src1(I) cmpop src2(I) ? 255 : 0``
    * ``dst(I) = src1(I) cmpop value ? 255 : 0``
    
The comparison operations can be replaced with the equivalent matrix expressions: ::

    Mat dst1 = src1 >= src2;
    Mat dst2 = src1 < 8;
    ...


See Also:
:func:`checkRange`,:func:`min`,:func:`max`,:func:`threshold`,:ref:`MatrixExpressions`

.. index:: completeSymm

.. _completeSymm:

completeSymm
------------

.. c:function:: void completeSymm(Mat& mtx, bool lowerToUpper=false)

    Copies the lower or the upper half of a square matrix to another half.

    :param mtx: Input-output floating-point square matrix.

    :param lowerToUpper: If true, the lower half is copied to the upper half. Otherwise, the upper half is copied to the lower half.

The function ``completeSymm`` copies the lower half of a square matrix to its another half. The matrix diagonal remains unchanged:

*
    :math:`\texttt{mtx}_{ij}=\texttt{mtx}_{ji}`     for
    :math:`i > j`     if ``lowerToUpper=false``
    
*
    :math:`\texttt{mtx}_{ij}=\texttt{mtx}_{ji}`     for
    :math:`i < j`     if ``lowerToUpper=true``
    
See Also: :func:`flip`,:func:`transpose`

.. index:: convertScaleAbs

.. _convertScaleAbs:

convertScaleAbs
---------------

.. c:function:: void convertScaleAbs(const Mat& src, Mat& dst, double alpha=1, double beta=0)

    Scales, computes absolute values, and converts the result to 8-bit.

    :param src: The source array.

    :param dst: The destination array.

    :param alpha: An optional scale factor.

    :param beta: An optional delta added to the scaled values.

On each element of the input array, the function ``convertScaleAbs`` performs three operations sequentially: scaling, taking an absolute value, conversion to an unsigned 8-bit type:

.. math::

    \texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)

In case of multi-channel arrays, the function processes each channel independently. When the output is not 8-bit, the operation can be emulated by calling the ``Mat::convertTo`` method (or by using matrix expressions) and then by computing an absolute value of the result. For example: ::

    Mat_<float> A(30,30);
    randu(A, Scalar(-100), Scalar(100));
    Mat_<float> B = A*5 + 3;
    B = abs(B);
    // Mat_<float> B = abs(A*5+3) will also do the job,
    // but it will allocate a temporary matrix


See Also:
:func:`Mat::convertTo`,:func:`abs`

.. index:: countNonZero

.. _countNonZero:

countNonZero
------------

.. c:function:: int countNonZero( const Mat& mtx )

    Counts non-zero array elements.

    :param mtx: Single-channel array.

The function ``cvCountNonZero`` returns the number of non-zero elements in ``mtx`` :

.. math::

    \sum _{I: \; \texttt{mtx} (I) \ne0 } 1

See Also:
:func:`mean`,:func:`meanStdDev`,:func:`norm`,:func:`minMaxLoc`,:func:`calcCovarMatrix`

.. index:: cubeRoot

.. _cubeRoot:

cubeRoot
--------

.. c:function:: float cubeRoot(float val)

    Computes the cube root of an argument.

    :param val: A function argument.

The function ``cubeRoot`` computes :math:`\sqrt[3]{\texttt{val}}`. Negative arguments are handled correctly. *NaN*
and :math:`\pm\infty` are not handled. The accuracy approaches the maximum possible accuracy for single-precision data.

.. index:: cvarrToMat

.. _cvarrToMat:

cvarrToMat
----------

.. c:function:: Mat cvarrToMat(const CvArr* src, bool copyData=false, bool allowND=true, int coiMode=0)

    Converts ``CvMat``, ``IplImage`` , or ``CvMatND`` to ``Mat``.

    :param src: The source ``CvMat``, ``IplImage`` , or  ``CvMatND`` .
    
    :param copyData: When it is false (default value), no data is copied and only the new header is created. In this case, the original array should not be deallocated while the new matrix header is used. If the parameter is true, all the data is copied and you may deallocate the original array right after the conversion.

    :param allowND: When it is true (default value),  ``CvMatND``  is converted to  ``Mat`` , if it is possible (for example, when the data is contiguous). If it is not possible, or when the parameter is false, the function will report an error.

    :param coiMode: The parameter specifies how the IplImage COI (when set) is handled.

        *  If  ``coiMode=0`` , the function reports an error if COI is set.

        *  If  ``coiMode=1`` , the function never reports an error. Instead, it returns the header to the whole original image and you will have to check and process COI manually. See  :func:`extractImageCOI` .

The function ``cvarrToMat`` converts ``CvMat``, ``IplImage`` , or ``CvMatND`` header to
:func:`Mat` header, and optionally duplicates the underlying data. The constructed header is returned by the function.

When ``copyData=false`` , the conversion is done really fast (in O(1) time) and the newly created matrix header will have ``refcount=0`` , which means that no reference counting is done for the matrix data. In this case, you have to preserve the data until the new header is destructed. Otherwise, when ``copyData=true`` , the new buffer is allocated and managed as if you created a new matrix from scratch and copied the data there. That is, ``cvarrToMat(src, true) :math:`\sim` cvarrToMat(src, false).clone()`` (assuming that COI is not set). The function provides a uniform way of supporting
``CvArr`` paradigm in the code that is migrated to use new-style data structures internally. The reverse transformation, from
:func:`Mat` to
``CvMat`` or
``IplImage`` can be done by a simple assignment: ::

    CvMat* A = cvCreateMat(10, 10, CV_32F);
    cvSetIdentity(A);
    IplImage A1; cvGetImage(A, &A1);
    Mat B = cvarrToMat(A);
    Mat B1 = cvarrToMat(&A1);
    IplImage C = B;
    CvMat C1 = B1;
    // now A, A1, B, B1, C and C1 are different headers
    // for the same 10x10 floating-point array.
    // note that you will need to use "&"
    // to pass C & C1 to OpenCV functions, for example:
    printf("


Normally, the function is used to convert an old-style 2D array (
``CvMat`` or
``IplImage`` ) to ``Mat`` . However, the function can also take
``CvMatND`` as an input and create
:func:`Mat` for it, if it is possible. And, for ``CvMatND A`` , it is possible if and only if ``A.dim[i].size*A.dim.step[i] == A.dim.step[i-1]`` for all or for all but one ``i, 0 < i < A.dims`` . That is, the matrix data should be continuous or it should be representable as a sequence of continuous matrices. By using this function in this way, you can process
``CvMatND`` using an arbitrary element-wise function. But for more complex operations, such as filtering functions, it will not work, and you need to convert
``CvMatND`` to
:func:`MatND` using the corresponding constructor of the latter.

The last parameter, ``coiMode`` , specifies how to deal with an image with COI set. By default, it is 0 and the function reports an error when an image with COI comes in. And ``coiMode=1`` means that no error is signalled. You have to check COI presence and handle it manually. The modern structures, such as
:func:`Mat` and
:func:`MatND` do not support COI natively. To process an individual channel of a new-style array, you need either to organize a loop over the array (for example, using matrix iterators) where the channel of interest will be processed, or extract the COI using
:func:`mixChannels` (for new-style arrays) or
:func:`extractImageCOI` (for old-style arrays), process this individual channel, and insert it back to the destination array if needed (using
:func:`mixChannel` or
:func:`insertImageCOI` , respectively).

See Also:
:func:`cvGetImage`,:func:`cvGetMat`,:func:`cvGetMatND`,:func:`extractImageCOI`,:func:`insertImageCOI`,:func:`mixChannels` 

.. index:: dct

.. _dct:

dct
-------
.. c:function:: void dct(const Mat& src, Mat& dst, int flags=0)

    Performs a forward or inverse discrete cosine transform of 1D or 2D array

    :param src: The source floating-point array

    :param dst: The destination array; will have the same size and same type as  ``src``
    
    :param flags: Transformation flags, a combination of the following values

            * **DCT_INVERSE** do an inverse 1D or 2D transform instead of the default forward transform.

            * **DCT_ROWS** do a forward or inverse transform of every individual row of the input matrix. This flag allows user to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself), to do 3D and higher-dimensional transforms and so forth.

The function ``dct`` performs a forward or inverse discrete cosine transform (DCT) of a 1D or 2D floating-point array:

Forward Cosine transform of 1D vector of
:math:`N` elements:

.. math::

    Y = C^{(N)}  \cdot X

where

.. math::

    C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )

and
:math:`\alpha_0=1`,:math:`\alpha_j=2` for
:math:`j > 0` .

Inverse Cosine transform of 1D vector of N elements:

.. math::

    X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y

(since
:math:`C^{(N)}` is orthogonal matrix,
:math:`C^{(N)} \cdot \left(C^{(N)}\right)^T = I` )

Forward Cosine transform of 2D
:math:`M \times N` matrix:

.. math::

    Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T

Inverse Cosine transform of 2D vector of
:math:`M \times N` elements:

.. math::

    X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}

The function chooses the mode of operation by looking at the flags and size of the input array:

*
    if ``(flags & DCT_INVERSE) == 0``     , the function does forward 1D or 2D transform, otherwise it is inverse 1D or 2D transform.

*
    if ``(flags & DCT_ROWS) :math:`\ne` 0``     , the function performs 1D transform of each row.

*
    otherwise, if the array is a single column or a single row, the function performs 1D transform

*
    otherwise it performs 2D transform.

**Important note**
: currently dct supports even-size arrays (2, 4, 6 ...). For data analysis and approximation you can pad the array when necessary.

Also, the function's performance depends very much, and not monotonically, on the array size, see
:func:`getOptimalDFTSize` . In the current implementation DCT of a vector of size ``N`` is computed via DFT of a vector of size ``N/2`` , thus the optimal DCT size
:math:`\texttt{N}^*\geq\texttt{N}` can be computed as: ::

    size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }


See Also:
:func:`dft`,:func:`getOptimalDFTSize`,:func:`idct`

.. index:: dft

.. _dft:

dft
---

.. c:function:: void dft(const Mat& src, Mat& dst, int flags=0, int nonzeroRows=0)

    Performs a forward or inverse Discrete Fourier transform of 1D or 2D floating-point array.

    :param src: The source array, real or complex

    :param dst: The destination array, which size and type depends on the  ``flags``
    
    :param flags: Transformation flags, a combination of the following values

            * **DFT_INVERSE** do an inverse 1D or 2D transform instead of the default forward transform.

            * **DFT_SCALE** scale the result: divide it by the number of array elements. Normally, it is combined with  ``DFT_INVERSE``             .
            * **DFT_ROWS** do a forward or inverse transform of every individual row of the input matrix. This flag allows the user to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself), to do 3D and higher-dimensional transforms and so forth.

            * **DFT_COMPLEX_OUTPUT** then the function performs forward transformation of 1D or 2D real array, the result, though being a complex array, has complex-conjugate symmetry ( *CCS* ), see the description below. Such an array can be packed into real array of the same size as input, which is the fastest option and which is what the function does by default. However, you may wish to get the full complex array (for simpler spectrum analysis etc.). Pass the flag to tell the function to produce full-size complex output array.

            * **DFT_REAL_OUTPUT** then the function performs inverse transformation of 1D or 2D complex array, the result is normally a complex array of the same size. However, if the source array has conjugate-complex symmetry (for example, it is a result of forward transformation with  ``DFT_COMPLEX_OUTPUT``  flag), then the output is real array. While the function itself does not check whether the input is symmetrical or not, you can pass the flag and then the function will assume the symmetry and produce the real output array. Note that when the input is packed real array and inverse transformation is executed, the function treats the input as packed complex-conjugate symmetrical array, so the output will also be real array

    :param nonzeroRows: When the parameter  :math:`\ne 0` , the function assumes that only the first  ``nonzeroRows``  rows of the input array ( ``DFT_INVERSE``  is not set) or only the first  ``nonzeroRows``  of the output array ( ``DFT_INVERSE``  is set) contain non-zeros, thus the function can handle the rest of the rows more efficiently and thus save some time. This technique is very useful for computing array cross-correlation or convolution using DFT

Forward Fourier transform of 1D vector of N elements:

.. math::

    Y = F^{(N)}  \cdot X,

where
:math:`F^{(N)}_{jk}=\exp(-2\pi i j k/N)` and
:math:`i=\sqrt{-1}` Inverse Fourier transform of 1D vector of N elements:

.. math::

    \begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}

where
:math:`F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T` Forward Fourier transform of 2D vector of
:math:`M \times N` elements:

.. math::

    Y = F^{(M)}  \cdot X  \cdot F^{(N)}

Inverse Fourier transform of 2D vector of
:math:`M \times N` elements:

.. math::

    \begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}

In the case of real (single-channel) data, the packed format called
*CCS*
(complex-conjugate-symmetrical) that was borrowed from IPL and used to represent the result of a forward Fourier transform or input for an inverse Fourier transform:

.. math::

    \begin{bmatrix} Re Y_{0,0} & Re Y_{0,1} & Im Y_{0,1} & Re Y_{0,2} & Im Y_{0,2} &  \cdots & Re Y_{0,N/2-1} & Im Y_{0,N/2-1} & Re Y_{0,N/2}  \\ Re Y_{1,0} & Re Y_{1,1} & Im Y_{1,1} & Re Y_{1,2} & Im Y_{1,2} &  \cdots & Re Y_{1,N/2-1} & Im Y_{1,N/2-1} & Re Y_{1,N/2}  \\ Im Y_{1,0} & Re Y_{2,1} & Im Y_{2,1} & Re Y_{2,2} & Im Y_{2,2} &  \cdots & Re Y_{2,N/2-1} & Im Y_{2,N/2-1} & Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &  Re Y_{M-3,1}  & Im Y_{M-3,1} &  \hdotsfor{3} & Re Y_{M-3,N/2-1} & Im Y_{M-3,N/2-1}& Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &  Re Y_{M-2,1}  & Im Y_{M-2,1} &  \hdotsfor{3} & Re Y_{M-2,N/2-1} & Im Y_{M-2,N/2-1}& Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &  Re Y_{M-1,1} &  Im Y_{M-1,1} &  \hdotsfor{3} & Re Y_{M-1,N/2-1} & Im Y_{M-1,N/2-1}& Re Y_{M/2,N/2} \end{bmatrix}

in the case of 1D transform of real vector, the output will look as the first row of the above matrix.

So, the function chooses the operation mode depending on the flags and size of the input array:

 * if ``DFT_ROWS`` is set or the input array has single row or single column then the function performs 1D forward or inverse transform (of each row of a matrix when ``DFT_ROWS`` is set, otherwise it will be 2D transform.

 * if input array is real and ``DFT_INVERSE`` is not set, the function does forward 1D or 2D transform:

    * when ``DFT_COMPLEX_OUTPUT`` is set then the output will be complex matrix of the same size as input.

    * otherwise the output will be a real matrix of the same size as input. in the case of 2D transform it will use the packed format as shown above; in the case of single 1D transform it will look as the first row of the above matrix; in the case of multiple 1D transforms (when using ``DCT_ROWS``         flag) each row of the output matrix will look like the first row of the above matrix.

 * otherwise, if the input array is complex and either ``DFT_INVERSE``     or ``DFT_REAL_OUTPUT``     are not set then the output will be a complex array of the same size as input and the function will perform the forward or inverse 1D or 2D transform of the whole input array or each row of the input array independently, depending on the flags ``DFT_INVERSE`` and ``DFT_ROWS``.

 * otherwise, i.e. when ``DFT_INVERSE`` is set, the input array is real, or it is complex but ``DFT_REAL_OUTPUT``     is set, the output will be a real array of the same size as input, and the function will perform 1D or 2D inverse transformation of the whole input array or each individual row, depending on the flags ``DFT_INVERSE`` and ``DFT_ROWS``.

The scaling is done after the transformation if ``DFT_SCALE`` is set.

Unlike
:func:`dct` , the function supports arrays of arbitrary size, but only those arrays are processed efficiently, which sizes can be factorized in a product of small prime numbers (2, 3 and 5 in the current implementation). Such an efficient DFT size can be computed using
:func:`getOptimalDFTSize` method.

Here is the sample on how to compute DFT-based convolution of two 2D real arrays: ::

    void convolveDFT(const Mat& A, const Mat& B, Mat& C)
    {
        // reallocate the output array if needed
        C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
        Size dftSize;
        // compute the size of DFT transform
        dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
        dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

        // allocate temporary buffers and initialize them with 0's
        Mat tempA(dftSize, A.type(), Scalar::all(0));
        Mat tempB(dftSize, B.type(), Scalar::all(0));

        // copy A and B to the top-left corners of tempA and tempB, respectively
        Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
        A.copyTo(roiA);
        Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
        B.copyTo(roiB);

        // now transform the padded A & B in-place;
        // use "nonzeroRows" hint for faster processing
        dft(tempA, tempA, 0, A.rows);
        dft(tempB, tempB, 0, B.rows);

        // multiply the spectrums;
        // the function handles packed spectrum representations well
        mulSpectrums(tempA, tempB, tempA);

        // transform the product back from the frequency domain.
        // Even though all the result rows will be non-zero,
        // we need only the first C.rows of them, and thus we
        // pass nonzeroRows == C.rows
        dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

        // now copy the result back to C.
        tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);

        // all the temporary buffers will be deallocated automatically
    }


What can be optimized in the above sample?

*
    since we passed :math:`\texttt{nonzeroRows} \ne 0`     to the forward transform calls and since we copied ``A``     / ``B``     to the top-left corners of ``tempA``     / ``tempB``     , respectively, it's not necessary to clear the whole ``tempA``     and ``tempB``     ; it is only necessary to clear the ``tempA.cols - A.cols``     ( ``tempB.cols - B.cols``     ) rightmost columns of the matrices.

* this DFT-based convolution does not have to be applied to the whole big arrays, especially if ``B``     is significantly smaller than ``A``     or vice versa. Instead, we can compute convolution by parts. For that we need to split the destination array ``C``     into multiple tiles and for each tile estimate, which parts of ``A``     and ``B``     are required to compute convolution in this tile. If the tiles in ``C``     are too small, the speed will decrease a lot, because of repeated work - in the ultimate case, when each tile in ``C``     is a single pixel, the algorithm becomes equivalent to the naive convolution algorithm. If the tiles are too big, the temporary arrays ``tempA``     and ``tempB``     become too big and there is also slowdown because of bad cache locality. So there is optimal tile size somewhere in the middle.

*
    if the convolution is done by parts, since different tiles in ``C``     can be computed in parallel, the loop can be threaded.

All of the above improvements have been implemented in :func:`matchTemplate` and :func:`filter2D` , therefore, by using them, you can get even better performance than with the above theoretically optimal implementation (though, those two functions actually compute cross-correlation, not convolution, so you will need to "flip" the kernel or the image around the center using :func:`flip` ).

See Also:
:func:`dct`,:func:`getOptimalDFTSize`,:func:`mulSpectrums`,:func:`filter2D`,:func:`matchTemplate`,:func:`flip`,:func:`cartToPolar`,:func:`magnitude`,:func:`phase`

.. index:: divide

.. _divide:

divide
----------
.. c:function:: void divide(const Mat& src1, const Mat& src2, Mat& dst, double scale=1)

.. c:function:: void divide(double scale, const Mat& src2, Mat& dst)

.. c:function:: void divide(const MatND& src1, const MatND& src2, MatND& dst, double scale=1)

.. c:function:: void divide(double scale, const MatND& src2, MatND& dst)

    Performs per-element division of two arrays or a scalar by an array.

    :param src1: The first source array

    :param src2: The second source array; should have the same size and same type as  ``src1``
    
    :param scale: Scale factor

    :param dst: The destination array; will have the same size and same type as  ``src2``
    
The functions ``divide`` divide one array by another:

.. math::

    \texttt{dst(I) = saturate(src1(I)*scale/src2(I))}

or a scalar by array, when there is no ``src1`` :

.. math::

    \texttt{dst(I) = saturate(scale/src2(I))}

The result will have the same type as ``src1`` . When ``src2(I)=0``,``dst(I)=0`` too.

See Also:
:func:`multiply`,:func:`add`,:func:`subtract`,:ref:`MatrixExpressions`

.. index:: determinant

.. _determinant:

determinant
-----------

.. c:function:: double determinant(const Mat& mtx)

    Returns determinant of a square floating-point matrix.

    :param mtx: The input matrix; must have  ``CV_32FC1``  or  ``CV_64FC1``  type and square size

The function ``determinant`` computes and returns determinant of the specified matrix. For small matrices ( ``mtx.cols=mtx.rows<=3`` )
the direct method is used; for larger matrices the function uses LU factorization.

For symmetric positive-determined matrices, it is also possible to compute
:func:`SVD` :
:math:`\texttt{mtx}=U \cdot W \cdot V^T` and then calculate the determinant as a product of the diagonal elements of
:math:`W` .

See Also:
:func:`SVD`,:func:`trace`,:func:`invert`,:func:`solve`,:ref:`MatrixExpressions`

.. index:: eigen

.. _eigen:

eigen
-----

.. c:function:: bool eigen(const Mat& src, Mat& eigenvalues, int lowindex=-1, int highindex=-1)

.. c:function:: bool eigen(const Mat& src, Mat& eigenvalues, Mat& eigenvectors, int lowindex=-1,int highindex=-1)

    Computes eigenvalues and eigenvectors of a symmetric matrix.

    :param src: The input matrix; must have  ``CV_32FC1``  or  ``CV_64FC1``  type, square size and be symmetric:  :math:`\texttt{src}^T=\texttt{src}`
    
    :param eigenvalues: The output vector of eigenvalues of the same type as  ``src`` ; The eigenvalues are stored in the descending order.

    :param eigenvectors: The output matrix of eigenvectors; It will have the same size and the same type as  ``src`` ; The eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues

    :param lowindex: Optional index of largest eigenvalue/-vector to calculate. (See below.)

    :param highindex: Optional index of smallest eigenvalue/-vector to calculate. (See below.)

The functions ``eigen`` compute just eigenvalues, or eigenvalues and eigenvectors of symmetric matrix ``src`` : ::

    src*eigenvectors(i,:)' = eigenvalues(i)*eigenvectors(i,:)' (in MATLAB notation)


If either low- or highindex is supplied the other is required, too.
Indexing is 0-based. Example: To calculate the largest eigenvector/-value set
lowindex = highindex = 0.
For legacy reasons this function always returns a square matrix the same size
as the source matrix with eigenvectors and a vector the length of the source
matrix with eigenvalues. The selected eigenvectors/-values are always in the
first highindex - lowindex + 1 rows.

See Also:
:func:`SVD`,:func:`completeSymm`,:func:`PCA`

.. index:: exp

.. _exp:

exp
---

.. c:function:: void exp(const Mat& src, Mat& dst)

.. c:function:: void exp(const MatND& src, MatND& dst)

    Calculates the exponent of every array element.

    :param src: The source array

    :param dst: The destination array; will have the same size and same type as  ``src``

The function ``exp`` calculates the exponent of every element of the input array:

.. math::

    \texttt{dst} [I] = e^{ \texttt{src} }(I)

The maximum relative error is about
:math:`7 \times 10^{-6}` for single-precision and less than
:math:`10^{-10}` for double-precision. Currently, the function converts denormalized values to zeros on output. Special values (NaN,
:math:`\pm \infty` ) are not handled.

See Also:
:func:`log`,:func:`cartToPolar`,:func:`polarToCart`,:func:`phase`,:func:`pow`,:func:`sqrt`,:func:`magnitude`

.. index:: extractImageCOI

.. _extractImageCOI:

extractImageCOI
---------------

.. c:function:: void extractImageCOI(const CvArr* src, Mat& dst, int coi=-1)

    Extract the selected image channel

    :param src: The source array. It should be a pointer to  ``CvMat``  or  ``IplImage``
    
    :param dst: The destination array; will have single-channel, and the same size and the same depth as  ``src``
    
    :param coi: If the parameter is  ``>=0`` , it specifies the channel to extract; If it is  ``<0`` , ``src``  must be a pointer to  ``IplImage``  with valid COI set - then the selected COI is extracted.

The function ``extractImageCOI`` is used to extract image COI from an old-style array and put the result to the new-style C++ matrix. As usual, the destination matrix is reallocated using ``Mat::create`` if needed.

To extract a channel from a new-style matrix, use
:func:`mixChannels` or
:func:`split` See Also:
:func:`mixChannels`,:func:`split`,:func:`merge`,:func:`cvarrToMat`,:func:`cvSetImageCOI`,:func:`cvGetImageCOI`

.. index:: fastAtan2

.. _fastAtan2:

fastAtan2
---------

.. c:function:: float fastAtan2(float y, float x)

    Calculates the angle of a 2D vector in degrees

    :param x: x-coordinate of the vector

    :param y: y-coordinate of the vector

The function ``fastAtan2`` calculates the full-range angle of an input 2D vector. The angle is
measured in degrees and varies from
:math:`0^\circ` to
:math:`360^\circ` . The accuracy is about
:math:`0.3^\circ` .

.. index:: flip

flip
--------
.. c:function:: void flip(const Mat& src, Mat& dst, int flipCode)

    Flips a 2D array around vertical, horizontal or both axes.

    :param src: The source array

    :param dst: The destination array; will have the same size and same type as  ``src``
    
    :param flipCode: Specifies how to flip the array: 0 means flipping around the x-axis, positive (e.g., 1) means flipping around y-axis, and negative (e.g., -1) means flipping around both axes. See Also the discussion below for the formulas.

The function ``flip`` flips the array in one of three different ways (row and column indices are 0-based):

.. math::

    \texttt{dst} _{ij} =  \forkthree{\texttt{src}_{\texttt{src.rows}-i-1,j} }{if  \texttt{flipCode} = 0}
    { \texttt{src} _{i, \texttt{src.cols} -j-1}}{if  \texttt{flipCode} > 0}
    { \texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1}}{if  \texttt{flipCode} < 0}

The example scenarios of function use are:

*
    vertical flipping of the image (
    :math:`\texttt{flipCode} = 0`     ) to switch between top-left and bottom-left image origin, which is a typical operation in video processing in Windows.

*
    horizontal flipping of the image with subsequent horizontal shift and absolute difference calculation to check for a vertical-axis symmetry (
    :math:`\texttt{flipCode} > 0`     )

*
    simultaneous horizontal and vertical flipping of the image with subsequent shift and absolute difference calculation to check for a central symmetry (
    :math:`\texttt{flipCode} < 0`     )

*
    reversing the order of 1d point arrays (
    :math:`\texttt{flipCode} > 0`     or
    :math:`\texttt{flipCode} = 0`     )

See Also: :func:`transpose`,:func:`repeat`,:func:`completeSymm`

.. index:: gemm

.. _gemm:

gemm
----

.. c:function:: void gemm(const Mat& src1, const Mat& src2, double alpha, const Mat& src3, double beta, Mat& dst, int flags=0)

    Performs generalized matrix multiplication.

    :param src1: The first multiplied input matrix; should have  ``CV_32FC1`` , ``CV_64FC1`` , ``CV_32FC2``  or  ``CV_64FC2``  type

    :param src2: The second multiplied input matrix; should have the same type as  ``src1``
    
    :param alpha: The weight of the matrix product

    :param src3: The third optional delta matrix added to the matrix product; should have the same type as  ``src1``  and  ``src2``
    
    :param beta: The weight of  ``src3``
    
    :param dst: The destination matrix; It will have the proper size and the same type as input matrices

    :param flags: Operation flags:

            * **GEMM_1_T** transpose  ``src1``
            * **GEMM_2_T** transpose  ``src2``
            * **GEMM_3_T** transpose  ``src3``
            
The function performs generalized matrix multiplication and similar to the corresponding functions ``*gemm`` in BLAS level 3. For example, ``gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)`` corresponds to

.. math::

    \texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T

The function can be replaced with a matrix expression, e.g. the above call can be replaced with: ::

    dst = alpha*src1.t()*src2 + beta*src3.t();


See Also:
:func:`mulTransposed`,:func:`transform`,:ref:`MatrixExpressions`

.. index:: getConvertElem

.. _getConvertItem:

getConvertElem
--------------

.. c:function:: ConvertData getConvertElem(int fromType, int toType)

.. c:function:: ConvertScaleData getConvertScaleElem(int fromType, int toType)

.. c:function:: typedef void (*ConvertData)(const void* from, void* to, int cn)

.. c:function:: typedef void (*ConvertScaleData)(const void* from, void* to, int cn, double alpha, double beta)

    Returns conversion function for a single pixel

    :param fromType: The source pixel type

    :param toType: The destination pixel type

    :param from: Callback parameter: pointer to the input pixel

    :param to: Callback parameter: pointer to the output pixel

    :param cn: Callback parameter: the number of channels; can be arbitrary, 1, 100, 100000, ...

    :param alpha: ConvertScaleData callback optional parameter: the scale factor

    :param beta: ConvertScaleData callback optional parameter: the delta or offset

The functions ``getConvertElem`` and ``getConvertScaleElem`` return pointers to the functions for converting individual pixels from one type to another. While the main function purpose is to convert single pixels (actually, for converting sparse matrices from one type to another), you can use them to convert the whole row of a dense matrix or the whole matrix at once, by setting ``cn = matrix.cols*matrix.rows*matrix.channels()`` if the matrix data is continuous.

See Also:
:func:`Mat::convertTo`,:func:`MatND::convertTo`,:func:`SparseMat::convertTo`

.. index:: getOptimalDFTSize

.. _getOptimalDFTSize:

getOptimalDFTSize
-----------------

.. c:function:: int getOptimalDFTSize(int vecsize)

    Returns optimal DFT size for a given vector size.

    :param vecsize: Vector size

DFT performance is not a monotonic function of a vector size, therefore, when you compute convolution of two arrays or do a spectral analysis of array, it usually makes sense to pad the input data with zeros to get a bit larger array that can be transformed much faster than the original one.
Arrays, which size is a power-of-two (2, 4, 8, 16, 32, ...) are the fastest to process, though, the arrays, which size is a product of 2's, 3's and 5's (e.g. 300 = 5*5*3*2*2), are also processed quite efficiently.

The function ``getOptimalDFTSize`` returns the minimum number ``N`` that is greater than or equal to ``vecsize`` , such that the DFT
of a vector of size ``N`` can be computed efficiently. In the current implementation
:math:`N=2^p \times 3^q \times 5^r` , for some
:math:`p`,:math:`q`,:math:`r` .

The function returns a negative number if ``vecsize`` is too large (very close to ``INT_MAX`` ).

While the function cannot be used directly to estimate the optimal vector size for DCT transform (since the current DCT implementation supports only even-size vectors), it can be easily computed as ``getOptimalDFTSize((vecsize+1)/2)*2`` .

See Also:
:func:`dft`,:func:`dct`,:func:`idft`,:func:`idct`,:func:`mulSpectrums`

.. index:: idct

.. _idct:

idct
----

.. c:function:: void idct(const Mat& src, Mat& dst, int flags=0)

    Computes inverse Discrete Cosine Transform of a 1D or 2D array

    :param src: The source floating-point single-channel array

    :param dst: The destination array. Will have the same size and same type as  ``src``
    
    :param flags: The operation flags.
    
``idct(src, dst, flags)`` is equivalent to ``dct(src, dst, flags | DCT_INVERSE)``.

See Also: :func:`dct`,:func:`dft`,:func:`idft`,:func:`getOptimalDFTSize`

.. index:: idft

.. _idft:

idft
----

.. c:function:: void idft(const Mat& src, Mat& dst, int flags=0, int outputRows=0)

    Computes inverse Discrete Fourier Transform of a 1D or 2D array

    :param src: The source floating-point real or complex array

    :param dst: The destination array, which size and type depends on the  ``flags``
    
    :param flags: The operation flags. See  :func:`dft`
    
    :param nonzeroRows: The number of  ``dst``  rows to compute. The rest of the rows will have undefined content. See the convolution sample in  :func:`dft`  description
    
``idft(src, dst, flags)`` is equivalent to ``dct(src, dst, flags | DFT_INVERSE)`` .

See :func:`dft` for details.
Note, that none of ``dft`` and ``idft`` scale the result by default.
Thus, you should pass ``DFT_SCALE`` to one of ``dft`` or ``idft`` explicitly to make these transforms mutually inverse.

See Also: :func:`dft`,:func:`dct`,:func:`idct`,:func:`mulSpectrums`,:func:`getOptimalDFTSize`

.. index:: inRange

.. _inRange:

inRange
-------

.. c:function:: void inRange(const Mat& src, const Mat& lowerb, const Mat& upperb, Mat& dst)

.. c:function:: void inRange(const Mat& src, const Scalar& lowerb, const Scalar& upperb, Mat& dst)

.. c:function:: void inRange(const MatND& src, const MatND& lowerb, const MatND& upperb, MatND& dst)

.. c:function:: void inRange(const MatND& src, const Scalar& lowerb, const Scalar& upperb, MatND& dst)

    Checks if array elements lie between the elements of two other arrays.

    :param src: The first source array

    :param lowerb: The inclusive lower boundary array of the same size and type as  ``src``
    
    :param upperb: The exclusive upper boundary array of the same size and type as  ``src``
    
    :param dst: The destination array, will have the same size as  ``src``  and  ``CV_8U``  type

The functions ``inRange`` do the range check for every element of the input array:

.. math::

    \texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 <  \texttt{upperb} (I)_0

for single-channel arrays,

.. math::

    \texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 <  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 <  \texttt{upperb} (I)_1

for two-channel arrays and so forth. ``dst`` (I) is set to 255 (all ``1`` -bits) if ``src`` (I) is within the specified range and 0 otherwise.

.. index:: invert

.. _invert:

invert
------

.. c:function:: double invert(const Mat& src, Mat& dst, int method=DECOMP_LU)

    Finds the inverse or pseudo-inverse of a matrix

    :param src: The source floating-point  :math:`M \times N`  matrix

    :param dst: The destination matrix; will have  :math:`N \times M`  size and the same type as  ``src``
    
    :param flags: The inversion method :

            * **DECOMP_LU** Gaussian elimination with optimal pivot element chosen

            * **DECOMP_SVD** Singular value decomposition (SVD) method

            * **DECOMP_CHOLESKY** Cholesky decomposion. The matrix must be symmetrical and positively defined

The function ``invert`` inverts matrix ``src`` and stores the result in ``dst`` .
When the matrix ``src`` is singular or non-square, the function computes the pseudo-inverse matrix, i.e. the matrix ``dst`` , such that
:math:`\|\texttt{src} \cdot \texttt{dst} - I\|` is minimal.

In the case of ``DECOMP_LU`` method, the function returns the ``src`` determinant ( ``src`` must be square). If it is 0, the matrix is not inverted and ``dst`` is filled with zeros.

In the case of ``DECOMP_SVD`` method, the function returns the inversed condition number of ``src`` (the ratio of the smallest singular value to the largest singular value) and 0 if ``src`` is singular. The SVD method calculates a pseudo-inverse matrix if ``src`` is singular.

Similarly to ``DECOMP_LU`` , the method ``DECOMP_CHOLESKY`` works only with non-singular square matrices. In this case the function stores the inverted matrix in ``dst`` and returns non-zero, otherwise it returns 0.

See Also:
:func:`solve`,:func:`SVD`

.. index:: log

.. _log:

log
---

.. c:function:: void log(const Mat& src, Mat& dst)

.. c:function:: void log(const MatND& src, MatND& dst)

    Calculates the natural logarithm of every array element.

    :param src: The source array

    :param dst: The destination array; will have the same size and same type as  ``src``
    
The function ``log`` calculates the natural logarithm of the absolute value of every element of the input array:

.. math::

    \texttt{dst} (I) =  \fork{\log |\texttt{src}(I)|}{if $\texttt{src}(I) \ne 0$ }{\texttt{C}}{otherwise}

Where ``C`` is a large negative number (about -700 in the current implementation).
The maximum relative error is about
:math:`7 \times 10^{-6}` for single-precision input and less than
:math:`10^{-10}` for double-precision input. Special values (NaN,
:math:`\pm \infty` ) are not handled.

See Also:
:func:`exp`,:func:`cartToPolar`,:func:`polarToCart`,:func:`phase`,:func:`pow`,:func:`sqrt`,:func:`magnitude`

.. index:: LUT

.. _LUT:

LUT
---

.. c:function:: void LUT(const Mat& src, const Mat& lut, Mat& dst)

    Performs a look-up table transform of an array.

    :param src: Source array of 8-bit elements

    :param lut: Look-up table of 256 elements. In the case of multi-channel source array, the table should either have a single channel (in this case the same table is used for all channels) or the same number of channels as in the source array

    :param dst: Destination array; will have the same size and the same number of channels as  ``src`` , and the same depth as  ``lut``
    
The function ``LUT`` fills the destination array with values from the look-up table. Indices of the entries are taken from the source array. That is, the function processes each element of ``src`` as follows:

.. math::

    \texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}

where

.. math::

    d =  \fork{0}{if \texttt{src} has depth \texttt{CV\_8U}}{128}{if \texttt{src} has depth \texttt{CV\_8S}}

See Also:
:func:`convertScaleAbs`,``Mat::convertTo``

.. index:: magnitude

.. _magnitude:

magnitude
---------

.. c:function:: void magnitude(const Mat& x, const Mat& y, Mat& magnitude)

    Calculates magnitude of 2D vectors.

    :param x: The floating-point array of x-coordinates of the vectors

    :param y: The floating-point array of y-coordinates of the vectors; must have the same size as  ``x``
    
    :param dst: The destination array; will have the same size and same type as  ``x``
    
The function ``magnitude`` calculates magnitude of 2D vectors formed from the corresponding elements of ``x`` and ``y`` arrays:

.. math::

    \texttt{dst} (I) =  \sqrt{\texttt{x}(I)^2 + \texttt{y}(I)^2}

See Also:
:func:`cartToPolar`,:func:`polarToCart`,:func:`phase`,:func:`sqrt`

.. index:: Mahalanobis

.. _Mahalanobis:

Mahalanobis
-----------

.. c:function:: double Mahalanobis(const Mat& vec1, const Mat& vec2, const Mat& icovar)

    Calculates the Mahalanobis distance between two vectors.

    :param vec1: The first 1D source vector

    :param vec2: The second 1D source vector

    :param icovar: The inverse covariance matrix

The function ``cvMahalonobis`` calculates and returns the weighted distance between two vectors:

.. math::

    d( \texttt{vec1} , \texttt{vec2} )= \sqrt{\sum_{i,j}{\texttt{icovar(i,j)}\cdot(\texttt{vec1}(I)-\texttt{vec2}(I))\cdot(\texttt{vec1(j)}-\texttt{vec2(j)})} }

The covariance matrix may be calculated using the
:func:`calcCovarMatrix` function and then inverted using the
:func:`invert` function (preferably using DECOMP_SVD method, as the most accurate).

.. index:: max

.. _max:

max
---

.. c:function:: Mat_Expr<...> max(const Mat& src1, const Mat& src2)

.. c:function:: Mat_Expr<...> max(const Mat& src1, double value)

.. c:function:: Mat_Expr<...> max(double value, const Mat& src1)

.. c:function:: void max(const Mat& src1, const Mat& src2, Mat& dst)

.. c:function:: void max(const Mat& src1, double value, Mat& dst)

.. c:function:: void max(const MatND& src1, const MatND& src2, MatND& dst)

.. c:function:: void max(const MatND& src1, double value, MatND& dst)

    Calculates per-element maximum of two arrays or array and a scalar

    :param src1: The first source array

    :param src2: The second source array of the same size and type as  ``src1``
    
    :param value: The real scalar value

    :param dst: The destination array; will have the same size and type as  ``src1``
    
The functions ``max`` compute per-element maximum of two arrays:

.. math::

    \texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))

or array and a scalar:

.. math::

    \texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )

In the second variant, when the source array is multi-channel, each channel is compared with ``value`` independently.

The first 3 variants of the function listed above are actually a part of
:ref:`MatrixExpressions` , they return the expression object that can be further transformed, or assigned to a matrix, or passed to a function etc.

See Also:
:func:`min`,:func:`compare`,:func:`inRange`,:func:`minMaxLoc`,:ref:`MatrixExpressions`

.. index:: mean

.. _mean:

mean
----

.. c:function:: Scalar mean(const Mat& mtx)

.. c:function:: Scalar mean(const Mat& mtx, const Mat& mask)

.. c:function:: Scalar mean(const MatND& mtx)

.. c:function:: Scalar mean(const MatND& mtx, const MatND& mask)

    Calculates average (mean) of array elements

    :param mtx: The source array; it should have 1 to 4 channels (so that the result can be stored in  :func:`Scalar` )

    :param mask: The optional operation mask

The functions ``mean`` compute mean value ``M`` of array elements, independently for each channel, and return it:

.. math::

    \begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}

When all the mask elements are 0's, the functions return ``Scalar::all(0)`` .

See Also:
:func:`countNonZero`,:func:`meanStdDev`,:func:`norm`,:func:`minMaxLoc`

.. index:: meanStdDev

.. _meanStdDev:

meanStdDev
----------

.. c:function:: void meanStdDev(const Mat& mtx, Scalar& mean, Scalar& stddev, const Mat& mask=Mat())

.. c:function:: void meanStdDev(const MatND& mtx, Scalar& mean, Scalar& stddev, const MatND& mask=MatND())

    Calculates mean and standard deviation of array elements

    :param mtx: The source array; it should have 1 to 4 channels (so that the results can be stored in  :func:`Scalar` 's)

    :param mean: The output parameter: computed mean value

    :param stddev: The output parameter: computed standard deviation

    :param mask: The optional operation mask

The functions ``meanStdDev`` compute the mean and the standard deviation ``M`` of array elements, independently for each channel, and return it via the output parameters:

.. math::

    \begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2} \end{array}

When all the mask elements are 0's, the functions return ``mean=stddev=Scalar::all(0)`` .
Note that the computed standard deviation is only the diagonal of the complete normalized covariance matrix. If the full matrix is needed, you can reshape the multi-channel array
:math:`M \times N` to the single-channel array
:math:`M*N \times \texttt{mtx.channels}()` (only possible when the matrix is continuous) and then pass the matrix to
:func:`calcCovarMatrix` .

See Also:
:func:`countNonZero`,:func:`mean`,:func:`norm`,:func:`minMaxLoc`,:func:`calcCovarMatrix`

.. index:: merge

.. _merge:

merge
-----

.. c:function:: void merge(const Mat* mv, size_t count, Mat& dst)

.. c:function:: void merge(const vector<Mat>& mv, Mat& dst)

.. c:function:: void merge(const MatND* mv, size_t count, MatND& dst)

.. c:function:: void merge(const vector<MatND>& mv, MatND& dst)

    Composes a multi-channel array from several single-channel arrays.

    :param mv: The source array or vector of the single-channel matrices to be merged. All the matrices in  ``mv``  must have the same size and the same type

    :param count: The number of source matrices when  ``mv``  is a plain C array; must be greater than zero

    :param dst: The destination array; will have the same size and the same depth as  ``mv[0]`` , the number of channels will match the number of source matrices

The functions ``merge`` merge several single-channel arrays (or rather interleave their elements) to make a single multi-channel array.

.. math::

    \texttt{dst} (I)_c =  \texttt{mv} [c](I)

The function
:func:`split` does the reverse operation and if you need to merge several multi-channel images or shuffle channels in some other advanced way, use
:func:`mixChannels` See Also:
:func:`mixChannels`,:func:`split`,:func:`reshape`

.. index:: min

.. _min:

min
---

.. c:function:: Mat_Expr<...> min(const Mat& src1, const Mat& src2)

.. c:function:: Mat_Expr<...> min(const Mat& src1, double value)

.. c:function:: Mat_Expr<...> min(double value, const Mat& src1)

.. c:function:: void min(const Mat& src1, const Mat& src2, Mat& dst)

.. c:function:: void min(const Mat& src1, double value, Mat& dst)

.. c:function:: void min(const MatND& src1, const MatND& src2, MatND& dst)

.. c:function:: void min(const MatND& src1, double value, MatND& dst)

    Calculates per-element minimum of two arrays or array and a scalar

    :param src1: The first source array

    :param src2: The second source array of the same size and type as  ``src1``
    
    :param value: The real scalar value

    :param dst: The destination array; will have the same size and type as  ``src1``
    
The functions ``min`` compute per-element minimum of two arrays:

.. math::

    \texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))

or array and a scalar:

.. math::

    \texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )

In the second variant, when the source array is multi-channel, each channel is compared with ``value`` independently.

The first 3 variants of the function listed above are actually a part of
:ref:`MatrixExpressions` , they return the expression object that can be further transformed, or assigned to a matrix, or passed to a function etc.

See Also:
:func:`max`,:func:`compare`,:func:`inRange`,:func:`minMaxLoc`,:ref:`MatrixExpressions`

.. index:: minMaxLoc

.. _minMaxLoc:

minMaxLoc
---------

.. c:function:: void minMaxLoc(const Mat& src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, const Mat& mask=Mat())

.. c:function:: void minMaxLoc(const MatND& src, double* minVal, double* maxVal, int* minIdx=0, int* maxIdx=0, const MatND& mask=MatND())

.. c:function:: void minMaxLoc(const SparseMat& src, double* minVal, double* maxVal, int* minIdx=0, int* maxIdx=0)

    Finds global minimum and maximum in a whole array or sub-array

    :param src: The source single-channel array

    :param minVal: Pointer to returned minimum value;  ``NULL``  if not required

    :param maxVal: Pointer to returned maximum value;  ``NULL``  if not required

    :param minLoc: Pointer to returned minimum location (in 2D case);  ``NULL``  if not required

    :param maxLoc: Pointer to returned maximum location (in 2D case);  ``NULL``  if not required

    :param minIdx: Pointer to returned minimum location (in nD case); ``NULL``  if not required, otherwise must point to an array of  ``src.dims``  elements and the coordinates of minimum element in each dimensions will be stored sequentially there.

    :param maxIdx: Pointer to returned maximum location (in nD case);  ``NULL``  if not required

    :param mask: The optional mask used to select a sub-array

The functions ``ninMaxLoc`` find minimum and maximum element values
and their positions. The extremums are searched across the whole array, or,
if ``mask`` is not an empty array, in the specified array region.

The functions do not work with multi-channel arrays. If you need to find minimum or maximum elements across all the channels, use
:func:`reshape` first to reinterpret the array as single-channel. Or you may extract the particular channel using
:func:`extractImageCOI` or
:func:`mixChannels` or
:func:`split` .

in the case of a sparse matrix the minimum is found among non-zero elements only.

See Also:
:func:`max`,:func:`min`,:func:`compare`,:func:`inRange`,:func:`extractImageCOI`,:func:`mixChannels`,:func:`split`,:func:`reshape` .

.. index:: mixChannels

.. _mixChannels:

mixChannels
-----------

.. c:function:: void mixChannels(const Mat* srcv, int nsrc, Mat* dstv, int ndst, const int* fromTo, size_t npairs)

.. c:function:: void mixChannels(const MatND* srcv, int nsrc, MatND* dstv, int ndst, const int* fromTo, size_t npairs)

.. c:function:: void mixChannels(const vector<Mat>& srcv, vector<Mat>& dstv, const int* fromTo, int npairs)

.. c:function:: void mixChannels(const vector<MatND>& srcv, vector<MatND>& dstv, const int* fromTo, int npairs)

    Copies specified channels from input arrays to the specified channels of output arrays

    :param srcv: The input array or vector of matrices.
        All the matrices must have the same size and the same depth

    :param nsrc: The number of elements in  ``srcv``
    
    :param dstv: The output array or vector of matrices. All the matrices  *must be allocated* , their size and depth must be the same as in  ``srcv[0]``
        
    :param ndst: The number of elements in  ``dstv``
    
    :param fromTo: The array of index pairs, specifying which channels are copied and where. ``fromTo[k*2]``  is the 0-based index of the input channel in  ``srcv``  and ``fromTo[k*2+1]``  is the index of the output channel in  ``dstv`` . Here the continuous channel numbering is used, that is, the first input image channels are indexed from  ``0``  to  ``srcv[0].channels()-1`` , the second input image channels are indexed from  ``srcv[0].channels()``  to ``srcv[0].channels() + srcv[1].channels()-1``  etc., and the same scheme is used for the output image channels. As a special case, when  ``fromTo[k*2]``  is negative, the corresponding output channel is filled with zero. ``npairs``
    
The functions ``mixChannels`` provide an advanced mechanism for shuffling image channels.
    
:func:`split` and
:func:`merge` and some forms of
:func:`cvtColor` are partial cases of ``mixChannels`` .

As an example, this code splits a 4-channel RGBA image into a 3-channel
BGR (i.e. with R and B channels swapped) and separate alpha channel image: ::

    Mat rgba( 100, 100, CV_8UC4, Scalar(1,2,3,4) );
    Mat bgr( rgba.rows, rgba.cols, CV_8UC3 );
    Mat alpha( rgba.rows, rgba.cols, CV_8UC1 );

    // forming array of matrices is quite efficient operations,
    // because the matrix data is not copied, only the headers
    Mat out[] = { bgr, alpha };
    // rgba[0] -> bgr[2], rgba[1] -> bgr[1],
    // rgba[2] -> bgr[0], rgba[3] -> alpha[0]
    int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
    mixChannels( &rgba, 1, out, 2, from_to, 4 );


Note that, unlike many other new-style C++ functions in OpenCV (see the introduction section and
:func:`Mat::create` ), ``mixChannels`` requires the destination arrays be pre-allocated before calling the function.

See Also:
:func:`split`,:func:`merge`,:func:`cvtColor`

.. index:: mulSpectrums

.. _mulSpectrums:

mulSpectrums
------------

.. c:function:: void mulSpectrums(const Mat& src1, const Mat& src2, Mat& dst, int flags, bool conj=false)

    Performs per-element multiplication of two Fourier spectrums.

    :param src1: The first source array

    :param src2: The second source array; must have the same size and the same type as  ``src1``
    
    :param dst: The destination array; will have the same size and the same type as  ``src1``
    
    :param flags: The same flags as passed to  :func:`dft` ; only the flag  ``DFT_ROWS``  is checked for

    :param conj: The optional flag that conjugate the second source array before the multiplication (true) or not (false)

The function ``mulSpectrums`` performs per-element multiplication of the two CCS-packed or complex matrices that are results of a real or complex Fourier transform.

The function, together with
:func:`dft` and
:func:`idft` , may be used to calculate convolution (pass ``conj=false`` ) or correlation (pass ``conj=false`` ) of two arrays rapidly. When the arrays are complex, they are simply multiplied (per-element) with optional conjugation of the second array elements. When the arrays are real, they assumed to be CCS-packed (see
:func:`dft` for details).

.. index:: multiply

.. _multiply:

multiply
--------

.. c:function:: void multiply(const Mat& src1, const Mat& src2, Mat& dst, double scale=1)

.. c:function:: void multiply(const MatND& src1, const MatND& src2, MatND& dst, double scale=1)

    Calculates the per-element scaled product of two arrays

    :param src1: The first source array

    :param src2: The second source array of the same size and the same type as  ``src1``
    
    :param dst: The destination array; will have the same size and the same type as  ``src1``
    
    :param scale: The optional scale factor

The function ``multiply`` calculates the per-element product of two arrays:

.. math::

    \texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))

There is also
:ref:`MatrixExpressions` -friendly variant of the first function, see
:func:`Mat::mul` .

If you are looking for a matrix product, not per-element product, see
:func:`gemm` .

See Also:
:func:`add`,:func:`substract`,:func:`divide`,:ref:`MatrixExpressions`,:func:`scaleAdd`,:func:`addWeighted`,:func:`accumulate`,:func:`accumulateProduct`,:func:`accumulateSquare`,:func:`Mat::convertTo`

.. index:: mulTransposed

.. mulTransposed:

mulTransposed
-------------

.. c:function:: void mulTransposed( const Mat& src, Mat& dst, bool aTa, const Mat& delta=Mat(), double scale=1, int rtype=-1 )

    Calculates the product of a matrix and its transposition.

    :param src: The source matrix

    :param dst: The destination square matrix

    :param aTa: Specifies the multiplication ordering; see the description below

    :param delta: The optional delta matrix, subtracted from  ``src``  before the multiplication. When the matrix is empty ( ``delta=Mat()`` ), it's assumed to be zero, i.e. nothing is subtracted, otherwise if it has the same size as  ``src`` , then it's simply subtracted, otherwise it is "repeated" (see  :func:`repeat` ) to cover the full  ``src``  and then subtracted. Type of the delta matrix, when it's not empty, must be the same as the type of created destination matrix, see the  ``rtype``  description

    :param scale: The optional scale factor for the matrix product

    :param rtype: When it's negative, the destination matrix will have the same type as  ``src`` . Otherwise, it will have  ``type=CV_MAT_DEPTH(rtype)`` , which should be either  ``CV_32F``  or  ``CV_64F``
    
The function ``mulTransposed`` calculates the product of ``src`` and its transposition:

.. math::

    \texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )

if ``aTa=true`` , and

.. math::

    \texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T

otherwise. The function is used to compute covariance matrix and with zero delta can be used as a faster substitute for general matrix product
:math:`A*B` when
:math:`B=A^T` .

See Also:
:func:`calcCovarMatrix`,:func:`gemm`,:func:`repeat`,:func:`reduce`

.. index:: norm

.. _norm:

norm
----

.. c:function:: double norm(const Mat& src1, int normType=NORM_L2)

.. c:function:: double norm(const Mat& src1, const Mat& src2, int normType=NORM_L2)

.. c:function:: double norm(const Mat& src1, int normType, const Mat& mask)

.. c:function:: double norm(const Mat& src1, const Mat& src2, int normType, const Mat& mask)

.. c:function:: double norm(const MatND& src1, int normType=NORM_L2, const MatND& mask=MatND())

.. c:function:: double norm(const MatND& src1, const MatND& src2, int normType=NORM_L2, const MatND& mask=MatND())

.. c:function:: double norm( const SparseMat& src, int normType )

    Calculates absolute array norm, absolute difference norm, or relative difference norm.

    :param src1: The first source array

    :param src2: The second source array of the same size and the same type as  ``src1``
    
    :param normType: Type of the norm; see the discussion below

    :param mask: The optional operation mask

The functions ``norm`` calculate the absolute norm of ``src1`` (when there is no ``src2`` ):

.. math::

    norm =  \forkthree{\|\texttt{src1}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I)|}{if  $\texttt{normType} = \texttt{NORM\_INF}$ }
    { \| \texttt{src1} \| _{L_1} =  \sum _I | \texttt{src1} (I)|}{if  $\texttt{normType} = \texttt{NORM\_L1}$ }
    { \| \texttt{src1} \| _{L_2} =  \sqrt{\sum_I \texttt{src1}(I)^2} }{if  $\texttt{normType} = \texttt{NORM\_L2}$ }

or an absolute or relative difference norm if ``src2`` is there:

.. math::

    norm =  \forkthree{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  $\texttt{normType} = \texttt{NORM\_INF}$ }
    { \| \texttt{src1} - \texttt{src2} \| _{L_1} =  \sum _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  $\texttt{normType} = \texttt{NORM\_L1}$ }
    { \| \texttt{src1} - \texttt{src2} \| _{L_2} =  \sqrt{\sum_I (\texttt{src1}(I) - \texttt{src2}(I))^2} }{if  $\texttt{normType} = \texttt{NORM\_L2}$ }

or

.. math::

    norm =  \forkthree{\frac{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}}    }{\|\texttt{src2}\|_{L_{\infty}} }}{if  $\texttt{normType} = \texttt{NORM\_RELATIVE\_INF}$ }
    { \frac{\|\texttt{src1}-\texttt{src2}\|_{L_1} }{\|\texttt{src2}\|_{L_1}} }{if  $\texttt{normType} = \texttt{NORM\_RELATIVE\_L1}$ }
    { \frac{\|\texttt{src1}-\texttt{src2}\|_{L_2} }{\|\texttt{src2}\|_{L_2}} }{if  $\texttt{normType} = \texttt{NORM\_RELATIVE\_L2}$ }

The functions ``norm`` return the calculated norm.

When there is ``mask`` parameter, and it is not empty (then it should have type ``CV_8U`` and the same size as ``src1`` ), the norm is computed only over the specified by the mask region.

A multiple-channel source arrays are treated as a single-channel, that is, the results for all channels are combined.

.. index:: normalize

.. _normalize:

normalize
---------

.. c:function:: void normalize( const Mat& src, Mat& dst, double alpha=1, double beta=0, int normType=NORM_L2, int rtype=-1, const Mat& mask=Mat())

.. c:function:: void normalize( const MatND& src, MatND& dst, double alpha=1, double beta=0, int normType=NORM_L2, int rtype=-1, const MatND& mask=MatND())

.. c:function:: void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType )

    Normalizes array's norm or the range

    :param src: The source array

    :param dst: The destination array; will have the same size as  ``src``
    
    :param alpha: The norm value to normalize to or the lower range boundary in the case of range normalization

    :param beta: The upper range boundary in the case of range normalization; not used for norm normalization

    :param normType: The normalization type, see the discussion

    :param rtype: When the parameter is negative, the destination array will have the same type as  ``src`` , otherwise it will have the same number of channels as  ``src``  and the depth ``=CV_MAT_DEPTH(rtype)``
    
    :param mask: The optional operation mask

The functions ``normalize`` scale and shift the source array elements, so that

.. math::

    \| \texttt{dst} \| _{L_p}= \texttt{alpha}

(where
:math:`p=\infty` , 1 or 2) when ``normType=NORM_INF``,``NORM_L1`` or ``NORM_L2``,or so that

.. math::

    \min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}

when ``normType=NORM_MINMAX`` (for dense arrays only).

The optional mask specifies the sub-array to be normalize, that is, the norm or min-n-max are computed over the sub-array and then this sub-array is modified to be normalized. If you want to only use the mask to compute the norm or min-max, but modify the whole array, you can use
:func:`norm` and
:func:`Mat::convertScale` /
:func:`MatND::convertScale` /cross{SparseMat::convertScale} separately.

in the case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this, the range transformation for sparse matrices is not allowed, since it can shift the zero level.

See Also:
:func:`norm`,:func:`Mat::convertScale`,:func:`MatND::convertScale`,:func:`SparseMat::convertScale`

.. index:: PCA

.. _PCA:

PCA
---
.. c:type:: PCA

Class for Principal Component Analysis ::

    class PCA
    {
    public:
        // default constructor
        PCA();
        // computes PCA for a set of vectors stored as data rows or columns.
        PCA(const Mat& data, const Mat& mean, int flags, int maxComponents=0);
        // computes PCA for a set of vectors stored as data rows or columns
        PCA& operator()(const Mat& data, const Mat& mean, int flags, int maxComponents=0);
        // projects vector into the principal components space
        Mat project(const Mat& vec) const;
        void project(const Mat& vec, Mat& result) const;
        // reconstructs the vector from its PC projection
        Mat backProject(const Mat& vec) const;
        void backProject(const Mat& vec, Mat& result) const;

        // eigenvectors of the PC space, stored as the matrix rows
        Mat eigenvectors;
        // the corresponding eigenvalues; not used for PCA compression/decompression
        Mat eigenvalues;
        // mean vector, subtracted from the projected vector
        // or added to the reconstructed vector
        Mat mean;
    };


The class ``PCA`` is used to compute the special basis for a set of vectors. The basis will consist of eigenvectors of the covariance matrix computed from the input set of vectors. And also the class ``PCA`` can transform vectors to/from the new coordinate space, defined by the basis. Usually, in this new coordinate system each vector from the original set (and any linear combination of such vectors) can be quite accurately approximated by taking just the first few its components, corresponding to the eigenvectors of the largest eigenvalues of the covariance matrix. Geometrically it means that we compute projection of the vector to a subspace formed by a few eigenvectors corresponding to the dominant eigenvalues of the covariation matrix. And usually such a projection is very close to the original vector. That is, we can represent the original vector from a high-dimensional space with a much shorter vector consisting of the projected vector's coordinates in the subspace. Such a transformation is also known as Karhunen-Loeve Transform, or KLT. See
http://en.wikipedia.org/wiki/Principal\_component\_analysis
The following sample is the function that takes two matrices. The first one stores the set of vectors (a row per vector) that is used to compute PCA, the second one stores another "test" set of vectors (a row per vector) that are first compressed with PCA, then reconstructed back and then the reconstruction error norm is computed and printed for each vector. ::

    PCA compressPCA(const Mat& pcaset, int maxComponents,
                    const Mat& testset, Mat& compressed)
    {
        PCA pca(pcaset, // pass the data
                Mat(), // we do not have a pre-computed mean vector,
                       // so let the PCA engine to compute it
                CV_PCA_DATA_AS_ROW, // indicate that the vectors
                                    // are stored as matrix rows
                                    // (use CV_PCA_DATA_AS_COL if the vectors are
                                    // the matrix columns)
                maxComponents // specify, how many principal components to retain
                );
        // if there is no test data, just return the computed basis, ready-to-use
        if( !testset.data )
            return pca;
        CV_Assert( testset.cols == pcaset.cols );

        compressed.create(testset.rows, maxComponents, testset.type());

        Mat reconstructed;
        for( int i = 0; i < testset.rows; i++ )
        {
            Mat vec = testset.row(i), coeffs = compressed.row(i);
            // compress the vector, the result will be stored
            // in the i-th row of the output matrix
            pca.project(vec, coeffs);
            // and then reconstruct it
            pca.backProject(coeffs, reconstructed);
            // and measure the error
            printf("
        }
        return pca;
    }


See Also:
:func:`calcCovarMatrix`,:func:`mulTransposed`,:func:`SVD`,:func:`dft`,:func:`dct`

.. index:: PCA::PCA

.. _PCA::PCA:

PCA::PCA
------------
.. c:function:: PCA::PCA()

.. c:function:: PCA::PCA(const Mat& data, const Mat& mean, int flags, int maxComponents=0)

    PCA constructors

    :param data: the input samples, stored as the matrix rows or as the matrix columns

    :param mean: the optional mean value. If the matrix is empty ( ``Mat()`` ), the mean is computed from the data.

    :param flags: operation flags. Currently the parameter is only used to specify the data layout.

        * **CV_PCA_DATA_AS_ROWS** Indicates that the input samples are stored as matrix rows.

        * **CV_PCA_DATA_AS_COLS** Indicates that the input samples are stored as matrix columns.

    :param maxComponents: The maximum number of components that PCA should retain. By default, all the components are retained.

The default constructor initializes empty PCA structure. The second constructor initializes the structure and calls
:func:`PCA::operator ()` .

.. index:: PCA::operator ()

.. _PCA::operator ():

PCA::operator ()
----------------

.. c:function:: PCA& PCA::operator()(const Mat& data, const Mat& mean, int flags, int maxComponents=0)

    Performs Principal Component Analysis of the supplied dataset.

    :param data: the input samples, stored as the matrix rows or as the matrix columns

    :param mean: the optional mean value. If the matrix is empty ( ``Mat()`` ), the mean is computed from the data.

    :param flags: operation flags. Currently the parameter is only used to specify the data layout.

        * **CV_PCA_DATA_AS_ROWS** Indicates that the input samples are stored as matrix rows.

        * **CV_PCA_DATA_AS_COLS** Indicates that the input samples are stored as matrix columns.

    :param maxComponents: The maximum number of components that PCA should retain. By default, all the components are retained.

The operator performs PCA of the supplied dataset. It is safe to reuse the same PCA structure for multiple dataset. That is, if the  structure has been previously used with another dataset, the existing internal data is reclaimed and the new ``eigenvalues``,``eigenvectors`` and ``mean`` are allocated and computed.

The computed eigenvalues are sorted from the largest to the smallest and the corresponding eigenvectors are stored as ``PCA::eigenvectors`` rows.

.. index:: PCA::project

.. _PCA::project:

PCA::project
------------

.. c:function:: Mat PCA::project(const Mat& vec) const

.. c:function:: void PCA::project(const Mat& vec, Mat& result) const

    Project vector(s) to the principal component subspace

    :param vec: the input vector(s). They have to have the same dimensionality and the same layout as the input data used at PCA phase. That is, if  ``CV_PCA_DATA_AS_ROWS``  had been specified, then  ``vec.cols==data.cols``  (that's vectors' dimensionality) and  ``vec.rows``  is the number of vectors to project; and similarly for the  ``CV_PCA_DATA_AS_COLS``  case.

    :param result: the output vectors. Let's now consider  ``CV_PCA_DATA_AS_COLS``  case. In this case the output matrix will have as many columns as the number of input vectors, i.e.  ``result.cols==vec.cols``  and the number of rows will match the number of principal components (e.g.  ``maxComponents``  parameter passed to the constructor).

The methods project one or more vectors to the principal component subspace, where each vector projection is represented by coefficients in the principal component basis. The first form of the method returns the matrix that the second form writes to the result. So the first form can be used as a part of expression, while the second form can be more efficient in a processing loop.

.. index:: PCA::backProject

.. _PCA::backProject:

PCA::backProject
----------------

.. c:function:: Mat PCA::backProject(const Mat& vec) const

.. c:function:: void PCA::backProject(const Mat& vec, Mat& result) const

    Reconstruct vectors from their PC projections.

    :param vec: Coordinates of the vectors in the principal component subspace. The layout and size are the same as of  ``PCA::project``  output vectors.

    :param result: The reconstructed vectors. The layout and size are the same as of  ``PCA::project``  input vectors.

The methods are inverse operations to
:func:`PCA::project` . They take PC coordinates of projected vectors and reconstruct the original vectors. Of course, unless all the principal components have been retained, the reconstructed vectors will be different from the originals, but typically the difference will be small is if the number of components is large enough (but still much smaller than the original vector dimensionality) - that's why PCA is used after all.

.. index:: perspectiveTransform

.. _perspectiveTransform:

perspectiveTransform
--------------------
.. c:function:: void perspectiveTransform(const Mat& src, Mat& dst, const Mat& mtx )

    Performs perspective matrix transformation of vectors.

    :param src: The source two-channel or three-channel floating-point array;
                    each element is 2D/3D vector to be transformed

    :param dst: The destination array; it will have the same size and same type as  ``src``
    
    :param mtx: :math:`3\times 3`  or  :math:`4 \times 4`  transformation matrix

The function ``perspectiveTransform`` transforms every element of ``src``,by treating it as 2D or 3D vector, in the following way (here 3D vector transformation is shown; in the case of 2D vector transformation the
:math:`z` component is omitted):

.. math::

    (x, y, z)  \rightarrow (x'/w, y'/w, z'/w)

where

.. math::

    (x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix}

and

.. math::

    w =  \fork{w'}{if $w' \ne 0$}{\infty}{otherwise}

Note that the function transforms a sparse set of 2D or 3D vectors. If you want to transform an image using perspective transformation, use
:func:`warpPerspective` . If you have an inverse task, i.e. want to compute the most probable perspective transformation out of several pairs of corresponding points, you can use
:func:`getPerspectiveTransform` or
:func:`findHomography` .

See Also:
:func:`transform`,:func:`warpPerspective`,:func:`getPerspectiveTransform`,:func:`findHomography`

.. index:: phase

.. _phase:

phase
-----

.. c:function:: void phase(const Mat& x, const Mat& y, Mat& angle, bool angleInDegrees=false)

    Calculates the rotation angle of 2d vectors

    :param x: The source floating-point array of x-coordinates of 2D vectors

    :param y: The source array of y-coordinates of 2D vectors; must have the same size and the same type as  ``x``     
    
    :param angle: The destination array of vector angles; it will have the same size and same type as  ``x``
    
    :param angleInDegrees: When it is true, the function will compute angle in degrees, otherwise they will be measured in radians

The function ``phase`` computes the rotation angle of each 2D vector that is formed from the corresponding elements of ``x`` and ``y`` :

.. math::

    \texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))

The angle estimation accuracy is
:math:`\sim\,0.3^\circ` , when ``x(I)=y(I)=0`` , the corresponding ``angle`` (I) is set to
:math:`0` .

See Also:

.. index:: polarToCart

.. _polarToCart:

polarToCart
-----------

.. c:function:: void polarToCart(const Mat& magnitude, const Mat& angle, Mat& x, Mat& y, bool angleInDegrees=false)

    Computes x and y coordinates of 2D vectors from their magnitude and angle.

    :param magnitude: The source floating-point array of magnitudes of 2D vectors. It can be an empty matrix ( ``=Mat()`` ) - in this case the function assumes that all the magnitudes are =1. If it's not empty, it must have the same size and same type as  ``angle``
    
    :param angle: The source floating-point array of angles of the 2D vectors

    :param x: The destination array of x-coordinates of 2D vectors; will have the same size and the same type as  ``angle``     
    
    :param y: The destination array of y-coordinates of 2D vectors; will have the same size and the same type as  ``angle``     
    
    :param angleInDegrees: When it is true, the input angles are measured in degrees, otherwise they are measured in radians

The function ``polarToCart`` computes the cartesian coordinates of each 2D vector represented by the corresponding elements of ``magnitude`` and ``angle`` :

.. math::

    \begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}

The relative accuracy of the estimated coordinates is
:math:`\sim\,10^{-6}` .

See Also:
:func:`cartToPolar`,:func:`magnitude`,:func:`phase`,:func:`exp`,:func:`log`,:func:`pow`,:func:`sqrt`

.. index:: pow

.. _pow:

pow
---

.. c:function:: void pow(const Mat& src, double p, Mat& dst)

.. c:function:: void pow(const MatND& src, double p, MatND& dst)

    Raises every array element to a power.

    :param src: The source array

    :param p: The exponent of power

    :param dst: The destination array; will have the same size and the same type as  ``src``

The function ``pow`` raises every element of the input array to ``p`` :

.. math::

    \texttt{dst} (I) =  \fork{\texttt{src}(I)^p}{if \texttt{p} is integer}{|\texttt{src}(I)|^p}{otherwise}

That is, for a non-integer power exponent the absolute values of input array elements are used. However, it is possible to get true values for negative values using some extra operations, as the following example, computing the 5th root of array ``src`` , shows: ::

    Mat mask = src < 0;
    pow(src, 1./5, dst);
    subtract(Scalar::all(0), dst, dst, mask);


For some values of ``p`` , such as integer values, 0.5, and -0.5, specialized faster algorithms are used.

See Also:
:func:`sqrt`,:func:`exp`,:func:`log`,:func:`cartToPolar`,:func:`polarToCart`

.. index:: RNG

.. _RNG:

RNG
---

Random number generator class. ::

    class CV_EXPORTS RNG
    {
    public:
        enum { UNIFORM=0, NORMAL=1 };

        // constructors
        RNG();
        RNG(uint64 state);

        // returns 32-bit unsigned random number
        unsigned next();

        // return random numbers of the specified type
        operator uchar();
        operator schar();
        operator ushort();
        operator short();
        operator unsigned();
            // returns a random integer sampled uniformly from [0, N).
            unsigned operator()(unsigned N);
            unsigned operator()();
        operator int();
        operator float();
        operator double();
        // returns a random number sampled uniformly from [a, b) range
        int uniform(int a, int b);
        float uniform(float a, float b);
        double uniform(double a, double b);

        // returns Gaussian random number with zero mean.
            double gaussian(double sigma);

        // fills array with random numbers sampled from the specified distribution
        void fill( Mat& mat, int distType, const Scalar& a, const Scalar& b );
        void fill( MatND& mat, int distType, const Scalar& a, const Scalar& b );

        // internal state of the RNG (could change in the future)
        uint64 state;
    };


The class ``RNG`` implements random number generator. It encapsulates the RNG state (currently, a 64-bit integer) and  has methods to return scalar random values and to fill arrays with random values. Currently it supports uniform and Gaussian (normal) distributions. The generator uses Multiply-With-Carry algorithm, introduced by G. Marsaglia (
http://en.wikipedia.org/wiki/Multiply-with-carry
). Gaussian-distribution random numbers are generated using Ziggurat algorithm (
http://en.wikipedia.org/wiki/Ziggurat_algorithm
), introduced by G. Marsaglia and W. W. Tsang.

.. index:: RNG::RNG

.. _RNG::RNG:

RNG::RNG
------------
.. c:function:: RNG::RNG()

.. c:function:: RNG::RNG(uint64 state)

    RNG constructors

    :param state: the 64-bit value used to initialize the RNG

These are the RNG constructors. The first form sets the state to some pre-defined value, equal to ``2**32-1`` in the current implementation. The second form sets the state to the specified value. If the user passed ``state=0`` , the constructor uses the above default value instead, to avoid the singular random number sequence, consisting of all zeros.

.. index:: RNG::next

.. _RNG::next:

RNG::next
-------------
.. c:function:: unsigned RNG::next()

    Returns the next random number

The method updates the state using MWC algorithm and returns the next 32-bit random number.

.. index:: RNG::operator T

.. _RNG::operator T:

RNG::operator T
---------------

.. cpp:function:: RNG::operator uchar()

.. cpp:function:: RNG::operator schar()

.. cpp:function:: RNG::operator ushort()

.. cpp:function:: RNG::operator short()

.. cpp:function:: RNG::operator int()

.. cpp:function:: RNG::operator float()

.. cpp:function:: RNG::operator double()

    Returns the next random number of the specified type

Each of the methods updates the state using MWC algorithm and returns the next random number of the specified type. In the case of integer types the returned number is from the whole available value range for the specified type. In the case of floating-point types the returned value is from ``[0,1)`` range.

.. index:: RNG::operator ()

.. _RNG::operator ():

RNG::operator ()
--------------------
.. c:function:: unsigned RNG::operator ()()

.. c:function:: unsigned RNG::operator ()(unsigned N)

    Returns the next random number

    :param N: The upper non-inclusive boundary of the returned random number

The methods transforms the state using MWC algorithm and returns the next random number. The first form is equivalent to
:func:`RNG::next` , the second form returns the random number modulo ``N`` , i.e. the result will be in the range ``[0, N)`` .

.. index:: RNG::uniform

.. _RNG::uniform:

RNG::uniform
----------------
.. c:function:: int RNG::uniform(int a, int b)

.. c:function:: float RNG::uniform(float a, float b)

.. c:function:: double RNG::uniform(double a, double b)

    Returns the next random number sampled from the uniform distribution

    :param a: The lower inclusive boundary of the returned random numbers

    :param b: The upper non-inclusive boundary of the returned random numbers

The methods transforms the state using MWC algorithm and returns the next uniformly-distributed random number of the specified type, deduced from the input parameter type, from the range ``[a, b)`` . There is one nuance, illustrated by the following sample: ::

    RNG rng;

    // will always produce 0
    double a = rng.uniform(0, 1);

    // will produce double from [0, 1)
    double a1 = rng.uniform((double)0, (double)1);

    // will produce float from [0, 1)
    double b = rng.uniform(0.f, 1.f);

    // will produce double from [0, 1)
    double c = rng.uniform(0., 1.);

    // will likely cause compiler error because of ambiguity:
    //  RNG::uniform(0, (int)0.999999)? or RNG::uniform((double)0, 0.99999)?
    double d = rng.uniform(0, 0.999999);


That is, the compiler does not take into account type of the variable that you assign the result of ``RNG::uniform`` to, the only thing that matters to it is the type of ``a`` and ``b`` parameters. So if you want a floating-point random number, but the range boundaries are integer numbers, either put dots in the end, if they are constants, or use explicit type cast operators, as in ``a1`` initialization above.

.. index:: RNG::gaussian

.. _RNG::gaussian:

RNG::gaussian
-----------------
.. c:function:: double RNG::gaussian(double sigma)

    Returns the next random number sampled from the Gaussian distribution

    :param sigma: The standard deviation of the distribution

The methods transforms the state using MWC algorithm and returns the next random number from the Gaussian distribution ``N(0,sigma)`` . That is, the mean value of the returned random numbers will be zero and the standard deviation will be the specified ``sigma`` .

.. index:: RNG::fill

.. _RNG::fill:

RNG::fill
-------------
.. c:function:: void RNG::fill( Mat& mat, int distType, const Scalar& a, const Scalar& b )

.. c:function:: void RNG::fill( MatND& mat, int distType, const Scalar& a, const Scalar& b )

    Fill arrays with random numbers

    :param mat: 2D or N-dimensional matrix. Currently matrices with more than 4 channels are not supported by the methods. Use  :func:`reshape`  as a possible workaround.

    :param distType: The distribution type, ``RNG::UNIFORM``  or  ``RNG::NORMAL``
    
    :param a: The first distribution parameter. In the case of uniform distribution this is inclusive lower boundary. In the case of normal distribution this is mean value.

    :param b: The second distribution parameter. In the case of uniform distribution this is non-inclusive upper boundary. In the case of normal distribution this is standard deviation.

Each of the methods fills the matrix with the random values from the specified distribution. As the new numbers are generated, the RNG state is updated accordingly. In the case of multiple-channel images every channel is filled independently, i.e. RNG can not generate samples from multi-dimensional Gaussian distribution with non-diagonal covariation matrix directly. To do that, first, generate matrix from the distribution
:math:`N(0, I_n)` , i.e. Gaussian distribution with zero mean and identity covariation matrix, and then transform it using
:func:`transform` and the specific covariation matrix.

.. index:: randu

.. _randu:

randu
-----

.. c:function:: template<typename _Tp> _Tp randu()

.. c:function:: void randu(Mat& mtx, const Scalar& low, const Scalar& high)

    Generates a single uniformly-distributed random number or array of random numbers

    :param mtx: The output array of random numbers. The array must be pre-allocated and have 1 to 4 channels

    :param low: The inclusive lower boundary of the generated random numbers

    :param high: The exclusive upper boundary of the generated random numbers

The template functions ``randu`` generate and return the next uniformly-distributed random value of the specified type. ``randu<int>()`` is equivalent to ``(int)theRNG();`` etc. See
:func:`RNG` description.

The second non-template variant of the function fills the matrix ``mtx`` with uniformly-distributed random numbers from the specified range:

.. math::

    \texttt{low} _c  \leq \texttt{mtx} (I)_c <  \texttt{high} _c

See Also:
:func:`RNG`,:func:`randn`,:func:`theRNG` .

.. index:: randn

.. _randn:

randn
-----

.. c:function:: void randn(Mat& mtx, const Scalar& mean, const Scalar& stddev)

    Fills array with normally distributed random numbers

    :param mtx: The output array of random numbers. The array must be pre-allocated and have 1 to 4 channels

    :param mean: The mean value (expectation) of the generated random numbers

    :param stddev: The standard deviation of the generated random numbers

The function ``randn`` fills the matrix ``mtx`` with normally distributed random numbers with the specified mean and standard deviation.
is applied to the generated numbers (i.e. the values are clipped)

See Also:
:func:`RNG`,:func:`randu`

.. index:: randShuffle

.. randShuffle:

randShuffle
-----------

.. c:function:: void randShuffle(Mat& mtx, double iterFactor=1., RNG* rng=0)

    Shuffles the array elements randomly

    :param mtx: The input/output numerical 1D array

    :param iterFactor: The scale factor that determines the number of random swap operations. See the discussion

    :param rng: The optional random number generator used for shuffling. If it is zero, :func:`theRNG` () is used instead

The function ``randShuffle`` shuffles the specified 1D array by randomly choosing pairs of elements and swapping them. The number of such swap operations will be ``mtx.rows*mtx.cols*iterFactor`` See Also:
:func:`RNG`,:func:`sort`

.. index:: reduce

.. _reduce:

reduce
------

.. c:function:: void reduce(const Mat& mtx, Mat& vec, int dim, int reduceOp, int dtype=-1)

    Reduces a matrix to a vector

    :param mtx: The source 2D matrix

    :param vec: The destination vector. Its size and type is defined by  ``dim``  and  ``dtype``  parameters

    :param dim: The dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row and 1 means that the matrix is reduced to a single column

    :param reduceOp: The reduction operation, one of:

            * **CV_REDUCE_SUM** The output is the sum of all of the matrix's rows/columns.

            * **CV_REDUCE_AVG** The output is the mean vector of all of the matrix's rows/columns.

            * **CV_REDUCE_MAX** The output is the maximum (column/row-wise) of all of the matrix's rows/columns.

            * **CV_REDUCE_MIN** The output is the minimum (column/row-wise) of all of the matrix's rows/columns.

    :param dtype: When it is negative, the destination vector will have the same type as the source matrix, otherwise, its type will be  ``CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), mtx.channels())``
    
The function ``reduce`` reduces matrix to a vector by treating the matrix rows/columns as a set of 1D vectors and performing the specified operation on the vectors until a single row/column is obtained. For example, the function can be used to compute horizontal and vertical projections of an raster image. In the case of ``CV_REDUCE_SUM`` and ``CV_REDUCE_AVG`` the output may have a larger element bit-depth to preserve accuracy. And multi-channel arrays are also supported in these two reduction modes.

See Also: :func:`repeat`

.. index:: repeat

.. _repeat:

repeat
------

.. c:function:: void repeat(const Mat& src, int ny, int nx, Mat& dst)

.. c:function:: Mat repeat(const Mat& src, int ny, int nx)

    Fill the destination array with repeated copies of the source array.

    :param src: The source array to replicate

    :param dst: The destination array; will have the same type as  ``src``
    
    :param ny: How many times the  ``src``  is repeated along the vertical axis

    :param nx: How many times the  ``src``  is repeated along the horizontal axis

The functions
:func:`repeat` duplicate the source array one or more times along each of the two axes:

.. math::

    \texttt{dst} _{ij}= \texttt{src} _{i \mod \texttt{src.rows} , \; j \mod \texttt{src.cols} }

The second variant of the function is more convenient to use with
:ref:`MatrixExpressions` See Also:
:func:`reduce`,:ref:`MatrixExpressions`

.. index:: saturate_cast

.. _saturate_cast_:

saturate_cast
-------------

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(unsigned char v)

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(signed char v)

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(unsigned short v)

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(signed short v)

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(int v)

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(unsigned int v)

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(float v)

.. c:function:: template<typename _Tp> inline _Tp saturate_cast(double v)

    Template function for accurate conversion from one primitive type to another

    :param v: The function parameter

The functions ``saturate_cast`` resembles the standard C++ cast operations, such as ``static_cast<T>()`` etc. They perform an efficient and accurate conversion from one primitive type to another, see the introduction. "saturate" in the name means that when the input value ``v`` is out of range of the target type, the result will not be formed just by taking low bits of the input, but instead the value will be clipped. For example: ::

    uchar a = saturate_cast<uchar>(-100); // a = 0 (UCHAR_MIN)
    short b = saturate_cast<short>(33333.33333); // b = 32767 (SHRT_MAX)


Such clipping is done when the target type is ``unsigned char, signed char, unsigned short or signed short`` - for 32-bit integers no clipping is done.

When the parameter is floating-point value and the target type is an integer (8-, 16- or 32-bit), the floating-point value is first rounded to the nearest integer and then clipped if needed (when the target type is 8- or 16-bit).

This operation is used in most simple or complex image processing functions in OpenCV.

See Also:
:func:`add`,:func:`subtract`,:func:`multiply`,:func:`divide`,:func:`Mat::convertTo`

.. index:: scaleAdd

.. _scaleAdd:

scaleAdd
--------

.. c:function:: void scaleAdd(const Mat& src1, double scale, const Mat& src2, Mat& dst)

.. c:function:: void scaleAdd(const MatND& src1, double scale, const MatND& src2, MatND& dst)

    Calculates the sum of a scaled array and another array.

    :param src1: The first source array

    :param scale: Scale factor for the first array

    :param src2: The second source array; must have the same size and the same type as  ``src1``
    
    :param dst: The destination array; will have the same size and the same type as  ``src1``
    
The function ``scaleAdd`` is one of the classical primitive linear algebra operations, known as ``DAXPY`` or ``SAXPY`` in `BLAS <http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_. It calculates the sum of a scaled array and another array:

.. math::

    \texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)

The function can also be emulated with a matrix expression, for example: ::

    Mat A(3, 3, CV_64F);
    ...
    A.row(0) = A.row(1)*2 + A.row(2);


See Also:
:func:`add`,:func:`addWeighted`,:func:`subtract`,:func:`Mat::dot`,:func:`Mat::convertTo`,:ref:`MatrixExpressions`

.. index:: setIdentity

.. _setIdentity:

setIdentity
-----------

.. c:function:: void setIdentity(Mat& dst, const Scalar& value=Scalar(1))

    Initializes a scaled identity matrix

    :param dst: The matrix to initialize (not necessarily square)

    :param value: The value to assign to the diagonal elements

The function
:func:`setIdentity` initializes a scaled identity matrix:

.. math::

    \texttt{dst} (i,j)= \fork{\texttt{value}}{ if $i=j$}{0}{otherwise}

The function can also be emulated using the matrix initializers and the matrix expressions: ::

    Mat A = Mat::eye(4, 3, CV_32F)*5;
    // A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]


See Also:
:func:`Mat::zeros`,:func:`Mat::ones`,:ref:`MatrixExpressions`,:func:`Mat::setTo`,:func:`Mat::operator=`

.. index:: solve

.. _solve:

solve
-----

.. c:function:: bool solve(const Mat& src1, const Mat& src2, Mat& dst, int flags=DECOMP_LU)

    Solves one or more linear systems or least-squares problems.

    :param src1: The input matrix on the left-hand side of the system

    :param src2: The input matrix on the right-hand side of the system

    :param dst: The output solution

    :param flags: The solution (matrix inversion) method

            * **DECOMP_LU** Gaussian elimination with optimal pivot element chosen

            * **DECOMP_CHOLESKY** Cholesky  :math:`LL^T`  factorization; the matrix  ``src1``  must be symmetrical and positively defined

            * **DECOMP_EIG** Eigenvalue decomposition; the matrix  ``src1``  must be symmetrical

            * **DECOMP_SVD** Singular value decomposition (SVD) method; the system can be over-defined and/or the matrix  ``src1``  can be singular

            * **DECOMP_QR** QR factorization; the system can be over-defined and/or the matrix  ``src1``  can be singular

            * **DECOMP_NORMAL** While all the previous flags are mutually exclusive, this flag can be used together with any of the previous. It means that the normal equations  :math:`\texttt{src1}^T\cdot\texttt{src1}\cdot\texttt{dst}=\texttt{src1}^T\texttt{src2}`  are solved instead of the original system  :math:`\texttt{src1}\cdot\texttt{dst}=\texttt{src2}`
            
The function ``solve`` solves a linear system or least-squares problem (the latter is possible with SVD or QR methods, or by specifying the flag ``DECOMP_NORMAL`` ):

.. math::

    \texttt{dst} =  \arg \min _X \| \texttt{src1} \cdot \texttt{X} -  \texttt{src2} \|

If ``DECOMP_LU`` or ``DECOMP_CHOLESKY`` method is used, the function returns 1 if ``src1`` (or
:math:`\texttt{src1}^T\texttt{src1}` ) is non-singular and 0 otherwise; in the latter case ``dst`` is not valid. Other methods find some pseudo-solution in the case of singular left-hand side part.

Note that if you want to find unity-norm solution of an under-defined singular system
:math:`\texttt{src1}\cdot\texttt{dst}=0` , the function ``solve`` will not do the work. Use
:func:`SVD::solveZ` instead.

See Also:
:func:`invert`,:func:`SVD`,:func:`eigen`

.. index:: solveCubic

.. _solveCubic:

solveCubic
--------------
.. c:function:: void solveCubic(const Mat& coeffs, Mat& roots)

    Finds the real roots of a cubic equation.

    :param coeffs: The equation coefficients, an array of 3 or 4 elements

    :param roots: The destination array of real roots which will have 1 or 3 elements

The function ``solveCubic`` finds the real roots of a cubic equation:

(if coeffs is a 4-element vector)

.. math::

    \texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0

or (if coeffs is 3-element vector):

.. math::

    x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0

The roots are stored to ``roots`` array.

.. index:: solvePoly

.. _solvePoly:

solvePoly
---------

.. c:function:: void solvePoly(const Mat& coeffs, Mat& roots, int maxIters=20, int fig=100)

    Finds the real or complex roots of a polynomial equation

    :param coeffs: The array of polynomial coefficients

    :param roots: The destination (complex) array of roots

    :param maxIters: The maximum number of iterations the algorithm does

    :param fig:

The function ``solvePoly`` finds real and complex roots of a polynomial equation:

.. math::

    \texttt{coeffs} [0] x^{n} +  \texttt{coeffs} [1] x^{n-1} + ... +  \texttt{coeffs} [n-1] x +  \texttt{coeffs} [n] = 0

.. index:: sort

.. _sort:

sort
----

.. c:function:: void sort(const Mat& src, Mat& dst, int flags)

    Sorts each row or each column of a matrix

    :param src: The source single-channel array

    :param dst: The destination array of the same size and the same type as  ``src``
    
    :param flags: The operation flags, a combination of the following values:

            * **CV_SORT_EVERY_ROW** Each matrix row is sorted independently

            * **CV_SORT_EVERY_COLUMN** Each matrix column is sorted independently. This flag and the previous one are mutually exclusive

            * **CV_SORT_ASCENDING** Each matrix row is sorted in the ascending order

            * **CV_SORT_DESCENDING** Each matrix row is sorted in the descending order. This flag and the previous one are also mutually exclusive

The function ``sort`` sorts each matrix row or each matrix column in ascending or descending order. If you want to sort matrix rows or columns lexicographically, you can use STL ``std::sort`` generic function with the proper comparison predicate.

See Also:
:func:`sortIdx`,:func:`randShuffle`

.. index:: sortIdx

.. _sortIdx:

sortIdx
-------

.. c:function:: void sortIdx(const Mat& src, Mat& dst, int flags)

    Sorts each row or each column of a matrix

    :param src: The source single-channel array

    :param dst: The destination integer array of the same size as  ``src``
    
    :param flags: The operation flags, a combination of the following values:

            * **CV_SORT_EVERY_ROW** Each matrix row is sorted independently

            * **CV_SORT_EVERY_COLUMN** Each matrix column is sorted independently. This flag and the previous one are mutually exclusive

            * **CV_SORT_ASCENDING** Each matrix row is sorted in the ascending order

            * **CV_SORT_DESCENDING** Each matrix row is sorted in the descending order. This flag and the previous one are also mutually exclusive

The function ``sortIdx`` sorts each matrix row or each matrix column in ascending or descending order. Instead of reordering the elements themselves, it stores the indices of sorted elements in the destination array. For example: ::

    Mat A = Mat::eye(3,3,CV_32F), B;
    sortIdx(A, B, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
    // B will probably contain
    // (because of equal elements in A some permutations are possible):
    // [[1, 2, 0], [0, 2, 1], [0, 1, 2]]


See Also:
:func:`sort`,:func:`randShuffle`

.. index:: split

.. _split:

split
-----

.. c:function:: void split(const Mat& mtx, Mat* mv)

.. c:function:: void split(const Mat& mtx, vector<Mat>& mv)

.. c:function:: void split(const MatND& mtx, MatND* mv)

.. c:function:: void split(const MatND& mtx, vector<MatND>& mv)

    Divides multi-channel array into several single-channel arrays

    :param mtx: The source multi-channel array

    :param mv: The destination array or vector of arrays; The number of arrays must match  ``mtx.channels()`` . The arrays themselves will be reallocated if needed

The functions ``split`` split multi-channel array into separate single-channel arrays:

.. math::

    \texttt{mv} [c](I) =  \texttt{mtx} (I)_c

If you need to extract a single-channel or do some other sophisticated channel permutation, use
:func:`mixChannels` See Also:
:func:`merge`,:func:`mixChannels`,:func:`cvtColor`

.. index:: sqrt

.. _sqrt:

sqrt
----

.. c:function:: void sqrt(const Mat& src, Mat& dst)

.. c:function:: void sqrt(const MatND& src, MatND& dst)

    Calculates square root of array elements

    :param src: The source floating-point array

    :param dst: The destination array; will have the same size and the same type as  ``src``
    
The functions ``sqrt`` calculate square root of each source array element. in the case of multi-channel arrays each channel is processed independently. The accuracy is approximately the same as of the built-in ``std::sqrt`` .

See Also:
:func:`pow`,:func:`magnitude`

.. index:: subtract

.. _subtract:

subtract
--------

.. c:function:: void subtract(const Mat& src1, const Mat& src2, Mat& dst)

.. c:function:: void subtract(const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask)

.. c:function:: void subtract(const Mat& src1, const Scalar& sc, Mat& dst, const Mat& mask=Mat())

.. c:function:: void subtract(const Scalar& sc, const Mat& src2, Mat& dst, const Mat& mask=Mat())

.. c:function:: void subtract(const MatND& src1, const MatND& src2, MatND& dst)

.. c:function:: void subtract(const MatND& src1, const MatND& src2, MatND& dst, const MatND& mask)

.. c:function:: void subtract(const MatND& src1, const Scalar& sc, MatND& dst, const MatND& mask=MatND())

.. c:function:: void subtract(const Scalar& sc, const MatND& src2, MatND& dst, const MatND& mask=MatND())

    Calculates per-element difference between two arrays or array and a scalar

    :param src1: The first source array

    :param src2: The second source array. It must have the same size and same type as  ``src1``
    
    :param sc: Scalar; the first or the second input parameter

    :param dst: The destination array; it will have the same size and same type as  ``src1`` ; see  ``Mat::create``     
    
    :param mask: The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

The functions ``subtract`` compute

*
    the difference between two arrays

    .. math::

        \texttt{dst} (I) =  \texttt{saturate} ( \texttt{src1} (I) -  \texttt{src2} (I)) \quad \texttt{if mask} (I) \ne0

*
    the difference between array and a scalar:

    .. math::

        \texttt{dst} (I) =  \texttt{saturate} ( \texttt{src1} (I) -  \texttt{sc} ) \quad \texttt{if mask} (I) \ne0

*
    the difference between scalar and an array:

    .. math::

        \texttt{dst} (I) =  \texttt{saturate} ( \texttt{sc} -  \texttt{src2} (I)) \quad \texttt{if mask} (I) \ne0

where ``I`` is multi-dimensional index of array elements.

The first function in the above list can be replaced with matrix expressions: ::

    dst = src1 - src2;
    dst -= src2; // equivalent to subtract(dst, src2, dst);


See Also:
:func:`add`,:func:`addWeighted`,:func:`scaleAdd`,:func:`convertScale`,:ref:`MatrixExpressions`,.

.. index:: SVD

.. _SVD:

SVD
---
.. c:type:: SVD

Class for computing Singular Value Decomposition ::

    class SVD
    {
    public:
        enum { MODIFY_A=1, NO_UV=2, FULL_UV=4 };
        // default empty constructor
        SVD();
        // decomposes A into u, w and vt: A = u*w*vt;
        // u and vt are orthogonal, w is diagonal
        SVD( const Mat& A, int flags=0 );
        // decomposes A into u, w and vt.
        SVD& operator ()( const Mat& A, int flags=0 );

        // finds such vector x, norm(x)=1, so that A*x = 0,
        // where A is singular matrix
        static void solveZ( const Mat& A, Mat& x );
        // does back-subsitution:
        // x = vt.t()*inv(w)*u.t()*rhs ~ inv(A)*rhs
        void backSubst( const Mat& rhs, Mat& x ) const;

        Mat u; // the left orthogonal matrix
        Mat w; // vector of singular values
        Mat vt; // the right orthogonal matrix
    };


The class ``SVD`` is used to compute Singular Value Decomposition of a floating-point matrix and then use it to solve least-square problems, under-determined linear systems, invert matrices, compute condition numbers etc.
For a bit faster operation you can pass ``flags=SVD::MODIFY_A|...`` to modify the decomposed matrix when it is not necessarily to preserve it. If you want to compute condition number of a matrix or absolute value of its determinant - you do not need ``u`` and ``vt`` , so you can pass ``flags=SVD::NO_UV|...`` . Another flag ``FULL_UV`` indicates that full-size ``u`` and ``vt`` must be computed, which is not necessary most of the time.

See Also:
:func:`invert`,:func:`solve`,:func:`eigen`,:func:`determinant`

.. index:: SVD::SVD

.. _SVD::SVD:

SVD::SVD
--------

.. c:function:: SVD::SVD()

.. c:function:: SVD::SVD( const Mat& A, int flags=0 )

    SVD constructors

    :param A: The decomposed matrix

    :param flags: Operation flags

        * **SVD::MODIFY_A** The algorithm can modify the decomposed matrix. It can save some space and speed-up processing a bit

        * **SVD::NO_UV** Indicates that only the vector of singular values  ``w``  is to be computed, while  ``u``  and  ``vt``  will be set to empty matrices

        * **SVD::FULL_UV** When the matrix is not square, by default the algorithm produces  ``u``  and  ``vt``  matrices of sufficiently large size for the further  ``A``  reconstruction. If, however, ``FULL_UV``  flag is specified, ``u``  and  ``vt``  will be full-size square orthogonal matrices.

The first constructor initializes empty ``SVD`` structure. The second constructor initializes empty ``SVD`` structure and then calls
:func:`SVD::operator ()` .

.. index:: SVD::operator ()

.. _SVD::operator ():

SVD::operator ()
----------------

.. c:function:: SVD& SVD::operator ()( const Mat& A, int flags=0 )

    Performs SVD of a matrix

    :param A: The decomposed matrix

    :param flags: Operation flags

        * **SVD::MODIFY_A** The algorithm can modify the decomposed matrix. It can save some space and speed-up processing a bit

        * **SVD::NO_UV** Only singular values are needed. The algorithm will not compute  ``u``  and  ``vt``  matrices

        * **SVD::FULL_UV** When the matrix is not square, by default the algorithm produces  ``u``  and  ``vt``  matrices of sufficiently large size for the further  ``A``  reconstruction. If, however, ``FULL_UV``  flag is specified, ``u``  and  ``vt``  will be full-size square orthogonal matrices.

The operator performs singular value decomposition of the supplied matrix. The ``u``,``vt`` and the vector of singular values ``w`` are stored in the structure. The same ``SVD`` structure can be reused many times with different matrices. Each time, if needed, the previous ``u``,``vt`` and ``w`` are reclaimed and the new matrices are created, which is all handled by
:func:`Mat::create` .

.. index:: SVD::solveZ

.. _SVD::solveZ:

SVD::solveZ
-----------

.. c:function:: static void SVD::solveZ( const Mat& A, Mat& x )

    Solves under-determined singular linear system

    :param A: The left-hand-side matrix.

    :param x: The found solution

The method finds unit-length solution
**x**
of the under-determined system
:math:`A x = 0` . Theory says that such system has infinite number of solutions, so the algorithm finds the unit-length solution as the right singular vector corresponding to the smallest singular value (which should be 0). In practice, because of round errors and limited floating-point accuracy, the input matrix can appear to be close-to-singular rather than just singular. So, strictly speaking, the algorithm solves the following problem:

.. math::

    x^* =  \arg \min _{x:  \| x \| =1}  \| A  \cdot x  \|

.. index:: SVD::backSubst

.. _SVD::backSubst:

SVD::backSubst
--------------

.. c:function:: void SVD::backSubst( const Mat& rhs, Mat& x ) const

    Performs singular value back substitution

    :param rhs: The right-hand side of a linear system  :math:`\texttt{A} \texttt{x} = \texttt{rhs}`  being solved, where  ``A``  is the matrix passed to  :func:`SVD::SVD`  or  :func:`SVD::operator ()`
    
    :param x: The found solution of the system

The method computes back substitution for the specified right-hand side:

.. math::

    \texttt{x} =  \texttt{vt} ^T  \cdot diag( \texttt{w} )^{-1}  \cdot \texttt{u} ^T  \cdot \texttt{rhs} \sim \texttt{A} ^{-1}  \cdot \texttt{rhs}

Using this technique you can either get a very accurate solution of convenient linear system, or the best (in the least-squares terms) pseudo-solution of an overdetermined linear system. Note that explicit SVD with the further back substitution only makes sense if you need to solve many linear systems with the same left-hand side (e.g. ``A`` ). If all you need is to solve a single system (possibly with multiple ``rhs`` immediately available), simply call
:func:`solve` add pass ``DECOMP_SVD`` there - it will do absolutely the same thing.

.. index:: sum

.. _sum:

sum
---

.. c:function:: Scalar sum(const Mat& mtx)

.. c:function:: Scalar sum(const MatND& mtx)

    Calculates sum of array elements

    :param mtx: The source array; must have 1 to 4 channels

The functions ``sum`` calculate and return the sum of array elements, independently for each channel.

See Also:
:func:`countNonZero`,:func:`mean`,:func:`meanStdDev`,:func:`norm`,:func:`minMaxLoc`,:func:`reduce`

.. index:: theRNG

theRNG
------

.. c:function:: RNG& theRNG()

    Returns the default random number generator

The function ``theRNG`` returns the default random number generator. For each thread there is separate random number generator, so you can use the function safely in multi-thread environments. If you just need to get a single random number using this generator or initialize an array, you can use
:func:`randu` or
:func:`randn` instead. But if you are going to generate many random numbers inside a loop, it will be much faster to use this function to retrieve the generator and then use ``RNG::operator _Tp()`` .

See Also:
:func:`RNG`,:func:`randu`,:func:`randn`

.. index:: trace

.. _trace:

trace
-----

.. c:function:: Scalar trace(const Mat& mtx)

    Returns the trace of a matrix

    :param mtx: The source matrix

The function ``trace`` returns the sum of the diagonal elements of the matrix ``mtx`` .

.. math::

    \mathrm{tr} ( \texttt{mtx} ) =  \sum _i  \texttt{mtx} (i,i)

.. index:: transform

.. _transform:

transform
---------

.. c:function:: void transform(const Mat& src, Mat& dst, const Mat& mtx )

    Performs matrix transformation of every array element.

    :param src: The source array; must have as many channels (1 to 4) as  ``mtx.cols``  or  ``mtx.cols-1``
    
    :param dst: The destination array; will have the same size and depth as  ``src``  and as many channels as  ``mtx.rows``     
    
    :param mtx: The transformation matrix

The function ``transform`` performs matrix transformation of every element of array ``src`` and stores the results in ``dst`` :

.. math::

    \texttt{dst} (I) =  \texttt{mtx} \cdot \texttt{src} (I)

(when ``mtx.cols=src.channels()`` ), or

.. math::

    \texttt{dst} (I) =  \texttt{mtx} \cdot [ \texttt{src} (I); 1]

(when ``mtx.cols=src.channels()+1`` )

That is, every element of an ``N`` -channel array ``src`` is
considered as ``N`` -element vector, which is transformed using
a
:math:`\texttt{M} \times \texttt{N}` or
:math:`\texttt{M} \times \texttt{N+1}` matrix ``mtx`` into
an element of ``M`` -channel array ``dst`` .

The function may be used for geometrical transformation of
:math:`N` -dimensional
points, arbitrary linear color space transformation (such as various kinds of RGB
:math:`\rightarrow` YUV transforms), shuffling the image channels and so forth.

See Also:
:func:`perspectiveTransform`,:func:`getAffineTransform`,:func:`estimateRigidTransform`,:func:`warpAffine`,:func:`warpPerspective`

.. index:: transpose

.. _transpose:

transpose
---------

.. c:function:: void transpose(const Mat& src, Mat& dst)

    Transposes a matrix

    :param src: The source array

    :param dst: The destination array of the same type as  ``src``
    
The function :func:`transpose` transposes the matrix ``src`` :

.. math::

    \texttt{dst} (i,j) =  \texttt{src} (j,i)

Note that no complex conjugation is done in the case of a complex
matrix, it should be done separately if needed.
