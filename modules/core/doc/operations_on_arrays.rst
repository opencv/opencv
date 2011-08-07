Operations on Arrays
====================

.. highlight:: cpp

abs
---
Computes an absolute value of each matrix element.

.. ocv:function:: MatExpr abs(const Mat& src)
.. ocv:function:: MatExpr abs(const MatExpr& src)

    :param src: Matrix or matrix expression.
    
``abs`` is a meta-function that is expanded to one of :ocv:func:`absdiff` forms:

    * ``C = abs(A-B)``     is equivalent to ``absdiff(A, B, C)``     

    * ``C = abs(A)``     is equivalent to ``absdiff(A, Scalar::all(0), C)``     

    * ``C = Mat_<Vec<uchar,n> >(abs(A*alpha + beta))``     is equivalent to :ocv:funcx:`convertScaleAbs` (A, C, alpha, beta)
    
The output matrix has the same size and the same type as the input one except for the last case, where ``C`` is ``depth=CV_8U`` .

    .. seealso:: :ref:`MatrixExpressions`, :ocv:func:`absdiff`


absdiff
-----------
Computes the per-element absolute difference between two arrays or between an array and a scalar.

.. ocv:function:: void absdiff(InputArray src1, InputArray src2, OutputArray dst)

.. ocv:pyfunction:: cv2.absdiff(src1, src2[, dst]) -> dst

.. ocv:cfunction:: void cvAbsDiff(const CvArr* src1, const CvArr* src2, CvArr* dst)
.. ocv:cfunction:: void cvAbsDiffS(const CvArr* src, CvArr* dst, CvScalar value)
.. ocv:pyoldfunction:: cv.AbsDiff(src1, src2, dst)-> None
.. ocv:pyoldfunction:: cv.AbsDiffS(src, dst, value)-> None

    :param src1: First input array or a scalar.
    
    :param src2: Second input array or a scalar.
    
    :param dst: Destination array that has the same size and type as ``src1`` (or ``src2``).
    
The function ``absdiff`` computes:

 *
    Absolute difference between two arrays when they have the same size and type:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)

 *
    Absolute difference between an array and a scalar when the second array is constructed from ``Scalar`` or has as many elements as the number of channels in ``src1``:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)

 *
    Absolute difference between a scalar and an array when the first array is constructed from ``Scalar`` or has as many elements as the number of channels in ``src2``:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)

    where  ``I`` is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently.


.. seealso:: :ocv:func:`abs`


add
-------

Computes the per-element sum of two arrays or an array and a scalar.

.. ocv:function:: void add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask=noArray(), int dtype=-1)

.. ocv:pyfunction:: cv2.add(src1, src2[, dst[, mask[, dtype]]]) -> dst

.. ocv:cfunction:: void cvAdd(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)
.. ocv:cfunction:: void cvAddS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.Add(src1, src2, dst, mask=None)-> None
.. ocv:pyoldfunction:: cv.AddS(src, value, dst, mask=None)-> None

    :param src1: First source array or a scalar.

    :param src2: Second source array or a scalar.
    
    :param dst: Destination array that has the same size and number of channels as the input array(s). The depth is defined by ``dtype`` or ``src1``/``src2``.
    
    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.
    
    :param dtype: Optional depth of the output array. See the discussion below.

The function ``add`` computes:

 *
    Sum of two arrays when both input arrays have the same size and the same number of channels:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0

 *
    Sum of an array and a scalar when ``src2`` is constructed from ``Scalar`` or has the same number of elements as ``src1.channels()``:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0

 *
    Sum of a scalar and an array when ``src1`` is constructed from ``Scalar`` or has the same number of elements as ``src2.channels()``:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0

    where ``I`` is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently.

The first function in the list above can be replaced with matrix expressions: ::

    dst = src1 + src2;
    dst += src1; // equivalent to add(dst, src1, dst);

The input arrays and the destination array can all have the same or different depths. For example, you can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit floating-point array. Depth of the output array is determined by the ``dtype`` parameter. In the second and third cases above, as well as in the first case, when ``src1.depth() == src2.depth()``, ``dtype`` can be set to the default ``-1``. In this case, the output array will have the same depth as the input array, be it ``src1``, ``src2`` or both.

.. seealso::
   
    :ocv:func:`subtract`,
    :ocv:func:`addWeighted`,
    :ocv:func:`scaleAdd`,
    :ocv:func:`Mat::convertTo`,
    :ref:`MatrixExpressions`



addWeighted
---------------
Computes the weighted sum of two arrays.

.. ocv:function:: void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1)

.. ocv:pyfunction:: cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst

.. ocv:cfunction:: void cvAddWeighted(const CvArr* src1, double alpha, const CvArr* src2, double beta, double gamma, CvArr* dst)
.. ocv:pyoldfunction:: cv.AddWeighted(src1, alpha, src2, beta, gamma, dst)-> None

    :param src1: First source array.

    :param alpha: Weight for the first array elements.

    :param src2: Second source array of the same size and channel number as  ``src1`` .
    
    :param beta: Weight for the second array elements.

    :param dst: Destination array that has the same size and number of channels as the input arrays.
    
    :param gamma: Scalar added to each sum.
    
    :param dtype: Optional depth of the destination array. When both input arrays have the same depth, ``dtype`` can be set to ``-1``, which will be equivalent to ``src1.depth()``.

The function ``addWeighted`` calculates the weighted sum of two arrays as follows:

.. math::

    \texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )

where ``I`` is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently.

The function can be replaced with a matrix expression: ::

    dst = src1*alpha + src2*beta + gamma;


.. seealso::

    :ocv:func:`add`,
    :ocv:func:`subtract`,
    :ocv:func:`scaleAdd`,
    :ocv:func:`Mat::convertTo`,
    :ref:`MatrixExpressions`



bitwise_and
-----------
Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar.

.. ocv:function:: void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask=noArray())

.. ocv:pyfunction:: cv2.bitwise_and(src1, src2[, dst[, mask]]) -> dst

.. ocv:cfunction:: void cvAnd(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)
.. ocv:cfunction:: void cvAndS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.And(src1, src2, dst, mask=None)-> None
.. ocv:pyoldfunction:: cv.AndS(src, value, dst, mask=None)-> None

    :param src1: First source array or a scalar.

    :param src2: Second source array or a scalar.

    :param dst: Destination arrayb that has the same size and type as the input array(s).
    
    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The function computes the per-element bit-wise logical conjunction for:

 *
    Two arrays when ``src1`` and ``src2`` have the same size:

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0

 *
    An array and a scalar when ``src2`` is constructed from ``Scalar`` or has the same number of elements as ``src1.channels()``:

    .. math::

        \texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0

 *
    A scalar and an array when ``src1`` is constructed from ``Scalar`` or has the same number of elements as ``src2.channels()``:

    .. math::

        \texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0


In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently. In the second and third cases above, the scalar is first converted to the array type.



bitwise_not
-----------
Inverts every bit of an array.

.. ocv:function:: void bitwise_not(InputArray src, OutputArray dst, InputArray mask=noArray())

.. ocv:pyfunction:: cv2.bitwise_not(src[, dst[, mask]]) -> dst

.. ocv:cfunction:: void cvNot(const CvArr* src, CvArr* dst)
.. ocv:pyoldfunction:: cv.Not(src, dst)-> None

    :param src: Source array.

    :param dst: Destination array that has the same size and type as the input array.
    
    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The function computes per-element bit-wise inversion of the source array:

.. math::

    \texttt{dst} (I) =  \neg \texttt{src} (I)

In case of a floating-point source array, its machine-specific bit representation (usually IEEE754-compliant) is used for the operation. In case of multi-channel arrays, each channel is processed independently.



bitwise_or
----------
Calculates the per-element bit-wise disjunction of two arrays or an array and a scalar.

.. ocv:function:: void bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask=noArray())

.. ocv:pyfunction:: cv2.bitwise_or(src1, src2[, dst[, mask]]) -> dst

.. ocv:cfunction:: void cvOr(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)
.. ocv:cfunction:: void cvOrS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.Or(src1, src2, dst, mask=None)-> None
.. ocv:pyoldfunction:: cv.OrS(src, value, dst, mask=None)-> None

    :param src1: First source array or a scalar.

    :param src2: Second source array or a scalar.

    :param dst: Destination array that has the same size and type as the input array(s).

    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The function computes the per-element bit-wise logical disjunction for:

 *
    Two arrays when ``src1`` and ``src2`` have the same size:

        .. math::

            \texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0

 *
    An array and a scalar when ``src2`` is constructed from ``Scalar`` or has the same number of elements as ``src1.channels()``:

        .. math::

            \texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0

 *
    A scalar and an array when ``src1`` is constructed from ``Scalar`` or has the same number of elements as ``src2.channels()``:

        .. math::

            \texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0


In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently. In the second and third cases above, the scalar is first converted to the array type.
    

bitwise_xor
-----------
Calculates the per-element bit-wise "exclusive or" operation on two arrays or an array and a scalar.

.. ocv:function:: void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask=noArray())

.. ocv:pyfunction:: cv2.bitwise_xor(src1, src2[, dst[, mask]]) -> dst

.. ocv:cfunction:: void cvXor(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)
.. ocv:cfunction:: void cvXorS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.Xor(src1, src2, dst, mask=None)-> None
.. ocv:pyoldfunction:: cv.XorS(src, value, dst, mask=None)-> None

    :param src1: First source array or a scalar.

    :param src2: Second source array or a scalar.

    :param dst: Destination array that has the same size and type as the input array(s).

    :param mask: Optional operation mask, 8-bit single channel array, that specifies elements of the destination array to be changed.

The function computes the per-element bit-wise logical "exclusive-or" operation for:

 *
    Two arrays when ``src1`` and ``src2`` have the same size:

        .. math::

            \texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0

 *
    An array and a scalar when ``src2`` is constructed from ``Scalar`` or has the same number of elements as ``src1.channels()``:

        .. math::

            \texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0

 *
    A scalar and an array when ``src1`` is constructed from ``Scalar`` or has the same number of elements as ``src2.channels()``:

        .. math::

            \texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0


In case of floating-point arrays, their machine-specific bit representations (usually IEEE754-compliant) are used for the operation. In case of multi-channel arrays, each channel is processed independently. In the 2nd and 3rd cases above, the scalar is first converted to the array type.
    

calcCovarMatrix
---------------
Calculates the covariance matrix of a set of vectors.

.. ocv:function:: void calcCovarMatrix( const Mat* samples, int nsamples, Mat& covar, Mat& mean, int flags, int ctype=CV_64F)

.. ocv:function:: void calcCovarMatrix( InputArray samples, OutputArray covar, OutputArray mean, int flags, int ctype=CV_64F)

.. ocv:pyfunction:: cv2.calcCovarMatrix(samples, flags[, covar[, mean[, ctype]]]) -> covar, mean

.. ocv:cfunction:: void cvCalcCovarMatrix( const CvArr** vects, int count, CvArr* covMat, CvArr* avg, int flags)
.. ocv:pyoldfunction:: cv.CalcCovarMatrix(vects, covMat, avg, flags)-> None

    :param samples: Samples stored either as separate matrices or as rows/columns of a single matrix.

    :param nsamples: Number of samples when they are stored separately.

    :param covar: Output covariance matrix of the type ``ctype``  and square size.

    :param mean: Input or output (depending on the flags) array as the average value of the input vectors.

    :param flags: Operation flags as a combination of the following values:

            * **CV_COVAR_SCRAMBLED** The output covariance matrix is calculated as:

                .. math::

                      \texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]^T  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...],
                      
                The covariance matrix will be  ``nsamples x nsamples``. Such an unusual covariance matrix is used for fast PCA of a set of very large vectors (see, for example, the EigenFaces technique for face recognition). Eigenvalues of this "scrambled" matrix match the eigenvalues of the true covariance matrix. The "true" eigenvectors can be easily calculated from the eigenvectors of the "scrambled" covariance matrix.

            * **CV_COVAR_NORMAL** The output covariance matrix is calculated as:

                .. math::

                      \texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...]^T,
                      
                ``covar``  will be a square matrix of the same size as the total number of elements in each input vector. One and only one of  ``CV_COVAR_SCRAMBLED``  and ``CV_COVAR_NORMAL``  must be specified.

            * **CV_COVAR_USE_AVG** If the flag is specified, the function does not calculate  ``mean``  from the input vectors but, instead, uses the passed  ``mean``  vector. This is useful if  ``mean``  has been pre-computed or known in advance, or if the covariance matrix is calculated by parts. In this case, ``mean``  is not a mean vector of the input sub-set of vectors but rather the mean vector of the whole set.

            * **CV_COVAR_SCALE** If the flag is specified, the covariance matrix is scaled. In the "normal" mode,  ``scale``  is  ``1./nsamples`` . In the "scrambled" mode,  ``scale``  is the reciprocal of the total number of elements in each input vector. By default (if the flag is not specified), the covariance matrix is not scaled (  ``scale=1`` ).

            * **CV_COVAR_ROWS** [Only useful in the second variant of the function] If the flag is specified, all the input vectors are stored as rows of the  ``samples``  matrix.  ``mean``  should be a single-row vector in this case.

            * **CV_COVAR_COLS** [Only useful in the second variant of the function] If the flag is specified, all the input vectors are stored as columns of the  ``samples``  matrix.  ``mean``  should be a single-column vector in this case.

The functions ``calcCovarMatrix`` calculate the covariance matrix and, optionally, the mean vector of the set of input vectors.

.. seealso::

    :ocv:class:`PCA`,
    :ocv:func:`mulTransposed`,
    :ocv:func:`Mahalanobis`



cartToPolar
-----------
Calculates the magnitude and angle of 2D vectors.

.. ocv:function:: void cartToPolar(InputArray x, InputArray y, OutputArray magnitude, OutputArray angle, bool angleInDegrees=false)

.. ocv:pyfunction:: cv2.cartToPolar(x, y[, magnitude[, angle[, angleInDegrees]]]) -> magnitude, angle

.. ocv:cfunction:: void cvCartToPolar( const CvArr* x, const CvArr* y, CvArr* magnitude, CvArr* angle=NULL, int angleInDegrees=0)
.. ocv:pyoldfunction:: cv.CartToPolar(x, y, magnitude, angle=None, angleInDegrees=0)-> None

    :param x: Array of x-coordinates. This must be a single-precision or double-precision floating-point array.

    :param y: Array of y-coordinates that must have the same size and same type as  ``x`` .
    
    :param magnitude: Destination array of magnitudes of the same size and type as  ``x`` .
    
    :param angle: Destination array of angles that has the same size and type as  ``x`` . The angles are measured in radians  (from 0 to 2*Pi) or in degrees (0 to 360 degrees).

    :param angleInDegrees: Flag indicating whether the angles are measured in radians, which is the default mode, or in degrees.

The function ``cartToPolar`` calculates either the magnitude, angle, or both for every 2D vector (x(I),y(I)):

.. math::

    \begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}

The angles are calculated with accuracy about 0.3 degrees. For the point (0,0), the angle is set to 0.



checkRange
----------
Checks every element of an input array for invalid values.

.. ocv:function:: bool checkRange(InputArray src, bool quiet=true, Point* pos=0, double minVal=-DBL_MAX, double maxVal=DBL_MAX)

.. ocv:pyfunction:: cv2.checkRange(a[, quiet[, minVal[, maxVal]]]) -> retval, pt

    :param src: Array to check.

    :param quiet: Flag indicating whether the functions quietly return false when the array elements are out of range or they throw an exception.

    :param pos: Optional output parameter, where the position of the first outlier is stored. In the second function  ``pos`` , when not NULL, must be a pointer to array of  ``src.dims``  elements.

    :param minVal: Inclusive lower boundary of valid values range.

    :param maxVal: Exclusive upper boundary of valid values range.

The functions ``checkRange`` check that every array element is neither NaN nor
infinite. When ``minVal < -DBL_MAX`` and ``maxVal < DBL_MAX`` , the functions also check that each value is between ``minVal`` and ``maxVal`` . In case of multi-channel arrays, each channel is processed independently.
If some values are out of range, position of the first outlier is stored in ``pos`` (when
``pos != NULL``). Then, the functions either return false (when ``quiet=true`` ) or throw an exception.



compare
-------
Performs the per-element comparison of two arrays or an array and scalar value.

.. ocv:function:: void compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop)

.. ocv:pyfunction:: cv2.compare(src1, src2, cmpop[, dst]) -> dst

.. ocv:cfunction:: void cvCmp(const CvArr* src1, const CvArr* src2, CvArr* dst, int cmpOp)

.. ocv:pyoldfunction:: cv.Cmp(src1, src2, dst, cmpOp)-> None

.. ocv:cfunction:: void cvCmpS(const CvArr* src1, double src2, CvArr* dst, int cmpOp)

.. ocv:pyoldfunction:: cv.CmpS(src1, src2, dst, cmpOp)-> None

    :param src1: First source array or a scalar (in the case of ``cvCmp``, ``cv.Cmp``, ``cvCmpS``, ``cv.CmpS`` it is always an array). When it is array, it must have a single channel.

    :param src2: Second source array or a scalar (in the case of ``cvCmp`` and ``cv.Cmp`` it is always an array; in the case of ``cvCmpS``, ``cv.CmpS`` it is always a scalar). When it is array, it must have a single channel.
    
    :param dst: Destination array that has the same size as the input array(s) and type= ``CV_8UC1`` .
    
    :param cmpop: Flag specifying the relation between the elements to be checked.

            * **CMP_EQ** ``src1`` equal to ``src2``.
            * **CMP_GT** ``src1`` greater than ``src2``.
            * **CMP_GE** ``src1`` greater than or equal to ``src2``.
            * **CMP_LT** ``src1`` less than ``src2``.   
            * **CMP_LE** ``src1`` less than or equal to ``src2``.             
            * **CMP_NE** ``src1`` not equal to ``src2``.
            
The function compares:


 *
   Elements of two arrays when ``src1`` and ``src2`` have the same size:

   .. math::

       \texttt{dst} (I) =  \texttt{src1} (I)  \,\texttt{cmpop}\, \texttt{src2} (I)

 *
   Elements of ``src1`` with a scalar ``src2` when ``src2`` is constructed from ``Scalar`` or has a single element:

   .. math::

       \texttt{dst} (I) =  \texttt{src1}(I) \,\texttt{cmpop}\,  \texttt{src2}

 *
   ``src1`` with elements of ``src2`` when ``src1`` is constructed from ``Scalar`` or has a single element:

   .. math::

       \texttt{dst} (I) =  \texttt{src1}  \,\texttt{cmpop}\, \texttt{src2} (I)


When the comparison result is true, the corresponding element of destination array is set to 255.    
The comparison operations can be replaced with the equivalent matrix expressions: ::

    Mat dst1 = src1 >= src2;
    Mat dst2 = src1 < 8;
    ...


.. seealso::

    :ocv:func:`checkRange`,
    :ocv:func:`min`,
    :ocv:func:`max`,
    :ocv:func:`threshold`,
    :ref:`MatrixExpressions`



completeSymm
------------
Copies the lower or the upper half of a square matrix to another half.

.. ocv:function:: void completeSymm(InputOutputArray mtx, bool lowerToUpper=false)

.. ocv:pyfunction:: cv2.completeSymm(mtx[, lowerToUpper]) -> None

    :param mtx: Input-output floating-point square matrix.

    :param lowerToUpper: Operation flag. If it is true, the lower half is copied to the upper half. Otherwise, the upper half is copied to the lower half.

The function ``completeSymm`` copies the lower half of a square matrix to its another half. The matrix diagonal remains unchanged:

 *
    :math:`\texttt{mtx}_{ij}=\texttt{mtx}_{ji}`     for
    :math:`i > j`     if ``lowerToUpper=false``
    
 *
    :math:`\texttt{mtx}_{ij}=\texttt{mtx}_{ji}`     for
    :math:`i < j`     if ``lowerToUpper=true``
    
.. seealso::

    :ocv:func:`flip`,
    :ocv:func:`transpose`



convertScaleAbs
---------------
Scales, computes absolute values, and converts the result to 8-bit.

.. ocv:function:: void convertScaleAbs(InputArray src, OutputArray dst, double alpha=1, double beta=0)

.. ocv:pyfunction:: cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst

.. ocv:cfunction:: void cvConvertScaleAbs(const CvArr* src, CvArr* dst, double scale=1, double shift=0)
.. ocv:pyoldfunction:: cv.ConvertScaleAbs(src, dst, scale=1.0, shift=0.0)-> None

    :param src: Source array.

    :param dst: Destination array.

    :param alpha: Optional scale factor.

    :param beta: Optional delta added to the scaled values.

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


.. seealso::

    :ocv:func:`Mat::convertTo`,
    :ocv:func:`abs`



countNonZero
------------
Counts non-zero array elements.

.. ocv:function:: int countNonZero( InputArray mtx )

.. ocv:pyfunction:: cv2.countNonZero(src) -> retval

.. ocv:cfunction:: int cvCountNonZero(const CvArr* arr)
.. ocv:pyoldfunction:: cv.CountNonZero(arr)-> int

    :param mtx: Single-channel array.

The function returns the number of non-zero elements in ``mtx`` :

.. math::

    \sum _{I: \; \texttt{mtx} (I) \ne0 } 1

.. seealso::

    :ocv:func:`mean`,
    :ocv:func:`meanStdDev`,
    :ocv:func:`norm`,
    :ocv:func:`minMaxLoc`,
    :ocv:func:`calcCovarMatrix`



cvarrToMat
----------
Converts ``CvMat``, ``IplImage`` , or ``CvMatND`` to ``Mat``.

.. ocv:function:: Mat cvarrToMat(const CvArr* src, bool copyData=false, bool allowND=true, int coiMode=0)

    :param src: Source ``CvMat``, ``IplImage`` , or  ``CvMatND`` .
    
    :param copyData: When it is false (default value), no data is copied and only the new header is created. In this case, the original array should not be deallocated while the new matrix header is used. If the parameter is true, all the data is copied and you may deallocate the original array right after the conversion.

    :param allowND: When it is true (default value), ``CvMatND`` is converted to 2-dimensional ``Mat``, if it is possible (see the discussion below). If it is not possible, or when the parameter is false, the function will report an error.

    :param coiMode: Parameter specifying how the IplImage COI (when set) is handled.

        *  If  ``coiMode=0`` and COI is set, the function reports an error.

        *  If  ``coiMode=1`` , the function never reports an error. Instead, it returns the header to the whole original image and you will have to check and process COI manually. See  :ocv:func:`extractImageCOI` .

The function ``cvarrToMat`` converts ``CvMat``, ``IplImage`` , or ``CvMatND`` header to
:ocv:class:`Mat` header, and optionally duplicates the underlying data. The constructed header is returned by the function.

When ``copyData=false`` , the conversion is done really fast (in O(1) time) and the newly created matrix header will have ``refcount=0`` , which means that no reference counting is done for the matrix data. In this case, you have to preserve the data until the new header is destructed. Otherwise, when ``copyData=true`` , the new buffer is allocated and managed as if you created a new matrix from scratch and copied the data there. That is, ``cvarrToMat(src, true)`` is equivalent to ``cvarrToMat(src, false).clone()`` (assuming that COI is not set). The function provides a uniform way of supporting
``CvArr`` paradigm in the code that is migrated to use new-style data structures internally. The reverse transformation, from
``Mat`` to
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
    printf("%g\n", cvNorm(&C1, 0, CV_L2));

Normally, the function is used to convert an old-style 2D array (
``CvMat`` or
``IplImage`` ) to ``Mat`` . However, the function can also take
``CvMatND`` as an input and create
:ocv:func:`Mat` for it, if it is possible. And, for ``CvMatND A`` , it is possible if and only if ``A.dim[i].size*A.dim.step[i] == A.dim.step[i-1]`` for all or for all but one ``i, 0 < i < A.dims`` . That is, the matrix data should be continuous or it should be representable as a sequence of continuous matrices. By using this function in this way, you can process
``CvMatND`` using an arbitrary element-wise function.

The last parameter, ``coiMode`` , specifies how to deal with an image with COI set. By default, it is 0 and the function reports an error when an image with COI comes in. And ``coiMode=1`` means that no error is signalled. You have to check COI presence and handle it manually. The modern structures, such as
:ocv:func:`Mat` and
:ocv:func:`MatND` do not support COI natively. To process an individual channel of a new-style array, you need either to organize a loop over the array (for example, using matrix iterators) where the channel of interest will be processed, or extract the COI using
:ocv:func:`mixChannels` (for new-style arrays) or
:ocv:func:`extractImageCOI` (for old-style arrays), process this individual channel, and insert it back to the destination array if needed (using
:ocv:func:`mixChannel` or
:ocv:func:`insertImageCOI` , respectively).

.. seealso::

    :c:func:`cvGetImage`,
    :c:func:`cvGetMat`,
    :c:func:`cvGetMatND`,
    :ocv:func:`extractImageCOI`,
    :ocv:func:`insertImageCOI`,
    :ocv:func:`mixChannels` 

dct
-------
Performs a forward or inverse discrete Cosine transform of 1D or 2D array.

.. ocv:function:: void dct(InputArray src, OutputArray dst, int flags=0)

.. ocv:pyfunction:: cv2.dct(src[, dst[, flags]]) -> dst

.. ocv:cfunction:: void cvDCT(const CvArr* src, CvArr* dst, int flags)
.. ocv:pyoldfunction:: cv.DCT(src, dst, flags)-> None

    :param src: Source floating-point array.

    :param dst: Destination array of the same size and type as  ``src`` .
    
    :param flags: Transformation flags as a combination of the following values:

            * **DCT_INVERSE** performs an inverse 1D or 2D transform instead of the default forward transform.

            * **DCT_ROWS** performs a forward or inverse transform of every individual row of the input matrix. This flag enables you to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself) to perform 3D and higher-dimensional transforms and so forth.

The function ``dct`` performs a forward or inverse discrete Cosine transform (DCT) of a 1D or 2D floating-point array:

*
    Forward Cosine transform of a 1D vector of ``N`` elements:

    .. math::

        Y = C^{(N)}  \cdot X

    where

    .. math::

        C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )

    and
    
    :math:`\alpha_0=1`, :math:`\alpha_j=2` for *j > 0*.

*
    Inverse Cosine transform of a 1D vector of ``N`` elements:

    .. math::

        X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y

    (since
    :math:`C^{(N)}` is an orthogonal matrix,
    :math:`C^{(N)} \cdot \left(C^{(N)}\right)^T = I` )

*
    Forward 2D Cosine transform of ``M x N`` matrix:

    .. math::

        Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T

*
    Inverse 2D Cosine transform of ``M x N`` matrix:

    .. math::

        X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}


The function chooses the mode of operation by looking at the flags and size of the input array:

*
    If ``(flags & DCT_INVERSE) == 0`` , the function does a forward 1D or 2D transform. Otherwise, it is an inverse 1D or 2D transform.

*
    If ``(flags & DCT_ROWS) != 0`` , the function performs a 1D transform of each row.

*
    If the array is a single column or a single row, the function performs a 1D transform.

*
    If none of the above is true, the function performs a 2D transform.

.. note::
 
    Currently ``dct`` supports even-size arrays (2, 4, 6 ...). For data analysis and approximation, you can pad the array when necessary.

    Also, the function performance depends very much, and not monotonically, on the array size (see
    :ocv:func:`getOptimalDFTSize` ). In the current implementation DCT of a vector of size ``N`` is computed via DFT of a vector of size ``N/2`` . Thus, the optimal DCT size ``N1 >= N`` can be computed as: ::

        size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }
        N1 = getOptimalDCTSize(N);

.. seealso:: :ocv:func:`dft` , :ocv:func:`getOptimalDFTSize` , :ocv:func:`idct`



dft
---
Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.

.. ocv:function:: void dft(InputArray src, OutputArray dst, int flags=0, int nonzeroRows=0)

.. ocv:pyfunction:: cv2.dft(src[, dst[, flags[, nonzeroRows]]]) -> dst

.. ocv:cfunction:: void cvDFT(const CvArr* src, CvArr* dst, int flags, int nonzeroRows=0)
.. ocv:pyoldfunction:: cv.DFT(src, dst, flags, nonzeroRows=0)-> None

    :param src: Source array that could be real or complex.

    :param dst: Destination array whose size and type depends on the  ``flags`` .
    
    :param flags: Transformation flags representing a combination of the following values:

            * **DFT_INVERSE** performs an inverse 1D or 2D transform instead of the default forward transform.

            * **DFT_SCALE** scales the result: divide it by the number of array elements. Normally, it is combined with  ``DFT_INVERSE`` .             
            * **DFT_ROWS** performs a forward or inverse transform of every individual row of the input matrix. This flag enables you to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself) to perform 3D and higher-dimensional transforms and so forth.

            * **DFT_COMPLEX_OUTPUT** performs a forward transformation of 1D or 2D real array. The result, though being a complex array, has complex-conjugate symmetry (*CCS*, see the function description below for details). Such an array can be packed into a real array of the same size as input, which is the fastest option and which is what the function does by default. However, you may wish to get a full complex array (for simpler spectrum analysis, and so on). Pass the flag to enable the function to produce a full-size complex output array.

            * **DFT_REAL_OUTPUT** performs an inverse transformation of a 1D or 2D complex array. The result is normally a complex array of the same size. However, if the source array has conjugate-complex symmetry (for example, it is a result of forward transformation with  ``DFT_COMPLEX_OUTPUT``  flag), the output is a real array. While the function itself does not check whether the input is symmetrical or not, you can pass the flag and then the function will assume the symmetry and produce the real output array. Note that when the input is packed into a real array and inverse transformation is executed, the function treats the input as a packed complex-conjugate symmetrical array. So, the output will also be a real array.

    :param nonzeroRows: When the parameter is not zero, the function assumes that only the first  ``nonzeroRows``  rows of the input array ( ``DFT_INVERSE``  is not set) or only the first  ``nonzeroRows``  of the output array ( ``DFT_INVERSE``  is set) contain non-zeros. Thus, the function can handle the rest of the rows more efficiently and save some time. This technique is very useful for computing array cross-correlation or convolution using DFT.


The function performs one of the following:

*
    Forward the Fourier transform of a 1D vector of ``N`` elements:

    .. math::

        Y = F^{(N)}  \cdot X,

    where
    :math:`F^{(N)}_{jk}=\exp(-2\pi i j k/N)` and
    :math:`i=\sqrt{-1}`
    
*
    Inverse the Fourier transform of a 1D vector of ``N`` elements:

    .. math::

        \begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}

    where
    :math:`F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T`

*    
    Forward the 2D Fourier transform of a ``M x N`` matrix:

    .. math::

        Y = F^{(M)}  \cdot X  \cdot F^{(N)}

*
    Inverse the 2D Fourier transform of a ``M x N`` matrix:

    .. math::

        \begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}


In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input spectrum of the inverse Fourier transform can be represented in a packed format called *CCS* (complex-conjugate-symmetrical). It was borrowed from IPL (Intel* Image Processing Library). Here is how 2D *CCS* spectrum looks:

.. math::

    \begin{bmatrix} Re Y_{0,0} & Re Y_{0,1} & Im Y_{0,1} & Re Y_{0,2} & Im Y_{0,2} &  \cdots & Re Y_{0,N/2-1} & Im Y_{0,N/2-1} & Re Y_{0,N/2}  \\ Re Y_{1,0} & Re Y_{1,1} & Im Y_{1,1} & Re Y_{1,2} & Im Y_{1,2} &  \cdots & Re Y_{1,N/2-1} & Im Y_{1,N/2-1} & Re Y_{1,N/2}  \\ Im Y_{1,0} & Re Y_{2,1} & Im Y_{2,1} & Re Y_{2,2} & Im Y_{2,2} &  \cdots & Re Y_{2,N/2-1} & Im Y_{2,N/2-1} & Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &  Re Y_{M-3,1}  & Im Y_{M-3,1} &  \hdotsfor{3} & Re Y_{M-3,N/2-1} & Im Y_{M-3,N/2-1}& Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &  Re Y_{M-2,1}  & Im Y_{M-2,1} &  \hdotsfor{3} & Re Y_{M-2,N/2-1} & Im Y_{M-2,N/2-1}& Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &  Re Y_{M-1,1} &  Im Y_{M-1,1} &  \hdotsfor{3} & Re Y_{M-1,N/2-1} & Im Y_{M-1,N/2-1}& Re Y_{M/2,N/2} \end{bmatrix}

In case of 1D transform of a real vector, the output looks like the first row of the matrix above.

So, the function chooses an operation mode depending on the flags and size of the input array:

 * If ``DFT_ROWS`` is set or the input array has a single row or single column, the function performs a 1D forward or inverse transform of each row of a matrix when ``DFT_ROWS`` is set. Otherwise, it performs a 2D transform.

 * If the input array is real and ``DFT_INVERSE`` is not set, the function performs a forward 1D or 2D transform:

    * When ``DFT_COMPLEX_OUTPUT`` is set, the output is a complex matrix of the same size as input.

    * When ``DFT_COMPLEX_OUTPUT`` is not set, the output is a real matrix of the same size as input. In case of 2D transform, it uses the packed format as shown above. In case of a single 1D transform, it looks like the first row of the matrix above. In case of multiple 1D transforms (when using the ``DCT_ROWS``         flag), each row of the output matrix looks like the first row of the matrix above.

 * If the input array is complex and either ``DFT_INVERSE``     or ``DFT_REAL_OUTPUT``     are not set, the output is a complex array of the same size as input. The function performs a forward or inverse 1D or 2D transform of the whole input array or each row of the input array independently, depending on the flags ``DFT_INVERSE`` and ``DFT_ROWS``.

 * When ``DFT_INVERSE`` is set and the input array is real, or it is complex but ``DFT_REAL_OUTPUT``     is set, the output is a real array of the same size as input. The function performs a 1D or 2D inverse transformation of the whole input array or each individual row, depending on the flags ``DFT_INVERSE`` and ``DFT_ROWS``.

If ``DFT_SCALE`` is set, the scaling is done after the transformation.

Unlike :ocv:func:`dct` , the function supports arrays of arbitrary size. But only those arrays are processed efficiently, whose sizes can be factorized in a product of small prime numbers (2, 3, and 5 in the current implementation). Such an efficient DFT size can be computed using the :ocv:func:`getOptimalDFTSize` method.

The sample below illustrates how to compute a DFT-based convolution of two 2D real arrays: ::

    void convolveDFT(InputArray A, InputArray B, OutputArray C)
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
        // you need only the first C.rows of them, and thus you
        // pass nonzeroRows == C.rows
        dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

        // now copy the result back to C.
        tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);

        // all the temporary buffers will be deallocated automatically
    }


To optimize this sample, consider the following approaches:

*
    Since ``nonzeroRows != 0`` is passed to the forward transform calls and since  ``A`` and ``B`` are copied to the top-left corners of ``tempA`` and ``tempB``, respectively, it is not necessary to clear the whole ``tempA`` and ``tempB``. It is only necessary to clear the ``tempA.cols - A.cols`` ( ``tempB.cols - B.cols``) rightmost columns of the matrices.

*
   This DFT-based convolution does not have to be applied to the whole big arrays, especially if ``B``     is significantly smaller than ``A`` or vice versa. Instead, you can compute convolution by parts. To do this, you need to split the destination array ``C``     into multiple tiles. For each tile, estimate which parts of ``A``     and ``B``     are required to compute convolution in this tile. If the tiles in ``C``     are too small, the speed will decrease a lot because of repeated work. In the ultimate case, when each tile in ``C``     is a single pixel, the algorithm becomes equivalent to the naive convolution algorithm. If the tiles are too big, the temporary arrays ``tempA``     and ``tempB``     become too big and there is also a slowdown because of bad cache locality. So, there is an optimal tile size somewhere in the middle.

*
    If different tiles in ``C``     can be computed in parallel and, thus, the convolution is done by parts, the loop can be threaded.

All of the above improvements have been implemented in :ocv:func:`matchTemplate` and :ocv:func:`filter2D` . Therefore, by using them, you can get the performance even better than with the above theoretically optimal implementation. Though, those two functions actually compute cross-correlation, not convolution, so you need to "flip" the second convolution operand ``B`` vertically and horizontally using :ocv:func:`flip` .

.. seealso:: :ocv:func:`dct` , :ocv:func:`getOptimalDFTSize` , :ocv:func:`mulSpectrums`, :ocv:func:`filter2D` , :ocv:func:`matchTemplate` , :ocv:func:`flip` , :ocv:func:`cartToPolar` , :ocv:func:`magnitude` , :ocv:func:`phase`



divide
----------
Performs per-element division of two arrays or a scalar by an array.

.. ocv:function:: void divide(InputArray src1, InputArray src2, OutputArray dst, double scale=1, int dtype=-1)

.. ocv:function:: void divide(double scale, InputArray src2, OutputArray dst, int dtype=-1)

.. ocv:pyfunction:: cv2.divide(src1, src2[, dst[, scale[, dtype]]]) -> dst
.. ocv:pyfunction:: cv2.divide(scale, src2[, dst[, dtype]]) -> dst

.. ocv:cfunction:: void cvDiv(const CvArr* src1, const CvArr* src2, CvArr* dst, double scale=1)
.. ocv:pyoldfunction:: cv.Div(src1, src2, dst, scale)-> None

    :param src1: First source array.

    :param src2: Second source array of the same size and type as  ``src1`` .
    
    :param scale: Scalar factor.

    :param dst: Destination array of the same size and type as  ``src2`` .
    
    :param dtype: Optional depth of the destination array. If it is ``-1``, ``dst`` will have depth ``src2.depth()``. In case of an array-by-array division, you can only pass ``-1`` when ``src1.depth()==src2.depth()``.
    
The functions ``divide`` divide one array by another:

.. math::

    \texttt{dst(I) = saturate(src1(I)*scale/src2(I))}

or a scalar by an array when there is no ``src1`` :

.. math::

    \texttt{dst(I) = saturate(scale/src2(I))}

When ``src2(I)`` is zero, ``dst(I)`` will also be zero. Different channels of multi-channel arrays are processed independently.

.. seealso::

    :ocv:func:`multiply`,
    :ocv:func:`add`,
    :ocv:func:`subtract`,
    :ref:`MatrixExpressions`



determinant
-----------
Returns the determinant of a square floating-point matrix.

.. ocv:function:: double determinant(InputArray mtx)

.. ocv:pyfunction:: cv2.determinant(mtx) -> retval

.. ocv:cfunction:: double cvDet(const CvArr* mtx)
.. ocv:pyoldfunction:: cv.Det(mtx)-> double

    :param mtx: Input matrix that must have  ``CV_32FC1``  or  ``CV_64FC1``  type and square size.

The function ``determinant`` computes and returns the determinant of the specified matrix. For small matrices ( ``mtx.cols=mtx.rows<=3`` ),
the direct method is used. For larger matrices, the function uses LU factorization with partial pivoting.

For symmetric positively-determined matrices, it is also possible to use :ocv:func:`eigen` decomposition to compute the determinant.

.. seealso::

    :ocv:func:`trace`,
    :ocv:func:`invert`,
    :ocv:func:`solve`,
    :ocv:func:`eigen`,
    :ref:`MatrixExpressions`



eigen
-----

.. ocv:function:: bool eigen(InputArray src, OutputArray eigenvalues, int lowindex=-1, int highindex=-1)

.. ocv:function:: bool eigen(InputArray src, OutputArray eigenvalues, OutputArray eigenvectors, int lowindex=-1,int highindex=-1)

.. ocv:cfunction:: void cvEigenVV( CvArr* src, CvArr* eigenvectors, CvArr* eigenvalues, double eps=0, int lowindex=-1, int highindex=-1)

.. ocv:pyoldfunction:: cv.EigenVV(src, eigenvectors, eigenvalues, eps, lowindex=-1, highindex=-1)-> None

    Computes eigenvalues and eigenvectors of a symmetric matrix.

.. ocv:pyfunction:: cv2.eigen(src, computeEigenvectors[, eigenvalues[, eigenvectors[, lowindex[, highindex]]]]) -> retval, eigenvalues, eigenvectors

    :param src: Input matrix that must have  ``CV_32FC1``  or  ``CV_64FC1``  type, square size and be symmetrical (``src`` :sup:`T` == ``src``).
    
    :param eigenvalues: Output vector of eigenvalues of the same type as  ``src`` . The eigenvalues are stored in the descending order.

    :param eigenvectors: Output matrix of eigenvectors. It has the same size and type as  ``src`` . The eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues.

    :param lowindex: Optional index of largest eigenvalue/-vector to calculate. The parameter is ignored in the current implementation.

    :param highindex: Optional index of smallest eigenvalue/-vector to calculate. The parameter is ignored in the current implementation.

The functions ``eigen`` compute just eigenvalues, or eigenvalues and eigenvectors of the symmetric matrix ``src`` : ::

    src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()

.. note:: in the new and the old interfaces different ordering of eigenvalues and eigenvectors parameters is used.

.. seealso:: :ocv:func:`completeSymm` , :ocv:class:`PCA`



exp
---
Calculates the exponent of every array element.

.. ocv:function:: void exp(InputArray src, OutputArray dst)

.. ocv:pyfunction:: cv2.exp(src[, dst]) -> dst

.. ocv:cfunction:: void cvExp(const CvArr* src, CvArr* dst)
.. ocv:pyoldfunction:: cv.Exp(src, dst)-> None

    :param src: Source array.

    :param dst: Destination array of the same size and type as ``src``.

The function ``exp`` calculates the exponent of every element of the input array:

.. math::

    \texttt{dst} [I] = e^{ src(I) }

The maximum relative error is about ``7e-6`` for single-precision input and less than ``1e-10`` for double-precision input. Currently, the function converts denormalized values to zeros on output. Special values (NaN, Inf) are not handled.

.. seealso::  :ocv:func:`log` , :ocv:func:`cartToPolar` , :ocv:func:`polarToCart` , :ocv:func:`phase` , :ocv:func:`pow` , :ocv:func:`sqrt` , :ocv:func:`magnitude`



extractImageCOI
---------------
Extracts the selected image channel.

.. ocv:function:: void extractImageCOI(const CvArr* src, OutputArray dst, int coi=-1)

    :param src: Source array. It should be a pointer to  ``CvMat``  or  ``IplImage`` .
    
    :param dst: Destination array with a single channel and the same size and depth as  ``src`` .
    
    :param coi: If the parameter is  ``>=0`` , it specifies the channel to extract. If it is  ``<0`` and ``src``  is a pointer to  ``IplImage``  with a  valid COI set, the selected COI is extracted.

The function ``extractImageCOI`` is used to extract an image COI from an old-style array and put the result to the new-style C++ matrix. As usual, the destination matrix is reallocated using ``Mat::create`` if needed.

To extract a channel from a new-style matrix, use
:ocv:func:`mixChannels` or
:ocv:func:`split` .

.. seealso::  :ocv:func:`mixChannels` , :ocv:func:`split` , :ocv:func:`merge` , :ocv:func:`cvarrToMat` , :ocv:cfunc:`cvSetImageCOI` , :ocv:cfunc:`cvGetImageCOI`



flip
--------
Flips a 2D array around vertical, horizontal, or both axes.

.. ocv:function:: void flip(InputArray src, OutputArray dst, int flipCode)

.. ocv:pyfunction:: cv2.flip(src, flipCode[, dst]) -> dst

.. ocv:cfunction:: void cvFlip(const CvArr* src, CvArr* dst=NULL, int flipMode=0)
.. ocv:pyoldfunction:: cv.Flip(src, dst=None, flipMode=0)-> None

    :param src: Source array.

    :param dst: Destination array of the same size and type as  ``src`` .
    
    :param flipCode: Flag to specify how to flip the array. 0 means flipping around the x-axis. Positive value (for example, 1) means flipping around y-axis. Negative value (for example, -1) means flipping around both axes. See the discussion below for the formulas.

The function ``flip`` flips the array in one of three different ways (row and column indices are 0-based):

.. math::

    \texttt{dst} _{ij} =
    \left\{
    \begin{array}{l l}
    \texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\
    \texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\
    \texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\
    \end{array}
    \right.

The example scenarios of using the function are the following:

 *
    Vertical flipping of the image (``flipCode == 0``) to switch between top-left and bottom-left image origin. This is a typical operation in video processing on Microsoft Windows* OS.

 *
    Horizontal flipping of the image with the subsequent horizontal shift and absolute difference calculation to check for a vertical-axis symmetry (``flipCode > 0``).

 *
    Simultaneous horizontal and vertical flipping of the image with the subsequent shift and absolute difference calculation to check for a central symmetry (``flipCode < 0``).

 *
    Reversing the order of point arrays (``flipCode > 0`` or ``flipCode == 0``).

.. seealso:: :ocv:func:`transpose` , :ocv:func:`repeat` , :ocv:func:`completeSymm`



gemm
----
Performs generalized matrix multiplication.

.. ocv:function:: void gemm(InputArray src1, InputArray src2, double alpha, InputArray src3, double beta, OutputArray dst, int flags=0)

.. ocv:pyfunction:: cv2.gemm(src1, src2, alpha, src3, gamma[, dst[, flags]]) -> dst

.. ocv:cfunction:: void cvGEMM( const CvArr* src1, const CvArr* src2, double alpha, const CvArr* src3, double beta, CvArr* dst, int tABC=0)
.. ocv:pyoldfunction:: cv.GEMM(src1, src2, alphs, src3, beta, dst, tABC=0)-> None

    :param src1: First multiplied input matrix that should have  ``CV_32FC1`` , ``CV_64FC1`` , ``CV_32FC2`` , or  ``CV_64FC2``  type.

    :param src2: Second multiplied input matrix of the same type as  ``src1`` .
    
    :param alpha: Weight of the matrix product.

    :param src3: Third optional delta matrix added to the matrix product. It should have the same type as  ``src1``  and  ``src2`` .
    
    :param beta: Weight of  ``src3`` .
    
    :param dst: Destination matrix. It has the proper size and the same type as input matrices.

    :param flags: Operation flags:

            * **GEMM_1_T** transpose  ``src1``
            * **GEMM_2_T** transpose  ``src2``
            * **GEMM_3_T** transpose  ``src3``
            
The function performs generalized matrix multiplication similar to the ``gemm`` functions in BLAS level 3. For example, ``gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)`` corresponds to

.. math::

    \texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T

The function can be replaced with a matrix expression. For example, the above call can be replaced with: ::

    dst = alpha*src1.t()*src2 + beta*src3.t();


.. seealso::  :ocv:func:`mulTransposed` , :ocv:func:`transform` , :ref:`MatrixExpressions`



getConvertElem
--------------
Returns a conversion function for a single pixel.

.. ocv:function:: ConvertData getConvertElem(int fromType, int toType)

.. ocv:function:: ConvertScaleData getConvertScaleElem(int fromType, int toType)

    :param fromType: Source pixel type.

    :param toType: Destination pixel type.

    :param from: Callback parameter: pointer to the input pixel.

    :param to: Callback parameter: pointer to the output pixel

    :param cn: Callback parameter: the number of channels. It can be arbitrary, 1, 100, 100000, ...

    :param alpha: ``ConvertScaleData`` callback optional parameter: the scale factor.

    :param beta: ``ConvertScaleData`` callback optional parameter: the delta or offset.

The functions ``getConvertElem`` and ``getConvertScaleElem`` return pointers to the functions for converting individual pixels from one type to another. While the main function purpose is to convert single pixels (actually, for converting sparse matrices from one type to another), you can use them to convert the whole row of a dense matrix or the whole matrix at once, by setting ``cn = matrix.cols*matrix.rows*matrix.channels()`` if the matrix data is continuous.

``ConvertData`` and ``ConvertScaleData`` are defined as: ::

    typedef void (*ConvertData)(const void* from, void* to, int cn)
    typedef void (*ConvertScaleData)(const void* from, void* to,
                                     int cn, double alpha, double beta)

.. seealso:: :ocv:func:`Mat::convertTo` , :ocv:func:`SparseMat::convertTo`



getOptimalDFTSize
-----------------
Returns the optimal DFT size for a given vector size.

.. ocv:function:: int getOptimalDFTSize(int vecsize)

.. ocv:pyfunction:: cv2.getOptimalDFTSize(vecsize) -> retval

.. ocv:cfunction:: int cvGetOptimalDFTSize(int size0)
.. ocv:pyoldfunction:: cv.GetOptimalDFTSize(size0)-> int

    :param vecsize: Vector size.

DFT performance is not a monotonic function of a vector size. Therefore, when you compute convolution of two arrays or perform the spectral analysis of an array, it usually makes sense to pad the input data with zeros to get a bit larger array that can be transformed much faster than the original one.
Arrays whose size is a power-of-two (2, 4, 8, 16, 32, ...) are the fastest to process. Though, the arrays whose size is a product of 2's, 3's, and 5's (for example, 300 = 5*5*3*2*2) are also processed quite efficiently.

The function ``getOptimalDFTSize`` returns the minimum number ``N`` that is greater than or equal to ``vecsize``  so that the DFT of a vector of size ``N`` can be computed efficiently. In the current implementation ``N`` = 2 :sup:`p` * 3 :sup:`q` * 5 :sup:`r` for some integer ``p``, ``q``, ``r``.

The function returns a negative number if ``vecsize`` is too large (very close to ``INT_MAX`` ).

While the function cannot be used directly to estimate the optimal vector size for DCT transform (since the current DCT implementation supports only even-size vectors), it can be easily computed as ``getOptimalDFTSize((vecsize+1)/2)*2``.

.. seealso:: :ocv:func:`dft` , :ocv:func:`dct` , :ocv:func:`idft` , :ocv:func:`idct` , :ocv:func:`mulSpectrums`



idct
----
Computes the inverse Discrete Cosine Transform of a 1D or 2D array.

.. ocv:function:: void idct(InputArray src, OutputArray dst, int flags=0)

.. ocv:pyfunction:: cv2.idct(src[, dst[, flags]]) -> dst

    :param src: Source floating-point single-channel array.

    :param dst: Destination array of the same size and type as  ``src`` .
    
    :param flags: Operation flags.
    
``idct(src, dst, flags)`` is equivalent to ``dct(src, dst, flags | DCT_INVERSE)``.

.. seealso::

    :ocv:func:`dct`,
    :ocv:func:`dft`,
    :ocv:func:`idft`,
    :ocv:func:`getOptimalDFTSize`



idft
----
Computes the inverse Discrete Fourier Transform of a 1D or 2D array.

.. ocv:function:: void idft(InputArray src, OutputArray dst, int flags=0, int outputRows=0)

.. ocv:pyfunction:: cv2.idft(src[, dst[, flags[, nonzeroRows]]]) -> dst

    :param src: Source floating-point real or complex array.

    :param dst: Destination array whose size and type depend on the  ``flags`` .
    
    :param flags: Operation flags. See  :ocv:func:`dft` .
    
    :param nonzeroRows: Number of  ``dst``  rows to compute. The rest of the rows have undefined content. See the convolution sample in  :ocv:func:`dft`  description.
    
``idft(src, dst, flags)`` is equivalent to ``dct(src, dst, flags | DFT_INVERSE)`` .

See :ocv:func:`dft` for details.

.. note:: None of ``dft`` and ``idft`` scales the result by default. So, you should pass ``DFT_SCALE`` to one of ``dft`` or ``idft`` explicitly to make these transforms mutually inverse.

.. seealso::

    :ocv:func:`dft`,
    :ocv:func:`dct`,
    :ocv:func:`idct`,
    :ocv:func:`mulSpectrums`,
    :ocv:func:`getOptimalDFTSize`



inRange
-------
Checks if array elements lie between the elements of two other arrays.

.. ocv:function:: void inRange(InputArray src, InputArray lowerb, InputArray upperb, OutputArray dst)

.. ocv:pyfunction:: cv2.inRange(src, lowerb, upperb[, dst]) -> dst

.. ocv:cfunction:: void cvInRange(const CvArr* src, const CvArr* lower, const CvArr* upper, CvArr* dst)
.. ocv:cfunction:: void cvInRangeS(const CvArr* src, CvScalar lower, CvScalar upper, CvArr* dst)
.. ocv:pyoldfunction:: cv.InRange(src, lower, upper, dst)-> None
.. ocv:pyoldfunction:: cv.InRangeS(src, lower, upper, dst)-> None

    :param src: First source array.

    :param lowerb: Inclusive lower boundary array or a scalar.
    
    :param upperb: Inclusive upper boundary array or a scalar.
    
    :param dst: Destination array of the same size as  ``src``  and  ``CV_8U``  type.

The function checks the range as follows:

 * For every element of a single-channel input array:

   .. math::

      \texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 <  \texttt{upperb} (I)_0

 * For two-channel arrays:

   .. math::

      \texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 <  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 <  \texttt{upperb} (I)_1

 * and so forth.

That is, ``dst`` (I) is set to 255 (all ``1`` -bits) if ``src`` (I) is within the specified 1D, 2D, 3D, ... box and 0 otherwise.

When the lower and/or upper bounary parameters are scalars, the indexes ``(I)`` at ``lowerb`` and ``upperb`` in the above formulas should be omitted.


invert
------
Finds the inverse or pseudo-inverse of a matrix.

.. ocv:function:: double invert(InputArray src, OutputArray dst, int method=DECOMP_LU)

.. ocv:pyfunction:: cv2.invert(src[, dst[, flags]]) -> retval, dst

.. ocv:cfunction:: double cvInvert(const CvArr* src, CvArr* dst, int method=CV_LU)
.. ocv:pyoldfunction:: cv.Invert(src, dst, method=CV_LU)-> double

    :param src: Source floating-point  ``M x N``  matrix.

    :param dst: Destination matrix of  ``N x M``  size and the same type as  ``src`` .
    
    :param flags: Inversion method :

            * **DECOMP_LU** Gaussian elimination with the optimal pivot element chosen.

            * **DECOMP_SVD** Singular value decomposition (SVD) method.

            * **DECOMP_CHOLESKY** Cholesky decomposion. The matrix must be symmetrical and positively defined.

The function ``invert`` inverts the matrix ``src`` and stores the result in ``dst`` .
When the matrix ``src`` is singular or non-square, the function computes the pseudo-inverse matrix (the ``dst`` matrix) so that ``norm(src*dst - I)`` is minimal, where I is an identity matrix.

In case of the ``DECOMP_LU`` method, the function returns the ``src`` determinant ( ``src`` must be square). If it is 0, the matrix is not inverted and ``dst`` is filled with zeros.

In case of the ``DECOMP_SVD`` method, the function returns the inverse condition number of ``src`` (the ratio of the smallest singular value to the largest singular value) and 0 if ``src`` is singular. The SVD method calculates a pseudo-inverse matrix if ``src`` is singular.

Similarly to ``DECOMP_LU`` , the method ``DECOMP_CHOLESKY`` works only with non-singular square matrices that should also be symmetrical and positively defined. In this case, the function stores the inverted matrix in ``dst`` and returns non-zero. Otherwise, it returns 0.

.. seealso::

    :ocv:func:`solve`,
    :ocv:class:`SVD`



log
---
Calculates the natural logarithm of every array element.

.. ocv:function:: void log(InputArray src, OutputArray dst)

.. ocv:pyfunction:: cv2.log(src[, dst]) -> dst

.. ocv:cfunction:: void cvLog(const CvArr* src, CvArr* dst)
.. ocv:pyoldfunction:: cv.Log(src, dst)-> None

    :param src: Source array.

    :param dst: Destination array of the same size and type as  ``src`` .
    
The function ``log`` calculates the natural logarithm of the absolute value of every element of the input array:

.. math::

    \texttt{dst} (I) =  \fork{\log |\texttt{src}(I)|}{if $\texttt{src}(I) \ne 0$ }{\texttt{C}}{otherwise}

where ``C`` is a large negative number (about -700 in the current implementation).
The maximum relative error is about ``7e-6`` for single-precision input and less than ``1e-10`` for double-precision input. Special values (NaN, Inf) are not handled.

.. seealso::

    :ocv:func:`exp`,
    :ocv:func:`cartToPolar`,
    :ocv:func:`polarToCart`,
    :ocv:func:`phase`,
    :ocv:func:`pow`,
    :ocv:func:`sqrt`,
    :ocv:func:`magnitude`



LUT
---
Performs a look-up table transform of an array.

.. ocv:function:: void LUT(InputArray src, InputArray lut, OutputArray dst)

.. ocv:pyfunction:: cv2.LUT(src, lut[, dst[, interpolation]]) -> dst

.. ocv:cfunction:: void cvLUT(const CvArr* src, CvArr* dst, const CvArr* lut)
.. ocv:pyoldfunction:: cv.LUT(src, dst, lut)-> None

    :param src: Source array of 8-bit elements.

    :param lut: Look-up table of 256 elements. In case of multi-channel source array, the table should either have a single channel (in this case the same table is used for all channels) or the same number of channels as in the source array.

    :param dst: Destination array of the same size and the same number of channels as  ``src`` , and the same depth as  ``lut`` .
    
The function ``LUT`` fills the destination array with values from the look-up table. Indices of the entries are taken from the source array. That is, the function processes each element of ``src`` as follows:

.. math::

    \texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}

where

.. math::

    d =  \fork{0}{if \texttt{src} has depth \texttt{CV\_8U}}{128}{if \texttt{src} has depth \texttt{CV\_8S}}

.. seealso::

    :ocv:func:`convertScaleAbs`,
    :ocv:func:`Mat::convertTo`



magnitude
---------
Calculates the magnitude of 2D vectors.

.. ocv:function:: void magnitude(InputArray x, InputArray y, OutputArray magnitude)

.. ocv:pyfunction:: cv2.magnitude(x, y[, magnitude]) -> magnitude

    :param x: Floating-point array of x-coordinates of the vectors.

    :param y: Floating-point array of y-coordinates of the vectors. It must have the same size as  ``x`` .
    
    :param dst: Destination array of the same size and type as  ``x`` .
    
The function ``magnitude`` calculates the magnitude of 2D vectors formed from the corresponding elements of ``x`` and ``y`` arrays:

.. math::

    \texttt{dst} (I) =  \sqrt{\texttt{x}(I)^2 + \texttt{y}(I)^2}

.. seealso::

    :ocv:func:`cartToPolar`,
    :ocv:func:`polarToCart`,
    :ocv:func:`phase`,
    :ocv:func:`sqrt`



Mahalanobis
-----------
Calculates the Mahalanobis distance between two vectors.

.. ocv:function:: double Mahalanobis(InputArray vec1, InputArray vec2, InputArray icovar)

.. ocv:pyfunction:: cv2.Mahalanobis(v1, v2, icovar) -> retval

.. ocv:cfunction:: double cvMahalanobis( const CvArr* vec1, const CvArr* vec2, CvArr* icovar)

.. ocv:pyoldfunction:: cv.Mahalanobis(vec1, vec2, icovar)-> None

    :param vec1: First 1D source vector.

    :param vec2: Second 1D source vector.

    :param icovar: Inverse covariance matrix.

The function ``Mahalanobis`` calculates and returns the weighted distance between two vectors:

.. math::

    d( \texttt{vec1} , \texttt{vec2} )= \sqrt{\sum_{i,j}{\texttt{icovar(i,j)}\cdot(\texttt{vec1}(I)-\texttt{vec2}(I))\cdot(\texttt{vec1(j)}-\texttt{vec2(j)})} }

The covariance matrix may be calculated using the
:ocv:func:`calcCovarMatrix` function and then inverted using the
:ocv:func:`invert` function (preferably using the ``DECOMP_SVD`` method, as the most accurate).



max
---
Calculates per-element maximum of two arrays or an array and a scalar.

.. ocv:function:: MatExpr max(const Mat& src1, const Mat& src2)

.. ocv:function:: MatExpr max(const Mat& src1, double value)

.. ocv:function:: MatExpr max(double value, const Mat& src1)

.. ocv:function:: void max(InputArray src1, InputArray src2, OutputArray dst)

.. ocv:function:: void max(const Mat& src1, const Mat& src2, Mat& dst)

.. ocv:function:: void max(const Mat& src1, double value, Mat& dst)

.. ocv:pyfunction:: cv2.max(src1, src2[, dst]) -> dst

.. ocv:cfunction:: void cvMax(const CvArr* src1, const CvArr* src2, CvArr* dst)
.. ocv:cfunction:: void cvMaxS(const CvArr* src, double value, CvArr* dst)
.. ocv:pyoldfunction:: cv.Max(src1, src2, dst)-> None
.. ocv:pyoldfunction:: cv.MaxS(src, value, dst)-> None

    :param src1: First source array.

    :param src2: Second source array of the same size and type as  ``src1`` .
    
    :param value: Real scalar value.

    :param dst: Destination array of the same size and type as  ``src1`` .
    
The functions ``max`` compute the per-element maximum of two arrays:

.. math::

    \texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))

or array and a scalar:

.. math::

    \texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )

In the second variant, when the source array is multi-channel, each channel is compared with ``value`` independently.

The first 3 variants of the function listed above are actually a part of
:ref:`MatrixExpressions` . They return an expression object that can be further either transformed/ assigned to a matrix, or passed to a function, and so on.

.. seealso::

    :ocv:func:`min`,
    :ocv:func:`compare`,
    :ocv:func:`inRange`,
    :ocv:func:`minMaxLoc`,
    :ref:`MatrixExpressions`


mean
----
Calculates an average (mean) of array elements.

.. ocv:function:: Scalar mean(InputArray src, InputArray mask=noArray())

.. ocv:pyfunction:: cv2.mean(src[, mask]) -> retval

.. ocv:cfunction:: CvScalar cvAvg(const CvArr* src, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.Avg(src, mask=None)-> CvScalar

    :param src: Source array that should have from 1 to 4 channels so that the result can be stored in  :ocv:func:`Scalar` .

    :param mask: Optional operation mask.

The function ``mean`` computes the mean value ``M`` of array elements, independently for each channel, and return it:

.. math::

    \begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}

When all the mask elements are 0's, the functions return ``Scalar::all(0)`` .

.. seealso::

    :ocv:func:`countNonZero`,
    :ocv:func:`meanStdDev`,
    :ocv:func:`norm`,
    :ocv:func:`minMaxLoc`



meanStdDev
----------
Calculates a mean and standard deviation of array elements.

.. ocv:function:: void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev, InputArray mask=noArray())

.. ocv:pyfunction:: cv2.meanStdDev(src[, mean[, stddev[, mask]]]) -> mean, stddev

.. ocv:cfunction:: void cvAvgSdv(const CvArr* src, CvScalar* mean, CvScalar* stdDev, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.AvgSdv(src, mask=None)-> (mean, stdDev)

    :param src: Source array that should have from 1 to 4 channels so that the results can be stored in  :ocv:func:`Scalar` 's.

    :param mean: Output parameter: computed mean value.

    :param stddev: Output parameter: computed standard deviation.

    :param mask: Optional operation mask.

The function ``meanStdDev`` computes the mean and the standard deviation ``M`` of array elements independently for each channel and returns it via the output parameters:

.. math::

    \begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2} \end{array}

When all the mask elements are 0's, the functions return ``mean=stddev=Scalar::all(0)`` .

.. note:: The computed standard deviation is only the diagonal of the complete normalized covariance matrix. If the full matrix is needed, you can reshape the multi-channel array ``M x N`` to the single-channel array ``M*N x mtx.channels()`` (only possible when the matrix is continuous) and then pass the matrix to :ocv:func:`calcCovarMatrix` .

.. seealso::

    :ocv:func:`countNonZero`,
    :ocv:func:`mean`,
    :ocv:func:`norm`,
    :ocv:func:`minMaxLoc`,
    :ocv:func:`calcCovarMatrix`



merge
-----
Composes a multi-channel array from several single-channel arrays.

.. ocv:function:: void merge(const Mat* mv, size_t count, OutputArray dst)

.. ocv:function:: void merge(const vector<Mat>& mv, OutputArray dst)

.. ocv:pyfunction:: cv2.merge(mv[, dst]) -> dst

.. ocv:cfunction:: void cvMerge(const CvArr* src0, const CvArr* src1, const CvArr* src2, const CvArr* src3, CvArr* dst)
.. ocv:pyoldfunction:: cv.Merge(src0, src1, src2, src3, dst)-> None

    :param mv: Source array or vector of matrices to be merged. All the matrices in ``mv``  must have the same size and the same depth.

    :param count: Number of source matrices when  ``mv``  is a plain C array. It must be greater than zero.

    :param dst: Destination array of the same size and the same depth as  ``mv[0]`` . The number of channels will be the total number of channels in the matrix array.

The functions ``merge`` merge several arrays to make a single multi-channel array. That is, each element of the output array will be a concatenation of the elements of the input arrays, where elements of i-th input array are treated as ``mv[i].channels()``-element vectors.

The function
:ocv:func:`split` does the reverse operation. If you need to shuffle channels in some other advanced way, use
:ocv:func:`mixChannels` .

.. seealso::

    :ocv:func:`mixChannels`,
    :ocv:func:`split`,
    :ocv:func:`Mat::reshape`



min
---
Calculates per-element minimum of two arrays or array and a scalar.

.. ocv:function:: MatExpr min(const Mat& src1, const Mat& src2)

.. ocv:function:: MatExpr min(const Mat& src1, double value)

.. ocv:function:: MatExpr min(double value, const Mat& src1)

.. ocv:function:: void min(InputArray src1, InputArray src2, OutputArray dst)

.. ocv:function:: void min(const Mat& src1, const Mat& src2, Mat& dst)

.. ocv:function:: void min(const Mat& src1, double value, Mat& dst)

.. ocv:pyfunction:: cv2.min(src1, src2[, dst]) -> dst

.. ocv:cfunction:: void cvMin(const CvArr* src1, const CvArr* src2, CvArr* dst)
.. ocv:cfunction:: void cvMinS(const CvArr* src, double value, CvArr* dst)
.. ocv:pyoldfunction:: cv.Min(src1, src2, dst)-> None
.. ocv:pyoldfunction:: cv.MinS(src, value, dst)-> None

    :param src1: First source array.

    :param src2: Second source array of the same size and type as  ``src1`` .
    
    :param value: Real scalar value.

    :param dst: Destination array of the same size and type as  ``src1`` .
    
The functions ``min`` compute the per-element minimum of two arrays:

.. math::

    \texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))

or array and a scalar:

.. math::

    \texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )

In the second variant, when the source array is multi-channel, each channel is compared with ``value`` independently.

The first three variants of the function listed above are actually a part of
:ref:`MatrixExpressions` . They return the expression object that can be further either transformed/assigned to a matrix, or passed to a function, and so on.

.. seealso::

    :ocv:func:`max`,
    :ocv:func:`compare`,
    :ocv:func:`inRange`,
    :ocv:func:`minMaxLoc`,
    :ref:`MatrixExpressions`



minMaxLoc
---------
Finds the global minimum and maximum in a whole array or sub-array.

.. ocv:function:: void minMaxLoc(InputArray src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())

.. ocv:function:: void minMaxLoc(const SparseMat& src, double* minVal, double* maxVal, int* minIdx=0, int* maxIdx=0)

.. ocv:pyfunction:: cv2.minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc

.. ocv:cfunction:: void cvMinMaxLoc(const CvArr* arr, double* minVal, double* maxVal, CvPoint* minLoc=NULL, CvPoint* maxLoc=NULL, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.MinMaxLoc(arr, mask=None)-> (minVal, maxVal, minLoc, maxLoc)

    :param src: Source single-channel array.

    :param minVal: Pointer to the returned minimum value.  ``NULL`` is used if not required.

    :param maxVal: Pointer to the returned maximum value.  ``NULL`` is used if not required.

    :param minLoc: Pointer to the returned minimum location (in 2D case).  ``NULL`` is used if not required.

    :param maxLoc: Pointer to the returned maximum location (in 2D case).  ``NULL`` is used if not required.

    :param minIdx: Pointer to the returned minimum location (in nD case). ``NULL`` is used if not required. Otherwise, it must point to an array of  ``src.dims``  elements. The coordinates of the minimum element in each dimension are stored there sequentially.

    :param maxIdx: Pointer to the returned maximum location (in nD case).  ``NULL`` is used if not required.

    :param mask: Optional mask used to select a sub-array.

The functions ``ninMaxLoc`` find the minimum and maximum element values and their positions. The extremums are searched across the whole array or,
if ``mask`` is not an empty array, in the specified array region.

The functions do not work with multi-channel arrays. If you need to find minimum or maximum elements across all the channels, use
:ocv:func:`reshape` first to reinterpret the array as single-channel. Or you may extract the particular channel using either
:ocv:func:`extractImageCOI` , or
:ocv:func:`mixChannels` , or
:ocv:func:`split` .

In case of a sparse matrix, the minimum is found among non-zero elements only.

.. seealso::

    :ocv:func:`max`,
    :ocv:func:`min`,
    :ocv:func:`compare`,
    :ocv:func:`inRange`,
    :ocv:func:`extractImageCOI`,
    :ocv:func:`mixChannels`,
    :ocv:func:`split`,
    :ocv:func:`reshape` 



mixChannels
-----------
Copies specified channels from input arrays to the specified channels of output arrays.

.. ocv:function:: void mixChannels(const Mat* src, int nsrc, Mat* dst, int ndst, const int* fromTo, size_t npairs)

.. ocv:function:: void mixChannels(const vector<Mat>& src, vector<Mat>& dst, const int* fromTo, int npairs)

.. ocv:pyfunction:: cv2.mixChannels(src, dst, fromTo) -> None

.. ocv:cfunction:: void cvMixChannels(const CvArr** src, int srcCount, CvArr** dst, int dstCount, const int* fromTo, int pairCount)
.. ocv:pyoldfunction:: cv.MixChannels(src, dst, fromTo) -> None

    :param src: Input array or vector of matrices. All the matrices must have the same size and the same depth.

    :param nsrc: Number of matrices in  ``src`` .
    
    :param dst: Output array or vector of matrices. All the matrices  *must be allocated* . Their size and depth must be the same as in  ``src[0]`` .
        
    :param ndst: Number of matrices in  ``dst`` .
    
    :param fromTo: Array of index pairs specifying which channels are copied and where. ``fromTo[k*2]``  is a 0-based index of the input channel in  ``src`` . ``fromTo[k*2+1]``  is an index of the output channel in  ``dst`` . The continuous channel numbering is used: the first input image channels are indexed from  ``0``  to  ``src[0].channels()-1`` , the second input image channels are indexed from  ``src[0].channels()``  to ``src[0].channels() + src[1].channels()-1``,  and so on. The same scheme is used for the output image channels. As a special case, when  ``fromTo[k*2]``  is negative, the corresponding output channel is filled with zero .
    
    :param npairs: Number of index pairs in ``fromTo``.
    
The functions ``mixChannels`` provide an advanced mechanism for shuffling image channels.
    
:ocv:func:`split` and
:ocv:func:`merge` and some forms of
:ocv:func:`cvtColor` are partial cases of ``mixChannels`` .

In the example below, the code splits a 4-channel RGBA image into a 3-channel BGR (with R and B channels swapped) and a separate alpha-channel image: ::

    Mat rgba( 100, 100, CV_8UC4, Scalar(1,2,3,4) );
    Mat bgr( rgba.rows, rgba.cols, CV_8UC3 );
    Mat alpha( rgba.rows, rgba.cols, CV_8UC1 );

    // forming an array of matrices is a quite efficient operation,
    // because the matrix data is not copied, only the headers
    Mat out[] = { bgr, alpha };
    // rgba[0] -> bgr[2], rgba[1] -> bgr[1],
    // rgba[2] -> bgr[0], rgba[3] -> alpha[0]
    int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
    mixChannels( &rgba, 1, out, 2, from_to, 4 );


.. note:: Unlike many other new-style C++ functions in OpenCV (see the introduction section and :ocv:func:`Mat::create` ), ``mixChannels`` requires the destination arrays to be pre-allocated before calling the function.

.. seealso::

    :ocv:func:`split`,
    :ocv:func:`merge`,
    :ocv:func:`cvtColor`



mulSpectrums
------------
Performs the per-element multiplication of two Fourier spectrums.

.. ocv:function:: void mulSpectrums(InputArray src1, InputArray src2, OutputArray dst, int flags, bool conj=false)

.. ocv:pyfunction:: cv2.mulSpectrums(a, b, flags[, c[, conjB]]) -> c

.. ocv:cfunction:: void cvMulSpectrums( const CvArr* src1, const CvArr* src2, CvArr* dst, int flags)
.. ocv:pyoldfunction:: cv.MulSpectrums(src1, src2, dst, flags)-> None

    :param src1: First source array.

    :param src2: Second source array of the same size and type as  ``src1`` .
    
    :param dst: Destination array of the same size and type as  ``src1`` .
    
    :param flags: Operation flags. Currently, the only supported flag is ``DFT_ROWS``, which indicates that each row of ``src1`` and ``src2`` is an independent 1D Fourier spectrum.

    :param conj: Optional flag that conjugates the second source array before the multiplication (true) or not (false).

The function ``mulSpectrums`` performs the per-element multiplication of the two CCS-packed or complex matrices that are results of a real or complex Fourier transform.

The function, together with
:ocv:func:`dft` and
:ocv:func:`idft` , may be used to calculate convolution (pass ``conj=false`` ) or correlation (pass ``conj=false`` ) of two arrays rapidly. When the arrays are complex, they are simply multiplied (per element) with an optional conjugation of the second-array elements. When the arrays are real, they are assumed to be CCS-packed (see
:ocv:func:`dft` for details).



multiply
--------
Calculates the per-element scaled product of two arrays.

.. ocv:function:: void multiply(InputArray src1, InputArray src2, OutputArray dst, double scale=1)

.. ocv:pyfunction:: cv2.multiply(src1, src2[, dst[, scale[, dtype]]]) -> dst

.. ocv:cfunction:: void cvMul(const CvArr* src1, const CvArr* src2, CvArr* dst, double scale=1)
.. ocv:pyoldfunction:: cv.Mul(src1, src2, dst, scale)-> None

    :param src1: First source array.

    :param src2: Second source array of the same size and the same type as  ``src1`` .
    
    :param dst: Destination array of the same size and type as  ``src1`` .
    
    :param scale: Optional scale factor.

The function ``multiply`` calculates the per-element product of two arrays:

.. math::

    \texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))

There is also a
:ref:`MatrixExpressions` -friendly variant of the first function. See
:ocv:func:`Mat::mul` .

For a not-per-element matrix product, see
:ocv:func:`gemm` .

.. seealso::

    :ocv:func:`add`,
    :ocv:func:`substract`,
    :ocv:func:`divide`,
    :ref:`MatrixExpressions`,
    :ocv:func:`scaleAdd`,
    :ocv:func:`addWeighted`,
    :ocv:func:`accumulate`,
    :ocv:func:`accumulateProduct`,
    :ocv:func:`accumulateSquare`,
    :ocv:func:`Mat::convertTo`



mulTransposed
-------------
Calculates the product of a matrix and its transposition.

.. ocv:function:: void mulTransposed(InputArray src, OutputArray dst, bool aTa, InputArray delta=noArray(), double scale=1, int rtype=-1)

.. ocv:pyfunction:: cv2.mulTransposed(src, aTa[, dst[, delta[, scale[, dtype]]]]) -> dst

.. ocv:cfunction:: void cvMulTransposed(const CvArr* src, CvArr* dst, int order, const CvArr* delta=NULL, double scale=1.0)
.. ocv:pyoldfunction:: cv.MulTransposed(src, dst, order, delta=None, scale)-> None

    :param src: Source single-channel matrix. Note that unlike :ocv:func:`gemm`, the function can multiply not only floating-point matrices.

    :param dst: Destination square matrix.

    :param aTa: Flag specifying the multiplication ordering. See the description below.

    :param delta: Optional delta matrix subtracted from  ``src``  before the multiplication. When the matrix is empty ( ``delta=noArray()`` ), it is assumed to be zero, that is, nothing is subtracted. If it has the same size as  ``src`` , it is simply subtracted. Otherwise, it is "repeated" (see  :ocv:func:`repeat` ) to cover the full  ``src``  and then subtracted. Type of the delta matrix, when it is not empty, must be the same as the type of created destination matrix. See the  ``rtype``  parameter description below.

    :param scale: Optional scale factor for the matrix product.

    :param rtype: Optional type of the destination matrix. When it is negative, the destination matrix will have the same type as  ``src`` . Otherwise, it will be ``type=CV_MAT_DEPTH(rtype)`` that should be either  ``CV_32F``  or  ``CV_64F`` .
    
The function ``mulTransposed`` calculates the product of ``src`` and its transposition:

.. math::

    \texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )

if ``aTa=true`` , and

.. math::

    \texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T

otherwise. The function is used to compute the covariance matrix. With zero delta, it can be used as a faster substitute for general matrix product ``A*B`` when ``B=A'``

.. seealso::

    :ocv:func:`calcCovarMatrix`,
    :ocv:func:`gemm`,
    :ocv:func:`repeat`,
    :ocv:func:`reduce`



norm
----
Calculates an absolute array norm, an absolute difference norm, or a relative difference norm.

.. ocv:function:: double norm(InputArray src1, int normType=NORM_L2, InputArray mask=noArray())

.. ocv:function:: double norm(InputArray src1, InputArray src2, int normType, InputArray mask=noArray())

.. ocv:function:: double norm( const SparseMat& src, int normType )

.. ocv:pyfunction:: cv2.norm(src1[, normType[, mask]]) -> retval
.. ocv:pyfunction:: cv2.norm(src1, src2[, normType[, mask]]) -> retval

.. ocv:cfunction:: double cvNorm(const CvArr* arr1, const CvArr* arr2=NULL, int normType=CV_L2, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.Norm(arr1, arr2, normType=CV_L2, mask=None)-> double

    :param src1: First source array.

    :param src2: Second source array of the same size and the same type as  ``src1`` .
    
    :param normType: Type of the norm. See the details below.

    :param mask: Optional operation mask. It must have the same size as ``src1`` and ``CV_8UC1`` type.

The functions ``norm`` calculate an absolute norm of ``src1`` (when there is no ``src2`` ):

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

When the ``mask`` parameter is specified and it is not empty, the norm is computed only over the region specified by the mask.

A multi-channel source arrays are treated as a single-channel, that is, the results for all channels are combined.



normalize
---------
Normalizes the norm or value range of an array.

.. ocv:function:: void normalize(const InputArray src, OutputArray dst, double alpha=1, double beta=0, int normType=NORM_L2, int rtype=-1, InputArray mask=noArray())

.. ocv:function:: void normalize(const SparseMat& src, SparseMat& dst, double alpha, int normType)

.. ocv:pyfunction:: cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]]) -> dst

    :param src: Source array.

    :param dst: Destination array of the same size as  ``src`` .
    
    :param alpha: Norm value to normalize to or the lower range boundary in case of the range normalization.

    :param beta: Upper range boundary in case ofthe range normalization. It is not used for the norm normalization.

    :param normType: Normalization type. See the details below.

    :param rtype: When the parameter is negative, the destination array has the same type as  ``src``. Otherwise, it has the same number of channels as  ``src``  and the depth ``=CV_MAT_DEPTH(rtype)`` .
    
    :param mask: Optional operation mask.


The functions ``normalize`` scale and shift the source array elements so that

.. math::

    \| \texttt{dst} \| _{L_p}= \texttt{alpha}

(where p=Inf, 1 or 2) when ``normType=NORM_INF``, ``NORM_L1``, or ``NORM_L2``, respectively; or so that

.. math::

    \min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}

when ``normType=NORM_MINMAX`` (for dense arrays only).
The optional mask specifies a sub-array to be normalized. This means that the norm or min-n-max are computed over the sub-array, and then this sub-array is modified to be normalized. If you want to only use the mask to compute the norm or min-max but modify the whole array, you can use
:ocv:func:`norm` and
:ocv:func:`Mat::convertTo`.

In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this, the range transformation for sparse matrices is not allowed since it can shift the zero level.

.. seealso::

    :ocv:func:`norm`,
    :ocv:func:`Mat::convertTo`,
    :ocv:func:`SparseMat::convertTo`



PCA
---
.. ocv:class:: PCA

Principal Component Analysis class.

The class is used to compute a special basis for a set of vectors. The basis will consist of eigenvectors of the covariance matrix computed from the input set of vectors. The class ``PCA`` can also transform vectors to/from the new coordinate space defined by the basis. Usually, in this new coordinate system, each vector from the original set (and any linear combination of such vectors) can be quite accurately approximated by taking its first few components, corresponding to the eigenvectors of the largest eigenvalues of the covariance matrix. Geometrically it means that you compute a projection of the vector to a subspace formed by a few eigenvectors corresponding to the dominant eigenvalues of the covariance matrix. And usually such a projection is very close to the original vector. So, you can represent the original vector from a high-dimensional space with a much shorter vector consisting of the projected vector's coordinates in the subspace. Such a transformation is also known as Karhunen-Loeve Transform, or KLT. See
http://en.wikipedia.org/wiki/Principal\_component\_analysis .

The sample below is the function that takes two matrices. The first function stores a set of vectors (a row per vector) that is used to compute PCA. The second function stores another "test" set of vectors (a row per vector). First, these vectors are compressed with PCA, then reconstructed back, and then the reconstruction error norm is computed and printed for each vector. ::

    PCA compressPCA(InputArray pcaset, int maxComponents,
                    const Mat& testset, OutputArray compressed)
    {
        PCA pca(pcaset, // pass the data
                Mat(), // there is no pre-computed mean vector,
                       // so let the PCA engine to compute it
                CV_PCA_DATA_AS_ROW, // indicate that the vectors
                                    // are stored as matrix rows
                                    // (use CV_PCA_DATA_AS_COL if the vectors are
                                    // the matrix columns)
                maxComponents // specify how many principal components to retain
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


.. seealso::

    :ocv:func:`calcCovarMatrix`,
    :ocv:func:`mulTransposed`,
    :ocv:class:`SVD`,
    :ocv:func:`dft`,
    :ocv:func:`dct`



PCA::PCA
------------
PCA constructors

.. ocv:function:: PCA::PCA()

.. ocv:function:: PCA::PCA(InputArray data, InputArray mean, int flags, int maxComponents=0)

    :param data: Input samples stored as matrix rows or matrix columns.

    :param mean: Optional mean value. If the matrix is empty ( ``noArray()`` ), the mean is computed from the data.

    :param flags: Operation flags. Currently the parameter is only used to specify the data layout.

        * **CV_PCA_DATA_AS_ROW** indicates that the input samples are stored as matrix rows.

        * **CV_PCA_DATA_AS_COL** indicates that the input samples are stored as matrix columns.

    :param maxComponents: Maximum number of components that PCA should retain. By default, all the components are retained.

The default constructor initializes an empty PCA structure. The second constructor initializes the structure and calls
:ocv:func:`PCA::operator()` .



PCA::operator ()
----------------
Performs Principal Component Analysis of the supplied dataset.

.. ocv:function:: PCA& PCA::operator()(InputArray data, InputArray mean, int flags, int maxComponents=0)

.. ocv:pyfunction:: cv2.PCACompute(data[, mean[, eigenvectors[, maxComponents]]]) -> mean, eigenvectors

    :param data: Input samples stored as the matrix rows or as the matrix columns.

    :param mean: Optional mean value. If the matrix is empty ( ``noArray()`` ), the mean is computed from the data.

    :param flags: Operation flags. Currently the parameter is only used to specify the data layout.

        * **CV_PCA_DATA_AS_ROW** indicates that the input samples are stored as matrix rows.

        * **CV_PCA_DATA_AS_COL** indicates that the input samples are stored as matrix columns.

    :param maxComponents: Maximum number of components that PCA should retain. By default, all the components are retained.

The operator performs PCA of the supplied dataset. It is safe to reuse the same PCA structure for multiple datasets. That is, if the  structure has been previously used with another dataset, the existing internal data is reclaimed and the new ``eigenvalues``, ``eigenvectors`` , and ``mean`` are allocated and computed.

The computed eigenvalues are sorted from the largest to the smallest and the corresponding eigenvectors are stored as ``PCA::eigenvectors`` rows.



PCA::project
------------
Projects vector(s) to the principal component subspace.

.. ocv:function:: Mat PCA::project(InputArray vec) const

.. ocv:function:: void PCA::project(InputArray vec, OutputArray result) const

.. ocv:pyfunction:: cv2.PCAProject(vec, mean, eigenvectors[, result]) -> result

    :param vec: Input vector(s). They must have the same dimensionality and the same layout as the input data used at PCA phase. That is, if  ``CV_PCA_DATA_AS_ROW``  are specified, then  ``vec.cols==data.cols``  (vector dimensionality) and  ``vec.rows``  is the number of vectors to project. The same is true for the  ``CV_PCA_DATA_AS_COL``  case.

    :param result: Output vectors. In case of  ``CV_PCA_DATA_AS_COL``  , the output matrix has as many columns as the number of input vectors. This means that  ``result.cols==vec.cols``  and the number of rows match the number of principal components (for example,  ``maxComponents``  parameter passed to the constructor).

The methods project one or more vectors to the principal component subspace, where each vector projection is represented by coefficients in the principal component basis. The first form of the method returns the matrix that the second form writes to the result. So the first form can be used as a part of expression while the second form can be more efficient in a processing loop.



PCA::backProject
----------------
Reconstructs vectors from their PC projections.

.. ocv:function:: Mat PCA::backProject(InputArray vec) const

.. ocv:function:: void PCA::backProject(InputArray vec, OutputArray result) const

.. ocv:pyfunction:: cv2.PCABackProject(vec, mean, eigenvectors[, result]) -> result

    :param vec: Coordinates of the vectors in the principal component subspace. The layout and size are the same as of  ``PCA::project``  output vectors.

    :param result: Reconstructed vectors. The layout and size are the same as of  ``PCA::project``  input vectors.

The methods are inverse operations to
:ocv:func:`PCA::project` . They take PC coordinates of projected vectors and reconstruct the original vectors. Unless all the principal components have been retained, the reconstructed vectors are different from the originals. But typically, the difference is small if the number of components is large enough (but still much smaller than the original vector dimensionality). As a result, PCA is used.



perspectiveTransform
--------------------
Performs the perspective matrix transformation of vectors.

.. ocv:function:: void perspectiveTransform(InputArray src, OutputArray dst, InputArray mtx)

.. ocv:pyfunction:: cv2.perspectiveTransform(src, m[, dst]) -> dst

.. ocv:cfunction:: void cvPerspectiveTransform(const CvArr* src, CvArr* dst, const CvMat* mat)
.. ocv:pyoldfunction:: cv.PerspectiveTransform(src, dst, mat)-> None

    :param src: Source two-channel or three-channel floating-point array. Each element is a 2D/3D vector to be transformed.

    :param dst: Destination array of the same size and type as  ``src`` .
    
    :param mtx: ``3x3`` or ``4x4`` floating-point transformation matrix.

The function ``perspectiveTransform`` transforms every element of ``src`` by treating it as a 2D or 3D vector, in the following way:

.. math::

    (x, y, z)  \rightarrow (x'/w, y'/w, z'/w)

where

.. math::

    (x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix}

and

.. math::

    w =  \fork{w'}{if $w' \ne 0$}{\infty}{otherwise}

Here a 3D vector transformation is shown. In case of a 2D vector transformation, the ``z`` component is omitted.

.. note:: The function transforms a sparse set of 2D or 3D vectors. If you want to transform an image using perspective transformation, use :ocv:func:`warpPerspective` . If you have an inverse problem, that is, you want to compute the most probable perspective transformation out of several pairs of corresponding points, you can use :ocv:func:`getPerspectiveTransform` or :ocv:func:`findHomography` .

.. seealso::

    :ocv:func:`transform`,
    :ocv:func:`warpPerspective`,
    :ocv:func:`getPerspectiveTransform`,
    :ocv:func:`findHomography`



phase
-----
Calculates the rotation angle of 2D vectors.

.. ocv:function:: void phase(InputArray x, InputArray y, OutputArray angle, bool angleInDegrees=false)

.. ocv:pyfunction:: cv2.phase(x, y[, angle[, angleInDegrees]]) -> angle

    :param x: Source floating-point array of x-coordinates of 2D vectors.

    :param y: Source array of y-coordinates of 2D vectors. It must have the same size and the same type as  ``x``  .   
    
    :param angle: Destination array of vector angles. It has the same size and same type as  ``x`` .
    
    :param angleInDegrees: When it is true, the function computes the angle in degrees. Otherwise, they are measured in radians.

The function ``phase`` computes the rotation angle of each 2D vector that is formed from the corresponding elements of ``x`` and ``y`` :

.. math::

    \texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))

The angle estimation accuracy is about 0.3 degrees. When ``x(I)=y(I)=0`` , the corresponding ``angle(I)`` is set to 0.


polarToCart
-----------
Computes x and y coordinates of 2D vectors from their magnitude and angle.

.. ocv:function:: void polarToCart(InputArray magnitude, InputArray angle, OutputArray x, OutputArray y, bool angleInDegrees=false)

.. ocv:pyfunction:: cv2.polarToCart(magnitude, angle[, x[, y[, angleInDegrees]]]) -> x, y

.. ocv:cfunction:: void cvPolarToCart( const CvArr* magnitude, const CvArr* angle, CvArr* x, CvArr* y, int angleInDegrees=0)
.. ocv:pyoldfunction:: cv.PolarToCart(magnitude, angle, x, y, angleInDegrees=0)-> None

    :param magnitude: Source floating-point array of magnitudes of 2D vectors. It can be an empty matrix ( ``=Mat()`` ). In this case, the function assumes that all the magnitudes are =1. If it is not empty, it must have the same size and type as  ``angle`` .
    
    :param angle: Source floating-point array of angles of 2D vectors.

    :param x: Destination array of x-coordinates of 2D vectors. It has the same size and type as  ``angle``.
    
    :param y: Destination array of y-coordinates of 2D vectors. It has the same size and type as  ``angle``.
    
    :param angleInDegrees: When it is true, the input angles are measured in degrees. Otherwise. they are measured in radians.

The function ``polarToCart`` computes the Cartesian coordinates of each 2D vector represented by the corresponding elements of ``magnitude`` and ``angle`` :

.. math::

    \begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}

The relative accuracy of the estimated coordinates is about ``1e-6``.

.. seealso::

    :ocv:func:`cartToPolar`,
    :ocv:func:`magnitude`,
    :ocv:func:`phase`,
    :ocv:func:`exp`,
    :ocv:func:`log`,
    :ocv:func:`pow`,
    :ocv:func:`sqrt`



pow
---
Raises every array element to a power.

.. ocv:function:: void pow(InputArray src, double p, OutputArray dst)

.. ocv:pyfunction:: cv2.pow(src, power[, dst]) -> dst

.. ocv:cfunction:: void cvPow( const CvArr* src, CvArr* dst, double power)
.. ocv:pyoldfunction:: cv.Pow(src, dst, power)-> None

    :param src: Source array.

    :param p: Exponent of power.

    :param dst: Destination array of the same size and type as  ``src`` .

The function ``pow`` raises every element of the input array to ``p`` :

.. math::

    \texttt{dst} (I) =  \fork{\texttt{src}(I)^p}{if \texttt{p} is integer}{|\texttt{src}(I)|^p}{otherwise}

So, for a non-integer power exponent, the absolute values of input array elements are used. However, it is possible to get true values for negative values using some extra operations. In the example below, computing the 5th root of array ``src``  shows: ::

    Mat mask = src < 0;
    pow(src, 1./5, dst);
    subtract(Scalar::all(0), dst, dst, mask);


For some values of ``p`` , such as integer values, 0.5 and -0.5, specialized faster algorithms are used.

.. seealso::

    :ocv:func:`sqrt`,
    :ocv:func:`exp`,
    :ocv:func:`log`,
    :ocv:func:`cartToPolar`,
    :ocv:func:`polarToCart`



RNG
---

.. ocv:class:: RNG

Random number generator. It encapsulates the state (currently, a 64-bit integer) and has methods to return scalar random values and to fill arrays with random values. Currently it supports uniform and Gaussian (normal) distributions. The generator uses Multiply-With-Carry algorithm, introduced by G. Marsaglia (
http://en.wikipedia.org/wiki/Multiply-with-carry
). Gaussian-distribution random numbers are generated using the Ziggurat algorithm (
http://en.wikipedia.org/wiki/Ziggurat_algorithm
), introduced by G. Marsaglia and W. W. Tsang.



RNG::RNG
------------
The constructors

.. ocv:function:: RNG::RNG()

.. ocv:function:: RNG::RNG(uint64 state)

    :param state: 64-bit value used to initialize the RNG.

These are the RNG constructors. The first form sets the state to some pre-defined value, equal to ``2**32-1`` in the current implementation. The second form sets the state to the specified value. If you passed ``state=0`` , the constructor uses the above default value instead to avoid the singular random number sequence, consisting of all zeros.



RNG::next
-------------
Returns the next random number.

.. ocv:function:: unsigned RNG::next()

The method updates the state using the MWC algorithm and returns the next 32-bit random number.



RNG::operator T
---------------
Returns the next random number of the specified type.

.. ocv:function:: RNG::operator uchar()

.. ocv:function:: RNG::operator schar()

.. ocv:function:: RNG::operator ushort()

.. ocv:function:: RNG::operator short()

.. ocv:function:: RNG::operator int()

.. ocv:function:: RNG::operator unsigned()

.. ocv:function:: RNG::operator float()

.. ocv:function:: RNG::operator double()

Each of the methods updates the state using the MWC algorithm and returns the next random number of the specified type. In case of integer types, the returned number is from the available value range for the specified type. In case of floating-point types, the returned value is from ``[0,1)`` range.



RNG::operator ()
--------------------
Returns the next random number.

.. ocv:function:: unsigned RNG::operator ()()

.. ocv:function:: unsigned RNG::operator ()(unsigned N)

    :param N: Upper non-inclusive boundary of the returned random number.

The methods transform the state using the MWC algorithm and return the next random number. The first form is equivalent to
:ocv:func:`RNG::next` . The second form returns the random number modulo ``N`` , which means that the result is in the range ``[0, N)`` .



RNG::uniform
----------------
Returns the next random number sampled from the uniform distribution.

.. ocv:function:: int RNG::uniform(int a, int b)

.. ocv:function:: float RNG::uniform(float a, float b)

.. ocv:function:: double RNG::uniform(double a, double b)

    :param a: Lower inclusive boundary of the returned random numbers.

    :param b: Upper non-inclusive boundary of the returned random numbers.

The methods transform the state using the MWC algorithm and return the next uniformly-distributed random number of the specified type, deduced from the input parameter type, from the range ``[a, b)`` . There is a nuance illustrated by the following sample: ::

    RNG rng;

    // always produces 0
    double a = rng.uniform(0, 1);

    // produces double from [0, 1)
    double a1 = rng.uniform((double)0, (double)1);

    // produces float from [0, 1)
    double b = rng.uniform(0.f, 1.f);

    // produces double from [0, 1)
    double c = rng.uniform(0., 1.);

    // may cause compiler error because of ambiguity:
    //  RNG::uniform(0, (int)0.999999)? or RNG::uniform((double)0, 0.99999)?
    double d = rng.uniform(0, 0.999999);


The compiler does not take into account the type of the variable to which you assign the result of ``RNG::uniform`` . The only thing that matters to the compiler is the type of ``a`` and ``b`` parameters. So, if you want a floating-point random number, but the range boundaries are integer numbers, either put dots in the end, if they are constants, or use explicit type cast operators, as in the ``a1`` initialization above.



RNG::gaussian
-----------------
Returns the next random number sampled from the Gaussian distribution.

.. ocv:function:: double RNG::gaussian(double sigma)

    :param sigma: Standard deviation of the distribution.

The method transforms the state using the MWC algorithm and returns the next random number from the Gaussian distribution ``N(0,sigma)`` . That is, the mean value of the returned random numbers is zero and the standard deviation is the specified ``sigma`` .



RNG::fill
-------------
Fills arrays with random numbers.

.. ocv:function:: void RNG::fill( InputOutputArray mat, int distType, InputArray a, InputArray b )

    :param mat: 2D or N-dimensional matrix. Currently matrices with more than 4 channels are not supported by the methods. Use  :ocv:func:`reshape`  as a possible workaround.

    :param distType: Distribution type, ``RNG::UNIFORM``  or  ``RNG::NORMAL`` .
    
    :param a: First distribution parameter. In case of the uniform distribution, this is an inclusive lower boundary. In case of the normal distribution, this is a mean value.

    :param b: Second distribution parameter. In case of the uniform distribution, this is a non-inclusive upper boundary. In case of the normal distribution, this is a standard deviation (diagonal of the standard deviation matrix or the full standard deviation matrix).

Each of the methods fills the matrix with the random values from the specified distribution. As the new numbers are generated, the RNG state is updated accordingly. In case of multiple-channel images, every channel is filled independently, which means that RNG cannot generate samples from the multi-dimensional Gaussian distribution with non-diagonal covariance matrix directly. To do that, the method generates samples from multi-dimensional standard Gaussian distribution with zero mean and identity covariation matrix, and then transforms them using :ocv:func:`transform` to get samples from the specified Gaussian distribution.

randu
-----
Generates a single uniformly-distributed random number or an array of random numbers.

.. ocv:function:: template<typename _Tp> _Tp randu()

.. ocv:function:: void randu(InputOutputArray mtx, InputArray low, InputArray high)

.. ocv:pyfunction:: cv2.randu(dst, low, high) -> None

    :param mtx: Output array of random numbers. The array must be pre-allocated.

    :param low: Inclusive lower boundary of the generated random numbers.

    :param high: Exclusive upper boundary of the generated random numbers.

The template functions ``randu`` generate and return the next uniformly-distributed random value of the specified type. ``randu<int>()`` is an equivalent to ``(int)theRNG();`` , and so on. See
:ocv:class:`RNG` description.

The second non-template variant of the function fills the matrix ``mtx`` with uniformly-distributed random numbers from the specified range:

.. math::

    \texttt{low} _c  \leq \texttt{mtx} (I)_c <  \texttt{high} _c

.. seealso::

    :ocv:class:`RNG`,
    :ocv:func:`randn`,
    :ocv:func:`theRNG` 



randn
-----
Fills the array with normally distributed random numbers.

.. ocv:function:: void randn(InputOutputArray mtx, InputArray mean, InputArray stddev)

.. ocv:pyfunction:: cv2.randn(dst, mean, stddev) -> None

    :param mtx: Output array of random numbers. The array must be pre-allocated and have 1 to 4 channels.

    :param mean: Mean value (expectation) of the generated random numbers.

    :param stddev: Standard deviation of the generated random numbers. It can be either a vector (in which case a diagonal standard deviation matrix is assumed) or a square matrix.

The function ``randn`` fills the matrix ``mtx`` with normally distributed random numbers with the specified mean vector and the standard deviation matrix. The generated random numbers are clipped to fit the value range of the destination array data type.

.. seealso::

    :ocv:class:`RNG`,
    :ocv:func:`randu`



randShuffle
-----------
Shuffles the array elements randomly.

.. ocv:function:: void randShuffle(InputOutputArray mtx, double iterFactor=1., RNG* rng=0)

.. ocv:pyfunction:: cv2.randShuffle(src[, dst[, iterFactor]]) -> dst

    :param mtx: Input/output numerical 1D array.

    :param iterFactor: Scale factor that determines the number of random swap operations. See the details below.

    :param rng: Optional random number generator used for shuffling. If it is zero, :ocv:func:`theRNG` () is used instead.

The function ``randShuffle`` shuffles the specified 1D array by randomly choosing pairs of elements and swapping them. The number of such swap operations will be ``mtx.rows*mtx.cols*iterFactor`` .

.. seealso::

    :ocv:class:`RNG`,
    :ocv:func:`sort`



reduce
------
Reduces a matrix to a vector.

.. ocv:function:: void reduce(InputArray mtx, OutputArray vec, int dim, int reduceOp, int dtype=-1)

.. ocv:pyfunction:: cv2.reduce(src, dim, rtype[, dst[, dtype]]) -> dst

.. ocv:cfunction:: void cvReduce(const CvArr* src, CvArr* dst, int dim=-1, int op=CV_REDUCE_SUM)
.. ocv:pyoldfunction:: cv.Reduce(src, dst, dim=-1, op=CV_REDUCE_SUM)-> None

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

.. seealso:: :ocv:func:`repeat`



repeat
------
Fills the destination array with repeated copies of the source array.

.. ocv:function:: void repeat(InputArray src, int ny, int nx, OutputArray dst)

.. ocv:function:: Mat repeat(InputArray src, int ny, int nx)

.. ocv:pyfunction:: cv2.repeat(src, ny, nx[, dst]) -> dst

.. ocv:cfunction:: void cvRepeat(const CvArr* src, CvArr* dst)
.. ocv:pyoldfunction:: cv.Repeat(src, dst)-> None

    :param src: Source array to replicate.

    :param dst: Destination array of the same type as  ``src`` .
    
    :param ny: Flag to specify how many times the  ``src``  is repeated along the vertical axis.

    :param nx: Flag to specify how many times the  ``src``  is repeated along the horizontal axis.

The functions
:ocv:func:`repeat` duplicate the source array one or more times along each of the two axes:

.. math::

    \texttt{dst} _{ij}= \texttt{src} _{i\mod src.rows, \; j\mod src.cols }

The second variant of the function is more convenient to use with
:ref:`MatrixExpressions` . 

.. seealso::

    :ocv:func:`reduce`,
    :ref:`MatrixExpressions`



scaleAdd
--------
Calculates the sum of a scaled array and another array.

.. ocv:function:: void scaleAdd(InputArray src1, double scale, InputArray src2, OutputArray dst)

.. ocv:pyfunction:: cv2.scaleAdd(src1, alpha, src2[, dst]) -> dst

.. ocv:cfunction:: void cvScaleAdd(const CvArr* src1, CvScalar scale, const CvArr* src2, CvArr* dst)
.. ocv:pyoldfunction:: cv.ScaleAdd(src1, scale, src2, dst)-> None

    :param src1: First source array.

    :param scale: Scale factor for the first array.

    :param src2: Second source array of the same size and type as  ``src1`` .
    
    :param dst: Destination array of the same size and type as  ``src1`` .
    
The function ``scaleAdd`` is one of the classical primitive linear algebra operations, known as ``DAXPY`` or ``SAXPY`` in `BLAS <http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_. It calculates the sum of a scaled array and another array:

.. math::

    \texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)

The function can also be emulated with a matrix expression, for example: ::

    Mat A(3, 3, CV_64F);
    ...
    A.row(0) = A.row(1)*2 + A.row(2);


.. seealso::

    :ocv:func:`add`,
    :ocv:func:`addWeighted`,
    :ocv:func:`subtract`,
    :ocv:func:`Mat::dot`,
    :ocv:func:`Mat::convertTo`,
    :ref:`MatrixExpressions`



setIdentity
-----------
Initializes a scaled identity matrix.

.. ocv:function:: void setIdentity(InputOutputArray dst, const Scalar& value=Scalar(1))

.. ocv:pyfunction:: cv2.setIdentity(mtx[, s]) -> None

.. ocv:cfunction:: void cvSetIdentity(CvArr* mat, CvScalar value=cvRealScalar(1))
.. ocv:pyoldfunction:: cv.SetIdentity(mat, value=1)-> None

    :param dst: Matrix to initialize (not necessarily square).

    :param value: Value to assign to diagonal elements.

The function
:ocv:func:`setIdentity` initializes a scaled identity matrix:

.. math::

    \texttt{dst} (i,j)= \fork{\texttt{value}}{ if $i=j$}{0}{otherwise}

The function can also be emulated using the matrix initializers and the matrix expressions: ::

    Mat A = Mat::eye(4, 3, CV_32F)*5;
    // A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]


.. seealso::

    :ocv:func:`Mat::zeros`,
    :ocv:func:`Mat::ones`,
    :ref:`MatrixExpressions`,
    :ocv:func:`Mat::setTo`,
    :ocv:func:`Mat::operator=`



solve
-----
Solves one or more linear systems or least-squares problems.

.. ocv:function:: bool solve(InputArray src1, InputArray src2, OutputArray dst, int flags=DECOMP_LU)

.. ocv:pyfunction:: cv2.solve(src1, src2[, dst[, flags]]) -> retval, dst

.. ocv:cfunction:: int cvSolve(const CvArr* src1, const CvArr* src2, CvArr* dst, int method=CV_LU)
.. ocv:pyoldfunction:: cv.Solve(A, B, X, method=CV_LU)-> None

    :param src1: Input matrix on the left-hand side of the system.

    :param src2: Input matrix on the right-hand side of the system.

    :param dst: Output solution.

    :param flags: Solution (matrix inversion) method.

            * **DECOMP_LU** Gaussian elimination with optimal pivot element chosen.

            * **DECOMP_CHOLESKY** Cholesky  :math:`LL^T`  factorization. The matrix  ``src1``  must be symmetrical and positively defined.

            * **DECOMP_EIG** Eigenvalue decomposition. The matrix  ``src1``  must be symmetrical.

            * **DECOMP_SVD** Singular value decomposition (SVD) method. The system can be over-defined and/or the matrix  ``src1``  can be singular.

            * **DECOMP_QR** QR factorization. The system can be over-defined and/or the matrix  ``src1``  can be singular.

            * **DECOMP_NORMAL** While all the previous flags are mutually exclusive, this flag can be used together with any of the previous. It means that the normal equations  :math:`\texttt{src1}^T\cdot\texttt{src1}\cdot\texttt{dst}=\texttt{src1}^T\texttt{src2}`  are solved instead of the original system  :math:`\texttt{src1}\cdot\texttt{dst}=\texttt{src2}` .
            
The function ``solve`` solves a linear system or least-squares problem (the latter is possible with SVD or QR methods, or by specifying the flag ``DECOMP_NORMAL`` ):

.. math::

    \texttt{dst} =  \arg \min _X \| \texttt{src1} \cdot \texttt{X} -  \texttt{src2} \|

If ``DECOMP_LU`` or ``DECOMP_CHOLESKY`` method is used, the function returns 1 if ``src1`` (or
:math:`\texttt{src1}^T\texttt{src1}` ) is non-singular. Otherwise, it returns 0. In the latter case, ``dst`` is not valid. Other methods find a pseudo-solution in case of a singular left-hand side part.

.. note:: If you want to find a unity-norm solution of an under-defined singular system :math:`\texttt{src1}\cdot\texttt{dst}=0` , the function ``solve`` will not do the work. Use :ocv:func:`SVD::solveZ` instead.

.. seealso::

    :ocv:func:`invert`,
    :ocv:class:`SVD`,
    :ocv:func:`eigen`



solveCubic
--------------
Finds the real roots of a cubic equation.

.. ocv:function:: void solveCubic(InputArray coeffs, OutputArray roots)

.. ocv:pyfunction:: cv2.solveCubic(coeffs[, roots]) -> retval, roots

.. ocv:cfunction:: void cvSolveCubic(const CvArr* coeffs, CvArr* roots)
.. ocv:pyoldfunction:: cv.SolveCubic(coeffs, roots)-> None

    :param coeffs: Equation coefficients, an array of 3 or 4 elements.

    :param roots: Destination array of real roots that has 1 or 3 elements.

The function ``solveCubic`` finds the real roots of a cubic equation:

* if ``coeffs`` is a 4-element vector:

.. math::

    \texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0

* if ``coeffs`` is a 3-element vector:

.. math::

    x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0

The roots are stored in the ``roots`` array.



solvePoly
---------
Finds the real or complex roots of a polynomial equation.

.. ocv:function:: void solvePoly(InputArray coeffs, OutputArray roots, int maxIters=300)

.. ocv:pyfunction:: cv2.solvePoly(coeffs[, roots[, maxIters]]) -> retval, roots

    :param coeffs: Array of polynomial coefficients.

    :param roots: Destination (complex) array of roots.

    :param maxIters: Maximum number of iterations the algorithm does.

The function ``solvePoly`` finds real and complex roots of a polynomial equation:

.. math::

    \texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0



sort
----
Sorts each row or each column of a matrix.

.. ocv:function:: void sort(InputArray src, OutputArray dst, int flags)

.. ocv:pyfunction:: cv2.sort(src, flags[, dst]) -> dst

    :param src: Source single-channel array.

    :param dst: Destination array of the same size and type as  ``src`` .
    
    :param flags: Operation flags, a combination of the following values:

            * **CV_SORT_EVERY_ROW** Each matrix row is sorted independently.

            * **CV_SORT_EVERY_COLUMN** Each matrix column is sorted independently. This flag and the previous one are mutually exclusive.

            * **CV_SORT_ASCENDING** Each matrix row is sorted in the ascending order.

            * **CV_SORT_DESCENDING** Each matrix row is sorted in the descending order. This flag and the previous one are also mutually exclusive.

The function ``sort`` sorts each matrix row or each matrix column in ascending or descending order. So you should pass two operation flags to get desired behaviour. If you want to sort matrix rows or columns lexicographically, you can use STL ``std::sort`` generic function with the proper comparison predicate.

.. seealso::

    :ocv:func:`sortIdx`,
    :ocv:func:`randShuffle`



sortIdx
-------
Sorts each row or each column of a matrix.

.. ocv:function:: void sortIdx(InputArray src, OutputArray dst, int flags)

.. ocv:pyfunction:: cv2.sortIdx(src, flags[, dst]) -> dst

    :param src: Source single-channel array.

    :param dst: Destination integer array of the same size as  ``src`` .
    
    :param flags: Operation flags that could be a combination of the following values:

            * **CV_SORT_EVERY_ROW** Each matrix row is sorted independently.

            * **CV_SORT_EVERY_COLUMN** Each matrix column is sorted independently. This flag and the previous one are mutually exclusive.

            * **CV_SORT_ASCENDING** Each matrix row is sorted in the ascending order.

            * **CV_SORT_DESCENDING** Each matrix row is sorted in the descending order. This flag and the previous one are also mutually exclusive.

The function ``sortIdx`` sorts each matrix row or each matrix column in the ascending or descending order. So you should pass two operation flags to get desired behaviour. Instead of reordering the elements themselves, it stores the indices of sorted elements in the destination array. For example: ::

    Mat A = Mat::eye(3,3,CV_32F), B;
    sortIdx(A, B, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
    // B will probably contain
    // (because of equal elements in A some permutations are possible):
    // [[1, 2, 0], [0, 2, 1], [0, 1, 2]]


.. seealso::

    :ocv:func:`sort`,
    :ocv:func:`randShuffle`



split
-----
Divides a multi-channel array into several single-channel arrays.

.. ocv:function:: void split(const Mat& mtx, Mat* mv)

.. ocv:function:: void split(const Mat& mtx, vector<Mat>& mv)

.. ocv:pyfunction:: cv2.split(m, mv) -> None

.. ocv:cfunction:: void cvSplit(const CvArr* src, CvArr* dst0, CvArr* dst1, CvArr* dst2, CvArr* dst3)
.. ocv:pyoldfunction:: cv.Split(src, dst0, dst1, dst2, dst3)-> None

    :param mtx: Source multi-channel array.

    :param mv: Destination array or vector of arrays. In the first variant of the function the number of arrays must match  ``mtx.channels()`` . The arrays themselves are reallocated, if needed.

The functions ``split`` split a multi-channel array into separate single-channel arrays:

.. math::

    \texttt{mv} [c](I) =  \texttt{mtx} (I)_c

If you need to extract a single channel or do some other sophisticated channel permutation, use
:ocv:func:`mixChannels` .

.. seealso::

    :ocv:func:`merge`,
    :ocv:func:`mixChannels`,
    :ocv:func:`cvtColor`



sqrt
----
Calculates a quare root of array elements.

.. ocv:function:: void sqrt(InputArray src, OutputArray dst)

.. ocv:pyfunction:: cv2.sqrt(src[, dst]) -> dst

.. ocv:cfunction:: float cvSqrt(float value)
.. ocv:pyoldfunction:: cv.Sqrt(value)-> float

    :param src: Source floating-point array.

    :param dst: Destination array of the same size and type as  ``src`` .
    
The functions ``sqrt`` calculate a square root of each source array element. In case of multi-channel arrays, each channel is processed independently. The accuracy is approximately the same as of the built-in ``std::sqrt`` .

.. seealso::

    :ocv:func:`pow`,
    :ocv:func:`magnitude`



subtract
--------
Calculates the per-element difference between two arrays or array and a scalar.

.. ocv:function:: void subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask=noArray(), int dtype=-1)

.. ocv:pyfunction:: cv2.subtract(src1, src2[, dst[, mask[, dtype]]]) -> dst

.. ocv:cfunction:: void cvSub(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)
.. ocv:cfunction:: void cvSubRS(const CvArr* src1, CvScalar src2, CvArr* dst, const CvArr* mask=NULL)
.. ocv:cfunction:: void cvSubS(const CvArr* src1, CvScalar src2, CvArr* dst, const CvArr* mask=NULL)

.. ocv:pyoldfunction:: cv.Sub(src1, src2, dst, mask=None)-> None
.. ocv:pyoldfunction:: cv.SubRS(src1, src2, dst, mask=None)-> None
.. ocv:pyoldfunction:: cv.SubS(src1, src2, dst, mask=None)-> None

    :param src1: First source array or a scalar.

    :param src2: Second source array or a scalar.
    
    :param dst: Destination array of the same size and the same number of channels as the input array.   
    
    :param mask: Optional operation mask. This is an 8-bit single channel array that specifies elements of the destination array to be changed.
    
    :param dtype: Optional depth of the output array. See the details below.

The function ``subtract`` computes:

 *
    Difference between two arrays, when both input arrays have the same size and the same number of channels:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0

 *
    Difference between an array and a scalar, when ``src2`` is constructed from ``Scalar`` or has the same number of elements as ``src1.channels()``:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0

 *
    Difference between a scalar and an array, when ``src1`` is constructed from ``Scalar`` or has the same number of elements as ``src2.channels()``:

    .. math::

        \texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0
        
 *
    The reverse difference between a scalar and an array in the case of ``SubRS``:
    
    .. math::

        \texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0

where ``I`` is a multi-dimensional index of array elements. In case of multi-channel arrays, each channel is processed independently.

The first function in the list above can be replaced with matrix expressions: ::

    dst = src1 - src2;
    dst -= src1; // equivalent to subtract(dst, src1, dst);

The input arrays and the destination array can all have the same or different depths. For example, you can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of the output array is determined by ``dtype`` parameter. In the second and third cases above, as well as in the first case, when ``src1.depth() == src2.depth()``, ``dtype`` can be set to the default ``-1``. In this case the output array will have the same depth as the input array, be it ``src1``, ``src2`` or both.

.. seealso::

    :ocv:func:`add`,
    :ocv:func:`addWeighted`,
    :ocv:func:`scaleAdd`,
    :ocv:func:`Mat::convertTo`,
    :ref:`MatrixExpressions`



SVD
---
.. ocv:class:: SVD

Class for computing Singular Value Decomposition of a floating-point matrix. The Singular Value Decomposition is used to solve least-square problems, under-determined linear systems, invert matrices, compute condition numbers, and so on.

For a faster operation, you can pass ``flags=SVD::MODIFY_A|...`` to modify the decomposed matrix when it is not necessary to preserve it. If you want to compute a condition number of a matrix or an absolute value of its determinant, you do not need ``u`` and ``vt`` . You can pass ``flags=SVD::NO_UV|...`` . Another flag ``FULL_UV`` indicates that full-size ``u`` and ``vt`` must be computed, which is not necessary most of the time.

.. seealso::

    :ocv:func:`invert`,
    :ocv:func:`solve`,
    :ocv:func:`eigen`,
    :ocv:func:`determinant`



SVD::SVD
--------
The constructors.

.. ocv:function:: SVD::SVD()

.. ocv:function:: SVD::SVD( InputArray A, int flags=0 )

    :param src: Decomposed matrix.

    :param flags: Operation flags.

        * **SVD::MODIFY_A** Use the algorithm to modify the decomposed matrix. It can save space and speed up processing.

        * **SVD::NO_UV** Indicate that only a vector of singular values  ``w``  is to be computed, while  ``u``  and  ``vt``  will be set to empty matrices.

        * **SVD::FULL_UV** When the matrix is not square, by default the algorithm produces  ``u``  and  ``vt``  matrices of sufficiently large size for the further  ``A``  reconstruction. If, however, ``FULL_UV``  flag is specified, ``u``  and  ``vt``  will be full-size square orthogonal matrices.

The first constructor initializes an empty ``SVD`` structure. The second constructor initializes an empty ``SVD`` structure and then calls
:ocv:func:`SVD::operator()` .


SVD::operator ()
----------------
Performs SVD of a matrix.

.. ocv:function:: SVD& SVD::operator()( InputArray src, int flags=0 )

    :param src: Decomposed matrix.

    :param flags: Operation flags.

        * **SVD::MODIFY_A** Use the algorithm to modify the decomposed matrix. It can save space and speed up processing.

        * **SVD::NO_UV** Use only singular values. The algorithm does not compute  ``u``  and  ``vt``  matrices.

        * **SVD::FULL_UV** When the matrix is not square, by default the algorithm produces  ``u``  and  ``vt``  matrices of sufficiently large size for the further  ``A``  reconstruction. If, however, the ``FULL_UV``  flag is specified, ``u``  and  ``vt``  are full-size square orthogonal matrices.

The operator performs the singular value decomposition of the supplied matrix. The ``u``,``vt`` , and the vector of singular values ``w`` are stored in the structure. The same ``SVD`` structure can be reused many times with different matrices. Each time, if needed, the previous ``u``,``vt`` , and ``w`` are reclaimed and the new matrices are created, which is all handled by
:ocv:func:`Mat::create` .


SVD::compute
------------
Performs SVD of a matrix

.. ocv:function:: static void SVD::compute( InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags=0 )

.. ocv:function:: static void SVD::compute( InputArray src, OutputArray w, int flags=0 )

.. ocv:pyfunction:: cv2.SVDecomp(src[, w[, u[, vt[, flags]]]]) -> w, u, vt

.. ocv:cfunction:: void cvSVD( CvArr* src, CvArr* w, CvArr* u=NULL, CvArr* v=NULL, int flags=0)

.. ocv:pyoldfunction:: cv.SVD(src, w, u=None, v=None, flags=0)-> None

    :param src: Decomposed matrix
    
    :param w: Computed singular values
    
    :param u: Computed left singular vectors
    
    :param v: Computed right singular vectors
    
    :param vt: Transposed matrix of right singular values
    
    :param flags: Opertion flags - see :ocv:func:`SVD::SVD`.

The methods/functions perform SVD of matrix. Unlike ``SVD::SVD`` constructor and ``SVD::operator()``, they store the results to the user-provided matrices. ::

    Mat A, w, u, vt;
    SVD::compute(A, w, u, vt);
    

SVD::solveZ
-----------
Solves an under-determined singular linear system.

.. ocv:function:: static void SVD::solveZ( InputArray src, OutputArray dst )

    :param src: Left-hand-side matrix.

    :param dst: Found solution.

The method finds a unit-length solution ``x`` of a singular linear system 
``A*x = 0``. Depending on the rank of ``A``, there can be no solutions, a single solution or an infinite number of solutions. In general, the algorithm solves the following problem:

.. math::

    dst =  \arg \min _{x:  \| x \| =1}  \| src  \cdot x  \|


SVD::backSubst
--------------
Performs a singular value back substitution.

.. ocv:function:: void SVD::backSubst( InputArray rhs, OutputArray dst ) const

.. ocv:function:: static void SVD::backSubst( InputArray w, InputArray u, InputArray vt, InputArray rhs, OutputArray dst )

.. ocv:pyfunction:: cv2.SVBackSubst(w, u, vt, rhs[, dst]) -> dst

.. ocv:cfunction:: void cvSVBkSb( const CvArr* w, const CvArr* u, const CvArr* v, const CvArr* rhs, CvArr* dst, int flags)

.. ocv:pyoldfunction:: cv.SVBkSb(w, u, v, rhs, dst, flags)-> None

    :param w: Singular values
    
    :param u: Left singular vectors
    
    :param v: Right singular vectors
    
    :param vt: Transposed matrix of right singular vectors.

    :param rhs: Right-hand side of a linear system ``(u*w*v')*dst = rhs`` to be solved, where ``A`` has been previously decomposed.
    
    :param dst: Found solution of the system.

The method computes a back substitution for the specified right-hand side:

.. math::

    \texttt{x} =  \texttt{vt} ^T  \cdot diag( \texttt{w} )^{-1}  \cdot \texttt{u} ^T  \cdot \texttt{rhs} \sim \texttt{A} ^{-1}  \cdot \texttt{rhs}

Using this technique you can either get a very accurate solution of the convenient linear system, or the best (in the least-squares terms) pseudo-solution of an overdetermined linear system. 

.. note:: Explicit SVD with the further back substitution only makes sense if you need to solve many linear systems with the same left-hand side (for example, ``src`` ). If all you need is to solve a single system (possibly with multiple ``rhs`` immediately available), simply call :ocv:func:`solve` add pass ``DECOMP_SVD`` there. It does absolutely the same thing.



sum
---
Calculates the sum of array elements.

.. ocv:function:: Scalar sum(InputArray arr)

.. ocv:pyfunction:: cv2.sumElems(arr) -> retval

.. ocv:cfunction:: CvScalar cvSum(const CvArr* arr)
.. ocv:pyoldfunction:: cv.Sum(arr)-> CvScalar

    :param arr: Source array that must have from 1 to 4 channels.

The functions ``sum`` calculate and return the sum of array elements, independently for each channel.

.. seealso::

    :ocv:func:`countNonZero`,
    :ocv:func:`mean`,
    :ocv:func:`meanStdDev`,
    :ocv:func:`norm`,
    :ocv:func:`minMaxLoc`,
    :ocv:func:`reduce`



theRNG
------
Returns the default random number generator.

.. ocv:function:: RNG& theRNG()

The function ``theRNG`` returns the default random number generator. For each thread, there is a separate random number generator, so you can use the function safely in multi-thread environments. If you just need to get a single random number using this generator or initialize an array, you can use
:ocv:func:`randu` or
:ocv:func:`randn` instead. But if you are going to generate many random numbers inside a loop, it is much faster to use this function to retrieve the generator and then use ``RNG::operator _Tp()`` .

.. seealso::

    :ocv:class:`RNG`,
    :ocv:func:`randu`,
    :ocv:func:`randn`



trace
-----
Returns the trace of a matrix.

.. ocv:function:: Scalar trace(InputArray mat)

.. ocv:pyfunction:: cv2.trace(mat) -> retval

.. ocv:cfunction:: CvScalar cvTrace(const CvArr* mat)
.. ocv:pyoldfunction:: cv.Trace(mat)-> CvScalar

    :param mtx: Source matrix.

The function ``trace`` returns the sum of the diagonal elements of the matrix ``mtx`` .

.. math::

    \mathrm{tr} ( \texttt{mtx} ) =  \sum _i  \texttt{mtx} (i,i)



transform
---------
Performs the matrix transformation of every array element.

.. ocv:function:: void transform(InputArray src, OutputArray dst, InputArray mtx )

.. ocv:pyfunction:: cv2.transform(src, mtx [, dst]) -> dst

.. ocv:cfunction:: void cvTransform(const CvArr* src, CvArr* dst, const CvMat* mtx, const CvMat* shiftvec=NULL)
.. ocv:pyoldfunction:: cv.Transform(src, dst, mtx, shiftvec=None)-> None

    :param src: Source array that must have as many channels (1 to 4) as  ``mtx.cols``  or  ``mtx.cols-1``.
    
    :param dst: Destination array of the same size and depth as  ``src`` . It has as many channels as  ``mtx.rows``  .   
    
    :param mtx: Transformation ``2x2`` or ``2x3`` floating-point matrix.
    
    :param shiftvec: Optional translation vector (when ``mtx`` is ``2x2``)

The function ``transform`` performs the matrix transformation of every element of the array ``src`` and stores the results in ``dst`` :

.. math::

    \texttt{dst} (I) =  \texttt{mtx} \cdot \texttt{src} (I)

(when ``mtx.cols=src.channels()`` ), or

.. math::

    \texttt{dst} (I) =  \texttt{mtx} \cdot [ \texttt{src} (I); 1]

(when ``mtx.cols=src.channels()+1`` )

Every element of the ``N`` -channel array ``src`` is interpreted as ``N`` -element vector that is transformed using
the ``M x N`` or ``M x (N+1)`` matrix ``mtx``
to ``M``-element vector - the corresponding element of the destination array ``dst`` .

The function may be used for geometrical transformation of
``N`` -dimensional
points, arbitrary linear color space transformation (such as various kinds of RGB to YUV transforms), shuffling the image channels, and so forth.

.. seealso::

    :ocv:func:`perspectiveTransform`,
    :ocv:func:`getAffineTransform`,
    :ocv:func:`estimateRigidTransform`,
    :ocv:func:`warpAffine`,
    :ocv:func:`warpPerspective`



transpose
---------
Transposes a matrix.

.. ocv:function:: void transpose(InputArray src, OutputArray dst)

.. ocv:pyfunction:: cv2.transpose(src[, dst]) -> dst

.. ocv:cfunction:: void cvTranspose(const CvArr* src, CvArr* dst)
.. ocv:pyoldfunction:: cv.Transpose(src, dst)-> None

    :param src: Source array.

    :param dst: Destination array of the same type as  ``src`` .
    
The function :ocv:func:`transpose` transposes the matrix ``src`` :

.. math::

    \texttt{dst} (i,j) =  \texttt{src} (j,i)

.. note:: No complex conjugation is done in case of a complex matrix. It it should be done separately if needed.
