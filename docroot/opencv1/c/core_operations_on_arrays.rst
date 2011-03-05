Operations on Arrays
====================

.. highlight:: c



.. index:: AbsDiff

.. _AbsDiff:

AbsDiff
-------

`id=0.389752508219 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/AbsDiff>`__




.. cfunction:: void cvAbsDiff(const CvArr* src1, const CvArr* src2, CvArr* dst)

    Calculates absolute difference between two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    
The function calculates absolute difference between two arrays.



.. math::

    \texttt{dst} (i)_c = | \texttt{src1} (I)_c -  \texttt{src2} (I)_c|  


All the arrays must have the same data type and the same size (or ROI size).


.. index:: AbsDiffS

.. _AbsDiffS:

AbsDiffS
--------

`id=0.906294304824 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/AbsDiffS>`__




.. cfunction:: void cvAbsDiffS(const CvArr* src, CvArr* dst, CvScalar value)

    Calculates absolute difference between an array and a scalar.






::


    
    #define cvAbs(src, dst) cvAbsDiffS(src, dst, cvScalarAll(0))
    

..



    
    :param src: The source array 
    
    
    :param dst: The destination array 
    
    
    :param value: The scalar 
    
    
    
The function calculates absolute difference between an array and a scalar.



.. math::

    \texttt{dst} (i)_c = | \texttt{src} (I)_c -  \texttt{value} _c|  


All the arrays must have the same data type and the same size (or ROI size).



.. index:: Add

.. _Add:

Add
---

`id=0.857040798932 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Add>`__




.. cfunction:: void cvAdd(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

    Computes the per-element sum of two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function adds one array to another:




::


    
    dst(I)=src1(I)+src2(I) if mask(I)!=0
    

..

All the arrays must have the same type, except the mask, and the same size (or ROI size).
For types that have limited range this operation is saturating.


.. index:: AddS

.. _AddS:

AddS
----

`id=0.475031728547 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/AddS>`__




.. cfunction:: void cvAddS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

    Computes the sum of an array and a scalar.





    
    :param src: The source array 
    
    
    :param value: Added scalar 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function adds a scalar 
``value``
to every element in the source array 
``src1``
and stores the result in 
``dst``
.
For types that have limited range this operation is saturating.




::


    
    dst(I)=src(I)+value if mask(I)!=0
    

..

All the arrays must have the same type, except the mask, and the same size (or ROI size).



.. index:: AddWeighted

.. _AddWeighted:

AddWeighted
-----------

`id=0.57991333562 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/AddWeighted>`__




.. cfunction:: void  cvAddWeighted(const CvArr* src1, double alpha,                      const CvArr* src2, double beta,                      double gamma, CvArr* dst)

    Computes the weighted sum of two arrays.





    
    :param src1: The first source array 
    
    
    :param alpha: Weight for the first array elements 
    
    
    :param src2: The second source array 
    
    
    :param beta: Weight for the second array elements 
    
    
    :param dst: The destination array 
    
    
    :param gamma: Scalar, added to each sum 
    
    
    
The function calculates the weighted sum of two arrays as follows:




::


    
    dst(I)=src1(I)*alpha+src2(I)*beta+gamma
    

..

All the arrays must have the same type and the same size (or ROI size).
For types that have limited range this operation is saturating.



.. index:: And

.. _And:

And
---

`id=0.185678982065 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/And>`__




.. cfunction:: void cvAnd(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

    Calculates per-element bit-wise conjunction of two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function calculates per-element bit-wise logical conjunction of two arrays:




::


    
    dst(I)=src1(I)&src2(I) if mask(I)!=0
    

..

In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.


.. index:: AndS

.. _AndS:

AndS
----

`id=0.18019335221 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/AndS>`__




.. cfunction:: void cvAndS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

    Calculates per-element bit-wise conjunction of an array and a scalar.





    
    :param src: The source array 
    
    
    :param value: Scalar to use in the operation 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function calculates per-element bit-wise conjunction of an array and a scalar:




::


    
    dst(I)=src(I)&value if mask(I)!=0
    

..

Prior to the actual operation, the scalar is converted to the same type as that of the array(s). In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.

The following sample demonstrates how to calculate the absolute value of floating-point array elements by clearing the most-significant bit:




::


    
    float a[] = { -1, 2, -3, 4, -5, 6, -7, 8, -9 };
    CvMat A = cvMat(3, 3, CV_32F, &a);
    int i, absMask = 0x7fffffff;
    cvAndS(&A, cvRealScalar(*(float*)&absMask), &A, 0);
    for(i = 0; i < 9; i++ )
        printf("
    

..

The code should print:




::


    
    1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
    

..


.. index:: Avg

.. _Avg:

Avg
---

`id=0.150599164969 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Avg>`__




.. cfunction:: CvScalar cvAvg(const CvArr* arr, const CvArr* mask=NULL)

    Calculates average (mean) of array elements.





    
    :param arr: The array 
    
    
    :param mask: The optional operation mask 
    
    
    
The function calculates the average value 
``M``
of array elements, independently for each channel:



.. math::

    \begin{array}{l} N =  \sum _I ( \texttt{mask} (I)  \ne 0) \\ M_c =  \frac{\sum_{I, \, \texttt{mask}(I) \ne 0} \texttt{arr} (I)_c}{N} \end{array} 


If the array is 
``IplImage``
and COI is set, the function processes the selected channel only and stores the average to the first scalar component 
:math:`S_0`
.


.. index:: AvgSdv

.. _AvgSdv:

AvgSdv
------

`id=0.239443049508 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/AvgSdv>`__




.. cfunction:: void cvAvgSdv(const CvArr* arr, CvScalar* mean, CvScalar* stdDev, const CvArr* mask=NULL)

    Calculates average (mean) of array elements.





    
    :param arr: The array 
    
    
    :param mean: Pointer to the output mean value, may be NULL if it is not needed 
    
    
    :param stdDev: Pointer to the output standard deviation 
    
    
    :param mask: The optional operation mask 
    
    
    
The function calculates the average value and standard deviation of array elements, independently for each channel:



.. math::

    \begin{array}{l} N =  \sum _I ( \texttt{mask} (I)  \ne 0) \\ mean_c =  \frac{1}{N} \, \sum _{ I,  \, \texttt{mask} (I)  \ne 0}  \texttt{arr} (I)_c \\ stdDev_c =  \sqrt{\frac{1}{N} \, \sum_{ I, \, \texttt{mask}(I) \ne 0} ( \texttt{arr} (I)_c - mean_c)^2} \end{array} 


If the array is 
``IplImage``
and COI is set, the function processes the selected channel only and stores the average and standard deviation to the first components of the output scalars (
:math:`mean_0`
and 
:math:`stdDev_0`
).


.. index:: CalcCovarMatrix

.. _CalcCovarMatrix:

CalcCovarMatrix
---------------

`id=0.533338739877 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CalcCovarMatrix>`__




.. cfunction:: void cvCalcCovarMatrix( const CvArr** vects, int count, CvArr* covMat, CvArr* avg, int flags)

    Calculates covariance matrix of a set of vectors.





    
    :param vects: The input vectors, all of which must have the same type and the same size. The vectors do not have to be 1D, they can be 2D (e.g., images) and so forth 
    
    
    :param count: The number of input vectors 
    
    
    :param covMat: The output covariance matrix that should be floating-point and square 
    
    
    :param avg: The input or output (depending on the flags) array - the mean (average) vector of the input vectors 
    
    
    :param flags: The operation flags, a combination of the following values 
         
            * **CV_COVAR_SCRAMBLED** The output covariance matrix is calculated as:  
                
                .. math::
                
                      \texttt{scale}  * [  \texttt{vects}  [0]-  \texttt{avg}  , \texttt{vects}  [1]-  \texttt{avg}  ,...]^T  \cdot  [ \texttt{vects}  [0]- \texttt{avg}  , \texttt{vects}  [1]- \texttt{avg}  ,...]  
                
                ,
                that is, the covariance matrix is :math:`\texttt{count} \times \texttt{count}` .
                Such an unusual covariance matrix is used for fast PCA
                of a set of very large vectors (see, for example, the EigenFaces technique
                for face recognition). Eigenvalues of this "scrambled" matrix will
                match the eigenvalues of the true covariance matrix and the "true"
                eigenvectors can be easily calculated from the eigenvectors of the
                "scrambled" covariance matrix. 
            
            * **CV_COVAR_NORMAL** The output covariance matrix is calculated as:  
                
                .. math::
                
                      \texttt{scale}  * [  \texttt{vects}  [0]-  \texttt{avg}  , \texttt{vects}  [1]-  \texttt{avg}  ,...]  \cdot  [ \texttt{vects}  [0]- \texttt{avg}  , \texttt{vects}  [1]- \texttt{avg}  ,...]^T  
                
                ,
                that is,  ``covMat``  will be a covariance matrix
                with the same linear size as the total number of elements in each
                input vector. One and only one of  ``CV_COVAR_SCRAMBLED``  and ``CV_COVAR_NORMAL``  must be specified 
            
            * **CV_COVAR_USE_AVG** If the flag is specified, the function does not calculate  ``avg``  from the input vectors, but, instead, uses the passed  ``avg``  vector. This is useful if  ``avg``  has been already calculated somehow, or if the covariance matrix is calculated by parts - in this case,  ``avg``  is not a mean vector of the input sub-set of vectors, but rather the mean vector of the whole set. 
            
            * **CV_COVAR_SCALE** If the flag is specified, the covariance matrix is scaled. In the "normal" mode  ``scale``  is '1./count'; in the "scrambled" mode  ``scale``  is the reciprocal of the total number of elements in each input vector. By default (if the flag is not specified) the covariance matrix is not scaled ('scale=1'). 
            
            
            * **CV_COVAR_ROWS** Means that all the input vectors are stored as rows of a single matrix,  ``vects[0]`` .  ``count``  is ignored in this case, and  ``avg``  should be a single-row vector of an appropriate size. 
            
            * **CV_COVAR_COLS** Means that all the input vectors are stored as columns of a single matrix,  ``vects[0]`` .  ``count``  is ignored in this case, and  ``avg``  should be a single-column vector of an appropriate size. 
            
            
            
    
    
    
The function calculates the covariance matrix
and, optionally, the mean vector of the set of input vectors. The function
can be used for PCA, for comparing vectors using Mahalanobis distance and so forth.


.. index:: CartToPolar

.. _CartToPolar:

CartToPolar
-----------

`id=0.387301730832 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CartToPolar>`__




.. cfunction:: void cvCartToPolar( const CvArr* x, const CvArr* y, CvArr* magnitude, CvArr* angle=NULL, int angleInDegrees=0)

    Calculates the magnitude and/or angle of 2d vectors.





    
    :param x: The array of x-coordinates 
    
    
    :param y: The array of y-coordinates 
    
    
    :param magnitude: The destination array of magnitudes, may be set to NULL if it is not needed 
    
    
    :param angle: The destination array of angles, may be set to NULL if it is not needed. The angles are measured in radians  :math:`(0`  to  :math:`2 \pi )`  or in degrees (0 to 360 degrees). 
    
    
    :param angleInDegrees: The flag indicating whether the angles are measured in radians, which is default mode, or in degrees 
    
    
    
The function calculates either the magnitude, angle, or both of every 2d vector (x(I),y(I)):




::


    
    
    magnitude(I)=sqrt(x(I)^2^+y(I)^2^ ),
    angle(I)=atan(y(I)/x(I) )
    
    

..

The angles are calculated with 0.1 degree accuracy. For the (0,0) point, the angle is set to 0.


.. index:: Cbrt

.. _Cbrt:

Cbrt
----

`id=0.47391511107 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Cbrt>`__




.. cfunction:: float cvCbrt(float value)

    Calculates the cubic root





    
    :param value: The input floating-point value 
    
    
    
The function calculates the cubic root of the argument, and normally it is faster than 
``pow(value,1./3)``
. In addition, negative arguments are handled properly. Special values (
:math:`\pm \infty`
, NaN) are not handled.


.. index:: ClearND

.. _ClearND:

ClearND
-------

`id=0.433568700573 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ClearND>`__




.. cfunction:: void cvClearND(CvArr* arr, int* idx)

    Clears a specific array element.




    
    :param arr: Input array 
    
    
    :param idx: Array of the element indices 
    
    
    
The function 
:ref:`ClearND`
clears (sets to zero) a specific element of a dense array or deletes the element of a sparse array. If the sparse array element does not exists, the function does nothing.


.. index:: CloneImage

.. _CloneImage:

CloneImage
----------

`id=0.968680686034 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CloneImage>`__




.. cfunction:: IplImage* cvCloneImage(const IplImage* image)

    Makes a full copy of an image, including the header, data, and ROI.





    
    :param image: The original image 
    
    
    
The returned 
``IplImage*``
points to the image copy.


.. index:: CloneMat

.. _CloneMat:

CloneMat
--------

`id=0.975713536969 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CloneMat>`__




.. cfunction:: CvMat* cvCloneMat(const CvMat* mat)

    Creates a full matrix copy.





    
    :param mat: Matrix to be copied 
    
    
    
Creates a full copy of a matrix and returns a pointer to the copy.


.. index:: CloneMatND

.. _CloneMatND:

CloneMatND
----------

`id=0.570248603442 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CloneMatND>`__




.. cfunction:: CvMatND* cvCloneMatND(const CvMatND* mat)

    Creates full copy of a multi-dimensional array and returns a pointer to the copy.





    
    :param mat: Input array 
    
    
    

.. index:: CloneSparseMat

.. _CloneSparseMat:

CloneSparseMat
--------------

`id=0.709316686508 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CloneSparseMat>`__




.. cfunction:: CvSparseMat* cvCloneSparseMat(const CvSparseMat* mat)

    Creates full copy of sparse array.





    
    :param mat: Input array 
    
    
    
The function creates a copy of the input array and returns pointer to the copy.

.. index:: Cmp

.. _Cmp:

Cmp
---

`id=0.802902555491 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Cmp>`__




.. cfunction:: void cvCmp(const CvArr* src1, const CvArr* src2, CvArr* dst, int cmpOp)

    Performs per-element comparison of two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array. Both source arrays must have a single channel. 
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    
    :param cmpOp: The flag specifying the relation between the elements to be checked 
        
               
            * **CV_CMP_EQ** src1(I) "equal to" value 
            
              
            * **CV_CMP_GT** src1(I) "greater than" value 
            
              
            * **CV_CMP_GE** src1(I) "greater or equal" value 
            
              
            * **CV_CMP_LT** src1(I) "less than" value 
            
              
            * **CV_CMP_LE** src1(I) "less or equal" value 
            
              
            * **CV_CMP_NE** src1(I) "not equal" value 
            
            
    
    
    
The function compares the corresponding elements of two arrays and fills the destination mask array:




::


    
    dst(I)=src1(I) op src2(I),
    

..

``dst(I)``
is set to 0xff (all 
``1``
-bits) if the specific relation between the elements is true and 0 otherwise. All the arrays must have the same type, except the destination, and the same size (or ROI size)


.. index:: CmpS

.. _CmpS:

CmpS
----

`id=0.590507866573 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CmpS>`__




.. cfunction:: void cvCmpS(const CvArr* src, double value, CvArr* dst, int cmpOp)

    Performs per-element comparison of an array and a scalar.





    
    :param src: The source array, must have a single channel 
    
    
    :param value: The scalar value to compare each array element with 
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    
    :param cmpOp: The flag specifying the relation between the elements to be checked 
        
               
            * **CV_CMP_EQ** src1(I) "equal to" value 
            
              
            * **CV_CMP_GT** src1(I) "greater than" value 
            
              
            * **CV_CMP_GE** src1(I) "greater or equal" value 
            
              
            * **CV_CMP_LT** src1(I) "less than" value 
            
              
            * **CV_CMP_LE** src1(I) "less or equal" value 
            
              
            * **CV_CMP_NE** src1(I) "not equal" value 
            
            
    
    
    
The function compares the corresponding elements of an array and a scalar and fills the destination mask array:




::


    
    dst(I)=src(I) op scalar
    

..

where 
``op``
is 
:math:`=,\; >,\; \ge,\; <,\; \le\; or\; \ne`
.

``dst(I)``
is set to 0xff (all 
``1``
-bits) if the specific relation between the elements is true and 0 otherwise. All the arrays must have the same size (or ROI size).


.. index:: ConvertScale

.. _ConvertScale:

ConvertScale
------------

`id=0.634428432556 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ConvertScale>`__




.. cfunction:: void cvConvertScale(const CvArr* src, CvArr* dst, double scale=1, double shift=0)

    Converts one array to another with optional linear transformation.






::


    
    #define cvCvtScale cvConvertScale
    #define cvScale  cvConvertScale
    #define cvConvert(src, dst )  cvConvertScale((src), (dst), 1, 0 )
    

..



    
    :param src: Source array 
    
    
    :param dst: Destination array 
    
    
    :param scale: Scale factor 
    
    
    :param shift: Value added to the scaled source array elements 
    
    
    
The function has several different purposes, and thus has several different names. It copies one array to another with optional scaling, which is performed first, and/or optional type conversion, performed after:



.. math::

    \texttt{dst} (I) =  \texttt{scale} \texttt{src} (I) + ( \texttt{shift} _0, \texttt{shift} _1,...) 


All the channels of multi-channel arrays are processed independently.

The type of conversion is done with rounding and saturation, that is if the
result of scaling + conversion can not be represented exactly by a value
of the destination array element type, it is set to the nearest representable
value on the real axis.

In the case of 
``scale=1, shift=0``
no prescaling is done. This is a specially
optimized case and it has the appropriate 
:ref:`Convert`
name. If
source and destination array types have equal types, this is also a
special case that can be used to scale and shift a matrix or an image
and that is caled 
:ref:`Scale`
.



.. index:: ConvertScaleAbs

.. _ConvertScaleAbs:

ConvertScaleAbs
---------------

`id=0.936176741204 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ConvertScaleAbs>`__




.. cfunction:: void cvConvertScaleAbs(const CvArr* src, CvArr* dst, double scale=1, double shift=0)

    Converts input array elements to another 8-bit unsigned integer with optional linear transformation.





    
    :param src: Source array 
    
    
    :param dst: Destination array (should have 8u depth) 
    
    
    :param scale: ScaleAbs factor 
    
    
    :param shift: Value added to the scaled source array elements 
    
    
    
The function is similar to 
:ref:`ConvertScale`
, but it stores absolute values of the conversion results:



.. math::

    \texttt{dst} (I) = | \texttt{scale} \texttt{src} (I) + ( \texttt{shift} _0, \texttt{shift} _1,...)| 


The function supports only destination arrays of 8u (8-bit unsigned integers) type; for other types the function can be emulated by a combination of 
:ref:`ConvertScale`
and 
:ref:`Abs`
functions.


.. index:: CvtScaleAbs

.. _CvtScaleAbs:

CvtScaleAbs
-----------

`id=0.460721939041 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvtScaleAbs>`__




.. cfunction:: void cvCvtScaleAbs(const CvArr* src, CvArr* dst, double scale=1, double shift=0)

    Converts input array elements to another 8-bit unsigned integer with optional linear transformation.





    
    :param src: Source array 
    
    
    :param dst: Destination array (should have 8u depth) 
    
    
    :param scale: ScaleAbs factor 
    
    
    :param shift: Value added to the scaled source array elements 
    
    
    
The function is similar to 
:ref:`ConvertScale`
, but it stores absolute values of the conversion results:



.. math::

    \texttt{dst} (I) = | \texttt{scale} \texttt{src} (I) + ( \texttt{shift} _0, \texttt{shift} _1,...)| 


The function supports only destination arrays of 8u (8-bit unsigned integers) type; for other types the function can be emulated by a combination of 
:ref:`ConvertScale`
and 
:ref:`Abs`
functions.


.. index:: Copy

.. _Copy:

Copy
----

`id=0.347619260884 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Copy>`__




.. cfunction:: void cvCopy(const CvArr* src, CvArr* dst, const CvArr* mask=NULL)

    Copies one array to another.





    
    :param src: The source array 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function copies selected elements from an input array to an output array:



.. math::

    \texttt{dst} (I)= \texttt{src} (I)  \quad \text{if} \quad \texttt{mask} (I)  \ne 0. 


If any of the passed arrays is of 
``IplImage``
type, then its ROI
and COI fields are used. Both arrays must have the same type, the same
number of dimensions, and the same size. The function can also copy sparse
arrays (mask is not supported in this case).


.. index:: CountNonZero

.. _CountNonZero:

CountNonZero
------------

`id=0.58249377667 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CountNonZero>`__




.. cfunction:: int cvCountNonZero(const CvArr* arr)

    Counts non-zero array elements.





    
    :param arr: The array must be a single-channel array or a multi-channel image with COI set 
    
    
    
The function returns the number of non-zero elements in arr:



.. math::

    \sum _I ( \texttt{arr} (I)  \ne 0)  


In the case of 
``IplImage``
both ROI and COI are supported.



.. index:: CreateData

.. _CreateData:

CreateData
----------

`id=0.638669203593 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateData>`__




.. cfunction:: void cvCreateData(CvArr* arr)

    Allocates array data





    
    :param arr: Array header 
    
    
    
The function allocates image, matrix or
multi-dimensional array data. Note that in the case of matrix types OpenCV
allocation functions are used and in the case of IplImage they are used
unless 
``CV_TURN_ON_IPL_COMPATIBILITY``
was called. In the
latter case IPL functions are used to allocate the data.


.. index:: CreateImage

.. _CreateImage:

CreateImage
-----------

`id=0.0131648371818 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateImage>`__




.. cfunction:: IplImage* cvCreateImage(CvSize size, int depth, int channels)

    Creates an image header and allocates the image data.





    
    :param size: Image width and height 
    
    
    :param depth: Bit depth of image elements. See  :ref:`IplImage`  for valid depths. 
    
    
    :param channels: Number of channels per pixel. See  :ref:`IplImage`  for details. This function only creates images with interleaved channels. 
    
    
    
This call is a shortened form of



::


    
    header = cvCreateImageHeader(size, depth, channels);
    cvCreateData(header);
    

..


.. index:: CreateImageHeader

.. _CreateImageHeader:

CreateImageHeader
-----------------

`id=0.810135262232 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateImageHeader>`__




.. cfunction:: IplImage* cvCreateImageHeader(CvSize size, int depth, int channels)

    Creates an image header but does not allocate the image data.





    
    :param size: Image width and height 
    
    
    :param depth: Image depth (see  :ref:`CreateImage` ) 
    
    
    :param channels: Number of channels (see  :ref:`CreateImage` ) 
    
    
    
This call is an analogue of



::


    
    hdr=iplCreateImageHeader(channels, 0, depth,
                          channels == 1 ? "GRAY" : "RGB",
                          channels == 1 ? "GRAY" : channels == 3 ? "BGR" :
                          channels == 4 ? "BGRA" : "",
                          IPL_DATA_ORDER_PIXEL, IPL_ORIGIN_TL, 4,
                          size.width, size.height,
                          0,0,0,0);
    

..

but it does not use IPL functions by default (see the 
``CV_TURN_ON_IPL_COMPATIBILITY``
macro).

.. index:: CreateMat

.. _CreateMat:

CreateMat
---------

`id=0.590155166978 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateMat>`__




.. cfunction:: CvMat* cvCreateMat( int rows, int cols, int type)

    Creates a matrix header and allocates the matrix data. 





    
    :param rows: Number of rows in the matrix 
    
    
    :param cols: Number of columns in the matrix 
    
    
    :param type: The type of the matrix elements in the form  ``CV_<bit depth><S|U|F>C<number of channels>`` , where S=signed, U=unsigned, F=float. For example, CV _ 8UC1 means the elements are 8-bit unsigned and the there is 1 channel, and CV _ 32SC2 means the elements are 32-bit signed and there are 2 channels. 
    
    
    
This is the concise form for:




::


    
    CvMat* mat = cvCreateMatHeader(rows, cols, type);
    cvCreateData(mat);
    

..


.. index:: CreateMatHeader

.. _CreateMatHeader:

CreateMatHeader
---------------

`id=0.130473841629 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateMatHeader>`__




.. cfunction:: CvMat* cvCreateMatHeader( int rows, int cols, int type)

    Creates a matrix header but does not allocate the matrix data.





    
    :param rows: Number of rows in the matrix 
    
    
    :param cols: Number of columns in the matrix 
    
    
    :param type: Type of the matrix elements, see  :ref:`CreateMat` 
    
    
    
The function allocates a new matrix header and returns a pointer to it. The matrix data can then be allocated using 
:ref:`CreateData`
or set explicitly to user-allocated data via 
:ref:`SetData`
.


.. index:: CreateMatND

.. _CreateMatND:

CreateMatND
-----------

`id=0.0659656407287 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateMatND>`__




.. cfunction:: CvMatND* cvCreateMatND( int dims, const int* sizes, int type)

    Creates the header and allocates the data for a multi-dimensional dense array.





    
    :param dims: Number of array dimensions. This must not exceed CV _ MAX _ DIM (32 by default, but can be changed at build time). 
    
    
    :param sizes: Array of dimension sizes. 
    
    
    :param type: Type of array elements, see  :ref:`CreateMat` . 
    
    
    
This is a short form for:




::


    
    CvMatND* mat = cvCreateMatNDHeader(dims, sizes, type);
    cvCreateData(mat);
    

..


.. index:: CreateMatNDHeader

.. _CreateMatNDHeader:

CreateMatNDHeader
-----------------

`id=0.132772998614 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateMatNDHeader>`__




.. cfunction:: CvMatND* cvCreateMatNDHeader( int dims, const int* sizes, int type)

    Creates a new matrix header but does not allocate the matrix data.





    
    :param dims: Number of array dimensions 
    
    
    :param sizes: Array of dimension sizes 
    
    
    :param type: Type of array elements, see  :ref:`CreateMat` 
    
    
    
The function allocates a header for a multi-dimensional dense array. The array data can further be allocated using 
:ref:`CreateData`
or set explicitly to user-allocated data via 
:ref:`SetData`
.


.. index:: CreateSparseMat

.. _CreateSparseMat:

CreateSparseMat
---------------

`id=0.206464913947 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateSparseMat>`__




.. cfunction:: CvSparseMat* cvCreateSparseMat(int dims, const int* sizes, int type)

    Creates sparse array.





    
    :param dims: Number of array dimensions. In contrast to the dense matrix, the number of dimensions is practically unlimited (up to  :math:`2^{16}` ). 
    
    
    :param sizes: Array of dimension sizes 
    
    
    :param type: Type of array elements. The same as for CvMat 
    
    
    
The function allocates a multi-dimensional sparse array. Initially the array contain no elements, that is 
:ref:`Get`
or 
:ref:`GetReal`
returns zero for every index.

.. index:: CrossProduct

.. _CrossProduct:

CrossProduct
------------

`id=0.63082262592 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CrossProduct>`__




.. cfunction:: void cvCrossProduct(const CvArr* src1, const CvArr* src2, CvArr* dst)

    Calculates the cross product of two 3D vectors.





    
    :param src1: The first source vector 
    
    
    :param src2: The second source vector 
    
    
    :param dst: The destination vector 
    
    
    
The function calculates the cross product of two 3D vectors:



.. math::

    \texttt{dst} =  \texttt{src1} \times \texttt{src2} 


or:


.. math::

    \begin{array}{l} \texttt{dst} _1 =  \texttt{src1} _2  \texttt{src2} _3 -  \texttt{src1} _3  \texttt{src2} _2 \\ \texttt{dst} _2 =  \texttt{src1} _3  \texttt{src2} _1 -  \texttt{src1} _1  \texttt{src2} _3 \\ \texttt{dst} _3 =  \texttt{src1} _1  \texttt{src2} _2 -  \texttt{src1} _2  \texttt{src2} _1 \end{array} 



CvtPixToPlane
-------------


Synonym for 
:ref:`Split`
.


.. index:: DCT

.. _DCT:

DCT
---

`id=0.811976099826 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/DCT>`__




.. cfunction:: void cvDCT(const CvArr* src, CvArr* dst, int flags)

    Performs a forward or inverse Discrete Cosine transform of a 1D or 2D floating-point array.





    
    :param src: Source array, real 1D or 2D array 
    
    
    :param dst: Destination array of the same size and same type as the source 
    
    
    :param flags: Transformation flags, a combination of the following values 
         
            * **CV_DXT_FORWARD** do a forward 1D or 2D transform. 
            
            * **CV_DXT_INVERSE** do an inverse 1D or 2D transform. 
            
            * **CV_DXT_ROWS** do a forward or inverse transform of every individual row of the input matrix. This flag allows user to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself), to do 3D and higher-dimensional transforms and so forth. 
            
            
    
    
    
The function performs a forward or inverse transform of a 1D or 2D floating-point array:

Forward Cosine transform of 1D vector of 
:math:`N`
elements:


.. math::

    Y = C^{(N)}  \cdot X 


where


.. math::

    C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right ) 


and 
:math:`\alpha_0=1`
, 
:math:`\alpha_j=2`
for 
:math:`j > 0`
.

Inverse Cosine transform of 1D vector of N elements:


.. math::

    X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y 


(since 
:math:`C^{(N)}`
is orthogonal matrix, 
:math:`C^{(N)} \cdot \left(C^{(N)}\right)^T = I`
)

Forward Cosine transform of 2D 
:math:`M \times N`
matrix:


.. math::

    Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T 


Inverse Cosine transform of 2D vector of 
:math:`M \times N`
elements:


.. math::

    X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)} 



.. index:: DFT

.. _DFT:

DFT
---

`id=0.604521057934 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/DFT>`__




.. cfunction:: void cvDFT(const CvArr* src, CvArr* dst, int flags, int nonzeroRows=0)

    Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.





    
    :param src: Source array, real or complex 
    
    
    :param dst: Destination array of the same size and same type as the source 
    
    
    :param flags: Transformation flags, a combination of the following values 
         
            * **CV_DXT_FORWARD** do a forward 1D or 2D transform. The result is not scaled. 
            
            * **CV_DXT_INVERSE** do an inverse 1D or 2D transform. The result is not scaled.  ``CV_DXT_FORWARD``  and  ``CV_DXT_INVERSE``  are mutually exclusive, of course. 
            
            * **CV_DXT_SCALE** scale the result: divide it by the number of array elements. Usually, it is combined with  ``CV_DXT_INVERSE`` , and one may use a shortcut  ``CV_DXT_INV_SCALE`` . 
            
            * **CV_DXT_ROWS** do a forward or inverse transform of every individual row of the input matrix. This flag allows the user to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself), to do 3D and higher-dimensional transforms and so forth. 
            
            * **CV_DXT_INVERSE_SCALE** same as  ``CV_DXT_INVERSE + CV_DXT_SCALE`` 
            
            
    
    
    :param nonzeroRows: Number of nonzero rows in the source array
        (in the case of a forward 2d transform), or a number of rows of interest in
        the destination array (in the case of an inverse 2d transform). If the value
        is negative, zero, or greater than the total number of rows, it is
        ignored. The parameter can be used to speed up 2d convolution/correlation
        when computing via DFT. See the example below. 
    
    
    
The function performs a forward or inverse transform of a 1D or 2D floating-point array:


Forward Fourier transform of 1D vector of N elements:


.. math::

    y = F^{(N)}  \cdot x, where F^{(N)}_{jk}=exp(-i  \cdot 2 \pi \cdot j  \cdot k/N) 


, 


.. math::

    i=sqrt(-1) 


Inverse Fourier transform of 1D vector of N elements:


.. math::

    x'= (F^{(N)})^{-1}  \cdot y = conj(F^(N))  \cdot y
    x = (1/N)  \cdot x 


Forward Fourier transform of 2D vector of M 
:math:`\times`
N elements:


.. math::

    Y = F^{(M)}  \cdot X  \cdot F^{(N)} 


Inverse Fourier transform of 2D vector of M 
:math:`\times`
N elements:


.. math::

    X'= conj(F^{(M)})  \cdot Y  \cdot conj(F^{(N)})
    X = (1/(M  \cdot N))  \cdot X' 


In the case of real (single-channel) data, the packed format, borrowed from IPL, is used to represent the result of a forward Fourier transform or input for an inverse Fourier transform:



.. math::

    \begin{bmatrix} Re Y_{0,0} & Re Y_{0,1} & Im Y_{0,1} & Re Y_{0,2} & Im Y_{0,2} &  \cdots & Re Y_{0,N/2-1} & Im Y_{0,N/2-1} & Re Y_{0,N/2}  \\ Re Y_{1,0} & Re Y_{1,1} & Im Y_{1,1} & Re Y_{1,2} & Im Y_{1,2} &  \cdots & Re Y_{1,N/2-1} & Im Y_{1,N/2-1} & Re Y_{1,N/2}  \\ Im Y_{1,0} & Re Y_{2,1} & Im Y_{2,1} & Re Y_{2,2} & Im Y_{2,2} &  \cdots & Re Y_{2,N/2-1} & Im Y_{2,N/2-1} & Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &  Re Y_{M-3,1}  & Im Y_{M-3,1} &  \hdotsfor{3} & Re Y_{M-3,N/2-1} & Im Y_{M-3,N/2-1}& Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &  Re Y_{M-2,1}  & Im Y_{M-2,1} &  \hdotsfor{3} & Re Y_{M-2,N/2-1} & Im Y_{M-2,N/2-1}& Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &  Re Y_{M-1,1} &  Im Y_{M-1,1} &  \hdotsfor{3} & Re Y_{M-1,N/2-1} & Im Y_{M-1,N/2-1}& Re Y_{M/2,N/2} \end{bmatrix} 


Note: the last column is present if 
``N``
is even, the last row is present if 
``M``
is even.
In the case of 1D real transform the result looks like the first row of the above matrix.

Here is the example of how to compute 2D convolution using DFT.




::


    
    CvMat* A = cvCreateMat(M1, N1, CVg32F);
    CvMat* B = cvCreateMat(M2, N2, A->type);
    
    // it is also possible to have only abs(M2-M1)+1 times abs(N2-N1)+1
    // part of the full convolution result
    CvMat* conv = cvCreateMat(A->rows + B->rows - 1, A->cols + B->cols - 1, 
                               A->type);
    
    // initialize A and B
    ...
    
    int dftgM = cvGetOptimalDFTSize(A->rows + B->rows - 1);
    int dftgN = cvGetOptimalDFTSize(A->cols + B->cols - 1);
    
    CvMat* dftgA = cvCreateMat(dft_M, dft_N, A->type);
    CvMat* dftgB = cvCreateMat(dft_M, dft_N, B->type);
    CvMat tmp;
    
    // copy A to dftgA and pad dft_A with zeros
    cvGetSubRect(dftgA, &tmp, cvRect(0,0,A->cols,A->rows));
    cvCopy(A, &tmp);
    cvGetSubRect(dftgA, &tmp, cvRect(A->cols,0,dft_A->cols - A->cols,A->rows));
    cvZero(&tmp);
    // no need to pad bottom part of dftgA with zeros because of
    // use nonzerogrows parameter in cvDFT() call below
    
    cvDFT(dftgA, dft_A, CV_DXT_FORWARD, A->rows);
    
    // repeat the same with the second array
    cvGetSubRect(dftgB, &tmp, cvRect(0,0,B->cols,B->rows));
    cvCopy(B, &tmp);
    cvGetSubRect(dftgB, &tmp, cvRect(B->cols,0,dft_B->cols - B->cols,B->rows));
    cvZero(&tmp);
    // no need to pad bottom part of dftgB with zeros because of
    // use nonzerogrows parameter in cvDFT() call below
    
    cvDFT(dftgB, dft_B, CV_DXT_FORWARD, B->rows);
    
    cvMulSpectrums(dftgA, dft_B, dft_A, 0 /* or CV_DXT_MUL_CONJ to get 
                    correlation rather than convolution */);
    
    cvDFT(dftgA, dft_A, CV_DXT_INV_SCALE, conv->rows); // calculate only 
                                                             // the top part
    cvGetSubRect(dftgA, &tmp, cvRect(0,0,conv->cols,conv->rows));
    
    cvCopy(&tmp, conv);
    

..


.. index:: DecRefData

.. _DecRefData:

DecRefData
----------

`id=0.253923047171 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/DecRefData>`__




.. cfunction:: void cvDecRefData(CvArr* arr)

    Decrements an array data reference counter.





    
    :param arr: Pointer to an array header 
    
    
    
The function decrements the data reference counter in a 
:ref:`CvMat`
or
:ref:`CvMatND`
if the reference counter pointer
is not NULL. If the counter reaches zero, the data is deallocated. In the
current implementation the reference counter is not NULL only if the data
was allocated using the 
:ref:`CreateData`
function. The counter will be NULL in other cases such as:
external data was assigned to the header using 
:ref:`SetData`
, the matrix
header is part of a larger matrix or image, or the header was converted from an image or n-dimensional matrix header. 


.. index:: Det

.. _Det:

Det
---

`id=0.437350985322 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Det>`__




.. cfunction:: double cvDet(const CvArr* mat)

    Returns the determinant of a matrix.





    
    :param mat: The source matrix 
    
    
    
The function returns the determinant of the square matrix 
``mat``
. The direct method is used for small matrices and Gaussian elimination is used for larger matrices. For symmetric positive-determined matrices, it is also possible to run
:ref:`SVD`
with 
:math:`U = V = 0`
and then calculate the determinant as a product of the diagonal elements of 
:math:`W`
.


.. index:: Div

.. _Div:

Div
---

`id=0.781734526018 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Div>`__




.. cfunction:: void cvDiv(const CvArr* src1, const CvArr* src2, CvArr* dst, double scale=1)

    Performs per-element division of two arrays.





    
    :param src1: The first source array. If the pointer is NULL, the array is assumed to be all 1's. 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    :param scale: Optional scale factor 
    
    
    
The function divides one array by another:



.. math::

    \texttt{dst} (I)= \fork{\texttt{scale} \cdot \texttt{src1}(I)/\texttt{src2}(I)}{if \texttt{src1} is not \texttt{NULL}}{\texttt{scale}/\texttt{src2}(I)}{otherwise} 


All the arrays must have the same type and the same size (or ROI size).



.. index:: DotProduct

.. _DotProduct:

DotProduct
----------

`id=0.166249445191 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/DotProduct>`__




.. cfunction:: double cvDotProduct(const CvArr* src1, const CvArr* src2)

    Calculates the dot product of two arrays in Euclidian metrics.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    
The function calculates and returns the Euclidean dot product of two arrays.



.. math::

    src1  \bullet src2 =  \sum _I ( \texttt{src1} (I)  \texttt{src2} (I)) 


In the case of multiple channel arrays, the results for all channels are accumulated. In particular, 
``cvDotProduct(a,a)``
where 
``a``
is a complex vector, will return 
:math:`||\texttt{a}||^2`
.
The function can process multi-dimensional arrays, row by row, layer by layer, and so on.


.. index:: EigenVV

.. _EigenVV:

EigenVV
-------

`id=0.843871751283 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/EigenVV>`__




.. cfunction:: void cvEigenVV( CvArr* mat, CvArr* evects, CvArr* evals, double eps=0,  int lowindex = -1,  int highindex = -1)

    Computes eigenvalues and eigenvectors of a symmetric matrix.





    
    :param mat: The input symmetric square matrix, modified during the processing 
    
    
    :param evects: The output matrix of eigenvectors, stored as subsequent rows 
    
    
    :param evals: The output vector of eigenvalues, stored in the descending order (order of eigenvalues and eigenvectors is syncronized, of course) 
    
    
    :param eps: Accuracy of diagonalization. Typically,  ``DBL_EPSILON``  (about  :math:`10^{-15}` ) works well.
        THIS PARAMETER IS CURRENTLY IGNORED. 
    
    
    :param lowindex: Optional index of largest eigenvalue/-vector to calculate.
        (See below.) 
    
    
    :param highindex: Optional index of smallest eigenvalue/-vector to calculate.
        (See below.) 
    
    
    
The function computes the eigenvalues and eigenvectors of matrix 
``A``
:




::


    
    mat*evects(i,:)' = evals(i)*evects(i,:)' (in MATLAB notation)
    

..

If either low- or highindex is supplied the other is required, too.
Indexing is 0-based. Example: To calculate the largest eigenvector/-value set
``lowindex=highindex=0``
. To calculate all the eigenvalues, leave 
``lowindex=highindex=-1``
.
For legacy reasons this function always returns a square matrix the same size
as the source matrix with eigenvectors and a vector the length of the source
matrix with eigenvalues. The selected eigenvectors/-values are always in the
first highindex - lowindex + 1 rows.

The contents of matrix 
``A``
is destroyed by the function.

Currently the function is slower than 
:ref:`SVD`
yet less accurate,
so if 
``A``
is known to be positively-defined (for example, it
is a covariance matrix)it is recommended to use 
:ref:`SVD`
to find
eigenvalues and eigenvectors of 
``A``
, especially if eigenvectors
are not required.


.. index:: Exp

.. _Exp:

Exp
---

`id=0.027762297646 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Exp>`__




.. cfunction:: void cvExp(const CvArr* src, CvArr* dst)

    Calculates the exponent of every array element.





    
    :param src: The source array 
    
    
    :param dst: The destination array, it should have  ``double``  type or the same type as the source 
    
    
    
The function calculates the exponent of every element of the input array:



.. math::

    \texttt{dst} [I] = e^{ \texttt{src} (I)} 


The maximum relative error is about 
:math:`7 \times 10^{-6}`
. Currently, the function converts denormalized values to zeros on output.


.. index:: FastArctan

.. _FastArctan:

FastArctan
----------

`id=0.535136484735 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/FastArctan>`__




.. cfunction:: float cvFastArctan(float y, float x)

    Calculates the angle of a 2D vector.





    
    :param x: x-coordinate of 2D vector 
    
    
    :param y: y-coordinate of 2D vector 
    
    
    
The function calculates the full-range angle of an input 2D vector. The angle is 
measured in degrees and varies from 0 degrees to 360 degrees. The accuracy is about 0.1 degrees.


.. index:: Flip

.. _Flip:

Flip
----

`id=0.83697433441 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Flip>`__




.. cfunction:: void  cvFlip(const CvArr* src, CvArr* dst=NULL, int flipMode=0)

    Flip a 2D array around vertical, horizontal or both axes.





    
    :param src: Source array 
    
    
    :param dst: Destination array.
        If  :math:`\texttt{dst} = \texttt{NULL}`  the flipping is done in place. 
    
    
    :param flipMode: Specifies how to flip the array:
        0 means flipping around the x-axis, positive (e.g., 1) means flipping around y-axis, and negative (e.g., -1) means flipping around both axes. See also the discussion below for the formulas: 
    
    
    
The function flips the array in one of three different ways (row and column indices are 0-based):



.. math::

    dst(i,j) =  \forkthree{\texttt{src}(rows(\texttt{src})-i-1,j)}{if $\texttt{flipMode} = 0$}{\texttt{src}(i,cols(\texttt{src})-j-1)}{if $\texttt{flipMode} > 0$}{\texttt{src}(rows(\texttt{src})-i-1,cols(\texttt{src})-j-1)}{if $\texttt{flipMode} < 0$} 


The example scenarios of function use are:


    

*
    vertical flipping of the image (flipMode = 0) to switch between top-left and bottom-left image origin, which is a typical operation in video processing under Win32 systems.
      
    

*
    horizontal flipping of the image with subsequent horizontal shift and absolute difference calculation to check for a vertical-axis symmetry (flipMode 
    :math:`>`
    0)
      
    

*
    simultaneous horizontal and vertical flipping of the image with subsequent shift and absolute difference calculation to check for a central symmetry (flipMode 
    :math:`<`
    0)
      
    

*
    reversing the order of 1d point arrays (flipMode > 0)
    
    

.. index:: GEMM

.. _GEMM:

GEMM
----

`id=0.183074301558 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GEMM>`__




.. cfunction:: void cvGEMM( const CvArr* src1,  const CvArr* src2, double alpha,                const CvArr* src3,  double beta,  CvArr* dst,  int tABC=0)



.. cfunction:: \#define cvMatMulAdd(src1, src2, src3, dst ) cvGEMM(src1, src2, 1, src3, 1, dst, 0 )\#define cvMatMul(src1, src2, dst ) cvMatMulAdd(src1, src2, 0, dst )

    Performs generalized matrix multiplication.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param src3: The third source array (shift). Can be NULL, if there is no shift. 
    
    
    :param dst: The destination array 
    
    
    :param tABC: The operation flags that can be 0 or a combination of the following values 
         
            * **CV_GEMM_A_T** transpose src1 
            
            * **CV_GEMM_B_T** transpose src2 
            
            * **CV_GEMM_C_T** transpose src3 
            
            
        
        For example,  ``CV_GEMM_A_T+CV_GEMM_C_T``  corresponds to 
        
        .. math::
        
            \texttt{alpha}   \,   \texttt{src1}  ^T  \,   \texttt{src2}  +  \texttt{beta}   \,   \texttt{src3}  ^T 
        
        
    
    
    
The function performs generalized matrix multiplication:



.. math::

    \texttt{dst} =  \texttt{alpha} \, op( \texttt{src1} )  \, op( \texttt{src2} ) +  \texttt{beta} \, op( \texttt{src3} )  \quad \text{where $op(X)$ is $X$ or $X^T$} 


All the matrices should have the same data type and coordinated sizes. Real or complex floating-point matrices are supported.


.. index:: Get?D

.. _Get?D:

Get?D
-----

`id=0.996029550845 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Get%3FD>`__




.. cfunction:: CvScalar cvGet1D(const CvArr* arr, int idx0) CvScalar cvGet2D(const CvArr* arr, int idx0, int idx1) CvScalar cvGet3D(const CvArr* arr, int idx0, int idx1, int idx2) CvScalar cvGetND(const CvArr* arr, int* idx)

    Return a specific array element.





    
    :param arr: Input array 
    
    
    :param idx0: The first zero-based component of the element index 
    
    
    :param idx1: The second zero-based component of the element index 
    
    
    :param idx2: The third zero-based component of the element index 
    
    
    :param idx: Array of the element indices 
    
    
    
The functions return a specific array element. In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).

.. index:: GetCol(s)

.. _GetCol(s):

GetCol(s)
---------

`id=0.311656091229 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetCol%28s%29>`__




.. cfunction:: CvMat* cvGetCol(const CvArr* arr, CvMat* submat, int col)

    Returns array column or column span.





.. cfunction:: CvMat* cvGetCols(const CvArr* arr, CvMat* submat, int startCol, int endCol)

    




    
    :param arr: Input array 
    
    
    :param submat: Pointer to the resulting sub-array header 
    
    
    :param col: Zero-based index of the selected column 
    
    
    :param startCol: Zero-based index of the starting column (inclusive) of the span 
    
    
    :param endCol: Zero-based index of the ending column (exclusive) of the span 
    
    
    
The functions 
``GetCol``
and 
``GetCols``
return the header, corresponding to a specified column span of the input array. 
``GetCol``
is a shortcut for 
:ref:`GetCols`
:




::


    
    cvGetCol(arr, submat, col); // ~ cvGetCols(arr, submat, col, col + 1);
    

..


.. index:: GetDiag

.. _GetDiag:

GetDiag
-------

`id=0.851887559121 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetDiag>`__




.. cfunction:: CvMat* cvGetDiag(const CvArr* arr, CvMat* submat, int diag=0)

    Returns one of array diagonals.





    
    :param arr: Input array 
    
    
    :param submat: Pointer to the resulting sub-array header 
    
    
    :param diag: Array diagonal. Zero corresponds to the main diagonal, -1 corresponds to the diagonal above the main , 1 corresponds to the diagonal below the main, and so forth. 
    
    
    
The function returns the header, corresponding to a specified diagonal of the input array.


cvGetDims, cvGetDimSize
-----------------------


Return number of array dimensions and their sizes or the size of a particular dimension.



.. cfunction:: int cvGetDims(const CvArr* arr, int* sizes=NULL)

    




.. cfunction:: int cvGetDimSize(const CvArr* arr, int index)

    




    
    :param arr: Input array 
    
    
    :param sizes: Optional output vector of the array dimension sizes. For
        2d arrays the number of rows (height) goes first, number of columns
        (width) next. 
    
    
    :param index: Zero-based dimension index (for matrices 0 means number
        of rows, 1 means number of columns; for images 0 means height, 1 means
        width) 
    
    
    
The function 
``cvGetDims``
returns the array dimensionality and the
array of dimension sizes. In the case of 
``IplImage``
or 
:ref:`CvMat`
it always
returns 2 regardless of number of image/matrix rows. The function
``cvGetDimSize``
returns the particular dimension size (number of
elements per that dimension). For example, the following code calculates
total number of array elements in two ways:




::


    
    // via cvGetDims()
    int sizes[CV_MAX_DIM];
    int i, total = 1;
    int dims = cvGetDims(arr, size);
    for(i = 0; i < dims; i++ )
        total *= sizes[i];
    
    // via cvGetDims() and cvGetDimSize()
    int i, total = 1;
    int dims = cvGetDims(arr);
    for(i = 0; i < dims; i++ )
        total *= cvGetDimsSize(arr, i);
    

..


.. index:: GetElemType

.. _GetElemType:

GetElemType
-----------

`id=0.664874075316 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetElemType>`__




.. cfunction:: int cvGetElemType(const CvArr* arr)

    Returns type of array elements.





    
    :param arr: Input array 
    
    
    
The function returns type of the array elements
as described in 
:ref:`CreateMat`
discussion: 
``CV_8UC1``
... 
``CV_64FC4``
.



.. index:: GetImage

.. _GetImage:

GetImage
--------

`id=0.868367677778 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetImage>`__




.. cfunction:: IplImage* cvGetImage(const CvArr* arr, IplImage* imageHeader)

    Returns image header for arbitrary array.





    
    :param arr: Input array 
    
    
    :param imageHeader: Pointer to  ``IplImage``  structure used as a temporary buffer 
    
    
    
The function returns the image header for the input array
that can be a matrix - 
:ref:`CvMat`
, or an image - 
``IplImage*``
. In
the case of an image the function simply returns the input pointer. In the
case of 
:ref:`CvMat`
it initializes an 
``imageHeader``
structure
with the parameters of the input matrix. Note that if we transform
``IplImage``
to 
:ref:`CvMat`
and then transform CvMat back to
IplImage, we can get different headers if the ROI is set, and thus some
IPL functions that calculate image stride from its width and align may
fail on the resultant image.


.. index:: GetImageCOI

.. _GetImageCOI:

GetImageCOI
-----------

`id=0.280055789523 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetImageCOI>`__




.. cfunction:: int cvGetImageCOI(const IplImage* image)

    Returns the index of the channel of interest. 





    
    :param image: A pointer to the image header 
    
    
    
Returns the channel of interest of in an IplImage. Returned values correspond to the 
``coi``
in 
:ref:`SetImageCOI`
.


.. index:: GetImageROI

.. _GetImageROI:

GetImageROI
-----------

`id=0.762224588004 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetImageROI>`__




.. cfunction:: CvRect cvGetImageROI(const IplImage* image)

    Returns the image ROI.





    
    :param image: A pointer to the image header 
    
    
    
If there is no ROI set, 
``cvRect(0,0,image->width,image->height)``
is returned.


.. index:: GetMat

.. _GetMat:

GetMat
------

`id=0.492159925052 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetMat>`__




.. cfunction:: CvMat* cvGetMat(const CvArr* arr, CvMat* header, int* coi=NULL, int allowND=0)

    Returns matrix header for arbitrary array.





    
    :param arr: Input array 
    
    
    :param header: Pointer to  :ref:`CvMat`  structure used as a temporary buffer 
    
    
    :param coi: Optional output parameter for storing COI 
    
    
    :param allowND: If non-zero, the function accepts multi-dimensional dense arrays (CvMatND*) and returns 2D (if CvMatND has two dimensions) or 1D matrix (when CvMatND has 1 dimension or more than 2 dimensions). The array must be continuous. 
    
    
    
The function returns a matrix header for the input array that can be a matrix - 

:ref:`CvMat`
, an image - 
``IplImage``
or a multi-dimensional dense array - 
:ref:`CvMatND`
(latter case is allowed only if 
``allowND != 0``
) . In the case of matrix the function simply returns the input pointer. In the case of 
``IplImage*``
or 
:ref:`CvMatND`
it initializes the 
``header``
structure with parameters of the current image ROI and returns the pointer to this temporary structure. Because COI is not supported by 
:ref:`CvMat`
, it is returned separately.

The function provides an easy way to handle both types of arrays - 
``IplImage``
and 
:ref:`CvMat`
- using the same code. Reverse transform from 
:ref:`CvMat`
to 
``IplImage``
can be done using the 
:ref:`GetImage`
function.

Input array must have underlying data allocated or attached, otherwise the function fails.

If the input array is 
``IplImage``
with planar data layout and COI set, the function returns the pointer to the selected plane and COI = 0. It enables per-plane processing of multi-channel images with planar data layout using OpenCV functions.


.. index:: GetNextSparseNode

.. _GetNextSparseNode:

GetNextSparseNode
-----------------

`id=0.693142857428 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetNextSparseNode>`__




.. cfunction:: CvSparseNode* cvGetNextSparseNode(CvSparseMatIterator* matIterator)

    Returns the next sparse matrix element





    
    :param matIterator: Sparse array iterator 
    
    
    
The function moves iterator to the next sparse matrix element and returns pointer to it. In the current version there is no any particular order of the elements, because they are stored in the hash table. The sample below demonstrates how to iterate through the sparse matrix:

Using 
:ref:`InitSparseMatIterator`
and 
:ref:`GetNextSparseNode`
to calculate sum of floating-point sparse array.




::


    
    double sum;
    int i, dims = cvGetDims(array);
    CvSparseMatIterator mat_iterator;
    CvSparseNode* node = cvInitSparseMatIterator(array, &mat_iterator);
    
    for(; node != 0; node = cvGetNextSparseNode(&mat_iterator ))
    {
        /* get pointer to the element indices */
        int* idx = CV_NODE_IDX(array, node);
        /* get value of the element (assume that the type is CV_32FC1) */
        float val = *(float*)CV_NODE_VAL(array, node);
        printf("(");
        for(i = 0; i < dims; i++ )
            printf("
        printf("
    
        sum += val;
    }
    
    printf("nTotal sum = 
    

..


.. index:: GetOptimalDFTSize

.. _GetOptimalDFTSize:

GetOptimalDFTSize
-----------------

`id=0.773925667267 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetOptimalDFTSize>`__




.. cfunction:: int cvGetOptimalDFTSize(int size0)

    Returns optimal DFT size for a given vector size.





    
    :param size0: Vector size 
    
    
    
The function returns the minimum number
``N``
that is greater than or equal to 
``size0``
, such that the DFT
of a vector of size 
``N``
can be computed fast. In the current
implementation 
:math:`N=2^p \times 3^q \times 5^r`
, for some 
:math:`p`
, 
:math:`q`
, 
:math:`r`
.

The function returns a negative number if 
``size0``
is too large
(very close to 
``INT_MAX``
)



.. index:: GetRawData

.. _GetRawData:

GetRawData
----------

`id=0.0637610069522 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetRawData>`__




.. cfunction:: void cvGetRawData(const CvArr* arr, uchar** data,                    int* step=NULL, CvSize* roiSize=NULL)

    Retrieves low-level information about the array.





    
    :param arr: Array header 
    
    
    :param data: Output pointer to the whole image origin or ROI origin if ROI is set 
    
    
    :param step: Output full row length in bytes 
    
    
    :param roiSize: Output ROI size 
    
    
    
The function fills output variables with low-level information about the array data. All output parameters are optional, so some of the pointers may be set to 
``NULL``
. If the array is 
``IplImage``
with ROI set, the parameters of ROI are returned.

The following example shows how to get access to array elements. GetRawData calculates the absolute value of the elements in a single-channel, floating-point array.




::


    
    float* data;
    int step;
    
    CvSize size;
    int x, y;
    
    cvGetRawData(array, (uchar**)&data, &step, &size);
    step /= sizeof(data[0]);
    
    for(y = 0; y < size.height; y++, data += step )
        for(x = 0; x < size.width; x++ )
            data[x] = (float)fabs(data[x]);
    
    

..


.. index:: GetReal1D

.. _GetReal1D:

GetReal1D
---------

`id=0.946925134724 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetReal1D>`__




.. cfunction:: double cvGetReal1D(const CvArr* arr, int idx0)

    Return a specific element of single-channel 1D array.





    
    :param arr: Input array. Must have a single channel. 
    
    
    :param idx0: The first zero-based component of the element index 
    
    
    
Returns a specific element of a single-channel array. If the array has
multiple channels, a runtime error is raised. Note that 
:ref:`Get`
function can be used safely for both single-channel and multiple-channel
arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).


.. index:: GetReal2D

.. _GetReal2D:

GetReal2D
---------

`id=0.949131529933 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetReal2D>`__




.. cfunction:: double cvGetReal2D(const CvArr* arr, int idx0, int idx1)

    Return a specific element of single-channel 2D array.





    
    :param arr: Input array. Must have a single channel. 
    
    
    :param idx0: The first zero-based component of the element index 
    
    
    :param idx1: The second zero-based component of the element index 
    
    
    
Returns a specific element of a single-channel array. If the array has
multiple channels, a runtime error is raised. Note that 
:ref:`Get`
function can be used safely for both single-channel and multiple-channel
arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).


.. index:: GetReal3D

.. _GetReal3D:

GetReal3D
---------

`id=0.0143815925526 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetReal3D>`__




.. cfunction:: double cvGetReal3D(const CvArr* arr, int idx0, int idx1, int idx2)

    Return a specific element of single-channel array.





    
    :param arr: Input array. Must have a single channel. 
    
    
    :param idx0: The first zero-based component of the element index 
    
    
    :param idx1: The second zero-based component of the element index 
    
    
    :param idx2: The third zero-based component of the element index 
    
    
    
Returns a specific element of a single-channel array. If the array has
multiple channels, a runtime error is raised. Note that 
:ref:`Get`
function can be used safely for both single-channel and multiple-channel
arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).


.. index:: GetRealND

.. _GetRealND:

GetRealND
---------

`id=0.276521262331 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetRealND>`__




.. cfunction:: double cvGetRealND(const CvArr* arr, int* idx)->float

    Return a specific element of single-channel array.





    
    :param arr: Input array. Must have a single channel. 
    
    
    :param idx: Array of the element indices 
    
    
    
Returns a specific element of a single-channel array. If the array has
multiple channels, a runtime error is raised. Note that 
:ref:`Get`
function can be used safely for both single-channel and multiple-channel
arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).



.. index:: GetRow(s)

.. _GetRow(s):

GetRow(s)
---------

`id=0.355110492705 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetRow%28s%29>`__




.. cfunction:: CvMat* cvGetRow(const CvArr* arr, CvMat* submat, int row)

    Returns array row or row span.





.. cfunction:: CvMat* cvGetRows(const CvArr* arr, CvMat* submat, int startRow, int endRow, int deltaRow=1)

    




    
    :param arr: Input array 
    
    
    :param submat: Pointer to the resulting sub-array header 
    
    
    :param row: Zero-based index of the selected row 
    
    
    :param startRow: Zero-based index of the starting row (inclusive) of the span 
    
    
    :param endRow: Zero-based index of the ending row (exclusive) of the span 
    
    
    :param deltaRow: Index step in the row span. That is, the function extracts every  ``deltaRow`` -th row from  ``startRow``  and up to (but not including)  ``endRow`` . 
    
    
    
The functions return the header, corresponding to a specified row/row span of the input array. Note that 
``GetRow``
is a shortcut for 
:ref:`GetRows`
:




::


    
    cvGetRow(arr, submat, row ) ~ cvGetRows(arr, submat, row, row + 1, 1);
    

..


.. index:: GetSize

.. _GetSize:

GetSize
-------

`id=0.248625107219 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetSize>`__




.. cfunction:: CvSize cvGetSize(const CvArr* arr)

    Returns size of matrix or image ROI.





    
    :param arr: array header 
    
    
    
The function returns number of rows (CvSize::height) and number of columns (CvSize::width) of the input matrix or image. In the case of image the size of ROI is returned.



.. index:: GetSubRect

.. _GetSubRect:

GetSubRect
----------

`id=0.0482029723737 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetSubRect>`__




.. cfunction:: CvMat* cvGetSubRect(const CvArr* arr, CvMat* submat, CvRect rect)

    Returns matrix header corresponding to the rectangular sub-array of input image or matrix.





    
    :param arr: Input array 
    
    
    :param submat: Pointer to the resultant sub-array header 
    
    
    :param rect: Zero-based coordinates of the rectangle of interest 
    
    
    
The function returns header, corresponding to
a specified rectangle of the input array. In other words, it allows
the user to treat a rectangular part of input array as a stand-alone
array. ROI is taken into account by the function so the sub-array of
ROI is actually extracted.


.. index:: InRange

.. _InRange:

InRange
-------

`id=0.549621347828 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InRange>`__




.. cfunction:: void cvInRange(const CvArr* src, const CvArr* lower, const CvArr* upper, CvArr* dst)

    Checks that array elements lie between the elements of two other arrays.





    
    :param src: The first source array 
    
    
    :param lower: The inclusive lower boundary array 
    
    
    :param upper: The exclusive upper boundary array 
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    
    
The function does the range check for every element of the input array:



.. math::

    \texttt{dst} (I)= \texttt{lower} (I)_0 <=  \texttt{src} (I)_0 <  \texttt{upper} (I)_0 


For single-channel arrays,



.. math::

    \texttt{dst} (I)= \texttt{lower} (I)_0 <=  \texttt{src} (I)_0 <  \texttt{upper} (I)_0  \land \texttt{lower} (I)_1 <=  \texttt{src} (I)_1 <  \texttt{upper} (I)_1 


For two-channel arrays and so forth,

dst(I) is set to 0xff (all 
``1``
-bits) if src(I) is within the range and 0 otherwise. All the arrays must have the same type, except the destination, and the same size (or ROI size).



.. index:: InRangeS

.. _InRangeS:

InRangeS
--------

`id=0.194953788625 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InRangeS>`__




.. cfunction:: void cvInRangeS(const CvArr* src, CvScalar lower, CvScalar upper, CvArr* dst)

    Checks that array elements lie between two scalars.





    
    :param src: The first source array 
    
    
    :param lower: The inclusive lower boundary 
    
    
    :param upper: The exclusive upper boundary 
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    
    
The function does the range check for every element of the input array:



.. math::

    \texttt{dst} (I)= \texttt{lower} _0 <=  \texttt{src} (I)_0 <  \texttt{upper} _0 


For single-channel arrays,



.. math::

    \texttt{dst} (I)= \texttt{lower} _0 <=  \texttt{src} (I)_0 <  \texttt{upper} _0  \land \texttt{lower} _1 <=  \texttt{src} (I)_1 <  \texttt{upper} _1 


For two-channel arrays nd so forth,

'dst(I)' is set to 0xff (all 
``1``
-bits) if 'src(I)' is within the range and 0 otherwise. All the arrays must have the same size (or ROI size).


.. index:: IncRefData

.. _IncRefData:

IncRefData
----------

`id=0.0936060506247 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/IncRefData>`__




.. cfunction:: int cvIncRefData(CvArr* arr)

    Increments array data reference counter.





    
    :param arr: Array header 
    
    
    
The function increments 
:ref:`CvMat`
or
:ref:`CvMatND`
data reference counter and returns the new counter value
if the reference counter pointer is not NULL, otherwise it returns zero.


.. index:: InitImageHeader

.. _InitImageHeader:

InitImageHeader
---------------

`id=0.742068243947 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InitImageHeader>`__




.. cfunction:: IplImage* cvInitImageHeader( IplImage* image, CvSize size, int depth, int channels, int origin=0, int align=4)

    Initializes an image header that was previously allocated.





    
    :param image: Image header to initialize 
    
    
    :param size: Image width and height 
    
    
    :param depth: Image depth (see  :ref:`CreateImage` ) 
    
    
    :param channels: Number of channels (see  :ref:`CreateImage` ) 
    
    
    :param origin: Top-left  ``IPL_ORIGIN_TL``  or bottom-left  ``IPL_ORIGIN_BL`` 
    
    
    :param align: Alignment for image rows, typically 4 or 8 bytes 
    
    
    
The returned 
``IplImage*``
points to the initialized header.


.. index:: InitMatHeader

.. _InitMatHeader:

InitMatHeader
-------------

`id=0.656867541884 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InitMatHeader>`__




.. cfunction:: CvMat* cvInitMatHeader( CvMat* mat, int rows, int cols, int type,  void* data=NULL, int step=CV_AUTOSTEP)

    Initializes a pre-allocated matrix header.





    
    :param mat: A pointer to the matrix header to be initialized 
    
    
    :param rows: Number of rows in the matrix 
    
    
    :param cols: Number of columns in the matrix 
    
    
    :param type: Type of the matrix elements, see  :ref:`CreateMat` . 
    
    
    :param data: Optional: data pointer assigned to the matrix header 
    
    
    :param step: Optional: full row width in bytes of the assigned data. By default, the minimal possible step is used which assumes there are no gaps between subsequent rows of the matrix. 
    
    
    
This function is often used to process raw data with OpenCV matrix functions. For example, the following code computes the matrix product of two matrices, stored as ordinary arrays:




::


    
    double a[] = { 1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12 };
    
    double b[] = { 1, 5, 9,
                   2, 6, 10,
                   3, 7, 11,
                   4, 8, 12 };
    
    double c[9];
    CvMat Ma, Mb, Mc ;
    
    cvInitMatHeader(&Ma, 3, 4, CV_64FC1, a);
    cvInitMatHeader(&Mb, 4, 3, CV_64FC1, b);
    cvInitMatHeader(&Mc, 3, 3, CV_64FC1, c);
    
    cvMatMulAdd(&Ma, &Mb, 0, &Mc);
    // the c array now contains the product of a (3x4) and b (4x3)
    
    

..


.. index:: InitMatNDHeader

.. _InitMatNDHeader:

InitMatNDHeader
---------------

`id=0.422685627081 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InitMatNDHeader>`__




.. cfunction:: CvMatND* cvInitMatNDHeader( CvMatND* mat, int dims, const int* sizes, int type, void* data=NULL)

    Initializes a pre-allocated multi-dimensional array header.





    
    :param mat: A pointer to the array header to be initialized 
    
    
    :param dims: The number of array dimensions 
    
    
    :param sizes: An array of dimension sizes 
    
    
    :param type: Type of array elements, see  :ref:`CreateMat` 
    
    
    :param data: Optional data pointer assigned to the matrix header 
    
    
    

.. index:: InitSparseMatIterator

.. _InitSparseMatIterator:

InitSparseMatIterator
---------------------

`id=0.201070631416 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InitSparseMatIterator>`__




.. cfunction:: CvSparseNode* cvInitSparseMatIterator(const CvSparseMat* mat,                                        CvSparseMatIterator* matIterator)

    Initializes sparse array elements iterator.





    
    :param mat: Input array 
    
    
    :param matIterator: Initialized iterator 
    
    
    
The function initializes iterator of
sparse array elements and returns pointer to the first element, or NULL
if the array is empty.


.. index:: InvSqrt

.. _InvSqrt:

InvSqrt
-------

`id=0.80254392991 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InvSqrt>`__




.. cfunction:: float cvInvSqrt(float value)

    Calculates the inverse square root.





    
    :param value: The input floating-point value 
    
    
    
The function calculates the inverse square root of the argument, and normally it is faster than 
``1./sqrt(value)``
. If the argument is zero or negative, the result is not determined. Special values (
:math:`\pm \infty`
, NaN) are not handled.


.. index:: Inv

.. _Inv:

Inv
---

`id=0.303857308817 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Inv>`__


:ref:`Invert`

.. index:: 

.. _:




`id=0.780643675122 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/>`__




.. cfunction:: double cvInvert(const CvArr* src, CvArr* dst, int method=CV_LU)

    Finds the inverse or pseudo-inverse of a matrix.





    
    :param src: The source matrix 
    
    
    :param dst: The destination matrix 
    
    
    :param method: Inversion method 
        
               
            * **CV_LU** Gaussian elimination with optimal pivot element chosen 
            
              
            * **CV_SVD** Singular value decomposition (SVD) method 
            
              
            * **CV_SVD_SYM** SVD method for a symmetric positively-defined matrix 
            
            
    
    
    
The function inverts matrix 
``src1``
and stores the result in 
``src2``
.

In the case of 
``LU``
method, the function returns the 
``src1``
determinant (src1 must be square). If it is 0, the matrix is not inverted and 
``src2``
is filled with zeros.

In the case of 
``SVD``
methods, the function returns the inversed condition of 
``src1``
(ratio of the smallest singular value to the largest singular value) and 0 if 
``src1``
is all zeros. The SVD methods calculate a pseudo-inverse matrix if 
``src1``
is singular.



.. index:: IsInf

.. _IsInf:

IsInf
-----

`id=0.308846865611 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/IsInf>`__




.. cfunction:: int cvIsInf(double value)

    Determines if the argument is Infinity.





    
    :param value: The input floating-point value 
    
    
    
The function returns 1 if the argument is 
:math:`\pm \infty`
(as defined by IEEE754 standard), 0 otherwise.


.. index:: IsNaN

.. _IsNaN:

IsNaN
-----

`id=0.651061735514 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/IsNaN>`__




.. cfunction:: int cvIsNaN(double value)

    Determines if the argument is Not A Number.





    
    :param value: The input floating-point value 
    
    
    
The function returns 1 if the argument is Not A Number (as defined by IEEE754 standard), 0 otherwise.



.. index:: LUT

.. _LUT:

LUT
---

`id=0.987743314885 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/LUT>`__




.. cfunction:: void cvLUT(const CvArr* src, CvArr* dst, const CvArr* lut)

    Performs a look-up table transform of an array.





    
    :param src: Source array of 8-bit elements 
    
    
    :param dst: Destination array of a given depth and of the same number of channels as the source array 
    
    
    :param lut: Look-up table of 256 elements; should have the same depth as the destination array. In the case of multi-channel source and destination arrays, the table should either have a single-channel (in this case the same table is used for all channels) or the same number of channels as the source/destination array. 
    
    
    
The function fills the destination array with values from the look-up table. Indices of the entries are taken from the source array. That is, the function processes each element of 
``src``
as follows:



.. math::

    \texttt{dst} _i  \leftarrow \texttt{lut} _{ \texttt{src} _i + d} 


where



.. math::

    d =  \fork{0}{if \texttt{src} has depth \texttt{CV\_8U}}{128}{if \texttt{src} has depth \texttt{CV\_8S}} 



.. index:: Log

.. _Log:

Log
---

`id=0.367129782627 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Log>`__




.. cfunction:: void cvLog(const CvArr* src, CvArr* dst)

    Calculates the natural logarithm of every array element's absolute value.





    
    :param src: The source array 
    
    
    :param dst: The destination array, it should have  ``double``  type or the same type as the source 
    
    
    
The function calculates the natural logarithm of the absolute value of every element of the input array:



.. math::

    \texttt{dst} [I] =  \fork{\log{|\texttt{src}(I)}}{if $\texttt{src}[I] \ne 0$ }{\texttt{C}}{otherwise} 


Where 
``C``
is a large negative number (about -700 in the current implementation).


.. index:: Mahalanobis

.. _Mahalanobis:

Mahalanobis
-----------

`id=0.146686782784 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Mahalanobis>`__




.. cfunction:: double cvMahalanobis( const CvArr* vec1, const CvArr* vec2, CvArr* mat)

    Calculates the Mahalanobis distance between two vectors.





    
    :param vec1: The first 1D source vector 
    
    
    :param vec2: The second 1D source vector 
    
    
    :param mat: The inverse covariance matrix 
    
    
    
The function calculates and returns the weighted distance between two vectors:



.. math::

    d( \texttt{vec1} , \texttt{vec2} )= \sqrt{\sum_{i,j}{\texttt{icovar(i,j)}\cdot(\texttt{vec1}(I)-\texttt{vec2}(I))\cdot(\texttt{vec1(j)}-\texttt{vec2(j)})} } 


The covariance matrix may be calculated using the 
:ref:`CalcCovarMatrix`
function and further inverted using the 
:ref:`Invert`
function (CV
_
SVD method is the prefered one because the matrix might be singular).



.. index:: Mat

.. _Mat:

Mat
---

`id=0.921640300869 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Mat>`__




.. cfunction:: CvMat cvMat( int rows, int cols, int type, void* data=NULL)

    Initializes matrix header (lightweight variant).





    
    :param rows: Number of rows in the matrix 
    
    
    :param cols: Number of columns in the matrix 
    
    
    :param type: Type of the matrix elements - see  :ref:`CreateMat` 
    
    
    :param data: Optional data pointer assigned to the matrix header 
    
    
    
Initializes a matrix header and assigns data to it. The matrix is filled 
*row*
-wise (the first 
``cols``
elements of data form the first row of the matrix, etc.)

This function is a fast inline substitution for 
:ref:`InitMatHeader`
. Namely, it is equivalent to:




::


    
    CvMat mat;
    cvInitMatHeader(&mat, rows, cols, type, data, CV_AUTOSTEP);
    

..


.. index:: Max

.. _Max:

Max
---

`id=0.802320083613 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Max>`__




.. cfunction:: void cvMax(const CvArr* src1, const CvArr* src2, CvArr* dst)

    Finds per-element maximum of two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    
The function calculates per-element maximum of two arrays:



.. math::

    \texttt{dst} (I)= \max ( \texttt{src1} (I),  \texttt{src2} (I)) 


All the arrays must have a single channel, the same data type and the same size (or ROI size).



.. index:: MaxS

.. _MaxS:

MaxS
----

`id=0.981553315291 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MaxS>`__




.. cfunction:: void cvMaxS(const CvArr* src, double value, CvArr* dst)

    Finds per-element maximum of array and scalar.





    
    :param src: The first source array 
    
    
    :param value: The scalar value 
    
    
    :param dst: The destination array 
    
    
    
The function calculates per-element maximum of array and scalar:



.. math::

    \texttt{dst} (I)= \max ( \texttt{src} (I),  \texttt{value} ) 


All the arrays must have a single channel, the same data type and the same size (or ROI size).



.. index:: Merge

.. _Merge:

Merge
-----

`id=0.57803259893 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Merge>`__




.. cfunction:: void cvMerge(const CvArr* src0, const CvArr* src1,               const CvArr* src2, const CvArr* src3, CvArr* dst)

    Composes a multi-channel array from several single-channel arrays or inserts a single channel into the array.






::


    
    #define cvCvtPlaneToPix cvMerge
    

..



    
    :param src0: Input channel 0 
    
    
    :param src1: Input channel 1 
    
    
    :param src2: Input channel 2 
    
    
    :param src3: Input channel 3 
    
    
    :param dst: Destination array 
    
    
    
The function is the opposite to 
:ref:`Split`
. If the destination array has N channels then if the first N input channels are not NULL, they all are copied to the destination array; if only a single source channel of the first N is not NULL, this particular channel is copied into the destination array; otherwise an error is raised. The rest of the source channels (beyond the first N) must always be NULL. For IplImage 
:ref:`Copy`
with COI set can be also used to insert a single channel into the image.


.. index:: Min

.. _Min:

Min
---

`id=0.696669339505 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Min>`__




.. cfunction:: void cvMin(const CvArr* src1, const CvArr* src2, CvArr* dst)

    Finds per-element minimum of two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    
The function calculates per-element minimum of two arrays:



.. math::

    \texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I)) 


All the arrays must have a single channel, the same data type and the same size (or ROI size).



.. index:: MinMaxLoc

.. _MinMaxLoc:

MinMaxLoc
---------

`id=0.836639641988 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MinMaxLoc>`__




.. cfunction:: void cvMinMaxLoc(const CvArr* arr, double* minVal, double* maxVal,                   CvPoint* minLoc=NULL, CvPoint* maxLoc=NULL, const CvArr* mask=NULL)

    Finds global minimum and maximum in array or subarray.





    
    :param arr: The source array, single-channel or multi-channel with COI set 
    
    
    :param minVal: Pointer to returned minimum value 
    
    
    :param maxVal: Pointer to returned maximum value 
    
    
    :param minLoc: Pointer to returned minimum location 
    
    
    :param maxLoc: Pointer to returned maximum location 
    
    
    :param mask: The optional mask used to select a subarray 
    
    
    
The function finds minimum and maximum element values
and their positions. The extremums are searched across the whole array,
selected 
``ROI``
(in the case of 
``IplImage``
) or, if 
``mask``
is not 
``NULL``
, in the specified array region. If the array has
more than one channel, it must be 
``IplImage``
with 
``COI``
set. In the case of multi-dimensional arrays, 
``minLoc->x``
and 
``maxLoc->x``
will contain raw (linear) positions of the extremums.


.. index:: MinS

.. _MinS:

MinS
----

`id=0.476843407849 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MinS>`__




.. cfunction:: void cvMinS(const CvArr* src, double value, CvArr* dst)

    Finds per-element minimum of an array and a scalar.





    
    :param src: The first source array 
    
    
    :param value: The scalar value 
    
    
    :param dst: The destination array 
    
    
    
The function calculates minimum of an array and a scalar:



.. math::

    \texttt{dst} (I)= \min ( \texttt{src} (I),  \texttt{value} ) 


All the arrays must have a single channel, the same data type and the same size (or ROI size).



Mirror
------


Synonym for 
:ref:`Flip`
.


.. index:: MixChannels

.. _MixChannels:

MixChannels
-----------

`id=0.147282411501 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MixChannels>`__




.. cfunction:: void cvMixChannels(const CvArr** src, int srcCount,                     CvArr** dst, int dstCount,                     const int* fromTo, int pairCount)

    Copies several channels from input arrays to certain channels of output arrays





    
    :param src: Input arrays 
    
    
    :param srcCount: The number of input arrays. 
    
    
    :param dst: Destination arrays 
    
    
    :param dstCount: The number of output arrays. 
    
    
    :param fromTo: The array of pairs of indices of the planes
        copied.  ``fromTo[k*2]``  is the 0-based index of the input channel in  ``src``  and ``fromTo[k*2+1]``  is the index of the output channel in  ``dst`` .
        Here the continuous channel numbering is used, that is, the first input image channels are indexed
        from  ``0``  to  ``channels(src[0])-1`` , the second input image channels are indexed from ``channels(src[0])``  to  ``channels(src[0]) + channels(src[1])-1``  etc., and the same
        scheme is used for the output image channels.
        As a special case, when  ``fromTo[k*2]``  is negative,
        the corresponding output channel is filled with zero.  
    
    
    
The function is a generalized form of 
:ref:`cvSplit`
and 
:ref:`Merge`
and some forms of 
:ref:`CvtColor`
. It can be used to change the order of the
planes, add/remove alpha channel, extract or insert a single plane or
multiple planes etc.

As an example, this code splits a 4-channel RGBA image into a 3-channel
BGR (i.e. with R and B swapped) and separate alpha channel image:




::


    
        CvMat* rgba = cvCreateMat(100, 100, CV_8UC4);
        CvMat* bgr = cvCreateMat(rgba->rows, rgba->cols, CV_8UC3);
        CvMat* alpha = cvCreateMat(rgba->rows, rgba->cols, CV_8UC1);
        cvSet(rgba, cvScalar(1,2,3,4));
    
        CvArr* out[] = { bgr, alpha };
        int from_to[] = { 0,2,  1,1,  2,0,  3,3 };
        cvMixChannels(&bgra, 1, out, 2, from_to, 4);
    

..


MulAddS
-------


Synonym for 
:ref:`ScaleAdd`
.


.. index:: Mul

.. _Mul:

Mul
---

`id=0.272808918308 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Mul>`__




.. cfunction:: void cvMul(const CvArr* src1, const CvArr* src2, CvArr* dst, double scale=1)

    Calculates the per-element product of two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    :param scale: Optional scale factor 
    
    
    
The function calculates the per-element product of two arrays:



.. math::

    \texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I) 


All the arrays must have the same type and the same size (or ROI size).
For types that have limited range this operation is saturating.


.. index:: MulSpectrums

.. _MulSpectrums:

MulSpectrums
------------

`id=0.824454753657 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MulSpectrums>`__




.. cfunction:: void cvMulSpectrums( const CvArr* src1, const CvArr* src2, CvArr* dst, int flags)

    Performs per-element multiplication of two Fourier spectrums.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array of the same type and the same size as the source arrays 
    
    
    :param flags: A combination of the following values; 
         
            * **CV_DXT_ROWS** treats each row of the arrays as a separate spectrum (see  :ref:`DFT`  parameters description). 
            
            * **CV_DXT_MUL_CONJ** conjugate the second source array before the multiplication. 
            
            
    
    
    
The function performs per-element multiplication of the two CCS-packed or complex matrices that are results of a real or complex Fourier transform.

The function, together with 
:ref:`DFT`
, may be used to calculate convolution of two arrays rapidly.



.. index:: MulTransposed

.. _MulTransposed:

MulTransposed
-------------

`id=0.918985398563 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MulTransposed>`__




.. cfunction:: void cvMulTransposed(const CvArr* src, CvArr* dst, int order, const CvArr* delta=NULL, double scale=1.0)

    Calculates the product of an array and a transposed array.





    
    :param src: The source matrix 
    
    
    :param dst: The destination matrix. Must be  ``CV_32F``  or  ``CV_64F`` . 
    
    
    :param order: Order of multipliers 
    
    
    :param delta: An optional array, subtracted from  ``src``  before multiplication 
    
    
    :param scale: An optional scaling 
    
    
    
The function calculates the product of src and its transposition:



.. math::

    \texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T 


if 
:math:`\texttt{order}=0`
, and



.. math::

    \texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} ) 


otherwise.


.. index:: Norm

.. _Norm:

Norm
----

`id=0.154207520216 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Norm>`__




.. cfunction:: double cvNorm(const CvArr* arr1, const CvArr* arr2=NULL, int normType=CV_L2, const CvArr* mask=NULL)

    Calculates absolute array norm, absolute difference norm, or relative difference norm.





    
    :param arr1: The first source image 
    
    
    :param arr2: The second source image. If it is NULL, the absolute norm of  ``arr1``  is calculated, otherwise the absolute or relative norm of  ``arr1`` - ``arr2``  is calculated. 
    
    
    :param normType: Type of norm, see the discussion 
    
    
    :param mask: The optional operation mask 
    
    
    
The function calculates the absolute norm of 
``arr1``
if 
``arr2``
is NULL:


.. math::

    norm =  \forkthree{||\texttt{arr1}||_C    = \max_I |\texttt{arr1}(I)|}{if $\texttt{normType} = \texttt{CV\_C}$}{||\texttt{arr1}||_{L1} = \sum_I |\texttt{arr1}(I)|}{if $\texttt{normType} = \texttt{CV\_L1}$}{||\texttt{arr1}||_{L2} = \sqrt{\sum_I \texttt{arr1}(I)^2}}{if $\texttt{normType} = \texttt{CV\_L2}$} 


or the absolute difference norm if 
``arr2``
is not NULL:


.. math::

    norm =  \forkthree{||\texttt{arr1}-\texttt{arr2}||_C    = \max_I |\texttt{arr1}(I) - \texttt{arr2}(I)|}{if $\texttt{normType} = \texttt{CV\_C}$}{||\texttt{arr1}-\texttt{arr2}||_{L1} = \sum_I |\texttt{arr1}(I) - \texttt{arr2}(I)|}{if $\texttt{normType} = \texttt{CV\_L1}$}{||\texttt{arr1}-\texttt{arr2}||_{L2} = \sqrt{\sum_I (\texttt{arr1}(I) - \texttt{arr2}(I))^2}}{if $\texttt{normType} = \texttt{CV\_L2}$} 


or the relative difference norm if 
``arr2``
is not NULL and 
``(normType & CV_RELATIVE) != 0``
:



.. math::

    norm =  \forkthree{\frac{||\texttt{arr1}-\texttt{arr2}||_C    }{||\texttt{arr2}||_C   }}{if $\texttt{normType} = \texttt{CV\_RELATIVE\_C}$}{\frac{||\texttt{arr1}-\texttt{arr2}||_{L1} }{||\texttt{arr2}||_{L1}}}{if $\texttt{normType} = \texttt{CV\_RELATIVE\_L1}$}{\frac{||\texttt{arr1}-\texttt{arr2}||_{L2} }{||\texttt{arr2}||_{L2}}}{if $\texttt{normType} = \texttt{CV\_RELATIVE\_L2}$} 


The function returns the calculated norm. A multiple-channel array is treated as a single-channel, that is, the results for all channels are combined.


.. index:: Not

.. _Not:

Not
---

`id=0.826629484119 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Not>`__




.. cfunction:: void cvNot(const CvArr* src, CvArr* dst)

    Performs per-element bit-wise inversion of array elements.





    
    :param src: The source array 
    
    
    :param dst: The destination array 
    
    
    
The function Not inverses every bit of every array element:




::


    
    dst(I)=~src(I)
    

..


.. index:: Or

.. _Or:

Or
--

`id=0.507374371267 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Or>`__




.. cfunction:: void cvOr(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

    Calculates per-element bit-wise disjunction of two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function calculates per-element bit-wise disjunction of two arrays:




::


    
    dst(I)=src1(I)|src2(I)
    

..

In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.


.. index:: OrS

.. _OrS:

OrS
---

`id=0.625318578996 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/OrS>`__




.. cfunction:: void cvOrS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

    Calculates a per-element bit-wise disjunction of an array and a scalar.





    
    :param src: The source array 
    
    
    :param value: Scalar to use in the operation 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function OrS calculates per-element bit-wise disjunction of an array and a scalar:




::


    
    dst(I)=src(I)|value if mask(I)!=0
    

..

Prior to the actual operation, the scalar is converted to the same type as that of the array(s). In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.



.. index:: PerspectiveTransform

.. _PerspectiveTransform:

PerspectiveTransform
--------------------

`id=0.41652773978 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/PerspectiveTransform>`__




.. cfunction:: void cvPerspectiveTransform(const CvArr* src, CvArr* dst, const CvMat* mat)

    Performs perspective matrix transformation of a vector array.





    
    :param src: The source three-channel floating-point array 
    
    
    :param dst: The destination three-channel floating-point array 
    
    
    :param mat: :math:`3\times 3`  or  :math:`4 \times 4`  transformation matrix 
    
    
    
The function transforms every element of 
``src``
(by treating it as 2D or 3D vector) in the following way:



.. math::

    (x, y, z)  \rightarrow (x'/w, y'/w, z'/w)  


where



.. math::

    (x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix} 


and


.. math::

    w =  \fork{w'}{if $w' \ne 0$}{\infty}{otherwise} 



.. index:: PolarToCart

.. _PolarToCart:

PolarToCart
-----------

`id=0.178570045111 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/PolarToCart>`__




.. cfunction:: void cvPolarToCart( const CvArr* magnitude, const CvArr* angle, CvArr* x, CvArr* y, int angleInDegrees=0)

    Calculates Cartesian coordinates of 2d vectors represented in polar form.





    
    :param magnitude: The array of magnitudes. If it is NULL, the magnitudes are assumed to be all 1's. 
    
    
    :param angle: The array of angles, whether in radians or degrees 
    
    
    :param x: The destination array of x-coordinates, may be set to NULL if it is not needed 
    
    
    :param y: The destination array of y-coordinates, mau be set to NULL if it is not needed 
    
    
    :param angleInDegrees: The flag indicating whether the angles are measured in radians, which is default mode, or in degrees 
    
    
    
The function calculates either the x-coodinate, y-coordinate or both of every vector 
``magnitude(I)*exp(angle(I)*j), j=sqrt(-1)``
:




::


    
    x(I)=magnitude(I)*cos(angle(I)),
    y(I)=magnitude(I)*sin(angle(I))
    

..


.. index:: Pow

.. _Pow:

Pow
---

`id=0.456179463072 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Pow>`__




.. cfunction:: void cvPow( const CvArr* src, CvArr* dst, double power)

    Raises every array element to a power.





    
    :param src: The source array 
    
    
    :param dst: The destination array, should be the same type as the source 
    
    
    :param power: The exponent of power 
    
    
    
The function raises every element of the input array to 
``p``
:



.. math::

    \texttt{dst} [I] =  \fork{\texttt{src}(I)^p}{if \texttt{p} is integer}{|\texttt{src}(I)^p|}{otherwise} 


That is, for a non-integer power exponent the absolute values of input array elements are used. However, it is possible to get true values for negative values using some extra operations, as the following example, computing the cube root of array elements, shows:




::


    
    CvSize size = cvGetSize(src);
    CvMat* mask = cvCreateMat(size.height, size.width, CV_8UC1);
    cvCmpS(src, 0, mask, CV_CMP_LT); /* find negative elements */
    cvPow(src, dst, 1./3);
    cvSubRS(dst, cvScalarAll(0), dst, mask); /* negate the results of negative inputs */
    cvReleaseMat(&mask);
    

..

For some values of 
``power``
, such as integer values, 0.5, and -0.5, specialized faster algorithms are used.


.. index:: Ptr?D

.. _Ptr?D:

Ptr?D
-----

`id=0.355198763108 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Ptr%3FD>`__




.. cfunction:: uchar* cvPtr1D(const CvArr* arr, int idx0, int* type=NULL)



.. cfunction:: uchar* cvPtr2D(const CvArr* arr, int idx0, int idx1, int* type=NULL)



.. cfunction:: uchar* cvPtr3D(const CvArr* arr, int idx0, int idx1, int idx2, int* type=NULL)



.. cfunction:: uchar* cvPtrND(const CvArr* arr, int* idx, int* type=NULL, int createNode=1, unsigned* precalcHashval=NULL)

    Return pointer to a particular array element.





    
    :param arr: Input array 
    
    
    :param idx0: The first zero-based component of the element index 
    
    
    :param idx1: The second zero-based component of the element index 
    
    
    :param idx2: The third zero-based component of the element index 
    
    
    :param idx: Array of the element indices 
    
    
    :param type: Optional output parameter: type of matrix elements 
    
    
    :param createNode: Optional input parameter for sparse matrices. Non-zero value of the parameter means that the requested element is created if it does not exist already. 
    
    
    :param precalcHashval: Optional input parameter for sparse matrices. If the pointer is not NULL, the function does not recalculate the node hash value, but takes it from the specified location. It is useful for speeding up pair-wise operations (TODO: provide an example) 
    
    
    
The functions return a pointer to a specific array element. Number of array dimension should match to the number of indices passed to the function except for 
``cvPtr1D``
function that can be used for sequential access to 1D, 2D or nD dense arrays.

The functions can be used for sparse arrays as well - if the requested node does not exist they create it and set it to zero.

All these as well as other functions accessing array elements (
:ref:`Get`
, 
:ref:`GetReal`
, 
:ref:`Set`
, 
:ref:`SetReal`
) raise an error in case if the element index is out of range.


.. index:: RNG

.. _RNG:

RNG
---

`id=0.334224465442 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/RNG>`__




.. cfunction:: CvRNG cvRNG(int64 seed=-1)

    Initializes a random number generator state.





    
    :param seed: 64-bit value used to initiate a random sequence 
    
    
    
The function initializes a random number generator
and returns the state. The pointer to the state can be then passed to the
:ref:`RandInt`
, 
:ref:`RandReal`
and 
:ref:`RandArr`
functions. In the
current implementation a multiply-with-carry generator is used.


.. index:: RandArr

.. _RandArr:

RandArr
-------

`id=0.617206781965 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/RandArr>`__




.. cfunction:: void cvRandArr( CvRNG* rng, CvArr* arr, int distType, CvScalar param1, CvScalar param2)

    Fills an array with random numbers and updates the RNG state.





    
    :param rng: RNG state initialized by  :ref:`RNG` 
    
    
    :param arr: The destination array 
    
    
    :param distType: Distribution type 
         
            * **CV_RAND_UNI** uniform distribution 
            
            * **CV_RAND_NORMAL** normal or Gaussian distribution 
            
            
    
    
    :param param1: The first parameter of the distribution. In the case of a uniform distribution it is the inclusive lower boundary of the random numbers range. In the case of a normal distribution it is the mean value of the random numbers. 
    
    
    :param param2: The second parameter of the distribution. In the case of a uniform distribution it is the exclusive upper boundary of the random numbers range. In the case of a normal distribution it is the standard deviation of the random numbers. 
    
    
    
The function fills the destination array with uniformly
or normally distributed random numbers.

In the example below, the function
is used to add a few normally distributed floating-point numbers to
random locations within a 2d array.




::


    
    /* let noisy_screen be the floating-point 2d array that is to be "crapped" */
    CvRNG rng_state = cvRNG(0xffffffff);
    int i, pointCount = 1000;
    /* allocate the array of coordinates of points */
    CvMat* locations = cvCreateMat(pointCount, 1, CV_32SC2);
    /* arr of random point values */
    CvMat* values = cvCreateMat(pointCount, 1, CV_32FC1);
    CvSize size = cvGetSize(noisy_screen);
    
    /* initialize the locations */
    cvRandArr(&rng_state, locations, CV_RAND_UNI, cvScalar(0,0,0,0), 
               cvScalar(size.width,size.height,0,0));
    
    /* generate values */
    cvRandArr(&rng_state, values, CV_RAND_NORMAL,
               cvRealScalar(100), // average intensity
               cvRealScalar(30) // deviation of the intensity
              );
    
    /* set the points */
    for(i = 0; i < pointCount; i++ )
    {
        CvPoint pt = *(CvPoint*)cvPtr1D(locations, i, 0);
        float value = *(float*)cvPtr1D(values, i, 0);
        *((float*)cvPtr2D(noisy_screen, pt.y, pt.x, 0 )) += value;
    }
    
    /* not to forget to release the temporary arrays */
    cvReleaseMat(&locations);
    cvReleaseMat(&values);
    
    /* RNG state does not need to be deallocated */
    

..


.. index:: RandInt

.. _RandInt:

RandInt
-------

`id=0.580357752305 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/RandInt>`__




.. cfunction:: unsigned cvRandInt(CvRNG* rng)

    Returns a 32-bit unsigned integer and updates RNG.





    
    :param rng: RNG state initialized by  ``RandInit``  and, optionally, customized by  ``RandSetRange``  (though, the latter function does not affect the discussed function outcome) 
    
    
    
The function returns a uniformly-distributed random
32-bit unsigned integer and updates the RNG state. It is similar to the rand()
function from the C runtime library, but it always generates a 32-bit number
whereas rand() returns a number in between 0 and 
``RAND_MAX``
which is 
:math:`2^{16}`
or 
:math:`2^{32}`
, depending on the platform.

The function is useful for generating scalar random numbers, such as
points, patch sizes, table indices, etc., where integer numbers of a certain
range can be generated using a modulo operation and floating-point numbers
can be generated by scaling from 0 to 1 or any other specific range.

Here is the example from the previous function discussion rewritten using
:ref:`RandInt`
:




::


    
    /* the input and the task is the same as in the previous sample. */
    CvRNG rnggstate = cvRNG(0xffffffff);
    int i, pointCount = 1000;
    /* ... - no arrays are allocated here */
    CvSize size = cvGetSize(noisygscreen);
    /* make a buffer for normally distributed numbers to reduce call overhead */
    #define bufferSize 16
    float normalValueBuffer[bufferSize];
    CvMat normalValueMat = cvMat(bufferSize, 1, CVg32F, normalValueBuffer);
    int valuesLeft = 0;
    
    for(i = 0; i < pointCount; i++ )
    {
        CvPoint pt;
        /* generate random point */
        pt.x = cvRandInt(&rnggstate ) 
        pt.y = cvRandInt(&rnggstate ) 
    
        if(valuesLeft <= 0 )
        {
            /* fulfill the buffer with normally distributed numbers 
               if the buffer is empty */
            cvRandArr(&rnggstate, &normalValueMat, CV_RAND_NORMAL, 
                       cvRealScalar(100), cvRealScalar(30));
            valuesLeft = bufferSize;
        }
        *((float*)cvPtr2D(noisygscreen, pt.y, pt.x, 0 ) = 
                                    normalValueBuffer[--valuesLeft];
    }
    
    /* there is no need to deallocate normalValueMat because we have
    both the matrix header and the data on stack. It is a common and efficient
    practice of working with small, fixed-size matrices */
    

..


.. index:: RandReal

.. _RandReal:

RandReal
--------

`id=0.350180512192 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/RandReal>`__




.. cfunction:: double cvRandReal(CvRNG* rng)

    Returns a floating-point random number and updates RNG.





    
    :param rng: RNG state initialized by  :ref:`RNG` 
    
    
    
The function returns a uniformly-distributed random floating-point number between 0 and 1 (1 is not included).


.. index:: Reduce

.. _Reduce:

Reduce
------

`id=0.0732892550064 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Reduce>`__




.. cfunction:: void cvReduce(const CvArr* src, CvArr* dst, int dim = -1, int op=CV_REDUCE_SUM)

    Reduces a matrix to a vector.





    
    :param src: The input matrix. 
    
    
    :param dst: The output single-row/single-column vector that accumulates somehow all the matrix rows/columns. 
    
    
    :param dim: The dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row, 1 means that the matrix is reduced to a single column and -1 means that the dimension is chosen automatically by analysing the dst size. 
    
    
    :param op: The reduction operation. It can take of the following values: 
         
            * **CV_REDUCE_SUM** The output is the sum of all of the matrix's rows/columns. 
            
            * **CV_REDUCE_AVG** The output is the mean vector of all of the matrix's rows/columns. 
            
            * **CV_REDUCE_MAX** The output is the maximum (column/row-wise) of all of the matrix's rows/columns. 
            
            * **CV_REDUCE_MIN** The output is the minimum (column/row-wise) of all of the matrix's rows/columns. 
            
            
    
    
    
The function reduces matrix to a vector by treating the matrix rows/columns as a set of 1D vectors and performing the specified operation on the vectors until a single row/column is obtained. For example, the function can be used to compute horizontal and vertical projections of an raster image. In the case of 
``CV_REDUCE_SUM``
and 
``CV_REDUCE_AVG``
the output may have a larger element bit-depth to preserve accuracy. And multi-channel arrays are also supported in these two reduction modes. 


.. index:: ReleaseData

.. _ReleaseData:

ReleaseData
-----------

`id=0.193575098708 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseData>`__




.. cfunction:: void cvReleaseData(CvArr* arr)

    Releases array data.





    
    :param arr: Array header 
    
    
    
The function releases the array data. In the case of 
:ref:`CvMat`
or 
:ref:`CvMatND`
it simply calls cvDecRefData(), that is the function can not deallocate external data. See also the note to 
:ref:`CreateData`
.


.. index:: ReleaseImage

.. _ReleaseImage:

ReleaseImage
------------

`id=0.44586180035 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseImage>`__




.. cfunction:: void cvReleaseImage(IplImage** image)

    Deallocates the image header and the image data.





    
    :param image: Double pointer to the image header 
    
    
    
This call is a shortened form of




::


    
    if(*image )
    {
        cvReleaseData(*image);
        cvReleaseImageHeader(image);
    }
    

..


.. index:: ReleaseImageHeader

.. _ReleaseImageHeader:

ReleaseImageHeader
------------------

`id=0.423555076157 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseImageHeader>`__




.. cfunction:: void cvReleaseImageHeader(IplImage** image)

    Deallocates an image header.





    
    :param image: Double pointer to the image header 
    
    
    
This call is an analogue of



::


    
    if(image )
    {
        iplDeallocate(*image, IPL_IMAGE_HEADER | IPL_IMAGE_ROI);
        *image = 0;
    }
    

..

but it does not use IPL functions by default (see the 
``CV_TURN_ON_IPL_COMPATIBILITY``
macro).



.. index:: ReleaseMat

.. _ReleaseMat:

ReleaseMat
----------

`id=0.627422807105 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseMat>`__




.. cfunction:: void cvReleaseMat(CvMat** mat)

    Deallocates a matrix.





    
    :param mat: Double pointer to the matrix 
    
    
    
The function decrements the matrix data reference counter and deallocates matrix header. If the data reference counter is 0, it also deallocates the data.




::


    
    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);
    

..


.. index:: ReleaseMatND

.. _ReleaseMatND:

ReleaseMatND
------------

`id=0.14075975211 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseMatND>`__




.. cfunction:: void cvReleaseMatND(CvMatND** mat)

    Deallocates a multi-dimensional array.





    
    :param mat: Double pointer to the array 
    
    
    
The function decrements the array data reference counter and releases the array header. If the reference counter reaches 0, it also deallocates the data.




::


    
    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);
    

..


.. index:: ReleaseSparseMat

.. _ReleaseSparseMat:

ReleaseSparseMat
----------------

`id=0.140784480973 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseSparseMat>`__




.. cfunction:: void cvReleaseSparseMat(CvSparseMat** mat)

    Deallocates sparse array.





    
    :param mat: Double pointer to the array 
    
    
    
The function releases the sparse array and clears the array pointer upon exit.


.. index:: Repeat

.. _Repeat:

Repeat
------

`id=0.923112302662 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Repeat>`__




.. cfunction:: void cvRepeat(const CvArr* src, CvArr* dst)

    Fill the destination array with repeated copies of the source array.





    
    :param src: Source array, image or matrix 
    
    
    :param dst: Destination array, image or matrix 
    
    
    
The function fills the destination array with repeated copies of the source array:




::


    
    dst(i,j)=src(i mod rows(src), j mod cols(src))
    

..

So the destination array may be as larger as well as smaller than the source array.


.. index:: ResetImageROI

.. _ResetImageROI:

ResetImageROI
-------------

`id=0.543905373341 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ResetImageROI>`__




.. cfunction:: void cvResetImageROI(IplImage* image)

    Resets the image ROI to include the entire image and releases the ROI structure.





    
    :param image: A pointer to the image header 
    
    
    
This produces a similar result to the following
, but in addition it releases the ROI structure.




::


    
    cvSetImageROI(image, cvRect(0, 0, image->width, image->height ));
    cvSetImageCOI(image, 0);
    

..


.. index:: Reshape

.. _Reshape:

Reshape
-------

`id=0.617983810813 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Reshape>`__




.. cfunction:: CvMat* cvReshape(const CvArr* arr, CvMat* header, int newCn, int newRows=0)

    Changes shape of matrix/image without copying data.





    
    :param arr: Input array 
    
    
    :param header: Output header to be filled 
    
    
    :param newCn: New number of channels. 'newCn = 0' means that the number of channels remains unchanged. 
    
    
    :param newRows: New number of rows. 'newRows = 0' means that the number of rows remains unchanged unless it needs to be changed according to  ``newCn``  value. 
    
    
    
The function initializes the CvMat header so that it points to the same data as the original array but has a different shape - different number of channels, different number of rows, or both.

The following example code creates one image buffer and two image headers, the first is for a 320x240x3 image and the second is for a 960x240x1 image:




::


    
    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    CvMat gray_mat_hdr;
    IplImage gray_img_hdr, *gray_img;
    cvReshape(color_img, &gray_mat_hdr, 1);
    gray_img = cvGetImage(&gray_mat_hdr, &gray_img_hdr);
    

..

And the next example converts a 3x3 matrix to a single 1x9 vector:




::


    
    CvMat* mat = cvCreateMat(3, 3, CV_32F);
    CvMat row_header, *row;
    row = cvReshape(mat, &row_header, 0, 1);
    

..


.. index:: ReshapeMatND

.. _ReshapeMatND:

ReshapeMatND
------------

`id=0.409528209175 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReshapeMatND>`__




.. cfunction:: CvArr* cvReshapeMatND(const CvArr* arr,                        int sizeofHeader, CvArr* header,                        int newCn, int newDims, int* newSizes)

    Changes the shape of a multi-dimensional array without copying the data.






::


    
    #define cvReshapeND(arr, header, newCn, newDims, newSizes )   \
          cvReshapeMatND((arr), sizeof(*(header)), (header),         \
                          (newCn), (newDims), (newSizes))
    

..



    
    :param arr: Input array 
    
    
    :param sizeofHeader: Size of output header to distinguish between IplImage, CvMat and CvMatND output headers 
    
    
    :param header: Output header to be filled 
    
    
    :param newCn: New number of channels.  :math:`\texttt{newCn} = 0`  means that the number of channels remains unchanged. 
    
    
    :param newDims: New number of dimensions.  :math:`\texttt{newDims} = 0`  means that the number of dimensions remains the same. 
    
    
    :param newSizes: Array of new dimension sizes. Only  :math:`\texttt{newDims}-1`  values are used, because the total number of elements must remain the same.
        Thus, if  :math:`\texttt{newDims} = 1` ,  ``newSizes``  array is not used. 
    
    
    
The function is an advanced version of 
:ref:`Reshape`
that can work with multi-dimensional arrays as well (though it can work with ordinary images and matrices) and change the number of dimensions.

Below are the two samples from the 
:ref:`Reshape`
description rewritten using 
:ref:`ReshapeMatND`
:




::


    
    
    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    IplImage gray_img_hdr, *gray_img;
    gray_img = (IplImage*)cvReshapeND(color_img, &gray_img_hdr, 1, 0, 0);
    
    ...
    
    /* second example is modified to convert 2x2x2 array to 8x1 vector */
    int size[] = { 2, 2, 2 };
    CvMatND* mat = cvCreateMatND(3, size, CV_32F);
    CvMat row_header, *row;
    row = (CvMat*)cvReshapeND(mat, &row_header, 0, 1, 0);
    
    

..


.. index:: cvRound, cvFloor, cvCeil

.. _cvRound, cvFloor, cvCeil:

cvRound, cvFloor, cvCeil
------------------------

`id=0.0596129889144 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/cvRound%2C%20cvFloor%2C%20cvCeil>`__




.. cfunction:: int cvRound(double value) int cvFloor(double value) int cvCeil(double value)

    Converts a floating-point number to an integer.





    
    :param value: The input floating-point value 
    
    
    
The functions convert the input floating-point number to an integer using one of the rounding
modes. 
``Round``
returns the nearest integer value to the
argument. 
``Floor``
returns the maximum integer value that is not
larger than the argument. 
``Ceil``
returns the minimum integer
value that is not smaller than the argument. On some architectures the
functions work much faster than the standard cast
operations in C. If the absolute value of the argument is greater than
:math:`2^{31}`
, the result is not determined. Special values (
:math:`\pm \infty`
, NaN)
are not handled.


.. index:: ScaleAdd

.. _ScaleAdd:

ScaleAdd
--------

`id=0.579340191614 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ScaleAdd>`__




.. cfunction:: void cvScaleAdd(const CvArr* src1, CvScalar scale, const CvArr* src2, CvArr* dst)

    Calculates the sum of a scaled array and another array.





    
    :param src1: The first source array 
    
    
    :param scale: Scale factor for the first array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    
The function calculates the sum of a scaled array and another array:



.. math::

    \texttt{dst} (I)= \texttt{scale} \, \texttt{src1} (I) +  \texttt{src2} (I) 


All array parameters should have the same type and the same size.


.. index:: Set

.. _Set:

Set
---

`id=0.861577153242 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Set>`__




.. cfunction:: void cvSet(CvArr* arr, CvScalar value, const CvArr* mask=NULL)

    Sets every element of an array to a given value.





    
    :param arr: The destination array 
    
    
    :param value: Fill value 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function copies the scalar 
``value``
to every selected element of the destination array:



.. math::

    \texttt{arr} (I)= \texttt{value} \quad \text{if} \quad \texttt{mask} (I)  \ne 0 


If array 
``arr``
is of 
``IplImage``
type, then is ROI used, but COI must not be set.


.. index:: Set?D

.. _Set?D:

Set?D
-----

`id=0.152512661076 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Set%3FD>`__




.. cfunction:: void cvSet1D(CvArr* arr, int idx0, CvScalar value)



.. cfunction:: void cvSet2D(CvArr* arr, int idx0, int idx1, CvScalar value)



.. cfunction:: void cvSet3D(CvArr* arr, int idx0, int idx1, int idx2, CvScalar value)



.. cfunction:: void cvSetND(CvArr* arr, int* idx, CvScalar value)

    Change the particular array element.





    
    :param arr: Input array 
    
    
    :param idx0: The first zero-based component of the element index 
    
    
    :param idx1: The second zero-based component of the element index 
    
    
    :param idx2: The third zero-based component of the element index 
    
    
    :param idx: Array of the element indices 
    
    
    :param value: The assigned value 
    
    
    
The functions assign the new value to a particular array element. In the case of a sparse array the functions create the node if it does not exist yet.


.. index:: SetData

.. _SetData:

SetData
-------

`id=0.107211131582 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetData>`__




.. cfunction:: void cvSetData(CvArr* arr, void* data, int step)

    Assigns user data to the array header.





    
    :param arr: Array header 
    
    
    :param data: User data 
    
    
    :param step: Full row length in bytes 
    
    
    
The function assigns user data to the array header. Header should be initialized before using 
``cvCreate*Header``
, 
``cvInit*Header``
or 
:ref:`Mat`
(in the case of matrix) function.


.. index:: SetIdentity

.. _SetIdentity:

SetIdentity
-----------

`id=0.77516298162 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetIdentity>`__




.. cfunction:: void cvSetIdentity(CvArr* mat, CvScalar value=cvRealScalar(1))

    Initializes a scaled identity matrix.





    
    :param mat: The matrix to initialize (not necesserily square) 
    
    
    :param value: The value to assign to the diagonal elements 
    
    
    
The function initializes a scaled identity matrix:



.. math::

    \texttt{arr} (i,j)= \fork{\texttt{value}}{ if $i=j$}{0}{otherwise} 



.. index:: SetImageCOI

.. _SetImageCOI:

SetImageCOI
-----------

`id=0.597376489371 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetImageCOI>`__




.. cfunction:: void cvSetImageCOI( IplImage* image, int coi)

    Sets the channel of interest in an IplImage.





    
    :param image: A pointer to the image header 
    
    
    :param coi: The channel of interest. 0 - all channels are selected, 1 - first channel is selected, etc. Note that the channel indices become 1-based. 
    
    
    
If the ROI is set to 
``NULL``
and the coi is 
*not*
0,
the ROI is allocated. Most OpenCV functions do 
*not*
support
the COI setting, so to process an individual image/matrix channel one
may copy (via 
:ref:`Copy`
or 
:ref:`Split`
) the channel to a separate
image/matrix, process it and then copy the result back (via 
:ref:`Copy`
or 
:ref:`Merge`
) if needed.


.. index:: SetImageROI

.. _SetImageROI:

SetImageROI
-----------

`id=0.699794583761 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetImageROI>`__




.. cfunction:: void cvSetImageROI( IplImage* image, CvRect rect)

    Sets an image Region Of Interest (ROI) for a given rectangle.





    
    :param image: A pointer to the image header 
    
    
    :param rect: The ROI rectangle 
    
    
    
If the original image ROI was 
``NULL``
and the 
``rect``
is not the whole image, the ROI structure is allocated.

Most OpenCV functions support the use of ROI and treat the image rectangle as a separate image. For example, all of the pixel coordinates are counted from the top-left (or bottom-left) corner of the ROI, not the original image.


.. index:: SetReal?D

.. _SetReal?D:

SetReal?D
---------

`id=0.771070365808 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetReal%3FD>`__




.. cfunction:: void cvSetReal1D(CvArr* arr, int idx0, double value)



.. cfunction:: void cvSetReal2D(CvArr* arr, int idx0, int idx1, double value)



.. cfunction:: void cvSetReal3D(CvArr* arr, int idx0, int idx1, int idx2, double value)



.. cfunction:: void cvSetRealND(CvArr* arr, int* idx, double value)

    Change a specific array element.





    
    :param arr: Input array 
    
    
    :param idx0: The first zero-based component of the element index 
    
    
    :param idx1: The second zero-based component of the element index 
    
    
    :param idx2: The third zero-based component of the element index 
    
    
    :param idx: Array of the element indices 
    
    
    :param value: The assigned value 
    
    
    
The functions assign a new value to a specific
element of a single-channel array. If the array has multiple channels,
a runtime error is raised. Note that the 
:ref:`Set*D`
function can be used
safely for both single-channel and multiple-channel arrays, though they
are a bit slower.

In the case of a sparse array the functions create the node if it does not yet exist.


.. index:: SetZero

.. _SetZero:

SetZero
-------

`id=0.0226075499078 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetZero>`__




.. cfunction:: void cvSetZero(CvArr* arr)

    Clears the array.






::


    
    #define cvZero cvSetZero
    

..



    
    :param arr: Array to be cleared 
    
    
    
The function clears the array. In the case of dense arrays (CvMat, CvMatND or IplImage), cvZero(array) is equivalent to cvSet(array,cvScalarAll(0),0).
In the case of sparse arrays all the elements are removed.


.. index:: Solve

.. _Solve:

Solve
-----

`id=0.516299173545 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Solve>`__




.. cfunction:: int cvSolve(const CvArr* src1, const CvArr* src2, CvArr* dst, int method=CV_LU)

    Solves a linear system or least-squares problem.





    
    :param A: The source matrix 
    
    
    :param B: The right-hand part of the linear system 
    
    
    :param X: The output solution 
    
    
    :param method: The solution (matrix inversion) method 
        
               
            * **CV_LU** Gaussian elimination with optimal pivot element chosen 
            
              
            * **CV_SVD** Singular value decomposition (SVD) method 
            
              
            * **CV_SVD_SYM** SVD method for a symmetric positively-defined matrix. 
            
            
    
    
    
The function solves a linear system or least-squares problem (the latter is possible with SVD methods):



.. math::

    \texttt{dst} = argmin_X|| \texttt{src1} \, \texttt{X} -  \texttt{src2} || 


If 
``CV_LU``
method is used, the function returns 1 if 
``src1``
is non-singular and 0 otherwise; in the latter case 
``dst``
is not valid.


.. index:: SolveCubic

.. _SolveCubic:

SolveCubic
----------

`id=0.317112254405 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SolveCubic>`__




.. cfunction:: void cvSolveCubic(const CvArr* coeffs, CvArr* roots)

    Finds the real roots of a cubic equation.





    
    :param coeffs: The equation coefficients, an array of 3 or 4 elements 
    
    
    :param roots: The output array of real roots which should have 3 elements 
    
    
    
The function finds the real roots of a cubic equation:

If coeffs is a 4-element vector:



.. math::

    \texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0 


or if coeffs is 3-element vector:



.. math::

    x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0 


The function returns the number of real roots found. The roots are
stored to 
``root``
array, which is padded with zeros if there is
only one root.


.. index:: Split

.. _Split:

Split
-----

`id=0.404799243335 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Split>`__




.. cfunction:: void cvSplit(const CvArr* src, CvArr* dst0, CvArr* dst1,               CvArr* dst2, CvArr* dst3)

    Divides multi-channel array into several single-channel arrays or extracts a single channel from the array.





    
    :param src: Source array 
    
    
    :param dst0: Destination channel 0 
    
    
    :param dst1: Destination channel 1 
    
    
    :param dst2: Destination channel 2 
    
    
    :param dst3: Destination channel 3 
    
    
    
The function divides a multi-channel array into separate
single-channel arrays. Two modes are available for the operation. If the
source array has N channels then if the first N destination channels
are not NULL, they all are extracted from the source array;
if only a single destination channel of the first N is not NULL, this
particular channel is extracted; otherwise an error is raised. The rest
of the destination channels (beyond the first N) must always be NULL. For
IplImage 
:ref:`Copy`
with COI set can be also used to extract a single
channel from the image.



.. index:: Sqrt

.. _Sqrt:

Sqrt
----

`id=0.688190940304 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Sqrt>`__




.. cfunction:: float cvSqrt(float value)

    Calculates the square root.





    
    :param value: The input floating-point value 
    
    
    
The function calculates the square root of the argument. If the argument is negative, the result is not determined.


.. index:: Sub

.. _Sub:

Sub
---

`id=0.952315283514 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Sub>`__




.. cfunction:: void cvSub(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

    Computes the per-element difference between two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function subtracts one array from another one:




::


    
    dst(I)=src1(I)-src2(I) if mask(I)!=0
    

..

All the arrays must have the same type, except the mask, and the same size (or ROI size).
For types that have limited range this operation is saturating.


.. index:: SubRS

.. _SubRS:

SubRS
-----

`id=0.239416677071 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SubRS>`__




.. cfunction:: void cvSubRS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

    Computes the difference between a scalar and an array.





    
    :param src: The first source array 
    
    
    :param value: Scalar to subtract from 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function subtracts every element of source array from a scalar:




::


    
    dst(I)=value-src(I) if mask(I)!=0
    

..

All the arrays must have the same type, except the mask, and the same size (or ROI size).
For types that have limited range this operation is saturating.


.. index:: SubS

.. _SubS:

SubS
----

`id=0.841148312387 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SubS>`__




.. cfunction:: void cvSubS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

    Computes the difference between an array and a scalar.





    
    :param src: The source array 
    
    
    :param value: Subtracted scalar 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function subtracts a scalar from every element of the source array:




::


    
    dst(I)=src(I)-value if mask(I)!=0
    

..

All the arrays must have the same type, except the mask, and the same size (or ROI size).
For types that have limited range this operation is saturating.



.. index:: Sum

.. _Sum:

Sum
---

`id=0.811470558337 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Sum>`__




.. cfunction:: CvScalar cvSum(const CvArr* arr)

    Adds up array elements.





    
    :param arr: The array 
    
    
    
The function calculates the sum 
``S``
of array elements, independently for each channel:



.. math::

    \sum _I  \texttt{arr} (I)_c  


If the array is 
``IplImage``
and COI is set, the function processes the selected channel only and stores the sum to the first scalar component.



.. index:: SVBkSb

.. _SVBkSb:

SVBkSb
------

`id=0.305531304006 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SVBkSb>`__




.. cfunction:: void  cvSVBkSb( const CvArr* W, const CvArr* U, const CvArr* V, const CvArr* B, CvArr* X, int flags)

    Performs singular value back substitution.





    
    :param W: Matrix or vector of singular values 
    
    
    :param U: Left orthogonal matrix (tranposed, perhaps) 
    
    
    :param V: Right orthogonal matrix (tranposed, perhaps) 
    
    
    :param B: The matrix to multiply the pseudo-inverse of the original matrix  ``A``  by. This is an optional parameter. If it is omitted then it is assumed to be an identity matrix of an appropriate size (so that  ``X``  will be the reconstructed pseudo-inverse of  ``A`` ). 
    
    
    :param X: The destination matrix: result of back substitution 
    
    
    :param flags: Operation flags, should match exactly to the  ``flags``  passed to  :ref:`SVD` 
    
    
    
The function calculates back substitution for decomposed matrix 
``A``
(see 
:ref:`SVD`
description) and matrix 
``B``
:



.. math::

    \texttt{X} =  \texttt{V} \texttt{W} ^{-1}  \texttt{U} ^T  \texttt{B} 


where



.. math::

    W^{-1}_{(i,i)}= \fork{1/W_{(i,i)}}{if $W_{(i,i)} > \epsilon \sum_i{W_{(i,i)}}$ }{0}{otherwise} 


and 
:math:`\epsilon`
is a small number that depends on the matrix data type.

This function together with 
:ref:`SVD`
is used inside 
:ref:`Invert`
and 
:ref:`Solve`
, and the possible reason to use these (svd and bksb)
"low-level" function, is to avoid allocation of temporary matrices inside
the high-level counterparts (inv and solve).


.. index:: SVD

.. _SVD:

SVD
---

`id=0.666817969466 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SVD>`__




.. cfunction:: void cvSVD( CvArr* A,  CvArr* W,  CvArr* U=NULL,  CvArr* V=NULL,  int flags=0)

    Performs singular value decomposition of a real floating-point matrix.





    
    :param A: Source  :math:`\texttt{M} \times \texttt{N}`  matrix 
    
    
    :param W: Resulting singular value diagonal matrix ( :math:`\texttt{M} \times \texttt{N}`  or  :math:`\min(\texttt{M}, \texttt{N})  \times \min(\texttt{M}, \texttt{N})` ) or  :math:`\min(\texttt{M},\texttt{N}) \times 1`  vector of the singular values 
    
    
    :param U: Optional left orthogonal matrix,  :math:`\texttt{M} \times \min(\texttt{M}, \texttt{N})`  (when  ``CV_SVD_U_T``  is not set), or  :math:`\min(\texttt{M},\texttt{N}) \times \texttt{M}`  (when  ``CV_SVD_U_T``  is set), or  :math:`\texttt{M} \times \texttt{M}`  (regardless of  ``CV_SVD_U_T``  flag). 
    
    
    :param V: Optional right orthogonal matrix,  :math:`\texttt{N} \times \min(\texttt{M}, \texttt{N})`  (when  ``CV_SVD_V_T``  is not set), or  :math:`\min(\texttt{M},\texttt{N}) \times \texttt{N}`  (when  ``CV_SVD_V_T``  is set), or  :math:`\texttt{N} \times \texttt{N}`  (regardless of  ``CV_SVD_V_T``  flag). 
    
    
    :param flags: Operation flags; can be 0 or a combination of the following values: 
        
                
            * **CV_SVD_MODIFY_A** enables modification of matrix  ``A``  during the operation. It speeds up the processing. 
            
               
            * **CV_SVD_U_T** means that the transposed matrix  ``U``  is returned. Specifying the flag speeds up the processing. 
            
               
            * **CV_SVD_V_T** means that the transposed matrix  ``V``  is returned. Specifying the flag speeds up the processing. 
            
            
    
    
    
The function decomposes matrix 
``A``
into the product of a diagonal matrix and two 

orthogonal matrices:



.. math::

    A=U  \, W  \, V^T 


where 
:math:`W`
is a diagonal matrix of singular values that can be coded as a
1D vector of singular values and 
:math:`U`
and 
:math:`V`
. All the singular values
are non-negative and sorted (together with 
:math:`U`
and 
:math:`V`
columns)
in descending order.

An SVD algorithm is numerically robust and its typical applications include:



    

*
    accurate eigenvalue problem solution when matrix 
    ``A``
    is a square, symmetric, and positively defined matrix, for example, when
      it is a covariance matrix. 
    :math:`W`
    in this case will be a vector/matrix
      of the eigenvalues, and 
    :math:`U = V`
    will be a matrix of the eigenvectors.
      
    

*
    accurate solution of a poor-conditioned linear system.
      
    

*
    least-squares solution of an overdetermined linear system. This and the preceeding is done by using the 
    :ref:`Solve`
    function with the 
    ``CV_SVD``
    method.
      
    

*
    accurate calculation of different matrix characteristics such as the matrix rank (the number of non-zero singular values), condition number (ratio of the largest singular value to the smallest one), and determinant (absolute value of the determinant is equal to the product of singular values). 
    
    

.. index:: Trace

.. _Trace:

Trace
-----

`id=0.173901751057 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Trace>`__




.. cfunction:: CvScalar cvTrace(const CvArr* mat)

    Returns the trace of a matrix.





    
    :param mat: The source matrix 
    
    
    
The function returns the sum of the diagonal elements of the matrix 
``src1``
.



.. math::

    tr( \texttt{mat} ) =  \sum _i  \texttt{mat} (i,i)  



.. index:: Transform

.. _Transform:

Transform
---------

`id=0.132381356501 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Transform>`__




.. cfunction:: void cvTransform(const CvArr* src, CvArr* dst, const CvMat* transmat, const CvMat* shiftvec=NULL)

    Performs matrix transformation of every array element.





    
    :param src: The first source array 
    
    
    :param dst: The destination array 
    
    
    :param transmat: Transformation matrix 
    
    
    :param shiftvec: Optional shift vector 
    
    
    
The function performs matrix transformation of every element of array 
``src``
and stores the results in 
``dst``
:



.. math::

    dst(I) = transmat  \cdot src(I) + shiftvec  


That is, every element of an 
``N``
-channel array 
``src``
is
considered as an 
``N``
-element vector which is transformed using
a 
:math:`\texttt{M} \times \texttt{N}`
matrix 
``transmat``
and shift
vector 
``shiftvec``
into an element of 
``M``
-channel array
``dst``
. There is an option to embedd 
``shiftvec``
into
``transmat``
. In this case 
``transmat``
should be a 
:math:`\texttt{M}
\times (N+1)`
matrix and the rightmost column is treated as the shift
vector.

Both source and destination arrays should have the same depth and the
same size or selected ROI size. 
``transmat``
and 
``shiftvec``
should be real floating-point matrices.

The function may be used for geometrical transformation of n dimensional
point set, arbitrary linear color space transformation, shuffling the
channels and so forth.


.. index:: Transpose

.. _Transpose:

Transpose
---------

`id=0.402895405287 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Transpose>`__




.. cfunction:: void cvTranspose(const CvArr* src, CvArr* dst)

    Transposes a matrix.





    
    :param src: The source matrix 
    
    
    :param dst: The destination matrix 
    
    
    
The function transposes matrix 
``src1``
:



.. math::

    \texttt{dst} (i,j) =  \texttt{src} (j,i)  


Note that no complex conjugation is done in the case of a complex
matrix. Conjugation should be done separately: look at the sample code
in 
:ref:`XorS`
for an example.


.. index:: Xor

.. _Xor:

Xor
---

`id=0.778881513254 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/Xor>`__




.. cfunction:: void cvXor(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

    Performs per-element bit-wise "exclusive or" operation on two arrays.





    
    :param src1: The first source array 
    
    
    :param src2: The second source array 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function calculates per-element bit-wise logical conjunction of two arrays:




::


    
    dst(I)=src1(I)^src2(I) if mask(I)!=0
    

..

In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.


.. index:: XorS

.. _XorS:

XorS
----

`id=0.0218684678783 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/XorS>`__




.. cfunction:: void cvXorS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

    Performs per-element bit-wise "exclusive or" operation on an array and a scalar.





    
    :param src: The source array 
    
    
    :param value: Scalar to use in the operation 
    
    
    :param dst: The destination array 
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    
    
The function XorS calculates per-element bit-wise conjunction of an array and a scalar:




::


    
    dst(I)=src(I)^value if mask(I)!=0
    

..

Prior to the actual operation, the scalar is converted to the same type as that of the array(s). In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size

The following sample demonstrates how to conjugate complex vector by switching the most-significant bit of imaging part:




::


    
    
    float a[] = { 1, 0, 0, 1, -1, 0, 0, -1 }; /* 1, j, -1, -j */
    CvMat A = cvMat(4, 1, CV_32FC2, &a);
    int i, negMask = 0x80000000;
    cvXorS(&A, cvScalar(0, *(float*)&negMask, 0, 0 ), &A, 0);
    for(i = 0; i < 4; i++ )
        printf("(%.1f, %.1f) ", a[i*2], a[i*2+1]);
    
    

..

The code should print:




::


    
    (1.0,0.0) (0.0,-1.0) (-1.0,0.0) (0.0,1.0)
    

..


.. index:: mGet

.. _mGet:

mGet
----

`id=0.966917154108 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/mGet>`__




.. cfunction:: double cvmGet(const CvMat* mat, int row, int col)

    Returns the particular element of single-channel floating-point matrix.





    
    :param mat: Input matrix 
    
    
    :param row: The zero-based index of row 
    
    
    :param col: The zero-based index of column 
    
    
    
The function is a fast replacement for 
:ref:`GetReal2D`
in the case of single-channel floating-point matrices. It is faster because
it is inline, it does fewer checks for array type and array element type,
and it checks for the row and column ranges only in debug mode.


.. index:: mSet

.. _mSet:

mSet
----

`id=0.367233373522 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/mSet>`__




.. cfunction:: void cvmSet(CvMat* mat, int row, int col, double value)

    Returns a specific element of a single-channel floating-point matrix.





    
    :param mat: The matrix 
    
    
    :param row: The zero-based index of row 
    
    
    :param col: The zero-based index of column 
    
    
    :param value: The new value of the matrix element 
    
    
    
The function is a fast replacement for 
:ref:`SetReal2D`
in the case of single-channel floating-point matrices. It is faster because
it is inline, it does fewer checks for array type and array element type, 
and it checks for the row and column ranges only in debug mode.

