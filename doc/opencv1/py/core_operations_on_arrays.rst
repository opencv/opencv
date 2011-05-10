Operations on Arrays
====================

.. highlight:: python



.. index:: AbsDiff

.. _AbsDiff:

AbsDiff
-------




.. function:: AbsDiff(src1,src2,dst)-> None

    Calculates absolute difference between two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The function calculates absolute difference between two arrays.



.. math::

    \texttt{dst} (i)_c = | \texttt{src1} (I)_c -  \texttt{src2} (I)_c|  


All the arrays must have the same data type and the same size (or ROI size).


.. index:: AbsDiffS

.. _AbsDiffS:

AbsDiffS
--------




.. function:: AbsDiffS(src,value,dst)-> None

    Calculates absolute difference between an array and a scalar.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param value: The scalar 
    
    :type value: :class:`CvScalar`
    
    
    
The function calculates absolute difference between an array and a scalar.



.. math::

    \texttt{dst} (i)_c = | \texttt{src} (I)_c -  \texttt{value} _c|  


All the arrays must have the same data type and the same size (or ROI size).



.. index:: Add

.. _Add:

Add
---




.. function:: Add(src1,src2,dst,mask=NULL)-> None

    Computes the per-element sum of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: AddS(src,value,dst,mask=NULL)-> None

    Computes the sum of an array and a scalar.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: Added scalar 
    
    :type value: :class:`CvScalar`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: AddWeighted(src1,alpha,src2,beta,gamma,dst)-> None

    Computes the weighted sum of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param alpha: Weight for the first array elements 
    
    :type alpha: float
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param beta: Weight for the second array elements 
    
    :type beta: float
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param gamma: Scalar, added to each sum 
    
    :type gamma: float
    
    
    
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




.. function:: And(src1,src2,dst,mask=NULL)-> None

    Calculates per-element bit-wise conjunction of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
The function calculates per-element bit-wise logical conjunction of two arrays:




::


    
    dst(I)=src1(I)&src2(I) if mask(I)!=0
    

..

In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.


.. index:: AndS

.. _AndS:

AndS
----




.. function:: AndS(src,value,dst,mask=NULL)-> None

    Calculates per-element bit-wise conjunction of an array and a scalar.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: Scalar to use in the operation 
    
    :type value: :class:`CvScalar`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
The function calculates per-element bit-wise conjunction of an array and a scalar:




::


    
    dst(I)=src(I)&value if mask(I)!=0
    

..

Prior to the actual operation, the scalar is converted to the same type as that of the array(s). In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.


.. index:: Avg

.. _Avg:

Avg
---




.. function:: Avg(arr,mask=NULL)-> CvScalar

    Calculates average (mean) of array elements.





    
    :param arr: The array 
    
    :type arr: :class:`CvArr`
    
    
    :param mask: The optional operation mask 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: AvgSdv(arr,mask=NULL)-> (mean, stdDev)

    Calculates average (mean) of array elements.





    
    :param arr: The array 
    
    :type arr: :class:`CvArr`
    
    
    :param mask: The optional operation mask 
    
    :type mask: :class:`CvArr`
    
    
    :param mean: Mean value, a CvScalar 
    
    :type mean: :class:`CvScalar`
    
    
    :param stdDev: Standard deviation, a CvScalar 
    
    :type stdDev: :class:`CvScalar`
    
    
    
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




.. function:: CalcCovarMatrix(vects,covMat,avg,flags)-> None

    Calculates covariance matrix of a set of vectors.





    
    :param vects: The input vectors, all of which must have the same type and the same size. The vectors do not have to be 1D, they can be 2D (e.g., images) and so forth 
    
    :type vects: :class:`cvarr_count`
    
    
    :param covMat: The output covariance matrix that should be floating-point and square 
    
    :type covMat: :class:`CvArr`
    
    
    :param avg: The input or output (depending on the flags) array - the mean (average) vector of the input vectors 
    
    :type avg: :class:`CvArr`
    
    
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
            
            
            
    
    :type flags: int
    
    
    
The function calculates the covariance matrix
and, optionally, the mean vector of the set of input vectors. The function
can be used for PCA, for comparing vectors using Mahalanobis distance and so forth.


.. index:: CartToPolar

.. _CartToPolar:

CartToPolar
-----------




.. function:: CartToPolar(x,y,magnitude,angle=NULL,angleInDegrees=0)-> None

    Calculates the magnitude and/or angle of 2d vectors.





    
    :param x: The array of x-coordinates 
    
    :type x: :class:`CvArr`
    
    
    :param y: The array of y-coordinates 
    
    :type y: :class:`CvArr`
    
    
    :param magnitude: The destination array of magnitudes, may be set to NULL if it is not needed 
    
    :type magnitude: :class:`CvArr`
    
    
    :param angle: The destination array of angles, may be set to NULL if it is not needed. The angles are measured in radians  :math:`(0`  to  :math:`2 \pi )`  or in degrees (0 to 360 degrees). 
    
    :type angle: :class:`CvArr`
    
    
    :param angleInDegrees: The flag indicating whether the angles are measured in radians, which is default mode, or in degrees 
    
    :type angleInDegrees: int
    
    
    
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




.. function:: Cbrt(value)-> float

    Calculates the cubic root





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
The function calculates the cubic root of the argument, and normally it is faster than 
``pow(value,1./3)``
. In addition, negative arguments are handled properly. Special values (
:math:`\pm \infty`
, NaN) are not handled.


.. index:: ClearND

.. _ClearND:

ClearND
-------




.. function:: ClearND(arr,idx)-> None

    Clears a specific array element.




    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx: Array of the element indices 
    
    :type idx: sequence of int
    
    
    
The function 
:ref:`ClearND`
clears (sets to zero) a specific element of a dense array or deletes the element of a sparse array. If the sparse array element does not exists, the function does nothing.


.. index:: CloneImage

.. _CloneImage:

CloneImage
----------




.. function:: CloneImage(image)-> copy

    Makes a full copy of an image, including the header, data, and ROI.





    
    :param image: The original image 
    
    :type image: :class:`IplImage`
    
    
    
The returned 
``IplImage*``
points to the image copy.


.. index:: CloneMat

.. _CloneMat:

CloneMat
--------




.. function:: CloneMat(mat)-> copy

    Creates a full matrix copy.





    
    :param mat: Matrix to be copied 
    
    :type mat: :class:`CvMat`
    
    
    
Creates a full copy of a matrix and returns a pointer to the copy.


.. index:: CloneMatND

.. _CloneMatND:

CloneMatND
----------




.. function:: CloneMatND(mat)-> copy

    Creates full copy of a multi-dimensional array and returns a pointer to the copy.





    
    :param mat: Input array 
    
    :type mat: :class:`CvMatND`
    
    
    

.. index:: Cmp

.. _Cmp:

Cmp
---




.. function:: Cmp(src1,src2,dst,cmpOp)-> None

    Performs per-element comparison of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array. Both source arrays must have a single channel. 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    :type dst: :class:`CvArr`
    
    
    :param cmpOp: The flag specifying the relation between the elements to be checked 
        
               
            * **CV_CMP_EQ** src1(I) "equal to" value 
            
              
            * **CV_CMP_GT** src1(I) "greater than" value 
            
              
            * **CV_CMP_GE** src1(I) "greater or equal" value 
            
              
            * **CV_CMP_LT** src1(I) "less than" value 
            
              
            * **CV_CMP_LE** src1(I) "less or equal" value 
            
              
            * **CV_CMP_NE** src1(I) "not equal" value 
            
            
    
    :type cmpOp: int
    
    
    
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




.. function:: CmpS(src,value,dst,cmpOp)-> None

    Performs per-element comparison of an array and a scalar.





    
    :param src: The source array, must have a single channel 
    
    :type src: :class:`CvArr`
    
    
    :param value: The scalar value to compare each array element with 
    
    :type value: float
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    :type dst: :class:`CvArr`
    
    
    :param cmpOp: The flag specifying the relation between the elements to be checked 
        
               
            * **CV_CMP_EQ** src1(I) "equal to" value 
            
              
            * **CV_CMP_GT** src1(I) "greater than" value 
            
              
            * **CV_CMP_GE** src1(I) "greater or equal" value 
            
              
            * **CV_CMP_LT** src1(I) "less than" value 
            
              
            * **CV_CMP_LE** src1(I) "less or equal" value 
            
              
            * **CV_CMP_NE** src1(I) "not equal" value 
            
            
    
    :type cmpOp: int
    
    
    
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


.. index:: Convert

.. _Convert:

Convert
-------




.. function:: Convert(src,dst)-> None

    Converts one array to another.





    
    :param src: Source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The type of conversion is done with rounding and saturation, that is if the
result of scaling + conversion can not be represented exactly by a value
of the destination array element type, it is set to the nearest representable
value on the real axis.

All the channels of multi-channel arrays are processed independently.


.. index:: ConvertScale

.. _ConvertScale:

ConvertScale
------------




.. function:: ConvertScale(src,dst,scale=1.0,shift=0.0)-> None

    Converts one array to another with optional linear transformation.





    
    :param src: Source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param scale: Scale factor 
    
    :type scale: float
    
    
    :param shift: Value added to the scaled source array elements 
    
    :type shift: float
    
    
    
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




.. function:: ConvertScaleAbs(src,dst,scale=1.0,shift=0.0)-> None

    Converts input array elements to another 8-bit unsigned integer with optional linear transformation.





    
    :param src: Source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array (should have 8u depth) 
    
    :type dst: :class:`CvArr`
    
    
    :param scale: ScaleAbs factor 
    
    :type scale: float
    
    
    :param shift: Value added to the scaled source array elements 
    
    :type shift: float
    
    
    
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




.. function:: CvtScaleAbs(src,dst,scale=1.0,shift=0.0)-> None

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




.. function:: Copy(src,dst,mask=NULL)-> None

    Copies one array to another.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: CountNonZero(arr)-> int

    Counts non-zero array elements.





    
    :param arr: The array must be a single-channel array or a multi-channel image with COI set 
    
    :type arr: :class:`CvArr`
    
    
    
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




.. function:: CreateData(arr) -> None

    Allocates array data





    
    :param arr: Array header 
    
    :type arr: :class:`CvArr`
    
    
    
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




.. function:: CreateImage(size, depth, channels)->image

    Creates an image header and allocates the image data.





    
    :param size: Image width and height 
    
    :type size: :class:`CvSize`
    
    
    :param depth: Bit depth of image elements. See  :ref:`IplImage`  for valid depths. 
    
    :type depth: int
    
    
    :param channels: Number of channels per pixel. See  :ref:`IplImage`  for details. This function only creates images with interleaved channels. 
    
    :type channels: int
    
    
    

.. index:: CreateImageHeader

.. _CreateImageHeader:

CreateImageHeader
-----------------




.. function:: CreateImageHeader(size, depth, channels) -> image

    Creates an image header but does not allocate the image data.





    
    :param size: Image width and height 
    
    :type size: :class:`CvSize`
    
    
    :param depth: Image depth (see  :ref:`CreateImage` ) 
    
    :type depth: int
    
    
    :param channels: Number of channels (see  :ref:`CreateImage` ) 
    
    :type channels: int
    
    
    

.. index:: CreateMat

.. _CreateMat:

CreateMat
---------




.. function:: CreateMat(rows, cols, type) -> mat

    Creates a matrix header and allocates the matrix data. 





    
    :param rows: Number of rows in the matrix 
    
    :type rows: int
    
    
    :param cols: Number of columns in the matrix 
    
    :type cols: int
    
    
    :param type: The type of the matrix elements in the form  ``CV_<bit depth><S|U|F>C<number of channels>`` , where S=signed, U=unsigned, F=float. For example, CV _ 8UC1 means the elements are 8-bit unsigned and the there is 1 channel, and CV _ 32SC2 means the elements are 32-bit signed and there are 2 channels. 
    
    :type type: int
    
    
    

.. index:: CreateMatHeader

.. _CreateMatHeader:

CreateMatHeader
---------------




.. function:: CreateMatHeader(rows, cols, type) -> mat

    Creates a matrix header but does not allocate the matrix data.





    
    :param rows: Number of rows in the matrix 
    
    :type rows: int
    
    
    :param cols: Number of columns in the matrix 
    
    :type cols: int
    
    
    :param type: Type of the matrix elements, see  :ref:`CreateMat` 
    
    :type type: int
    
    
    
The function allocates a new matrix header and returns a pointer to it. The matrix data can then be allocated using 
:ref:`CreateData`
or set explicitly to user-allocated data via 
:ref:`SetData`
.


.. index:: CreateMatND

.. _CreateMatND:

CreateMatND
-----------




.. function:: CreateMatND(dims, type) -> None

    Creates the header and allocates the data for a multi-dimensional dense array.





    
    :param dims: List or tuple of array dimensions, up to 32 in length. 
    
    :type dims: sequence of int
    
    
    :param type: Type of array elements, see  :ref:`CreateMat` . 
    
    :type type: int
    
    
    
This is a short form for:


.. index:: CreateMatNDHeader

.. _CreateMatNDHeader:

CreateMatNDHeader
-----------------




.. function:: CreateMatNDHeader(dims, type) -> None

    Creates a new matrix header but does not allocate the matrix data.





    
    :param dims: List or tuple of array dimensions, up to 32 in length. 
    
    :type dims: sequence of int
    
    
    :param type: Type of array elements, see  :ref:`CreateMat` 
    
    :type type: int
    
    
    
The function allocates a header for a multi-dimensional dense array. The array data can further be allocated using 
:ref:`CreateData`
or set explicitly to user-allocated data via 
:ref:`SetData`
.


.. index:: CrossProduct

.. _CrossProduct:

CrossProduct
------------




.. function:: CrossProduct(src1,src2,dst)-> None

    Calculates the cross product of two 3D vectors.





    
    :param src1: The first source vector 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source vector 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination vector 
    
    :type dst: :class:`CvArr`
    
    
    
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




.. function:: DCT(src,dst,flags)-> None

    Performs a forward or inverse Discrete Cosine transform of a 1D or 2D floating-point array.





    
    :param src: Source array, real 1D or 2D array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array of the same size and same type as the source 
    
    :type dst: :class:`CvArr`
    
    
    :param flags: Transformation flags, a combination of the following values 
         
            * **CV_DXT_FORWARD** do a forward 1D or 2D transform. 
            
            * **CV_DXT_INVERSE** do an inverse 1D or 2D transform. 
            
            * **CV_DXT_ROWS** do a forward or inverse transform of every individual row of the input matrix. This flag allows user to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself), to do 3D and higher-dimensional transforms and so forth. 
            
            
    
    :type flags: int
    
    
    
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




.. function:: DFT(src,dst,flags,nonzeroRows=0)-> None

    Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.





    
    :param src: Source array, real or complex 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array of the same size and same type as the source 
    
    :type dst: :class:`CvArr`
    
    
    :param flags: Transformation flags, a combination of the following values 
         
            * **CV_DXT_FORWARD** do a forward 1D or 2D transform. The result is not scaled. 
            
            * **CV_DXT_INVERSE** do an inverse 1D or 2D transform. The result is not scaled.  ``CV_DXT_FORWARD``  and  ``CV_DXT_INVERSE``  are mutually exclusive, of course. 
            
            * **CV_DXT_SCALE** scale the result: divide it by the number of array elements. Usually, it is combined with  ``CV_DXT_INVERSE`` , and one may use a shortcut  ``CV_DXT_INV_SCALE`` . 
            
            * **CV_DXT_ROWS** do a forward or inverse transform of every individual row of the input matrix. This flag allows the user to transform multiple vectors simultaneously and can be used to decrease the overhead (which is sometimes several times larger than the processing itself), to do 3D and higher-dimensional transforms and so forth. 
            
            * **CV_DXT_INVERSE_SCALE** same as  ``CV_DXT_INVERSE + CV_DXT_SCALE`` 
            
            
    
    :type flags: int
    
    
    :param nonzeroRows: Number of nonzero rows in the source array
        (in the case of a forward 2d transform), or a number of rows of interest in
        the destination array (in the case of an inverse 2d transform). If the value
        is negative, zero, or greater than the total number of rows, it is
        ignored. The parameter can be used to speed up 2d convolution/correlation
        when computing via DFT. See the example below. 
    
    :type nonzeroRows: int
    
    
    
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


.. index:: Det

.. _Det:

Det
---




.. function:: Det(mat)-> double

    Returns the determinant of a matrix.





    
    :param mat: The source matrix 
    
    :type mat: :class:`CvArr`
    
    
    
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




.. function:: Div(src1,src2,dst,scale)-> None

    Performs per-element division of two arrays.





    
    :param src1: The first source array. If the pointer is NULL, the array is assumed to be all 1's. 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param scale: Optional scale factor 
    
    :type scale: float
    
    
    
The function divides one array by another:



.. math::

    \texttt{dst} (I)= \fork{\texttt{scale} \cdot \texttt{src1}(I)/\texttt{src2}(I)}{if \texttt{src1} is not \texttt{NULL}}{\texttt{scale}/\texttt{src2}(I)}{otherwise} 


All the arrays must have the same type and the same size (or ROI size).



.. index:: DotProduct

.. _DotProduct:

DotProduct
----------




.. function:: DotProduct(src1,src2)-> double

    Calculates the dot product of two arrays in Euclidian metrics.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    
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




.. function:: EigenVV(mat,evects,evals,eps,lowindex,highindex)-> None

    Computes eigenvalues and eigenvectors of a symmetric matrix.





    
    :param mat: The input symmetric square matrix, modified during the processing 
    
    :type mat: :class:`CvArr`
    
    
    :param evects: The output matrix of eigenvectors, stored as subsequent rows 
    
    :type evects: :class:`CvArr`
    
    
    :param evals: The output vector of eigenvalues, stored in the descending order (order of eigenvalues and eigenvectors is syncronized, of course) 
    
    :type evals: :class:`CvArr`
    
    
    :param eps: Accuracy of diagonalization. Typically,  ``DBL_EPSILON``  (about  :math:`10^{-15}` ) works well.
        THIS PARAMETER IS CURRENTLY IGNORED. 
    
    :type eps: float
    
    
    :param lowindex: Optional index of largest eigenvalue/-vector to calculate.
        (See below.) 
    
    :type lowindex: int
    
    
    :param highindex: Optional index of smallest eigenvalue/-vector to calculate.
        (See below.) 
    
    :type highindex: int
    
    
    
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




.. function:: Exp(src,dst)-> None

    Calculates the exponent of every array element.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination array, it should have  ``double``  type or the same type as the source 
    
    :type dst: :class:`CvArr`
    
    
    
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




.. function:: FastArctan(y,x)-> float

    Calculates the angle of a 2D vector.





    
    :param x: x-coordinate of 2D vector 
    
    :type x: float
    
    
    :param y: y-coordinate of 2D vector 
    
    :type y: float
    
    
    
The function calculates the full-range angle of an input 2D vector. The angle is 
measured in degrees and varies from 0 degrees to 360 degrees. The accuracy is about 0.1 degrees.


.. index:: Flip

.. _Flip:

Flip
----




.. function:: Flip(src,dst=NULL,flipMode=0)-> None

    Flip a 2D array around vertical, horizontal or both axes.





    
    :param src: Source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array.
        If  :math:`\texttt{dst} = \texttt{NULL}`  the flipping is done in place. 
    
    :type dst: :class:`CvArr`
    
    
    :param flipMode: Specifies how to flip the array:
        0 means flipping around the x-axis, positive (e.g., 1) means flipping around y-axis, and negative (e.g., -1) means flipping around both axes. See also the discussion below for the formulas: 
    
    :type flipMode: int
    
    
    
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
    
    

.. index:: fromarray

.. _fromarray:

fromarray
---------




.. function:: fromarray(object, allowND = False) -> CvMat

    Create a CvMat from an object that supports the array interface.





    
    :param object: Any object that supports the array interface 
    
    
    :param allowND: If true, will return a CvMatND 
    
    
    
If the object supports the
`array interface <http://docs.scipy.org/doc/numpy/reference/arrays.interface.html>`_
,
return a 
:ref:`CvMat`
(
``allowND = False``
) or 
:ref:`CvMatND`
(
``allowND = True``
).

If 
``allowND = False``
, then the object's array must be either 2D or 3D.  If it is 2D, then the returned CvMat has a single channel.  If it is 3D, then the returned CvMat will have N channels, where N is the last dimension of the array. In this case, N cannot be greater than OpenCV's channel limit, 
``CV_CN_MAX``
.

If 
``allowND = True``
, then 
``fromarray``
returns a single-channel 
:ref:`CvMatND`
with the same shape as the original array.

For example, 
`NumPy <http://numpy.scipy.org/>`_
arrays support the array interface, so can be converted to OpenCV objects:




.. doctest::


    
    >>> import cv, numpy
    >>> a = numpy.ones((480, 640))
    >>> mat = cv.fromarray(a)
    >>> print cv.GetDims(mat), cv.CV_MAT_CN(cv.GetElemType(mat))
    (480, 640) 1
    >>> a = numpy.ones((480, 640, 3))
    >>> mat = cv.fromarray(a)
    >>> print cv.GetDims(mat), cv.CV_MAT_CN(cv.GetElemType(mat))
    (480, 640) 3
    >>> a = numpy.ones((480, 640, 3))
    >>> mat = cv.fromarray(a, allowND = True)
    >>> print cv.GetDims(mat), cv.CV_MAT_CN(cv.GetElemType(mat))
    (480, 640, 3) 1
    

..


.. index:: GEMM

.. _GEMM:

GEMM
----




.. function:: GEMM(src1,src2,alphs,src3,beta,dst,tABC=0)-> None

    Performs generalized matrix multiplication.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param src3: The third source array (shift). Can be NULL, if there is no shift. 
    
    :type src3: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param tABC: The operation flags that can be 0 or a combination of the following values 
         
            * **CV_GEMM_A_T** transpose src1 
            
            * **CV_GEMM_B_T** transpose src2 
            
            * **CV_GEMM_C_T** transpose src3 
            
            
        
        For example,  ``CV_GEMM_A_T+CV_GEMM_C_T``  corresponds to 
        
        .. math::
        
            \texttt{alpha}   \,   \texttt{src1}  ^T  \,   \texttt{src2}  +  \texttt{beta}   \,   \texttt{src3}  ^T 
        
        
    
    :type tABC: int
    
    
    
The function performs generalized matrix multiplication:



.. math::

    \texttt{dst} =  \texttt{alpha} \, op( \texttt{src1} )  \, op( \texttt{src2} ) +  \texttt{beta} \, op( \texttt{src3} )  \quad \text{where $op(X)$ is $X$ or $X^T$} 


All the matrices should have the same data type and coordinated sizes. Real or complex floating-point matrices are supported.


.. index:: Get1D

.. _Get1D:

Get1D
-----




.. function:: Get1D(arr, idx) -> scalar

    Return a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx: Zero-based element index 
    
    :type idx: int
    
    
    
Return a specific array element.  Array must have dimension 3.


.. index:: Get2D

.. _Get2D:

Get2D
-----




.. function::  Get2D(arr, idx0, idx1) -> scalar 

    Return a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: Zero-based element row index 
    
    :type idx0: int
    
    
    :param idx1: Zero-based element column index 
    
    :type idx1: int
    
    
    
Return a specific array element.  Array must have dimension 2.


.. index:: Get3D

.. _Get3D:

Get3D
-----




.. function::  Get3D(arr, idx0, idx1, idx2) -> scalar 

    Return a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: Zero-based element index 
    
    :type idx0: int
    
    
    :param idx1: Zero-based element index 
    
    :type idx1: int
    
    
    :param idx2: Zero-based element index 
    
    :type idx2: int
    
    
    
Return a specific array element.  Array must have dimension 3.


.. index:: GetND

.. _GetND:

GetND
-----




.. function::  GetND(arr, indices) -> scalar 

    Return a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param indices: List of zero-based element indices 
    
    :type indices: sequence of int
    
    
    
Return a specific array element.  The length of array indices must be the same as the dimension of the array.


.. index:: GetCol

.. _GetCol:

GetCol
------




.. function:: GetCol(arr,col)-> submat

    Returns array column.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param col: Zero-based index of the selected column 
    
    :type col: int
    
    
    :param submat: resulting single-column array 
    
    :type submat: :class:`CvMat`
    
    
    
The function 
``GetCol``
returns a single column from the input array.


.. index:: GetCols

.. _GetCols:

GetCols
-------




.. function:: GetCols(arr,startCol,endCol)-> submat

    Returns array column span.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param startCol: Zero-based index of the starting column (inclusive) of the span 
    
    :type startCol: int
    
    
    :param endCol: Zero-based index of the ending column (exclusive) of the span 
    
    :type endCol: int
    
    
    :param submat: resulting multi-column array 
    
    :type submat: :class:`CvMat`
    
    
    
The function 
``GetCols``
returns a column span from the input array.


.. index:: GetDiag

.. _GetDiag:

GetDiag
-------




.. function:: GetDiag(arr,diag=0)-> submat

    Returns one of array diagonals.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param submat: Pointer to the resulting sub-array header 
    
    :type submat: :class:`CvMat`
    
    
    :param diag: Array diagonal. Zero corresponds to the main diagonal, -1 corresponds to the diagonal above the main , 1 corresponds to the diagonal below the main, and so forth. 
    
    :type diag: int
    
    
    
The function returns the header, corresponding to a specified diagonal of the input array.


.. index:: GetDims

.. _GetDims:

GetDims
-------




.. function:: GetDims(arr)-> list

    Returns list of array dimensions





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    
The function returns a list of array dimensions.
In the case of 
``IplImage``
or 
:ref:`CvMat`
it always
returns a list of length 2.

.. index:: GetElemType

.. _GetElemType:

GetElemType
-----------




.. function:: GetElemType(arr)-> int

    Returns type of array elements.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    
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




.. function:: GetImage(arr) -> iplimage

    Returns image header for arbitrary array.





    
    :param arr: Input array 
    
    :type arr: :class:`CvMat`
    
    
    
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




.. function:: GetImageCOI(image)-> channel

    Returns the index of the channel of interest. 





    
    :param image: A pointer to the image header 
    
    :type image: :class:`IplImage`
    
    
    
Returns the channel of interest of in an IplImage. Returned values correspond to the 
``coi``
in 
:ref:`SetImageCOI`
.


.. index:: GetImageROI

.. _GetImageROI:

GetImageROI
-----------




.. function:: GetImageROI(image)-> CvRect

    Returns the image ROI.





    
    :param image: A pointer to the image header 
    
    :type image: :class:`IplImage`
    
    
    
If there is no ROI set, 
``cvRect(0,0,image->width,image->height)``
is returned.


.. index:: GetMat

.. _GetMat:

GetMat
------




.. function:: GetMat(arr, allowND=0) -> cvmat 

    Returns matrix header for arbitrary array.





    
    :param arr: Input array 
    
    :type arr: :class:`IplImage`
    
    
    :param allowND: If non-zero, the function accepts multi-dimensional dense arrays (CvMatND*) and returns 2D (if CvMatND has two dimensions) or 1D matrix (when CvMatND has 1 dimension or more than 2 dimensions). The array must be continuous. 
    
    :type allowND: int
    
    
    
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


.. index:: GetOptimalDFTSize

.. _GetOptimalDFTSize:

GetOptimalDFTSize
-----------------




.. function:: GetOptimalDFTSize(size0)-> int

    Returns optimal DFT size for a given vector size.





    
    :param size0: Vector size 
    
    :type size0: int
    
    
    
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



.. index:: GetReal1D

.. _GetReal1D:

GetReal1D
---------




.. function:: GetReal1D(arr, idx0)->float

    Return a specific element of single-channel 1D array.





    
    :param arr: Input array. Must have a single channel. 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: The first zero-based component of the element index 
    
    :type idx0: int
    
    
    
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




.. function:: GetReal2D(arr, idx0, idx1)->float

    Return a specific element of single-channel 2D array.





    
    :param arr: Input array. Must have a single channel. 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: The first zero-based component of the element index 
    
    :type idx0: int
    
    
    :param idx1: The second zero-based component of the element index 
    
    :type idx1: int
    
    
    
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




.. function:: GetReal3D(arr, idx0, idx1, idx2)->float

    Return a specific element of single-channel array.





    
    :param arr: Input array. Must have a single channel. 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: The first zero-based component of the element index 
    
    :type idx0: int
    
    
    :param idx1: The second zero-based component of the element index 
    
    :type idx1: int
    
    
    :param idx2: The third zero-based component of the element index 
    
    :type idx2: int
    
    
    
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




.. function:: GetRealND(arr, idx)->float

    Return a specific element of single-channel array.





    
    :param arr: Input array. Must have a single channel. 
    
    :type arr: :class:`CvArr`
    
    
    :param idx: Array of the element indices 
    
    :type idx: sequence of int
    
    
    
Returns a specific element of a single-channel array. If the array has
multiple channels, a runtime error is raised. Note that 
:ref:`Get`
function can be used safely for both single-channel and multiple-channel
arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).



.. index:: GetRow

.. _GetRow:

GetRow
------




.. function:: GetRow(arr,row)-> submat

    Returns array row.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param row: Zero-based index of the selected row 
    
    :type row: int
    
    
    :param submat: resulting single-row array 
    
    :type submat: :class:`CvMat`
    
    
    
The function 
``GetRow``
returns a single row from the input array.


.. index:: GetRows

.. _GetRows:

GetRows
-------




.. function:: GetRows(arr,startRow,endRow,deltaRow=1)-> submat

    Returns array row span.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param startRow: Zero-based index of the starting row (inclusive) of the span 
    
    :type startRow: int
    
    
    :param endRow: Zero-based index of the ending row (exclusive) of the span 
    
    :type endRow: int
    
    
    :param deltaRow: Index step in the row span. 
    
    :type deltaRow: int
    
    
    :param submat: resulting multi-row array 
    
    :type submat: :class:`CvMat`
    
    
    
The function 
``GetRows``
returns a row span from the input array.


.. index:: GetSize

.. _GetSize:

GetSize
-------




.. function:: GetSize(arr)-> CvSize

    Returns size of matrix or image ROI.





    
    :param arr: array header 
    
    :type arr: :class:`CvArr`
    
    
    
The function returns number of rows (CvSize::height) and number of columns (CvSize::width) of the input matrix or image. In the case of image the size of ROI is returned.



.. index:: GetSubRect

.. _GetSubRect:

GetSubRect
----------




.. function:: GetSubRect(arr, rect) -> cvmat

    Returns matrix header corresponding to the rectangular sub-array of input image or matrix.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param rect: Zero-based coordinates of the rectangle of interest 
    
    :type rect: :class:`CvRect`
    
    
    
The function returns header, corresponding to
a specified rectangle of the input array. In other words, it allows
the user to treat a rectangular part of input array as a stand-alone
array. ROI is taken into account by the function so the sub-array of
ROI is actually extracted.


.. index:: InRange

.. _InRange:

InRange
-------




.. function:: InRange(src,lower,upper,dst)-> None

    Checks that array elements lie between the elements of two other arrays.





    
    :param src: The first source array 
    
    :type src: :class:`CvArr`
    
    
    :param lower: The inclusive lower boundary array 
    
    :type lower: :class:`CvArr`
    
    
    :param upper: The exclusive upper boundary array 
    
    :type upper: :class:`CvArr`
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    :type dst: :class:`CvArr`
    
    
    
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




.. function:: InRangeS(src,lower,upper,dst)-> None

    Checks that array elements lie between two scalars.





    
    :param src: The first source array 
    
    :type src: :class:`CvArr`
    
    
    :param lower: The inclusive lower boundary 
    
    :type lower: :class:`CvScalar`
    
    
    :param upper: The exclusive upper boundary 
    
    :type upper: :class:`CvScalar`
    
    
    :param dst: The destination array, must have 8u or 8s type 
    
    :type dst: :class:`CvArr`
    
    
    
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


.. index:: InvSqrt

.. _InvSqrt:

InvSqrt
-------




.. function:: InvSqrt(value)-> float

    Calculates the inverse square root.





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
The function calculates the inverse square root of the argument, and normally it is faster than 
``1./sqrt(value)``
. If the argument is zero or negative, the result is not determined. Special values (
:math:`\pm \infty`
, NaN) are not handled.


.. index:: Inv

.. _Inv:

Inv
---




:ref:`Invert`

.. index:: 

.. _:







.. function:: Invert(src,dst,method=CV_LU)-> double

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




.. function:: IsInf(value)-> int

    Determines if the argument is Infinity.





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
The function returns 1 if the argument is 
:math:`\pm \infty`
(as defined by IEEE754 standard), 0 otherwise.


.. index:: IsNaN

.. _IsNaN:

IsNaN
-----




.. function:: IsNaN(value)-> int

    Determines if the argument is Not A Number.





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
The function returns 1 if the argument is Not A Number (as defined by IEEE754 standard), 0 otherwise.



.. index:: LUT

.. _LUT:

LUT
---




.. function:: LUT(src,dst,lut)-> None

    Performs a look-up table transform of an array.





    
    :param src: Source array of 8-bit elements 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array of a given depth and of the same number of channels as the source array 
    
    :type dst: :class:`CvArr`
    
    
    :param lut: Look-up table of 256 elements; should have the same depth as the destination array. In the case of multi-channel source and destination arrays, the table should either have a single-channel (in this case the same table is used for all channels) or the same number of channels as the source/destination array. 
    
    :type lut: :class:`CvArr`
    
    
    
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




.. function:: Log(src,dst)-> None

    Calculates the natural logarithm of every array element's absolute value.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination array, it should have  ``double``  type or the same type as the source 
    
    :type dst: :class:`CvArr`
    
    
    
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




.. function:: Mahalonobis(vec1,vec2,mat)-> None

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



.. index:: Max

.. _Max:

Max
---




.. function:: Max(src1,src2,dst)-> None

    Finds per-element maximum of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The function calculates per-element maximum of two arrays:



.. math::

    \texttt{dst} (I)= \max ( \texttt{src1} (I),  \texttt{src2} (I)) 


All the arrays must have a single channel, the same data type and the same size (or ROI size).



.. index:: MaxS

.. _MaxS:

MaxS
----




.. function:: MaxS(src,value,dst)-> None

    Finds per-element maximum of array and scalar.





    
    :param src: The first source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: The scalar value 
    
    :type value: float
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The function calculates per-element maximum of array and scalar:



.. math::

    \texttt{dst} (I)= \max ( \texttt{src} (I),  \texttt{value} ) 


All the arrays must have a single channel, the same data type and the same size (or ROI size).



.. index:: Merge

.. _Merge:

Merge
-----




.. function:: Merge(src0,src1,src2,src3,dst)-> None

    Composes a multi-channel array from several single-channel arrays or inserts a single channel into the array.





    
    :param src0: Input channel 0 
    
    :type src0: :class:`CvArr`
    
    
    :param src1: Input channel 1 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: Input channel 2 
    
    :type src2: :class:`CvArr`
    
    
    :param src3: Input channel 3 
    
    :type src3: :class:`CvArr`
    
    
    :param dst: Destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The function is the opposite to 
:ref:`Split`
. If the destination array has N channels then if the first N input channels are not NULL, they all are copied to the destination array; if only a single source channel of the first N is not NULL, this particular channel is copied into the destination array; otherwise an error is raised. The rest of the source channels (beyond the first N) must always be NULL. For IplImage 
:ref:`Copy`
with COI set can be also used to insert a single channel into the image.


.. index:: Min

.. _Min:

Min
---




.. function:: Min(src1,src2,dst)-> None

    Finds per-element minimum of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The function calculates per-element minimum of two arrays:



.. math::

    \texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I)) 


All the arrays must have a single channel, the same data type and the same size (or ROI size).



.. index:: MinMaxLoc

.. _MinMaxLoc:

MinMaxLoc
---------




.. function:: MinMaxLoc(arr,mask=NULL)-> (minVal,maxVal,minLoc,maxLoc)

    Finds global minimum and maximum in array or subarray.





    
    :param arr: The source array, single-channel or multi-channel with COI set 
    
    :type arr: :class:`CvArr`
    
    
    :param minVal: Pointer to returned minimum value 
    
    :type minVal: float
    
    
    :param maxVal: Pointer to returned maximum value 
    
    :type maxVal: float
    
    
    :param minLoc: Pointer to returned minimum location 
    
    :type minLoc: :class:`CvPoint`
    
    
    :param maxLoc: Pointer to returned maximum location 
    
    :type maxLoc: :class:`CvPoint`
    
    
    :param mask: The optional mask used to select a subarray 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: MinS(src,value,dst)-> None

    Finds per-element minimum of an array and a scalar.





    
    :param src: The first source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: The scalar value 
    
    :type value: float
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    
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




.. function:: MixChannels(src, dst, fromTo) -> None

    Copies several channels from input arrays to certain channels of output arrays





    
    :param src: Input arrays 
    
    :type src: :class:`cvarr_count`
    
    
    :param dst: Destination arrays 
    
    :type dst: :class:`cvarr_count`
    
    
    :param fromTo: The array of pairs of indices of the planes
        copied.  Each pair  ``fromTo[k]=(i,j)`` 
        means that i-th plane from  ``src``  is copied to the j-th plane in  ``dst`` , where continuous
        plane numbering is used both in the input array list and the output array list.
        As a special case, when the  ``fromTo[k][0]``  is negative, the corresponding output plane  ``j`` 
         is filled with zero.  
    
    :type fromTo: :class:`intpair`
    
    
    
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


    
            rgba = cv.CreateMat(100, 100, cv.CV_8UC4)
            bgr =  cv.CreateMat(100, 100, cv.CV_8UC3)
            alpha = cv.CreateMat(100, 100, cv.CV_8UC1)
            cv.Set(rgba, (1,2,3,4))
            cv.MixChannels([rgba], [bgr, alpha], [
               (0, 2),    # rgba[0] -> bgr[2]
               (1, 1),    # rgba[1] -> bgr[1]
               (2, 0),    # rgba[2] -> bgr[0]
               (3, 3)     # rgba[3] -> alpha[0]
            ])
    

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




.. function:: Mul(src1,src2,dst,scale)-> None

    Calculates the per-element product of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param scale: Optional scale factor 
    
    :type scale: float
    
    
    
The function calculates the per-element product of two arrays:



.. math::

    \texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I) 


All the arrays must have the same type and the same size (or ROI size).
For types that have limited range this operation is saturating.


.. index:: MulSpectrums

.. _MulSpectrums:

MulSpectrums
------------




.. function:: MulSpectrums(src1,src2,dst,flags)-> None

    Performs per-element multiplication of two Fourier spectrums.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array of the same type and the same size as the source arrays 
    
    :type dst: :class:`CvArr`
    
    
    :param flags: A combination of the following values; 
         
            * **CV_DXT_ROWS** treats each row of the arrays as a separate spectrum (see  :ref:`DFT`  parameters description). 
            
            * **CV_DXT_MUL_CONJ** conjugate the second source array before the multiplication. 
            
            
    
    :type flags: int
    
    
    
The function performs per-element multiplication of the two CCS-packed or complex matrices that are results of a real or complex Fourier transform.

The function, together with 
:ref:`DFT`
, may be used to calculate convolution of two arrays rapidly.



.. index:: MulTransposed

.. _MulTransposed:

MulTransposed
-------------




.. function:: MulTransposed(src,dst,order,delta=NULL,scale)-> None

    Calculates the product of an array and a transposed array.





    
    :param src: The source matrix 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination matrix. Must be  ``CV_32F``  or  ``CV_64F`` . 
    
    :type dst: :class:`CvArr`
    
    
    :param order: Order of multipliers 
    
    :type order: int
    
    
    :param delta: An optional array, subtracted from  ``src``  before multiplication 
    
    :type delta: :class:`CvArr`
    
    
    :param scale: An optional scaling 
    
    :type scale: float
    
    
    
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




.. function:: Norm(arr1,arr2,normType=CV_L2,mask=NULL)-> double

    Calculates absolute array norm, absolute difference norm, or relative difference norm.





    
    :param arr1: The first source image 
    
    :type arr1: :class:`CvArr`
    
    
    :param arr2: The second source image. If it is NULL, the absolute norm of  ``arr1``  is calculated, otherwise the absolute or relative norm of  ``arr1`` - ``arr2``  is calculated. 
    
    :type arr2: :class:`CvArr`
    
    
    :param normType: Type of norm, see the discussion 
    
    :type normType: int
    
    
    :param mask: The optional operation mask 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: Not(src,dst)-> None

    Performs per-element bit-wise inversion of array elements.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The function Not inverses every bit of every array element:




::


    
    dst(I)=~src(I)
    

..


.. index:: Or

.. _Or:

Or
--




.. function:: Or(src1,src2,dst,mask=NULL)-> None

    Calculates per-element bit-wise disjunction of two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
The function calculates per-element bit-wise disjunction of two arrays:




::


    
    dst(I)=src1(I)|src2(I)
    

..

In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.


.. index:: OrS

.. _OrS:

OrS
---




.. function:: OrS(src,value,dst,mask=NULL)-> None

    Calculates a per-element bit-wise disjunction of an array and a scalar.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: Scalar to use in the operation 
    
    :type value: :class:`CvScalar`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
The function OrS calculates per-element bit-wise disjunction of an array and a scalar:




::


    
    dst(I)=src(I)|value if mask(I)!=0
    

..

Prior to the actual operation, the scalar is converted to the same type as that of the array(s). In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.



.. index:: PerspectiveTransform

.. _PerspectiveTransform:

PerspectiveTransform
--------------------




.. function:: PerspectiveTransform(src,dst,mat)-> None

    Performs perspective matrix transformation of a vector array.





    
    :param src: The source three-channel floating-point array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination three-channel floating-point array 
    
    :type dst: :class:`CvArr`
    
    
    :param mat: :math:`3\times 3`  or  :math:`4 \times 4`  transformation matrix 
    
    :type mat: :class:`CvMat`
    
    
    
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




.. function:: PolarToCart(magnitude,angle,x,y,angleInDegrees=0)-> None

    Calculates Cartesian coordinates of 2d vectors represented in polar form.





    
    :param magnitude: The array of magnitudes. If it is NULL, the magnitudes are assumed to be all 1's. 
    
    :type magnitude: :class:`CvArr`
    
    
    :param angle: The array of angles, whether in radians or degrees 
    
    :type angle: :class:`CvArr`
    
    
    :param x: The destination array of x-coordinates, may be set to NULL if it is not needed 
    
    :type x: :class:`CvArr`
    
    
    :param y: The destination array of y-coordinates, mau be set to NULL if it is not needed 
    
    :type y: :class:`CvArr`
    
    
    :param angleInDegrees: The flag indicating whether the angles are measured in radians, which is default mode, or in degrees 
    
    :type angleInDegrees: int
    
    
    
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




.. function:: Pow(src,dst,power)-> None

    Raises every array element to a power.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination array, should be the same type as the source 
    
    :type dst: :class:`CvArr`
    
    
    :param power: The exponent of power 
    
    :type power: float
    
    
    
The function raises every element of the input array to 
``p``
:



.. math::

    \texttt{dst} [I] =  \fork{\texttt{src}(I)^p}{if \texttt{p} is integer}{|\texttt{src}(I)^p|}{otherwise} 


That is, for a non-integer power exponent the absolute values of input array elements are used. However, it is possible to get true values for negative values using some extra operations, as the following example, computing the cube root of array elements, shows:




.. doctest::


    
    >>> import cv
    >>> src = cv.CreateMat(1, 10, cv.CV_32FC1)
    >>> mask = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    >>> dst = cv.CreateMat(src.rows, src.cols, cv.CV_32FC1)
    >>> cv.CmpS(src, 0, mask, cv.CV_CMP_LT)         # find negative elements
    >>> cv.Pow(src, dst, 1. / 3)
    >>> cv.SubRS(dst, cv.ScalarAll(0), dst, mask)   # negate the results of negative inputs
    

..

For some values of 
``power``
, such as integer values, 0.5, and -0.5, specialized faster algorithms are used.


.. index:: RNG

.. _RNG:

RNG
---




.. function:: RNG(seed=-1LL)-> CvRNG

    Initializes a random number generator state.





    
    :param seed: 64-bit value used to initiate a random sequence 
    
    :type seed: :class:`int64`
    
    
    
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




.. function:: RandArr(rng,arr,distType,param1,param2)-> None

    Fills an array with random numbers and updates the RNG state.





    
    :param rng: RNG state initialized by  :ref:`RNG` 
    
    :type rng: :class:`CvRNG`
    
    
    :param arr: The destination array 
    
    :type arr: :class:`CvArr`
    
    
    :param distType: Distribution type 
         
            * **CV_RAND_UNI** uniform distribution 
            
            * **CV_RAND_NORMAL** normal or Gaussian distribution 
            
            
    
    :type distType: int
    
    
    :param param1: The first parameter of the distribution. In the case of a uniform distribution it is the inclusive lower boundary of the random numbers range. In the case of a normal distribution it is the mean value of the random numbers. 
    
    :type param1: :class:`CvScalar`
    
    
    :param param2: The second parameter of the distribution. In the case of a uniform distribution it is the exclusive upper boundary of the random numbers range. In the case of a normal distribution it is the standard deviation of the random numbers. 
    
    :type param2: :class:`CvScalar`
    
    
    
The function fills the destination array with uniformly
or normally distributed random numbers.


.. index:: RandInt

.. _RandInt:

RandInt
-------




.. function:: RandInt(rng)-> unsigned

    Returns a 32-bit unsigned integer and updates RNG.





    
    :param rng: RNG state initialized by  ``RandInit``  and, optionally, customized by  ``RandSetRange``  (though, the latter function does not affect the discussed function outcome) 
    
    :type rng: :class:`CvRNG`
    
    
    
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


.. index:: RandReal

.. _RandReal:

RandReal
--------




.. function:: RandReal(rng)-> double

    Returns a floating-point random number and updates RNG.





    
    :param rng: RNG state initialized by  :ref:`RNG` 
    
    :type rng: :class:`CvRNG`
    
    
    
The function returns a uniformly-distributed random floating-point number between 0 and 1 (1 is not included).


.. index:: Reduce

.. _Reduce:

Reduce
------




.. function:: Reduce(src,dst,dim=-1,op=CV_REDUCE_SUM)-> None

    Reduces a matrix to a vector.





    
    :param src: The input matrix. 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The output single-row/single-column vector that accumulates somehow all the matrix rows/columns. 
    
    :type dst: :class:`CvArr`
    
    
    :param dim: The dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row, 1 means that the matrix is reduced to a single column and -1 means that the dimension is chosen automatically by analysing the dst size. 
    
    :type dim: int
    
    
    :param op: The reduction operation. It can take of the following values: 
         
            * **CV_REDUCE_SUM** The output is the sum of all of the matrix's rows/columns. 
            
            * **CV_REDUCE_AVG** The output is the mean vector of all of the matrix's rows/columns. 
            
            * **CV_REDUCE_MAX** The output is the maximum (column/row-wise) of all of the matrix's rows/columns. 
            
            * **CV_REDUCE_MIN** The output is the minimum (column/row-wise) of all of the matrix's rows/columns. 
            
            
    
    :type op: int
    
    
    
The function reduces matrix to a vector by treating the matrix rows/columns as a set of 1D vectors and performing the specified operation on the vectors until a single row/column is obtained. For example, the function can be used to compute horizontal and vertical projections of an raster image. In the case of 
``CV_REDUCE_SUM``
and 
``CV_REDUCE_AVG``
the output may have a larger element bit-depth to preserve accuracy. And multi-channel arrays are also supported in these two reduction modes. 


.. index:: Repeat

.. _Repeat:

Repeat
------




.. function:: Repeat(src,dst)-> None

    Fill the destination array with repeated copies of the source array.





    
    :param src: Source array, image or matrix 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array, image or matrix 
    
    :type dst: :class:`CvArr`
    
    
    
The function fills the destination array with repeated copies of the source array:




::


    
    dst(i,j)=src(i mod rows(src), j mod cols(src))
    

..

So the destination array may be as larger as well as smaller than the source array.


.. index:: ResetImageROI

.. _ResetImageROI:

ResetImageROI
-------------




.. function:: ResetImageROI(image)-> None

    Resets the image ROI to include the entire image and releases the ROI structure.





    
    :param image: A pointer to the image header 
    
    :type image: :class:`IplImage`
    
    
    
This produces a similar result to the following



::


    
    cv.SetImageROI(image, (0, 0, image.width, image.height))
    cv.SetImageCOI(image, 0)
    

..


.. index:: Reshape

.. _Reshape:

Reshape
-------




.. function:: Reshape(arr, newCn, newRows=0) -> cvmat

    Changes shape of matrix/image without copying data.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param newCn: New number of channels. 'newCn = 0' means that the number of channels remains unchanged. 
    
    :type newCn: int
    
    
    :param newRows: New number of rows. 'newRows = 0' means that the number of rows remains unchanged unless it needs to be changed according to  ``newCn``  value. 
    
    :type newRows: int
    
    
    
The function initializes the CvMat header so that it points to the same data as the original array but has a different shape - different number of channels, different number of rows, or both.


.. index:: ReshapeMatND

.. _ReshapeMatND:

ReshapeMatND
------------




.. function:: ReshapeMatND(arr, newCn, newDims) -> cvmat

    Changes the shape of a multi-dimensional array without copying the data.





    
    :param arr: Input array 
    
    :type arr: :class:`CvMat`
    
    
    :param newCn: New number of channels.  :math:`\texttt{newCn} = 0`  means that the number of channels remains unchanged. 
    
    :type newCn: int
    
    
    :param newDims: List of new dimensions. 
    
    :type newDims: sequence of int
    
    
    
Returns a new 
:ref:`CvMatND`
that shares the same data as 
``arr``
but has different dimensions or number of channels.  The only requirement
is that the total length of the data is unchanged.




.. doctest::


    
    >>> import cv
    >>> mat = cv.CreateMatND([24], cv.CV_32FC1)
    >>> print cv.GetDims(cv.ReshapeMatND(mat, 0, [8, 3]))
    (8, 3)
    >>> m2 = cv.ReshapeMatND(mat, 4, [3, 2])
    >>> print cv.GetDims(m2)
    (3, 2)
    >>> print m2.channels
    4
    

..


.. index:: Round

.. _Round:

Round
-----




.. function:: Round(value) -> int

    Converts a floating-point number to the nearest integer value.





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
On some architectures this function is much faster than the standard cast
operations. If the absolute value of the argument is greater than
:math:`2^{31}`
, the result is not determined. Special values (
:math:`\pm \infty`
, NaN)
are not handled.


.. index:: Floor

.. _Floor:

Floor
-----




.. function:: Floor(value) -> int

    Converts a floating-point number to the nearest integer value that is not larger than the argument.





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
On some architectures this function is much faster than the standard cast
operations. If the absolute value of the argument is greater than
:math:`2^{31}`
, the result is not determined. Special values (
:math:`\pm \infty`
, NaN)
are not handled.


.. index:: Ceil

.. _Ceil:

Ceil
----




.. function:: Ceil(value) -> int

    Converts a floating-point number to the nearest integer value that is not smaller than the argument.





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
On some architectures this function is much faster than the standard cast
operations. If the absolute value of the argument is greater than
:math:`2^{31}`
, the result is not determined. Special values (
:math:`\pm \infty`
, NaN)
are not handled.


.. index:: ScaleAdd

.. _ScaleAdd:

ScaleAdd
--------




.. function:: ScaleAdd(src1,scale,src2,dst)-> None

    Calculates the sum of a scaled array and another array.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param scale: Scale factor for the first array 
    
    :type scale: :class:`CvScalar`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    
The function calculates the sum of a scaled array and another array:



.. math::

    \texttt{dst} (I)= \texttt{scale} \, \texttt{src1} (I) +  \texttt{src2} (I) 


All array parameters should have the same type and the same size.


.. index:: Set

.. _Set:

Set
---




.. function:: Set(arr,value,mask=NULL)-> None

    Sets every element of an array to a given value.





    
    :param arr: The destination array 
    
    :type arr: :class:`CvArr`
    
    
    :param value: Fill value 
    
    :type value: :class:`CvScalar`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
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


.. index:: Set1D

.. _Set1D:

Set1D
-----




.. function::  Set1D(arr, idx, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx: Zero-based element index 
    
    :type idx: int
    
    
    :param value: The value to assign to the element 
    
    :type value: :class:`CvScalar`
    
    
    
Sets a specific array element.  Array must have dimension 1.


.. index:: Set2D

.. _Set2D:

Set2D
-----




.. function::  Set2D(arr, idx0, idx1, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: Zero-based element row index 
    
    :type idx0: int
    
    
    :param idx1: Zero-based element column index 
    
    :type idx1: int
    
    
    :param value: The value to assign to the element 
    
    :type value: :class:`CvScalar`
    
    
    
Sets a specific array element.  Array must have dimension 2.


.. index:: Set3D

.. _Set3D:

Set3D
-----




.. function::  Set3D(arr, idx0, idx1, idx2, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: Zero-based element index 
    
    :type idx0: int
    
    
    :param idx1: Zero-based element index 
    
    :type idx1: int
    
    
    :param idx2: Zero-based element index 
    
    :type idx2: int
    
    
    :param value: The value to assign to the element 
    
    :type value: :class:`CvScalar`
    
    
    
Sets a specific array element.  Array must have dimension 3.


.. index:: SetND

.. _SetND:

SetND
-----




.. function::  SetND(arr, indices, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param indices: List of zero-based element indices 
    
    :type indices: sequence of int
    
    
    :param value: The value to assign to the element 
    
    :type value: :class:`CvScalar`
    
    
    
Sets a specific array element.  The length of array indices must be the same as the dimension of the array.

.. index:: SetData

.. _SetData:

SetData
-------




.. function:: SetData(arr, data, step)-> None

    Assigns user data to the array header.





    
    :param arr: Array header 
    
    :type arr: :class:`CvArr`
    
    
    :param data: User data 
    
    :type data: object
    
    
    :param step: Full row length in bytes 
    
    :type step: int
    
    
    
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




.. function:: SetIdentity(mat,value=1)-> None

    Initializes a scaled identity matrix.





    
    :param mat: The matrix to initialize (not necesserily square) 
    
    :type mat: :class:`CvArr`
    
    
    :param value: The value to assign to the diagonal elements 
    
    :type value: :class:`CvScalar`
    
    
    
The function initializes a scaled identity matrix:



.. math::

    \texttt{arr} (i,j)= \fork{\texttt{value}}{ if $i=j$}{0}{otherwise} 



.. index:: SetImageCOI

.. _SetImageCOI:

SetImageCOI
-----------




.. function:: SetImageCOI(image, coi)-> None

    Sets the channel of interest in an IplImage.





    
    :param image: A pointer to the image header 
    
    :type image: :class:`IplImage`
    
    
    :param coi: The channel of interest. 0 - all channels are selected, 1 - first channel is selected, etc. Note that the channel indices become 1-based. 
    
    :type coi: int
    
    
    
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




.. function:: SetImageROI(image, rect)-> None

    Sets an image Region Of Interest (ROI) for a given rectangle.





    
    :param image: A pointer to the image header 
    
    :type image: :class:`IplImage`
    
    
    :param rect: The ROI rectangle 
    
    :type rect: :class:`CvRect`
    
    
    
If the original image ROI was 
``NULL``
and the 
``rect``
is not the whole image, the ROI structure is allocated.

Most OpenCV functions support the use of ROI and treat the image rectangle as a separate image. For example, all of the pixel coordinates are counted from the top-left (or bottom-left) corner of the ROI, not the original image.


.. index:: SetReal1D

.. _SetReal1D:

SetReal1D
---------




.. function::  SetReal1D(arr, idx, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx: Zero-based element index 
    
    :type idx: int
    
    
    :param value: The value to assign to the element 
    
    :type value: float
    
    
    
Sets a specific array element.  Array must have dimension 1.


.. index:: SetReal2D

.. _SetReal2D:

SetReal2D
---------




.. function::  SetReal2D(arr, idx0, idx1, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: Zero-based element row index 
    
    :type idx0: int
    
    
    :param idx1: Zero-based element column index 
    
    :type idx1: int
    
    
    :param value: The value to assign to the element 
    
    :type value: float
    
    
    
Sets a specific array element.  Array must have dimension 2.


.. index:: SetReal3D

.. _SetReal3D:

SetReal3D
---------




.. function::  SetReal3D(arr, idx0, idx1, idx2, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param idx0: Zero-based element index 
    
    :type idx0: int
    
    
    :param idx1: Zero-based element index 
    
    :type idx1: int
    
    
    :param idx2: Zero-based element index 
    
    :type idx2: int
    
    
    :param value: The value to assign to the element 
    
    :type value: float
    
    
    
Sets a specific array element.  Array must have dimension 3.


.. index:: SetRealND

.. _SetRealND:

SetRealND
---------




.. function::  SetRealND(arr, indices, value) -> None 

    Set a specific array element.





    
    :param arr: Input array 
    
    :type arr: :class:`CvArr`
    
    
    :param indices: List of zero-based element indices 
    
    :type indices: sequence of int
    
    
    :param value: The value to assign to the element 
    
    :type value: float
    
    
    
Sets a specific array element.  The length of array indices must be the same as the dimension of the array.

.. index:: SetZero

.. _SetZero:

SetZero
-------




.. function:: SetZero(arr)-> None

    Clears the array.





    
    :param arr: Array to be cleared 
    
    :type arr: :class:`CvArr`
    
    
    
The function clears the array. In the case of dense arrays (CvMat, CvMatND or IplImage), cvZero(array) is equivalent to cvSet(array,cvScalarAll(0),0).
In the case of sparse arrays all the elements are removed.


.. index:: Solve

.. _Solve:

Solve
-----




.. function:: Solve(A,B,X,method=CV_LU)-> None

    Solves a linear system or least-squares problem.





    
    :param A: The source matrix 
    
    :type A: :class:`CvArr`
    
    
    :param B: The right-hand part of the linear system 
    
    :type B: :class:`CvArr`
    
    
    :param X: The output solution 
    
    :type X: :class:`CvArr`
    
    
    :param method: The solution (matrix inversion) method 
        
               
            * **CV_LU** Gaussian elimination with optimal pivot element chosen 
            
              
            * **CV_SVD** Singular value decomposition (SVD) method 
            
              
            * **CV_SVD_SYM** SVD method for a symmetric positively-defined matrix. 
            
            
    
    :type method: int
    
    
    
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




.. function:: SolveCubic(coeffs,roots)-> None

    Finds the real roots of a cubic equation.





    
    :param coeffs: The equation coefficients, an array of 3 or 4 elements 
    
    :type coeffs: :class:`CvMat`
    
    
    :param roots: The output array of real roots which should have 3 elements 
    
    :type roots: :class:`CvMat`
    
    
    
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




.. function:: Split(src,dst0,dst1,dst2,dst3)-> None

    Divides multi-channel array into several single-channel arrays or extracts a single channel from the array.





    
    :param src: Source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst0: Destination channel 0 
    
    :type dst0: :class:`CvArr`
    
    
    :param dst1: Destination channel 1 
    
    :type dst1: :class:`CvArr`
    
    
    :param dst2: Destination channel 2 
    
    :type dst2: :class:`CvArr`
    
    
    :param dst3: Destination channel 3 
    
    :type dst3: :class:`CvArr`
    
    
    
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




.. function:: Sqrt(value)-> float

    Calculates the square root.





    
    :param value: The input floating-point value 
    
    :type value: float
    
    
    
The function calculates the square root of the argument. If the argument is negative, the result is not determined.


.. index:: Sub

.. _Sub:

Sub
---




.. function:: Sub(src1,src2,dst,mask=NULL)-> None

    Computes the per-element difference between two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: SubRS(src,value,dst,mask=NULL)-> None

    Computes the difference between a scalar and an array.





    
    :param src: The first source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: Scalar to subtract from 
    
    :type value: :class:`CvScalar`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: SubS(src,value,dst,mask=NULL)-> None

    Computes the difference between an array and a scalar.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: Subtracted scalar 
    
    :type value: :class:`CvScalar`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
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




.. function:: Sum(arr)-> CvScalar

    Adds up array elements.





    
    :param arr: The array 
    
    :type arr: :class:`CvArr`
    
    
    
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




.. function:: SVBkSb(W,U,V,B,X,flags)-> None

    Performs singular value back substitution.





    
    :param W: Matrix or vector of singular values 
    
    :type W: :class:`CvArr`
    
    
    :param U: Left orthogonal matrix (tranposed, perhaps) 
    
    :type U: :class:`CvArr`
    
    
    :param V: Right orthogonal matrix (tranposed, perhaps) 
    
    :type V: :class:`CvArr`
    
    
    :param B: The matrix to multiply the pseudo-inverse of the original matrix  ``A``  by. This is an optional parameter. If it is omitted then it is assumed to be an identity matrix of an appropriate size (so that  ``X``  will be the reconstructed pseudo-inverse of  ``A`` ). 
    
    :type B: :class:`CvArr`
    
    
    :param X: The destination matrix: result of back substitution 
    
    :type X: :class:`CvArr`
    
    
    :param flags: Operation flags, should match exactly to the  ``flags``  passed to  :ref:`SVD` 
    
    :type flags: int
    
    
    
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




.. function:: SVD(A,W, U = None, V = None, flags=0)-> None

    Performs singular value decomposition of a real floating-point matrix.





    
    :param A: Source  :math:`\texttt{M} \times \texttt{N}`  matrix 
    
    :type A: :class:`CvArr`
    
    
    :param W: Resulting singular value diagonal matrix ( :math:`\texttt{M} \times \texttt{N}`  or  :math:`\min(\texttt{M}, \texttt{N})  \times \min(\texttt{M}, \texttt{N})` ) or  :math:`\min(\texttt{M},\texttt{N}) \times 1`  vector of the singular values 
    
    :type W: :class:`CvArr`
    
    
    :param U: Optional left orthogonal matrix,  :math:`\texttt{M} \times \min(\texttt{M}, \texttt{N})`  (when  ``CV_SVD_U_T``  is not set), or  :math:`\min(\texttt{M},\texttt{N}) \times \texttt{M}`  (when  ``CV_SVD_U_T``  is set), or  :math:`\texttt{M} \times \texttt{M}`  (regardless of  ``CV_SVD_U_T``  flag). 
    
    :type U: :class:`CvArr`
    
    
    :param V: Optional right orthogonal matrix,  :math:`\texttt{N} \times \min(\texttt{M}, \texttt{N})`  (when  ``CV_SVD_V_T``  is not set), or  :math:`\min(\texttt{M},\texttt{N}) \times \texttt{N}`  (when  ``CV_SVD_V_T``  is set), or  :math:`\texttt{N} \times \texttt{N}`  (regardless of  ``CV_SVD_V_T``  flag). 
    
    :type V: :class:`CvArr`
    
    
    :param flags: Operation flags; can be 0 or a combination of the following values: 
        
                
            * **CV_SVD_MODIFY_A** enables modification of matrix  ``A``  during the operation. It speeds up the processing. 
            
               
            * **CV_SVD_U_T** means that the transposed matrix  ``U``  is returned. Specifying the flag speeds up the processing. 
            
               
            * **CV_SVD_V_T** means that the transposed matrix  ``V``  is returned. Specifying the flag speeds up the processing. 
            
            
    
    :type flags: int
    
    
    
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




.. function:: Trace(mat)-> CvScalar

    Returns the trace of a matrix.





    
    :param mat: The source matrix 
    
    :type mat: :class:`CvArr`
    
    
    
The function returns the sum of the diagonal elements of the matrix 
``src1``
.



.. math::

    tr( \texttt{mat} ) =  \sum _i  \texttt{mat} (i,i)  



.. index:: Transform

.. _Transform:

Transform
---------




.. function:: Transform(src,dst,transmat,shiftvec=NULL)-> None

    Performs matrix transformation of every array element.





    
    :param src: The first source array 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param transmat: Transformation matrix 
    
    :type transmat: :class:`CvMat`
    
    
    :param shiftvec: Optional shift vector 
    
    :type shiftvec: :class:`CvMat`
    
    
    
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




.. function:: Transpose(src,dst)-> None

    Transposes a matrix.





    
    :param src: The source matrix 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination matrix 
    
    :type dst: :class:`CvArr`
    
    
    
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




.. function:: Xor(src1,src2,dst,mask=NULL)-> None

    Performs per-element bit-wise "exclusive or" operation on two arrays.





    
    :param src1: The first source array 
    
    :type src1: :class:`CvArr`
    
    
    :param src2: The second source array 
    
    :type src2: :class:`CvArr`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
The function calculates per-element bit-wise logical conjunction of two arrays:




::


    
    dst(I)=src1(I)^src2(I) if mask(I)!=0
    

..

In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size.


.. index:: XorS

.. _XorS:

XorS
----




.. function:: XorS(src,value,dst,mask=NULL)-> None

    Performs per-element bit-wise "exclusive or" operation on an array and a scalar.





    
    :param src: The source array 
    
    :type src: :class:`CvArr`
    
    
    :param value: Scalar to use in the operation 
    
    :type value: :class:`CvScalar`
    
    
    :param dst: The destination array 
    
    :type dst: :class:`CvArr`
    
    
    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed 
    
    :type mask: :class:`CvArr`
    
    
    
The function XorS calculates per-element bit-wise conjunction of an array and a scalar:




::


    
    dst(I)=src(I)^value if mask(I)!=0
    

..

Prior to the actual operation, the scalar is converted to the same type as that of the array(s). In the case of floating-point arrays their bit representations are used for the operation. All the arrays must have the same type, except the mask, and the same size


.. index:: mGet

.. _mGet:

mGet
----




.. function:: mGet(mat,row,col)-> double

    Returns the particular element of single-channel floating-point matrix.





    
    :param mat: Input matrix 
    
    :type mat: :class:`CvMat`
    
    
    :param row: The zero-based index of row 
    
    :type row: int
    
    
    :param col: The zero-based index of column 
    
    :type col: int
    
    
    
The function is a fast replacement for 
:ref:`GetReal2D`
in the case of single-channel floating-point matrices. It is faster because
it is inline, it does fewer checks for array type and array element type,
and it checks for the row and column ranges only in debug mode.


.. index:: mSet

.. _mSet:

mSet
----




.. function:: mSet(mat,row,col,value)-> None

    Returns a specific element of a single-channel floating-point matrix.





    
    :param mat: The matrix 
    
    :type mat: :class:`CvMat`
    
    
    :param row: The zero-based index of row 
    
    :type row: int
    
    
    :param col: The zero-based index of column 
    
    :type col: int
    
    
    :param value: The new value of the matrix element 
    
    :type value: float
    
    
    
The function is a fast replacement for 
:ref:`SetReal2D`
in the case of single-channel floating-point matrices. It is faster because
it is inline, it does fewer checks for array type and array element type, 
and it checks for the row and column ranges only in debug mode.

