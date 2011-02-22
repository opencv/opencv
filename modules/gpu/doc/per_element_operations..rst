Per-element Operations.
=======================

.. highlight:: cpp



.. index:: gpu::add


cv::gpu::add
------------

`id=0.387694196113 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Aadd>`__




.. cfunction:: void add(const GpuMat\& a, const GpuMat\& b, GpuMat\& c)

    Computes matrix-matrix or matrix-scalar sum.





    
    :param a: First source matrix.  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_32SC1``  and  ``CV_32FC1``  matrices are supported for now. 
    
    
    :param b: Second source matrix. Must have the same size and type as  ``a`` . 
    
    
    :param c: Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    


.. cfunction:: void add(const GpuMat\& a, const Scalar\& sc, GpuMat\& c)





    
    * **a** Source matrix.  ``CV_32FC1``  and  ``CV_32FC2``  matrixes are supported for now. 
    
    
    * **b** Source scalar to be added to the source matrix. 
    
    
    * **c** Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    
See also: 
:func:`add`
.



.. index:: gpu::subtract


cv::gpu::subtract
-----------------

`id=0.316386979537 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Asubtract>`__




.. cfunction:: void subtract(const GpuMat\& a, const GpuMat\& b, GpuMat\& c)

    Subtracts matrix from another matrix (or scalar from matrix).





    
    :param a: First source matrix.  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_32SC1``  and  ``CV_32FC1``  matrices are supported for now. 
    
    
    :param b: Second source matrix. Must have the same size and type as  ``a`` . 
    
    
    :param c: Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    


.. cfunction:: void subtract(const GpuMat\& a, const Scalar\& sc, GpuMat\& c)





    
    * **a** Source matrix.   ``CV_32FC1``  and  ``CV_32FC2``  matrixes are supported for now. 
    
    
    * **b** Scalar to be subtracted from the source matrix elements. 
    
    
    * **c** Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    
See also: 
:func:`subtract`
.



.. index:: gpu::multiply


cv::gpu::multiply
-----------------

`id=0.12843407457 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Amultiply>`__




.. cfunction:: void multiply(const GpuMat\& a, const GpuMat\& b, GpuMat\& c)

    Computes per-element product of two matrices (or of matrix and scalar).





    
    :param a: First source matrix.  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_32SC1``  and  ``CV_32FC1``  matrices are supported for now. 
    
    
    :param b: Second source matrix. Must have the same size and type as  ``a`` . 
    
    
    :param c: Destionation matrix. Will have the same size and type as  ``a`` . 
    
    
    


.. cfunction:: void multiply(const GpuMat\& a, const Scalar\& sc, GpuMat\& c)





    
    * **a** Source matrix.   ``CV_32FC1``  and  ``CV_32FC2``  matrixes are supported for now. 
    
    
    * **b** Scalar to be multiplied by. 
    
    
    * **c** Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    
See also: 
:func:`multiply`
.



.. index:: gpu::divide


cv::gpu::divide
---------------

`id=0.178699823123 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Adivide>`__




.. cfunction:: void divide(const GpuMat\& a, const GpuMat\& b, GpuMat\& c)

    Performs per-element division of two matrices (or division of matrix by scalar).





    
    :param a: First source matrix.  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_32SC1``  and  ``CV_32FC1``  matrices are supported for now. 
    
    
    :param b: Second source matrix. Must have the same size and type as  ``a`` . 
    
    
    :param c: Destionation matrix. Will have the same size and type as  ``a`` . 
    
    
    


.. cfunction:: void divide(const GpuMat\& a, const Scalar\& sc, GpuMat\& c)





    
    * **a** Source matrix.   ``CV_32FC1``  and  ``CV_32FC2``  matrixes are supported for now. 
    
    
    * **b** Scalar to be divided by. 
    
    
    * **c** Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    
This function in contrast to 
:func:`divide`
uses round-down rounding mode.

See also: 
:func:`divide`
.



.. index:: gpu::exp


cv::gpu::exp
------------

`id=0.0437158645609 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Aexp>`__




.. cfunction:: void exp(const GpuMat\& a, GpuMat\& b)

    Computes exponent of each matrix element.





    
    :param a: Source matrix.  ``CV_32FC1``  matrixes are supported for now. 
    
    
    :param b: Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    
See also: 
:func:`exp`
.



.. index:: gpu::log


cv::gpu::log
------------

`id=0.726514219732 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Alog>`__




.. cfunction:: void log(const GpuMat\& a, GpuMat\& b)

    Computes natural logarithm of absolute value of each matrix element.





    
    :param a: Source matrix.  ``CV_32FC1``  matrixes are supported for now. 
    
    
    :param b: Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    
See also: 
:func:`log`
.



.. index:: gpu::absdiff


cv::gpu::absdiff
----------------

`id=0.0449517502969 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Aabsdiff>`__




.. cfunction:: void absdiff(const GpuMat\& a, const GpuMat\& b, GpuMat\& c)

    Computes per-element absolute difference of two matrices (or of matrix and scalar).





    
    :param a: First source matrix.  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_32SC1``  and  ``CV_32FC1``  matrices are supported for now. 
    
    
    :param b: Second source matrix. Must have the same size and type as  ``a`` . 
    
    
    :param c: Destionation matrix. Will have the same size and type as  ``a`` . 
    
    
    


.. cfunction:: void absdiff(const GpuMat\& a, const Scalar\& s, GpuMat\& c)





    
    * **a** Source matrix.  ``CV_32FC1``  matrixes are supported for now. 
    
    
    * **b** Scalar to be subtracted from the source matrix elements. 
    
    
    * **c** Destination matrix. Will have the same size and type as  ``a`` . 
    
    
    
See also: 
:func:`absdiff`
.



.. index:: gpu::compare


cv::gpu::compare
----------------

`id=0.346307736999 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Acompare>`__




.. cfunction:: void compare(const GpuMat\& a, const GpuMat\& b, GpuMat\& c, int cmpop)

    Compares elements of two matrices.





    
    :param a: First source matrix.  ``CV_8UC4``  and  ``CV_32FC1``  matrices are supported for now. 
    
    
    :param b: Second source matrix. Must have the same size and type as  ``a`` . 
    
    
    :param c: Destination matrix. Will have the same size as  ``a``  and be  ``CV_8UC1``  type. 
    
    
    :param cmpop: Flag specifying the relation between the elements to be checked: 
          
            * **CMP_EQ** :math:`=` 
             
            * **CMP_GT** :math:`>` 
             
            * **CMP_GE** :math:`\ge` 
             
            * **CMP_LT** :math:`<` 
             
            * **CMP_LE** :math:`\le` 
             
            * **CMP_NE** :math:`\ne` 
             
             
    
    
    
See also: 
:func:`compare`
.



.. index:: cv::gpu::bitwise_not

.. _cv::gpu::bitwise_not:

cv::gpu::bitwise_not
--------------------

`id=0.242780097451 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/cv%3A%3Agpu%3A%3Abitwise_not>`__




.. cfunction:: void bitwise_not(const GpuMat\& src, GpuMat\& dst,   const GpuMat\& mask=GpuMat())



.. cfunction:: void bitwise_not(const GpuMat\& src, GpuMat\& dst,   const GpuMat\& mask, const Stream\& stream)

    Performs per-element bitwise inversion.





    
    :param src: Source matrix. 
    
    
    :param dst: Destination matrix. Will have the same size and type as  ``src`` . 
    
    
    :param mask: Optional operation mask. 8-bit single channel image. 
    
    
    :param stream: Stream for the asynchronous version. 
    
    
    
See also: 
.



.. index:: cv::gpu::bitwise_or

.. _cv::gpu::bitwise_or:

cv::gpu::bitwise_or
-------------------

`id=0.762303417062 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/cv%3A%3Agpu%3A%3Abitwise_or>`__




.. cfunction:: void bitwise_or(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const GpuMat\& mask=GpuMat())



.. cfunction:: void bitwise_or(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const GpuMat\& mask, const Stream\& stream)

    Performs per-element bitwise disjunction of two matrices.





    
    :param src1: First source matrix. 
    
    
    :param src2: Second source matrix. It must have the same size and type as  ``src1`` . 
    
    
    :param dst: Destination matrix. Will have the same size and type as  ``src1`` . 
    
    
    :param mask: Optional operation mask. 8-bit single channel image. 
    
    
    :param stream: Stream for the asynchronous version. 
    
    
    
See also: 
.



.. index:: cv::gpu::bitwise_and

.. _cv::gpu::bitwise_and:

cv::gpu::bitwise_and
--------------------

`id=0.621591376205 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/cv%3A%3Agpu%3A%3Abitwise_and>`__




.. cfunction:: void bitwise_and(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const GpuMat\& mask=GpuMat())



.. cfunction:: void bitwise_and(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const GpuMat\& mask, const Stream\& stream)

    Performs per-element bitwise conjunction of two matrices.





    
    :param src1: First source matrix. 
    
    
    :param src2: Second source matrix. It must have the same size and type as  ``src1`` . 
    
    
    :param dst: Destination matrix. Will have the same size and type as  ``src1`` . 
    
    
    :param mask: Optional operation mask. 8-bit single channel image. 
    
    
    :param stream: Stream for the asynchronous version. 
    
    
    
See also: 
.



.. index:: cv::gpu::bitwise_xor

.. _cv::gpu::bitwise_xor:

cv::gpu::bitwise_xor
--------------------

`id=0.684217951074 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/cv%3A%3Agpu%3A%3Abitwise_xor>`__




.. cfunction:: void bitwise_xor(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const GpuMat\& mask=GpuMat())



.. cfunction:: void bitwise_xor(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const GpuMat\& mask, const Stream\& stream)

    Performs per-element bitwise "exclusive or" of two matrices.





    
    :param src1: First source matrix. 
    
    
    :param src2: Second source matrix. It must have the same size and type as  ``src1`` . 
    
    
    :param dst: Destination matrix. Will have the same size and type as  ``src1`` . 
    
    
    :param mask: Optional operation mask. 8-bit single channel image. 
    
    
    :param stream: Stream for the asynchronous version. 
    
    
    
See also: 
.



.. index:: gpu::min


cv::gpu::min
------------

`id=0.276176266158 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Amin>`__




.. cfunction:: void min(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst)



.. cfunction:: void min(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const Stream\& stream)

    Computes per-element minimum of two matrices (or of matrix and scalar).





    
    :param src1: First source matrix. 
    
    
    :param src2: Second source matrix. 
    
    
    :param dst: Destination matrix. Will have the same size and type as  ``src1`` . 
    
    
    :param stream: Stream for the asynchronous version. 
    
    
    


.. cfunction:: void min(const GpuMat\& src1, double src2, GpuMat\& dst)



.. cfunction:: void min(const GpuMat\& src1, double src2, GpuMat\& dst,   const Stream\& stream)





    
    * **src1** Source matrix. 
    
    
    * **src2** Scalar to be compared with. 
    
    
    * **dst** Destination matrix. Will have the same size and type as  ``src1`` . 
    
    
    * **stream** Stream for the asynchronous version. 
    
    
    
See also: 
:func:`min`
.



.. index:: gpu::max


cv::gpu::max
------------

`id=0.175554622377 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3Amax>`__




.. cfunction:: void max(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst)



.. cfunction:: void max(const GpuMat\& src1, const GpuMat\& src2, GpuMat\& dst,   const Stream\& stream)

    Computes per-element maximum of two matrices (or of matrix and scalar).





    
    :param src1: First source matrix. 
    
    
    :param src2: Second source matrix. 
    
    
    :param dst: Destination matrix. Will have the same size and type as  ``src1`` . 
    
    
    :param stream: Stream for the asynchronous version. 
    
    
    


.. cfunction:: void max(const GpuMat\& src1, double src2, GpuMat\& dst)



.. cfunction:: void max(const GpuMat\& src1, double src2, GpuMat\& dst,   const Stream\& stream)





    
    * **src1** Source matrix. 
    
    
    * **src2** Scalar to be compared with. 
    
    
    * **dst** Destination matrix. Will have the same size and type as  ``src1`` . 
    
    
    * **stream** Stream for the asynchronous version. 
    
    
    
See also: 
:func:`max`
.
