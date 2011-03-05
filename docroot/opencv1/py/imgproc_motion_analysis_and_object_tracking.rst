Motion Analysis and Object Tracking
===================================

.. highlight:: python



.. index:: Acc

.. _Acc:

Acc
---

`id=0.629029815041 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/Acc>`__


.. function:: Acc(image,sum,mask=NULL)-> None

    Adds a frame to an accumulator.





    
    :param image: Input image, 1- or 3-channel, 8-bit or 32-bit floating point. (each channel of multi-channel image is processed independently) 
    
    :type image: :class:`CvArr`
    
    
    :param sum: Accumulator with the same number of channels as input image, 32-bit or 64-bit floating-point 
    
    :type sum: :class:`CvArr`
    
    
    :param mask: Optional operation mask 
    
    :type mask: :class:`CvArr`
    
    
    
The function adds the whole image 
``image``
or its selected region to the accumulator 
``sum``
:



.. math::

    \texttt{sum} (x,y)  \leftarrow \texttt{sum} (x,y) +  \texttt{image} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0  



.. index:: MultiplyAcc

.. _MultiplyAcc:

MultiplyAcc
-----------

`id=0.767428702085 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/MultiplyAcc>`__


.. function:: MultiplyAcc(image1,image2,acc,mask=NULL)-> None

    Adds the product of two input images to the accumulator.





    
    :param image1: First input image, 1- or 3-channel, 8-bit or 32-bit floating point (each channel of multi-channel image is processed independently) 
    
    :type image1: :class:`CvArr`
    
    
    :param image2: Second input image, the same format as the first one 
    
    :type image2: :class:`CvArr`
    
    
    :param acc: Accumulator with the same number of channels as input images, 32-bit or 64-bit floating-point 
    
    :type acc: :class:`CvArr`
    
    
    :param mask: Optional operation mask 
    
    :type mask: :class:`CvArr`
    
    
    
The function adds the product of 2 images or their selected regions to the accumulator 
``acc``
:



.. math::

    \texttt{acc} (x,y)  \leftarrow \texttt{acc} (x,y) +  \texttt{image1} (x,y)  \cdot \texttt{image2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0  



.. index:: RunningAvg

.. _RunningAvg:

RunningAvg
----------

`id=0.136357383909 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/RunningAvg>`__


.. function:: RunningAvg(image,acc,alpha,mask=NULL)-> None

    Updates the running average.





    
    :param image: Input image, 1- or 3-channel, 8-bit or 32-bit floating point (each channel of multi-channel image is processed independently) 
    
    :type image: :class:`CvArr`
    
    
    :param acc: Accumulator with the same number of channels as input image, 32-bit or 64-bit floating-point 
    
    :type acc: :class:`CvArr`
    
    
    :param alpha: Weight of input image 
    
    :type alpha: float
    
    
    :param mask: Optional operation mask 
    
    :type mask: :class:`CvArr`
    
    
    
The function calculates the weighted sum of the input image
``image``
and the accumulator 
``acc``
so that 
``acc``
becomes a running average of frame sequence:



.. math::

    \texttt{acc} (x,y)  \leftarrow (1- \alpha )  \cdot \texttt{acc} (x,y) +  \alpha \cdot \texttt{image} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0  


where 
:math:`\alpha`
regulates the update speed (how fast the accumulator forgets about previous frames).


.. index:: SquareAcc

.. _SquareAcc:

SquareAcc
---------

`id=0.606012635939 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/SquareAcc>`__


.. function:: SquareAcc(image,sqsum,mask=NULL)-> None

    Adds the square of the source image to the accumulator.





    
    :param image: Input image, 1- or 3-channel, 8-bit or 32-bit floating point (each channel of multi-channel image is processed independently) 
    
    :type image: :class:`CvArr`
    
    
    :param sqsum: Accumulator with the same number of channels as input image, 32-bit or 64-bit floating-point 
    
    :type sqsum: :class:`CvArr`
    
    
    :param mask: Optional operation mask 
    
    :type mask: :class:`CvArr`
    
    
    
The function adds the input image 
``image``
or its selected region, raised to power 2, to the accumulator 
``sqsum``
:



.. math::

    \texttt{sqsum} (x,y)  \leftarrow \texttt{sqsum} (x,y) +  \texttt{image} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0  


