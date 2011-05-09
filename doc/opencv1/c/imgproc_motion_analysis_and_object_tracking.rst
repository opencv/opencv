Motion Analysis and Object Tracking
===================================

.. highlight:: c



.. index:: Acc

.. _Acc:

Acc
---

`id=0.999960514281 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/Acc>`__




.. cfunction:: void cvAcc(  const CvArr* image, CvArr* sum, const CvArr* mask=NULL )

    Adds a frame to an accumulator.





    
    :param image: Input image, 1- or 3-channel, 8-bit or 32-bit floating point. (each channel of multi-channel image is processed independently) 
    
    
    :param sum: Accumulator with the same number of channels as input image, 32-bit or 64-bit floating-point 
    
    
    :param mask: Optional operation mask 
    
    
    
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

`id=0.550586168837 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/MultiplyAcc>`__




.. cfunction:: void cvMultiplyAcc(  const CvArr* image1, const CvArr* image2, CvArr* acc, const CvArr* mask=NULL )

    Adds the product of two input images to the accumulator.





    
    :param image1: First input image, 1- or 3-channel, 8-bit or 32-bit floating point (each channel of multi-channel image is processed independently) 
    
    
    :param image2: Second input image, the same format as the first one 
    
    
    :param acc: Accumulator with the same number of channels as input images, 32-bit or 64-bit floating-point 
    
    
    :param mask: Optional operation mask 
    
    
    
The function adds the product of 2 images or their selected regions to the accumulator 
``acc``
:



.. math::

    \texttt{acc} (x,y)  \leftarrow \texttt{acc} (x,y) +  \texttt{image1} (x,y)  \cdot \texttt{image2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0  



.. index:: RunningAvg

.. _RunningAvg:

RunningAvg
----------

`id=0.0736920452652 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/RunningAvg>`__




.. cfunction:: void cvRunningAvg(  const CvArr* image, CvArr* acc, double alpha, const CvArr* mask=NULL )

    Updates the running average.





    
    :param image: Input image, 1- or 3-channel, 8-bit or 32-bit floating point (each channel of multi-channel image is processed independently) 
    
    
    :param acc: Accumulator with the same number of channels as input image, 32-bit or 64-bit floating-point 
    
    
    :param alpha: Weight of input image 
    
    
    :param mask: Optional operation mask 
    
    
    
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

`id=0.22065009551 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/SquareAcc>`__




.. cfunction:: void cvSquareAcc(  const CvArr* image, CvArr* sqsum, const CvArr* mask=NULL )

    Adds the square of the source image to the accumulator.





    
    :param image: Input image, 1- or 3-channel, 8-bit or 32-bit floating point (each channel of multi-channel image is processed independently) 
    
    
    :param sqsum: Accumulator with the same number of channels as input image, 32-bit or 64-bit floating-point 
    
    
    :param mask: Optional operation mask 
    
    
    
The function adds the input image 
``image``
or its selected region, raised to power 2, to the accumulator 
``sqsum``
:



.. math::

    \texttt{sqsum} (x,y)  \leftarrow \texttt{sqsum} (x,y) +  \texttt{image} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0  


