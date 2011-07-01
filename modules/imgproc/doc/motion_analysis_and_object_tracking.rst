Motion Analysis and Object Tracking
===================================

.. highlight:: cpp

accumulate
--------------
Adds an image to the accumulator.

.. ocv:function:: void accumulate( InputArray src, InputOutputArray dst, InputArray mask=noArray() )

.. ocv:pyfunction:: cv2.accumulate(src, dst[, mask]) -> dst

.. ocv:cfunction:: void cvAcc( const CvArr* src, CvArr* dst, const CvArr* mask=NULL )
.. ocv:pyoldfunction:: cv.Acc(src, dst, mask=None)-> None

    :param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.

    :param dst: Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.

    :param mask: Optional operation mask.

The function adds ``src``  or some of its elements to ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images. Each channel is processed independently.

The functions ``accumulate*`` can be used, for example, to collect statistics of a scene background viewed by a still camera and for the further foreground-background segmentation.

.. seealso::

    :ocv:func:`accumulateSquare`,
    :ocv:func:`accumulateProduct`,
    :ocv:func:`accumulateWeighted`



accumulateSquare
--------------------
Adds the square of a source image to the accumulator.

.. ocv:function:: void accumulateSquare( InputArray src, InputOutputArray dst,  InputArray mask=noArray() )

.. ocv:pyfunction:: cv2.accumulateSquare(src, dst[, mask]) -> dst

.. ocv:cfunction:: void cvSquareAcc( const CvArr* src, CvArr* dst, const CvArr* mask=NULL )
.. ocv:pyoldfunction:: cv.SquareAcc(src, dst, mask=None)-> None

    :param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.

    :param dst: Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.

    :param mask: Optional operation mask.

The function adds the input image ``src`` or its selected region, raised to a power of 2, to the accumulator ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images. Each channel is processed independently.

.. seealso::

    :ocv:func:`accumulateSquare`,
    :ocv:func:`accumulateProduct`,
    :ocv:func:`accumulateWeighted`



accumulateProduct
---------------------
Adds the per-element product of two input images to the accumulator.

.. ocv:function:: void accumulateProduct( InputArray src1, InputArray src2, InputOutputArray dst, InputArray mask=noArray() )

.. ocv:pyfunction:: cv2.accumulateProduct(src1, src2, dst[, mask]) -> dst

.. ocv:cfunction:: void cvMultiplyAcc( const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL )
.. ocv:pyoldfunction:: cv.MultiplyAcc(src1, src2, dst, mask=None)-> None

    :param src1: First input image, 1- or 3-channel, 8-bit or 32-bit floating point.

    :param src2: Second input image of the same type and the same size as  ``src1`` .
	
    :param dst: Accumulator with the same number of channels as input images, 32-bit or 64-bit floating-point.

    :param mask: Optional operation mask.

The function adds the product of two images or their selected regions to the accumulator ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images. Each channel is processed independently.

.. seealso::

    :ocv:func:`accumulate`,
    :ocv:func:`accumulateSquare`,
    :ocv:func:`accumulateWeighted`



accumulateWeighted
----------------------
Updates a running average.

.. ocv:function:: void accumulateWeighted( InputArray src, InputOutputArray dst, double alpha, InputArray mask=noArray() )

.. ocv:pyfunction:: cv2.accumulateWeighted(src, dst, alpha[, mask]) -> dst

.. ocv:cfunction:: void cvRunningAvg( const CvArr* src, CvArr* dst, double alpha, const CvArr* mask=NULL )
.. ocv:pyoldfunction:: cv.RunningAvg(src, dst, alpha, mask=None)-> None

    :param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.

    :param dst: Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.

    :param alpha: Weight of the input image.

    :param mask: Optional operation mask.

The function calculates the weighted sum of the input image ``src`` and the accumulator ``dst`` so that ``dst`` becomes a running average of a frame sequence:

.. math::

    \texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

That is, ``alpha`` regulates the update speed (how fast the accumulator "forgets" about earlier images).
The function supports multi-channel images. Each channel is processed independently.

.. seealso::

    :ocv:func:`accumulate`,
    :ocv:func:`accumulateSquare`,
    :ocv:func:`accumulateProduct` 
