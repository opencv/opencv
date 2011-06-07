Motion Analysis and Object Tracking
===================================

.. highlight:: cpp

.. index:: accumulate

accumulate
--------------
.. cpp:function:: void accumulate( InputArray src, InputOutputArray dst, InputArray mask=None() )

    Adds an image to the accumulator.

    :param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.

    :param dst: Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.

    :param mask: Optional operation mask.

The function adds ``src``  or some of its elements to ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images. Each channel is processed independently.

The functions ``accumulate*`` can be used, for example, to collect statistics of a scene background viewed by a still camera and for the further foreground-background segmentation.

See Also:
:cpp:func:`accumulateSquare`,
:cpp:func:`accumulateProduct`,
:cpp:func:`accumulateWeighted`

.. index:: accumulateSquare

accumulateSquare
--------------------
.. cpp:function:: void accumulateSquare( InputArray src, InputOutputArray dst,  InputArray mask=None() )

    Adds the square of a source image to the accumulator.

    :param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.

    :param dst: Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.

    :param mask: Optional operation mask.

The function adds the input image ``src`` or its selected region, raised to power 2, to the accumulator ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images Each channel is processed independently.

See Also:
:cpp:func:`accumulateSquare`,
:cpp:func:`accumulateProduct`,
:cpp:func:`accumulateWeighted`

.. index:: accumulateProduct

accumulateProduct
---------------------
.. cpp:function:: void accumulateProduct( InputArray src1, InputArray src2, InputOutputArray dst, InputArray mask=None() )

    Adds the per-element product of two input images to the accumulator.

    :param src1: The first input image, 1- or 3-channel, 8-bit or 32-bit floating point.

    :param src2: The second input image of the same type and the same size as  ``src1`` .
	
    :param dst: Accumulator with the same number of channels as input images, 32-bit or 64-bit floating-point.

    :param mask: Optional operation mask.

The function adds the product of 2 images or their selected regions to the accumulator ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images. Each channel is processed independently.

See Also:
:cpp:func:`accumulate`,
:cpp:func:`accumulateSquare`,
:cpp:func:`accumulateWeighted`

.. index:: accumulateWeighted

accumulateWeighted
----------------------
.. cpp:function:: void accumulateWeighted( InputArray src, InputOutputArray dst, double alpha, InputArray mask=None() )

    Updates a running average.

    :param src: Input image as 1- or 3-channel, 8-bit or 32-bit floating point.

    :param dst: Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.

    :param alpha: Weight of the input image.

    :param mask: Optional operation mask.

The function calculates the weighted sum of the input image ``src`` and the accumulator ``dst`` so that ``dst`` becomes a running average of a frame sequence:

.. math::

    \texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

That is, ``alpha`` regulates the update speed (how fast the accumulator "forgets" about earlier images).
The function supports multi-channel images. Each channel is processed independently.

See Also:
:cpp:func:`accumulate`,
:cpp:func:`accumulateSquare`,
:cpp:func:`accumulateProduct` 