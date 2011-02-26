Motion Analysis and Object Tracking
===================================

.. highlight:: cpp

.. index:: accumulate

cv::accumulate
--------------
.. cfunction:: void accumulate( const Mat\& src, Mat\& dst, const Mat\& mask=Mat() )

    Adds image to the accumulator.

    :param src: The input image, 1- or 3-channel, 8-bit or 32-bit floating point

    :param dst: The accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point

    :param mask: Optional operation mask

The function adds ``src`` , or some of its elements, to ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images; each channel is processed independently.

The functions ``accumulate*`` can be used, for example, to collect statistic of background of a scene, viewed by a still camera, for the further foreground-background segmentation.

See also:
:func:`accumulateSquare`,:func:`accumulateProduct`,:func:`accumulateWeighted`
.. index:: accumulateSquare

cv::accumulateSquare
--------------------
.. cfunction:: void accumulateSquare( const Mat\& src, Mat\& dst,  const Mat\& mask=Mat() )

    Adds the square of the source image to the accumulator.

    :param src: The input image, 1- or 3-channel, 8-bit or 32-bit floating point

    :param dst: The accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point

    :param mask: Optional operation mask

The function adds the input image ``src`` or its selected region, raised to power 2, to the accumulator ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images; each channel is processed independently.

See also:
:func:`accumulateSquare`,:func:`accumulateProduct`,:func:`accumulateWeighted`
.. index:: accumulateProduct

cv::accumulateProduct
---------------------
.. cfunction:: void accumulateProduct( const Mat\& src1, const Mat\& src2,                        Mat\& dst, const Mat\& mask=Mat() )

    Adds the per-element product of two input images to the accumulator.

    :param src1: The first input image, 1- or 3-channel, 8-bit or 32-bit floating point

    :param src2: The second input image of the same type and the same size as  ``src1``
    :param dst: Accumulator with the same number of channels as input images, 32-bit or 64-bit floating-point

    :param mask: Optional operation mask

The function adds the product of 2 images or their selected regions to the accumulator ``dst`` :

.. math::

    \texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

The function supports multi-channel images; each channel is processed independently.

See also:
:func:`accumulate`,:func:`accumulateSquare`,:func:`accumulateWeighted`
.. index:: accumulateWeighted

cv::accumulateWeighted
----------------------
.. cfunction:: void accumulateWeighted( const Mat\& src, Mat\& dst,                         double alpha, const Mat\& mask=Mat() )

    Updates the running average.

    :param src: The input image, 1- or 3-channel, 8-bit or 32-bit floating point

    :param dst: The accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point

    :param alpha: Weight of the input image

    :param mask: Optional operation mask

The function calculates the weighted sum of the input image ``src`` and the accumulator ``dst`` so that ``dst`` becomes a running average of frame sequence:

.. math::

    \texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0

that is, ``alpha`` regulates the update speed (how fast the accumulator "forgets" about earlier images).
The function supports multi-channel images; each channel is processed independently.

See also:
:func:`accumulate`,:func:`accumulateSquare`,:func:`accumulateProduct` 