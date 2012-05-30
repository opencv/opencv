Motion Analysis and Object Tracking
===================================

.. highlight:: cpp

accumulate
--------------
Adds an image to the accumulator.

.. ocv:function:: void accumulate( InputArray src, InputOutputArray dst, InputArray mask=noArray() )

.. ocv:pyfunction:: cv2.accumulate(src, dst[, mask]) -> None

.. ocv:cfunction:: void cvAcc( const CvArr* image, CvArr* sum, const CvArr* mask=NULL )

.. ocv:pyoldfunction:: cv.Acc(image, sum, mask=None) -> None

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

.. ocv:pyfunction:: cv2.accumulateSquare(src, dst[, mask]) -> None

.. ocv:cfunction:: void cvSquareAcc( const CvArr* image, CvArr* sqsum, const CvArr* mask=NULL )

.. ocv:pyoldfunction:: cv.SquareAcc(image, sqsum, mask=None) -> None

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

.. ocv:pyfunction:: cv2.accumulateProduct(src1, src2, dst[, mask]) -> None

.. ocv:cfunction:: void cvMultiplyAcc( const CvArr* image1, const CvArr* image2, CvArr* acc, const CvArr* mask=NULL )

.. ocv:pyoldfunction:: cv.MultiplyAcc(image1, image2, acc, mask=None)-> None

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

.. ocv:pyfunction:: cv2.accumulateWeighted(src, dst, alpha[, mask]) -> None

.. ocv:cfunction:: void cvRunningAvg( const CvArr* image, CvArr* acc, double alpha, const CvArr* mask=NULL )
.. ocv:pyoldfunction:: cv.RunningAvg(image, acc, alpha, mask=None)-> None

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



phaseCorrelate
--------------
The function is used to detect translational shifts that occur between two images. The operation takes advantage of the Fourier shift theorem for detecting the translational shift in the frequency domain. It can be used for fast image registration as well as motion estimation. For more information please see http://en.wikipedia.org/wiki/Phase\_correlation .

Calculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed with :ocv:func:`getOptimalDFTSize`.

.. ocv:function:: Point2d phaseCorrelate(InputArray src1, InputArray src2, InputArray window = noArray())

    :param src1: Source floating point array (CV_32FC1 or CV_64FC1)
    :param src2: Source floating point array (CV_32FC1 or CV_64FC1)
    :param window: Floating point array with windowing coefficients to reduce edge effects (optional).

Return value: detected phase shift (sub-pixel) between the two arrays.

The function performs the following equations

*
    First it applies a Hanning window (see http://en.wikipedia.org/wiki/Hann\_function) to each image to remove possible edge effects. This window is cached until the array size changes to speed up processing time.

*
    Next it computes the forward DFTs of each source array:
    .. math::

        \mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}

    where
    :math:`\mathcal{F}` is the forward DFT.

*
    It then computes the cross-power spectrum of each frequency domain array:
    .. math::

        R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}

*
    Next the cross-correlation is converted back into the time domain via the inverse DFT:
    .. math::

        r = \mathcal{F}^{-1}\{R\}
*
    Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to achieve sub-pixel accuracy.
    .. math::

       (\Delta x, \Delta y) = \texttt{weighted_centroid}\{\arg \max_{(x, y)}\{r\}\}

.. seealso::
    :ocv:func:`dft`,
    :ocv:func:`getOptimalDFTSize`,
    :ocv:func:`idft`,
    :ocv:func:`mulSpectrums`
    :ocv:func:`createHanningWindow`

createHanningWindow
-------------------------------
This function computes a Hanning window coefficients in two dimensions. See http://en.wikipedia.org/wiki/Hann\_function and http://en.wikipedia.org/wiki/Window\_function for more information.

.. ocv:function:: void createHanningWindow(OutputArray dst, Size winSize, int type)

    :param dst: Destination array to place Hann coefficients in
    :param winSize: The window size specifications
    :param type: Created array type

An example is shown below: ::

    // create hanning window of size 100x100 and type CV_32F
    Mat hann;
    createHanningWindow(hann, Size(100, 100), CV_32F);

.. seealso::
    :ocv:func:`phaseCorrelate`
