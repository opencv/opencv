Histogram Calculation
=====================

.. highlight:: cpp



gpu::evenLevels
-------------------
Computes levels with even distribution.

.. ocv:function:: void gpu::evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel)

    :param levels: Destination array.  ``levels`` has 1 row, ``nLevels`` columns, and the ``CV_32SC1`` type.

    :param nLevels: Number of computed levels.  ``nLevels`` must be at least 2.

    :param lowerLevel: Lower boundary value of the lowest level.

    :param upperLevel: Upper boundary value of the greatest level.



gpu::histEven
-----------------
Calculates a histogram with evenly distributed bins.

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat& hist, GpuMat& buf, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histEven( const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::histEven( const GpuMat& src, GpuMat hist[4], GpuMat& buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream=Stream::Null() )

    :param src: Source image. ``CV_8U``, ``CV_16U``, or ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``histSize`` columns, and the ``CV_32S`` type.

    :param histSize: Size of the histogram.

    :param lowerLevel: Lower boundary of lowest-level bin.

    :param upperLevel: Upper boundary of highest-level bin.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



gpu::histRange
------------------
Calculates a histogram with bins determined by the ``levels`` array.

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U`` , ``CV_16U`` , or  ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``(levels.cols-1)`` columns, and the  ``CV_32SC1`` type.

    :param levels: Number of levels in the histogram.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



gpu::calcHist
------------------
Calculates histogram for one channel 8-bit image.

.. ocv:function:: void gpu::calcHist(const GpuMat& src, GpuMat& hist, Stream& stream = Stream::Null())

    :param src: Source image.

    :param hist: Destination histogram with one row, 256 columns, and the  ``CV_32SC1`` type.

    :param stream: Stream for the asynchronous version.



gpu::equalizeHist
------------------
Equalizes the histogram of a grayscale image.

.. ocv:function:: void gpu::equalizeHist(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image.

    :param hist: Destination histogram with one row, 256 columns, and the  ``CV_32SC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`equalizeHist`
