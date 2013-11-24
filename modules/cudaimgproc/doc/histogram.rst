Histogram Calculation
=====================

.. highlight:: cpp



cuda::calcHist
--------------
Calculates histogram for one channel 8-bit image.

.. ocv:function:: void cuda::calcHist(InputArray src, OutputArray hist, Stream& stream = Stream::Null())

    :param src: Source image with ``CV_8UC1`` type.

    :param hist: Destination histogram with one row, 256 columns, and the  ``CV_32SC1`` type.

    :param stream: Stream for the asynchronous version.



cuda::equalizeHist
------------------
Equalizes the histogram of a grayscale image.

.. ocv:function:: void cuda::equalizeHist(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::equalizeHist(InputArray src, OutputArray dst, InputOutputArray buf, Stream& stream = Stream::Null())

    :param src: Source image with ``CV_8UC1`` type.

    :param dst: Destination image.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`equalizeHist`



cuda::CLAHE
-----------
.. ocv:class:: cuda::CLAHE : public cv::CLAHE

Base class for Contrast Limited Adaptive Histogram Equalization. ::

    class CV_EXPORTS CLAHE : public cv::CLAHE
    {
    public:
        using cv::CLAHE::apply;
        virtual void apply(InputArray src, OutputArray dst, Stream& stream) = 0;
    };



cuda::CLAHE::apply
------------------
Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.

.. ocv:function:: void cuda::CLAHE::apply(InputArray src, OutputArray dst)

.. ocv:function:: void cuda::CLAHE::apply(InputArray src, OutputArray dst, Stream& stream)

    :param src: Source image with ``CV_8UC1`` type.

    :param dst: Destination image.

    :param stream: Stream for the asynchronous version.



cuda::createCLAHE
-----------------
Creates implementation for :ocv:class:`cuda::CLAHE` .

.. ocv:function:: Ptr<cuda::CLAHE> createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8))

    :param clipLimit: Threshold for contrast limiting.

    :param tileGridSize: Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles. ``tileGridSize`` defines the number of tiles in row and column.




cuda::evenLevels
----------------
Computes levels with even distribution.

.. ocv:function:: void cuda::evenLevels(OutputArray levels, int nLevels, int lowerLevel, int upperLevel)

    :param levels: Destination array.  ``levels`` has 1 row, ``nLevels`` columns, and the ``CV_32SC1`` type.

    :param nLevels: Number of computed levels.  ``nLevels`` must be at least 2.

    :param lowerLevel: Lower boundary value of the lowest level.

    :param upperLevel: Upper boundary value of the greatest level.



cuda::histEven
--------------
Calculates a histogram with evenly distributed bins.

.. ocv:function:: void cuda::histEven(InputArray src, OutputArray hist, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::histEven(InputArray src, OutputArray hist, InputOutputArray buf, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::histEven(InputArray src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null())

.. ocv:function:: void cuda::histEven(InputArray src, GpuMat hist[4], InputOutputArray buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U``, ``CV_16U``, or ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``histSize`` columns, and the ``CV_32S`` type.

    :param histSize: Size of the histogram.

    :param lowerLevel: Lower boundary of lowest-level bin.

    :param upperLevel: Upper boundary of highest-level bin.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



cuda::histRange
---------------
Calculates a histogram with bins determined by the ``levels`` array.

.. ocv:function:: void cuda::histRange(InputArray src, OutputArray hist, InputArray levels, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::histRange(InputArray src, OutputArray hist, InputArray levels, InputOutputArray buf, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::histRange(InputArray src, GpuMat hist[4], const GpuMat levels[4], Stream& stream = Stream::Null())

.. ocv:function:: void cuda::histRange(InputArray src, GpuMat hist[4], const GpuMat levels[4], InputOutputArray buf, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U`` , ``CV_16U`` , or  ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``(levels.cols-1)`` columns, and the  ``CV_32SC1`` type.

    :param levels: Number of levels in the histogram.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.
