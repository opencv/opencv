Feature Detection
=================

.. highlight:: cpp



gpu::cornerHarris
---------------------
Computes the Harris cornerness criteria at each image pixel.

.. ocv:function:: void gpu::cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType=BORDER_REFLECT101)

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. It has the same size as ``src`` and ``CV_32FC1`` type.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101`` and  ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerHarris`



gpu::cornerMinEigenVal
--------------------------
Computes the minimum eigen value of a 2x2 derivative covariation matrix at each pixel (the cornerness criteria).

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType=BORDER_REFLECT101)

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int borderType=BORDER_REFLECT101)

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, int borderType=BORDER_REFLECT101, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. The size is the same. The type is  ``CV_32FC1`` .

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param borderType: Pixel extrapolation method. Only ``BORDER_REFLECT101`` and ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerMinEigenVal`



gpu::GoodFeaturesToTrackDetector_GPU
------------------------------------
.. ocv:class:: gpu::GoodFeaturesToTrackDetector_GPU

Class used for strong corners detection on an image. ::

    class GoodFeaturesToTrackDetector_GPU
    {
    public:
        explicit GoodFeaturesToTrackDetector_GPU(int maxCorners_ = 1000, double qualityLevel_ = 0.01, double minDistance_ = 0.0,
            int blockSize_ = 3, bool useHarrisDetector_ = false, double harrisK_ = 0.04);

        void operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask = GpuMat());

        int maxCorners;
        double qualityLevel;
        double minDistance;

        int blockSize;
        bool useHarrisDetector;
        double harrisK;

        void releaseMemory();
    };

The class finds the most prominent corners in the image.

.. seealso:: :ocv:func:`goodFeaturesToTrack`
