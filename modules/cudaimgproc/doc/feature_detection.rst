Feature Detection
=================

.. highlight:: cpp



cuda::CornernessCriteria
------------------------
.. ocv:class:: cuda::CornernessCriteria : public Algorithm

Base class for Cornerness Criteria computation. ::

    class CV_EXPORTS CornernessCriteria : public Algorithm
    {
    public:
        virtual void compute(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
    };



cuda::CornernessCriteria::compute
---------------------------------
Computes the cornerness criteria at each image pixel.

.. ocv:function:: void cuda::CornernessCriteria::compute(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image containing cornerness values. It will have the same size as ``src`` and ``CV_32FC1`` type.

    :param stream: Stream for the asynchronous version.



cuda::createHarrisCorner
------------------------
Creates implementation for Harris cornerness criteria.

.. ocv:function:: Ptr<CornernessCriteria> cuda::createHarrisCorner(int srcType, int blockSize, int ksize, double k, int borderType = BORDER_REFLECT101)

    :param srcType: Input source type. Only  ``CV_8UC1`` and  ``CV_32FC1`` are supported for now.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101`` and  ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerHarris`



cuda::createMinEigenValCorner
-----------------------------
Creates implementation for the minimum eigen value of a 2x2 derivative covariation matrix (the cornerness criteria).

.. ocv:function:: Ptr<CornernessCriteria> cuda::createMinEigenValCorner(int srcType, int blockSize, int ksize, int borderType = BORDER_REFLECT101)

    :param srcType: Input source type. Only  ``CV_8UC1`` and  ``CV_32FC1`` are supported for now.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101`` and  ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerMinEigenVal`



cuda::CornersDetector
---------------------
.. ocv:class:: cuda::CornersDetector : public Algorithm

Base class for Corners Detector. ::

    class CV_EXPORTS CornersDetector : public Algorithm
    {
    public:
        virtual void detect(InputArray image, OutputArray corners, InputArray mask = noArray()) = 0;
    };



cuda::CornersDetector::detect
-----------------------------
Determines strong corners on an image.

.. ocv:function:: void cuda::CornersDetector::detect(InputArray image, OutputArray corners, InputArray mask = noArray())

    :param image: Input 8-bit or floating-point 32-bit, single-channel image.

    :param corners: Output vector of detected corners (1-row matrix with CV_32FC2 type with corners positions).

    :param mask: Optional region of interest. If the image is not empty (it needs to have the type  ``CV_8UC1``  and the same size as  ``image`` ), it  specifies the region in which the corners are detected.



cuda::createGoodFeaturesToTrackDetector
---------------------------------------
Creates implementation for :ocv:class:`cuda::CornersDetector` .

.. ocv:function:: Ptr<CornersDetector> cuda::createGoodFeaturesToTrackDetector(int srcType, int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0, int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04)

    :param srcType: Input source type. Only  ``CV_8UC1`` and  ``CV_32FC1`` are supported for now.

    :param maxCorners: Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.

    :param qualityLevel: Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see  :ocv:func:`cornerMinEigenVal` ) or the Harris function response (see  :ocv:func:`cornerHarris` ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the  ``qualityLevel=0.01`` , then all the corners with the quality measure less than 15 are rejected.

    :param minDistance: Minimum possible Euclidean distance between the returned corners.

    :param blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See  :ocv:func:`cornerEigenValsAndVecs` .

    :param useHarrisDetector: Parameter indicating whether to use a Harris detector (see :ocv:func:`cornerHarris`) or :ocv:func:`cornerMinEigenVal`.

    :param harrisK: Free parameter of the Harris detector.

.. seealso:: :ocv:func:`goodFeaturesToTrack`
