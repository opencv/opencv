Feature Detection and Description
=================================

.. highlight:: cpp



cuda::FAST_CUDA
---------------
.. ocv:class:: cuda::FAST_CUDA

Class used for corner detection using the FAST algorithm. ::

    class FAST_CUDA
    {
    public:
        enum
        {
            LOCATION_ROW = 0,
            RESPONSE_ROW,
            ROWS_COUNT
        };

        // all features have same size
        static const int FEATURE_SIZE = 7;

        explicit FAST_CUDA(int threshold, bool nonmaxSupression = true,
                          double keypointsRatio = 0.05);

        void operator ()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints);
        void operator ()(const GpuMat& image, const GpuMat& mask,
                         std::vector<KeyPoint>& keypoints);

        void downloadKeypoints(const GpuMat& d_keypoints,
                               std::vector<KeyPoint>& keypoints);

        void convertKeypoints(const Mat& h_keypoints,
                              std::vector<KeyPoint>& keypoints);

        void release();

        bool nonmaxSupression;

        int threshold;

        double keypointsRatio;

        int calcKeyPointsLocation(const GpuMat& image, const GpuMat& mask);

        int getKeyPoints(GpuMat& keypoints);
    };


The class ``FAST_CUDA`` implements FAST corner detection algorithm.

.. seealso:: :ocv:func:`FAST`



cuda::FAST_CUDA::FAST_CUDA
--------------------------
Constructor.

.. ocv:function:: cuda::FAST_CUDA::FAST_CUDA(int threshold, bool nonmaxSupression = true, double keypointsRatio = 0.05)

    :param threshold: Threshold on difference between intensity of the central pixel and pixels on a circle around this pixel.

    :param nonmaxSupression: If it is true, non-maximum suppression is applied to detected corners (keypoints).

    :param keypointsRatio: Inner buffer size for keypoints store is determined as (keypointsRatio * image_width * image_height).



cuda::FAST_CUDA::operator ()
----------------------------
Finds the keypoints using FAST detector.

.. ocv:function:: void cuda::FAST_CUDA::operator ()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints)
.. ocv:function:: void cuda::FAST_CUDA::operator ()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints)

    :param image: Image where keypoints (corners) are detected. Only 8-bit grayscale images are supported.

    :param mask: Optional input mask that marks the regions where we should detect features.

    :param keypoints: The output vector of keypoints. Can be stored both in CPU and GPU memory. For GPU memory:

            * keypoints.ptr<Vec2s>(LOCATION_ROW)[i] will contain location of i'th point
            * keypoints.ptr<float>(RESPONSE_ROW)[i] will contain response of i'th point (if non-maximum suppression is applied)



cuda::FAST_CUDA::downloadKeypoints
----------------------------------
Download keypoints from GPU to CPU memory.

.. ocv:function:: void cuda::FAST_CUDA::downloadKeypoints(const GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints)



cuda::FAST_CUDA::convertKeypoints
---------------------------------
Converts keypoints from CUDA representation to vector of ``KeyPoint``.

.. ocv:function:: void cuda::FAST_CUDA::convertKeypoints(const Mat& h_keypoints, std::vector<KeyPoint>& keypoints)



cuda::FAST_CUDA::release
------------------------
Releases inner buffer memory.

.. ocv:function:: void cuda::FAST_CUDA::release()



cuda::FAST_CUDA::calcKeyPointsLocation
--------------------------------------
Find keypoints and compute it's response if ``nonmaxSupression`` is true.

.. ocv:function:: int cuda::FAST_CUDA::calcKeyPointsLocation(const GpuMat& image, const GpuMat& mask)

    :param image: Image where keypoints (corners) are detected. Only 8-bit grayscale images are supported.

    :param mask: Optional input mask that marks the regions where we should detect features.

The function returns count of detected keypoints.



cuda::FAST_CUDA::getKeyPoints
-----------------------------
Gets final array of keypoints.

.. ocv:function:: int cuda::FAST_CUDA::getKeyPoints(GpuMat& keypoints)

    :param keypoints: The output vector of keypoints.

The function performs non-max suppression if needed and returns final count of keypoints.



cuda::ORB_CUDA
--------------
.. ocv:class:: cuda::ORB_CUDA

Class for extracting ORB features and descriptors from an image. ::

    class ORB_CUDA
    {
    public:
        enum
        {
            X_ROW = 0,
            Y_ROW,
            RESPONSE_ROW,
            ANGLE_ROW,
            OCTAVE_ROW,
            SIZE_ROW,
            ROWS_COUNT
        };

        enum
        {
            DEFAULT_FAST_THRESHOLD = 20
        };

        explicit ORB_CUDA(int nFeatures = 500, float scaleFactor = 1.2f,
                         int nLevels = 8, int edgeThreshold = 31,
                         int firstLevel = 0, int WTA_K = 2,
                         int scoreType = 0, int patchSize = 31);

        void operator()(const GpuMat& image, const GpuMat& mask,
                        std::vector<KeyPoint>& keypoints);
        void operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints);

        void operator()(const GpuMat& image, const GpuMat& mask,
                        std::vector<KeyPoint>& keypoints, GpuMat& descriptors);
        void operator()(const GpuMat& image, const GpuMat& mask,
                        GpuMat& keypoints, GpuMat& descriptors);

        void downloadKeyPoints(GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints);

        void convertKeyPoints(Mat& d_keypoints, std::vector<KeyPoint>& keypoints);

        int descriptorSize() const;

        void setParams(size_t n_features, const ORB::CommonParams& detector_params);
        void setFastParams(int threshold, bool nonmaxSupression = true);

        void release();

        bool blurForDescriptor;
    };

The class implements ORB feature detection and description algorithm.



cuda::ORB_CUDA::ORB_CUDA
------------------------
Constructor.

.. ocv:function:: cuda::ORB_CUDA::ORB_CUDA(int nFeatures = 500, float scaleFactor = 1.2f, int nLevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2, int scoreType = 0, int patchSize = 31)

    :param nFeatures: The number of desired features.

    :param scaleFactor: Coefficient by which we divide the dimensions from one scale pyramid level to the next.

    :param nLevels: The number of levels in the scale pyramid.

    :param edgeThreshold: How far from the boundary the points should be.

    :param firstLevel: The level at which the image is given. If 1, that means we will also look at the image  `scaleFactor`  times bigger.



cuda::ORB_CUDA::operator()
--------------------------
Detects keypoints and computes descriptors for them.

.. ocv:function:: void cuda::ORB_CUDA::operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints)

.. ocv:function:: void cuda::ORB_CUDA::operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints)

.. ocv:function:: void cuda::ORB_CUDA::operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints, GpuMat& descriptors)

.. ocv:function:: void cuda::ORB_CUDA::operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors)

    :param image: Input 8-bit grayscale image.

    :param mask: Optional input mask that marks the regions where we should detect features.

    :param keypoints: The input/output vector of keypoints. Can be stored both in CPU and GPU memory. For GPU memory:

            * ``keypoints.ptr<float>(X_ROW)[i]`` contains x coordinate of the i'th feature.
            * ``keypoints.ptr<float>(Y_ROW)[i]`` contains y coordinate of the i'th feature.
            * ``keypoints.ptr<float>(RESPONSE_ROW)[i]`` contains the response of the i'th feature.
            * ``keypoints.ptr<float>(ANGLE_ROW)[i]`` contains orientation of the i'th feature.
            * ``keypoints.ptr<float>(OCTAVE_ROW)[i]`` contains the octave of the i'th feature.
            * ``keypoints.ptr<float>(SIZE_ROW)[i]`` contains the size of the i'th feature.

    :param descriptors: Computed descriptors. if ``blurForDescriptor`` is true, image will be blurred before descriptors calculation.



cuda::ORB_CUDA::downloadKeyPoints
---------------------------------
Download keypoints from GPU to CPU memory.

.. ocv:function:: static void cuda::ORB_CUDA::downloadKeyPoints( const GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints )



cuda::ORB_CUDA::convertKeyPoints
--------------------------------
Converts keypoints from CUDA representation to vector of ``KeyPoint``.

.. ocv:function:: static void cuda::ORB_CUDA::convertKeyPoints( const Mat& d_keypoints, std::vector<KeyPoint>& keypoints )



cuda::ORB_CUDA::release
-----------------------
Releases inner buffer memory.

.. ocv:function:: void cuda::ORB_CUDA::release()



cuda::BFMatcher_CUDA
--------------------
.. ocv:class:: cuda::BFMatcher_CUDA

Brute-force descriptor matcher. For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches between descriptor sets. ::

    class BFMatcher_CUDA
    {
    public:
        explicit BFMatcher_CUDA(int norm = cv::NORM_L2);

        // Add descriptors to train descriptor collection.
        void add(const std::vector<GpuMat>& descCollection);

        // Get train descriptors collection.
        const std::vector<GpuMat>& getTrainDescriptors() const;

        // Clear train descriptors collection.
        void clear();

        // Return true if there are no train descriptors in collection.
        bool empty() const;

        // Return true if the matcher supports mask in match methods.
        bool isMaskSupported() const;

        void matchSingle(const GpuMat& query, const GpuMat& train,
            GpuMat& trainIdx, GpuMat& distance,
            const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

        static void matchDownload(const GpuMat& trainIdx,
            const GpuMat& distance, std::vector<DMatch>& matches);
        static void matchConvert(const Mat& trainIdx,
            const Mat& distance, std::vector<DMatch>& matches);

        void match(const GpuMat& query, const GpuMat& train,
            std::vector<DMatch>& matches, const GpuMat& mask = GpuMat());

        void makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection,
            const vector<GpuMat>& masks = std::vector<GpuMat>());

        void matchCollection(const GpuMat& query, const GpuMat& trainCollection,
            GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
            const GpuMat& maskCollection, Stream& stream = Stream::Null());

        static void matchDownload(const GpuMat& trainIdx, GpuMat& imgIdx,
            const GpuMat& distance, std::vector<DMatch>& matches);
        static void matchConvert(const Mat& trainIdx, const Mat& imgIdx,
            const Mat& distance, std::vector<DMatch>& matches);

        void match(const GpuMat& query, std::vector<DMatch>& matches,
            const std::vector<GpuMat>& masks = std::vector<GpuMat>());

        void knnMatchSingle(const GpuMat& query, const GpuMat& train,
            GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k,
            const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

        static void knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void knnMatchConvert(const Mat& trainIdx, const Mat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void knnMatch(const GpuMat& query, const GpuMat& train,
            std::vector< std::vector<DMatch> >& matches, int k,
            const GpuMat& mask = GpuMat(), bool compactResult = false);

        void knnMatch2Collection(const GpuMat& query, const GpuMat& trainCollection,
            GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
            const GpuMat& maskCollection = GpuMat(), Stream& stream = Stream::Null());

        static void knnMatch2Download(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void knnMatch(const GpuMat& query, std::vector< std::vector<DMatch> >& matches, int k,
            const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
            bool compactResult = false);

        void radiusMatchSingle(const GpuMat& query, const GpuMat& train,
            GpuMat& trainIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance,
            const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

        static void radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, const GpuMat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void radiusMatch(const GpuMat& query, const GpuMat& train,
            std::vector< std::vector<DMatch> >& matches, float maxDistance,
            const GpuMat& mask = GpuMat(), bool compactResult = false);

        void radiusMatchCollection(const GpuMat& query, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance,
            const std::vector<GpuMat>& masks = std::vector<GpuMat>(), Stream& stream = Stream::Null());

        static void radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, const GpuMat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
        static void radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches,
            std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

        void radiusMatch(const GpuMat& query, std::vector< std::vector<DMatch> >& matches, float maxDistance,
            const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false);

    private:
        std::vector<GpuMat> trainDescCollection;
    };


The class ``BFMatcher_CUDA`` has an interface similar to the class :ocv:class:`DescriptorMatcher`. It has two groups of ``match`` methods: for matching descriptors of one image with another image or with an image set. Also, all functions have an alternative to save results either to the GPU memory or to the CPU memory.

.. seealso:: :ocv:class:`DescriptorMatcher`, :ocv:class:`BFMatcher`



cuda::BFMatcher_CUDA::match
---------------------------
Finds the best match for each descriptor from a query set with train descriptors.

.. ocv:function:: void cuda::BFMatcher_CUDA::match(const GpuMat& query, const GpuMat& train, std::vector<DMatch>& matches, const GpuMat& mask = GpuMat())

.. ocv:function:: void cuda::BFMatcher_CUDA::matchSingle(const GpuMat& query, const GpuMat& train, GpuMat& trainIdx, GpuMat& distance, const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null())

.. ocv:function:: void cuda::BFMatcher_CUDA::match(const GpuMat& query, std::vector<DMatch>& matches, const std::vector<GpuMat>& masks = std::vector<GpuMat>())

.. ocv:function:: void cuda::BFMatcher_CUDA::matchCollection( const GpuMat& query, const GpuMat& trainCollection, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, const GpuMat& masks=GpuMat(), Stream& stream=Stream::Null() )

.. seealso:: :ocv:func:`DescriptorMatcher::match`



cuda::BFMatcher_CUDA::makeGpuCollection
---------------------------------------
Performs a GPU collection of train descriptors and masks in a suitable format for the :ocv:func:`cuda::BFMatcher_CUDA::matchCollection` function.

.. ocv:function:: void cuda::BFMatcher_CUDA::makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection, const vector<GpuMat>& masks = std::vector<GpuMat>())



cuda::BFMatcher_CUDA::matchDownload
-----------------------------------
Downloads matrices obtained via :ocv:func:`cuda::BFMatcher_CUDA::matchSingle` or :ocv:func:`cuda::BFMatcher_CUDA::matchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: static void cuda::BFMatcher_CUDA::matchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector<DMatch>&matches)

.. ocv:function:: static void cuda::BFMatcher_CUDA::matchDownload( const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, std::vector<DMatch>& matches )



cuda::BFMatcher_CUDA::matchConvert
----------------------------------
Converts matrices obtained via :ocv:func:`cuda::BFMatcher_CUDA::matchSingle` or :ocv:func:`cuda::BFMatcher_CUDA::matchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void cuda::BFMatcher_CUDA::matchConvert(const Mat& trainIdx, const Mat& distance, std::vector<DMatch>&matches)

.. ocv:function:: void cuda::BFMatcher_CUDA::matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>&matches)



cuda::BFMatcher_CUDA::knnMatch
------------------------------
Finds the ``k`` best matches for each descriptor from a query set with train descriptors.

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatch(const GpuMat& query, const GpuMat& train, std::vector< std::vector<DMatch> >&matches, int k, const GpuMat& mask = GpuMat(), bool compactResult = false)

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatchSingle(const GpuMat& query, const GpuMat& train, GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k, const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null())

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatch(const GpuMat& query, std::vector< std::vector<DMatch> >&matches, int k, const std::vector<GpuMat>&masks = std::vector<GpuMat>(), bool compactResult = false )

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatch2Collection(const GpuMat& query, const GpuMat& trainCollection, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, const GpuMat& maskCollection = GpuMat(), Stream& stream = Stream::Null())

    :param query: Query set of descriptors.

    :param train: Training set of descriptors. It is not be added to train descriptors collection stored in the class object.

    :param k: Number of the best matches per each query descriptor (or less if it is not possible).

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

    :param compactResult: If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.

    :param stream: Stream for the asynchronous version.

The function returns detected ``k`` (or less if not possible) matches in the increasing order by distance.

The third variant of the method stores the results in GPU memory.

.. seealso:: :ocv:func:`DescriptorMatcher::knnMatch`



cuda::BFMatcher_CUDA::knnMatchDownload
--------------------------------------
Downloads matrices obtained via :ocv:func:`cuda::BFMatcher_CUDA::knnMatchSingle` or :ocv:func:`cuda::BFMatcher_CUDA::knnMatch2Collection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatch2Download(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.



cuda::BFMatcher_CUDA::knnMatchConvert
-------------------------------------
Converts matrices obtained via :ocv:func:`cuda::BFMatcher_CUDA::knnMatchSingle` or :ocv:func:`cuda::BFMatcher_CUDA::knnMatch2Collection` to CPU vector with :ocv:class:`DMatch`.

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatchConvert(const Mat& trainIdx, const Mat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void cuda::BFMatcher_CUDA::knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.



cuda::BFMatcher_CUDA::radiusMatch
---------------------------------
For each query descriptor, finds the best matches with a distance less than a given threshold.

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatch(const GpuMat& query, const GpuMat& train, std::vector< std::vector<DMatch> >&matches, float maxDistance, const GpuMat& mask = GpuMat(), bool compactResult = false)

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatchSingle(const GpuMat& query, const GpuMat& train, GpuMat& trainIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance, const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null())

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatch(const GpuMat& query, std::vector< std::vector<DMatch> >&matches, float maxDistance, const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false)

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatchCollection(const GpuMat& query, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance, const std::vector<GpuMat>& masks = std::vector<GpuMat>(), Stream& stream = Stream::Null())

    :param query: Query set of descriptors.

    :param train: Training set of descriptors. It is not added to train descriptors collection stored in the class object.

    :param maxDistance: Distance threshold.

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

    :param compactResult: If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.

    :param stream: Stream for the asynchronous version.

The function returns detected matches in the increasing order by distance.

The methods work only on devices with the compute capability  :math:`>=` 1.1.

The third variant of the method stores the results in GPU memory and does not store the points by the distance.

.. seealso:: :ocv:func:`DescriptorMatcher::radiusMatch`



cuda::BFMatcher_CUDA::radiusMatchDownload
-----------------------------------------
Downloads matrices obtained via :ocv:func:`cuda::BFMatcher_CUDA::radiusMatchSingle` or :ocv:func:`cuda::BFMatcher_CUDA::radiusMatchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, const GpuMat& nMatches, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, const GpuMat& nMatches, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.




cuda::BFMatcher_CUDA::radiusMatchConvert
----------------------------------------
Converts matrices obtained via :ocv:func:`cuda::BFMatcher_CUDA::radiusMatchSingle` or :ocv:func:`cuda::BFMatcher_CUDA::radiusMatchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void cuda::BFMatcher_CUDA::radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.
