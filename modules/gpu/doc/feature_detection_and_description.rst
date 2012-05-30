Feature Detection and Description
=================================

.. highlight:: cpp



gpu::SURF_GPU
-------------
.. ocv:class:: gpu::SURF_GPU

Class used for extracting Speeded Up Robust Features (SURF) from an image. ::

    class SURF_GPU
    {
    public:
        enum KeypointLayout
        {
            X_ROW = 0,
            Y_ROW,
            LAPLACIAN_ROW,
            OCTAVE_ROW,
            SIZE_ROW,
            ANGLE_ROW,
            HESSIAN_ROW,
            ROWS_COUNT
        };

        //! the default constructor
        SURF_GPU();
        //! the full constructor taking all the necessary parameters
        explicit SURF_GPU(double _hessianThreshold, int _nOctaves=4,
             int _nOctaveLayers=2, bool _extended=false, float _keypointsRatio=0.01f);

        //! returns the descriptor size in float's (64 or 128)
        int descriptorSize() const;

        //! upload host keypoints to device memory
        void uploadKeypoints(const vector<KeyPoint>& keypoints,
            GpuMat& keypointsGPU);
        //! download keypoints from device to host memory
        void downloadKeypoints(const GpuMat& keypointsGPU,
            vector<KeyPoint>& keypoints);

        //! download descriptors from device to host memory
        void downloadDescriptors(const GpuMat& descriptorsGPU,
            vector<float>& descriptors);

        void operator()(const GpuMat& img, const GpuMat& mask,
            GpuMat& keypoints);

        void operator()(const GpuMat& img, const GpuMat& mask,
            GpuMat& keypoints, GpuMat& descriptors,
            bool useProvidedKeypoints = false,
            bool calcOrientation = true);

        void operator()(const GpuMat& img, const GpuMat& mask,
            std::vector<KeyPoint>& keypoints);

        void operator()(const GpuMat& img, const GpuMat& mask,
            std::vector<KeyPoint>& keypoints, GpuMat& descriptors,
            bool useProvidedKeypoints = false,
            bool calcOrientation = true);

        void operator()(const GpuMat& img, const GpuMat& mask,
            std::vector<KeyPoint>& keypoints,
            std::vector<float>& descriptors,
            bool useProvidedKeypoints = false,
            bool calcOrientation = true);

        void releaseMemory();

        // SURF parameters
        double hessianThreshold;
        int nOctaves;
        int nOctaveLayers;
        bool extended;
        bool upright;

        //! max keypoints = keypointsRatio * img.size().area()
        float keypointsRatio;

        GpuMat sum, mask1, maskSum, intBuffer;

        GpuMat det, trace;

        GpuMat maxPosBuffer;
    };


The class ``SURF_GPU`` implements Speeded Up Robust Features descriptor. There is a fast multi-scale Hessian keypoint detector that can be used to find the keypoints (which is the default option). But the descriptors can also be computed for the user-specified keypoints. Only 8-bit grayscale images are supported.

The class ``SURF_GPU`` can store results in the GPU and CPU memory. It provides functions to convert results between CPU and GPU version ( ``uploadKeypoints``, ``downloadKeypoints``, ``downloadDescriptors`` ). The format of CPU results is the same as ``SURF`` results. GPU results are stored in ``GpuMat``. The ``keypoints`` matrix is :math:`\texttt{nFeatures} \times 7` matrix with the ``CV_32FC1`` type.

* ``keypoints.ptr<float>(X_ROW)[i]`` contains x coordinate of the i-th feature.
* ``keypoints.ptr<float>(Y_ROW)[i]`` contains y coordinate of the i-th feature.
* ``keypoints.ptr<float>(LAPLACIAN_ROW)[i]``  contains the laplacian sign of the i-th feature.
* ``keypoints.ptr<float>(OCTAVE_ROW)[i]`` contains the octave of the i-th feature.
* ``keypoints.ptr<float>(SIZE_ROW)[i]`` contains the size of the i-th feature.
* ``keypoints.ptr<float>(ANGLE_ROW)[i]`` contain orientation of the i-th feature.
* ``keypoints.ptr<float>(HESSIAN_ROW)[i]`` contains the response of the i-th feature.

The ``descriptors`` matrix is :math:`\texttt{nFeatures} \times \texttt{descriptorSize}` matrix with the ``CV_32FC1`` type.

The class ``SURF_GPU`` uses some buffers and provides access to it. All buffers can be safely released between function calls.

.. seealso:: :ocv:class:`SURF`



gpu::FAST_GPU
-------------
.. ocv:class:: gpu::FAST_GPU

Class used for corner detection using the FAST algorithm. ::

    class FAST_GPU
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

        explicit FAST_GPU(int threshold, bool nonmaxSupression = true,
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


The class ``FAST_GPU`` implements FAST corner detection algorithm.

.. seealso:: :ocv:func:`FAST`



gpu::FAST_GPU::FAST_GPU
-------------------------------------
Constructor.

.. ocv:function:: gpu::FAST_GPU::FAST_GPU(int threshold, bool nonmaxSupression = true, double keypointsRatio = 0.05)

    :param threshold: Threshold on difference between intensity of the central pixel and pixels on a circle around this pixel.

    :param nonmaxSupression: If it is true, non-maximum suppression is applied to detected corners (keypoints).

    :param keypointsRatio: Inner buffer size for keypoints store is determined as (keypointsRatio * image_width * image_height).



gpu::FAST_GPU::operator ()
-------------------------------------
Finds the keypoints using FAST detector.

.. ocv:function:: void gpu::FAST_GPU::operator ()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints)
.. ocv:function:: void gpu::FAST_GPU::operator ()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints)

    :param image: Image where keypoints (corners) are detected. Only 8-bit grayscale images are supported.

    :param mask: Optional input mask that marks the regions where we should detect features.

    :param keypoints: The output vector of keypoints. Can be stored both in CPU and GPU memory. For GPU memory:

            * keypoints.ptr<Vec2s>(LOCATION_ROW)[i] will contain location of i'th point
            * keypoints.ptr<float>(RESPONSE_ROW)[i] will contain response of i'th point (if non-maximum suppression is applied)



gpu::FAST_GPU::downloadKeypoints
-------------------------------------
Download keypoints from GPU to CPU memory.

.. ocv:function:: void gpu::FAST_GPU::downloadKeypoints(const GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints)



gpu::FAST_GPU::convertKeypoints
-------------------------------------
Converts keypoints from GPU representation to vector of ``KeyPoint``.

.. ocv:function:: void gpu::FAST_GPU::convertKeypoints(const Mat& h_keypoints, std::vector<KeyPoint>& keypoints)



gpu::FAST_GPU::release
-------------------------------------
Releases inner buffer memory.

.. ocv:function:: void gpu::FAST_GPU::release()



gpu::FAST_GPU::calcKeyPointsLocation
-------------------------------------
Find keypoints and compute it's response if ``nonmaxSupression`` is true.

.. ocv:function:: int gpu::FAST_GPU::calcKeyPointsLocation(const GpuMat& image, const GpuMat& mask)

    :param image: Image where keypoints (corners) are detected. Only 8-bit grayscale images are supported.

    :param mask: Optional input mask that marks the regions where we should detect features.

The function returns count of detected keypoints.



gpu::FAST_GPU::getKeyPoints
-------------------------------------
Gets final array of keypoints.

.. ocv:function:: int gpu::FAST_GPU::getKeyPoints(GpuMat& keypoints)

    :param keypoints: The output vector of keypoints.

The function performs non-max suppression if needed and returns final count of keypoints.



gpu::ORB_GPU
-------------
.. ocv:class:: gpu::ORB_GPU

Class for extracting ORB features and descriptors from an image. ::

    class ORB_GPU
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

        explicit ORB_GPU(int nFeatures = 500, float scaleFactor = 1.2f,
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



gpu::ORB_GPU::ORB_GPU
-------------------------------------
Constructor.

.. ocv:function:: gpu::ORB_GPU::ORB_GPU(int nFeatures = 500, float scaleFactor = 1.2f, int nLevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2, int scoreType = 0, int patchSize = 31)

    :param nFeatures: The number of desired features.

    :param scaleFactor: Coefficient by which we divide the dimensions from one scale pyramid level to the next.

    :param nLevels: The number of levels in the scale pyramid.

    :param edgeThreshold: How far from the boundary the points should be.

    :param firstLevel: The level at which the image is given. If 1, that means we will also look at the image  `scaleFactor`  times bigger.



gpu::ORB_GPU::operator()
-------------------------------------
Detects keypoints and computes descriptors for them.

.. ocv:function:: void gpu::ORB_GPU::operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints)

.. ocv:function:: void gpu::ORB_GPU::operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints)

.. ocv:function:: void gpu::ORB_GPU::operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints, GpuMat& descriptors)

.. ocv:function:: void gpu::ORB_GPU::operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors)

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



gpu::ORB_GPU::downloadKeyPoints
-------------------------------------
Download keypoints from GPU to CPU memory.

.. ocv:function:: void gpu::ORB_GPU::downloadKeyPoints( GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints )



gpu::ORB_GPU::convertKeyPoints
-------------------------------------
Converts keypoints from GPU representation to vector of ``KeyPoint``.

.. ocv:function:: void gpu::ORB_GPU::convertKeyPoints( Mat& d_keypoints, std::vector<KeyPoint>& keypoints )



gpu::ORB_GPU::release
-------------------------------------
Releases inner buffer memory.

.. ocv:function:: void gpu::ORB_GPU::release()



gpu::BruteForceMatcher_GPU
--------------------------
.. ocv:class:: template<class Distance> gpu::BruteForceMatcher_GPU

Brute-force descriptor matcher. For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches between descriptor sets. ::

    template<class Distance>
    class BruteForceMatcher_GPU
    {
    public:
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


The class ``BruteForceMatcher_GPU`` has an interface similar to the class :ocv:class:`DescriptorMatcher`. It has two groups of ``match`` methods: for matching descriptors of one image with another image or with an image set. Also, all functions have an alternative to save results either to the GPU memory or to the CPU memory. The ``Distance`` template parameter is kept for CPU/GPU interfaces similarity. ``BruteForceMatcher_GPU`` supports only the ``L1<float>``, ``L2<float>``, and ``Hamming`` distance types.

.. seealso:: :ocv:class:`DescriptorMatcher`, :ocv:class:`BruteForceMatcher`



gpu::BruteForceMatcher_GPU::match
-------------------------------------
Finds the best match for each descriptor from a query set with train descriptors.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::match(const GpuMat& query, const GpuMat& train, std::vector<DMatch>& matches, const GpuMat& mask = GpuMat())

.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchSingle(const GpuMat& query, const GpuMat& train, GpuMat& trainIdx, GpuMat& distance, const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null())

.. ocv:function:: void gpu::BruteForceMatcher_GPU::match(const GpuMat& query, std::vector<DMatch>& matches, const std::vector<GpuMat>& masks = std::vector<GpuMat>())

.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchCollection(const GpuMat& query, const GpuMat& trainCollection, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, const GpuMat& masks, Stream& stream = Stream::Null())

.. seealso:: :ocv:func:`DescriptorMatcher::match`



gpu::BruteForceMatcher_GPU::makeGpuCollection
-------------------------------------------------
Performs a GPU collection of train descriptors and masks in a suitable format for the :ocv:func:`gpu::BruteForceMatcher_GPU::matchCollection` function.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection, const vector<GpuMat>& masks = std::vector<GpuMat>())



gpu::BruteForceMatcher_GPU::matchDownload
---------------------------------------------
Downloads matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::matchSingle` or :ocv:func:`gpu::BruteForceMatcher_GPU::matchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector<DMatch>&matches)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchDownload(const GpuMat& trainIdx, GpuMat& imgIdx, const GpuMat& distance, std::vector<DMatch>&matches)



gpu::BruteForceMatcher_GPU::matchConvert
---------------------------------------------
Converts matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::matchSingle` or :ocv:func:`gpu::BruteForceMatcher_GPU::matchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchConvert(const Mat& trainIdx, const Mat& distance, std::vector<DMatch>&matches)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>&matches)



gpu::BruteForceMatcher_GPU::knnMatch
----------------------------------------
Finds the ``k`` best matches for each descriptor from a query set with train descriptors.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatch(const GpuMat& query, const GpuMat& train, std::vector< std::vector<DMatch> >&matches, int k, const GpuMat& mask = GpuMat(), bool compactResult = false)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatchSingle(const GpuMat& query, const GpuMat& train, GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k, const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null())

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatch(const GpuMat& query, std::vector< std::vector<DMatch> >&matches, int k, const std::vector<GpuMat>&masks = std::vector<GpuMat>(), bool compactResult = false )

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatch2Collection(const GpuMat& query, const GpuMat& trainCollection, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, const GpuMat& maskCollection = GpuMat(), Stream& stream = Stream::Null())

    :param query: Query set of descriptors.

    :param train: Training set of descriptors. It is not be added to train descriptors collection stored in the class object.

    :param k: Number of the best matches per each query descriptor (or less if it is not possible).

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

    :param compactResult: If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.

    :param stream: Stream for the asynchronous version.

The function returns detected ``k`` (or less if not possible) matches in the increasing order by distance.

The third variant of the method stores the results in GPU memory.

.. seealso:: :ocv:func:`DescriptorMatcher::knnMatch`



gpu::BruteForceMatcher_GPU::knnMatchDownload
------------------------------------------------
Downloads matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::knnMatchSingle` or :ocv:func:`gpu::BruteForceMatcher_GPU::knnMatch2Collection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatch2Download(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.



gpu::BruteForceMatcher_GPU::knnMatchConvert
------------------------------------------------
Converts matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::knnMatchSingle` or :ocv:func:`gpu::BruteForceMatcher_GPU::knnMatch2Collection` to CPU vector with :ocv:class:`DMatch`.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatchConvert(const Mat& trainIdx, const Mat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.



gpu::BruteForceMatcher_GPU::radiusMatch
-------------------------------------------
For each query descriptor, finds the best matches with a distance less than a given threshold.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat& query, const GpuMat& train, std::vector< std::vector<DMatch> >&matches, float maxDistance, const GpuMat& mask = GpuMat(), bool compactResult = false)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatchSingle(const GpuMat& query, const GpuMat& train, GpuMat& trainIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance, const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null())

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat& query, std::vector< std::vector<DMatch> >&matches, float maxDistance, const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatchCollection(const GpuMat& query, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance, const std::vector<GpuMat>& masks = std::vector<GpuMat>(), Stream& stream = Stream::Null())

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



gpu::BruteForceMatcher_GPU::radiusMatchDownload
---------------------------------------------------
Downloads matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::radiusMatchSingle` or :ocv:func:`gpu::BruteForceMatcher_GPU::radiusMatchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, const GpuMat& nMatches, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, const GpuMat& nMatches, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.




gpu::BruteForceMatcher_GPU::radiusMatchConvert
---------------------------------------------------
Converts matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::radiusMatchSingle` or :ocv:func:`gpu::BruteForceMatcher_GPU::radiusMatchCollection` to vector with :ocv:class:`DMatch`.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches, std::vector< std::vector<DMatch> >& matches, bool compactResult = false)

If ``compactResult`` is ``true`` , the ``matches`` vector does not contain matches for fully masked-out query descriptors.

