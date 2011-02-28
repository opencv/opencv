Feature Detection and Description
=================================

.. highlight:: cpp

.. index:: gpu::SURF_GPU

.. _gpu::SURF_GPU:

gpu::SURF_GPU
-------------
.. c:type:: gpu::SURF_GPU

Class for extracting Speeded Up Robust Features from an image. ::

    class SURF_GPU : public SURFParams_GPU
    {
    public:
        //! returns the descriptor size in float's (64 or 128)
        int descriptorSize() const;

        //! upload host keypoints to device memory
        static void uploadKeypoints(const vector<KeyPoint>& keypoints,
            GpuMat& keypointsGPU);
        //! download keypoints from device to host memory
        static void downloadKeypoints(const GpuMat& keypointsGPU,
            vector<KeyPoint>& keypoints);

        //! download descriptors from device to host memory
        static void downloadDescriptors(const GpuMat& descriptorsGPU,
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

        GpuMat sum;
        GpuMat sumf;

        GpuMat mask1;
        GpuMat maskSum;

        GpuMat hessianBuffer;
        GpuMat maxPosBuffer;
        GpuMat featuresBuffer;
    };


The class ``SURF_GPU`` implements Speeded Up Robust Features descriptor. There is fast multi-scale Hessian keypoint detector that can be used to find the keypoints (which is the default option), but the descriptors can be also computed for the user-specified keypoints. Supports only 8 bit grayscale images.

The class ``SURF_GPU`` can store results to GPU and CPU memory and provides static functions to convert results between CPU and GPU version ( ``uploadKeypoints``,``downloadKeypoints``,``downloadDescriptors`` ). CPU results has the same format as
results. GPU results are stored to ``GpuMat`` . ``keypoints`` matrix is one row matrix with ``CV_32FC6`` type. It contains 6 float values per feature: ``x, y, size, response, angle, octave`` . ``descriptors`` matrix is
:math:`\texttt{nFeatures} \times \texttt{descriptorSize}` matrix with ``CV_32FC1`` type.

The class ``SURF_GPU`` uses some buffers and provides access to it. All buffers can be safely released between function calls.

See also:
.

.. index:: gpu::BruteForceMatcher_GPU

.. _gpu::BruteForceMatcher_GPU:

gpu::BruteForceMatcher_GPU
--------------------------
.. c:type:: gpu::BruteForceMatcher_GPU

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

        // Return true if there are not train descriptors in collection.
        bool empty() const;

        // Return true if the matcher supports mask in match methods.
        bool isMaskSupported() const;

        void matchSingle(const GpuMat& queryDescs, const GpuMat& trainDescs,
            GpuMat& trainIdx, GpuMat& distance,
            const GpuMat& mask = GpuMat());

        static void matchDownload(const GpuMat& trainIdx,
            const GpuMat& distance, std::vector<DMatch>& matches);

        void match(const GpuMat& queryDescs, const GpuMat& trainDescs,
            std::vector<DMatch>& matches, const GpuMat& mask = GpuMat());

        void makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection,
            const vector<GpuMat>& masks = std::vector<GpuMat>());

        void matchCollection(const GpuMat& queryDescs,
            const GpuMat& trainCollection,
            GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
            const GpuMat& maskCollection);

        static void matchDownload(const GpuMat& trainIdx, GpuMat& imgIdx,
            const GpuMat& distance, std::vector<DMatch>& matches);

        void match(const GpuMat& queryDescs, std::vector<DMatch>& matches,
            const std::vector<GpuMat>& masks = std::vector<GpuMat>());

        void knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
            GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k,
            const GpuMat& mask = GpuMat());

        static void knnMatchDownload(const GpuMat& trainIdx,
            const GpuMat& distance, std::vector< std::vector<DMatch> >& matches,
            bool compactResult = false);

        void knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
            std::vector< std::vector<DMatch> >& matches, int k,
            const GpuMat& mask = GpuMat(), bool compactResult = false);

        void knnMatch(const GpuMat& queryDescs,
            std::vector< std::vector<DMatch> >& matches, int knn,
            const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
            bool compactResult = false );

        void radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
            GpuMat& trainIdx, GpuMat& nMatches, GpuMat& distance,
            float maxDistance, const GpuMat& mask = GpuMat());

        static void radiusMatchDownload(const GpuMat& trainIdx,
            const GpuMat& nMatches, const GpuMat& distance,
            std::vector< std::vector<DMatch> >& matches,
            bool compactResult = false);

        void radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
            std::vector< std::vector<DMatch> >& matches, float maxDistance,
            const GpuMat& mask = GpuMat(), bool compactResult = false);

        void radiusMatch(const GpuMat& queryDescs,
            std::vector< std::vector<DMatch> >& matches, float maxDistance,
            const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
            bool compactResult = false);

    private:
        std::vector<GpuMat> trainDescCollection;
    };


The class ``BruteForceMatcher_GPU`` has the similar interface to class
. It has two groups of match methods: for matching descriptors of one image with other image or with image set. Also all functions have alternative: save results to GPU memory or to CPU memory.
 ``Distance`` template parameter is kept for CPU/GPU interfaces similarity. ``BruteForceMatcher_GPU`` supports only ``L1<float>`` and ``L2<float>`` distance types.

See also:,.

.. index:: cv::gpu::BruteForceMatcher_GPU::match

.. _cv::gpu::BruteForceMatcher_GPU::match:

cv::gpu::BruteForceMatcher_GPU::match
-------------------------------------
.. c:function:: void match(const GpuMat\& queryDescs,  const GpuMat\& trainDescs,  std::vector<DMatch>\& matches,  const GpuMat\& mask = GpuMat())

.. c:function:: void match(const GpuMat\& queryDescs,  std::vector<DMatch>\& matches,  const std::vector<GpuMat>\& masks = std::vector<GpuMat>())

    Finds the best match for each descriptor from a query set with train descriptors.

See also:
:func:`DescriptorMatcher::match` .

.. index:: cv::gpu::BruteForceMatcher_GPU::matchSingle

.. _cv::gpu::BruteForceMatcher_GPU::matchSingle:

cv::gpu::BruteForceMatcher_GPU::matchSingle
-------------------------------------------
.. c:function:: void matchSingle(const GpuMat\& queryDescs,  const GpuMat\& trainDescs,  GpuMat\& trainIdx,  GpuMat\& distance,  const GpuMat\& mask = GpuMat())

    Finds the best match for each query descriptor. Results will be stored to GPU memory.

    {Query set of descriptors.}
    {Train set of descriptors. This will not be added to train descriptors collection stored in class object.}
    {One row ``CV_32SC1``     matrix. Will contain the best train index for each query. If some query descriptors are masked out in ``mask``     it will contain -1.}
    {One row ``CV_32FC1``     matrix. Will contain the best distance for each query. If some query descriptors are masked out in ``mask``     it will contain ``FLT_MAX``     .}

    :param mask: Mask specifying permissible matches between input query and train matrices of descriptors.

.. index:: cv::gpu::BruteForceMatcher_GPU::matchCollection

.. _cv::gpu::BruteForceMatcher_GPU::matchCollection:

cv::gpu::BruteForceMatcher_GPU::matchCollection
-----------------------------------------------
.. c:function:: void matchCollection(const GpuMat\& queryDescs,  const GpuMat\& trainCollection,  GpuMat\& trainIdx,  GpuMat\& imgIdx,  GpuMat\& distance,  const GpuMat\& maskCollection)

    Find the best match for each query descriptor from train collection. Results will be stored to GPU memory.

    {Query set of descriptors.}
    { ``GpuMat``     containing train collection. It can be obtained from train descriptors collection that was set using ``add``     method by
    . Or it can contain user defined collection. It must be one row matrix, each element is a ``DevMem2D``     that points to one train descriptors matrix.}
    {One row ``CV_32SC1``     matrix. Will contain the best train index for each query. If some query descriptors are masked out in ``maskCollection``     it will contain -1.}
    {One row ``CV_32SC1``     matrix. Will contain image train index for each query. If some query descriptors are masked out in ``maskCollection``     it will contain -1.}
    {One row ``CV_32FC1``     matrix. Will contain the best distance for each query. If some query descriptors are masked out in ``maskCollection``     it will contain ``FLT_MAX``     .}

    :param maskCollection: ``GpuMat``  containing set of masks. It can be obtained from  ``std::vector<GpuMat>``  by  . Or it can contain user defined mask set. It must be empty matrix or one row matrix, each element is a  ``PtrStep``  that points to one mask.

.. index:: cv::gpu::BruteForceMatcher_GPU::makeGpuCollection

.. _cv::gpu::BruteForceMatcher_GPU::makeGpuCollection:

cv::gpu::BruteForceMatcher_GPU::makeGpuCollection
-------------------------------------------------
.. c:function:: void makeGpuCollection(GpuMat\& trainCollection,  GpuMat\& maskCollection,  const vector<GpuMat>\& masks = std::vector<GpuMat>())

    Makes gpu collection of train descriptors and masks in suitable format for function.

.. index:: cv::gpu::BruteForceMatcher_GPU::matchDownload

.. _cv::gpu::BruteForceMatcher_GPU::matchDownload:

cv::gpu::BruteForceMatcher_GPU::matchDownload
--------------------------------------------- ```` ```` ````
.. c:function:: void matchDownload(const GpuMat\& trainIdx,  const GpuMat\& distance,  std::vector<DMatch>\& matches)

.. c:function:: void matchDownload(const GpuMat\& trainIdx,  GpuMat\& imgIdx,  const GpuMat\& distance,  std::vector<DMatch>\& matches)

    Downloads trainIdx, imgIdxand distancematrices obtained via or to CPU vector with .

.. index:: cv::gpu::BruteForceMatcher_GPU::knnMatch

.. _cv::gpu::BruteForceMatcher_GPU::knnMatch:

cv::gpu::BruteForceMatcher_GPU::knnMatch
----------------------------------------
.. c:function:: void knnMatch(const GpuMat\& queryDescs,  const GpuMat\& trainDescs,  std::vector< std::vector<DMatch> >\& matches,  int k,  const GpuMat\& mask = GpuMat(),  bool compactResult = false)

    Finds the k best matches for each descriptor from a query set with train descriptors. Found k (or less if not possible) matches are returned in distance increasing order.

.. c:function:: void knnMatch(const GpuMat\& queryDescs,  std::vector< std::vector<DMatch> >\& matches,  int k,  const std::vector<GpuMat>\& masks = std::vector<GpuMat>(),  bool compactResult = false )

See also:
:func:`DescriptorMatcher::knnMatch` .

.. index:: cv::gpu::BruteForceMatcher_GPU::knnMatch

.. _cv::gpu::BruteForceMatcher_GPU::knnMatch:

cv::gpu::BruteForceMatcher_GPU::knnMatch
----------------------------------------
.. c:function:: void knnMatch(const GpuMat\& queryDescs,  const GpuMat\& trainDescs,  GpuMat\& trainIdx,  GpuMat\& distance,  GpuMat\& allDist,  int k,  const GpuMat\& mask = GpuMat())

    Finds the k best matches for each descriptor from a query set with train descriptors. Found k (or less if not possible) matches are returned in distance increasing order. Results will be stored to GPU memory.

    {Query set of descriptors.}
    {Train set of descriptors. This will not be added to train descriptors collection stored in class object.}
    {Matrix with
    :math:`\texttt{nQueries} \times \texttt{k}`     size and ``CV_32SC1``     type. ``trainIdx.at<int>(queryIdx, i)``     will contain index of the i'th best trains. If some query descriptors are masked out in ``mask``     it will contain -1.}
    {Matrix with
    :math:`\texttt{nQuery} \times \texttt{k}`     and ``CV_32FC1``     type. Will contain distance for each query and the i'th best trains. If some query descriptors are masked out in ``mask``     it will contain ``FLT_MAX``     .}
    {Buffer to store all distances between query descriptors and train descriptors. It will have
    :math:`\texttt{nQuery} \times \texttt{nTrain}`     size and ``CV_32FC1``     type. ``allDist.at<float>(queryIdx, trainIdx)``     will contain ``FLT_MAX``     , if ``trainIdx``     is one from k best, otherwise it will contain distance between ``queryIdx``     and ``trainIdx``     descriptors.}

    :param k: Number of the best matches will be found per each query descriptor (or less if it's not possible).

    :param mask: Mask specifying permissible matches between input query and train matrices of descriptors.

.. index:: cv::gpu::BruteForceMatcher_GPU::knnMatchDownload

.. _cv::gpu::BruteForceMatcher_GPU::knnMatchDownload:

cv::gpu::BruteForceMatcher_GPU::knnMatchDownload
------------------------------------------------ ```` ```` ```` ````
.. c:function:: void knnMatchDownload(const GpuMat\& trainIdx,  const GpuMat\& distance,  std::vector< std::vector<DMatch> >\& matches,  bool compactResult = false)

    Downloads trainIdxand distancematrices obtained via to CPU vector with . If compactResultis true matchesvector will not contain matches for fully masked out query descriptors.

.. index:: cv::gpu::BruteForceMatcher_GPU::radiusMatch

.. _cv::gpu::BruteForceMatcher_GPU::radiusMatch:

cv::gpu::BruteForceMatcher_GPU::radiusMatch
-------------------------------------------
.. c:function:: void radiusMatch(const GpuMat\& queryDescs,  const GpuMat\& trainDescs,  std::vector< std::vector<DMatch> >\& matches,  float maxDistance,  const GpuMat\& mask = GpuMat(),  bool compactResult = false)

    Finds the best matches for each query descriptor which have distance less than given threshold. Found matches are returned in distance increasing order.

.. c:function:: void radiusMatch(const GpuMat\& queryDescs,  std::vector< std::vector<DMatch> >\& matches,  float maxDistance,  const std::vector<GpuMat>\& masks = std::vector<GpuMat>(),  bool compactResult = false)

This function works only on devices with Compute Capability
:math:`>=` 1.1.

See also:
:func:`DescriptorMatcher::radiusMatch` .

.. index:: cv::gpu::BruteForceMatcher_GPU::radiusMatch

.. _cv::gpu::BruteForceMatcher_GPU::radiusMatch:

cv::gpu::BruteForceMatcher_GPU::radiusMatch
-------------------------------------------
.. c:function:: void radiusMatch(const GpuMat\& queryDescs,  const GpuMat\& trainDescs,  GpuMat\& trainIdx,  GpuMat\& nMatches,  GpuMat\& distance,  float maxDistance,  const GpuMat\& mask = GpuMat())

    Finds the best matches for each query descriptor which have distance less than given threshold. Results will be stored to GPU memory.

    {Query set of descriptors.}
    {Train set of descriptors. This will not be added to train descriptors collection stored in class object.}
    { ``trainIdx.at<int>(queryIdx, i)``     will contain i'th train index ``(i < min(nMatches.at<unsigned int>(0, queryIdx), trainIdx.cols)``     . If ``trainIdx``     is empty, it will be created with size
    :math:`\texttt{nQuery} \times \texttt{nTrain}`     . Or it can be allocated by user (it must have ``nQuery``     rows and ``CV_32SC1``     type). Cols can be less than ``nTrain``     , but it can be that matcher won't find all matches, because it haven't enough memory to store results.}
    { ``nMatches.at<unsigned int>(0, queryIdx)``     will contain matches count for ``queryIdx``     . Carefully, ``nMatches``     can be greater than ``trainIdx.cols``     - it means that matcher didn't find all matches, because it didn't have enough memory.}
    { ``distance.at<int>(queryIdx, i)``     will contain i'th distance ``(i < min(nMatches.at<unsigned int>(0, queryIdx), trainIdx.cols)``     . If ``trainIdx``     is empty, it will be created with size
    :math:`\texttt{nQuery} \times \texttt{nTrain}`     . Otherwise it must be also allocated by user (it must have the same size as ``trainIdx``     and ``CV_32FC1``     type).}

    :param maxDistance: Distance threshold.

    :param mask: Mask specifying permissible matches between input query and train matrices of descriptors.

In contrast to
results are not sorted by distance increasing order.

This function works only on devices with Compute Capability
:math:`>=` 1.1.

.. index:: cv::gpu::BruteForceMatcher_GPU::radiusMatchDownload

.. _cv::gpu::BruteForceMatcher_GPU::radiusMatchDownload:

cv::gpu::BruteForceMatcher_GPU::radiusMatchDownload
--------------------------------------------------- ```` ```` ```` ```` ````
.. c:function:: void radiusMatchDownload(const GpuMat\& trainIdx,  const GpuMat\& nMatches,  const GpuMat\& distance,  std::vector< std::vector<DMatch> >\& matches,  bool compactResult = false)

    Downloads trainIdx, nMatchesand distancematrices obtained via to CPU vector with . If compactResultis true matchesvector will not contain matches for fully masked out query descriptors.

