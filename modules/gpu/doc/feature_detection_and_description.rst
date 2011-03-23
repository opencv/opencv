Feature Detection and Description
=================================

.. highlight:: cpp



.. index:: gpu::SURF_GPU

gpu::SURF_GPU
-------------
.. cpp:class:: gpu::SURF_GPU

Class for extracting Speeded Up Robust Features from an image. ::

    class SURF_GPU : public CvSURFParams
    {
    public:
        //! the default constructor
        SURF_GPU();
        //! the full constructor taking all the necessary parameters
        explicit SURF_GPU(double _hessianThreshold, int _nOctaves=4,
             int _nOctaveLayers=2, bool _extended=false, float _keypointsRatio=0.01f, 
             bool _upright = false);

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

        //! max keypoints = keypointsRatio * img.size().area()
        float keypointsRatio;
        
        bool upright;

        GpuMat sum, mask1, maskSum, intBuffer;

        GpuMat det, trace;

        GpuMat maxPosBuffer;
        GpuMat featuresBuffer;
        GpuMat keypointsBuffer;
    };

The class ``SURF_GPU`` implements Speeded Up Robust Features descriptor. There is fast multi-scale Hessian keypoint detector that can be used to find the keypoints (which is the default option), but the descriptors can be also computed for the user-specified keypoints. Supports only 8 bit grayscale images.

The class ``SURF_GPU`` can store results to GPU and CPU memory and provides functions to convert results between CPU and GPU version (``uploadKeypoints``, ``downloadKeypoints``, ``downloadDescriptors``). CPU results has the same format as :c:type:`SURF` results. GPU results are stored to :cpp:class:`gpu::GpuMat`. ``keypoints`` matrix is one row matrix with ``CV_32FC6`` type. It contains 6 float values per feature: ``x, y, laplacian, size, dir, hessian``. ``descriptors`` matrix is ``nFeatures`` :math:`\times` ``descriptorSize`` matrix with ``CV_32FC1`` type.

The class ``SURF_GPU`` uses some buffers and provides access to it. All buffers can be safely released between function calls.

See also: :c:type:`SURF`.



.. index:: gpu::BruteForceMatcher_GPU

gpu::BruteForceMatcher_GPU
--------------------------
.. cpp:class:: gpu::BruteForceMatcher_GPU

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

The class ``BruteForceMatcher_GPU`` has the similar interface to class :c:type:`DescriptorMatcher`. It has two groups of match methods: for matching descriptors of one image with other image or with image set. Also all functions have alternative: save results to GPU memory or to CPU memory.

``Distance`` template parameter is kept for CPU/GPU interfaces similarity. ``BruteForceMatcher_GPU`` supports only ``L1<float>`` and ``L2<float>`` distance types.

See also: :c:type:`DescriptorMatcher`, :c:type:`BruteForceMatcher`.



.. index:: gpu::BruteForceMatcher_GPU::match

gpu::BruteForceMatcher_GPU::match
-------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::match(const GpuMat& queryDescs, const GpuMat& trainDescs, vector<DMatch>& matches, const GpuMat& mask = GpuMat())

.. cpp:function:: void gpu::BruteForceMatcher_GPU::match(const GpuMat& queryDescs, vector<DMatch>& matches, const vector<GpuMat>& masks = vector<GpuMat>())

    Finds the best match for each descriptor from a query set with train descriptors.

See also: :c:func:`DescriptorMatcher::match`.



.. index:: gpu::BruteForceMatcher_GPU::matchSingle

gpu::BruteForceMatcher_GPU::matchSingle
-------------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::matchSingle(const GpuMat& queryDescs, const GpuMat& trainDescs, GpuMat& trainIdx, GpuMat& distance, const GpuMat& mask = GpuMat())

    Finds the best match for each query descriptor. Results will be stored to GPU memory.
    
    :param queryDescs: Query set of descriptors.

    :param trainDescs: Train set of descriptors. This will not be added to train descriptors collection stored in class object.

    :param trainIdx: One row ``CV_32SC1`` matrix. Will contain the best train index for each query. If some query descriptors are masked out in ``mask`` it will contain -1.

    :param distance: One row ``CV_32FC1`` matrix. Will contain the best distance for each query. If some query descriptors are masked out in ``mask`` it will contain ``FLT_MAX``.

    :param mask: Mask specifying permissible matches between input query and train matrices of descriptors.



.. index:: gpu::BruteForceMatcher_GPU::matchCollection

gpu::BruteForceMatcher_GPU::matchCollection
-----------------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::matchCollection(const GpuMat& queryDescs, const GpuMat& trainCollection, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, const GpuMat& maskCollection)

    Find the best match for each query descriptor from train collection. Results will be stored to GPU memory.
    
    :param queryDescs: Query set of descriptors.

    :param trainCollection: :cpp:class:`gpu::GpuMat` containing train collection. It can be obtained from train descriptors collection that was set using ``add`` method by :cpp:func:`gpu::BruteForceMatcher_GPU::makeGpuCollection`. Or it can contain user defined collection. It must be one row matrix, each element is a :cpp:class:`gpu::DevMem2D_` that points to one train descriptors matrix.

    :param trainIdx: One row ``CV_32SC1`` matrix. Will contain the best train index for each query. If some query descriptors are masked out in ``maskCollection`` it will contain -1.

    :param imgIdx: One row ``CV_32SC1`` matrix. Will contain image train index for each query. If some query descriptors are masked out in ``maskCollection`` it will contain -1.

    :param distance: One row ``CV_32FC1`` matrix. Will contain the best distance for each query. If some query descriptors are masked out in ``maskCollection`` it will contain ``FLT_MAX``.

    :param maskCollection: :cpp:class:`gpu::GpuMat` containing set of masks. It can be obtained from ``vector<GpuMat>`` by :cpp:func:`gpu::BruteForceMatcher_GPU::makeGpuCollection`. Or it can contain user defined mask set. It must be empty matrix or one row matrix, each element is a :cpp:class:`gpu::PtrStep_` that points to one mask.



.. index:: gpu::BruteForceMatcher_GPU::makeGpuCollection

gpu::BruteForceMatcher_GPU::makeGpuCollection
-------------------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection, const vector<GpuMat>& masks = vector<GpuMat>())

    Makes gpu collection of train descriptors and masks in suitable format for :cpp:func:`gpu::BruteForceMatcher_GPU::matchCollection` function.



.. index:: gpu::BruteForceMatcher_GPU::matchDownload

gpu::BruteForceMatcher_GPU::matchDownload
---------------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::matchDownload(const GpuMat& trainIdx, const GpuMat& distance, vector<DMatch>& matches)

.. cpp:function:: void gpu::BruteForceMatcher_GPU::matchDownload(const GpuMat& trainIdx, GpuMat&imgIdx, const GpuMat& distance, vector<DMatch>& matches)

    Downloads ``trainIdx``, ``imgIdx`` and ``distance`` matrices obtained via :cpp:func:`gpu::BruteForceMatcher_GPU::matchSingle` or :cpp:func:`gpu::BruteForceMatcher_GPU::matchCollection` to CPU vector with :c:type:`DMatch`.



.. index:: gpu::BruteForceMatcher_GPU::knnMatch

gpu::BruteForceMatcher_GPU::knnMatch
----------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs, vector< vector<DMatch> >& matches, int k, const GpuMat& mask = GpuMat(), bool compactResult = false)

.. cpp:function:: void gpu::BruteForceMatcher_GPU::knnMatch(const GpuMat& queryDescs, vector< vector<DMatch> >& matches, int k, const vector<GpuMat>& masks = vector<GpuMat>(), bool compactResult = false)

    Finds the k best matches for each descriptor from a query set with train descriptors. Found k (or less if not possible) matches are returned in distance increasing order.

.. cpp:function:: void gpu::BruteForceMatcher_GPU::knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs, GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k, const GpuMat& mask = GpuMat())

    Finds the k best matches for each descriptor from a query set with train descriptors. Found k (or less if not possible) matches are returned in distance increasing order. Results will be stored to GPU memory.
    
    :param queryDescs: Query set of descriptors.

    :param trainDescs; Train set of descriptors. This will not be added to train descriptors collection stored in class object.

    :param trainIdx: Matrix with ``nQueries`` :math:`\times` ``k`` size and ``CV_32SC1`` type. ``trainIdx.at<int>(queryIdx, i)`` will contain index of the i'th best trains. If some query descriptors are masked out in ``mask`` it will contain -1.

    :param distance: Matrix with ``nQuery`` :math:`\times` ``k`` and ``CV_32FC1`` type. Will contain distance for each query and the i'th best trains. If some query descriptors are masked out in ``mask`` it will contain ``FLT_MAX``.

    :param allDist: Buffer to store all distances between query descriptors and train descriptors. It will have ``nQuery`` :math:`\times` ``nTrain`` size and ``CV_32FC1`` type. ``allDist.at<float>(queryIdx, trainIdx)`` will contain ``FLT_MAX``, if ``trainIdx`` is one from k best, otherwise it will contain distance between ``queryIdx`` and ``trainIdx`` descriptors.

    :param k: Number of the best matches will be found per each query descriptor (or less if it's not possible).

    :param mask: Mask specifying permissible matches between input query and train matrices of descriptors.

See also: :c:func:`DescriptorMatcher::knnMatch`.


.. index:: gpu::BruteForceMatcher_GPU::knnMatchDownload

gpu::BruteForceMatcher_GPU::knnMatchDownload
------------------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, vector< vector<DMatch> >& matches, bool compactResult = false)

    Downloads ``trainIdx`` and ``distance`` matrices obtained via :cpp:func:`gpu::BruteForceMatcher_GPU::knnMatch` to CPU vector with :c:type:`DMatch`. If ``compactResult`` is true ``matches`` vector will not contain matches for fully masked out query descriptors.



.. index:: gpu::BruteForceMatcher_GPU::radiusMatch

gpu::BruteForceMatcher_GPU::radiusMatch
-------------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs, vector< vector<DMatch> >& matches, float maxDistance, const GpuMat& mask = GpuMat(), bool compactResult = false)

.. cpp:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat& queryDescs, vector< vector<DMatch> >& matches, float maxDistance, const vector<GpuMat>& masks = vector<GpuMat>(), bool compactResult = false)

    Finds the best matches for each query descriptor which have distance less than given threshold. Found matches are returned in distance increasing order.

.. cpp:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat&queryDescs,  const GpuMat&trainDescs,  GpuMat&trainIdx,  GpuMat&nMatches,  GpuMat&distance,  float maxDistance,  const GpuMat&mask = GpuMat())

    Finds the best matches for each query descriptor which have distance less than given threshold. Results will be stored to GPU memory. Results are not sorted by distance increasing order.
    
    :param queryDescs: Query set of descriptors.

    :param trainDescs: Train set of descriptors. This will not be added to train descriptors collection stored in class object.

    :param trainIdx: ``trainIdx.at<int>(queryIdx, i)`` will contain i'th train index ``(i < min(nMatches.at<unsigned int>(0, queryIdx), trainIdx.cols)``. If ``trainIdx`` is empty, it will be created with size ``nQuery`` :math:`\times` ``nTrain``. Or it can be allocated by user (it must have ``nQuery`` rows and ``CV_32SC1`` type). Cols can be less than ``nTrain``, but it can be that matcher won't find all matches, because it haven't enough memory to store results.

    :param nMatches: ``nMatches.at<unsigned int>(0, queryIdx)`` will contain matches count for ``queryIdx``. Carefully, ``nMatches`` can be greater than ``trainIdx.cols`` - it means that matcher didn't find all matches, because it didn't have enough memory.

    :param distance: ``distance.at<int>(queryIdx, i)`` will contain i'th distance ``(i < min(nMatches.at<unsigned int>(0, queryIdx), trainIdx.cols)``. If ``trainIdx`` is empty, it will be created with size ``nQuery`` :math:`\times` ``nTrain``. Otherwise it must be also allocated by user (it must have the same size as ``trainIdx`` and ``CV_32FC1`` type).

    :param maxDistance: Distance threshold.

    :param mask: Mask specifying permissible matches between input query and train matrices of descriptors.

**Please note:** This function works only on devices with Compute Capability :math:`>=` 1.1.

See also: :c:func:`DescriptorMatcher::radiusMatch`.



.. index:: gpu::BruteForceMatcher_GPU::radiusMatchDownload

gpu::BruteForceMatcher_GPU::radiusMatchDownload
---------------------------------------------------
.. cpp:function:: void gpu::BruteForceMatcher_GPU::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& nMatches, const GpuMat& distance, vector< vector<DMatch> >& matches, bool compactResult = false)

    Downloads ``trainIdx``, ``nMatches`` and ``distance`` matrices obtained via :cpp:func:`gpu::BruteForceMatcher_GPU::radiusMatch` to CPU vector with :c:type:`DMatch`. If ``compactResult`` is true ``matches`` vector will not contain matches for fully masked out query descriptors.
