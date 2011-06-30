Feature Detection and Description
=================================

.. highlight:: cpp

.. index:: gpu::SURF_GPU

gpu::SURF_GPU
-------------
.. ocv:class:: gpu::SURF_GPU

Class used for extracting Speeded Up Robust Features (SURF) from an image. 
::

    class SURF_GPU : public CvSURFParams
    {
    public:
        enum KeypointLayout 
        {
            SF_X = 0,
            SF_Y,
            SF_LAPLACIAN,
            SF_SIZE,
            SF_DIR,
            SF_HESSIAN,
            SF_FEATURE_STRIDE
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

        //! max keypoints = keypointsRatio * img.size().area()
        float keypointsRatio;

        bool upright;

        GpuMat sum, mask1, maskSum, intBuffer;

        GpuMat det, trace;

        GpuMat maxPosBuffer;
    };


The class ``SURF_GPU`` implements Speeded Up Robust Features descriptor. There is a fast multi-scale Hessian keypoint detector that can be used to find the keypoints (which is the default option). But the descriptors can also be computed for the user-specified keypoints. Only 8-bit grayscale images are supported.

The class ``SURF_GPU`` can store results in the GPU and CPU memory. It provides functions to convert results between CPU and GPU version ( ``uploadKeypoints``, ``downloadKeypoints``, ``downloadDescriptors``). The format of CPU results is the same as ``SURF`` results. GPU results are stored in  ``GpuMat``. The ``keypoints`` matrix is :math:`\texttt{nFeatures} \times 6` matrix with the ``CV_32FC1`` type.

* ``keypoints.ptr<float>(SF_X)[i]`` contains x coordinate of the i-th feature. 
* ``keypoints.ptr<float>(SF_Y)[i]`` contains y coordinate of the i-th feature. 
* ``keypoints.ptr<float>(SF_LAPLACIAN)[i]``  contains the laplacian sign of the i-th feature. 
* ``keypoints.ptr<float>(SF_SIZE)[i]`` contains the size of the i-th feature. 
* ``keypoints.ptr<float>(SF_DIR)[i]`` contain orientation of the i-th feature. 
* ``keypoints.ptr<float>(SF_HESSIAN)[i]`` contains the response of the i-th feature. 

The ``descriptors`` matrix is :math:`\texttt{nFeatures} \times \texttt{descriptorSize}` matrix with the ``CV_32FC1`` type.

The class ``SURF_GPU`` uses some buffers and provides access to it. All buffers can be safely released between function calls.

.. seealso:: 
   :ocv:class:`SURF`

.. index:: gpu::BruteForceMatcher_GPU

gpu::BruteForceMatcher_GPU
--------------------------
.. ocv:class:: gpu::BruteForceMatcher_GPU

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


The class ``BruteForceMatcher_GPU`` has an interface similar to the class :ocv:class:`DescriptorMatcher`. It has two groups of ``match`` methods: for matching descriptors of one image with another image or with an image set. Also, all functions have an alternative to save results either to the GPU memory or to the CPU memory. The ``Distance`` template parameter is kept for CPU/GPU interfaces similarity. ``BruteForceMatcher_GPU`` supports only the ``L1<float>``, ``L2<float>``, and ``Hamming`` distance types.

.. seealso:: 
  :ocv:class:`DescriptorMatcher`, 
  :ocv:class:`BruteForceMatcher`

.. index:: gpu::BruteForceMatcher_GPU::match

gpu::BruteForceMatcher_GPU::match
-------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::match(const GpuMat& queryDescs, const GpuMat& trainDescs, std::vector<DMatch>& matches, const GpuMat& mask = GpuMat())

.. ocv:function:: void gpu::BruteForceMatcher_GPU::match(const GpuMat& queryDescs, std::vector<DMatch>& matches, const std::vector<GpuMat>& masks = std::vector<GpuMat>())

    Finds the best match for each descriptor from a query set with train descriptors.

.. seealso::
   :ocv:func:`DescriptorMatcher::match` 

.. index:: gpu::BruteForceMatcher_GPU::matchSingle

gpu::BruteForceMatcher_GPU::matchSingle
-------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchSingle(const GpuMat& queryDescs, const GpuMat& trainDescs, GpuMat& trainIdx, GpuMat& distance, const GpuMat& mask = GpuMat())

    Finds the best match for each query descriptor. Results are stored in the GPU memory.

    :param queryDescs: Query set of descriptors.
    
    :param trainDescs: Training set of descriptors. It is not added to train descriptors collection stored in the class object.
    
    :param trainIdx: Output single-row ``CV_32SC1`` matrix that contains the best train index for each query. If some query descriptors are masked out in ``mask`` , it contains -1.
    
    :param distance: Output single-row ``CV_32FC1`` matrix that contains the best distance for each query. If some query descriptors are masked out in ``mask``, it contains ``FLT_MAX``.

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

.. index:: gpu::BruteForceMatcher_GPU::matchCollection

gpu::BruteForceMatcher_GPU::matchCollection
-----------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchCollection(const GpuMat& queryDescs, const GpuMat& trainCollection, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, const GpuMat& maskCollection)

    Finds the best match for each query descriptor from train collection. Results are stored in the GPU memory.

   :param queryDescs: Query set of descriptors.
    
   :param trainCollection: :ocv:class:`gpu::GpuMat` containing train collection. It can be obtained from the collection of train descriptors that was set using the ``add``     method by :ocv:func:`gpu::BruteForceMatcher_GPU::makeGpuCollection`. Or it may contain a user-defined collection. This is a one-row matrix where each element is ``DevMem2D`` pointing out to a matrix of train descriptors.
    
   :param trainIdx: Output single-row ``CV_32SC1`` matrix that contains the best train index for each query. If some query descriptors are masked out in ``maskCollection``  , it contains -1.
    
   :param imgIdx: Output single-row ``CV_32SC1`` matrix that contains image train index for each query. If some query descriptors are masked out in ``maskCollection``  , it contains -1.
    
   :param distance: Output single-row ``CV_32FC1`` matrix that contains the best distance for each query. If some query descriptors are masked out in ``maskCollection``  , it contains ``FLT_MAX``.

   :param maskCollection: ``GpuMat``  containing a set of masks. It can be obtained from  ``std::vector<GpuMat>``  by  :ocv:func:`gpu::BruteForceMatcher_GPU::makeGpuCollection` or it may contain  a user-defined mask set. This is an empty matrix or one-row matrix where each element is a  ``PtrStep``  that points to one mask.

.. index:: gpu::BruteForceMatcher_GPU::makeGpuCollection

gpu::BruteForceMatcher_GPU::makeGpuCollection
-------------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection, const vector<GpuMat>&masks = std::vector<GpuMat>())

    Performs a GPU collection of train descriptors and masks in a suitable format for the 
    :ocv:func:`gpu::BruteForceMatcher_GPU::matchCollection` function.

.. index:: gpu::BruteForceMatcher_GPU::matchDownload

gpu::BruteForceMatcher_GPU::matchDownload
---------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector<DMatch>&matches)

.. ocv:function:: void gpu::BruteForceMatcher_GPU::matchDownload(const GpuMat& trainIdx, GpuMat& imgIdx, const GpuMat& distance, std::vector<DMatch>&matches)

    Downloads ``trainIdx``, ``imgIdx``, and ``distance`` matrices obtained via 
    :ocv:func:`gpu::BruteForceMatcher_GPU::matchSingle` or 
    :ocv:func:`gpu::BruteForceMatcher_GPU::matchCollection` to CPU vector with :ocv:class:`DMatch`.

.. index:: gpu::BruteForceMatcher_GPU::knnMatch

gpu::BruteForceMatcher_GPU::knnMatch
----------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs, std::vector< std::vector<DMatch> >&matches, int k, const GpuMat& mask = GpuMat(), bool compactResult = false)

    Finds the k best matches for each descriptor from a query set with train descriptors. The function returns detected k (or less if not possible) matches in the increasing order by distance.

.. ocv:function:: void knnMatch(const GpuMat& queryDescs, std::vector< std::vector<DMatch> >&matches, int k, const std::vector<GpuMat>&masks = std::vector<GpuMat>(), bool compactResult = false )

.. seealso::
   :ocv:func:`DescriptorMatcher::knnMatch` 

.. index:: gpu::BruteForceMatcher_GPU::knnMatch

gpu::BruteForceMatcher_GPU::knnMatch
----------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs, GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k, const GpuMat& mask = GpuMat())

    Finds the k best matches for each descriptor from a query set with train descriptors. The function returns detected k (or less if not possible) matches in the increasing order by distance. Results are stored in the GPU memory.

    :param queryDescs: Query set of descriptors.
    :param trainDescs: Training set of descriptors. It is not be added to train descriptors collection stored in the class object.
    :param trainIdx: Output matrix of ``queryDescs.rows x k`` size and ``CV_32SC1`` type. ``trainIdx.at<int>(i, j)`` contains an index of the j-th best match for the i-th query descriptor. If some query descriptors are masked out in ``mask``, it contains -1.
    :param distance: Output matrix of ``queryDescs.rows x k`` size and ``CV_32FC1`` type. ``distance.at<float>(i, j)`` contains a distance from the j-th best match for the i-th query descriptor to the query descriptor. If some query descriptors are masked out in ``mask``, it contains ``FLT_MAX``.
    :param allDist: Floating-point matrix of the size ``queryDescs.rows x trainDescs.rows``. This is a buffer to store all distances between each query descriptors and each train descriptor. On output, ``allDist.at<float>(queryIdx, trainIdx)`` contains ``FLT_MAX`` if ``trainIdx`` is one from k best.

    :param k: Number of the best matches per each query descriptor (or less if it is not possible).

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

.. index:: gpu::BruteForceMatcher_GPU::knnMatchDownload

gpu::BruteForceMatcher_GPU::knnMatchDownload
------------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

    Downloads ``trainIdx`` and ``distance`` matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::knnMatch` to CPU vector with :ocv:class:`DMatch`. If ``compactResult`` is true, the ``matches`` vector does not contain matches for fully masked-out query descriptors.

.. index:: gpu::BruteForceMatcher_GPU::radiusMatch

gpu::BruteForceMatcher_GPU::radiusMatch
-------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs, std::vector< std::vector<DMatch> >&matches, float maxDistance, const GpuMat& mask = GpuMat(), bool compactResult = false)

    For each query descriptor, finds the best matches with a distance less than a given threshold. The function returns detected matches in the increasing order by distance.

.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat& queryDescs, std::vector< std::vector<DMatch> >&matches, float maxDistance, const std::vector<GpuMat>&masks = std::vector<GpuMat>(), bool compactResult = false)

This function works only on devices with the compute capability  :math:`>=` 1.1.

.. seealso::
   :ocv:func:`DescriptorMatcher::radiusMatch` 

.. index:: gpu::BruteForceMatcher_GPU::radiusMatch

gpu::BruteForceMatcher_GPU::radiusMatch
-------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs, GpuMat& trainIdx, GpuMat& nMatches, GpuMat& distance, float maxDistance, const GpuMat& mask = GpuMat())

    For each query descriptor, finds the best matches with a distance less than a given threshold (``maxDistance``). The results are stored in the GPU memory.

    :param queryDescs: Query set of descriptors.
    
    :param trainDescs: Training set of descriptors. It is not added to train descriptors collection stored in the class object.
    
    :param trainIdx: ``trainIdx.at<int>(i, j)`` , the index of j-th training descriptor, which is close enough to i-th query descriptor. If ``trainIdx`` is empty, it is created with the size ``queryDescs.rows x trainDescs.rows``. When the matrix is pre-allocated, it can have less than ``trainDescs.rows`` columns. Then, the function returns as many matches for each query descriptor as fit into the matrix.
    
    :param nMatches: ``nMatches.at<unsigned int>(0, i)`` containing the number of matching descriptors for the i-th query descriptor. The value can be larger than ``trainIdx.cols`` , which means that the function could not store all the matches since it does not have enough memory.
    
    :param distance: Distance ``distance.at<int>(i, j)``  between the j-th match for the j-th query descriptor and this very query descriptor. The matrix has the ``CV_32FC1`` type and the same size as ``trainIdx``.

    :param maxDistance: Distance threshold.

    :param mask: Mask specifying permissible matches between the input query and train matrices of descriptors.

    In contrast to :ocv:func:`gpu::BruteForceMatcher_GPU::knnMatch`, here the results are not sorted by the distance. This function works only on devices with the compute capability >= 1.1.

.. index:: gpu::BruteForceMatcher_GPU::radiusMatchDownload

gpu::BruteForceMatcher_GPU::radiusMatchDownload
---------------------------------------------------
.. ocv:function:: void gpu::BruteForceMatcher_GPU::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& nMatches, const GpuMat& distance, std::vector< std::vector<DMatch> >&matches, bool compactResult = false)

    Downloads ``trainIdx``, ``nMatches`` and ``distance`` matrices obtained via :ocv:func:`gpu::BruteForceMatcher_GPU::radiusMatch` to CPU vector with :ocv:class:`DMatch`. If ``compactResult`` is true, the ``matches`` vector does not contain matches for fully masked-out query descriptors.

