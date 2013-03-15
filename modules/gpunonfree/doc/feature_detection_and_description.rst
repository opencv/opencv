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
