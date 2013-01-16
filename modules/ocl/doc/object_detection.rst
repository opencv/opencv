Object Detection
=============================

.. highlight:: cpp

ocl::OclCascadeClassifier
-------------------------
.. ocv:class:: ocl::OclCascadeClassifier : public CascadeClassifier

Cascade classifier class used for object detection. Supports HAAR cascade classifier  in the form of cross link ::

    class CV_EXPORTS OclCascadeClassifier : public CascadeClassifier
    {
    public:
          OclCascadeClassifier() {};
          ~OclCascadeClassifier() {};
           CvSeq *oclHaarDetectObjects(oclMat &gimg, CvMemStorage *storage,
                                      double scaleFactor,int minNeighbors,
                                      int flags, CvSize minSize = cvSize(0, 0),
                                      CvSize maxSize = cvSize(0, 0));
    };

ocl::OclCascadeClassifier::oclHaarDetectObjects
------------------------------------------------------
Returns the detected objects by a list of rectangles

.. ocv:function:: CvSeq* ocl::OclCascadeClassifier::oclHaarDetectObjects(oclMat &gimg, CvMemStorage *storage, double scaleFactor,int minNeighbors, int flags, CvSize minSize = cvSize(0, 0), CvSize maxSize = cvSize(0, 0))

    :param image:  Matrix of type CV_8U containing an image where objects should be detected.

    :param imageobjectsBuff: Buffer to store detected objects (rectangles). If it is empty, it is allocated with the defaultsize. If not empty, the function searches not more than N  objects, where N = sizeof(objectsBufers data)/sizeof(cv::Rect).

    :param scaleFactor: Parameter specifying how much the image size is reduced at each image scale.

    :param minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.

    :param minSize: Minimum possible object size. Objects smaller than that are ignored.

Detects objects of different sizes in the input image,only tested for face detection now. The function returns the number of detected objects.

ocl::MatchTemplateBuf
---------------------
.. ocv:struct:: ocl::MatchTemplateBuf

Class providing memory buffers for :ocv:func:`ocl::matchTemplate` function, plus it allows to adjust some specific parameters. ::

    struct CV_EXPORTS MatchTemplateBuf
    {
        Size user_block_size;
        oclMat imagef, templf;
        std::vector<oclMat> images;
        std::vector<oclMat> image_sums;
        std::vector<oclMat> image_sqsums;
    };

You can use field `user_block_size` to set specific block size for :ocv:func:`ocl::matchTemplate` function. If you leave its default value `Size(0,0)` then automatic estimation of block size will be used (which is optimized for speed). By varying `user_block_size` you can reduce memory requirements at the cost of speed.

ocl::matchTemplate
------------------
Computes a proximity map for a raster template and an image where the template is searched for.

.. ocv:function:: void ocl::matchTemplate(const oclMat& image, const oclMat& templ, oclMat& result, int method)

.. ocv:function:: void ocl::matchTemplate(const oclMat& image, const oclMat& templ, oclMat& result, int method, MatchTemplateBuf &buf)

    :param image: Source image.  ``CV_32F`` and  ``CV_8U`` depth images (1..4 channels) are supported for now.

    :param templ: Template image with the size and type the same as  ``image`` .

    :param result: Map containing comparison results ( ``CV_32FC1`` ). If  ``image`` is  *W x H*  and ``templ`` is  *w x h*, then  ``result`` must be *W-w+1 x H-h+1*.

    :param method: Specifies the way to compare the template with the image.

    :param buf: Optional buffer to avoid extra memory allocations and to adjust some specific parameters. See :ocv:struct:`ocl::MatchTemplateBuf`.

    The following methods are supported for the ``CV_8U`` depth images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_SQDIFF_NORMED``
    * ``CV_TM_CCORR``
    * ``CV_TM_CCORR_NORMED``
    * ``CV_TM_CCOEFF``
    * ``CV_TM_CCOEFF_NORMED``

    The following methods are supported for the ``CV_32F`` images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_CCORR``

.. seealso:: :ocv:func:`matchTemplate`


ocl::SURF_OCL
-------------
.. ocv:class:: ocl::SURF_OCL

Class used for extracting Speeded Up Robust Features (SURF) from an image. ::

    class SURF_OCL
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
        SURF_OCL();
        //! the full constructor taking all the necessary parameters
        explicit SURF_OCL(double _hessianThreshold, int _nOctaves=4,
             int _nOctaveLayers=2, bool _extended=false, float _keypointsRatio=0.01f, bool _upright = false);

        //! returns the descriptor size in float's (64 or 128)
        int descriptorSize() const;

        //! upload host keypoints to device memory
        void uploadKeypoints(const vector<KeyPoint>& keypoints,
            oclMat& keypointsocl);
        //! download keypoints from device to host memory
        void downloadKeypoints(const oclMat& keypointsocl,
            vector<KeyPoint>& keypoints);

        //! download descriptors from device to host memory
        void downloadDescriptors(const oclMat& descriptorsocl,
            vector<float>& descriptors);

        void operator()(const oclMat& img, const oclMat& mask,
            oclMat& keypoints);

        void operator()(const oclMat& img, const oclMat& mask,
            oclMat& keypoints, oclMat& descriptors,
            bool useProvidedKeypoints = false);

        void operator()(const oclMat& img, const oclMat& mask,
            std::vector<KeyPoint>& keypoints);

        void operator()(const oclMat& img, const oclMat& mask,
            std::vector<KeyPoint>& keypoints, oclMat& descriptors,
            bool useProvidedKeypoints = false);

        void operator()(const oclMat& img, const oclMat& mask,
            std::vector<KeyPoint>& keypoints,
            std::vector<float>& descriptors,
            bool useProvidedKeypoints = false);

        void releaseMemory();

        // SURF parameters
        double hessianThreshold;
        int nOctaves;
        int nOctaveLayers;
        bool extended;
        bool upright;

        //! max keypoints = min(keypointsRatio * img.size().area(), 65535)
        float keypointsRatio;

        oclMat sum, mask1, maskSum, intBuffer;

        oclMat det, trace;

        oclMat maxPosBuffer;
    };


The class ``SURF_OCL`` implements Speeded Up Robust Features descriptor. There is a fast multi-scale Hessian keypoint detector that can be used to find the keypoints (which is the default option). But the descriptors can also be computed for the user-specified keypoints. Only 8-bit grayscale images are supported.

The class ``SURF_OCL`` can store results in the GPU and CPU memory. It provides functions to convert results between CPU and GPU version ( ``uploadKeypoints``, ``downloadKeypoints``, ``downloadDescriptors`` ). The format of CPU results is the same as ``SURF`` results. GPU results are stored in ``oclMat``. The ``keypoints`` matrix is :math:`\texttt{nFeatures} \times 7` matrix with the ``CV_32FC1`` type.

* ``keypoints.ptr<float>(X_ROW)[i]`` contains x coordinate of the i-th feature.
* ``keypoints.ptr<float>(Y_ROW)[i]`` contains y coordinate of the i-th feature.
* ``keypoints.ptr<float>(LAPLACIAN_ROW)[i]``  contains the laplacian sign of the i-th feature.
* ``keypoints.ptr<float>(OCTAVE_ROW)[i]`` contains the octave of the i-th feature.
* ``keypoints.ptr<float>(SIZE_ROW)[i]`` contains the size of the i-th feature.
* ``keypoints.ptr<float>(ANGLE_ROW)[i]`` contain orientation of the i-th feature.
* ``keypoints.ptr<float>(HESSIAN_ROW)[i]`` contains the response of the i-th feature.

The ``descriptors`` matrix is :math:`\texttt{nFeatures} \times \texttt{descriptorSize}` matrix with the ``CV_32FC1`` type.

The class ``SURF_OCL`` uses some buffers and provides access to it. All buffers can be safely released between function calls.

.. seealso:: :ocv:class:`SURF`