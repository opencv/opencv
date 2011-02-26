Object Detection
================

.. highlight:: cpp

.. index:: gpu::HOGDescriptor

.. _gpu::HOGDescriptor:

gpu::HOGDescriptor
------------------
.. ctype:: gpu::HOGDescriptor

Histogram of Oriented Gradients
dalal_hog
descriptor and detector. ::

    struct CV_EXPORTS HOGDescriptor
    {
        enum { DEFAULT_WIN_SIGMA = -1 };
        enum { DEFAULT_NLEVELS = 64 };
        enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };

        HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
                      Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
                      int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
                      double threshold_L2hys=0.2, bool gamma_correction=true,
                      int nlevels=DEFAULT_NLEVELS);

        size_t getDescriptorSize() const;
        size_t getBlockHistogramSize() const;

        void setSVMDetector(const vector<float>& detector);

        static vector<float> getDefaultPeopleDetector();
        static vector<float> getPeopleDetector48x96();
        static vector<float> getPeopleDetector64x128();

        void detect(const GpuMat& img, vector<Point>& found_locations,
                    double hit_threshold=0, Size win_stride=Size(),
                    Size padding=Size());

        void detectMultiScale(const GpuMat& img, vector<Rect>& found_locations,
                              double hit_threshold=0, Size win_stride=Size(),
                              Size padding=Size(), double scale0=1.05,
                              int group_threshold=2);

        void getDescriptors(const GpuMat& img, Size win_stride,
                            GpuMat& descriptors,
                            int descr_format=DESCR_FORMAT_COL_BY_COL);

        Size win_size;
        Size block_size;
        Size block_stride;
        Size cell_size;
        int nbins;
        double win_sigma;
        double threshold_L2hys;
        bool gamma_correction;
        int nlevels;

    private:
        // Hidden
    }
..

Interfaces of all methods are kept similar to CPU HOG descriptor and detector analogues as much as possible.

.. index:: gpu::HOGDescriptor::HOGDescriptor

cv::gpu::HOGDescriptor::HOGDescriptor
-------------------------------------
.. cfunction:: HOGDescriptor::HOGDescriptor(Size win_size=Size(64, 128),   Size block_size=Size(16, 16), Size block_stride=Size(8, 8),   Size cell_size=Size(8, 8), int nbins=9,   double win_sigma=DEFAULT_WIN_SIGMA,   double threshold_L2hys=0.2, bool gamma_correction=true,   int nlevels=DEFAULT_NLEVELS)

    Creates HOG descriptor and detector.

    :param win_size: Detection window size. Must be aligned to block size and block stride.

    :param block_size: Block size in pixels. Must be aligned to cell size. Only (16,16) is supported for now.

    :param block_stride: Block stride. Must be a multiple of cell size.

    :param cell_size: Cell size. Only (8, 8) is supported for now.

    :param nbins: Number of bins. Only 9 bins per cell is supported for now.

    :param win_sigma: Gaussian smoothing window parameter.

    :param threshold_L2Hys: L2-Hys normalization method shrinkage.

    :param gamma_correction: Do gamma correction preprocessing or not.

    :param nlevels: Maximum number of detection window increases.

.. index:: gpu::HOGDescriptor::getDescriptorSize

cv::gpu::HOGDescriptor::getDescriptorSize
-----------------------------------------
.. cfunction:: size_t HOGDescriptor::getDescriptorSize() const

    Returns number of coefficients required for the classification.

.. index:: gpu::HOGDescriptor::getBlockHistogramSize

cv::gpu::HOGDescriptor::getBlockHistogramSize
---------------------------------------------
.. cfunction:: size_t HOGDescriptor::getBlockHistogramSize() const

    Returns block histogram size.

.. index:: gpu::HOGDescriptor::setSVMDetector

cv::gpu::HOGDescriptor::setSVMDetector
--------------------------------------
.. cfunction:: void HOGDescriptor::setSVMDetector(const vector<float>\& detector)

    Sets coefficients for the linear SVM classifier.

.. index:: gpu::HOGDescriptor::getDefaultPeopleDetector

cv::gpu::HOGDescriptor::getDefaultPeopleDetector
------------------------------------------------
.. cfunction:: static vector<float> HOGDescriptor::getDefaultPeopleDetector()

    Returns coefficients of the classifier trained for people detection (for default window size).

.. index:: gpu::HOGDescriptor::getPeopleDetector48x96

cv::gpu::HOGDescriptor::getPeopleDetector48x96
----------------------------------------------
.. cfunction:: static vector<float> HOGDescriptor::getPeopleDetector48x96()

    Returns coefficients of the classifier trained for people detection (for 48x96 windows).

.. index:: gpu::HOGDescriptor::getPeopleDetector64x128

cv::gpu::HOGDescriptor::getPeopleDetector64x128
-----------------------------------------------
.. cfunction:: static vector<float> HOGDescriptor::getPeopleDetector64x128()

    Returns coefficients of the classifier trained for people detection (for 64x128 windows).

.. index:: gpu::HOGDescriptor::detect

cv::gpu::HOGDescriptor::detect
------------------------------
.. cfunction:: void HOGDescriptor::detect(const GpuMat\& img,   vector<Point>\& found_locations, double hit_threshold=0,   Size win_stride=Size(), Size padding=Size())

    Perfroms object detection without multiscale window.

    :param img: Source image.  ``CV_8UC1``  and  ``CV_8UC4`` types are supported for now.

    :param found_locations: Will contain left-top corner points of detected objects boundaries.

    :param hit_threshold: Threshold for the distance between features and SVM classifying plane. Usually it's 0 and should be specfied in the detector coefficients (as the last free coefficient), but if the free coefficient is omitted (it's allowed) you can specify it manually here.

    :param win_stride: Window stride. Must be a multiple of block stride.

    :param padding: Mock parameter to keep CPU interface compatibility. Must be (0,0).

.. index:: gpu::HOGDescriptor::detectMultiScale

cv::gpu::HOGDescriptor::detectMultiScale
----------------------------------------
.. cfunction:: void HOGDescriptor::detectMultiScale(const GpuMat\& img,   vector<Rect>\& found_locations, double hit_threshold=0,   Size win_stride=Size(), Size padding=Size(),   double scale0=1.05, int group_threshold=2)

    Perfroms object detection with multiscale window.

    :param img: Source image. See  :func:`gpu::HOGDescriptor::detect`  for type limitations.

    :param found_locations: Will contain detected objects boundaries.

    :param hit_threshold: The threshold for the distance between features and SVM classifying plane. See  :func:`gpu::HOGDescriptor::detect`  for details.

    :param win_stride: Window stride. Must be a multiple of block stride.

    :param padding: Mock parameter to keep CPU interface compatibility. Must be (0,0).

    :param scale0: Coefficient of the detection window increase.

    :param group_threshold: After detection some objects could be covered by many rectangles. This coefficient regulates similarity threshold. 0 means don't perform grouping.
        See  :func:`groupRectangles` .

.. index:: gpu::HOGDescriptor::getDescriptors

cv::gpu::HOGDescriptor::getDescriptors
--------------------------------------
.. cfunction:: void HOGDescriptor::getDescriptors(const GpuMat\& img,   Size win_stride, GpuMat\& descriptors,   int descr_format=DESCR_FORMAT_COL_BY_COL)

    Returns block descriptors computed for the whole image. It's mainly used for classifier learning purposes.

    :param img: Source image. See  :func:`gpu::HOGDescriptor::detect`  for type limitations.

    :param win_stride: Window stride. Must be a multiple of block stride.

    :param descriptors: 2D array of descriptors.

    :param descr_format: Descriptor storage format: 

            * **DESCR_FORMAT_ROW_BY_ROW** Row-major order.

            * **DESCR_FORMAT_COL_BY_COL** Column-major order.
            

.. index:: gpu::CascadeClassifier_GPU

.. _gpu::CascadeClassifier_GPU:

gpu::CascadeClassifier_GPU
--------------------------
.. ctype:: gpu::CascadeClassifier_GPU

The cascade classifier class for object detection. ::

    class CV_EXPORTS CascadeClassifier_GPU
    {
    public:
            CascadeClassifier_GPU();
            CascadeClassifier_GPU(const string& filename);
            ~CascadeClassifier_GPU();

            bool empty() const;
            bool load(const string& filename);
            void release();

            /* returns number of detected objects */
            int detectMultiScale( const GpuMat& image, GpuMat& objectsBuf, double scaleFactor=1.2, int minNeighbors=4, Size minSize=Size());

            /* Finds only the largest object. Special mode for need to training*/
            bool findLargestObject;

            /* Draws rectangles in input image */
            bool visualizeInPlace;

            Size getClassifierSize() const;
    };
..

.. index:: cv::gpu::CascadeClassifier_GPU::CascadeClassifier_GPU

.. _cv::gpu::CascadeClassifier_GPU::CascadeClassifier_GPU:

cv::gpu::CascadeClassifier_GPU::CascadeClassifier_GPU
-----------------------------------------------------
.. cfunction:: cv::CascadeClassifier_GPU(const string\& filename)

    Loads the classifier from file.

    :param filename: Name of file from which classifier will be load. Only old haar classifier (trained by haartraining application) and NVidia's nvbin are supported.

.. index:: cv::gpu::CascadeClassifier_GPU::empty

.. _cv::gpu::CascadeClassifier_GPU::empty:

cv::gpu::CascadeClassifier_GPU::empty
-------------------------------------
.. cfunction:: bool CascadeClassifier_GPU::empty() const

    Checks if the classifier has been loaded or not.

.. index:: cv::gpu::CascadeClassifier_GPU::load

.. _cv::gpu::CascadeClassifier_GPU::load:

cv::gpu::CascadeClassifier_GPU::load
------------------------------------
.. cfunction:: bool CascadeClassifier_GPU::load(const string\& filename)

    Loads the classifier from file. The previous content is destroyed.

    :param filename: Name of file from which classifier will be load. Only old haar classifier (trained by haartraining application) and NVidia's nvbin are supported.

.. index:: cv::gpu::CascadeClassifier_GPU::release

.. _cv::gpu::CascadeClassifier_GPU::release:

cv::gpu::CascadeClassifier_GPU::release
---------------------------------------
.. cfunction:: void CascadeClassifier_GPU::release()

    Destroys loaded classifier.

.. index:: cv::gpu::CascadeClassifier_GPU::detectMultiScale

.. _cv::gpu::CascadeClassifier_GPU::detectMultiScale:

cv::gpu::CascadeClassifier_GPU::detectMultiScale
------------------------------------------------
.. cfunction:: int CascadeClassifier_GPU::detectMultiScale(const GpuMat\& image, GpuMat\& objectsBuf, double scaleFactor=1.2, int minNeighbors=4, Size minSize=Size())

    Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

    :param image: Matrix of type  ``CV_8U``  containing the image in which to detect objects.

    :param objects: Buffer to store detected objects (rectangles). If it is empty, it will be allocated with default size. If not empty, function will search not more than N objects, where N = sizeof(objectsBufer's data)/sizeof(cv::Rect).

    :param scaleFactor: Specifies how much the image size is reduced at each image scale.

    :param minNeighbors: Specifies how many neighbors should each candidate rectangle have to retain it.

    :param minSize: The minimum possible object size. Objects smaller than that are ignored.

The function returns number of detected objects, so you can retrieve them as in following example: ::

    cv::gpu::CascadeClassifier_GPU cascade_gpu(...);

    Mat image_cpu = imread(...)
    GpuMat image_gpu(image_cpu);

    GpuMat objbuf;
    int detections_number = cascade_gpu.detectMultiScale( image_gpu,
              objbuf, 1.2, minNeighbors);

    Mat obj_host;
    // download only detected number of rectangles
    objbuf.colRange(0, detections_number).download(obj_host);

    Rect* faces = obj_host.ptr<Rect>();
    for(int i = 0; i < detections_num; ++i)
       cv::rectangle(image_cpu, faces[i], Scalar(255));

    imshow("Faces", image_cpu);
..

See also:
:func:`CascadeClassifier::detectMultiScale` .

