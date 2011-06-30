Object Detection
================

.. highlight:: cpp

.. index:: gpu::HOGDescriptor

gpu::HOGDescriptor
------------------
.. ocv:class:: gpu::HOGDescriptor

The class implements Histogram of Oriented Gradients ([Dalal2005]_) object detector.
::

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


Interfaces of all methods are kept similar to the ``CPU HOG`` descriptor and detector analogues as much as possible.

.. index:: gpu::HOGDescriptor::HOGDescriptor

gpu::HOGDescriptor::HOGDescriptor
-------------------------------------
.. ocv:function:: gpu::HOGDescriptor::HOGDescriptor(Size win_size=Size(64, 128),
   Size block_size=Size(16, 16), Size block_stride=Size(8, 8),
   Size cell_size=Size(8, 8), int nbins=9,
   double win_sigma=DEFAULT_WIN_SIGMA,
   double threshold_L2hys=0.2, bool gamma_correction=true,
   int nlevels=DEFAULT_NLEVELS)??check the output??

    Creates the ``HOG`` descriptor and detector.

   :param win_size: Detection window size. Align to block size and block stride.

   :param block_size: Block size in pixels. Align to cell size. Only (16,16) is supported for now.

   :param block_stride: Block stride. It must be a multiple of cell size.

   :param cell_size: Cell size. Only (8, 8) is supported for now.

   :param nbins: Number of bins. Only 9 bins per cell are supported for now.

   :param win_sigma: Gaussian smoothing window parameter.

   :param threshold_L2Hys: L2-Hys normalization method shrinkage.

   :param gamma_correction: Flag to specify whether the gamma correction preprocessing is required or not.

   :param nlevels: Maximum number of detection window increases.

.. index:: gpu::HOGDescriptor::getDescriptorSize

gpu::HOGDescriptor::getDescriptorSize
-----------------------------------------
.. ocv:function:: size_t gpu::HOGDescriptor::getDescriptorSize() const

    Returns the number of coefficients required for the classification.

.. index:: gpu::HOGDescriptor::getBlockHistogramSize

gpu::HOGDescriptor::getBlockHistogramSize
---------------------------------------------
.. ocv:function:: size_t gpu::HOGDescriptor::getBlockHistogramSize() const

    Returns the block histogram size.

.. index:: gpu::HOGDescriptor::setSVMDetector

gpu::HOGDescriptor::setSVMDetector
--------------------------------------
.. ocv:function:: void gpu::HOGDescriptor::setSVMDetector(const vector<float>\& detector)

    Sets coefficients for the linear SVM classifier.

.. index:: gpu::HOGDescriptor::getDefaultPeopleDetector

gpu::HOGDescriptor::getDefaultPeopleDetector
------------------------------------------------
.. ocv:function:: static vector<float> gpu::HOGDescriptor::getDefaultPeopleDetector()

    Returns coefficients of the classifier trained for people detection (for default window size).

.. index:: gpu::HOGDescriptor::getPeopleDetector48x96

gpu::HOGDescriptor::getPeopleDetector48x96
----------------------------------------------
.. ocv:function:: static vector<float> gpu::HOGDescriptor::getPeopleDetector48x96()

    Returns coefficients of the classifier trained for people detection (for 48x96 windows).

.. index:: gpu::HOGDescriptor::getPeopleDetector64x128

gpu::HOGDescriptor::getPeopleDetector64x128
-----------------------------------------------
.. ocv:function:: static vector<float> gpu::HOGDescriptor::getPeopleDetector64x128()

    Returns coefficients of the classifier trained for people detection (for 64x128 windows).

.. index:: gpu::HOGDescriptor::detect

gpu::HOGDescriptor::detect
------------------------------
.. ocv:function:: void gpu::HOGDescriptor::detect(const GpuMat\& img,
   vector<Point>\& found_locations, double hit_threshold=0,
   Size win_stride=Size(), Size padding=Size())??see output??

    Performs object detection without a multi-scale window.

   :param img: Source image.  ``CV_8UC1``  and  ``CV_8UC4`` types are supported for now.

   :param found_locations: Left-top corner points of detected objects boundaries.

   :param hit_threshold: Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specfied in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param padding: Mock parameter to keep the CPU interface compatibility. It must be (0,0).

.. index:: gpu::HOGDescriptor::detectMultiScale

gpu::HOGDescriptor::detectMultiScale
----------------------------------------
.. ocv:function:: void gpu::HOGDescriptor::detectMultiScale(const GpuMat\& img,
   vector<Rect>\& found_locations, double hit_threshold=0,
   Size win_stride=Size(), Size padding=Size(),
   double scale0=1.05, int group_threshold=2)??the same??

    Performs object detection with a multi-scale window.

   :param img: Source image. See  :ocv:func:`gpu::HOGDescriptor::detect`  for type limitations.

   :param found_locations: Detected objects boundaries.

   :param hit_threshold: Threshold for the distance between features and SVM classifying plane. See  :ocv:func:`gpu::HOGDescriptor::detect`  for details.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param padding: Mock parameter to keep the CPU interface compatibility. It must be (0,0).

   :param scale0: Coefficient of the detection window increase.

   :param group_threshold: Coefficient to regulate the similarity threshold. When detected, some objects can be covered by many rectangles. 0 means not to perform grouping. See  :ocv:func:`groupRectangles` .

.. index:: gpu::HOGDescriptor::getDescriptors

gpu::HOGDescriptor::getDescriptors
--------------------------------------
.. ocv:function:: void gpu::HOGDescriptor::getDescriptors(const GpuMat\& img,
   Size win_stride, GpuMat\& descriptors,
   int descr_format=DESCR_FORMAT_COL_BY_COL)?? the same??

    Returns block descriptors computed for the whole image. The function is mainly used to learn the classifier.

   :param img: Source image. See  :ocv:func:`gpu::HOGDescriptor::detect`  for type limitations.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param descriptors: 2D array of descriptors.

   :param descr_format: Descriptor storage format: 

        * **DESCR_FORMAT_ROW_BY_ROW** - Row-major order.

        * **DESCR_FORMAT_COL_BY_COL** - Column-major order.
            

.. index:: gpu::CascadeClassifier_GPU

gpu::CascadeClassifier_GPU
--------------------------
.. ocv:class:: gpu::CascadeClassifier_GPU

Cascade classifier class used for object detection. 
::

    class CV_EXPORTS CascadeClassifier_GPU
    {
    public:
            CascadeClassifier_GPU();
            CascadeClassifier_GPU(const string& filename);
            ~CascadeClassifier_GPU();

            bool empty() const;
            bool load(const string& filename);
            void release();

            /* Returns number of detected objects */
            int detectMultiScale( const GpuMat& image, GpuMat& objectsBuf, double scaleFactor=1.2, int minNeighbors=4, Size minSize=Size());

            /* Finds only the largest object. Special mode if training is required.*/
            bool findLargestObject;

            /* Draws rectangles in input image */
            bool visualizeInPlace;

            Size getClassifierSize() const;
    };


.. index:: gpu::CascadeClassifier_GPU::CascadeClassifier_GPU

gpu::CascadeClassifier_GPU::CascadeClassifier_GPU
-----------------------------------------------------
.. ocv:function:: gpu::CascadeClassifier_GPU(const string\& filename)

    Loads the classifier from a file.

    :param filename: Name of the file from which the classifier is loaded. Only the old ``haar`` classifier (trained by the ``haar`` training application) and NVIDIA's ``nvbin`` are supported.

.. index:: gpu::CascadeClassifier_GPU::empty

.. _gpu::CascadeClassifier_GPU::empty:

gpu::CascadeClassifier_GPU::empty
-------------------------------------
.. ocv:function:: bool gpu::CascadeClassifier_GPU::empty() const

    Checks whether the classifier is loaded or not.

.. index:: gpu::CascadeClassifier_GPU::load

.. _gpu::CascadeClassifier_GPU::load:

gpu::CascadeClassifier_GPU::load
------------------------------------
.. ocv:function:: bool gpu::CascadeClassifier_GPU::load(const string\& filename)

    Loads the classifier from a file. The previous content is destroyed.

    :param filename: Name of the file from which the classifier is loaded. Only the old ``haar`` classifier (trained by the ``haar`` training application) and NVIDIA's ``nvbin`` are supported.

.. index:: gpu::CascadeClassifier_GPU::release

gpu::CascadeClassifier_GPU::release
---------------------------------------
.. ocv:function:: void gpu::CascadeClassifier_GPU::release()

    Destroys the loaded classifier.

.. index:: gpu::CascadeClassifier_GPU::detectMultiScale

gpu::CascadeClassifier_GPU::detectMultiScale
------------------------------------------------
.. ocv:function:: int gpu::CascadeClassifier_GPU::detectMultiScale(const GpuMat\& image, GpuMat\& objectsBuf, double scaleFactor=1.2, int minNeighbors=4, Size minSize=Size())

    Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

    :param image: Matrix of type  ``CV_8U``  containing an image where objects should be detected.

    :param objects: Buffer to store detected objects (rectangles). If it is empty, it is allocated with the default size. If not empty, the function searches not more than N objects, where ``N = sizeof(objectsBufer's data)/sizeof(cv::Rect)``.

    :param scaleFactor: Value to specify how much the image size is reduced at each image scale.

    :param minNeighbors: Value to specify how many neighbours each candidate rectangle has to retain.

    :param minSize: Minimum possible object size. Objects smaller than that are ignored.

    The function returns the number of detected objects, so you can retrieve them as in the following example: 

::

    gpu::CascadeClassifier_GPU cascade_gpu(...);

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


.. seealso:: :ocv:func:`CascadeClassifier::detectMultiScale` 

.. [Dalal2005] Navneet Dalal and Bill Triggs. *Histogram of oriented gradients for human detection*. 2005.
