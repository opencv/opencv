Object Detection
================

.. highlight:: cpp



gpu::HOGDescriptor
------------------
.. ocv:struct:: gpu::HOGDescriptor

The class implements Histogram of Oriented Gradients ([Dalal2005]_) object detector. ::

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



gpu::HOGDescriptor::HOGDescriptor
-------------------------------------
Creates the ``HOG`` descriptor and detector.

.. ocv:function:: gpu::HOGDescriptor::HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)

   :param win_size: Detection window size. Align to block size and block stride.

   :param block_size: Block size in pixels. Align to cell size. Only (16,16) is supported for now.

   :param block_stride: Block stride. It must be a multiple of cell size.

   :param cell_size: Cell size. Only (8, 8) is supported for now.

   :param nbins: Number of bins. Only 9 bins per cell are supported for now.

   :param win_sigma: Gaussian smoothing window parameter.

   :param threshold_L2hys: L2-Hys normalization method shrinkage.

   :param gamma_correction: Flag to specify whether the gamma correction preprocessing is required or not.

   :param nlevels: Maximum number of detection window increases.



gpu::HOGDescriptor::getDescriptorSize
-----------------------------------------
Returns the number of coefficients required for the classification.

.. ocv:function:: size_t gpu::HOGDescriptor::getDescriptorSize() const



gpu::HOGDescriptor::getBlockHistogramSize
---------------------------------------------
Returns the block histogram size.

.. ocv:function:: size_t gpu::HOGDescriptor::getBlockHistogramSize() const



gpu::HOGDescriptor::setSVMDetector
--------------------------------------
Sets coefficients for the linear SVM classifier.

.. ocv:function:: void gpu::HOGDescriptor::setSVMDetector(const vector<float>& detector)



gpu::HOGDescriptor::getDefaultPeopleDetector
------------------------------------------------
Returns coefficients of the classifier trained for people detection (for default window size).

.. ocv:function:: static vector<float> gpu::HOGDescriptor::getDefaultPeopleDetector()



gpu::HOGDescriptor::getPeopleDetector48x96
----------------------------------------------
Returns coefficients of the classifier trained for people detection (for 48x96 windows).

.. ocv:function:: static vector<float> gpu::HOGDescriptor::getPeopleDetector48x96()



gpu::HOGDescriptor::getPeopleDetector64x128
-----------------------------------------------
Returns coefficients of the classifier trained for people detection (for 64x128 windows).

.. ocv:function:: static vector<float> gpu::HOGDescriptor::getPeopleDetector64x128()



gpu::HOGDescriptor::detect
------------------------------
Performs object detection without a multi-scale window.

.. ocv:function:: void gpu::HOGDescriptor::detect(const GpuMat& img, vector<Point>& found_locations, double hit_threshold=0, Size win_stride=Size(), Size padding=Size())

   :param img: Source image.  ``CV_8UC1``  and  ``CV_8UC4`` types are supported for now.

   :param found_locations: Left-top corner points of detected objects boundaries.

   :param hit_threshold: Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specfied in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param padding: Mock parameter to keep the CPU interface compatibility. It must be (0,0).



gpu::HOGDescriptor::detectMultiScale
----------------------------------------
Performs object detection with a multi-scale window.

.. ocv:function:: void gpu::HOGDescriptor::detectMultiScale(const GpuMat& img, vector<Rect>& found_locations, double hit_threshold=0, Size win_stride=Size(), Size padding=Size(), double scale0=1.05, int group_threshold=2)

   :param img: Source image. See  :ocv:func:`gpu::HOGDescriptor::detect`  for type limitations.

   :param found_locations: Detected objects boundaries.

   :param hit_threshold: Threshold for the distance between features and SVM classifying plane. See  :ocv:func:`gpu::HOGDescriptor::detect`  for details.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param padding: Mock parameter to keep the CPU interface compatibility. It must be (0,0).

   :param scale0: Coefficient of the detection window increase.

   :param group_threshold: Coefficient to regulate the similarity threshold. When detected, some objects can be covered by many rectangles. 0 means not to perform grouping. See  :ocv:func:`groupRectangles` .



gpu::HOGDescriptor::getDescriptors
--------------------------------------
Returns block descriptors computed for the whole image.

.. ocv:function:: void gpu::HOGDescriptor::getDescriptors(const GpuMat& img, Size win_stride, GpuMat& descriptors, int descr_format=DESCR_FORMAT_COL_BY_COL)

   :param img: Source image. See  :ocv:func:`gpu::HOGDescriptor::detect`  for type limitations.

   :param win_stride: Window stride. It must be a multiple of block stride.

   :param descriptors: 2D array of descriptors.

   :param descr_format: Descriptor storage format:

        * **DESCR_FORMAT_ROW_BY_ROW** - Row-major order.

        * **DESCR_FORMAT_COL_BY_COL** - Column-major order.

The function is mainly used to learn the classifier.


Soft Cascade Classifier
======================

Soft Cascade Classifier for Object Detection
----------------------------------------------------------

Cascade detectors have been shown to operate extremely rapidly, with high accuracy, and have important applications in different spheres. The initial goal for this cascade implementation was the fast and accurate pedestrian detector but it also useful in general. Soft cascade is trained with AdaBoost. But instead of training sequence of stages, the soft cascade is trained as a one long stage of T weak classifiers. Soft cascade is formulated as follows:

.. math::
    \texttt{H}(x) = \sum _{\texttt{t}=1..\texttt{T}} {\texttt{s}_t(x)}

where :math:`\texttt{s}_t(x) = \alpha_t\texttt{h}_t(x)` are the set of thresholded weak classifiers selected during AdaBoost training scaled by the associated weights. Let

.. math::
    \texttt{H}_t(x) = \sum _{\texttt{i}=1..\texttt{t}} {\texttt{s}_i(x)}

be the partial sum of sample responses before :math:`t`-the weak classifier will be applied. The funtcion :math:`\texttt{H}_t(x)` of :math:`t` for sample :math:`x` named *sample trace*.
After each weak classifier evaluation, the sample trace at the point :math:`t` is compared with the rejection threshold :math:`r_t`. The sequence of :math:`r_t` named *rejection trace*.

The sample has been rejected if it fall rejection threshold. So stageless cascade allows to reject not-object sample as soon as possible. Another meaning of the sample trace is a confidence with that sample recognized as desired object. At each :math:`t` that confidence depend on all previous weak classifier. This feature of soft cascade is resulted in more accurate detection. The original formulation of soft cascade can be found in [BJ05]_.

.. [BJ05] Lubomir Bourdev and Jonathan Brandt. tRobust Object Detection Via Soft Cascade. IEEE CVPR, 2005.
.. [BMTG12] Rodrigo Benenson, Markus Mathias, Radu Timofte and Luc Van Gool. Pedestrian detection at 100 frames per second. IEEE CVPR, 2012.


SCascade
----------------
.. ocv:class:: SCascade

Implementation of soft (stageless) cascaded detector. ::

    class CV_EXPORTS SCascade : public Algorithm
    {
        struct CV_EXPORTS Detection
        {
              ushort x;
              ushort y;
              ushort w;
              ushort h;
              float confidence;
              int kind;

              enum {PEDESTRIAN = 0};
        };

        SCascade(const double minScale = 0.4, const double maxScale = 5., const int scales = 55, const int rejfactor = 1);
        virtual ~SCascade();
        virtual bool load(const FileNode& fn);
        virtual void detect(InputArray image, InputArray rois, OutputArray objects, Stream& stream = Stream::Null()) const;
        void genRoi(InputArray roi, OutputArray mask, Stream& stream = Stream::Null()) const;
    };


SCascade::SCascade
--------------------------
An empty cascade will be created.

.. ocv:function:: bool SCascade::SCascade(const float minScale = 0.4f, const float maxScale = 5.f, const int scales = 55, const int rejfactor = 1)

    :param minScale: a minimum scale relative to the original size of the image on which cascade will be applyed.

    :param maxScale: a maximum scale relative to the original size of the image on which cascade will be applyed.

    :param scales: a number of scales from minScale to maxScale.

    :param rejfactor: used for non maximum suppression.



SCascade::~SCascade
---------------------------
Destructor for SCascade.

.. ocv:function:: SCascade::~SCascade()



SCascade::load
--------------------------
Load cascade from FileNode.

.. ocv:function:: bool SCascade::load(const FileNode& fn)

    :param fn: File node from which the soft cascade are read.



SCascade::detect
--------------------------
Apply cascade to an input frame and return the vector of Decection objcts.

.. ocv:function:: void detect(InputArray image, InputArray rois, OutputArray objects, Stream& stream = Stream::Null()) const

    :param image: a frame on which detector will be applied.

    :param rois: a regions of interests mask generated by genRoi. Only the objects that fall into one of the regions will be returned.

    :param objects: an output array of Detections represented as GpuMat of detections (SCascade::Detection). The first element of the matrix is  actually a count of detections.

    :param stream: a high-level CUDA stream abstraction used for asynchronous execution.


SCascade::genRoi
--------------------------
Convert ROI matrix into the suitable for detect method.

.. ocv:function:: void genRoi(InputArray roi, OutputArray mask, Stream& stream = Stream::Null()) const

    :param rois: an input matrix of the same size as the image. There non zero value mean that detector should be executed in this point.

    :param mask: an output mask

    :param stream: a high-level CUDA stream abstraction used for asynchronous execution.



gpu::CascadeClassifier_GPU
--------------------------
.. ocv:class:: gpu::CascadeClassifier_GPU

Cascade classifier class used for object detection. Supports HAAR and LBP cascades. ::

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
            int detectMultiScale( const GpuMat& image, GpuMat& objectsBuf, Size maxObjectSize, Size minSize = Size(), double scaleFactor = 1.1, int minNeighbors = 4);

            /* Finds only the largest object. Special mode if training is required.*/
            bool findLargestObject;

            /* Draws rectangles in input image */
            bool visualizeInPlace;

            Size getClassifierSize() const;
    };



gpu::CascadeClassifier_GPU::CascadeClassifier_GPU
-----------------------------------------------------
Loads the classifier from a file. Cascade type is detected automatically by constructor parameter.

.. ocv:function:: gpu::CascadeClassifier_GPU::CascadeClassifier_GPU(const string& filename)

    :param filename: Name of the file from which the classifier is loaded. Only the old ``haar`` classifier (trained by the ``haar`` training application) and NVIDIA's ``nvbin`` are supported for HAAR and only new type of OpenCV XML cascade supported for LBP.



gpu::CascadeClassifier_GPU::empty
-------------------------------------
Checks whether the classifier is loaded or not.

.. ocv:function:: bool gpu::CascadeClassifier_GPU::empty() const



gpu::CascadeClassifier_GPU::load
------------------------------------
Loads the classifier from a file. The previous content is destroyed.

.. ocv:function:: bool gpu::CascadeClassifier_GPU::load(const string& filename)

    :param filename: Name of the file from which the classifier is loaded. Only the old ``haar`` classifier (trained by the ``haar`` training application) and NVIDIA's ``nvbin`` are supported for HAAR and only new type of OpenCV XML cascade supported for LBP.


gpu::CascadeClassifier_GPU::release
---------------------------------------
Destroys the loaded classifier.

.. ocv:function:: void gpu::CascadeClassifier_GPU::release()



gpu::CascadeClassifier_GPU::detectMultiScale
------------------------------------------------
Detects objects of different sizes in the input image.

.. ocv:function:: int gpu::CascadeClassifier_GPU::detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, double scaleFactor=1.2, int minNeighbors=4, Size minSize=Size())

.. ocv:function:: int gpu::CascadeClassifier_GPU::detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, Size maxObjectSize, Size minSize = Size(), double scaleFactor = 1.1, int minNeighbors = 4)

    :param image: Matrix of type  ``CV_8U``  containing an image where objects should be detected.

    :param objectsBuf: Buffer to store detected objects (rectangles). If it is empty, it is allocated with the default size. If not empty, the function searches not more than N objects, where ``N = sizeof(objectsBufer's data)/sizeof(cv::Rect)``.

    :param maxObjectSize: Maximum possible object size. Objects larger than that are ignored. Used for second signature and supported only for LBP cascades.

    :param scaleFactor:  Parameter specifying how much the image size is reduced at each image scale.

    :param minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.

    :param minSize: Minimum possible object size. Objects smaller than that are ignored.

The detected objects are returned as a list of rectangles.

The function returns the number of detected objects, so you can retrieve them as in the following example: ::

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
