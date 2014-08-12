Common Interfaces of Feature Detectors
======================================

.. highlight:: cpp

Feature detectors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. All objects that implement keypoint detectors
inherit the
:ocv:class:`FeatureDetector` interface.

.. note::

   * An example explaining keypoint detection can be found at opencv_source_code/samples/cpp/descriptor_extractor_matcher.cpp

FeatureDetector
---------------
.. ocv:class:: FeatureDetector : public Algorithm

Abstract base class for 2D image feature detectors. ::

    class CV_EXPORTS FeatureDetector
    {
    public:
        virtual ~FeatureDetector();

        void detect( InputArray image, vector<KeyPoint>& keypoints,
                     InputArray mask=noArray() ) const;

        void detect( InputArrayOfArrays images,
                     vector<vector<KeyPoint> >& keypoints,
                     InputArrayOfArrays masks=noArray() ) const;

        virtual void read(const FileNode&);
        virtual void write(FileStorage&) const;

        static Ptr<FeatureDetector> create( const String& detectorType );

    protected:
    ...
    };

FeatureDetector::detect
---------------------------
Detects keypoints in an image (first variant) or image set (second variant).

.. ocv:function:: void FeatureDetector::detect( InputArray image, vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const

.. ocv:function:: void FeatureDetector::detect( InputArrayOfArrays images, vector<vector<KeyPoint> >& keypoints, InputArrayOfArrays masks=noArray() ) const

.. ocv:pyfunction:: cv2.FeatureDetector_create.detect(image[, mask]) -> keypoints

    :param image: Image.

    :param images: Image set.

    :param keypoints: The detected keypoints. In the second variant of the method ``keypoints[i]`` is a set of keypoints detected in ``images[i]`` .

    :param mask: Mask specifying where to look for keypoints (optional). It must be a 8-bit integer matrix with non-zero values in the region of interest.

    :param masks: Masks for each input image specifying where to look for keypoints (optional). ``masks[i]`` is a mask for ``images[i]``.

FeatureDetector::create
-----------------------
Creates a feature detector by its name.

.. ocv:function:: Ptr<FeatureDetector> FeatureDetector::create( const String& detectorType )

.. ocv:pyfunction:: cv2.FeatureDetector_create(detectorType) -> retval

    :param detectorType: Feature detector type.

The following detector types are supported:

* ``"FAST"`` -- :ocv:class:`FastFeatureDetector`
* ``"ORB"`` -- :ocv:class:`ORB`
* ``"BRISK"`` -- :ocv:class:`BRISK`
* ``"MSER"`` -- :ocv:class:`MSER`
* ``"GFTT"`` -- :ocv:class:`GoodFeaturesToTrackDetector`
* ``"HARRIS"`` -- :ocv:class:`GoodFeaturesToTrackDetector` with Harris detector enabled
* ``"SimpleBlob"`` -- :ocv:class:`SimpleBlobDetector`

FastFeatureDetector
-------------------
.. ocv:class:: FastFeatureDetector : public FeatureDetector

Wrapping class for feature detection using the
:ocv:func:`FAST` method. ::

    class FastFeatureDetector : public FeatureDetector
    {
    public:
        FastFeatureDetector( int threshold=1, bool nonmaxSuppression=true, type=FastFeatureDetector::TYPE_9_16 );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

GoodFeaturesToTrackDetector
---------------------------
.. ocv:class:: GoodFeaturesToTrackDetector : public FeatureDetector

Wrapping class for feature detection using the
:ocv:func:`goodFeaturesToTrack` function. ::

    class GoodFeaturesToTrackDetector : public FeatureDetector
    {
    public:
        class Params
        {
        public:
            Params( int maxCorners=1000, double qualityLevel=0.01,
                    double minDistance=1., int blockSize=3,
                    bool useHarrisDetector=false, double k=0.04 );
            void read( const FileNode& fn );
            void write( FileStorage& fs ) const;

            int maxCorners;
            double qualityLevel;
            double minDistance;
            int blockSize;
            bool useHarrisDetector;
            double k;
        };

        GoodFeaturesToTrackDetector( const GoodFeaturesToTrackDetector::Params& params=
                                                GoodFeaturesToTrackDetector::Params() );
        GoodFeaturesToTrackDetector( int maxCorners, double qualityLevel,
                                     double minDistance, int blockSize=3,
                                     bool useHarrisDetector=false, double k=0.04 );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

MserFeatureDetector
-------------------
.. ocv:class:: MserFeatureDetector : public FeatureDetector

Wrapping class for feature detection using the
:ocv:class:`MSER` class. ::

    class MserFeatureDetector : public FeatureDetector
    {
    public:
        MserFeatureDetector( CvMSERParams params=cvMSERParams() );
        MserFeatureDetector( int delta, int minArea, int maxArea,
                             double maxVariation, double minDiversity,
                             int maxEvolution, double areaThreshold,
                             double minMargin, int edgeBlurSize );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

SimpleBlobDetector
-------------------
.. ocv:class:: SimpleBlobDetector : public FeatureDetector

Class for extracting blobs from an image. ::

    class SimpleBlobDetector : public FeatureDetector
    {
    public:
    struct Params
    {
        Params();
        float thresholdStep;
        float minThreshold;
        float maxThreshold;
        size_t minRepeatability;
        float minDistBetweenBlobs;

        bool filterByColor;
        uchar blobColor;

        bool filterByArea;
        float minArea, maxArea;

        bool filterByCircularity;
        float minCircularity, maxCircularity;

        bool filterByInertia;
        float minInertiaRatio, maxInertiaRatio;

        bool filterByConvexity;
        float minConvexity, maxConvexity;
    };

    SimpleBlobDetector(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());

    protected:
        ...
    };

The class implements a simple algorithm for extracting blobs from an image:

#. Convert the source image to binary images by applying thresholding with several thresholds from ``minThreshold`` (inclusive) to ``maxThreshold`` (exclusive) with distance ``thresholdStep`` between neighboring thresholds.

#. Extract connected components from every binary image by  :ocv:func:`findContours`  and calculate their centers.

#. Group centers from several binary images by their coordinates. Close centers form one group that corresponds to one blob, which is controlled by the ``minDistBetweenBlobs`` parameter.

#. From the groups, estimate final centers of blobs and their radiuses and return as locations and sizes of keypoints.

This class performs several filtrations of returned blobs. You should set ``filterBy*`` to true/false to turn on/off corresponding filtration. Available filtrations:

 * **By color**. This filter compares the intensity of a binary image at the center of a blob to ``blobColor``. If they differ, the blob is filtered out. Use ``blobColor = 0`` to extract dark blobs and ``blobColor = 255`` to extract light blobs.

 * **By area**. Extracted blobs have an area between ``minArea`` (inclusive) and ``maxArea`` (exclusive).

 * **By circularity**. Extracted blobs have circularity (:math:`\frac{4*\pi*Area}{perimeter * perimeter}`) between ``minCircularity`` (inclusive) and ``maxCircularity`` (exclusive).

 * **By ratio of the minimum inertia to maximum inertia**. Extracted blobs have this ratio between ``minInertiaRatio`` (inclusive) and ``maxInertiaRatio`` (exclusive).

 * **By convexity**. Extracted blobs have convexity (area / area of blob convex hull) between ``minConvexity`` (inclusive) and ``maxConvexity`` (exclusive).


Default values of parameters are tuned to extract dark circular blobs.
