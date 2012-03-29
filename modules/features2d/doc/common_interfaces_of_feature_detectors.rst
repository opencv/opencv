Common Interfaces of Feature Detectors
======================================

.. highlight:: cpp

Feature detectors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. All objects that implement keypoint detectors
inherit the
:ocv:class:`FeatureDetector` interface.

KeyPoint
--------
.. ocv:class:: KeyPoint

Data structure for salient point detectors.

    .. ocv:member:: Point2f pt

        coordinates of the keypoint

    .. ocv:member:: float size

         diameter of the meaningful keypoint neighborhood

    .. ocv:member:: float angle

        computed orientation of the keypoint (-1 if not applicable)

    .. ocv:member:: float response

        the response by which the most strong keypoints have been selected. Can be used for further sorting or subsampling

    .. ocv:member:: int octave

        octave (pyramid layer) from which the keypoint has been extracted

    .. ocv:member:: int class_id

        object id that can be used to clustered keypoints by an object they belong to

KeyPoint::KeyPoint
------------------
The keypoint constructors

.. ocv:function:: KeyPoint::KeyPoint()

.. ocv:function:: KeyPoint::KeyPoint(Point2f _pt, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)

.. ocv:function:: KeyPoint::KeyPoint(float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)

.. ocv:pyfunction:: cv2.KeyPoint(x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]) -> <KeyPoint object>

    :param x: x-coordinate of the keypoint

    :param y: y-coordinate of the keypoint

    :param _pt: x & y coordinates of the keypoint

    :param _size: keypoint diameter

    :param _angle: keypoint orientation

    :param _response: keypoint detector response on the keypoint (that is, strength of the keypoint)

    :param _octave: pyramid octave in which the keypoint has been detected

    :param _class_id: object id


FeatureDetector
---------------
.. ocv:class:: FeatureDetector

Abstract base class for 2D image feature detectors. ::

    class CV_EXPORTS FeatureDetector
    {
    public:
        virtual ~FeatureDetector();

        void detect( const Mat& image, vector<KeyPoint>& keypoints,
                     const Mat& mask=Mat() ) const;

        void detect( const vector<Mat>& images,
                     vector<vector<KeyPoint> >& keypoints,
                     const vector<Mat>& masks=vector<Mat>() ) const;

        virtual void read(const FileNode&);
        virtual void write(FileStorage&) const;

        static Ptr<FeatureDetector> create( const string& detectorType );

    protected:
    ...
    };

FeatureDetector::detect
---------------------------
Detects keypoints in an image (first variant) or image set (second variant).

.. ocv:function:: void FeatureDetector::detect( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const

.. ocv:function:: void FeatureDetector::detect( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, const vector<Mat>& masks=vector<Mat>() ) const

    :param image: Image.

    :param images: Image set.

    :param keypoints: The detected keypoints. In the second variant of the method ``keypoints[i]`` is a set of keypoints detected in ``images[i]`` .

    :param mask: Mask specifying where to look for keypoints (optional). It must be a 8-bit integer matrix with non-zero values in the region of interest.

    :param masks: Masks for each input image specifying where to look for keypoints (optional). ``masks[i]`` is a mask for ``images[i]``.

FeatureDetector::read
-------------------------
Reads a feature detector object from a file node.

.. ocv:function:: void FeatureDetector::read( const FileNode& fn )

    :param fn: File node from which the detector is read.

FeatureDetector::write
--------------------------
Writes a feature detector object to a file storage.

.. ocv:function:: void FeatureDetector::write( FileStorage& fs ) const

    :param fs: File storage where the detector is written.

FeatureDetector::create
---------------------------
Creates a feature detector by its name.

.. ocv:function:: Ptr<FeatureDetector> FeatureDetector::create( const string& detectorType )

    :param detectorType: Feature detector type.

The following detector types are supported:

* ``"FAST"`` -- :ocv:class:`FastFeatureDetector`
* ``"STAR"`` -- :ocv:class:`StarFeatureDetector`
* ``"SIFT"`` -- :ocv:class:`SiftFeatureDetector`
* ``"SURF"`` -- :ocv:class:`SurfFeatureDetector`
* ``"ORB"`` -- :ocv:class:`OrbFeatureDetector`
* ``"MSER"`` -- :ocv:class:`MserFeatureDetector`
* ``"GFTT"`` -- :ocv:class:`GoodFeaturesToTrackDetector`
* ``"HARRIS"`` -- :ocv:class:`GoodFeaturesToTrackDetector` with Harris detector enabled
* ``"Dense"`` -- :ocv:class:`DenseFeatureDetector`
* ``"SimpleBlob"`` -- :ocv:class:`SimpleBlobDetector`

Also a combined format is supported: feature detector adapter name ( ``"Grid"`` --
:ocv:class:`GridAdaptedFeatureDetector`, ``"Pyramid"`` --
:ocv:class:`PyramidAdaptedFeatureDetector` ) + feature detector name (see above),
for example: ``"GridFAST"``, ``"PyramidSTAR"`` .

FastFeatureDetector
-------------------
.. ocv:class:: FastFeatureDetector

Wrapping class for feature detection using the
:ocv:func:`FAST` method. ::

    class FastFeatureDetector : public FeatureDetector
    {
    public:
        FastFeatureDetector( int threshold=1, bool nonmaxSuppression=true );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

GoodFeaturesToTrackDetector
---------------------------
.. ocv:class:: GoodFeaturesToTrackDetector

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
.. ocv:class:: MserFeatureDetector

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


StarFeatureDetector
-------------------
.. ocv:class:: StarFeatureDetector

Wrapping class for feature detection using the
:ocv:class:`StarDetector` class. ::

    class StarFeatureDetector : public FeatureDetector
    {
    public:
        StarFeatureDetector( int maxSize=16, int responseThreshold=30,
                             int lineThresholdProjected = 10,
                             int lineThresholdBinarized=8, int suppressNonmaxSize=5 );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

SiftFeatureDetector
-------------------
.. ocv:class:: SiftFeatureDetector

Wrapping class for feature detection using the
:ocv:class:`SIFT` class. ::

    class SiftFeatureDetector : public FeatureDetector
    {
    public:
        SiftFeatureDetector(
            const SIFT::DetectorParams& detectorParams=SIFT::DetectorParams(),
            const SIFT::CommonParams& commonParams=SIFT::CommonParams() );
        SiftFeatureDetector( double threshold, double edgeThreshold,
                             int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES,
                             int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,
                             int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,
                             int angleMode=SIFT::CommonParams::FIRST_ANGLE );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

SurfFeatureDetector
-------------------
.. ocv:class:: SurfFeatureDetector

Wrapping class for feature detection using the
:ocv:class:`SURF` class. ::

    class SurfFeatureDetector : public FeatureDetector
    {
    public:
        SurfFeatureDetector( double hessianThreshold = 400., int octaves = 3,
                             int octaveLayers = 4 );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };


OrbFeatureDetector
-------------------
.. ocv:class:: OrbFeatureDetector

Wrapping class for feature detection using the
:ocv:class:`ORB` class. ::

    class OrbFeatureDetector : public FeatureDetector
    {
    public:
        OrbFeatureDetector( size_t n_features );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };


DenseFeatureDetector
--------------------
.. ocv:class:: DenseFeatureDetector

Class for generation of image features which are distributed densely and regularly over the image. ::

        class DenseFeatureDetector : public FeatureDetector
        {
        public:
                DenseFeatureDetector( float initFeatureScale=1.f, int featureScaleLevels=1,
                              float featureScaleMul=0.1f,
                              int initXyStep=6, int initImgBound=0,
                              bool varyXyStepWithScale=true,
                              bool varyImgBoundWithScale=false );
        protected:
        ...
    };

The detector generates several levels (in the amount of ``featureScaleLevels``) of features. Features of each level are located in the nodes of a regular grid over the image (excluding the image boundary of given size). The level parameters (a feature scale, a node size, a size of boundary) are multiplied by ``featureScaleMul`` with level index growing depending on input flags, viz.:

* Feature scale is multiplied always.

* The grid node size is multiplied if ``varyXyStepWithScale`` is ``true``.

* Size of image boundary is multiplied if ``varyImgBoundWithScale`` is ``true``.


SimpleBlobDetector
-------------------
.. ocv:class:: SimpleBlobDetector

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

GridAdaptedFeatureDetector
--------------------------
.. ocv:class:: GridAdaptedFeatureDetector

Class adapting a detector to partition the source image into a grid and detect points in each cell. ::

    class GridAdaptedFeatureDetector : public FeatureDetector
    {
    public:
        /*
         * detector            Detector that will be adapted.
         * maxTotalKeypoints   Maximum count of keypoints detected on the image.
         *                     Only the strongest keypoints will be kept.
         * gridRows            Grid row count.
         * gridCols            Grid column count.
         */
        GridAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector,
                                    int maxTotalKeypoints, int gridRows=4,
                                    int gridCols=4 );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };

PyramidAdaptedFeatureDetector
-----------------------------
.. ocv:class:: PyramidAdaptedFeatureDetector

Class adapting a detector to detect points over multiple levels of a Gaussian pyramid. Consider using this class for detectors that are not inherently scaled. ::

    class PyramidAdaptedFeatureDetector : public FeatureDetector
    {
    public:
        PyramidAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector,
                                       int levels=2 );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };


DynamicAdaptedFeatureDetector
-----------------------------
.. ocv:class:: DynamicAdaptedFeatureDetector

Adaptively adjusting detector that iteratively detects features until the desired number is found. ::

       class DynamicAdaptedFeatureDetector: public FeatureDetector
       {
       public:
           DynamicAdaptedFeatureDetector( const Ptr<AdjusterAdapter>& adjuster,
               int min_features=400, int max_features=500, int max_iters=5 );
           ...
       };

If the detector is persisted, it "remembers" the parameters
used for the last detection. In this case, the detector may be used for consistent numbers
of keypoints in a set of temporally related images, such as video streams or
panorama series.

``DynamicAdaptedFeatureDetector``  uses another detector, such as FAST or SURF, to do the dirty work,
with the help of ``AdjusterAdapter`` .
If the detected number of features is not large enough,
``AdjusterAdapter`` adjusts the detection parameters so that the next detection
results in a bigger or smaller number of features.  This is repeated until either the number of desired features are found
or the parameters are maxed out.

Adapters can be easily implemented for any detector via the
``AdjusterAdapter`` interface.

Beware that this is not thread-safe since the adjustment of parameters requires modification of the feature detector class instance.

Example of creating ``DynamicAdaptedFeatureDetector`` : ::

    //sample usage:
    //will create a detector that attempts to find
    //100 - 110 FAST Keypoints, and will at most run
    //FAST feature detection 10 times until that
    //number of keypoints are found
    Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector (100, 110, 10,
                                  new FastAdjuster(20,true)));


DynamicAdaptedFeatureDetector::DynamicAdaptedFeatureDetector
----------------------------------------------------------------
The constructor

.. ocv:function:: DynamicAdaptedFeatureDetector::DynamicAdaptedFeatureDetector(       const Ptr<AdjusterAdapter>& adjuster,       int min_features,   int max_features,   int max_iters )

    :param adjuster:  :ocv:class:`AdjusterAdapter`  that detects features and adjusts parameters.

    :param min_features: Minimum desired number of features.

    :param max_features: Maximum desired number of features.

    :param max_iters: Maximum number of times to try adjusting the feature detector parameters. For :ocv:class:`FastAdjuster` , this number can be high, but with ``Star`` or ``Surf``  many iterations can be time-comsuming.  At each iteration the detector is rerun.

AdjusterAdapter
---------------
.. ocv:class:: AdjusterAdapter

Class providing an interface for adjusting parameters of a feature detector. This interface is used by :ocv:class:`DynamicAdaptedFeatureDetector` . It is a wrapper for :ocv:class:`FeatureDetector` that enables adjusting parameters after feature detection. ::

     class AdjusterAdapter: public FeatureDetector
     {
     public:
        virtual ~AdjusterAdapter() {}
        virtual void tooFew(int min, int n_detected) = 0;
        virtual void tooMany(int max, int n_detected) = 0;
        virtual bool good() const = 0;
        virtual Ptr<AdjusterAdapter> clone() const = 0;
        static Ptr<AdjusterAdapter> create( const string& detectorType );
     };


See
:ocv:class:`FastAdjuster`,
:ocv:class:`StarAdjuster`, and
:ocv:class:`SurfAdjuster` for concrete implementations.

AdjusterAdapter::tooFew
---------------------------
Adjusts the detector parameters to detect more features.

.. ocv:function:: void AdjusterAdapter::tooFew(int min, int n_detected)

    :param min: Minimum desired number of features.

    :param n_detected: Number of features detected during the latest run.

Example: ::

    void FastAdjuster::tooFew(int min, int n_detected)
    {
            thresh_--;
    }

AdjusterAdapter::tooMany
----------------------------
Adjusts the detector parameters to detect less features.

.. ocv:function:: void AdjusterAdapter::tooMany(int max, int n_detected)

    :param max: Maximum desired number of features.

    :param n_detected: Number of features detected during the latest run.

Example: ::

    void FastAdjuster::tooMany(int min, int n_detected)
    {
            thresh_++;
    }


AdjusterAdapter::good
-------------------------
Returns false if the detector parameters cannot be adjusted any more.

.. ocv:function:: bool AdjusterAdapter::good() const

Example: ::

        bool FastAdjuster::good() const
        {
                return (thresh_ > 1) && (thresh_ < 200);
        }

AdjusterAdapter::create
-------------------------
Creates an adjuster adapter by name

.. ocv:function:: Ptr<AdjusterAdapter> AdjusterAdapter::create( const string& detectorType )

    Creates an adjuster adapter by name ``detectorType``. The detector name is the same as in :ocv:func:`FeatureDetector::create`, but now supports ``"FAST"``, ``"STAR"``, and ``"SURF"`` only.

FastAdjuster
------------
.. ocv:class:: FastAdjuster

:ocv:class:`AdjusterAdapter` for :ocv:class:`FastFeatureDetector`. This class decreases or increases the threshold value by 1. ::

        class FastAdjuster FastAdjuster: public AdjusterAdapter
        {
        public:
                FastAdjuster(int init_thresh = 20, bool nonmax = true);
                ...
        };

StarAdjuster
------------
.. ocv:class:: StarAdjuster

:ocv:class:`AdjusterAdapter` for :ocv:class:`StarFeatureDetector`. This class adjusts the ``responseThreshhold`` of ``StarFeatureDetector``.  ::

        class StarAdjuster: public AdjusterAdapter
        {
                StarAdjuster(double initial_thresh = 30.0);
                ...
        };

SurfAdjuster
------------
.. ocv:class:: SurfAdjuster

:ocv:class:`AdjusterAdapter` for :ocv:class:`SurfFeatureDetector`. This class adjusts the ``hessianThreshold`` of ``SurfFeatureDetector``. ::

        class SurfAdjuster: public SurfAdjuster
        {
                SurfAdjuster();
                ...
        };
