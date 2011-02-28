Common Interfaces of Feature Detectors
======================================

.. highlight:: cpp

Feature detectors in OpenCV have wrappers with common interface that enables to switch easily
between different algorithms solving the same problem. All objects that implement keypoint detectors
inherit
:func:`FeatureDetector` interface.

.. index:: KeyPoint

.. _KeyPoint:

KeyPoint
--------
.. c:type:: KeyPoint

Data structure for salient point detectors. ::

    class KeyPoint
    {
    public:
        // the default constructor
        KeyPoint() : pt(0,0), size(0), angle(-1), response(0), octave(0),
                     class_id(-1) {}
        // the full constructor
        KeyPoint(Point2f _pt, float _size, float _angle=-1,
                float _response=0, int _octave=0, int _class_id=-1)
                : pt(_pt), size(_size), angle(_angle), response(_response),
                  octave(_octave), class_id(_class_id) {}
        // another form of the full constructor
        KeyPoint(float x, float y, float _size, float _angle=-1,
                float _response=0, int _octave=0, int _class_id=-1)
                : pt(x, y), size(_size), angle(_angle), response(_response),
                  octave(_octave), class_id(_class_id) {}
        // converts vector of keypoints to vector of points
        static void convert(const std::vector<KeyPoint>& keypoints,
                            std::vector<Point2f>& points2f,
                            const std::vector<int>& keypointIndexes=std::vector<int>());
        // converts vector of points to the vector of keypoints, where each
        // keypoint is assigned the same size and the same orientation
        static void convert(const std::vector<Point2f>& points2f,
                            std::vector<KeyPoint>& keypoints,
                            float size=1, float response=1, int octave=0,
                            int class_id=-1);

        // computes overlap for pair of keypoints;
        // overlap is a ratio between area of keypoint regions intersection and
        // area of keypoint regions union (now keypoint region is circle)
        static float overlap(const KeyPoint& kp1, const KeyPoint& kp2);

        Point2f pt; // coordinates of the keypoints
        float size; // diameter of the meaningfull keypoint neighborhood
        float angle; // computed orientation of the keypoint (-1 if not applicable)
        float response; // the response by which the most strong keypoints
                        // have been selected. Can be used for the further sorting
                        // or subsampling
        int octave; // octave (pyramid layer) from which the keypoint has been extracted
        int class_id; // object class (if the keypoints need to be clustered by
                      // an object they belong to)
    };

    // writes vector of keypoints to the file storage
    void write(FileStorage& fs, const string& name, const vector<KeyPoint>& keypoints);
    // reads vector of keypoints from the specified file storage node
    void read(const FileNode& node, CV_OUT vector<KeyPoint>& keypoints);
..

.. index:: FeatureDetector

.. _FeatureDetector:

FeatureDetector
---------------
.. c:type:: FeatureDetector

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
..

.. index:: FeatureDetector::detect

FeatureDetector::detect
---------------------------
.. c:function:: void FeatureDetector::detect( const Mat\& image,                                vector<KeyPoint>\& keypoints,                                 const Mat\& mask=Mat() ) const

    Detect keypoints in an image (first variant) or image set (second variant).

    :param image: The image.

    :param keypoints: The detected keypoints.

    :param mask: Mask specifying where to look for keypoints (optional). Must be a char matrix
                             with non-zero values in the region of interest.

.. c:function:: void FeatureDetector::detect( const vector<Mat>\& images,                                                            vector<vector<KeyPoint> >\& keypoints,                                                             const vector<Mat>\& masks=vector<Mat>() ) const

    * **images** Images set.

    * **keypoints** Collection of keypoints detected in an input images. keypoints[i] is a set of keypoints detected in an images[i].

    * **masks** Masks for each input image specifying where to look for keypoints (optional). masks[i] is a mask for images[i].
                      Each element of  ``masks``  vector must be a char matrix with non-zero values in the region of interest.

.. index:: FeatureDetector::read

FeatureDetector::read
-------------------------
.. c:function:: void FeatureDetector::read( const FileNode\& fn )

    Read feature detector object from file node.

    :param fn: File node from which detector will be read.

.. index:: FeatureDetector::write

FeatureDetector::write
--------------------------
.. c:function:: void FeatureDetector::write( FileStorage\& fs ) const

    Write feature detector object to file storage.

    :param fs: File storage in which detector will be written.

.. index:: FeatureDetector::create

FeatureDetector::create
---------------------------
:func:`FeatureDetector`
.. c:function:: Ptr<FeatureDetector> FeatureDetector::create( const string\& detectorType )

    Feature detector factory that creates of given type with
default parameters (rather using default constructor).

    :param detectorType: Feature detector type.

Now the following detector types are supported:
\ ``"FAST"`` --
:func:`FastFeatureDetector`,\ ``"STAR"`` --
:func:`StarFeatureDetector`,\ ``"SIFT"`` --
:func:`SiftFeatureDetector`,\ ``"SURF"`` --
:func:`SurfFeatureDetector`,\ ``"MSER"`` --
:func:`MserFeatureDetector`,\ ``"GFTT"`` --
:func:`GfttFeatureDetector`,\ ``"HARRIS"`` --
:func:`HarrisFeatureDetector` .
\
Also combined format is supported: feature detector adapter name ( ``"Grid"`` --
:func:`GridAdaptedFeatureDetector`,``"Pyramid"`` --
:func:`PyramidAdaptedFeatureDetector` ) + feature detector name (see above),
e.g. ``"GridFAST"``,``"PyramidSTAR"`` , etc.

.. index:: FastFeatureDetector

.. _FastFeatureDetector:

FastFeatureDetector
-------------------
.. c:type:: FastFeatureDetector

Wrapping class for feature detection using
:func:`FAST` method. ::

    class FastFeatureDetector : public FeatureDetector
    {
    public:
        FastFeatureDetector( int threshold=1, bool nonmaxSuppression=true );
        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;
    protected:
        ...
    };
..

.. index:: GoodFeaturesToTrackDetector

.. _GoodFeaturesToTrackDetector:

GoodFeaturesToTrackDetector
---------------------------
.. c:type:: GoodFeaturesToTrackDetector

Wrapping class for feature detection using
:func:`goodFeaturesToTrack` function. ::

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
..

.. index:: MserFeatureDetector

.. _MserFeatureDetector:

MserFeatureDetector
-------------------
.. c:type:: MserFeatureDetector

Wrapping class for feature detection using
:func:`MSER` class. ::

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
..

.. index:: StarFeatureDetector

.. _StarFeatureDetector:

StarFeatureDetector
-------------------
.. c:type:: StarFeatureDetector

Wrapping class for feature detection using
:func:`StarDetector` class. ::

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
..

.. index:: SiftFeatureDetector

.. _SiftFeatureDetector:

SiftFeatureDetector
-------------------
.. c:type:: SiftFeatureDetector

Wrapping class for feature detection using
:func:`SIFT` class. ::

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
..

.. index:: SurfFeatureDetector

.. _SurfFeatureDetector:

SurfFeatureDetector
-------------------
.. c:type:: SurfFeatureDetector

Wrapping class for feature detection using
:func:`SURF` class. ::

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
..

.. index:: GridAdaptedFeatureDetector

.. _GridAdaptedFeatureDetector:

GridAdaptedFeatureDetector
--------------------------
.. c:type:: GridAdaptedFeatureDetector

Adapts a detector to partition the source image into a grid and detect
points in each cell. ::

    class GridAdaptedFeatureDetector : public FeatureDetector
    {
    public:
        /*
         * detector            Detector that will be adapted.
         * maxTotalKeypoints   Maximum count of keypoints detected on the image.
         *                     Only the strongest keypoints will be keeped.
         * gridRows            Grid rows count.
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
..

.. index:: PyramidAdaptedFeatureDetector

.. _PyramidAdaptedFeatureDetector:

PyramidAdaptedFeatureDetector
-----------------------------
.. c:type:: PyramidAdaptedFeatureDetector

Adapts a detector to detect points over multiple levels of a Gaussian
pyramid. Useful for detectors that are not inherently scaled. ::

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
..

.. index:: DynamicAdaptedFeatureDetector

.. _DynamicAdaptedFeatureDetector:

DynamicAdaptedFeatureDetector
-----------------------------
.. c:type:: DynamicAdaptedFeatureDetector

An adaptively adjusting detector that iteratively detects until the desired number
of features are found.

If the detector is persisted, it will "remember" the parameters
used on the last detection. In this way, the detector may be used for consistent numbers
of keypoints in a sets of images that are temporally related such as video streams or
panorama series.

The DynamicAdaptedFeatureDetector uses another detector such as FAST or SURF to do the dirty work,
with the help of an AdjusterAdapter.
After a detection, and an unsatisfactory number of features are detected,
the AdjusterAdapter will adjust the detection parameters so that the next detection will
result in more or less features.  This is repeated until either the number of desired features are found
or the parameters are maxed out.

Adapters can easily be implemented for any detector via the
AdjusterAdapter interface.

Beware that this is not thread safe - as the adjustment of parameters breaks the const
of the detection routine...

Here is a sample of how to create a DynamicAdaptedFeatureDetector. ::

    //sample usage:
    //will create a detector that attempts to find
    //100 - 110 FAST Keypoints, and will at most run
    //FAST feature detection 10 times until that
    //number of keypoints are found
    Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector (100, 110, 10,
                                  new FastAdjuster(20,true)));
.. ::

    class DynamicAdaptedFeatureDetector: public FeatureDetector
    {
    public:
        DynamicAdaptedFeatureDetector( const Ptr<AdjusterAdapter>& adjaster,
            int min_features=400, int max_features=500, int max_iters=5 );
        ...
    };
..

.. index:: DynamicAdaptedFeatureDetector::DynamicAdaptedFeatureDetector

DynamicAdaptedFeatureDetector::DynamicAdaptedFeatureDetector
----------------------------------------------------------------
.. c:function:: DynamicAdaptedFeatureDetector::DynamicAdaptedFeatureDetector(       const Ptr<AdjusterAdapter>\& adjaster,       int min_features,   int max_features,   int max_iters )

    DynamicAdaptedFeatureDetector constructor.

    :param adjaster:  An  :func:`AdjusterAdapter`  that will do the detection and parameter
                  adjustment

    :param min_features: This minimum desired number features.

    :param max_features: The maximum desired number of features.

    :param max_iters: The maximum number of times to try to adjust the feature detector parameters. For the  :func:`FastAdjuster`  this number can be high,
                         but with Star or Surf, many iterations can get time consuming.  At each iteration the detector is rerun, so keep this in mind when choosing this value.

.. index:: AdjusterAdapter

.. _AdjusterAdapter:

AdjusterAdapter
---------------
.. c:type:: AdjusterAdapter

A feature detector parameter adjuster interface, this is used by the
:func:`DynamicAdaptedFeatureDetector` and is a wrapper for
:func:`FeatureDetecto` r that allow them to be adjusted after a detection.

See
:func:`FastAdjuster`,:func:`StarAdjuster`,:func:`SurfAdjuster` for concrete implementations. ::

    class AdjusterAdapter: public FeatureDetector
    {
    public:
            virtual ~AdjusterAdapter() {}
            virtual void tooFew(int min, int n_detected) = 0;
            virtual void tooMany(int max, int n_detected) = 0;
            virtual bool good() const = 0;
    };
..

.. index:: AdjusterAdapter::tooFew

AdjusterAdapter::tooFew
---------------------------
.. c:function:: virtual void tooFew(int min, int n_detected) = 0

Too few features were detected so, adjust the detector parameters accordingly - so that the next
detection detects more features.

    :param min: This minimum desired number features.

    :param n_detected: The actual number detected last run.

An example implementation of this is ::

    void FastAdjuster::tooFew(int min, int n_detected)
    {
            thresh_--;
    }
..

.. index:: AdjusterAdapter::tooMany

AdjusterAdapter::tooMany
----------------------------
.. c:function:: virtual void tooMany(int max, int n_detected) = 0

    Too many features were detected so, adjust the detector parameters accordingly - so that the next
detection detects less features.

    :param max: This maximum desired number features.

    :param n_detected: The actual number detected last run.

An example implementation of this is ::

    void FastAdjuster::tooMany(int min, int n_detected)
    {
            thresh_++;
    }
..

.. index:: AdjusterAdapter::good

AdjusterAdapter::good
-------------------------
.. c:function:: virtual bool good() const = 0

    Are params maxed out or still valid? Returns false if the parameters can't be adjusted any more.

An example implementation of this is ::

    bool FastAdjuster::good() const
    {
            return (thresh_ > 1) && (thresh_ < 200);
    }
..

.. index:: FastAdjuster

.. _FastAdjuster:

FastAdjuster
------------
.. c:type:: FastAdjuster

An
:func:`AdjusterAdapter` for the
:func:`FastFeatureDetector` . This will basically decrement or increment the
threshhold by 1 ::

    class FastAdjuster FastAdjuster: public AdjusterAdapter
    {
    public:
            FastAdjuster(int init_thresh = 20, bool nonmax = true);
            ...
    };
..

.. index:: StarAdjuster

.. _StarAdjuster:

StarAdjuster
------------
.. c:type:: StarAdjuster

An
:func:`AdjusterAdapter` for the
:func:`StarFeatureDetector` .  This adjusts the responseThreshhold of
StarFeatureDetector. ::

    class StarAdjuster: public AdjusterAdapter
    {
            StarAdjuster(double initial_thresh = 30.0);
            ...
    };
..

.. index:: SurfAdjuster

.. _SurfAdjuster:

SurfAdjuster
------------
.. c:type:: SurfAdjuster

An
:func:`AdjusterAdapter` for the
:func:`SurfFeatureDetector` .  This adjusts the hessianThreshold of
SurfFeatureDetector. ::

    class SurfAdjuster: public SurfAdjuster
    {
            SurfAdjuster();
            ...
    };
..

