Common Interfaces of Generic Descriptor Matchers
================================================

.. highlight:: cpp

Matchers of keypoint descriptors in OpenCV have wrappers with common interface that enables to switch easily
between different algorithms solving the same problem. This section is devoted to matching descriptors
that can not be represented as vectors in a multidimensional space. ``GenericDescriptorMatcher`` is a more generic interface for descriptors. It does not make any assumptions about descriptor representation.
Every descriptor with
:func:`DescriptorExtractor` interface has a wrapper with ``GenericDescriptorMatcher`` interface (see
:func:`VectorDescriptorMatcher` ).
There are descriptors such as One way descriptor and Ferns that have ``GenericDescriptorMatcher`` interface implemented, but do not support
:func:`DescriptorExtractor` .

.. index:: GenericDescriptorMatcher

.. _GenericDescriptorMatcher:

GenericDescriptorMatcher
------------------------
.. ctype:: GenericDescriptorMatcher

Abstract interface for a keypoint descriptor extracting and matching.
There is
:func:`DescriptorExtractor` and
:func:`DescriptorMatcher` for these purposes too, but their interfaces are intended for descriptors
represented as vectors in a multidimensional space. ``GenericDescriptorMatcher`` is a more generic interface for descriptors.
As
:func:`DescriptorMatcher`,``GenericDescriptorMatcher`` has two groups
of match methods: for matching keypoints of one image with other image or
with image set. ::

    class GenericDescriptorMatcher
    {
    public:
        GenericDescriptorMatcher();
        virtual ~GenericDescriptorMatcher();

        virtual void add( const vector<Mat>& images,
                          vector<vector<KeyPoint> >& keypoints );

        const vector<Mat>& getTrainImages() const;
        const vector<vector<KeyPoint> >& getTrainKeypoints() const;
        virtual void clear();

        virtual void train() = 0;

        virtual bool isMaskSupported() = 0;

        void classify( const Mat& queryImage,
                       vector<KeyPoint>& queryKeypoints,
                       const Mat& trainImage,
                       vector<KeyPoint>& trainKeypoints ) const;
        void classify( const Mat& queryImage,
                       vector<KeyPoint>& queryKeypoints );

        /*
         * Group of methods to match keypoints from image pair.
         */
        void match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                    const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                    vector<DMatch>& matches, const Mat& mask=Mat() ) const;
        void knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                       const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                       vector<vector<DMatch> >& matches, int k,
                       const Mat& mask=Mat(), bool compactResult=false ) const;
        void radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                          const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                          vector<vector<DMatch> >& matches, float maxDistance,
                          const Mat& mask=Mat(), bool compactResult=false ) const;
        /*
         * Group of methods to match keypoints from one image to image set.
         */
        void match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                    vector<DMatch>& matches, const vector<Mat>& masks=vector<Mat>() );
        void knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                       vector<vector<DMatch> >& matches, int k,
                       const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
        void radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                          vector<vector<DMatch> >& matches, float maxDistance,
                          const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );

        virtual void read( const FileNode& );
        virtual void write( FileStorage& ) const;

        virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

    protected:
        ...
    };
..

.. index:: GenericDescriptorMatcher::add

cv::GenericDescriptorMatcher::add
---------------------------------
.. cfunction:: void GenericDescriptorMatcher::add( const vector<Mat>\& images,                        vector<vector<KeyPoint> >\& keypoints )

    Adds images and keypoints from them to the train collection (descriptors are supposed to be calculated here).
If train collection is not empty new image and keypoints from them will be added to
existing data.

    :param images: Image collection.

    :param keypoints: Point collection. Assumes that  ``keypoints[i]``  are keypoints
                          detected in an image  ``images[i]`` .

.. index:: GenericDescriptorMatcher::getTrainImages

cv::GenericDescriptorMatcher::getTrainImages
--------------------------------------------
.. cfunction:: const vector<Mat>\& GenericDescriptorMatcher::getTrainImages() const

    Returns train image collection.

.. index:: GenericDescriptorMatcher::getTrainKeypoints

cv::GenericDescriptorMatcher::getTrainKeypoints
-----------------------------------------------
.. cfunction:: const vector<vector<KeyPoint> >\&  GenericDescriptorMatcher::getTrainKeypoints() const

    Returns train keypoints collection.

.. index:: GenericDescriptorMatcher::clear

cv::GenericDescriptorMatcher::clear
-----------------------------------
.. cfunction:: void GenericDescriptorMatcher::clear()

    Clear train collection (iamges and keypoints).

.. index:: GenericDescriptorMatcher::train

cv::GenericDescriptorMatcher::train
-----------------------------------
.. cfunction:: void GenericDescriptorMatcher::train()

    Train the object, e.g. tree-based structure to extract descriptors or
to optimize descriptors matching.

.. index:: GenericDescriptorMatcher::isMaskSupported

cv::GenericDescriptorMatcher::isMaskSupported
---------------------------------------------
.. cfunction:: void GenericDescriptorMatcher::isMaskSupported()

    Returns true if generic descriptor matcher supports masking permissible matches.

.. index:: GenericDescriptorMatcher::classify

cv::GenericDescriptorMatcher::classify
--------------------------------------
:func:`GenericDescriptorMatcher::add`
.. cfunction:: void GenericDescriptorMatcher::classify(  const Mat\& queryImage,           vector<KeyPoint>\& queryKeypoints,           const Mat\& trainImage,           vector<KeyPoint>\& trainKeypoints ) const

    Classifies query keypoints under keypoints of one train image qiven as input argument
(first version of the method) or train image collection that set using (second version).

.. cfunction:: void GenericDescriptorMatcher::classify( const Mat\& queryImage,           vector<KeyPoint>\& queryKeypoints )

    :param queryImage: The query image.

    :param queryKeypoints: Keypoints from the query image.

    :param trainImage: The train image.

    :param trainKeypoints: Keypoints from the train image.

.. index:: GenericDescriptorMatcher::match

cv::GenericDescriptorMatcher::match
-----------------------------------
:func:`GenericDescriptorMatcher::add` :func:`DescriptorMatcher::match`
.. cfunction:: void GenericDescriptorMatcher::match(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      const Mat\& trainImage, vector<KeyPoint>\& trainKeypoints,      vector<DMatch>\& matches, const Mat\& mask=Mat() ) const

    Find best match for query keypoints to the training set. In first version of method
one train image and keypoints detected on it - are input arguments. In second version
query keypoints are matched to training collectin that set using . As in the mask can be set.

.. cfunction:: void GenericDescriptorMatcher::match(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,          vector<DMatch>\& matches,           const vector<Mat>\& masks=vector<Mat>() )

    :param queryImage: Query image.

    :param queryKeypoints: Keypoints detected in  ``queryImage`` .

    :param trainImage: Train image. This will not be added to train image collection
                                        stored in class object.

    :param trainKeypoints: Keypoints detected in  ``trainImage`` . They will not be added to train points collection
                                           stored in class object.

    :param matches: Matches. If some query descriptor (keypoint) masked out in  ``mask``                         no match will be added for this descriptor.
                                        So  ``matches``  size may be less query keypoints count.

    :param mask: Mask specifying permissible matches between input query and train keypoints.

    :param masks: The set of masks. Each  ``masks[i]``  specifies permissible matches between input query keypoints
                      and stored train keypointss from i-th image.

.. index:: GenericDescriptorMatcher::knnMatch

cv::GenericDescriptorMatcher::knnMatch
--------------------------------------
:func:`GenericDescriptorMatcher::match` :func:`DescriptorMatcher::knnMatch`
.. cfunction:: void GenericDescriptorMatcher::knnMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      const Mat\& trainImage, vector<KeyPoint>\& trainKeypoints,      vector<vector<DMatch> >\& matches, int k,       const Mat\& mask=Mat(), bool compactResult=false ) const

    Find the knn best matches for each keypoint from a query set with train keypoints.
Found knn (or less if not possible) matches are returned in distance increasing order.
Details see in and .

.. cfunction:: void GenericDescriptorMatcher::knnMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      vector<vector<DMatch> >\& matches, int k,       const vector<Mat>\& masks=vector<Mat>(),       bool compactResult=false )

.. index:: GenericDescriptorMatcher::radiusMatch

cv::GenericDescriptorMatcher::radiusMatch
-----------------------------------------
:func:`GenericDescriptorMatcher::match` :func:`DescriptorMatcher::radiusMatch`
.. cfunction:: void GenericDescriptorMatcher::radiusMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      const Mat\& trainImage, vector<KeyPoint>\& trainKeypoints,      vector<vector<DMatch> >\& matches, float maxDistance,       const Mat\& mask=Mat(), bool compactResult=false ) const

    Find the best matches for each query keypoint which have distance less than given threshold.
Found matches are returned in distance increasing order. Details see in and .

.. cfunction:: void GenericDescriptorMatcher::radiusMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      vector<vector<DMatch> >\& matches, float maxDistance,       const vector<Mat>\& masks=vector<Mat>(),       bool compactResult=false )

.. index:: GenericDescriptorMatcher::read

cv::GenericDescriptorMatcher::read
----------------------------------
.. cfunction:: void GenericDescriptorMatcher::read( const FileNode\& fn )

    Reads matcher object from a file node.

.. index:: GenericDescriptorMatcher::write

cv::GenericDescriptorMatcher::write
-----------------------------------
.. cfunction:: void GenericDescriptorMatcher::write( FileStorage\& fs ) const

    Writes match object to a file storage

.. index:: GenericDescriptorMatcher::clone

cv::GenericDescriptorMatcher::clone
-----------------------------------
.. cfunction:: Ptr<GenericDescriptorMatcher>\\GenericDescriptorMatcher::clone( bool emptyTrainData ) const

    Clone the matcher.

    :param emptyTrainData: If emptyTrainData is false the method create deep copy of the object, i.e. copies
            both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
            but with empty train data.

.. index:: OneWayDescriptorMatcher

.. _OneWayDescriptorMatcher:

OneWayDescriptorMatcher
-----------------------
.. ctype:: OneWayDescriptorMatcher

Wrapping class for computing, matching and classification of descriptors using
:func:`OneWayDescriptorBase` class. ::

    class OneWayDescriptorMatcher : public GenericDescriptorMatcher
    {
    public:
        class Params
        {
        public:
            static const int POSE_COUNT = 500;
            static const int PATCH_WIDTH = 24;
            static const int PATCH_HEIGHT = 24;
            static float GET_MIN_SCALE() { return 0.7f; }
            static float GET_MAX_SCALE() { return 1.5f; }
            static float GET_STEP_SCALE() { return 1.2f; }

            Params( int poseCount = POSE_COUNT,
                    Size patchSize = Size(PATCH_WIDTH, PATCH_HEIGHT),
                    string pcaFilename = string(),
                    string trainPath = string(), string trainImagesList = string(),
                    float minScale = GET_MIN_SCALE(), float maxScale = GET_MAX_SCALE(),
                    float stepScale = GET_STEP_SCALE() );

            int poseCount;
            Size patchSize;
            string pcaFilename;
            string trainPath;
            string trainImagesList;

            float minScale, maxScale, stepScale;
        };

        OneWayDescriptorMatcher( const Params& params=Params() );
        virtual ~OneWayDescriptorMatcher();

        void initialize( const Params& params, const Ptr<OneWayDescriptorBase>& base=Ptr<OneWayDescriptorBase>() );

        // Clears keypoints storing in collection and OneWayDescriptorBase
        virtual void clear();

        virtual void train();

        virtual bool isMaskSupported();

        virtual void read( const FileNode &fn );
        virtual void write( FileStorage& fs ) const;

        virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;
    protected:
        ...
    };
..

.. index:: FernDescriptorMatcher

.. _FernDescriptorMatcher:

FernDescriptorMatcher
---------------------
.. ctype:: FernDescriptorMatcher

Wrapping class for computing, matching and classification of descriptors using
:func:`FernClassifier` class. ::

    class FernDescriptorMatcher : public GenericDescriptorMatcher
    {
    public:
        class Params
        {
        public:
            Params( int nclasses=0,
                    int patchSize=FernClassifier::PATCH_SIZE,
                    int signatureSize=FernClassifier::DEFAULT_SIGNATURE_SIZE,
                    int nstructs=FernClassifier::DEFAULT_STRUCTS,
                    int structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                    int nviews=FernClassifier::DEFAULT_VIEWS,
                    int compressionMethod=FernClassifier::COMPRESSION_NONE,
                    const PatchGenerator& patchGenerator=PatchGenerator() );

            Params( const string& filename );

            int nclasses;
            int patchSize;
            int signatureSize;
            int nstructs;
            int structSize;
            int nviews;
            int compressionMethod;
            PatchGenerator patchGenerator;

            string filename;
        };

        FernDescriptorMatcher( const Params& params=Params() );
        virtual ~FernDescriptorMatcher();

        virtual void clear();

        virtual void train();

        virtual bool isMaskSupported();

        virtual void read( const FileNode &fn );
        virtual void write( FileStorage& fs ) const;

        virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;

    protected:
            ...
    };
..

.. index:: VectorDescriptorMatcher

.. _VectorDescriptorMatcher:

VectorDescriptorMatcher
-----------------------
.. ctype:: VectorDescriptorMatcher

Class used for matching descriptors that can be described as vectors in a finite-dimensional space. ::

    class CV_EXPORTS VectorDescriptorMatcher : public GenericDescriptorMatcher
    {
    public:
        VectorDescriptorMatcher( const Ptr<DescriptorExtractor>& extractor, const Ptr<DescriptorMatcher>& matcher );
        virtual ~VectorDescriptorMatcher();

        virtual void add( const vector<Mat>& imgCollection,
                          vector<vector<KeyPoint> >& pointCollection );
        virtual void clear();
        virtual void train();
        virtual bool isMaskSupported();

        virtual void read( const FileNode& fn );
        virtual void write( FileStorage& fs ) const;

        virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;

    protected:
        ...
    };
..

Example of creating: ::

    VectorDescriptorMatcher matcher( new SurfDescriptorExtractor,
                                     new BruteForceMatcher<L2<float> > );
..

