Common Interfaces of Generic Descriptor Matchers
================================================

.. highlight:: cpp

Matchers of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to easily switch 
between different algorithms solving the same problem. This section is devoted to matching descriptors
that cannot be represented as vectors in a multidimensional space. ``GenericDescriptorMatcher`` is a more generic interface for descriptors. It does not make any assumptions about descriptor representation.
Every descriptor with the
:ref:`DescriptorExtractor` interface has a wrapper with the ``GenericDescriptorMatcher`` interface (see
:ref:`VectorDescriptorMatcher` ).
There are descriptors such as the One-way descriptor and Ferns that have the ``GenericDescriptorMatcher`` interface implemented but do not support
:ref:`DescriptorExtractor` .

.. index:: GenericDescriptorMatcher

GenericDescriptorMatcher
------------------------
.. c:type:: GenericDescriptorMatcher

Abstract interface for extracting and matching a keypoint descriptor. There are aslo :ref:`DescriptorExtractor` and :ref:`DescriptorMatcher` for these purposes but their interfaces are intended for descriptors represented as vectors in a multidimensional space. ``GenericDescriptorMatcher`` is a more generic interface for descriptors. :ref:`DescriptorMatcher` and``GenericDescriptorMatcher`` have two groups of match methods: for matching keypoints of an image with another image or with an image set. ::

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
         * Group of methods to match keypoints from an image pair.
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
         * Group of methods to match keypoints from one image to an image set.
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


.. index:: GenericDescriptorMatcher::add

GenericDescriptorMatcher::add
---------------------------------
.. c:function:: void GenericDescriptorMatcher::add( const vector<Mat>\& images,                        vector<vector<KeyPoint> >\& keypoints )

    Adds images and their keypoints to a train collection (descriptors are supposed to be calculated here). If the train collection is not empty, a new image and its keypoints is added to existing data.??

    :param images: Image collection.

    :param keypoints: Point collection. It is assumed that ``keypoints[i]``  are keypoints detected in the image  ``images[i]`` .

.. index:: GenericDescriptorMatcher::getTrainImages

GenericDescriptorMatcher::getTrainImages
--------------------------------------------
.. c:function:: const vector<Mat>\& GenericDescriptorMatcher::getTrainImages() const

    Returns a train image collection.

.. index:: GenericDescriptorMatcher::getTrainKeypoints

GenericDescriptorMatcher::getTrainKeypoints
-----------------------------------------------
.. c:function:: const vector<vector<KeyPoint> >\&  GenericDescriptorMatcher::getTrainKeypoints() const

    Returns a train keypoints collection.

.. index:: GenericDescriptorMatcher::clear

GenericDescriptorMatcher::clear
-----------------------------------
.. c:function:: void GenericDescriptorMatcher::clear()

    Clears a train collection (images and keypoints).

.. index:: GenericDescriptorMatcher::train

GenericDescriptorMatcher::train
-----------------------------------
.. c:function:: void GenericDescriptorMatcher::train()

    Trains an object, for example, a tree-based structure, to extract descriptors or to optimize descriptors matching.

.. index:: GenericDescriptorMatcher::isMaskSupported

GenericDescriptorMatcher::isMaskSupported
---------------------------------------------
.. c:function:: void GenericDescriptorMatcher::isMaskSupported()

    Returns true if a generic descriptor matcher supports masking permissible matches.

.. index:: GenericDescriptorMatcher::classify

GenericDescriptorMatcher::classify
--------------------------------------
.. c:function:: void GenericDescriptorMatcher::classify(  const Mat\& queryImage,           vector<KeyPoint>\& queryKeypoints,           const Mat\& trainImage,           vector<KeyPoint>\& trainKeypoints ) const

    Classifies query keypoints under keypoints of a train image qiven as an input argument (first version of the method) or a train image collection (second version).??

.. c:function:: void GenericDescriptorMatcher::classify( const Mat\& queryImage,           vector<KeyPoint>\& queryKeypoints )

    :param queryImage: Query image.

    :param queryKeypoints: Keypoints from the query image.

    :param trainImage: Train image.

    :param trainKeypoints: Keypoints from the train image.

.. index:: GenericDescriptorMatcher::match

GenericDescriptorMatcher::match
-----------------------------------
:func:`GenericDescriptorMatcher::add` :func:`DescriptorMatcher::match`
.. c:function:: void GenericDescriptorMatcher::match(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      const Mat\& trainImage, vector<KeyPoint>\& trainKeypoints,      vector<DMatch>\& matches, const Mat\& mask=Mat() ) const

    Finds the best match for query keypoints to the training set. In the first version of the method, a train image and keypoints detected on it are input arguments. In the second version, query keypoints are matched to a training collection set using ??. As in the mask can be set.??

.. c:function:: void GenericDescriptorMatcher::match(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,          vector<DMatch>\& matches,           const vector<Mat>\& masks=vector<Mat>() )

    :param queryImage: Query image.

    :param queryKeypoints: Keypoints detected in  ``queryImage`` .

    :param trainImage: Train image. It is not added to a train image collection  stored in the class object.

    :param trainKeypoints: Keypoints detected in  ``trainImage`` . They are not added to a train points collection stored in the class object.

    :param matches: Matches. If a query descriptor (keypoint) is masked out in  ``mask`` ,  match is added for this descriptor. So,  ``matches``  size may be smaller than the query keypoints count.

    :param mask: Mask specifying permissible matches between input query and train keypoints.

    :param masks: Set of masks. Each  ``masks[i]``  specifies permissible matches between input query keypoints and stored train keypoints from the i-th image.

.. index:: GenericDescriptorMatcher::knnMatch

GenericDescriptorMatcher::knnMatch
--------------------------------------
.. c:function:: void GenericDescriptorMatcher::knnMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      const Mat\& trainImage, vector<KeyPoint>\& trainKeypoints,      vector<vector<DMatch> >\& matches, int k,       const Mat\& mask=Mat(), bool compactResult=false ) const

    Finds the knn best matches for each keypoint from a query set with train keypoints. Found knn (or less if not possible) matches are returned in the distance increasing order. See details in ??.

.. c:function:: void GenericDescriptorMatcher::knnMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      vector<vector<DMatch> >\& matches, int k,       const vector<Mat>\& masks=vector<Mat>(),       bool compactResult=false )

.. index:: GenericDescriptorMatcher::radiusMatch

GenericDescriptorMatcher::radiusMatch
-----------------------------------------
.. c:function:: void GenericDescriptorMatcher::radiusMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      const Mat\& trainImage, vector<KeyPoint>\& trainKeypoints,      vector<vector<DMatch> >\& matches, float maxDistance,       const Mat\& mask=Mat(), bool compactResult=false ) const

    Finds the best matches for each query keypoint that has a distance smaller than the given threshold. Found matches are returned in the distance increasing order. See details see in ??.

.. c:function:: void GenericDescriptorMatcher::radiusMatch(           const Mat\& queryImage, vector<KeyPoint>\& queryKeypoints,      vector<vector<DMatch> >\& matches, float maxDistance,       const vector<Mat>\& masks=vector<Mat>(),       bool compactResult=false )

.. index:: GenericDescriptorMatcher::read

GenericDescriptorMatcher::read
----------------------------------
.. c:function:: void GenericDescriptorMatcher::read( const FileNode\& fn )

    Reads a matcher object from a file node.

.. index:: GenericDescriptorMatcher::write

GenericDescriptorMatcher::write
-----------------------------------
.. c:function:: void GenericDescriptorMatcher::write( FileStorage\& fs ) const

    Writes a match object to a file storage.

.. index:: GenericDescriptorMatcher::clone

GenericDescriptorMatcher::clone
-----------------------------------
.. c:function:: Ptr<GenericDescriptorMatcher>\\GenericDescriptorMatcher::clone( bool emptyTrainData ) const

    Clones the matcher.

    :param emptyTrainData: If ``emptyTrainData`` is false, the method creates a deep copy of the object, that is, copies
            both parameters and train data. If ``emptyTrainData`` is true, the method creates an object copy with the current parameters
            but with empty train data.

.. index:: OneWayDescriptorMatcher

.. _OneWayDescriptorMatcher:

OneWayDescriptorMatcher
-----------------------
.. c:type:: OneWayDescriptorMatcher

Wrapping class for computing, matching, and classifying descriptors using the
:ref:`OneWayDescriptorBase` class ::

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

        // Clears keypoints stored in collection and OneWayDescriptorBase
        virtual void clear();

        virtual void train();

        virtual bool isMaskSupported();

        virtual void read( const FileNode &fn );
        virtual void write( FileStorage& fs ) const;

        virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;
    protected:
        ...
    };


.. index:: FernDescriptorMatcher

FernDescriptorMatcher
---------------------
.. c:type:: FernDescriptorMatcher

Wrapping class for computing, matching, and classifying descriptors using the
:ref:`FernClassifier` class ::

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


.. index:: VectorDescriptorMatcher

.. _VectorDescriptorMatcher:

VectorDescriptorMatcher
-----------------------
.. c:type:: VectorDescriptorMatcher

Class used for matching descriptors that can be described as vectors in a finite-dimensional space ::

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


Example: ::

    VectorDescriptorMatcher matcher( new SurfDescriptorExtractor,
                                     new BruteForceMatcher<L2<float> > );


