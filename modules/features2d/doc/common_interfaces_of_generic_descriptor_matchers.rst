Common Interfaces of Generic Descriptor Matchers
================================================

.. highlight:: cpp

Matchers of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to easily switch 
between different algorithms solving the same problem. This section is devoted to matching descriptors
that cannot be represented as vectors in a multidimensional space. ``GenericDescriptorMatcher`` is a more generic interface for descriptors. It does not make any assumptions about descriptor representation.
Every descriptor with the
:ocv:class:`DescriptorExtractor` interface has a wrapper with the ``GenericDescriptorMatcher`` interface (see
:ocv:class:`VectorDescriptorMatcher` ).
There are descriptors such as the One-way descriptor and Ferns that have the ``GenericDescriptorMatcher`` interface implemented but do not support ``DescriptorExtractor``.



GenericDescriptorMatcher
------------------------
.. ocv:class:: GenericDescriptorMatcher

Abstract interface for extracting and matching a keypoint descriptor. There are also :ocv:class:`DescriptorExtractor` and :ocv:class:`DescriptorMatcher` for these purposes but their interfaces are intended for descriptors represented as vectors in a multidimensional space. ``GenericDescriptorMatcher`` is a more generic interface for descriptors. ``DescriptorMatcher`` and ``GenericDescriptorMatcher`` have two groups of match methods: for matching keypoints of an image with another image or with an image set. ::

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




GenericDescriptorMatcher::add
---------------------------------
Adds images and their keypoints to the training collection stored in the class instance.

.. ocv:function:: void GenericDescriptorMatcher::add( const vector<Mat>& images,                        vector<vector<KeyPoint> >& keypoints )

    :param images: Image collection.

    :param keypoints: Point collection. It is assumed that ``keypoints[i]``  are keypoints detected in the image  ``images[i]`` .



GenericDescriptorMatcher::getTrainImages
--------------------------------------------
Returns a train image collection.

.. ocv:function:: const vector<Mat>& GenericDescriptorMatcher::getTrainImages() const



GenericDescriptorMatcher::getTrainKeypoints
-----------------------------------------------
Returns a train keypoints collection.

.. ocv:function:: const vector<vector<KeyPoint> >&  GenericDescriptorMatcher::getTrainKeypoints() const



GenericDescriptorMatcher::clear
-----------------------------------
Clears a train collection (images and keypoints).

.. ocv:function:: void GenericDescriptorMatcher::clear()



GenericDescriptorMatcher::train
-----------------------------------
Trains descriptor matcher

.. ocv:function:: void GenericDescriptorMatcher::train()

Prepares descriptor matcher, for example, creates a tree-based structure, to extract descriptors or to optimize descriptors matching.


GenericDescriptorMatcher::isMaskSupported
---------------------------------------------
Returns ``true`` if a generic descriptor matcher supports masking permissible matches.

.. ocv:function:: void GenericDescriptorMatcher::isMaskSupported()



GenericDescriptorMatcher::classify
--------------------------------------
Classifies keypoints from a query set.

.. ocv:function:: void GenericDescriptorMatcher::classify(  const Mat& queryImage,           vector<KeyPoint>& queryKeypoints,           const Mat& trainImage,           vector<KeyPoint>& trainKeypoints ) const

.. ocv:function:: void GenericDescriptorMatcher::classify( const Mat& queryImage,           vector<KeyPoint>& queryKeypoints )

    :param queryImage: Query image.

    :param queryKeypoints: Keypoints from a query image.

    :param trainImage: Train image.

    :param trainKeypoints: Keypoints from a train image.

The method classifies each keypoint from a query set. The first variant of the method takes a train image and its keypoints as an input argument. The second variant uses the internally stored training collection that can be built using the ``GenericDescriptorMatcher::add`` method.
    
The methods do the following:
    
#.
    Call the ``GenericDescriptorMatcher::match`` method to find correspondence between the query set and the training set.
        
#.
    Set the ``class_id`` field of each keypoint from the query set to ``class_id`` of the corresponding keypoint from the training set.



GenericDescriptorMatcher::match
-----------------------------------
Finds the best match in the training set for each keypoint from the query set.

.. ocv:function:: void GenericDescriptorMatcher::match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, const Mat& trainImage, vector<KeyPoint>& trainKeypoints, vector<DMatch>& matches, const Mat& mask=Mat() ) const

.. ocv:function:: void GenericDescriptorMatcher::match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, vector<DMatch>& matches, const vector<Mat>& masks=vector<Mat>() )

    :param queryImage: Query image.

    :param queryKeypoints: Keypoints detected in  ``queryImage`` .

    :param trainImage: Train image. It is not added to a train image collection  stored in the class object.

    :param trainKeypoints: Keypoints detected in  ``trainImage`` . They are not added to a train points collection stored in the class object.

    :param matches: Matches. If a query descriptor (keypoint) is masked out in  ``mask`` ,  match is added for this descriptor. So,  ``matches``  size may be smaller than the query keypoints count.

    :param mask: Mask specifying permissible matches between an input query and train keypoints.

    :param masks: Set of masks. Each  ``masks[i]``  specifies permissible matches between input query keypoints and stored train keypoints from the i-th image.

The methods find the best match for each query keypoint. In the first variant of the method, a train image and its keypoints are the input arguments. In the second variant, query keypoints are matched to the internally stored training collection that can be built using the ``GenericDescriptorMatcher::add`` method.     Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, ``queryKeypoints[i]`` can be matched with ``trainKeypoints[j]`` only if ``mask.at<uchar>(i,j)`` is non-zero.



GenericDescriptorMatcher::knnMatch
--------------------------------------
Finds the ``k`` best matches for each query keypoint.

.. ocv:function:: void GenericDescriptorMatcher::knnMatch(           const Mat& queryImage, vector<KeyPoint>& queryKeypoints,      const Mat& trainImage, vector<KeyPoint>& trainKeypoints,      vector<vector<DMatch> >& matches, int k,       const Mat& mask=Mat(), bool compactResult=false ) const

.. ocv:function:: void GenericDescriptorMatcher::knnMatch(           const Mat& queryImage, vector<KeyPoint>& queryKeypoints,      vector<vector<DMatch> >& matches, int k,       const vector<Mat>& masks=vector<Mat>(),       bool compactResult=false )
    
The methods are extended variants of ``GenericDescriptorMatch::match``. The parameters are similar, and the semantics is similar to ``DescriptorMatcher::knnMatch``. But this class does not require explicitly computed keypoint descriptors.



GenericDescriptorMatcher::radiusMatch
-----------------------------------------
For each query keypoint, finds the training keypoints not farther than the specified distance.

.. ocv:function:: void GenericDescriptorMatcher::radiusMatch(           const Mat& queryImage, vector<KeyPoint>& queryKeypoints,      const Mat& trainImage, vector<KeyPoint>& trainKeypoints,      vector<vector<DMatch> >& matches, float maxDistance,       const Mat& mask=Mat(), bool compactResult=false ) const

.. ocv:function:: void GenericDescriptorMatcher::radiusMatch(           const Mat& queryImage, vector<KeyPoint>& queryKeypoints,      vector<vector<DMatch> >& matches, float maxDistance,       const vector<Mat>& masks=vector<Mat>(),       bool compactResult=false )

The methods are similar to ``DescriptorMatcher::radius``. But this class does not require explicitly computed keypoint descriptors.



GenericDescriptorMatcher::read
----------------------------------
Reads a matcher object from a file node.

.. ocv:function:: void GenericDescriptorMatcher::read( const FileNode& fn )



GenericDescriptorMatcher::write
-----------------------------------
Writes a match object to a file storage.

.. ocv:function:: void GenericDescriptorMatcher::write( FileStorage& fs ) const


GenericDescriptorMatcher::clone
-----------------------------------
Clones the matcher.

.. ocv:function:: Ptr<GenericDescriptorMatcher> GenericDescriptorMatcher::clone( bool emptyTrainData ) const

    :param emptyTrainData: If ``emptyTrainData`` is false, the method creates a deep copy of the object, that is, copies
            both parameters and train data. If ``emptyTrainData`` is true, the method creates an object copy with the current parameters
            but with empty train data.


OneWayDescriptorMatcher
-----------------------
.. ocv:class:: OneWayDescriptorMatcher

Wrapping class for computing, matching, and classifying descriptors using the
:ocv:class:`OneWayDescriptorBase` class. ::

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




FernDescriptorMatcher
---------------------
.. ocv:class:: FernDescriptorMatcher

Wrapping class for computing, matching, and classifying descriptors using the
:ocv:class:`FernClassifier` class. ::

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




VectorDescriptorMatcher
-----------------------
.. ocv:class:: VectorDescriptorMatcher

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


Example: ::

    VectorDescriptorMatcher matcher( new SurfDescriptorExtractor,
                                     new BruteForceMatcher<L2<float> > );


