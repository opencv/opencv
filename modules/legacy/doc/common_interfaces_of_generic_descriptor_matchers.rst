Common Interfaces of Generic Descriptor Matchers
================================================

.. highlight:: cpp

OneWayDescriptorBase
--------------------
.. ocv:class:: OneWayDescriptorBase

Class encapsulates functionality for training/loading a set of one way descriptors
and finding the nearest closest descriptor to an input feature. ::

    class CV_EXPORTS OneWayDescriptorBase
    {
    public:

        // creates an instance of OneWayDescriptor from a set of training files
        // - patch_size: size of the input (large) patch
        // - pose_count: the number of poses to generate for each descriptor
        // - train_path: path to training files
        // - pca_config: the name of the file that contains PCA for small patches (2 times smaller
        // than patch_size each dimension
        // - pca_hr_config: the name of the file that contains PCA for large patches (of patch_size size)
        // - pca_desc_config: the name of the file that contains descriptors of PCA components
        OneWayDescriptorBase(CvSize patch_size, int pose_count, const char* train_path = 0, const char* pca_config = 0,
                            const char* pca_hr_config = 0, const char* pca_desc_config = 0, int pyr_levels = 1,
                            int pca_dim_high = 100, int pca_dim_low = 100);

        OneWayDescriptorBase(CvSize patch_size, int pose_count, const String &pca_filename, const String &train_path = String(), const String &images_list = String(),
                            float _scale_min = 0.7f, float _scale_max=1.5f, float _scale_step=1.2f, int pyr_levels = 1,
                            int pca_dim_high = 100, int pca_dim_low = 100);


        virtual ~OneWayDescriptorBase();
        void clear ();


        // Allocate: allocates memory for a given number of descriptors
        void Allocate(int train_feature_count);

        // AllocatePCADescriptors: allocates memory for pca descriptors
        void AllocatePCADescriptors();

        // returns patch size
        CvSize GetPatchSize() const {return m_patch_size;};
        // returns the number of poses for each descriptor
        int GetPoseCount() const {return m_pose_count;};

        // returns the number of pyramid levels
        int GetPyrLevels() const {return m_pyr_levels;};

        // returns the number of descriptors
        int GetDescriptorCount() const {return m_train_feature_count;};

        // CreateDescriptorsFromImage: creates descriptors for each of the input features
        // - src: input image
        // - features: input features
        // - pyr_levels: the number of pyramid levels
        void CreateDescriptorsFromImage(IplImage* src, const vector<KeyPoint>& features);

        // CreatePCADescriptors: generates descriptors for PCA components, needed for fast generation of feature descriptors
        void CreatePCADescriptors();

        // returns a feature descriptor by feature index
        const OneWayDescriptor* GetDescriptor(int desc_idx) const {return &m_descriptors[desc_idx];};

        // FindDescriptor: finds the closest descriptor
        // - patch: input image patch
        // - desc_idx: output index of the closest descriptor to the input patch
        // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
        // - distance: distance from the input patch to the closest feature pose
        // - _scales: scales of the input patch for each descriptor
        // - scale_ranges: input scales variation (float[2])
        void FindDescriptor(IplImage* patch, int& desc_idx, int& pose_idx, float& distance, float* _scale = 0, float* scale_ranges = 0) const;

        // - patch: input image patch
        // - n: number of the closest indexes
        // - desc_idxs: output indexes of the closest descriptor to the input patch (n)
        // - pose_idx: output indexes of the closest pose of the closest descriptor to the input patch (n)
        // - distances: distance from the input patch to the closest feature pose (n)
        // - _scales: scales of the input patch
        // - scale_ranges: input scales variation (float[2])
        void FindDescriptor(IplImage* patch, int n, vector<int>& desc_idxs, vector<int>& pose_idxs,
                            vector<float>& distances, vector<float>& _scales, float* scale_ranges = 0) const;

        // FindDescriptor: finds the closest descriptor
        // - src: input image
        // - pt: center of the feature
        // - desc_idx: output index of the closest descriptor to the input patch
        // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
        // - distance: distance from the input patch to the closest feature pose
        void FindDescriptor(IplImage* src, cv::Point2f pt, int& desc_idx, int& pose_idx, float& distance) const;

        // InitializePoses: generates random poses
        void InitializePoses();

        // InitializeTransformsFromPoses: generates 2x3 affine matrices from poses (initializes m_transforms)
        void InitializeTransformsFromPoses();

        // InitializePoseTransforms: subsequently calls InitializePoses and InitializeTransformsFromPoses
        void InitializePoseTransforms();

        // InitializeDescriptor: initializes a descriptor
        // - desc_idx: descriptor index
        // - train_image: image patch (ROI is supported)
        // - feature_label: feature textual label
        void InitializeDescriptor(int desc_idx, IplImage* train_image, const char* feature_label);

        void InitializeDescriptor(int desc_idx, IplImage* train_image, const KeyPoint& keypoint, const char* feature_label);

        // InitializeDescriptors: load features from an image and create descriptors for each of them
        void InitializeDescriptors(IplImage* train_image, const vector<KeyPoint>& features,
                                  const char* feature_label = "", int desc_start_idx = 0);

        // Write: writes this object to a file storage
        // - fs: output filestorage
        void Write (FileStorage &fs) const;

        // Read: reads OneWayDescriptorBase object from a file node
        // - fn: input file node
        void Read (const FileNode &fn);

        // LoadPCADescriptors: loads PCA descriptors from a file
        // - filename: input filename
        int LoadPCADescriptors(const char* filename);

        // LoadPCADescriptors: loads PCA descriptors from a file node
        // - fn: input file node
        int LoadPCADescriptors(const FileNode &fn);

        // SavePCADescriptors: saves PCA descriptors to a file
        // - filename: output filename
        void SavePCADescriptors(const char* filename);

        // SavePCADescriptors: saves PCA descriptors to a file storage
        // - fs: output file storage
        void SavePCADescriptors(CvFileStorage* fs) const;

        // GeneratePCA: calculate and save PCA components and descriptors
        // - img_path: path to training PCA images directory
        // - images_list: filename with filenames of training PCA images
        void GeneratePCA(const char* img_path, const char* images_list, int pose_count=500);

        // SetPCAHigh: sets the high resolution pca matrices (copied to internal structures)
        void SetPCAHigh(CvMat* avg, CvMat* eigenvectors);

        // SetPCALow: sets the low resolution pca matrices (copied to internal structures)
        void SetPCALow(CvMat* avg, CvMat* eigenvectors);

        int GetLowPCA(CvMat** avg, CvMat** eigenvectors)
        {
            *avg = m_pca_avg;
            *eigenvectors = m_pca_eigenvectors;
            return m_pca_dim_low;
        };

        int GetPCADimLow() const {return m_pca_dim_low;};
        int GetPCADimHigh() const {return m_pca_dim_high;};

        void ConvertDescriptorsArrayToTree(); // Converting pca_descriptors array to KD tree

        // GetPCAFilename: get default PCA filename
        static String GetPCAFilename () { return "pca.yml"; }

        virtual bool empty() const { return m_train_feature_count <= 0 ? true : false; }

    protected:
        ...
    };

OneWayDescriptorMatcher
-----------------------
.. ocv:class:: OneWayDescriptorMatcher : public GenericDescriptorMatcher

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
                    String pcaFilename = String(),
                    String trainPath = String(), String trainImagesList = String(),
                    float minScale = GET_MIN_SCALE(), float maxScale = GET_MAX_SCALE(),
                    float stepScale = GET_STEP_SCALE() );

            int poseCount;
            Size patchSize;
            String pcaFilename;
            String trainPath;
            String trainImagesList;

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

FernClassifier
--------------
.. ocv:class:: FernClassifier

::

    class CV_EXPORTS FernClassifier
    {
    public:
        FernClassifier();
        FernClassifier(const FileNode& node);
        FernClassifier(const vector<vector<Point2f> >& points,
                      const vector<Mat>& refimgs,
                      const vector<vector<int> >& labels=vector<vector<int> >(),
                      int _nclasses=0, int _patchSize=PATCH_SIZE,
                      int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                      int _nstructs=DEFAULT_STRUCTS,
                      int _structSize=DEFAULT_STRUCT_SIZE,
                      int _nviews=DEFAULT_VIEWS,
                      int _compressionMethod=COMPRESSION_NONE,
                      const PatchGenerator& patchGenerator=PatchGenerator());
        virtual ~FernClassifier();
        virtual void read(const FileNode& n);
        virtual void write(FileStorage& fs, const String& name=String()) const;
        virtual void trainFromSingleView(const Mat& image,
                                        const vector<KeyPoint>& keypoints,
                                        int _patchSize=PATCH_SIZE,
                                        int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                                        int _nstructs=DEFAULT_STRUCTS,
                                        int _structSize=DEFAULT_STRUCT_SIZE,
                                        int _nviews=DEFAULT_VIEWS,
                                        int _compressionMethod=COMPRESSION_NONE,
                                        const PatchGenerator& patchGenerator=PatchGenerator());
        virtual void train(const vector<vector<Point2f> >& points,
                          const vector<Mat>& refimgs,
                          const vector<vector<int> >& labels=vector<vector<int> >(),
                          int _nclasses=0, int _patchSize=PATCH_SIZE,
                          int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                          int _nstructs=DEFAULT_STRUCTS,
                          int _structSize=DEFAULT_STRUCT_SIZE,
                          int _nviews=DEFAULT_VIEWS,
                          int _compressionMethod=COMPRESSION_NONE,
                          const PatchGenerator& patchGenerator=PatchGenerator());
        virtual int operator()(const Mat& img, Point2f kpt, vector<float>& signature) const;
        virtual int operator()(const Mat& patch, vector<float>& signature) const;
        virtual void clear();
        virtual bool empty() const;
        void setVerbose(bool verbose);

        int getClassCount() const;
        int getStructCount() const;
        int getStructSize() const;
        int getSignatureSize() const;
        int getCompressionMethod() const;
        Size getPatchSize() const;

        struct Feature
        {
            uchar x1, y1, x2, y2;
            Feature() : x1(0), y1(0), x2(0), y2(0) {}
            Feature(int _x1, int _y1, int _x2, int _y2)
            : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
            {}
            template<typename _Tp> bool operator ()(const Mat_<_Tp>& patch) const
            { return patch(y1,x1) > patch(y2, x2); }
        };

        enum
        {
            PATCH_SIZE = 31,
            DEFAULT_STRUCTS = 50,
            DEFAULT_STRUCT_SIZE = 9,
            DEFAULT_VIEWS = 5000,
            DEFAULT_SIGNATURE_SIZE = 176,
            COMPRESSION_NONE = 0,
            COMPRESSION_RANDOM_PROJ = 1,
            COMPRESSION_PCA = 2,
            DEFAULT_COMPRESSION_METHOD = COMPRESSION_NONE
        };

    protected:
        ...
    };

FernDescriptorMatcher
---------------------
.. ocv:class:: FernDescriptorMatcher : public GenericDescriptorMatcher

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

            Params( const String& filename );

            int nclasses;
            int patchSize;
            int signatureSize;
            int nstructs;
            int structSize;
            int nviews;
            int compressionMethod;
            PatchGenerator patchGenerator;

            String filename;
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
