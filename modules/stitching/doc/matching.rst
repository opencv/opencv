Features Finding and Images Matching
====================================

.. highlight:: cpp

detail::ImageFeatures
-----------------------
.. ocv:struct:: detail::ImageFeatures 

Structure containing image keypoints and descriptors. ::

    struct CV_EXPORTS ImageFeatures
    {
        int img_idx;
        Size img_size;
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
    };

detail::FeaturesFinder
----------------------
.. ocv:class:: detail::FeaturesFinder

Feature finders base class. ::

    class CV_EXPORTS FeaturesFinder
    {
    public:
        virtual ~FeaturesFinder() {}
        void operator ()(const Mat &image, ImageFeatures &features);
        void operator ()(const Mat &image, ImageFeatures &features, const std::vector<cv::Rect> &rois);
        virtual void collectGarbage() {}

    protected:
        virtual void find(const Mat &image, ImageFeatures &features) = 0;
    };

detail::SurfFeaturesFinder
--------------------------
.. ocv:class:: detail::SurfFeaturesFinder

SURF features finder. ::

    class CV_EXPORTS SurfFeaturesFinder : public FeaturesFinder
    {
    public:
        SurfFeaturesFinder(double hess_thresh = 300., int num_octaves = 3, int num_layers = 4,
                           int num_octaves_descr = /*4*/3, int num_layers_descr = /*2*/4);

    private:
        /* hidden */
    };

.. seealso::
    :ocv:class:`detail::FeaturesFinder`
    :ocv:class:`SURF`

detail::OrbFeaturesFinder
-------------------------
.. ocv:class:: detail::OrbFeaturesFinder

ORB features finder. ::

    class CV_EXPORTS OrbFeaturesFinder : public FeaturesFinder
    {
    public:
        OrbFeaturesFinder(Size _grid_size = Size(3,1), size_t n_features = 1500, 
                          const ORB::CommonParams &detector_params = ORB::CommonParams(1.3f, 5));

    private:
        /* hidden */
    };

.. seealso:: 
    :ocv:class:`detail::FeaturesFinder`,
    :ocv:class:`ORB`

detail::MatchesInfo
-------------------
.. ocv:struct: detail::MatchesInfo

Structure containing information about matches between two images. It's assumed that there is a homography between those images. ::

    struct CV_EXPORTS MatchesInfo
    {
        MatchesInfo();
        MatchesInfo(const MatchesInfo &other);
        const MatchesInfo& operator =(const MatchesInfo &other);

        int src_img_idx, dst_img_idx;       // Images indices (optional)
        std::vector<DMatch> matches;
        std::vector<uchar> inliers_mask;    // Geometrically consistent matches mask
        int num_inliers;                    // Number of geometrically consistent matches
        Mat H;                              // Estimated homography
        double confidence;                  // Confidence two images are from the same panorama
    };

detail::FeaturesMatcher
-----------------------
.. ocv:class:: detail::FeaturesMatcher

Feature matchers base class. ::

    class CV_EXPORTS FeaturesMatcher
    {
    public:
        virtual ~FeaturesMatcher() {}

        void operator ()(const ImageFeatures &features1, const ImageFeatures &features2, 
                         MatchesInfo& matches_info) { match(features1, features2, matches_info); }

        void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches, 
                         const cv::Mat &mask = cv::Mat());

        bool isThreadSafe() const { return is_thread_safe_; }

        virtual void collectGarbage() {}

    protected:
        FeaturesMatcher(bool is_thread_safe = false) : is_thread_safe_(is_thread_safe) {}

        virtual void match(const ImageFeatures &features1, const ImageFeatures &features2, 
                           MatchesInfo& matches_info) = 0;

        bool is_thread_safe_;
    };

detail::BestOf2NearestMatcher
-----------------------------
.. ocv:class:: detail::BestOf2NearestMatcher

Features matcher which finds two best matches for each feature and leaves the best one only if the ratio between descriptor distances is greater than the threshold ``match_conf``. ::

    class CV_EXPORTS BestOf2NearestMatcher : public FeaturesMatcher
    {
    public:
        BestOf2NearestMatcher(bool try_use_gpu = false, float match_conf = 0.65f, 
                              int num_matches_thresh1 = 6, int num_matches_thresh2 = 6);

        void collectGarbage();

    protected:
        void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

        int num_matches_thresh1_;
        int num_matches_thresh2_;
        Ptr<FeaturesMatcher> impl_;
    };

.. seealso:: :ocv:class:`detail::FeaturesMatcher`

