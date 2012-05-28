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

detail::FeaturesFinder::operator()
----------------------------------

Finds features in the given image.

.. ocv:function:: void detail::FeaturesFinder::operator ()(const Mat &image, ImageFeatures &features)

.. ocv:function:: void detail::FeaturesFinder::operator ()(const Mat &image, ImageFeatures &features, const std::vector<cv::Rect> &rois)

    :param image: Source image

    :param features: Found features

    :param rois: Regions of interest

.. seealso:: :ocv:struct:`detail::ImageFeatures`, :ocv:class:`Rect_`

detail::FeaturesFinder::collectGarbage
--------------------------------------

Frees unused memory allocated before if there is any.

.. ocv:function:: void detail::FeaturesFinder::collectGarbage()

detail::FeaturesFinder::find
----------------------------

This method must implement features finding logic in order to make the wrappers `detail::FeaturesFinder::operator()`_ work.

.. ocv:function:: void find(const Mat &image, ImageFeatures &features)

    :param image: Source image

    :param features: Found features

.. seealso:: :ocv:struct:`detail::ImageFeatures`

detail::SurfFeaturesFinder
--------------------------
.. ocv:class:: detail::SurfFeaturesFinder : public FeaturesFinder

SURF features finder. ::

    class CV_EXPORTS SurfFeaturesFinder : public FeaturesFinder
    {
    public:
        SurfFeaturesFinder(double hess_thresh = 300., int num_octaves = 3, int num_layers = 4,
                           int num_octaves_descr = /*4*/3, int num_layers_descr = /*2*/4);

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::FeaturesFinder`, :ocv:class:`SURF`

detail::OrbFeaturesFinder
-------------------------
.. ocv:class:: detail::OrbFeaturesFinder : public FeaturesFinder

ORB features finder. ::

    class CV_EXPORTS OrbFeaturesFinder : public FeaturesFinder
    {
    public:
        OrbFeaturesFinder(Size _grid_size = Size(3,1), size_t n_features = 1500,
                          const ORB::CommonParams &detector_params = ORB::CommonParams(1.3f, 5));

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::FeaturesFinder`, :ocv:class:`ORB`

detail::MatchesInfo
-------------------
.. ocv:struct:: detail::MatchesInfo

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
                         const Mat &mask = cv::Mat());

        bool isThreadSafe() const { return is_thread_safe_; }

        virtual void collectGarbage() {}

    protected:
        FeaturesMatcher(bool is_thread_safe = false) : is_thread_safe_(is_thread_safe) {}

        virtual void match(const ImageFeatures &features1, const ImageFeatures &features2,
                           MatchesInfo& matches_info) = 0;

        bool is_thread_safe_;
    };

detail::FeaturesMatcher::operator()
-----------------------------------

Performs images matching.

.. ocv:function:: void detail::FeaturesMatcher::operator ()(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)

    :param features1: First image features

    :param features2: Second image features

    :param matches_info: Found matches

.. ocv:function:: void detail::FeaturesMatcher::operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches, const Mat &mask)

    :param features: Features of the source images

    :param pairwise_matches: Found pairwise matches

    :param mask: Mask indicating which image pairs must be matched

The function is parallelized with the TBB library.

.. seealso:: :ocv:struct:`detail::MatchesInfo`

detail::FeaturesMatcher::isThreadSafe
-------------------------------------

.. ocv:function:: bool detail::FeaturesMatcher::isThreadSafe() const

    :return: True, if it's possible to use the same matcher instance in parallel, false otherwise

detail::FeaturesMatcher::collectGarbage
---------------------------------------

Frees unused memory allocated before if there is any.

.. ocv:function:: void detail::FeaturesMatcher::collectGarbage()

detail::FeaturesMatcher::match
------------------------------

This method must implement matching logic in order to make the wrappers `detail::FeaturesMatcher::operator()`_ work.

.. ocv:function:: void detail::FeaturesMatcher::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)

    :param features1: First image features

    :param features2: Second image features

    :param matches_info: Found matches

detail::BestOf2NearestMatcher
-----------------------------
.. ocv:class:: detail::BestOf2NearestMatcher : public FeaturesMatcher

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

detail::BestOf2NearestMatcher::BestOf2NearestMatcher
----------------------------------------------------

Constructs a "best of 2 nearest" matcher.

.. ocv:function:: detail::BestOf2NearestMatcher::BestOf2NearestMatcher(bool try_use_gpu = false, float match_conf = 0.3f, int num_matches_thresh1 = 6, int num_matches_thresh2 = 6)

    :param try_use_gpu: Should try to use GPU or not

    :param match_conf: Match distances ration threshold

    :param num_matches_thresh1: Minimum number of matches required for the 2D projective transform estimation used in the inliers classification step

    :param num_matches_thresh2: Minimum number of matches required for the 2D projective transform re-estimation on inliers
