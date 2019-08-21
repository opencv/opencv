#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/flann.hpp"

namespace opencv_test
{
using namespace perf;

typedef TestBaseWithParam<size_t> FeaturesFinderVec;
typedef TestBaseWithParam<string> match;
typedef tuple<string, int> matchVector_t;
typedef TestBaseWithParam<matchVector_t> matchVector;

#define NUMBER_IMAGES testing::Values(1, 5, 20)
#define SURF_MATCH_CONFIDENCE 0.65f
#define ORB_MATCH_CONFIDENCE  0.3f
#define WORK_MEGAPIX 0.6

#ifdef HAVE_OPENCV_XFEATURES2D
#define TEST_DETECTORS testing::Values("surf", "orb")
#else
#define TEST_DETECTORS testing::Values<string>("orb")
#endif

PERF_TEST_P(FeaturesFinderVec, ParallelFeaturesFinder, NUMBER_IMAGES)
{
    Mat img = imread( getDataPath("stitching/a1.png") );
    vector<Mat> imgs(GetParam(), img);
    vector<detail::ImageFeatures> features(imgs.size());

    Ptr<detail::FeaturesFinder> featuresFinder = makePtr<detail::OrbFeaturesFinder>();

    TEST_CYCLE()
    {
        (*featuresFinder)(imgs, features);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(FeaturesFinderVec, SerialFeaturesFinder, NUMBER_IMAGES)
{
    Mat img = imread( getDataPath("stitching/a1.png") );
    vector<Mat> imgs(GetParam(), img);
    vector<detail::ImageFeatures> features(imgs.size());

    Ptr<detail::FeaturesFinder> featuresFinder = makePtr<detail::OrbFeaturesFinder>();

    TEST_CYCLE()
    {
        for (size_t i = 0; i < imgs.size(); ++i)
            (*featuresFinder)(imgs[i], features[i]);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( match, bestOf2Nearest, TEST_DETECTORS)
{
    Mat img1, img1_full = imread( getDataPath("stitching/boat1.jpg") );
    Mat img2, img2_full = imread( getDataPath("stitching/boat2.jpg") );
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1, INTER_LINEAR_EXACT);
    resize(img2_full, img2, Size(), scale2, scale2, INTER_LINEAR_EXACT);

    Ptr<detail::FeaturesFinder> finder;
    Ptr<detail::FeaturesMatcher> matcher;
    if (GetParam() == "surf")
    {
        finder = makePtr<detail::SurfFeaturesFinder>();
        matcher = makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);
    }
    else if (GetParam() == "orb")
    {
        finder = makePtr<detail::OrbFeaturesFinder>();
        matcher = makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE);
    }
    else
    {
        FAIL() << "Unknown 2D features type: " << GetParam();
    }

    detail::ImageFeatures features1, features2;
    (*finder)(img1, features1);
    (*finder)(img2, features2);

    detail::MatchesInfo pairwise_matches;

    declare.in(features1.descriptors, features2.descriptors);

    while(next())
    {
        cvflann::seed_random(42);//for predictive FlannBasedMatcher
        startTimer();
        (*matcher)(features1, features2, pairwise_matches);
        stopTimer();
        matcher->collectGarbage();
    }

    Mat dist (pairwise_matches.H, Range::all(), Range(2, 3));
    Mat R (pairwise_matches.H, Range::all(), Range(0, 2));
    // separate transform matrix, use lower error on rotations
    SANITY_CHECK(dist, 3., ERROR_ABSOLUTE);
    SANITY_CHECK(R, .06, ERROR_ABSOLUTE);
}

PERF_TEST_P( matchVector, bestOf2NearestVectorFeatures, testing::Combine(
                 TEST_DETECTORS,
                 testing::Values(2, 4, 8))
             )
{
    Mat img1, img1_full = imread( getDataPath("stitching/boat1.jpg") );
    Mat img2, img2_full = imread( getDataPath("stitching/boat2.jpg") );
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1, INTER_LINEAR_EXACT);
    resize(img2_full, img2, Size(), scale2, scale2, INTER_LINEAR_EXACT);

    Ptr<detail::FeaturesFinder> finder;
    Ptr<detail::FeaturesMatcher> matcher;
    string detectorName = get<0>(GetParam());
    int featuresVectorSize = get<1>(GetParam());
    if (detectorName == "surf")
    {
        finder = makePtr<detail::SurfFeaturesFinder>();
        matcher = makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);
    }
    else if (detectorName == "orb")
    {
        finder = makePtr<detail::OrbFeaturesFinder>();
        matcher = makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE);
    }
    else
    {
        FAIL() << "Unknown 2D features type: " << get<0>(GetParam());
    }

    detail::ImageFeatures features1, features2;
    (*finder)(img1, features1);
    (*finder)(img2, features2);
    vector<detail::ImageFeatures> features;
    vector<detail::MatchesInfo> pairwise_matches;
    for(int i = 0; i < featuresVectorSize/2; i++)
    {
        features.push_back(features1);
        features.push_back(features2);
    }

    declare.time(200);
    while(next())
    {
        cvflann::seed_random(42);//for predictive FlannBasedMatcher
        startTimer();
        (*matcher)(features, pairwise_matches);
        stopTimer();
        matcher->collectGarbage();
    }

    size_t matches_count = 0;
    for (size_t i = 0; i < pairwise_matches.size(); ++i)
    {
        if (pairwise_matches[i].src_img_idx < 0)
            continue;

        EXPECT_GT(pairwise_matches[i].matches.size(), 95u);
        EXPECT_FALSE(pairwise_matches[i].H.empty());
        ++matches_count;
    }

    EXPECT_GT(matches_count, 0u);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( match, affineBestOf2Nearest, TEST_DETECTORS)
{
    Mat img1, img1_full = imread( getDataPath("stitching/s1.jpg") );
    Mat img2, img2_full = imread( getDataPath("stitching/s2.jpg") );
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1, INTER_LINEAR_EXACT);
    resize(img2_full, img2, Size(), scale2, scale2, INTER_LINEAR_EXACT);

    Ptr<detail::FeaturesFinder> finder;
    Ptr<detail::FeaturesMatcher> matcher;
    if (GetParam() == "surf")
    {
        finder = makePtr<detail::SurfFeaturesFinder>();
        matcher = makePtr<detail::AffineBestOf2NearestMatcher>(false, false, SURF_MATCH_CONFIDENCE);
    }
    else if (GetParam() == "orb")
    {
        finder = makePtr<detail::OrbFeaturesFinder>();
        matcher = makePtr<detail::AffineBestOf2NearestMatcher>(false, false, ORB_MATCH_CONFIDENCE);
    }
    else
    {
        FAIL() << "Unknown 2D features type: " << GetParam();
    }

    detail::ImageFeatures features1, features2;
    (*finder)(img1, features1);
    (*finder)(img2, features2);

    detail::MatchesInfo pairwise_matches;

    declare.in(features1.descriptors, features2.descriptors);

    while(next())
    {
        cvflann::seed_random(42);//for predictive FlannBasedMatcher
        startTimer();
        (*matcher)(features1, features2, pairwise_matches);
        stopTimer();
        matcher->collectGarbage();
    }

    // separate rotation and translation in transform matrix
    Mat T (pairwise_matches.H, Range(0, 2), Range(2, 3));
    Mat R (pairwise_matches.H, Range(0, 2), Range(0, 2));
    Mat h (pairwise_matches.H, Range(2, 3), Range::all());
    SANITY_CHECK(T, 5, ERROR_ABSOLUTE); // allow 5 pixels diff in translations
    SANITY_CHECK(R, .01, ERROR_ABSOLUTE); // rotations must be more precise
    // last row should be precisely (0, 0, 1) as it is just added for representation in homogeneous
    // coordinates
    EXPECT_DOUBLE_EQ(h.at<double>(0), 0.);
    EXPECT_DOUBLE_EQ(h.at<double>(1), 0.);
    EXPECT_DOUBLE_EQ(h.at<double>(2), 1.);
}

PERF_TEST_P( matchVector, affineBestOf2NearestVectorFeatures, testing::Combine(
                 TEST_DETECTORS,
                 testing::Values(2, 4, 8))
             )
{
    Mat img1, img1_full = imread( getDataPath("stitching/s1.jpg") );
    Mat img2, img2_full = imread( getDataPath("stitching/s2.jpg") );
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1, INTER_LINEAR_EXACT);
    resize(img2_full, img2, Size(), scale2, scale2, INTER_LINEAR_EXACT);

    Ptr<detail::FeaturesFinder> finder;
    Ptr<detail::FeaturesMatcher> matcher;
    string detectorName = get<0>(GetParam());
    int featuresVectorSize = get<1>(GetParam());
    if (detectorName == "surf")
    {
        finder = makePtr<detail::SurfFeaturesFinder>();
        matcher = makePtr<detail::AffineBestOf2NearestMatcher>(false, false, SURF_MATCH_CONFIDENCE);
    }
    else if (detectorName == "orb")
    {
        finder = makePtr<detail::OrbFeaturesFinder>();
        matcher = makePtr<detail::AffineBestOf2NearestMatcher>(false, false, ORB_MATCH_CONFIDENCE);
    }
    else
    {
        FAIL() << "Unknown 2D features type: " << get<0>(GetParam());
    }

    detail::ImageFeatures features1, features2;
    (*finder)(img1, features1);
    (*finder)(img2, features2);
    vector<detail::ImageFeatures> features;
    vector<detail::MatchesInfo> pairwise_matches;
    for(int i = 0; i < featuresVectorSize/2; i++)
    {
        features.push_back(features1);
        features.push_back(features2);
    }

    declare.time(200);
    while(next())
    {
        cvflann::seed_random(42);//for predictive FlannBasedMatcher
        startTimer();
        (*matcher)(features, pairwise_matches);
        stopTimer();
        matcher->collectGarbage();
    }

    size_t matches_count = 0;
    for (size_t i = 0; i < pairwise_matches.size(); ++i)
    {
        if (pairwise_matches[i].src_img_idx < 0)
            continue;

        EXPECT_GT(pairwise_matches[i].matches.size(), (size_t)300);
        EXPECT_FALSE(pairwise_matches[i].H.empty());
        ++matches_count;
    }

    EXPECT_TRUE(matches_count > 0);

    SANITY_CHECK_NOTHING();
}

} // namespace
