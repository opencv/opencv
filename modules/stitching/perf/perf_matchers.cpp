#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/flann.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef TestBaseWithParam<size_t> FeaturesFinderVec;
typedef TestBaseWithParam<string> match;
typedef std::tr1::tuple<string, int> matchVector_t;
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
    Mat img1, img1_full = imread( getDataPath("stitching/b1.png") );
    Mat img2, img2_full = imread( getDataPath("stitching/b2.png") );
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1);
    resize(img2_full, img2, Size(), scale2, scale2);

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

    Mat& estimated_transform = pairwise_matches.H;
    SANITY_CHECK(estimated_transform, .1, ERROR_RELATIVE);
}

PERF_TEST_P( matchVector, bestOf2NearestVectorFeatures, testing::Combine(
                 TEST_DETECTORS,
                 testing::Values(2, 4, 8))
             )
{
    Mat img1, img1_full = imread( getDataPath("stitching/b1.png") );
    Mat img2, img2_full = imread( getDataPath("stitching/b2.png") );
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1);
    resize(img2_full, img2, Size(), scale2, scale2);

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

        EXPECT_TRUE(pairwise_matches[i].matches.size() > 10);
        EXPECT_FALSE(pairwise_matches[i].H.empty());
        ++matches_count;
    }

    EXPECT_TRUE(matches_count > 0);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( match, affineBestOf2Nearest, TEST_DETECTORS)
{
    Mat img1, img1_full = imread( getDataPath("stitching/s1.jpg") );
    Mat img2, img2_full = imread( getDataPath("stitching/s2.jpg") );
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1);
    resize(img2_full, img2, Size(), scale2, scale2);

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

    Mat& estimated_transform = pairwise_matches.H;
    SANITY_CHECK(estimated_transform, .02, ERROR_RELATIVE);
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
    resize(img1_full, img1, Size(), scale1, scale1);
    resize(img2_full, img2, Size(), scale2, scale2);

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

        EXPECT_TRUE(pairwise_matches[i].matches.size() > 400);
        EXPECT_FALSE(pairwise_matches[i].H.empty());
        ++matches_count;
    }

    EXPECT_TRUE(matches_count > 0);

    SANITY_CHECK_NOTHING();
}
