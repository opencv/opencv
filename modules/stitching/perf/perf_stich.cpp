#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/opencv_modules.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

#define SURF_MATCH_CONFIDENCE 0.65f
#define ORB_MATCH_CONFIDENCE  0.3f
#define WORK_MEGAPIX 0.6

typedef TestBaseWithParam<string> stitch;
typedef TestBaseWithParam<string> match;
typedef std::tr1::tuple<string, int> matchVector_t;
typedef TestBaseWithParam<matchVector_t> matchVector;

#ifdef HAVE_OPENCV_XFEATURES2D_TODO_FIND_WHY_SURF_IS_NOT_ABLE_TO_STITCH_PANOS
#define TEST_DETECTORS testing::Values("surf", "orb")
#else
#define TEST_DETECTORS testing::Values<string>("orb")
#endif

PERF_TEST_P(stitch, a123, TEST_DETECTORS)
{
    Mat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.png") ) );

    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? Ptr<detail::FeaturesFinder>(new detail::OrbFeaturesFinder())
            : Ptr<detail::FeaturesFinder>(new detail::SurfFeaturesFinder());

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE)
            : makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);
        stitcher.setWarper(makePtr<SphericalWarper>());
        stitcher.setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher.stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, 1182, 50);
    EXPECT_NEAR(pano.size().height, 682, 30);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(stitch, b12, TEST_DETECTORS)
{
    Mat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/b1.png") ) );
    imgs.push_back( imread( getDataPath("stitching/b2.png") ) );

    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? Ptr<detail::FeaturesFinder>(new detail::OrbFeaturesFinder())
            : Ptr<detail::FeaturesFinder>(new detail::SurfFeaturesFinder());

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE)
            : makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);
        stitcher.setWarper(makePtr<SphericalWarper>());
        stitcher.setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher.stitch(imgs, pano);
        stopTimer();
    }

    Mat pano_small;
    if (!pano.empty())
        resize(pano, pano_small, Size(320, 240), 0, 0, INTER_AREA);

    SANITY_CHECK(pano_small, 5);
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

    std::vector<DMatch>& matches = pairwise_matches.matches;
    if (GetParam() == "orb") matches.resize(0);
    for(size_t q = 0; q < matches.size(); ++q)
        if (matches[q].imgIdx < 0) { matches.resize(q); break;}
    SANITY_CHECK_MATCHES(matches);
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


    std::vector<DMatch>& matches = pairwise_matches[detectorName == "surf" ? 1 : 0].matches;
    for(size_t q = 0; q < matches.size(); ++q)
        if (matches[q].imgIdx < 0) { matches.resize(q); break;}
    SANITY_CHECK_MATCHES(matches);
}
