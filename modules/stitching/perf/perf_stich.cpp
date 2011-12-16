#include "perf_precomp.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/flann.hpp"

using namespace std;
using namespace cv;
using namespace perf;

#define SURF_MATCH_CONFIDENCE 0.65f
#define ORB_MATCH_CONFIDENCE  0.3f
#define WORK_MEGAPIX 0.6

typedef TestBaseWithParam<String> stitch;
typedef TestBaseWithParam<String> match;

PERF_TEST_P( stitch, a123, testing::Values("surf", "orb"))
{
    Mat pano;
    
    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.jpg") ) );

    Stitcher::Status status;
    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? (detail::FeaturesFinder*)new detail::OrbFeaturesFinder()
            : (detail::FeaturesFinder*)new detail::SurfFeaturesFinder();

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? new detail::BestOf2NearestMatcher(false, ORB_MATCH_CONFIDENCE)
            : new detail::BestOf2NearestMatcher(false, SURF_MATCH_CONFIDENCE);

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);
        stitcher.setWarper(new CylindricalWarper());
        stitcher.setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        status = stitcher.stitch(imgs, pano);
        stopTimer();
    }
}

PERF_TEST_P( stitch, b12, testing::Values("surf", "orb"))
{
    Mat pano;
    
    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/b1.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/b2.jpg") ) );

    Stitcher::Status status;
    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? (detail::FeaturesFinder*)new detail::OrbFeaturesFinder()
            : (detail::FeaturesFinder*)new detail::SurfFeaturesFinder();

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? new detail::BestOf2NearestMatcher(false, ORB_MATCH_CONFIDENCE)
            : new detail::BestOf2NearestMatcher(false, SURF_MATCH_CONFIDENCE);

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);
        stitcher.setWarper(new CylindricalWarper());
        stitcher.setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        status = stitcher.stitch(imgs, pano);
        stopTimer();
    }
}

PERF_TEST_P( match, bestOf2Nearest, testing::Values("surf", "orb"))
{
    Mat img1, img1_full = imread( getDataPath("stitching/b1.jpg") );
    Mat img2, img2_full = imread( getDataPath("stitching/b2.jpg") );
    float scale1 = std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1);
    resize(img2_full, img2, Size(), scale2, scale2);

    Ptr<detail::FeaturesFinder> finder;
    Ptr<detail::FeaturesMatcher> matcher;
    if (GetParam() == "surf")
    {
        finder = new detail::SurfFeaturesFinder();
        matcher = new detail::BestOf2NearestMatcher(false, SURF_MATCH_CONFIDENCE);
    }
    else if (GetParam() == "orb")
    {
        finder = new detail::OrbFeaturesFinder();
        matcher = new detail::BestOf2NearestMatcher(false, ORB_MATCH_CONFIDENCE);
    }
    else
    {
        FAIL() << "Unknown 2D features type: " << GetParam();
    }

    detail::ImageFeatures features1, features2;
    (*finder)(img1, features1);
    (*finder)(img2, features2);

    detail::MatchesInfo pairwise_matches;

    declare.in(features1.descriptors, features2.descriptors)
           .iterations(100);

    while(next())
    {
        cvflann::seed_random(42);//for predictive FlannBasedMatcher
        startTimer();
        (*matcher)(features1, features2, pairwise_matches);
        stopTimer();
        matcher->collectGarbage();
    }
}
