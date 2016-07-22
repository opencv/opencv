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

#ifdef HAVE_OPENCV_XFEATURES2D
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

    EXPECT_NEAR(pano.size().width, 1117, 50);
    EXPECT_NEAR(pano.size().height, 642, 30);

    Mat pano_small;
    resize(pano, pano_small, Size(320, 240), 0, 0, INTER_AREA);

    // results from orb and surf are slightly different (but both of them are good). Regression data
    // are for orb.
    /* transformations are:
    orb:
    [0.99213386, 0.062509097, -351.83731;
     -0.073042989, 1.0615162, -89.869858;
     0.0005330033, -4.0937066e-05, 1]
    surf:
    [1.0034728, 0.022535477, -352.76849;
     -0.080653802, 1.0742083, -89.602058;
     0.0004876224, 0.00012311155, 1]
    */
    if (GetParam() == "orb")
        SANITY_CHECK(pano_small, 5);
    else
        SANITY_CHECK_NOTHING();
}

PERF_TEST_P(stitch, affineS12, TEST_DETECTORS)
{
    Mat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/s1.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/s2.jpg") ) );

    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? Ptr<detail::FeaturesFinder>(new detail::OrbFeaturesFinder())
            : Ptr<detail::FeaturesFinder>(new detail::SurfFeaturesFinder());

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS, false);
        stitcher->setFeaturesFinder(featuresFinder);
        stitcher->setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher->stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, 1815, 10);
    EXPECT_NEAR(pano.size().height, 700, 5);

    SANITY_CHECK_NOTHING();
}
