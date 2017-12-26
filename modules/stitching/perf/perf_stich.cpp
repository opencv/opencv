#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"

#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::tuple;
using std::tr1::get;

#define SURF_MATCH_CONFIDENCE 0.65f
#define ORB_MATCH_CONFIDENCE  0.3f
#define WORK_MEGAPIX 0.6

typedef TestBaseWithParam<string> stitch;
typedef TestBaseWithParam<tuple<string, string> > stitchDatasets;

#ifdef HAVE_OPENCV_XFEATURES2D
#define TEST_DETECTORS testing::Values("surf", "orb", "akaze")
#else
#define TEST_DETECTORS testing::Values("orb", "akaze")
#endif
#define AFFINE_DATASETS testing::Values("s", "budapest", "newspaper", "prague")

PERF_TEST_P(stitch, a123, TEST_DETECTORS)
{
    Mat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.png") ) );

    Ptr<detail::FeaturesFinder> featuresFinder = getFeatureFinder(GetParam());

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

    Ptr<detail::FeaturesFinder> featuresFinder = getFeatureFinder(GetParam());

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

    EXPECT_NEAR(pano.size().width, 1117, GetParam() == "surf" ? 100 : 50);
    EXPECT_NEAR(pano.size().height, 642, GetParam() == "surf" ? 60 : 30);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(stitchDatasets, affine, testing::Combine(AFFINE_DATASETS, TEST_DETECTORS))
{
    string dataset = get<0>(GetParam());
    string detector = get<1>(GetParam());

    Mat pano;
    vector<Mat> imgs;
    int width, height, allowed_diff = 20;
    Ptr<detail::FeaturesFinder> featuresFinder = getFeatureFinder(detector);

    if(dataset == "budapest")
    {
        imgs.push_back(imread(getDataPath("stitching/budapest1.jpg")));
        imgs.push_back(imread(getDataPath("stitching/budapest2.jpg")));
        imgs.push_back(imread(getDataPath("stitching/budapest3.jpg")));
        imgs.push_back(imread(getDataPath("stitching/budapest4.jpg")));
        imgs.push_back(imread(getDataPath("stitching/budapest5.jpg")));
        imgs.push_back(imread(getDataPath("stitching/budapest6.jpg")));
        width = 2313;
        height = 1158;
        // this dataset is big, the results between surf and orb differ slightly,
        // but both are still good
        allowed_diff = 50;
    }
    else if (dataset == "newspaper")
    {
        imgs.push_back(imread(getDataPath("stitching/newspaper1.jpg")));
        imgs.push_back(imread(getDataPath("stitching/newspaper2.jpg")));
        imgs.push_back(imread(getDataPath("stitching/newspaper3.jpg")));
        imgs.push_back(imread(getDataPath("stitching/newspaper4.jpg")));
        width = 1791;
        height = 1136;
        // we need to boost ORB number of features to be able to stitch this dataset
        // SURF works just fine with default settings
        if(detector == "orb")
            featuresFinder = makePtr<detail::OrbFeaturesFinder>(Size(3,1), 3000);
    }
    else if (dataset == "prague")
    {
        imgs.push_back(imread(getDataPath("stitching/prague1.jpg")));
        imgs.push_back(imread(getDataPath("stitching/prague2.jpg")));
        width = 983;
        height = 1759;
    }
    else // dataset == "s"
    {
        imgs.push_back(imread(getDataPath("stitching/s1.jpg")));
        imgs.push_back(imread(getDataPath("stitching/s2.jpg")));
        width = 1815;
        height = 700;
    }

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS, false);
        stitcher->setFeaturesFinder(featuresFinder);

        if (cv::ocl::useOpenCL())
            cv::theRNG() = cv::RNG(12345); // prevent fails of Windows OpenCL builds (see #8294)

        startTimer();
        stitcher->stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, width, allowed_diff);
    EXPECT_NEAR(pano.size().height, height, allowed_diff);

    SANITY_CHECK_NOTHING();
}
