#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"

#include "opencv2/core/ocl.hpp"

namespace opencv_test
{
using namespace perf;

#define SURF_MATCH_CONFIDENCE 0.65f
#define ORB_MATCH_CONFIDENCE  0.3f
#define WORK_MEGAPIX 0.6

typedef TestBaseWithParam<string> stitch;
typedef TestBaseWithParam<int> stitchExposureCompensation;
typedef TestBaseWithParam<tuple<string, string> > stitchDatasets;
typedef TestBaseWithParam<tuple<string, int>> stitchExposureCompMultiFeed;

#if defined(HAVE_OPENCV_XFEATURES2D) && defined(OPENCV_ENABLE_NONFREE)
#define TEST_DETECTORS testing::Values("surf", "orb", "akaze")
#else
#define TEST_DETECTORS testing::Values("orb")
#endif
#define TEST_EXP_COMP_BS testing::Values(32, 16, 12, 10, 8)
#define TEST_EXP_COMP_NR_FEED testing::Values(1, 2, 3, 4, 5)
#define TEST_EXP_COMP_MODE testing::Values("gain", "channels", "blocks_gain", "blocks_channels")
#define AFFINE_DATASETS testing::Values("s", "budapest", "newspaper", "prague")

PERF_TEST_P(stitch, a123, TEST_DETECTORS)
{
    Mat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.png") ) );

    Ptr<Feature2D> featuresFinder = getFeatureFinder(GetParam());

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE)
            : makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Ptr<Stitcher> stitcher = Stitcher::create();
        stitcher->setFeaturesFinder(featuresFinder);
        stitcher->setFeaturesMatcher(featuresMatcher);
        stitcher->setWarper(makePtr<SphericalWarper>());
        stitcher->setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher->stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, 1182, 50);
    EXPECT_NEAR(pano.size().height, 682, 30);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(stitchExposureCompensation, a123, TEST_EXP_COMP_BS)
{
    Mat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.png") ) );

    int bs = GetParam();

    declare.time(30 * 10).iterations(10);

    while(next())
    {
        Ptr<Stitcher> stitcher = Stitcher::create();
        stitcher->setWarper(makePtr<SphericalWarper>());
        stitcher->setRegistrationResol(WORK_MEGAPIX);
        stitcher->setExposureCompensator(
            makePtr<detail::BlocksGainCompensator>(bs, bs));

        startTimer();
        stitcher->stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, 1182, 50);
    EXPECT_NEAR(pano.size().height, 682, 30);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(stitchExposureCompMultiFeed, a123, testing::Combine(TEST_EXP_COMP_MODE, TEST_EXP_COMP_NR_FEED))
{
    const int block_size = 32;
    Mat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.png") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.png") ) );

    string mode = get<0>(GetParam());
    int nr_feeds = get<1>(GetParam());

    declare.time(30 * 10).iterations(10);

    Ptr<detail::ExposureCompensator> exp_comp;
    if (mode == "gain")
        exp_comp = makePtr<detail::GainCompensator>(nr_feeds);
    else if (mode == "channels")
        exp_comp = makePtr<detail::ChannelsCompensator>(nr_feeds);
    else if (mode == "blocks_gain")
        exp_comp = makePtr<detail::BlocksGainCompensator>(block_size, block_size, nr_feeds);
    else if (mode == "blocks_channels")
        exp_comp = makePtr<detail::BlocksChannelsCompensator>(block_size, block_size, nr_feeds);

    while(next())
    {
        Ptr<Stitcher> stitcher = Stitcher::create();
        stitcher->setWarper(makePtr<SphericalWarper>());
        stitcher->setRegistrationResol(WORK_MEGAPIX);
        stitcher->setExposureCompensator(exp_comp);

        startTimer();
        stitcher->stitch(imgs, pano);
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

    Ptr<Feature2D> featuresFinder = getFeatureFinder(GetParam());

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE)
            : makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);

    declare.time(30 * 20).iterations(20);

    while(next())
    {
        Ptr<Stitcher> stitcher = Stitcher::create();
        stitcher->setFeaturesFinder(featuresFinder);
        stitcher->setFeaturesMatcher(featuresMatcher);
        stitcher->setWarper(makePtr<SphericalWarper>());
        stitcher->setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher->stitch(imgs, pano);
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
    Ptr<Feature2D> featuresFinder = getFeatureFinder(detector);

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
        // we need to boost ORB number of features to be able to stitch this dataset
        // SURF works just fine with default settings
        if(detector == "orb")
            featuresFinder = ORB::create(1500);
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
            featuresFinder = ORB::create(3000);
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
        Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
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

} // namespace
