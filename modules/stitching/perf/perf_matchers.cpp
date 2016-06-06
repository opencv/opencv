#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef TestBaseWithParam<size_t> FeaturesFinderVec;

#define NUMBER_IMAGES testing::Values(1, 5, 20)

PERF_TEST_P(FeaturesFinderVec, ParallelFeaturesFinder, NUMBER_IMAGES)
{
    Mat img = imread( getDataPath("stitching/a1.png") );
    vector<Mat> imgs(GetParam(), img);
    vector<detail::ImageFeatures> features(imgs.size());

    Ptr<detail::FeaturesFinder2> featuresFinder = makePtr<detail::OrbFeaturesFinder2>();

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
