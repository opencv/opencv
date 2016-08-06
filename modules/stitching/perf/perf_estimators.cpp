#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::tuple;
using std::tr1::get;

typedef TestBaseWithParam<tuple<string, string> > bundleAdjuster;

#ifdef HAVE_OPENCV_XFEATURES2D
#define TEST_DETECTORS testing::Values("surf", "orb")
#else
#define TEST_DETECTORS testing::Values<string>("orb")
#endif
#define WORK_MEGAPIX 0.6
#define AFFINE_FUNCTIONS testing::Values("affinePartial", "affine")

PERF_TEST_P(bundleAdjuster, affine, testing::Combine(TEST_DETECTORS, AFFINE_FUNCTIONS))
{
    Mat img1, img1_full = imread(getDataPath("stitching/s1.jpg"));
    Mat img2, img2_full = imread(getDataPath("stitching/s2.jpg"));
    float scale1 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img1_full.total()));
    float scale2 = (float)std::min(1.0, sqrt(WORK_MEGAPIX * 1e6 / img2_full.total()));
    resize(img1_full, img1, Size(), scale1, scale1);
    resize(img2_full, img2, Size(), scale2, scale2);

    string detector = get<0>(GetParam());
    string affine_fun = get<1>(GetParam());

    Ptr<detail::FeaturesFinder> finder;
    Ptr<detail::FeaturesMatcher> matcher;
    Ptr<detail::BundleAdjusterBase> bundle_adjuster;
    if (detector == "surf")
        finder = makePtr<detail::SurfFeaturesFinder>();
    else if (detector == "orb")
        finder = makePtr<detail::OrbFeaturesFinder>();
    if (affine_fun == "affinePartial")
    {
        matcher = makePtr<detail::AffineBestOf2NearestMatcher>(false);
        bundle_adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    }
    else if (affine_fun == "affine")
    {
        matcher = makePtr<detail::AffineBestOf2NearestMatcher>(true);
        bundle_adjuster = makePtr<detail::BundleAdjusterAffine>();
    }
    Ptr<detail::Estimator> estimator = makePtr<detail::AffineBasedEstimator>();

    std::vector<Mat> images;
    images.push_back(img1), images.push_back(img2);
    std::vector<detail::ImageFeatures> features;
    std::vector<detail::MatchesInfo> pairwise_matches;
    std::vector<detail::CameraParams> cameras;
    std::vector<detail::CameraParams> cameras2;

    (*finder)(images, features);
    (*matcher)(features, pairwise_matches);
    if (!(*estimator)(features, pairwise_matches, cameras))
        FAIL() << "estimation failed. this should never happen.";
    // this is currently required
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    cameras2 = cameras;
    bool success = true;
    while(next())
    {
        cameras = cameras2; // revert cameras back to original initial guess
        startTimer();
        success = (*bundle_adjuster)(features, pairwise_matches, cameras);
        stopTimer();
    }

    EXPECT_TRUE(success);
    EXPECT_TRUE(cameras.size() == 2);

    // fist camera should be just identity
    Mat &first = cameras[0].R;
    SANITY_CHECK(first, 1e-3, ERROR_ABSOLUTE);
    // second camera should be the estimated transform between images
    // separate rotation and translation in transform matrix
    Mat T_second (cameras[1].R, Range(0, 2), Range(2, 3));
    Mat R_second (cameras[1].R, Range(0, 2), Range(0, 2));
    Mat h (cameras[1].R, Range(2, 3), Range::all());
    SANITY_CHECK(T_second, 5, ERROR_ABSOLUTE); // allow 5 pixels diff in translations
    SANITY_CHECK(R_second, .01, ERROR_ABSOLUTE); // rotations must be more precise
    // last row should be precisely (0, 0, 1) as it is just added for representation in homogeneous
    // coordinates
    EXPECT_TRUE(h.type() == CV_32F);
    EXPECT_FLOAT_EQ(h.at<float>(0), 0.f);
    EXPECT_FLOAT_EQ(h.at<float>(1), 0.f);
    EXPECT_FLOAT_EQ(h.at<float>(2), 1.f);
}
