#include "perf_precomp.hpp"
#include "perf_feature2d.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<int, int, bool, string> Fast_Params_t;
typedef perf::TestBaseWithParam<Fast_Params_t> Fast_Params;

PERF_TEST_P(Fast_Params, detect,
    testing::Combine(
        testing::Values(20,30,100),                   // threshold
        testing::Values(
            // (int)FastFeatureDetector::TYPE_5_8,
            // (int)FastFeatureDetector::TYPE_7_12,
            (int)FastFeatureDetector::TYPE_9_16       // detector_type
        ),
        testing::Bool(),                              // nonmaxSuppression
        testing::Values("cv/inpaint/orig.png",
                        "cv/cameracalibration/chess9.png")
    ))
{
    int threshold_p = get<0>(GetParam());
    int type_p = get<1>(GetParam());
    bool nonmaxSuppression_p = get<2>(GetParam());
    string filename = getDataPath(get<3>(GetParam()));

    Mat img = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty()) << "Failed to load image: " << filename;

    vector<KeyPoint> keypoints;

    int64 t1 = cv::getTickCount();
    for(int i = 0; i < 10; i++) {
        FAST(img, keypoints, threshold_p, nonmaxSuppression_p, (FastFeatureDetector::DetectorType)type_p);
    }
    int64 t2 = cv::getTickCount();
    declare.in(img);
    TEST_CYCLE()
    {
        FAST(img, keypoints, threshold_p, nonmaxSuppression_p, (FastFeatureDetector::DetectorType)type_p);
    }

    // SANITY_CHECK_KEYPOINTS(keypoints);
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
