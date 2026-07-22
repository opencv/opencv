#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<std::string, cv::Size> String_Size_t;
typedef perf::TestBaseWithParam<String_Size_t> String_Size;

PERF_TEST_P(String_Size, asymm_circles_grid, testing::Values(
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles1.png", Size(7,13)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles2.png", Size(7,13)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles3.png", Size(7,13)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles4.png", Size(5,5)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles5.png", Size(5,5)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles6.png", Size(5,5)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles7.png", Size(3,9)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles8.png", Size(3,9)),
                String_Size_t("cv/cameracalibration/asymmetric_circles/acircles9.png", Size(3,9))
                )
            )
{
    string filename = getDataPath(get<0>(GetParam()));
    Size gridSize = get<1>(GetParam());

    Mat frame = imread(filename);
    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    vector<Point2f> ptvec;
    ptvec.resize(gridSize.area());

    cvtColor(frame, frame, COLOR_BGR2GRAY);

    declare.in(frame).out(ptvec);

    TEST_CYCLE() ASSERT_TRUE(findCirclesGrid(frame, gridSize, ptvec, CALIB_CB_CLUSTERING | CALIB_CB_ASYMMETRIC_GRID));

    SANITY_CHECK(ptvec, 2);
}

// Perf test using synthetic keypoints (no image I/O). Exercises the RNG and
// findLongestPath code paths directly with a pre-detected point set.
typedef perf::TestBaseWithParam<cv::Size> CirclesGrid_RNG_Size;

PERF_TEST_P(CirclesGrid_RNG_Size, detect_keypoints_symmetric,
            testing::Values(cv::Size(6, 5), cv::Size(8, 6), cv::Size(10, 8), cv::Size(15, 12), cv::Size(20, 15)))
{
    const cv::Size patternSize = GetParam();
    const float spacing = 30.f;

    std::vector<cv::Point2f> pts;
    pts.reserve(patternSize.area());
    for (int r = 0; r < patternSize.height; r++)
        for (int c = 0; c < patternSize.width; c++)
            pts.push_back(cv::Point2f(c * spacing, r * spacing));

    // Shuffle so the detector works from an unordered set, same as real use.
    cv::RNG& rng = cv::theRNG();
    for (int k = (int)pts.size() - 1; k > 0; k--)
        std::swap(pts[k], pts[rng.uniform(0, k + 1)]);

    std::vector<cv::Point2f> centers;
    centers.resize(patternSize.area());
    declare.in(cv::Mat(pts)).out(centers);

    TEST_CYCLE() ASSERT_TRUE(findCirclesGrid(cv::Mat(pts), patternSize, centers,
                                             CALIB_CB_SYMMETRIC_GRID, cv::Ptr<cv::FeatureDetector>()));

    SANITY_CHECK_NOTHING();
}

} // namespace
