#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{
PERF_TEST(Photo, ChromaticAberrationCorrector)
{
    std::string calib_file = getDataPath("cv/cameracalibration/chromatic_aberration/ca_photo_calib.yaml");
    std::string image_file = getDataPath("cv/cameracalibration/chromatic_aberration/ca_photo.png");

    Mat src = imread(image_file);
    ASSERT_FALSE(src.empty()) << "Could not load input image";

    cv::ChromaticAberrationCorrector corrector(calib_file);

    Mat dst;

    TEST_CYCLE()
    {
        dst = corrector.correctImage(src);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST(Photo, CorrectChromaticAberrationFunction)
{
    std::string calib_file = getDataPath("cv/cameracalibration/chromatic_aberration/calib_result.yaml");
    std::string image_file = getDataPath("cv/cameracalibration/chromatic_aberration/ca_photo.png");

    Mat src = imread(image_file);
    ASSERT_FALSE(src.empty()) << "Could not load input image";

    TEST_CYCLE()
    {
        Mat dst = cv::correctChromaticAberration(src, calib_file);
        (void)dst;
    }

    SANITY_CHECK_NOTHING();
}
} // namespace
} // namespace opencv_test
