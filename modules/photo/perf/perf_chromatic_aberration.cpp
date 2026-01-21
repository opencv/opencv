// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{

PERF_TEST(ChromaticAberration, CorrectChromaticAberration)
{
    std::string calib_file = getDataPath("cv/cameracalibration/chromatic_aberration/ca_photo_calib.yaml");
    std::string image_file = getDataPath("cv/cameracalibration/chromatic_aberration/ca_photo.png");

    cv::Mat src = cv::imread(image_file);
    ASSERT_FALSE(src.empty()) << "Could not load input image";
    ASSERT_EQ(src.type(), CV_8UC3);

    cv::Mat coeffMat;
    int degree = -1;
    Size calib_size = {-1,-1};

    FileStorage fs(calib_file, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_NO_THROW(cv::loadChromaticAberrationParams(fs.root(), coeffMat, calib_size, degree));

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::correctChromaticAberration(src, coeffMat, dst, calib_size, degree);
    }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
