// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {
namespace {

detail::WaveCorrectKind correctionKind(const std::vector<UMat>& images)
{

    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    stitcher->estimateTransform(images);

    std::vector<Mat> rmats;
    auto cameras = stitcher->cameras();
    for (const auto& camera: cameras)
        rmats.push_back(camera.R);

    return detail::autoDetectWaveCorrectKind(rmats);
}

TEST(WaveCorrection, AutoWaveCorrection)
{
    std::vector<UMat> images(2);
    imread(cvtest::TS::ptr()->get_data_path() + "stitching/s1.jpg").copyTo(images[0]);
    imread(cvtest::TS::ptr()->get_data_path() + "stitching/s2.jpg").copyTo(images[1]);

    EXPECT_EQ(detail::WAVE_CORRECT_HORIZ, correctionKind(images));

    std::vector<UMat> rotated_images(2);
    rotate(images[0], rotated_images[0], cv::ROTATE_90_CLOCKWISE);
    rotate(images[1], rotated_images[1], cv::ROTATE_90_CLOCKWISE);

    EXPECT_EQ(detail::WAVE_CORRECT_VERT, correctionKind(rotated_images));

    rotate(images[0], rotated_images[0], cv::ROTATE_90_COUNTERCLOCKWISE);
    rotate(images[1], rotated_images[1], cv::ROTATE_90_COUNTERCLOCKWISE);

    EXPECT_EQ(detail::WAVE_CORRECT_VERT, correctionKind(rotated_images));

    rotate(images[0], rotated_images[0], cv::ROTATE_180);
    rotate(images[1], rotated_images[1], cv::ROTATE_180);

    EXPECT_EQ(detail::WAVE_CORRECT_HORIZ, correctionKind(rotated_images));
}

} // namespace
} // namespace opencv_test
