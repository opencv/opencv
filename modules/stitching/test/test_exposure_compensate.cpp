// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {
namespace {

double minPSNR(UMat src1, UMat src2)
{
    std::vector<UMat> src1_channels, src2_channels;
    split(src1, src1_channels);
    split(src2, src2_channels);

    double psnr = cvtest::PSNR(src1_channels[0], src2_channels[0]);
    psnr = std::min(psnr, cvtest::PSNR(src1_channels[1], src2_channels[1]));
    return std::min(psnr, cvtest::PSNR(src1_channels[2], src2_channels[2]));
}

TEST(ExposureCompensate, SimilarityThreshold)
{
    UMat source;
    imread(cvtest::TS::ptr()->get_data_path() + "stitching/s1.jpg").copyTo(source);

    UMat image1 = source.clone();
    UMat image2 = source.clone();

    // Add a big artifact
    image2(Rect(150, 150, 100, 100)).setTo(Scalar(0, 0, 255));

    UMat mask(image1.size(), CV_8U);
    mask.setTo(255);

    detail::BlocksChannelsCompensator compensator;
    compensator.setNrGainsFilteringIterations(0); // makes it more clear

    // Feed the compensator, image 1 and 2 are perfectly
    // identical, except for the red artifact in image 2
    // Apart from that artifact, there is no exposure to compensate
    compensator.setSimilarityThreshold(1);
    uchar xff = 255;
    compensator.feed(
        {{}, {}},
        {image1, image2},
        {{mask, xff}, {mask, xff}}
    );
    // Verify that the artifact in image 2 did create
    // an artifact in image1 during the exposure compensation
    UMat image1_result = image1.clone();
    compensator.apply(0, {}, image1_result, mask);
    double psnr_no_similarity_mask = minPSNR(image1, image1_result);
    EXPECT_LT(psnr_no_similarity_mask, 45);

    // Add a similarity threshold and verify that
    // the artifact in image1 is gone
    compensator.setSimilarityThreshold(0.1);
    compensator.feed(
        {{}, {}},
        {image1, image2},
        {{mask, xff}, {mask, xff}}
    );
    image1_result = image1.clone();
    compensator.apply(0, {}, image1_result, mask);
    double psnr_similarity_mask = minPSNR(image1, image1_result);
    EXPECT_GT(psnr_similarity_mask, 300);
}

} // namespace
} // namespace opencv_test
