// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Features2d_AKAZE, detect_and_compute_split)
{
    Mat testImg(100, 100, CV_8U);
    RNG rng(101);
    rng.fill(testImg, RNG::UNIFORM, Scalar(0), Scalar(255), true);

    Ptr<Feature2D> ext = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 1, 1, KAZE::DIFF_PM_G2);
    vector<KeyPoint> detAndCompKps;
    Mat desc;
    ext->detectAndCompute(testImg, noArray(), detAndCompKps, desc);

    vector<KeyPoint> detKps;
    ext->detect(testImg, detKps);

    ASSERT_EQ(detKps.size(), detAndCompKps.size());

    for(size_t i = 0; i < detKps.size(); i++)
        ASSERT_EQ(detKps[i].hash(), detAndCompKps[i].hash());
}

/**
 * This test is here to guard propagation of NaNs that happens on this image. NaNs are guarded
 * by debug asserts in AKAZE, which should fire for you if you are lucky.
 *
 * This test also reveals problems with uninitialized memory that happens only on this image.
 * This is very hard to hit and depends a lot on particular allocator. Run this test in valgrind and check
 * for uninitialized values if you think you are hitting this problem again.
 */
TEST(Features2d_AKAZE, uninitialized_and_nans)
{
    Mat b1 = imread(cvtest::TS::ptr()->get_data_path() + "../stitching/b1.png");
    ASSERT_FALSE(b1.empty());

    vector<KeyPoint> keypoints;
    Mat desc;
    Ptr<Feature2D> akaze = AKAZE::create();
    akaze->detectAndCompute(b1, noArray(), keypoints, desc);
}

TEST(Features2d_KAZE, issue_27134_charbonnier_subpixel)
{
    // Test for issue #27134: KAZE with DIFF_CHARBONNIER should produce subpixel coordinates
    cv::Mat img(100, 100, CV_8UC1);
    cv::randu(img, cv::Scalar(0), cv::Scalar(255));
    
    // Create two keypoints at the same location with different angles
    std::vector<cv::KeyPoint> kp_in;
    kp_in.push_back(cv::KeyPoint(50.0f, 50.0f, 20.0f, 45.0f));
    kp_in.push_back(cv::KeyPoint(50.0f, 50.0f, 20.0f, 225.0f));
    
    // Test with DIFF_CHARBONNIER
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create(false, false, 0.001f, 4, 4, cv::KAZE::DIFF_CHARBONNIER);
    cv::Mat desc;
    std::vector<cv::KeyPoint> kp_out = kp_in;
    kaze->compute(img, kp_out, desc);
    
    ASSERT_EQ(kp_out.size(), (size_t)2);
    
    // Verify that keypoints have subpixel precision (not integer coordinates)
    for (size_t i = 0; i < kp_out.size(); i++)
    {
        float x = kp_out[i].pt.x;
        float y = kp_out[i].pt.y;
        
        // Check that coordinates are not exactly integers
        bool is_subpixel = (x != std::floor(x)) || (y != std::floor(y));
        EXPECT_TRUE(is_subpixel) << "Keypoint " << i << " should have subpixel coordinates, got (" << x << ", " << y << ")";
    }
    
    // Verify that keypoints with different angles produce different descriptors
    ASSERT_FALSE(desc.empty());
    ASSERT_EQ(desc.rows, 2);
    
    cv::Mat desc1 = desc.row(0);
    cv::Mat desc2 = desc.row(1);
    
    double dist = cv::norm(desc1, desc2, cv::NORM_L2);
    EXPECT_GT(dist, 0.0) << "Descriptors for keypoints with different angles should be different";
}

}} // namespace
