// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

using namespace std;
using namespace cv;

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
