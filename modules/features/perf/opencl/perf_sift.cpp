// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

typedef TestBaseWithParam<Size> SIFTOclFixture;

OCL_PERF_TEST_P(SIFTOclFixture, DetectAndCompute, ::testing::Values(
    Size(320, 240), Size(640, 480), Size(960, 540),
    Size(1280, 720), Size(1920, 1080)))
{
    Size sz = GetParam();
    Mat msrc = imread(getDataPath("stitching/s2.jpg"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(msrc.empty()) << "Failed to load stitching/s2.jpg";
    Mat mimg;
    resize(msrc, mimg, sz, 0, 0, INTER_LINEAR);
    UMat img = mimg.getUMat(ACCESS_READ), mask;
    declare.in(img, WARMUP_READ);

    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> points;
    UMat descriptors;

    OCL_TEST_CYCLE() sift->detectAndCompute(img, mask, points, descriptors, false);

    EXPECT_GT(points.size(), 20u);
    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(SIFTOclFixture, Compute, ::testing::Values(
    Size(640, 480), Size(1280, 720), Size(1920, 1080)))
{
    Size sz = GetParam();
    Mat msrc = imread(getDataPath("stitching/s2.jpg"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(msrc.empty()) << "Failed to load stitching/s2.jpg";
    Mat mimg;
    resize(msrc, mimg, sz, 0, 0, INTER_LINEAR);
    UMat img = mimg.getUMat(ACCESS_READ);
    declare.in(img, WARMUP_READ);

    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> points;
    sift->detect(img, points);

    UMat descriptors;
    OCL_TEST_CYCLE() sift->compute(img, points, descriptors);

    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

} // ocl
} // opencv_test

#endif // HAVE_OPENCL
