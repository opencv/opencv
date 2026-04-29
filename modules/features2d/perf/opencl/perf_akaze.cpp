// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

typedef Size_MatType AKAZEFixture;

OCL_PERF_TEST_P(AKAZEFixture, detectAndCompute, ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3), OCL_PERF_ENUM((MatType)CV_8UC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat img(srcSize, type), mask;
    declare.in(img, WARMUP_RNG);

    Ptr<AKAZE> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 0, 3, 0.001f, 1, 1, KAZE::DIFF_PM_G2);
    vector<KeyPoint> points;
    UMat descriptors;

    OCL_TEST_CYCLE() akaze->detectAndCompute(img, mask, points, descriptors, false);

    EXPECT_GT(points.size(), 20u);
    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

typedef Size_MatType AKAZEOclMldbUprightFixture;

OCL_PERF_TEST_P(AKAZEOclMldbUprightFixture, detectAndComputeMLDB,
    ::testing::Combine(
        ::testing::Values(
            cv::Size(320, 240), cv::Size(640, 480), cv::Size(960, 540),
            cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(2560, 1440), cv::Size(3840, 2160)
        ),
        OCL_PERF_ENUM((MatType)CV_8UC1)
    ))
{
    const Size_MatType_t params = GetParam();
    const cv::Size srcSize = get<0>(params);

    UMat img(srcSize, CV_8U);
    declare.in(img, WARMUP_RNG);

    Ptr<Feature2D> det = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 0, 3, 0.001f, 4, 4, KAZE::DIFF_PM_G2);
    vector<KeyPoint> kps;
    UMat descriptors;

    OCL_TEST_CYCLE() det->detectAndCompute(img, noArray(), kps, descriptors, false);

    EXPECT_GT(kps.size(), 0u);
    SANITY_CHECK_NOTHING();
}

typedef tuple<std::string, int> AKAZEOclRealImagesParams;
typedef TestBaseWithParam<AKAZEOclRealImagesParams> AKAZEOclRealImagesFixture;

OCL_PERF_TEST_P(AKAZEOclRealImagesFixture, detectAndComputeMLDBRealImages,
    ::testing::Combine(
        ::testing::Values(
            std::string("stitching/boat1.jpg"),
            std::string("stitching/boat2.jpg"),
            std::string("stitching/boat3.jpg"),
            std::string("stitching/boat4.jpg"),
            std::string("stitching/boat5.jpg"),
            std::string("stitching/boat6.jpg")
        ),
        ::testing::Values(0)
    ))
{
    const std::string imgName = get<0>(GetParam());

    Mat img_mat = imread(getDataPath(imgName), IMREAD_GRAYSCALE);
    if (img_mat.empty())
        throw cvtest::SkipTestException("Image not found: " + imgName);

    UMat img;
    img_mat.copyTo(img);
    declare.in(img);

    Ptr<Feature2D> det = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 0, 3, 0.001f, 4, 4, KAZE::DIFF_PM_G2);
    vector<KeyPoint> kps;
    UMat descriptors;

    OCL_TEST_CYCLE() det->detectAndCompute(img, noArray(), kps, descriptors, false);

    EXPECT_GT(kps.size(), 0u);
    SANITY_CHECK_NOTHING();
}

typedef Size_MatType AKAZEOclKazeUprightFixture;

OCL_PERF_TEST_P(AKAZEOclKazeUprightFixture, detectAndComputeKAZEUpright,
    ::testing::Combine(
        ::testing::Values(cv::Size(320, 240), cv::Size(640, 480), cv::Size(1280, 720)),
        OCL_PERF_ENUM((MatType)CV_8UC1)
    ))
{
    const Size_MatType_t params = GetParam();
    const cv::Size srcSize = get<0>(params);

    UMat img(srcSize, CV_8U);
    declare.in(img, WARMUP_RNG);

    Ptr<Feature2D> det = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT, 0, 3, 0.001f, 4, 4, KAZE::DIFF_PM_G2);
    vector<KeyPoint> kps;
    UMat descriptors;

    OCL_TEST_CYCLE() det->detectAndCompute(img, noArray(), kps, descriptors, false);

    EXPECT_GT(kps.size(), 0u);
    SANITY_CHECK_NOTHING();
}

} // ocl
} // opencv_test

#endif // HAVE_OPENCL
