// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <limits>

namespace opencv_test { namespace {

TEST(Imgproc_Remap3D, linear_ramp_and_depth_displacement)
{
    const int size[] = {4, 5, 6}, outSize[] = {2, 3, 4};
    Mat src(3, size, CV_32F), map(3, outSize, CV_32FC3), dst;
    for (int z = 0; z < size[0]; ++z)
        for (int y = 0; y < size[1]; ++y)
            for (int x = 0; x < size[2]; ++x)
                src.at<float>(z, y, x) = float(100 * z + 10 * y + x);
    for (int z = 0; z < outSize[0]; ++z)
        for (int y = 0; y < outSize[1]; ++y)
            for (int x = 0; x < outSize[2]; ++x)
                map.at<Vec3f>(z, y, x) = Vec3f(x + 0.23f, y + 0.37f, z + x * 0.19f);
    cv::remap3D(src, dst, map, INTER_LINEAR);
    ASSERT_EQ(3, dst.dims);
    ASSERT_EQ(map.size, dst.size);
    for (int z = 0; z < outSize[0]; ++z)
        for (int y = 0; y < outSize[1]; ++y)
            for (int x = 0; x < outSize[2]; ++x)
            {
                Vec3f p = map.at<Vec3f>(z, y, x);
                EXPECT_NEAR(p[0] + 10 * p[1] + 100 * p[2], dst.at<float>(z, y, x), 5e-5);
            }
}

typedef testing::TestWithParam<testing::tuple<int, int, int, int> > Remap3DTest;

TEST_P(Remap3DTest, reference_and_strides)
{
    const int depth = testing::get<0>(GetParam()), cn = testing::get<1>(GetParam());
    const int interpolation = testing::get<2>(GetParam()), border = testing::get<3>(GetParam());
    const int type = CV_MAKETYPE(depth, cn), storageSize[] = {5, 6, 8};
    const Range roi[] = {Range(1, 4), Range(1, 5), Range(1, 7)};
    Mat storage(3, storageSize, type), mapStorage(3, storageSize, CV_32FC3);
    Mat dstStorage(3, storageSize, type, Scalar::all(0));
    Mat src = storage(roi), map = mapStorage(roi), dst = dstStorage(roi);
    uchar* dstData = dst.data;
    RNG rng(29768);
    rng.fill(storage, RNG::UNIFORM, 0, 200);
    for (int z = 0; z < map.size[0]; ++z)
        for (int y = 0; y < map.size[1]; ++y)
            for (int x = 0; x < map.size[2]; ++x)
                map.at<Vec3f>(z, y, x) = Vec3f(rng.uniform(-2.f, 7.f),
                                               rng.uniform(-2.f, 5.f), rng.uniform(-2.f, 4.f));
    const Scalar borderValue(11, 23, 37, 49);
    cv::remap3D(src, dst, map, interpolation, border, borderValue);
    EXPECT_EQ(dstData, dst.data);
    Mat source64, actual64;
    src.convertTo(source64, CV_64F);
    dst.convertTo(actual64, CV_64F);
    // Independent reference: sum the separable tent kernels at the surrounding lattice points.
    for (int z = 0; z < map.size[0]; ++z)
        for (int y = 0; y < map.size[1]; ++y)
            for (int x = 0; x < map.size[2]; ++x)
            {
                const Vec3f p = map.at<Vec3f>(z, y, x);
                const int ix = cvFloor(p[0] + (interpolation == INTER_NEAREST ? 0.5 : 0.0));
                const int iy = cvFloor(p[1] + (interpolation == INTER_NEAREST ? 0.5 : 0.0));
                const int iz = cvFloor(p[2] + (interpolation == INTER_NEAREST ? 0.5 : 0.0));
                for (int c = 0; c < cn; ++c)
                {
                    double expected = 0;
                    const int n = interpolation == INTER_NEAREST ? 1 : 2;
                    for (int k = iz; k < iz + n; ++k)
                        for (int j = iy; j < iy + n; ++j)
                            for (int i = ix; i < ix + n; ++i)
                            {
                                double w = n == 1 ? 1.0 : (1.0 - std::abs(p[0] - double(i))) *
                                    (1.0 - std::abs(p[1] - double(j))) * (1.0 - std::abs(p[2] - double(k)));
                                int sx = cv::borderInterpolate(i, src.size[2], border);
                                int sy = cv::borderInterpolate(j, src.size[1], border);
                                int sz = cv::borderInterpolate(k, src.size[0], border);
                                double v = sx < 0 || sy < 0 || sz < 0 ? borderValue[c] :
                                    source64.ptr<double>(sz, sy)[sx * cn + c];
                                expected += w * v;
                            }
                    if (depth == CV_8U)
                        expected = saturate_cast<uchar>(expected);
                    EXPECT_NEAR(expected, actual64.ptr<double>(z, y)[x * cn + c], depth == CV_8U ? 0 : 2e-5);
                }
            }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Remap3DTest, testing::Combine(
    testing::Values(CV_8U, CV_32F), testing::Values(1, 2, 3, 4),
    testing::Values(INTER_NEAREST, INTER_LINEAR), testing::Values(BORDER_CONSTANT, BORDER_REPLICATE)));

TEST(Imgproc_Remap3D, singleton_borders_and_extreme_coordinates)
{
    const int size[] = {1, 1, 1}, mapSize[] = {1, 1, 5};
    Mat src(3, size, CV_32F, Scalar(80)), map(3, mapSize, CV_32FC3), dst;
    map.at<Vec3f>(0, 0, 0) = Vec3f(0, 0, 0);
    map.at<Vec3f>(0, 0, 1) = Vec3f(-0.5f, -0.5f, -0.5f);
    map.at<Vec3f>(0, 0, 2) = Vec3f(0.5f, 0.5f, 0.5f);
    map.at<Vec3f>(0, 0, 3) = Vec3f(std::numeric_limits<float>::max(), 0, 0);
    map.at<Vec3f>(0, 0, 4) = Vec3f(0, -std::numeric_limits<float>::max(), 0);
    cv::remap3D(src, dst, map, INTER_LINEAR, BORDER_CONSTANT, Scalar(8));
    const float expected[] = {80, 17, 17, 8, 8};
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(expected[i], dst.at<float>(0, 0, i));
    cv::remap3D(src, dst, map, INTER_NEAREST, BORDER_CONSTANT, Scalar(8));
    const float nearest[] = {80, 80, 8, 8, 8};
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(nearest[i], dst.at<float>(0, 0, i));
    for (int interpolation = INTER_NEAREST; interpolation <= INTER_LINEAR; ++interpolation)
    {
        cv::remap3D(src, dst, map, interpolation, BORDER_REPLICATE);
        for (int i = 0; i < 5; ++i)
            EXPECT_EQ(80, dst.at<float>(0, 0, i));
    }
}

TEST(Imgproc_Remap3D, invalid_arguments)
{
    const int size[] = {2, 3, 4};
    Mat src(3, size, CV_8U, Scalar(1)), map(3, size, CV_32FC3, Scalar(0)), dst;
    EXPECT_THROW(cv::remap3D(Mat(), dst, map, INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(Mat(3, 4, CV_8U), dst, map, INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(Mat(3, size, CV_16U), dst, map, INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(Mat(3, size, CV_8UC(5)), dst, map, INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(src, dst, Mat(), INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(src, dst, Mat(3, 4, CV_32FC3), INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(src, dst, Mat(3, size, CV_32FC2), INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(src, dst, Mat(3, size, CV_64FC3), INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(src, dst, map, INTER_CUBIC), cv::Exception);
    EXPECT_THROW(cv::remap3D(src, dst, map, INTER_LINEAR, BORDER_REFLECT), cv::Exception);
    EXPECT_THROW(cv::remap3D(src, src, map, INTER_LINEAR), cv::Exception);
    EXPECT_THROW(cv::remap3D(Mat(3, size, CV_32FC3), map, map, INTER_LINEAR), cv::Exception);
    map.at<Vec3f>(0, 0, 0)[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(cv::remap3D(src, dst, map, INTER_LINEAR), cv::Exception);
    map.at<Vec3f>(0, 0, 0)[0] = std::numeric_limits<float>::infinity();
    EXPECT_THROW(cv::remap3D(src, dst, map, INTER_NEAREST, BORDER_REPLICATE), cv::Exception);
}

TEST(Imgproc_Remap3D, constant_border_conversion)
{
    const int size[] = {1, 1, 1};
    Mat src(3, size, CV_8UC3, Scalar(100, 100, 100));
    Mat map(3, size, CV_32FC3, Scalar(-0.5, 0, 0)), dst;
    cv::remap3D(src, dst, map, INTER_LINEAR, BORDER_CONSTANT, Scalar(-100, 400, 40));
    const Vec3b result = dst.at<Vec3b>(0, 0, 0);
    EXPECT_EQ(50, result[0]);
    EXPECT_EQ(saturate_cast<uchar>(177.5), result[1]);
    EXPECT_EQ(70, result[2]);
}

TEST(Imgproc_Remap3D, integer_coordinate_ignores_zero_weight_neighbors)
{
    const int size[] = {2, 2, 2}, mapSize[] = {1, 1, 1};
    Mat src(3, size, CV_32F, Scalar(std::numeric_limits<float>::quiet_NaN()));
    src.at<float>(0, 0, 0) = 42;
    Mat map(3, mapSize, CV_32FC3, Scalar(0)), dst;
    cv::remap3D(src, dst, map, INTER_LINEAR);
    EXPECT_EQ(42, dst.at<float>(0, 0, 0));
}

}} // namespace
