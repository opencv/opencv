// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{

static Mat makeDenoisingInput(const std::string& path, const Size& size, int depth)
{
    Mat original = imread(perf::TestBase::getDataPath(path), IMREAD_GRAYSCALE);
    if (original.empty())
        return Mat();

    Mat tiled;
    repeat(original, (size.height + original.rows - 1) / original.rows,
                     (size.width + original.cols - 1) / original.cols, tiled);
    Mat src = tiled(Rect(0, 0, size.width, size.height)).clone();
    if (depth != CV_8U)
        src.convertTo(src, depth);
    return src;
}

static void makeDenoisingMultiInput(const Size& size, int depth, std::vector<Mat>& srcs)
{
    const int imgs_count = 3;
    srcs.resize(imgs_count);
    for (int i = 0; i < imgs_count; i++)
        srcs[i] = makeDenoisingInput(format("cv/denoising/lena_noised_gaussian_sigma=20_multi_%d.png", i),
                                     size, depth);
}

typedef perf::TestBaseWithParam<Size> Size_Denoising;

PERF_TEST_P(Size_Denoising, fastNlMeansDenoising,
            testing::Values(::perf::szVGA, ::perf::sz720p, Size(2592, 1944)))
{
    const Size size = GetParam();

    Mat src = makeDenoisingInput("cv/denoising/lena_noised_gaussian_sigma=10.png", size, CV_8U);
    ASSERT_FALSE(src.empty()) << "Could not load input image";
    Mat dst(size, CV_8UC1);

    declare.in(src).out(dst).time(120);

    TEST_CYCLE() cv::fastNlMeansDenoising(src, dst, 10, 7, 21);

    SANITY_CHECK_NOTHING();
}

typedef tuple<Size, perf::MatDepth> Size_Depth_t;
typedef perf::TestBaseWithParam<Size_Depth_t> Size_Depth_Denoising;

PERF_TEST_P(Size_Depth_Denoising, fastNlMeansDenoising_L1,
            testing::Combine(testing::Values(::perf::szVGA, ::perf::sz720p, Size(2592, 1944)),
                             testing::Values(CV_8U, CV_16U)))
{
    const Size size = get<0>(GetParam());
    const int depth = get<1>(GetParam());

    Mat src = makeDenoisingInput("cv/denoising/lena_noised_gaussian_sigma=10.png", size, depth);
    ASSERT_FALSE(src.empty()) << "Could not load input image";
    Mat dst(size, CV_MAKETYPE(depth, 1));
    std::vector<float> h(1, 10.f);

    declare.in(src).out(dst).time(120);

    TEST_CYCLE() cv::fastNlMeansDenoising(src, dst, h, 7, 21, NORM_L1);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_Denoising, fastNlMeansDenoisingMulti,
            testing::Values(::perf::szVGA, ::perf::sz720p))
{
    const Size size = GetParam();

    std::vector<Mat> srcs;
    makeDenoisingMultiInput(size, CV_8U, srcs);
    for (size_t i = 0; i < srcs.size(); i++)
        ASSERT_FALSE(srcs[i].empty()) << "Could not load input image " << i;
    const int imgs_count = (int)srcs.size();
    Mat dst(size, CV_8UC1);

    declare.in(srcs[0], srcs[1], srcs[2]).out(dst).time(180);

    TEST_CYCLE() cv::fastNlMeansDenoisingMulti(srcs, dst, imgs_count / 2, imgs_count, 15, 7, 21);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_Depth_Denoising, fastNlMeansDenoisingMulti_L1,
            testing::Combine(testing::Values(::perf::szVGA, ::perf::sz720p),
                             testing::Values(CV_8U, CV_16U)))
{
    const Size size = get<0>(GetParam());
    const int depth = get<1>(GetParam());

    std::vector<Mat> srcs;
    makeDenoisingMultiInput(size, depth, srcs);
    for (size_t i = 0; i < srcs.size(); i++)
        ASSERT_FALSE(srcs[i].empty()) << "Could not load input image " << i;
    const int imgs_count = (int)srcs.size();
    Mat dst(size, CV_MAKETYPE(depth, 1));
    std::vector<float> h(1, 15.f);

    declare.in(srcs[0], srcs[1], srcs[2]).out(dst).time(180);

    TEST_CYCLE() cv::fastNlMeansDenoisingMulti(srcs, dst, imgs_count / 2, imgs_count, h, 7, 21, NORM_L1);

    SANITY_CHECK_NOTHING();
}

} // namespace
