// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

OCL_PERF_TEST(Photo, DenoisingGrayscale)
{
    Mat _original = imread(getDataPath("cv/denoising/lena_noised_gaussian_sigma=10.png"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(_original.empty()) << "Could not load input image";

    UMat result(_original.size(), _original.type()), original;
    _original.copyTo(original);

    declare.in(original).out(result).iterations(10);

    OCL_TEST_CYCLE()
            cv::fastNlMeansDenoising(original, result, 10);

    SANITY_CHECK(result, 1);
}

OCL_PERF_TEST(Photo, DenoisingColored)
{
    Mat _original = imread(getDataPath("cv/denoising/lena_noised_gaussian_sigma=10.png"));
    ASSERT_FALSE(_original.empty()) << "Could not load input image";

    UMat result(_original.size(), _original.type()), original;
    _original.copyTo(original);

    declare.in(original).out(result).iterations(10);

    OCL_TEST_CYCLE()
            cv::fastNlMeansDenoisingColored(original, result, 10, 10);

    SANITY_CHECK(result, 2);
}

OCL_PERF_TEST(Photo, DISABLED_DenoisingGrayscaleMulti)
{
    const int imgs_count = 3;

    vector<UMat> original(imgs_count);
    Mat tmp;
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("cv/denoising/lena_noised_gaussian_sigma=20_multi_%d.png", i);
        tmp = imread(getDataPath(original_path), IMREAD_GRAYSCALE);
        ASSERT_FALSE(tmp.empty()) << "Could not load input image " << original_path;
        tmp.copyTo(original[i]);
        declare.in(original[i]);
    }
    UMat result(tmp.size(), tmp.type());
    declare.out(result).iterations(10);

    OCL_TEST_CYCLE()
            cv::fastNlMeansDenoisingMulti(original, result, imgs_count / 2, imgs_count, 15);

    SANITY_CHECK(result);
}

OCL_PERF_TEST(Photo, DISABLED_DenoisingColoredMulti)
{
    const int imgs_count = 3;

    vector<UMat> original(imgs_count);
    Mat tmp;
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("cv/denoising/lena_noised_gaussian_sigma=20_multi_%d.png", i);
        tmp = imread(getDataPath(original_path), IMREAD_COLOR);
        ASSERT_FALSE(tmp.empty()) << "Could not load input image " << original_path;

        tmp.copyTo(original[i]);
        declare.in(original[i]);
    }
    UMat result(tmp.size(), tmp.type());
    declare.out(result).iterations(10);

    OCL_TEST_CYCLE()
            cv::fastNlMeansDenoisingColoredMulti(original, result, imgs_count / 2, imgs_count, 10, 15);

    SANITY_CHECK(result);
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
