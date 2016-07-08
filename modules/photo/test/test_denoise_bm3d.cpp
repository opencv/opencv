/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "opencv2/photo.hpp"
#include <string>

#define DUMP_RESULTS
#define TEST_TRANSFORMS

#ifdef TEST_TRANSFORMS
#include "..\..\photo\src\bm3d_denoising_transforms.hpp"
#include "..\..\photo\src\bm3d_denoising_invoker.hpp"
#endif

#ifdef DUMP_RESULTS
#  define DUMP(image, path) imwrite(path, image)
#else
#  define DUMP(image, path)
#endif


TEST(Photo_DenoisingBm3dGrayscale, regression)
{
    std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "denoising/";
    std::string original_path = folder + "lena_noised_gaussian_sigma=10.png";
    std::string expected_path = folder + "lena_noised_gaussian_sigma=10.png";

    cv::Mat original = cv::imread(original_path, cv::IMREAD_GRAYSCALE);
    cv::Mat expected = cv::imread(expected_path, cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    cv::Mat result;
    cv::bm3dDenoising(original, result, 20, 4, 16, 50);

    original_path = folder + "lena_noised_noisy_bm3d_grayscale.png";
    DUMP(original, original_path + ".res.png");
    expected_path = folder + "lena_noised_denoised_bm3d_grayscale.png";
    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, cv::NORM_L2));
}

#ifdef TEST_TRANSFORMS

TEST(Photo_DenoisingBm3dTransforms, regression_2D)
{
    const int templateWindowSize = 4;
    const int templateWindowSizeSq = templateWindowSize * templateWindowSize;

    uchar src[templateWindowSizeSq];
    short dst[templateWindowSizeSq];

    // Initialize array
    for (uchar i = 0; i < templateWindowSizeSq; ++i)
    {
        src[i] = i;
    }

    Haar4x4(src, dst, templateWindowSize);
    InvHaar4x4(dst);

    for (uchar i = 0; i < templateWindowSizeSq; ++i)
        ASSERT_EQ(static_cast<short>(src[i]), dst[i]);
}

static void Test1dTransform(
    short *thrMap,
    int groupSize,
    int templateWindowSizeSq,
    short **z,
    short **zOrig,
    short (*HaarTransformShrink)(short **z, const int &n, short *&thrMap),
    void (*InverseHaarTransform)(short **src, const int &n),
    int expectedNonZeroCount = -1)
{
    if (expectedNonZeroCount < 0)
        expectedNonZeroCount = groupSize * templateWindowSizeSq;

    // Test group size
    short sumNonZero = 0;
    short *thrMapPtr1D = thrMap + (groupSize - 1) * templateWindowSizeSq;
    for (int n = 0; n < templateWindowSizeSq; n++)
    {
        sumNonZero += HaarTransformShrink(z, n, thrMapPtr1D);
        InverseHaarTransform(z, n);
    }

    //// Print
    //for (int i = 0; i < groupSize; ++i)
    //{
    //    for (int j = 0; j < templateWindowSizeSq; ++j)
    //    {
    //        if (j % 4 == 0)
    //            std::cout << std::endl;
    //        std::cout << z[i][j] << " (" << zOrig[i][j] << ") ";
    //    }
    //    std::cout << std::endl;
    //}

    // Assert transform
    if (expectedNonZeroCount == groupSize * templateWindowSizeSq)
    {
        for (int i = 0; i < groupSize; ++i)
            for (int j = 0; j < templateWindowSizeSq; ++j)
                ASSERT_EQ(z[i][j], zOrig[i][j]);
    }

    // Assert shrinkage
    printf("sumNonZero: %d\n", sumNonZero);
    ASSERT_EQ(sumNonZero, expectedNonZeroCount);
}

TEST(Photo_DenoisingBm3dTransforms, regression_1D_transform)
{
    const int templateWindowSize = 4;
    const int templateWindowSizeSq = templateWindowSize * templateWindowSize;
    const int searchWindowSize = 16;
    const int searchWindowSizeSq = searchWindowSize * searchWindowSize;
    const float h = 10;
    int maxGroupSize = 8;

    // Precompute separate maps for transform and shrinkage verification
    short *thrMapTransform = new short[templateWindowSizeSq * ((BM3D_MAX_3D_SIZE << 1) - 1)];
    short *thrMapShrinkage = new short[templateWindowSizeSq * ((BM3D_MAX_3D_SIZE << 1) - 1)];
    ComputeThresholdMap1D(thrMapTransform, kThrMap1D, kThrMap4x4, 0, kCoeff, templateWindowSizeSq);
    ComputeThresholdMap1D(thrMapShrinkage, kThrMap1D, kThrMap4x4, h, kCoeff, templateWindowSizeSq);

    // Generate some data
    short **z = new short*[maxGroupSize];
    short **zOrig = new short*[maxGroupSize];
    for (int i = 0; i < maxGroupSize; ++i)
    {
        z[i] = new short[templateWindowSizeSq];
        zOrig[i] = new short[templateWindowSizeSq];
    }

    for (int i = 0; i < maxGroupSize; ++i)
    {
        for (int j = 0; j < templateWindowSizeSq; ++j)
        {
            z[i][j] = (j + 1);
            zOrig[i][j] = z[i][j];
        }
    }

    // Verify transforms
    Test1dTransform(thrMapTransform, 2, templateWindowSizeSq, z, zOrig, HaarTransformShrink2, InverseHaarTransform2);
    Test1dTransform(thrMapTransform, 4, templateWindowSizeSq, z, zOrig, HaarTransformShrink4, InverseHaarTransform4);
    Test1dTransform(thrMapTransform, 8, templateWindowSizeSq, z, zOrig, HaarTransformShrink8, InverseHaarTransform8);

    // Verify shrinkage
    Test1dTransform(thrMapShrinkage, 2, templateWindowSizeSq, z, zOrig, HaarTransformShrink2, InverseHaarTransform2, 6);
    Test1dTransform(thrMapShrinkage, 4, templateWindowSizeSq, z, zOrig, HaarTransformShrink4, InverseHaarTransform4, 6);
    Test1dTransform(thrMapShrinkage, 8, templateWindowSizeSq, z, zOrig, HaarTransformShrink8, InverseHaarTransform8, 6);
}

#endif

//TEST(Photo_Bm3dDenoising, speed)
//{
//    std::string imgname = std::string(cvtest::TS::ptr()->get_data_path()) + "shared/5MP.png";
//    Mat src = imread(imgname, 0), dst;
//
//    double t = (double)getTickCount();
//    bm3dDenoising(src, dst, 1, 4, 16, 1);
//    t = (double)getTickCount() - t;
//    printf("execution time: %gms\n", t*1000. / getTickFrequency());
//}
