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
    std::string expected_path = folder + "lena_noised_denoised_grayscale_tw=7_sw=21_h=10.png";

    cv::Mat original = cv::imread(original_path, cv::IMREAD_GRAYSCALE);
    cv::Mat expected = cv::imread(expected_path, cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    cv::Mat result;
    cv::bm3dDenoising(original, result, 1);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, cv::NORM_L2));
}
//
//#ifdef TEST_TRANSFORMS
//TEST(Photo_DenoisingBm3dTransforms, regression)
//{
//    const int templateWindowSize = 4;
//    const int templateWindowSizeSq = templateWindowSize * templateWindowSize;
//
//    uchar src[templateWindowSizeSq];
//    short dst[templateWindowSizeSq];
//
//    for (uchar i = 0; i < templateWindowSizeSq; ++i)
//    {
//        src[i] = i;
//    }
//
//    Haar4x4(src, dst, templateWindowSize);
//    InvHaar4x4(dst);
//
//    for (uchar i = 0; i < templateWindowSizeSq; ++i)
//        ASSERT_EQ(static_cast<short>(src[i]), dst[i]);
//}
//#endif
//
//TEST(Photo_Bm3dDenoising, speed)
//{
//    std::string imgname = std::string(cvtest::TS::ptr()->get_data_path()) + "shared/5MP.png";
//    Mat src = imread(imgname, 0), dst;
//
//    double t = (double)getTickCount();
//    bm3dDenoising(src, dst, 1);
//    t = (double)getTickCount() - t;
//    printf("execution time: %gms\n", t*1000. / getTickFrequency());
//}
