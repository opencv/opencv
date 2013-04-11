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

#ifdef HAVE_CUDA

using namespace cvtest;

///////////////////////////////////////////////////////////////////
// Gold implementation

namespace
{
    template <typename T, template <typename> class Interpolator> void remapImpl(const cv::Mat& src, const cv::Mat& xmap, const cv::Mat& ymap, cv::Mat& dst, int borderType, cv::Scalar borderVal)
    {
        const int cn = src.channels();

        cv::Size dsize = xmap.size();

        dst.create(dsize, src.type());

        for (int y = 0; y < dsize.height; ++y)
        {
            for (int x = 0; x < dsize.width; ++x)
            {
                for (int c = 0; c < cn; ++c)
                    dst.at<T>(y, x * cn + c) = Interpolator<T>::getValue(src, ymap.at<float>(y, x), xmap.at<float>(y, x), c, borderType, borderVal);
            }
        }
    }

    void remapGold(const cv::Mat& src, const cv::Mat& xmap, const cv::Mat& ymap, cv::Mat& dst, int interpolation, int borderType, cv::Scalar borderVal)
    {
        typedef void (*func_t)(const cv::Mat& src, const cv::Mat& xmap, const cv::Mat& ymap, cv::Mat& dst, int borderType, cv::Scalar borderVal);

        static const func_t nearest_funcs[] =
        {
            remapImpl<unsigned char, NearestInterpolator>,
            remapImpl<signed char, NearestInterpolator>,
            remapImpl<unsigned short, NearestInterpolator>,
            remapImpl<short, NearestInterpolator>,
            remapImpl<int, NearestInterpolator>,
            remapImpl<float, NearestInterpolator>
        };

        static const func_t linear_funcs[] =
        {
            remapImpl<unsigned char, LinearInterpolator>,
            remapImpl<signed char, LinearInterpolator>,
            remapImpl<unsigned short, LinearInterpolator>,
            remapImpl<short, LinearInterpolator>,
            remapImpl<int, LinearInterpolator>,
            remapImpl<float, LinearInterpolator>
        };

        static const func_t cubic_funcs[] =
        {
            remapImpl<unsigned char, CubicInterpolator>,
            remapImpl<signed char, CubicInterpolator>,
            remapImpl<unsigned short, CubicInterpolator>,
            remapImpl<short, CubicInterpolator>,
            remapImpl<int, CubicInterpolator>,
            remapImpl<float, CubicInterpolator>
        };

        static const func_t* funcs[] = {nearest_funcs, linear_funcs, cubic_funcs};

        funcs[interpolation][src.depth()](src, xmap, ymap, dst, borderType, borderVal);
    }
}

///////////////////////////////////////////////////////////////////
// Test

PARAM_TEST_CASE(Remap, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int interpolation;
    int borderType;
    bool useRoi;

    cv::Mat xmap;
    cv::Mat ymap;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        interpolation = GET_PARAM(3);
        borderType = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());

        // rotation matrix

        const double aplha = CV_PI / 4;
        static double M[2][3] = { {std::cos(aplha), -std::sin(aplha), size.width / 2.0},
                                  {std::sin(aplha),  std::cos(aplha), 0.0}};

        xmap.create(size, CV_32FC1);
        ymap.create(size, CV_32FC1);

        for (int y = 0; y < size.height; ++y)
        {
            for (int x = 0; x < size.width; ++x)
            {
                xmap.at<float>(y, x) = static_cast<float>(M[0][0] * x + M[0][1] * y + M[0][2]);
                ymap.at<float>(y, x) = static_cast<float>(M[1][0] * x + M[1][1] * y + M[1][2]);
            }
        }
    }
};

GPU_TEST_P(Remap, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Scalar val = randomScalar(0.0, 255.0);

    cv::gpu::GpuMat dst = createMat(xmap.size(), type, useRoi);
    cv::gpu::remap(loadMat(src, useRoi), dst, loadMat(xmap, useRoi), loadMat(ymap, useRoi), interpolation, borderType, val);

    cv::Mat dst_gold;
    remapGold(src, xmap, ymap, dst_gold, interpolation, borderType, val);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-3 : 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Remap, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT), BorderType(cv::BORDER_WRAP)),
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
