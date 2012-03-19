/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "precomp.hpp"

#ifdef HAVE_CUDA

////////////////////////////////////////////////////////////////////////////////
// Add_Array

PARAM_TEST_CASE(Add_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    int channels;
    bool useRoi;

    int stype;
    int dtype;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

TEST_P(Add_Array, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
    dst.setTo(cv::Scalar::all(0));
    cv::gpu::add(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, channels == 1 ? loadMat(mask, useRoi) : cv::gpu::GpuMat(), depth.second);

    cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
    cv::add(mat1, mat2, dst_gold, channels == 1 ? mask : cv::noArray(), depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Add_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    testing::Values(1, 2, 3, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Add_Scalar

PARAM_TEST_CASE(Add_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Add_Scalar, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
    dst.setTo(cv::Scalar::all(0));
    cv::gpu::add(loadMat(mat, useRoi), val, dst, loadMat(mask, useRoi), depth.second);

    cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
    cv::add(mat, val, dst_gold, mask, depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Add_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Subtract_Array

PARAM_TEST_CASE(Subtract_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    int channels;
    bool useRoi;

    int stype;
    int dtype;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

TEST_P(Subtract_Array, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
    dst.setTo(cv::Scalar::all(0));
    cv::gpu::subtract(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, channels == 1 ? loadMat(mask, useRoi) : cv::gpu::GpuMat(), depth.second);

    cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
    cv::subtract(mat1, mat2, dst_gold, channels == 1 ? mask : cv::noArray(), depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Subtract_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    testing::Values(1, 2, 3, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Subtract_Scalar

PARAM_TEST_CASE(Subtract_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Subtract_Scalar, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
    dst.setTo(cv::Scalar::all(0));
    cv::gpu::subtract(loadMat(mat, useRoi), val, dst, loadMat(mask, useRoi), depth.second);

    cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
    cv::subtract(mat, val, dst_gold, mask, depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Subtract_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Array

PARAM_TEST_CASE(Multiply_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    int channels;
    bool useRoi;

    int stype;
    int dtype;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

TEST_P(Multiply_Array, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    double scale = randomDouble(0.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
    cv::gpu::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, scale, depth.second);

    cv::Mat dst_gold;
    cv::multiply(mat1, mat2, dst_gold, scale, depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Multiply_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    testing::Values(1, 2, 3, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Array_Special_Case

PARAM_TEST_CASE(Multiply_Array_Special_Case, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Multiply_Array_Special_Case, _8UC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_8UC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat dst = createMat(size, CV_8UC4, useRoi);
    cv::gpu::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

    cv::Mat h_dst(dst);

    for (int y = 0; y < h_dst.rows; ++y)
    {
        const cv::Vec4b* mat1_row = mat1.ptr<cv::Vec4b>(y);
        const float* mat2_row = mat2.ptr<float>(y);
        const cv::Vec4b* dst_row = h_dst.ptr<cv::Vec4b>(y);

        for (int x = 0; x < h_dst.cols; ++x)
        {
            cv::Vec4b val1 = mat1_row[x];
            float val2 = mat2_row[x];
            cv::Vec4b actual = dst_row[x];

            cv::Vec4b gold;

            gold[0] = cv::saturate_cast<uchar>(val1[0] * val2);
            gold[1] = cv::saturate_cast<uchar>(val1[1] * val2);
            gold[2] = cv::saturate_cast<uchar>(val1[2] * val2);
            gold[3] = cv::saturate_cast<uchar>(val1[3] * val2);

            ASSERT_LE(std::abs(gold[0] - actual[0]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
        }
    }
}

TEST_P(Multiply_Array_Special_Case, _16SC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_16SC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat dst = createMat(size, CV_16SC4, useRoi);
    cv::gpu::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

    cv::Mat h_dst(dst);

    for (int y = 0; y < h_dst.rows; ++y)
    {
        const cv::Vec4s* mat1_row = mat1.ptr<cv::Vec4s>(y);
        const float* mat2_row = mat2.ptr<float>(y);
        const cv::Vec4s* dst_row = h_dst.ptr<cv::Vec4s>(y);

        for (int x = 0; x < h_dst.cols; ++x)
        {
            cv::Vec4s val1 = mat1_row[x];
            float val2 = mat2_row[x];
            cv::Vec4s actual = dst_row[x];

            cv::Vec4s gold;

            gold[0] = cv::saturate_cast<short>(val1[0] * val2);
            gold[1] = cv::saturate_cast<short>(val1[1] * val2);
            gold[2] = cv::saturate_cast<short>(val1[2] * val2);
            gold[3] = cv::saturate_cast<short>(val1[3] * val2);

            ASSERT_LE(std::abs(gold[0] - actual[0]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Multiply_Array_Special_Case, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Scalar

PARAM_TEST_CASE(Multiply_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Multiply_Scalar, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    double scale = randomDouble(0.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
    cv::gpu::multiply(loadMat(mat, useRoi), val, dst, scale, depth.second);

    cv::Mat dst_gold;
    cv::multiply(mat, val, dst_gold, scale, depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Multiply_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Array

PARAM_TEST_CASE(Divide_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    int channels;
    bool useRoi;

    int stype;
    int dtype;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

TEST_P(Divide_Array, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype, 1.0, 255.0);
    double scale = randomDouble(0.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
    cv::gpu::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, scale, depth.second);

    cv::Mat dst_gold;
    cv::divide(mat1, mat2, dst_gold, scale, depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Divide_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    testing::Values(1, 2, 3, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Array_Special_Case

PARAM_TEST_CASE(Divide_Array_Special_Case, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Divide_Array_Special_Case, _8UC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_8UC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1, 1.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, CV_8UC4, useRoi);
    cv::gpu::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

    cv::Mat h_dst(dst);

    for (int y = 0; y < h_dst.rows; ++y)
    {
        const cv::Vec4b* mat1_row = mat1.ptr<cv::Vec4b>(y);
        const float* mat2_row = mat2.ptr<float>(y);
        const cv::Vec4b* dst_row = h_dst.ptr<cv::Vec4b>(y);

        for (int x = 0; x < h_dst.cols; ++x)
        {
            cv::Vec4b val1 = mat1_row[x];
            float val2 = mat2_row[x];
            cv::Vec4b actual = dst_row[x];

            cv::Vec4b gold;

            gold[0] = cv::saturate_cast<uchar>(val1[0] / val2);
            gold[1] = cv::saturate_cast<uchar>(val1[1] / val2);
            gold[2] = cv::saturate_cast<uchar>(val1[2] / val2);
            gold[3] = cv::saturate_cast<uchar>(val1[3] / val2);

            ASSERT_LE(std::abs(gold[0] - actual[0]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
        }
    }
}

TEST_P(Divide_Array_Special_Case, _16SC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_16SC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1, 1.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, CV_16SC4, useRoi);
    cv::gpu::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

    cv::Mat h_dst(dst);

    for (int y = 0; y < h_dst.rows; ++y)
    {
        const cv::Vec4s* mat1_row = mat1.ptr<cv::Vec4s>(y);
        const float* mat2_row = mat2.ptr<float>(y);
        const cv::Vec4s* dst_row = h_dst.ptr<cv::Vec4s>(y);

        for (int x = 0; x < h_dst.cols; ++x)
        {
            cv::Vec4s val1 = mat1_row[x];
            float val2 = mat2_row[x];
            cv::Vec4s actual = dst_row[x];

            cv::Vec4s gold;

            gold[0] = cv::saturate_cast<short>(val1[0] / val2);
            gold[1] = cv::saturate_cast<short>(val1[1] / val2);
            gold[2] = cv::saturate_cast<short>(val1[2] / val2);
            gold[3] = cv::saturate_cast<short>(val1[3] / val2);

            ASSERT_LE(std::abs(gold[0] - actual[0]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
            ASSERT_LE(std::abs(gold[1] - actual[1]), 1.0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Divide_Array_Special_Case, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Scalar

PARAM_TEST_CASE(Divide_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Divide_Scalar, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(1.0, 255.0);
    double scale = randomDouble(0.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
    cv::gpu::divide(loadMat(mat, useRoi), val, dst, scale, depth.second);

    cv::Mat dst_gold;
    cv::divide(mat, val, dst_gold, scale, depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Divide_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Scalar_Inv

PARAM_TEST_CASE(Divide_Scalar_Inv, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatType, MatType> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Divide_Scalar_Inv, Accuracy)
{
    if (depth.first == CV_64F || depth.second == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    double scale = randomDouble(0.0, 255.0);
    cv::Mat mat = randomMat(size, depth.first, 1.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
    cv::gpu::divide(scale, loadMat(mat, useRoi), dst, depth.second);

    cv::Mat dst_gold;
    cv::divide(scale, mat, dst_gold, depth.second);

    EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Divide_Scalar_Inv, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// AbsDiff

PARAM_TEST_CASE(AbsDiff, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(AbsDiff, Array)
{
    if (depth == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::absdiff(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

    cv::Mat dst_gold;
    cv::absdiff(src1, src2, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(AbsDiff, Scalar)
{
    if (depth == CV_64F)
    {
        if (!devInfo.supports(cv::gpu::NATIVE_DOUBLE))
            return;
    }

    cv::Mat src = randomMat(size, depth);
    cv::Scalar val = randomScalar(0.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::absdiff(loadMat(src, useRoi), val, dst);

    cv::Mat dst_gold;
    cv::absdiff(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, depth <= CV_32F ? 1.0 : 1e-5);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, AbsDiff, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Abs

PARAM_TEST_CASE(Abs, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Abs, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::abs(loadMat(src, useRoi), dst);

    cv::Mat dst_gold = cv::abs(src);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Abs, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_16SC1), MatType(CV_32FC1)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Sqr

PARAM_TEST_CASE(Sqr, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Sqr, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::sqr(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::multiply(src, src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Sqr, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_16UC1), MatType(CV_16SC1), MatType(CV_32FC1)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Sqrt

namespace
{
    template <typename T> void sqrtImpl(const cv::Mat& src, cv::Mat& dst)
    {
        dst.create(src.size(), src.type());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
                dst.at<T>(y, x) = static_cast<T>(std::sqrt(static_cast<float>(src.at<T>(y, x))));
        }
    }

    void sqrtGold(const cv::Mat& src, cv::Mat& dst)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Mat& dst);

        const func_t funcs[] =
        {
            sqrtImpl<uchar>, sqrtImpl<schar>, sqrtImpl<ushort>, sqrtImpl<short>,
            sqrtImpl<int>, sqrtImpl<float>
        };

        funcs[src.depth()](src, dst);
    }
}

PARAM_TEST_CASE(Sqrt, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Sqrt, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::sqrt(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    sqrtGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Sqrt, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_16UC1), MatType(CV_16SC1), MatType(CV_32FC1)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Log

namespace
{
    template <typename T> void logImpl(const cv::Mat& src, cv::Mat& dst)
    {
        dst.create(src.size(), src.type());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
                dst.at<T>(y, x) = static_cast<T>(std::log(static_cast<float>(src.at<T>(y, x))));
        }
    }

    void logGold(const cv::Mat& src, cv::Mat& dst)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Mat& dst);

        const func_t funcs[] =
        {
            logImpl<uchar>, logImpl<schar>, logImpl<ushort>, logImpl<short>,
            logImpl<int>, logImpl<float>
        };

        funcs[src.depth()](src, dst);
    }
}

PARAM_TEST_CASE(Log, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Log, Accuracy)
{
    cv::Mat src = randomMat(size, type, 1.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::log(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    logGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-6);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Log, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_16UC1), MatType(CV_16SC1), MatType(CV_32FC1)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Exp

PARAM_TEST_CASE(Exp, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Exp, Accuracy)
{
    cv::Mat src = randomMat(size, type, 0.0, 10.0);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::exp(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::exp(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-2);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Exp, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_32FC1)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// compare

PARAM_TEST_CASE(Compare, cv::gpu::DeviceInfo, cv::Size, MatDepth, CmpCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cmp_code;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cmp_code = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Compare, Accuracy)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    cv::gpu::GpuMat dst = createMat(size, CV_8UC1, useRoi);
    cv::gpu::compare(loadMat(src1, useRoi), loadMat(src2, useRoi), dst, cmp_code);

    cv::Mat dst_gold;
    cv::compare(src1, src2, dst_gold, cmp_code);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Compare, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    ALL_CMP_CODES,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Bitwise_Array

PARAM_TEST_CASE(Bitwise_Array, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;

    cv::Mat src1;
    cv::Mat src2;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        src1 = randomMat(size, type, 0.0, std::numeric_limits<int>::max());
        src2 = randomMat(size, type, 0.0, std::numeric_limits<int>::max());
    }
};

TEST_P(Bitwise_Array, Not)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_not(loadMat(src1), dst);

    cv::Mat dst_gold = ~src1;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise_Array, Or)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_or(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 | src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise_Array, And)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_and(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 & src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise_Array, Xor)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_xor(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 ^ src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Bitwise_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    TYPES(CV_8U, CV_32S, 1, 4)));

//////////////////////////////////////////////////////////////////////////////
// Bitwise_Scalar

PARAM_TEST_CASE(Bitwise_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth, int)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int channels;

    cv::Mat src;
    cv::Scalar val;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        src = randomMat(size, CV_MAKE_TYPE(depth, channels));
        cv::Scalar_<int> ival = randomScalar(0.0, 255.0);
        val = ival;
    }
};

TEST_P(Bitwise_Scalar, Or)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_or(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_or(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise_Scalar, And)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_and(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_and(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise_Scalar, Xor)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_xor(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_xor(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Bitwise_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32S)),
    testing::Values(1, 3, 4)));

//////////////////////////////////////////////////////////////////////////////
// RShift

namespace
{
    template <typename T> void rhiftImpl(const cv::Mat& src, cv::Scalar_<int> val, cv::Mat& dst)
    {
        const int cn = src.channels();

        dst.create(src.size(), src.type());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                    dst.at<T>(y, x * cn + c) = src.at<T>(y, x * cn + c) >> val.val[c];
            }
        }
    }

    void rhiftGold(const cv::Mat& src, cv::Scalar_<int> val, cv::Mat& dst)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Scalar_<int> val, cv::Mat& dst);

        const func_t funcs[] =
        {
            rhiftImpl<uchar>, rhiftImpl<schar>, rhiftImpl<ushort>, rhiftImpl<short>, rhiftImpl<int>
        };

        funcs[src.depth()](src, val, dst);
    }
}

PARAM_TEST_CASE(RShift, cv::gpu::DeviceInfo, cv::Size, MatDepth, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int channels;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(RShift, Accuracy)
{
    int type = CV_MAKE_TYPE(depth, channels);
    cv::Mat src = randomMat(size, type);
    cv::Scalar_<int> val = randomScalar(0.0, 8.0);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::rshift(loadMat(src, useRoi), val, dst);

    cv::Mat dst_gold;
    rhiftGold(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, RShift, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_8S), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32S)),
    testing::Values(1, 3, 4),
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// LShift

namespace
{
    template <typename T> void lhiftImpl(const cv::Mat& src, cv::Scalar_<int> val, cv::Mat& dst)
    {
        const int cn = src.channels();

        dst.create(src.size(), src.type());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                    dst.at<T>(y, x * cn + c) = src.at<T>(y, x * cn + c) << val.val[c];
            }
        }
    }

    void lhiftGold(const cv::Mat& src, cv::Scalar_<int> val, cv::Mat& dst)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Scalar_<int> val, cv::Mat& dst);

        const func_t funcs[] =
        {
            lhiftImpl<uchar>, lhiftImpl<schar>, lhiftImpl<ushort>, lhiftImpl<short>, lhiftImpl<int>
        };

        funcs[src.depth()](src, val, dst);
    }
}

PARAM_TEST_CASE(LShift, cv::gpu::DeviceInfo, cv::Size, MatDepth, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int channels;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(LShift, Accuracy)
{
    int type = CV_MAKE_TYPE(depth, channels);
    cv::Mat src = randomMat(size, type);
    cv::Scalar_<int> val = randomScalar(0.0, 8.0);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::rshift(loadMat(src, useRoi), val, dst);

    cv::Mat dst_gold;
    rhiftGold(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, LShift, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),  MatDepth(CV_16U), MatDepth(CV_32S)),
    testing::Values(1, 3, 4),
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Min

PARAM_TEST_CASE(Min, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Min, Accuracy)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::min(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

    cv::Mat dst_gold = cv::min(src1, src2);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Min, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Max

PARAM_TEST_CASE(Max, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Max, Accuracy)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::max(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

    cv::Mat dst_gold = cv::max(src1, src2);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Max, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));































using namespace cvtest;
using namespace testing;

PARAM_TEST_CASE(ArithmTestBase, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1;
    cv::Mat mat2;
    cv::Scalar val;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, type, 5, 16, false);
        mat2 = randomMat(rng, size, type, 5, 16, false);

        val = cv::Scalar(rng.uniform(1, 3), rng.uniform(1, 3), rng.uniform(1, 3), rng.uniform(1, 3));
    }
};

////////////////////////////////////////////////////////////////////////////////
// transpose

struct Transpose : ArithmTestBase {};

TEST_P(Transpose, Accuracy)
{
    cv::Mat dst_gold;
    cv::transpose(mat1, dst_gold);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::transpose(loadMat(mat1, useRoi), gpuRes);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Transpose, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC4, CV_8SC1, CV_8SC4, CV_16UC2, CV_16SC2, CV_32SC1, CV_32SC2, CV_32FC1, CV_32FC2, CV_64FC1),
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// meanStdDev

PARAM_TEST_CASE(MeanStdDev, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    cv::Scalar mean_gold;
    cv::Scalar stddev_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, CV_8UC1, 1, 255, false);

        cv::meanStdDev(mat, mean_gold, stddev_gold);
    }
};

TEST_P(MeanStdDev, Accuracy)
{
    cv::Scalar mean;
    cv::Scalar stddev;

    cv::gpu::meanStdDev(loadMat(mat, useRoi), mean, stddev);

    EXPECT_NEAR(mean_gold[0], mean[0], 1e-5);
    EXPECT_NEAR(mean_gold[1], mean[1], 1e-5);
    EXPECT_NEAR(mean_gold[2], mean[2], 1e-5);
    EXPECT_NEAR(mean_gold[3], mean[3], 1e-5);

    EXPECT_NEAR(stddev_gold[0], stddev[0], 1e-5);
    EXPECT_NEAR(stddev_gold[1], stddev[1], 1e-5);
    EXPECT_NEAR(stddev_gold[2], stddev[2], 1e-5);
    EXPECT_NEAR(stddev_gold[3], stddev[3], 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, MeanStdDev, Combine(
                        ALL_DEVICES,
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// normDiff

PARAM_TEST_CASE(NormDiff, cv::gpu::DeviceInfo, NormCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int normCode;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    double norm_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        normCode = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, CV_8UC1, 1, 255, false);
        mat2 = randomMat(rng, size, CV_8UC1, 1, 255, false);

        norm_gold = cv::norm(mat1, mat2, normCode);
    }
};

TEST_P(NormDiff, Accuracy)
{
    double norm = cv::gpu::norm(loadMat(mat1, useRoi), loadMat(mat2, useRoi), normCode);

    EXPECT_NEAR(norm_gold, norm, 1e-6);
}

INSTANTIATE_TEST_CASE_P(Arithm, NormDiff, Combine(
                        ALL_DEVICES,
                        Values((int) cv::NORM_INF, (int) cv::NORM_L1, (int) cv::NORM_L2),
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// flip

PARAM_TEST_CASE(Flip, cv::gpu::DeviceInfo, MatType, FlipCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flip_code;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        flip_code = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 1, 255, false);

        cv::flip(mat, dst_gold, flip_code);
    }
};

TEST_P(Flip, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpu_res;

    cv::gpu::flip(loadMat(mat, useRoi), gpu_res, flip_code);

    gpu_res.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Flip, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values((int)FLIP_BOTH, (int)FLIP_X, (int)FLIP_Y),
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// LUT

PARAM_TEST_CASE(LUT, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;
    cv::Mat lut;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 1, 255, false);
        lut = randomMat(rng, cv::Size(256, 1), CV_8UC1, 100, 200, false);

        cv::LUT(mat, lut, dst_gold);
    }
};

TEST_P(LUT, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpu_res;

    cv::gpu::LUT(loadMat(mat, useRoi), lut, gpu_res);

    gpu_res.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, LUT, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC3),
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// pow

PARAM_TEST_CASE(Pow, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    double power;
    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 0.0, 100.0, false);

        if (mat.depth() == CV_32F)
            power = rng.uniform(1.2f, 3.f);
        else
        {
            int ipower = rng.uniform(2, 8);
            power = (float)ipower;
        }

        cv::pow(mat, power, dst_gold);
    }
};

TEST_P(Pow, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpu_res;

    cv::gpu::pow(loadMat(mat, useRoi), power, gpu_res);

    gpu_res.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 2);
}

INSTANTIATE_TEST_CASE_P(Arithm, Pow, Combine(
                        ALL_DEVICES,
                        Values(CV_32F, CV_32FC3),
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// magnitude

PARAM_TEST_CASE(Magnitude, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);
        mat2 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);

        cv::magnitude(mat1, mat2, dst_gold);
    }
};

TEST_P(Magnitude, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpu_res;

    cv::gpu::magnitude(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpu_res);

    gpu_res.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

INSTANTIATE_TEST_CASE_P(Arithm, Magnitude, Combine(
                        ALL_DEVICES,
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// phase

PARAM_TEST_CASE(Phase, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);
        mat2 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);

        cv::phase(mat1, mat2, dst_gold);
    }
};

TEST_P(Phase, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpu_res;

    cv::gpu::phase(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpu_res);

    gpu_res.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-3);
}

INSTANTIATE_TEST_CASE_P(Arithm, Phase, Combine(
                        ALL_DEVICES,
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// cartToPolar

PARAM_TEST_CASE(CartToPolar, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat mag_gold;
    cv::Mat angle_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);
        mat2 = randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);

        cv::cartToPolar(mat1, mat2, mag_gold, angle_gold);
    }
};

TEST_P(CartToPolar, Accuracy)
{
    cv::Mat mag, angle;

    cv::gpu::GpuMat gpuMag;
    cv::gpu::GpuMat gpuAngle;

    cv::gpu::cartToPolar(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuMag, gpuAngle);

    gpuMag.download(mag);
    gpuAngle.download(angle);

    EXPECT_MAT_NEAR(mag_gold, mag, 1e-4);
    EXPECT_MAT_NEAR(angle_gold, angle, 1e-3);
}

INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, Combine(
                        ALL_DEVICES,
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// polarToCart

PARAM_TEST_CASE(PolarToCart, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mag;
    cv::Mat angle;

    cv::Mat x_gold;
    cv::Mat y_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mag = randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);
        angle = randomMat(rng, size, CV_32FC1, 0.0, 2.0 * CV_PI, false);

        cv::polarToCart(mag, angle, x_gold, y_gold);
    }
};

TEST_P(PolarToCart, Accuracy)
{
    cv::Mat x, y;

    cv::gpu::GpuMat gpuX;
    cv::gpu::GpuMat gpuY;

    cv::gpu::polarToCart(loadMat(mag, useRoi), loadMat(angle, useRoi), gpuX, gpuY);

    gpuX.download(x);
    gpuY.download(y);

    EXPECT_MAT_NEAR(x_gold, x, 1e-4);
    EXPECT_MAT_NEAR(y_gold, y, 1e-4);
}

INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, Combine(
                        ALL_DEVICES,
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// minMax

PARAM_TEST_CASE(MinMax, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;
    cv::Mat mask;

    double minVal_gold;
    double maxVal_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 0.0, 127.0, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2, false);

        if (type != CV_8S)
        {
            cv::minMaxLoc(mat, &minVal_gold, &maxVal_gold, 0, 0, mask);
        }
        else
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type
            minVal_gold = std::numeric_limits<double>::max();
            maxVal_gold = -std::numeric_limits<double>::max();
            for (int i = 0; i < mat.rows; ++i)
            {
                const signed char* mat_row = mat.ptr<signed char>(i);
                const unsigned char* mask_row = mask.ptr<unsigned char>(i);
                for (int j = 0; j < mat.cols; ++j)
                {
                    if (mask_row[j])
                    {
                        signed char val = mat_row[j];
                        if (val < minVal_gold) minVal_gold = val;
                        if (val > maxVal_gold) maxVal_gold = val;
                    }
                }
            }
        }
    }
};

TEST_P(MinMax, Accuracy)
{
    if (type == CV_64F && !supportFeature(devInfo,  cv::gpu::NATIVE_DOUBLE))
        return;

    double minVal, maxVal;

    cv::gpu::minMax(loadMat(mat, useRoi), &minVal, &maxVal, loadMat(mask, useRoi));

    EXPECT_DOUBLE_EQ(minVal_gold, minVal);
    EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
}

INSTANTIATE_TEST_CASE_P(Arithm, MinMax, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// minMaxLoc

PARAM_TEST_CASE(MinMaxLoc, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;
    cv::Mat mask;

    double minVal_gold;
    double maxVal_gold;
    cv::Point minLoc_gold;
    cv::Point maxLoc_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 0.0, 127.0, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2, false);

        if (type != CV_8S)
        {
            cv::minMaxLoc(mat, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold, mask);
        }
        else
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type
            minVal_gold = std::numeric_limits<double>::max();
            maxVal_gold = -std::numeric_limits<double>::max();
            for (int i = 0; i < mat.rows; ++i)
            {
                const signed char* mat_row = mat.ptr<signed char>(i);
                const unsigned char* mask_row = mask.ptr<unsigned char>(i);
                for (int j = 0; j < mat.cols; ++j)
                {
                    if (mask_row[j])
                    {
                        signed char val = mat_row[j];
                        if (val < minVal_gold) { minVal_gold = val; minLoc_gold = cv::Point(j, i); }
                        if (val > maxVal_gold) { maxVal_gold = val; maxLoc_gold = cv::Point(j, i); }
                    }
                }
            }
        }
    }
};

TEST_P(MinMaxLoc, Accuracy)
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    cv::gpu::minMaxLoc(loadMat(mat, useRoi), &minVal, &maxVal, &minLoc, &maxLoc, loadMat(mask, useRoi));

    EXPECT_DOUBLE_EQ(minVal_gold, minVal);
    EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

    int cmpMinVals = memcmp(mat.data + minLoc_gold.y * mat.step + minLoc_gold.x * mat.elemSize(),
                            mat.data + minLoc.y * mat.step + minLoc.x * mat.elemSize(),
                            mat.elemSize());
    int cmpMaxVals = memcmp(mat.data + maxLoc_gold.y * mat.step + maxLoc_gold.x * mat.elemSize(),
                            mat.data + maxLoc.y * mat.step + maxLoc.x * mat.elemSize(),
                            mat.elemSize());

    EXPECT_EQ(0, cmpMinVals);
    EXPECT_EQ(0, cmpMaxVals);
}

INSTANTIATE_TEST_CASE_P(Arithm, MinMaxLoc, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////
// countNonZero

PARAM_TEST_CASE(CountNonZero, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    int n_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        cv::Mat matBase = randomMat(rng, size, CV_8U, 0.0, 1.0, false);
        matBase.convertTo(mat, type);

        n_gold = cv::countNonZero(mat);
    }
};

TEST_P(CountNonZero, Accuracy)
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    int n = cv::gpu::countNonZero(loadMat(mat, useRoi));

    ASSERT_EQ(n_gold, n);
}

INSTANTIATE_TEST_CASE_P(Arithm, CountNonZero, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// sum

PARAM_TEST_CASE(Sum, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, CV_8U, 0.0, 10.0, false);
    }
};

TEST_P(Sum, Simple)
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Scalar sum_gold = cv::sum(mat);

    cv::Scalar sum = cv::gpu::sum(loadMat(mat, useRoi));

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

TEST_P(Sum, Abs)
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Scalar sum_gold = cv::norm(mat, cv::NORM_L1);

    cv::Scalar sum = cv::gpu::absSum(loadMat(mat, useRoi));

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

TEST_P(Sum, Sqr)
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat sqrmat;
    multiply(mat, mat, sqrmat);
    cv::Scalar sum_gold = sum(sqrmat);

    cv::Scalar sum = cv::gpu::sqrSum(loadMat(mat, useRoi));

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Sum, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// addWeighted

PARAM_TEST_CASE(AddWeighted, cv::gpu::DeviceInfo, MatType, MatType, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type1;
    int type2;
    int dtype;
    bool useRoi;

    cv::Size size;
    cv::Mat src1;
    cv::Mat src2;
    double alpha;
    double beta;
    double gamma;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type1 = GET_PARAM(1);
        type2 = GET_PARAM(2);
        dtype = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        src1 = randomMat(rng, size, type1, 0.0, 255.0, false);
        src2 = randomMat(rng, size, type2, 0.0, 255.0, false);

        alpha = rng.uniform(-10.0, 10.0);
        beta = rng.uniform(-10.0, 10.0);
        gamma = rng.uniform(-10.0, 10.0);

        cv::addWeighted(src1, alpha, src2, beta, gamma, dst_gold, dtype);
    }
};

TEST_P(AddWeighted, Accuracy)
{
    if ((src1.depth() == CV_64F || src2.depth() == CV_64F || dst_gold.depth() == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::addWeighted(loadMat(src1, useRoi), alpha, loadMat(src2, useRoi), beta, gamma, dev_dst, dtype);

    dev_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, dtype < CV_32F ? 1.0 : 1e-12);
}

INSTANTIATE_TEST_CASE_P(Arithm, AddWeighted, Combine(
                        ALL_DEVICES,
                        TYPES(CV_8U, CV_64F, 1, 1),
                        TYPES(CV_8U, CV_64F, 1, 1),
                        TYPES(CV_8U, CV_64F, 1, 1),
                        WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// reduce

PARAM_TEST_CASE(Reduce, cv::gpu::DeviceInfo, MatType, int, ReduceOp, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int dim;
    int reduceOp;
    bool useRoi;

    cv::Size size;
    cv::Mat src;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        dim = GET_PARAM(2);
        reduceOp = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 400), rng.uniform(100, 400));

        src = randomMat(rng, size, type, 0.0, 255.0, false);

        cv::reduce(src, dst_gold, dim, reduceOp, reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG ? CV_32F : CV_MAT_DEPTH(type));

        if (dim == 1)
        {
            dst_gold.cols = dst_gold.rows;
            dst_gold.rows = 1;
            dst_gold.step = dst_gold.cols * dst_gold.elemSize();
        }
    }
};

TEST_P(Reduce, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::reduce(loadMat(src, useRoi), dev_dst, dim, reduceOp, reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG ? CV_32F : CV_MAT_DEPTH(type));

    dev_dst.download(dst);

    double norm = reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG ? 1e-1 : 0.0;
    EXPECT_MAT_NEAR(dst_gold, dst, norm);
}

INSTANTIATE_TEST_CASE_P(Arithm, Reduce, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values(0, 1),
                        Values((int)CV_REDUCE_SUM, (int)CV_REDUCE_AVG, (int)CV_REDUCE_MAX, (int)CV_REDUCE_MIN),
                        WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// gemm

PARAM_TEST_CASE(GEMM, cv::gpu::DeviceInfo, MatType, GemmFlags, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flags;
    bool useRoi;

    int size;
    cv::Mat src1;
    cv::Mat src2;
    cv::Mat src3;
    double alpha;
    double beta;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        flags = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = rng.uniform(100, 200);

        src1 = randomMat(rng, cv::Size(size, size), type, -10.0, 10.0, false);
        src2 = randomMat(rng, cv::Size(size, size), type, -10.0, 10.0, false);
        src3 = randomMat(rng, cv::Size(size, size), type, -10.0, 10.0, false);
        alpha = rng.uniform(-10.0, 10.0);
        beta = rng.uniform(-10.0, 10.0);

        cv::gemm(src1, src2, alpha, src3, beta, dst_gold, flags);
    }
};

TEST_P(GEMM, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::gemm(loadMat(src1, useRoi), loadMat(src2, useRoi), alpha, loadMat(src3, useRoi), beta, dev_dst, flags);

    dev_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-1);
}

INSTANTIATE_TEST_CASE_P(Arithm, GEMM, Combine(
                        ALL_DEVICES,
                        Values(CV_32FC1, CV_32FC2),
                        Values(0, (int) cv::GEMM_1_T, (int) cv::GEMM_2_T, (int) cv::GEMM_3_T),
                        WHOLE_SUBMAT));

#endif // HAVE_CUDA
