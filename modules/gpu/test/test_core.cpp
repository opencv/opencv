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

////////////////////////////////////////////////////////////////////////////////
// Merge

PARAM_TEST_CASE(Merge, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
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

GPU_TEST_P(Merge, Accuracy)
{
    std::vector<cv::Mat> src;
    src.reserve(channels);
    for (int i = 0; i < channels; ++i)
        src.push_back(cv::Mat(size, depth, cv::Scalar::all(i)));

    std::vector<cv::gpu::GpuMat> d_src;
    for (int i = 0; i < channels; ++i)
        d_src.push_back(loadMat(src[i], useRoi));

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::merge(d_src, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst;
        cv::gpu::merge(d_src, dst);

        cv::Mat dst_gold;
        cv::merge(src, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Merge, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    testing::Values(1, 2, 3, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Split

PARAM_TEST_CASE(Split, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int channels;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, channels);
    }
};

GPU_TEST_P(Split, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            std::vector<cv::gpu::GpuMat> dst;
            cv::gpu::split(loadMat(src), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        std::vector<cv::gpu::GpuMat> dst;
        cv::gpu::split(loadMat(src, useRoi), dst);

        std::vector<cv::Mat> dst_gold;
        cv::split(src, dst_gold);

        ASSERT_EQ(dst_gold.size(), dst.size());

        for (size_t i = 0; i < dst_gold.size(); ++i)
        {
            EXPECT_MAT_NEAR(dst_gold[i], dst[i], 0.0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Split, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    testing::Values(1, 2, 3, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Add_Array

PARAM_TEST_CASE(Add_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Add_Array, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::add(loadMat(mat1), loadMat(mat2), dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::add(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, cv::gpu::GpuMat(), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::add(mat1, mat2, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Add_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

PARAM_TEST_CASE(Add_Array_Mask, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    int stype;
    int dtype;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, 1);
        dtype = CV_MAKE_TYPE(depth.second, 1);
    }
};

GPU_TEST_P(Add_Array_Mask, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    cv::Mat mask = randomMat(size, CV_8UC1, 0, 2);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::add(loadMat(mat1), loadMat(mat2), dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::add(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::add(mat1, mat2, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Add_Array_Mask, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Add_Scalar

PARAM_TEST_CASE(Add_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Add_Scalar, WithOutMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::add(loadMat(mat), val, dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::add(loadMat(mat, useRoi), val, dst, cv::gpu::GpuMat(), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::add(mat, val, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

GPU_TEST_P(Add_Scalar, WithMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::add(loadMat(mat), val, dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::add(loadMat(mat, useRoi), val, dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::add(mat, val, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Add_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Subtract_Array

PARAM_TEST_CASE(Subtract_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Subtract_Array, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::subtract(loadMat(mat1), loadMat(mat2), dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::subtract(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, cv::gpu::GpuMat(), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::subtract(mat1, mat2, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Subtract_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

PARAM_TEST_CASE(Subtract_Array_Mask, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    int stype;
    int dtype;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, 1);
        dtype = CV_MAKE_TYPE(depth.second, 1);
    }
};

GPU_TEST_P(Subtract_Array_Mask, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::subtract(loadMat(mat1), loadMat(mat2), dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::subtract(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::subtract(mat1, mat2, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Subtract_Array_Mask, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Subtract_Scalar

PARAM_TEST_CASE(Subtract_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Subtract_Scalar, WithOutMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::subtract(loadMat(mat), val, dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::subtract(loadMat(mat, useRoi), val, dst, cv::gpu::GpuMat(), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::subtract(mat, val, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

GPU_TEST_P(Subtract_Scalar, WithMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::subtract(loadMat(mat), val, dst, cv::gpu::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::gpu::subtract(loadMat(mat, useRoi), val, dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::subtract(mat, val, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Subtract_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Array

PARAM_TEST_CASE(Multiply_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Multiply_Array, WithOutScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::multiply(loadMat(mat1), loadMat(mat2), dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        cv::gpu::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat1, mat2, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-2 : 0.0);
    }
}

GPU_TEST_P(Multiply_Array, WithScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::multiply(loadMat(mat1), loadMat(mat2), dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        cv::gpu::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat1, mat2, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 2.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Multiply_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Array_Special

PARAM_TEST_CASE(Multiply_Array_Special, cv::gpu::DeviceInfo, cv::Size, UseRoi)
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

GPU_TEST_P(Multiply_Array_Special, Case_8UC4x_32FC1)
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

GPU_TEST_P(Multiply_Array_Special, Case_16SC4x_32FC1)
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

INSTANTIATE_TEST_CASE_P(GPU_Core, Multiply_Array_Special, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Scalar

PARAM_TEST_CASE(Multiply_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Multiply_Scalar, WithOutScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::multiply(loadMat(mat), val, dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::gpu::multiply(loadMat(mat, useRoi), val, dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat, val, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
    }
}


GPU_TEST_P(Multiply_Scalar, WithScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::multiply(loadMat(mat), val, dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::gpu::multiply(loadMat(mat, useRoi), val, dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat, val, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Multiply_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Array

PARAM_TEST_CASE(Divide_Array, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Divide_Array, WithOutScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype, 1.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::divide(loadMat(mat1), loadMat(mat2), dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        cv::gpu::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat1, mat2, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

GPU_TEST_P(Divide_Array, WithScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype, 1.0, 255.0);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::divide(loadMat(mat1), loadMat(mat2), dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dtype, useRoi);
        cv::gpu::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat1, mat2, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-2 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Divide_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Array_Special

PARAM_TEST_CASE(Divide_Array_Special, cv::gpu::DeviceInfo, cv::Size, UseRoi)
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

GPU_TEST_P(Divide_Array_Special, Case_8UC4x_32FC1)
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

GPU_TEST_P(Divide_Array_Special, Case_16SC4x_32FC1)
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

INSTANTIATE_TEST_CASE_P(GPU_Core, Divide_Array_Special, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Scalar

PARAM_TEST_CASE(Divide_Scalar, cv::gpu::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Divide_Scalar, WithOutScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(1.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::divide(loadMat(mat), val, dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::gpu::divide(loadMat(mat, useRoi), val, dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat, val, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

GPU_TEST_P(Divide_Scalar, WithScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(1.0, 255.0);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::divide(loadMat(mat), val, dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::gpu::divide(loadMat(mat, useRoi), val, dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat, val, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-2 : 1.0);
    }
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
    std::pair<MatDepth, MatDepth> depth;
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

GPU_TEST_P(Divide_Scalar_Inv, Accuracy)
{
    double scale = randomDouble(0.0, 255.0);
    cv::Mat mat = randomMat(size, depth.first, 1.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::divide(scale, loadMat(mat), dst, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::gpu::divide(scale, loadMat(mat, useRoi), dst, depth.second);

        cv::Mat dst_gold;
        cv::divide(scale, mat, dst_gold, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
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

GPU_TEST_P(AbsDiff, Array)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::absdiff(loadMat(src1), loadMat(src2), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
        cv::gpu::absdiff(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

        cv::Mat dst_gold;
        cv::absdiff(src1, src2, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

GPU_TEST_P(AbsDiff, Scalar)
{
    cv::Mat src = randomMat(size, depth);
    cv::Scalar val = randomScalar(0.0, 255.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::absdiff(loadMat(src), val, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
        cv::gpu::absdiff(loadMat(src, useRoi), val, dst);

        cv::Mat dst_gold;
        cv::absdiff(src, val, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, depth <= CV_32F ? 1.0 : 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, AbsDiff, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Abs

PARAM_TEST_CASE(Abs, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(Abs, Accuracy)
{
    cv::Mat src = randomMat(size, depth);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::abs(loadMat(src, useRoi), dst);

    cv::Mat dst_gold = cv::abs(src);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Abs, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_16S), MatDepth(CV_32F)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Sqr

PARAM_TEST_CASE(Sqr, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(Sqr, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 0, depth == CV_8U ? 16 : 255);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::sqr(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::multiply(src, src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Sqr, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32F)),
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

PARAM_TEST_CASE(Sqrt, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(Sqrt, Accuracy)
{
    cv::Mat src = randomMat(size, depth);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::sqrt(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    sqrtGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-5);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Sqrt, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32F)),
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

PARAM_TEST_CASE(Log, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(Log, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 1.0, 255.0);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::log(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    logGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-6);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Log, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32F)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Exp

namespace
{
    template <typename T> void expImpl(const cv::Mat& src, cv::Mat& dst)
    {
        dst.create(src.size(), src.type());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
                dst.at<T>(y, x) = cv::saturate_cast<T>(static_cast<int>(std::exp(static_cast<float>(src.at<T>(y, x)))));
        }
    }
    void expImpl_float(const cv::Mat& src, cv::Mat& dst)
    {
        dst.create(src.size(), src.type());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
                dst.at<float>(y, x) = std::exp(static_cast<float>(src.at<float>(y, x)));
        }
    }

    void expGold(const cv::Mat& src, cv::Mat& dst)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Mat& dst);

        const func_t funcs[] =
        {
            expImpl<uchar>, expImpl<schar>, expImpl<ushort>, expImpl<short>,
            expImpl<int>, expImpl_float
        };

        funcs[src.depth()](src, dst);
    }
}

PARAM_TEST_CASE(Exp, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(Exp, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 0.0, 10.0);

    cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
    cv::gpu::exp(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    expGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-2);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Exp, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32F)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Compare_Array

CV_ENUM(CmpCode, CMP_EQ, CMP_NE, CMP_GT, CMP_GE, CMP_LT, CMP_LE)

PARAM_TEST_CASE(Compare_Array, cv::gpu::DeviceInfo, cv::Size, MatDepth, CmpCode, UseRoi)
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

GPU_TEST_P(Compare_Array, Accuracy)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::compare(loadMat(src1), loadMat(src2), dst, cmp_code);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, CV_8UC1, useRoi);
        cv::gpu::compare(loadMat(src1, useRoi), loadMat(src2, useRoi), dst, cmp_code);

        cv::Mat dst_gold;
        cv::compare(src1, src2, dst_gold, cmp_code);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Compare_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    CmpCode::all(),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Compare_Scalar

namespace
{
    template <template <typename> class Op, typename T>
    void compareScalarImpl(const cv::Mat& src, cv::Scalar sc, cv::Mat& dst)
    {
        Op<T> op;

        const int cn = src.channels();

        dst.create(src.size(), CV_MAKE_TYPE(CV_8U, cn));

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    T src_val = src.at<T>(y, x * cn + c);
                    T sc_val = cv::saturate_cast<T>(sc.val[c]);
                    dst.at<uchar>(y, x * cn + c) = static_cast<uchar>(static_cast<int>(op(src_val, sc_val)) * 255);
                }
            }
        }
    }

    void compareScalarGold(const cv::Mat& src, cv::Scalar sc, cv::Mat& dst, int cmpop)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Scalar sc, cv::Mat& dst);
        static const func_t funcs[7][6] =
        {
            {compareScalarImpl<std::equal_to, unsigned char> , compareScalarImpl<std::greater, unsigned char> , compareScalarImpl<std::greater_equal, unsigned char> , compareScalarImpl<std::less, unsigned char> , compareScalarImpl<std::less_equal, unsigned char> , compareScalarImpl<std::not_equal_to, unsigned char> },
            {compareScalarImpl<std::equal_to, signed char>   , compareScalarImpl<std::greater, signed char>   , compareScalarImpl<std::greater_equal, signed char>   , compareScalarImpl<std::less, signed char>   , compareScalarImpl<std::less_equal, signed char>   , compareScalarImpl<std::not_equal_to, signed char>   },
            {compareScalarImpl<std::equal_to, unsigned short>, compareScalarImpl<std::greater, unsigned short>, compareScalarImpl<std::greater_equal, unsigned short>, compareScalarImpl<std::less, unsigned short>, compareScalarImpl<std::less_equal, unsigned short>, compareScalarImpl<std::not_equal_to, unsigned short>},
            {compareScalarImpl<std::equal_to, short>         , compareScalarImpl<std::greater, short>         , compareScalarImpl<std::greater_equal, short>         , compareScalarImpl<std::less, short>         , compareScalarImpl<std::less_equal, short>         , compareScalarImpl<std::not_equal_to, short>         },
            {compareScalarImpl<std::equal_to, int>           , compareScalarImpl<std::greater, int>           , compareScalarImpl<std::greater_equal, int>           , compareScalarImpl<std::less, int>           , compareScalarImpl<std::less_equal, int>           , compareScalarImpl<std::not_equal_to, int>           },
            {compareScalarImpl<std::equal_to, float>         , compareScalarImpl<std::greater, float>         , compareScalarImpl<std::greater_equal, float>         , compareScalarImpl<std::less, float>         , compareScalarImpl<std::less_equal, float>         , compareScalarImpl<std::not_equal_to, float>         },
            {compareScalarImpl<std::equal_to, double>        , compareScalarImpl<std::greater, double>        , compareScalarImpl<std::greater_equal, double>        , compareScalarImpl<std::less, double>        , compareScalarImpl<std::less_equal, double>        , compareScalarImpl<std::not_equal_to, double>        }
        };

        funcs[src.depth()][cmpop](src, sc, dst);
    }
}

PARAM_TEST_CASE(Compare_Scalar, cv::gpu::DeviceInfo, cv::Size, MatType, CmpCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int cmp_code;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        cmp_code = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(Compare_Scalar, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Scalar sc = randomScalar(0.0, 255.0);

    if (src.depth() < CV_32F)
    {
        sc.val[0] = cvRound(sc.val[0]);
        sc.val[1] = cvRound(sc.val[1]);
        sc.val[2] = cvRound(sc.val[2]);
        sc.val[3] = cvRound(sc.val[3]);
    }

    if (src.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::compare(loadMat(src), sc, dst, cmp_code);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, CV_MAKE_TYPE(CV_8U, src.channels()), useRoi);

        cv::gpu::compare(loadMat(src, useRoi), sc, dst, cmp_code);

        cv::Mat dst_gold;
        compareScalarGold(src, sc, dst_gold, cmp_code);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Compare_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    TYPES(CV_8U, CV_64F, 1, 4),
    CmpCode::all(),
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

GPU_TEST_P(Bitwise_Array, Not)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_not(loadMat(src1), dst);

    cv::Mat dst_gold = ~src1;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

GPU_TEST_P(Bitwise_Array, Or)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_or(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 | src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

GPU_TEST_P(Bitwise_Array, And)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_and(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 & src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

GPU_TEST_P(Bitwise_Array, Xor)
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

PARAM_TEST_CASE(Bitwise_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
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
        cv::Scalar_<int> ival = randomScalar(0.0, std::numeric_limits<int>::max());
        val = ival;
    }
};

GPU_TEST_P(Bitwise_Scalar, Or)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_or(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_or(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

GPU_TEST_P(Bitwise_Scalar, And)
{
    cv::gpu::GpuMat dst;
    cv::gpu::bitwise_and(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_and(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

GPU_TEST_P(Bitwise_Scalar, Xor)
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
    IMAGE_CHANNELS));

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

PARAM_TEST_CASE(RShift, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
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

GPU_TEST_P(RShift, Accuracy)
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
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_8S),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32S)),
    IMAGE_CHANNELS,
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

PARAM_TEST_CASE(LShift, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
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

GPU_TEST_P(LShift, Accuracy)
{
    int type = CV_MAKE_TYPE(depth, channels);
    cv::Mat src = randomMat(size, type);
    cv::Scalar_<int> val = randomScalar(0.0, 8.0);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::lshift(loadMat(src, useRoi), val, dst);

    cv::Mat dst_gold;
    lhiftGold(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, LShift, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32S)),
    IMAGE_CHANNELS,
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

GPU_TEST_P(Min, Array)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::min(loadMat(src1), loadMat(src2), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
        cv::gpu::min(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

        cv::Mat dst_gold = cv::min(src1, src2);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

GPU_TEST_P(Min, Scalar)
{
    cv::Mat src = randomMat(size, depth);
    double val = randomDouble(0.0, 255.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::min(loadMat(src), val, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
        cv::gpu::min(loadMat(src, useRoi), val, dst);

        cv::Mat dst_gold = cv::min(src, val);

        EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-5);
    }
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

GPU_TEST_P(Max, Array)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::max(loadMat(src1), loadMat(src2), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
        cv::gpu::max(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

        cv::Mat dst_gold = cv::max(src1, src2);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

GPU_TEST_P(Max, Scalar)
{
    cv::Mat src = randomMat(size, depth);
    double val = randomDouble(0.0, 255.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::max(loadMat(src), val, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
        cv::gpu::max(loadMat(src, useRoi), val, dst);

        cv::Mat dst_gold = cv::max(src, val);

        EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Max, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Pow

PARAM_TEST_CASE(Pow, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(Pow, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 0.0, 10.0);
    double power = randomDouble(2.0, 4.0);

    if (src.depth() < CV_32F)
        power = static_cast<int>(power);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::pow(loadMat(src), power, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, depth, useRoi);
        cv::gpu::pow(loadMat(src, useRoi), power, dst);

        cv::Mat dst_gold;
        cv::pow(src, power, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 0.0 : 1e-1);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Pow, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// AddWeighted

PARAM_TEST_CASE(AddWeighted, cv::gpu::DeviceInfo, cv::Size, MatDepth, MatDepth, MatDepth, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth1;
    int depth2;
    int dst_depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth1 = GET_PARAM(2);
        depth2 = GET_PARAM(3);
        dst_depth = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(AddWeighted, Accuracy)
{
    cv::Mat src1 = randomMat(size, depth1);
    cv::Mat src2 = randomMat(size, depth2);
    double alpha = randomDouble(-10.0, 10.0);
    double beta = randomDouble(-10.0, 10.0);
    double gamma = randomDouble(-10.0, 10.0);

    if ((depth1 == CV_64F || depth2 == CV_64F || dst_depth == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::addWeighted(loadMat(src1), alpha, loadMat(src2), beta, gamma, dst, dst_depth);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, dst_depth, useRoi);
        cv::gpu::addWeighted(loadMat(src1, useRoi), alpha, loadMat(src2, useRoi), beta, gamma, dst, dst_depth);

        cv::Mat dst_gold;
        cv::addWeighted(src1, alpha, src2, beta, gamma, dst_gold, dst_depth);

        EXPECT_MAT_NEAR(dst_gold, dst, dst_depth < CV_32F ? 2.0 : 1e-3);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, AddWeighted, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    ALL_DEPTH,
    ALL_DEPTH,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// GEMM

#ifdef HAVE_CUBLAS

CV_FLAGS(GemmFlags, 0, GEMM_1_T, GEMM_2_T, GEMM_3_T);
#define ALL_GEMM_FLAGS testing::Values(GemmFlags(0), GemmFlags(cv::GEMM_1_T), GemmFlags(cv::GEMM_2_T), GemmFlags(cv::GEMM_3_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_3_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T | cv::GEMM_3_T))

PARAM_TEST_CASE(GEMM, cv::gpu::DeviceInfo, cv::Size, MatType, GemmFlags, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int flags;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        flags = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(GEMM, Accuracy)
{
    cv::Mat src1 = randomMat(size, type, -10.0, 10.0);
    cv::Mat src2 = randomMat(size, type, -10.0, 10.0);
    cv::Mat src3 = randomMat(size, type, -10.0, 10.0);
    double alpha = randomDouble(-10.0, 10.0);
    double beta = randomDouble(-10.0, 10.0);

    if (CV_MAT_DEPTH(type) == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::gemm(loadMat(src1), loadMat(src2), alpha, loadMat(src3), beta, dst, flags);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else if (type == CV_64FC2 && flags != 0)
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::gemm(loadMat(src1), loadMat(src2), alpha, loadMat(src3), beta, dst, flags);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, type, useRoi);
        cv::gpu::gemm(loadMat(src1, useRoi), loadMat(src2, useRoi), alpha, loadMat(src3, useRoi), beta, dst, flags);

        cv::Mat dst_gold;
        cv::gemm(src1, src2, alpha, src3, beta, dst_gold, flags);

        EXPECT_MAT_NEAR(dst_gold, dst, CV_MAT_DEPTH(type) == CV_32F ? 1e-1 : 1e-10);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, GEMM, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_32FC1), MatType(CV_32FC2), MatType(CV_64FC1), MatType(CV_64FC2)),
    ALL_GEMM_FLAGS,
    WHOLE_SUBMAT));

#endif // HAVE_CUBLAS

////////////////////////////////////////////////////////////////////////////////
// Transpose

PARAM_TEST_CASE(Transpose, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
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

GPU_TEST_P(Transpose, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    if (CV_MAT_DEPTH(type) == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::transpose(loadMat(src), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(cv::Size(size.height, size.width), type, useRoi);
        cv::gpu::transpose(loadMat(src, useRoi), dst);

        cv::Mat dst_gold;
        cv::transpose(src, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Transpose, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1),
                    MatType(CV_8UC4),
                    MatType(CV_16UC2),
                    MatType(CV_16SC2),
                    MatType(CV_32SC1),
                    MatType(CV_32SC2),
                    MatType(CV_64FC1)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Flip

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)
#define ALL_FLIP_CODES testing::Values(FlipCode(FLIP_BOTH), FlipCode(FLIP_X), FlipCode(FLIP_Y))

PARAM_TEST_CASE(Flip, cv::gpu::DeviceInfo, cv::Size, MatType, FlipCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int flip_code;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        flip_code = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(Flip, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::flip(loadMat(src, useRoi), dst, flip_code);

    cv::Mat dst_gold;
    cv::flip(src, dst_gold, flip_code);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Flip, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1),
                    MatType(CV_8UC3),
                    MatType(CV_8UC4),
                    MatType(CV_16UC1),
                    MatType(CV_16UC3),
                    MatType(CV_16UC4),
                    MatType(CV_32SC1),
                    MatType(CV_32SC3),
                    MatType(CV_32SC4),
                    MatType(CV_32FC1),
                    MatType(CV_32FC3),
                    MatType(CV_32FC4)),
    ALL_FLIP_CODES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// LUT

PARAM_TEST_CASE(LUT, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
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

GPU_TEST_P(LUT, OneChannel)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat lut = randomMat(cv::Size(256, 1), CV_8UC1);

    cv::gpu::GpuMat dst = createMat(size, CV_MAKE_TYPE(lut.depth(), src.channels()));
    cv::gpu::LUT(loadMat(src, useRoi), lut, dst);

    cv::Mat dst_gold;
    cv::LUT(src, lut, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

GPU_TEST_P(LUT, MultiChannel)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat lut = randomMat(cv::Size(256, 1), CV_MAKE_TYPE(CV_8U, src.channels()));

    cv::gpu::GpuMat dst = createMat(size, CV_MAKE_TYPE(lut.depth(), src.channels()), useRoi);
    cv::gpu::LUT(loadMat(src, useRoi), lut, dst);

    cv::Mat dst_gold;
    cv::LUT(src, lut, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, LUT, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Magnitude

PARAM_TEST_CASE(Magnitude, cv::gpu::DeviceInfo, cv::Size, UseRoi)
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

GPU_TEST_P(Magnitude, NPP)
{
    cv::Mat src = randomMat(size, CV_32FC2);

    cv::gpu::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::gpu::magnitude(loadMat(src, useRoi), dst);

    cv::Mat arr[2];
    cv::split(src, arr);
    cv::Mat dst_gold;
    cv::magnitude(arr[0], arr[1], dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

GPU_TEST_P(Magnitude, Sqr_NPP)
{
    cv::Mat src = randomMat(size, CV_32FC2);

    cv::gpu::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::gpu::magnitudeSqr(loadMat(src, useRoi), dst);

    cv::Mat arr[2];
    cv::split(src, arr);
    cv::Mat dst_gold;
    cv::magnitude(arr[0], arr[1], dst_gold);
    cv::multiply(dst_gold, dst_gold, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-1);
}

GPU_TEST_P(Magnitude, Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::gpu::magnitude(loadMat(x, useRoi), loadMat(y, useRoi), dst);

    cv::Mat dst_gold;
    cv::magnitude(x, y, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

GPU_TEST_P(Magnitude, Sqr_Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::gpu::magnitudeSqr(loadMat(x, useRoi), loadMat(y, useRoi), dst);

    cv::Mat dst_gold;
    cv::magnitude(x, y, dst_gold);
    cv::multiply(dst_gold, dst_gold, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-1);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Magnitude, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Phase

namespace
{
    IMPLEMENT_PARAM_CLASS(AngleInDegrees, bool)
}

PARAM_TEST_CASE(Phase, cv::gpu::DeviceInfo, cv::Size, AngleInDegrees, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    bool angleInDegrees;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        angleInDegrees = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(Phase, Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::gpu::phase(loadMat(x, useRoi), loadMat(y, useRoi), dst, angleInDegrees);

    cv::Mat dst_gold;
    cv::phase(x, y, dst_gold, angleInDegrees);

    EXPECT_MAT_NEAR(dst_gold, dst, angleInDegrees ? 1e-2 : 1e-3);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Phase, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(AngleInDegrees(false), AngleInDegrees(true)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// CartToPolar

PARAM_TEST_CASE(CartToPolar, cv::gpu::DeviceInfo, cv::Size, AngleInDegrees, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    bool angleInDegrees;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        angleInDegrees = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(CartToPolar, Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat mag = createMat(size, CV_32FC1, useRoi);
    cv::gpu::GpuMat angle = createMat(size, CV_32FC1, useRoi);
    cv::gpu::cartToPolar(loadMat(x, useRoi), loadMat(y, useRoi), mag, angle, angleInDegrees);

    cv::Mat mag_gold;
    cv::Mat angle_gold;
    cv::cartToPolar(x, y, mag_gold, angle_gold, angleInDegrees);

    EXPECT_MAT_NEAR(mag_gold, mag, 1e-4);
    EXPECT_MAT_NEAR(angle_gold, angle, angleInDegrees ? 1e-2 : 1e-3);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, CartToPolar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(AngleInDegrees(false), AngleInDegrees(true)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// polarToCart

PARAM_TEST_CASE(PolarToCart, cv::gpu::DeviceInfo, cv::Size, AngleInDegrees, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    bool angleInDegrees;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        angleInDegrees = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(PolarToCart, Accuracy)
{
    cv::Mat magnitude = randomMat(size, CV_32FC1);
    cv::Mat angle = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat x = createMat(size, CV_32FC1, useRoi);
    cv::gpu::GpuMat y = createMat(size, CV_32FC1, useRoi);
    cv::gpu::polarToCart(loadMat(magnitude, useRoi), loadMat(angle, useRoi), x, y, angleInDegrees);

    cv::Mat x_gold;
    cv::Mat y_gold;
    cv::polarToCart(magnitude, angle, x_gold, y_gold, angleInDegrees);

    EXPECT_MAT_NEAR(x_gold, x, 1e-4);
    EXPECT_MAT_NEAR(y_gold, y, 1e-4);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, PolarToCart, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(AngleInDegrees(false), AngleInDegrees(true)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MeanStdDev

PARAM_TEST_CASE(MeanStdDev, cv::gpu::DeviceInfo, cv::Size, UseRoi)
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

GPU_TEST_P(MeanStdDev, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    if (!supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_13))
    {
        try
        {
            cv::Scalar mean;
            cv::Scalar stddev;
            cv::gpu::meanStdDev(loadMat(src, useRoi), mean, stddev);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::gpu::meanStdDev(loadMat(src, useRoi), mean, stddev);

        cv::Scalar mean_gold;
        cv::Scalar stddev_gold;
        cv::meanStdDev(src, mean_gold, stddev_gold);

        EXPECT_SCALAR_NEAR(mean_gold, mean, 1e-5);
        EXPECT_SCALAR_NEAR(stddev_gold, stddev, 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, MeanStdDev, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Norm

PARAM_TEST_CASE(Norm, cv::gpu::DeviceInfo, cv::Size, MatDepth, NormCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int normCode;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        normCode = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(Norm, Accuracy)
{
    cv::Mat src = randomMat(size, depth);
    cv::Mat mask = randomMat(size, CV_8UC1, 0, 2);

    cv::gpu::GpuMat d_buf;
    double val = cv::gpu::norm(loadMat(src, useRoi), normCode, loadMat(mask, useRoi), d_buf);

    double val_gold = cv::norm(src, normCode, mask);

    EXPECT_NEAR(val_gold, val, depth < CV_32F ? 0.0 : 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Norm, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_8S),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32S),
                    MatDepth(CV_32F)),
    testing::Values(NormCode(cv::NORM_L1), NormCode(cv::NORM_L2), NormCode(cv::NORM_INF)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// normDiff

PARAM_TEST_CASE(NormDiff, cv::gpu::DeviceInfo, cv::Size, NormCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int normCode;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        normCode = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(NormDiff, Accuracy)
{
    cv::Mat src1 = randomMat(size, CV_8UC1);
    cv::Mat src2 = randomMat(size, CV_8UC1);

    double val = cv::gpu::norm(loadMat(src1, useRoi), loadMat(src2, useRoi), normCode);

    double val_gold = cv::norm(src1, src2, normCode);

    EXPECT_NEAR(val_gold, val, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, NormDiff, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(NormCode(cv::NORM_L1), NormCode(cv::NORM_L2), NormCode(cv::NORM_INF)),
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Sum

namespace
{
    template <typename T>
    cv::Scalar absSumImpl(const cv::Mat& src)
    {
        const int cn = src.channels();

        cv::Scalar sum = cv::Scalar::all(0);

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                    sum[c] += std::abs(src.at<T>(y, x * cn + c));
            }
        }

        return sum;
    }

    cv::Scalar absSumGold(const cv::Mat& src)
    {
        typedef cv::Scalar (*func_t)(const cv::Mat& src);

        static const func_t funcs[] =
        {
            absSumImpl<uchar>,
            absSumImpl<schar>,
            absSumImpl<ushort>,
            absSumImpl<short>,
            absSumImpl<int>,
            absSumImpl<float>,
            absSumImpl<double>
        };

        return funcs[src.depth()](src);
    }

    template <typename T>
    cv::Scalar sqrSumImpl(const cv::Mat& src)
    {
        const int cn = src.channels();

        cv::Scalar sum = cv::Scalar::all(0);

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    const T val = src.at<T>(y, x * cn + c);
                    sum[c] += val * val;
                }
            }
        }

        return sum;
    }

    cv::Scalar sqrSumGold(const cv::Mat& src)
    {
        typedef cv::Scalar (*func_t)(const cv::Mat& src);

        static const func_t funcs[] =
        {
            sqrSumImpl<uchar>,
            sqrSumImpl<schar>,
            sqrSumImpl<ushort>,
            sqrSumImpl<short>,
            sqrSumImpl<int>,
            sqrSumImpl<float>,
            sqrSumImpl<double>
        };

        return funcs[src.depth()](src);
    }
}

PARAM_TEST_CASE(Sum, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    cv::Mat src;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        src = randomMat(size, type, -128.0, 128.0);
    }
};

GPU_TEST_P(Sum, Simple)
{
    cv::Scalar val = cv::gpu::sum(loadMat(src, useRoi));

    cv::Scalar val_gold = cv::sum(src);

    EXPECT_SCALAR_NEAR(val_gold, val, CV_MAT_DEPTH(type) < CV_32F ? 0.0 : 0.5);
}

GPU_TEST_P(Sum, Abs)
{
    cv::Scalar val = cv::gpu::absSum(loadMat(src, useRoi));

    cv::Scalar val_gold = absSumGold(src);

    EXPECT_SCALAR_NEAR(val_gold, val, CV_MAT_DEPTH(type) < CV_32F ? 0.0 : 0.5);
}

GPU_TEST_P(Sum, Sqr)
{
    cv::Scalar val = cv::gpu::sqrSum(loadMat(src, useRoi));

    cv::Scalar val_gold = sqrSumGold(src);

    EXPECT_SCALAR_NEAR(val_gold, val, CV_MAT_DEPTH(type) < CV_32F ? 0.0 : 0.5);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Sum, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    TYPES(CV_8U, CV_64F, 1, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MinMax

PARAM_TEST_CASE(MinMax, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(MinMax, WithoutMask)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::gpu::minMax(loadMat(src), &minVal, &maxVal);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::gpu::minMax(loadMat(src, useRoi), &minVal, &maxVal);

        double minVal_gold, maxVal_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
    }
}

GPU_TEST_P(MinMax, WithMask)
{
    cv::Mat src = randomMat(size, depth);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::gpu::minMax(loadMat(src), &minVal, &maxVal, loadMat(mask));
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::gpu::minMax(loadMat(src, useRoi), &minVal, &maxVal, loadMat(mask, useRoi));

        double minVal_gold, maxVal_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, 0, 0, mask);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
    }
}

GPU_TEST_P(MinMax, NullPtr)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::gpu::minMax(loadMat(src), &minVal, 0);
            cv::gpu::minMax(loadMat(src), 0, &maxVal);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::gpu::minMax(loadMat(src, useRoi), &minVal, 0);
        cv::gpu::minMax(loadMat(src, useRoi), 0, &maxVal);

        double minVal_gold, maxVal_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, 0, 0);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, MinMax, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MinMaxLoc

namespace
{
    template <typename T>
    void expectEqualImpl(const cv::Mat& src, cv::Point loc_gold, cv::Point loc)
    {
        EXPECT_EQ(src.at<T>(loc_gold.y, loc_gold.x), src.at<T>(loc.y, loc.x));
    }

    void expectEqual(const cv::Mat& src, cv::Point loc_gold, cv::Point loc)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Point loc_gold, cv::Point loc);

        static const func_t funcs[] =
        {
            expectEqualImpl<uchar>,
            expectEqualImpl<schar>,
            expectEqualImpl<ushort>,
            expectEqualImpl<short>,
            expectEqualImpl<int>,
            expectEqualImpl<float>,
            expectEqualImpl<double>
        };

        funcs[src.depth()](src, loc_gold, loc);
    }
}

PARAM_TEST_CASE(MinMaxLoc, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(MinMaxLoc, WithoutMask)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::gpu::minMaxLoc(loadMat(src), &minVal, &maxVal, &minLoc, &maxLoc);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::gpu::minMaxLoc(loadMat(src, useRoi), &minVal, &maxVal, &minLoc, &maxLoc);

        double minVal_gold, maxVal_gold;
        cv::Point minLoc_gold, maxLoc_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

        expectEqual(src, minLoc_gold, minLoc);
        expectEqual(src, maxLoc_gold, maxLoc);
    }
}

GPU_TEST_P(MinMaxLoc, WithMask)
{
    cv::Mat src = randomMat(size, depth);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::gpu::minMaxLoc(loadMat(src), &minVal, &maxVal, &minLoc, &maxLoc, loadMat(mask));
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::gpu::minMaxLoc(loadMat(src, useRoi), &minVal, &maxVal, &minLoc, &maxLoc, loadMat(mask, useRoi));

        double minVal_gold, maxVal_gold;
        cv::Point minLoc_gold, maxLoc_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold, mask);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

        expectEqual(src, minLoc_gold, minLoc);
        expectEqual(src, maxLoc_gold, maxLoc);
    }
}

GPU_TEST_P(MinMaxLoc, NullPtr)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::gpu::minMaxLoc(loadMat(src, useRoi), &minVal, 0, 0, 0);
            cv::gpu::minMaxLoc(loadMat(src, useRoi), 0, &maxVal, 0, 0);
            cv::gpu::minMaxLoc(loadMat(src, useRoi), 0, 0, &minLoc, 0);
            cv::gpu::minMaxLoc(loadMat(src, useRoi), 0, 0, 0, &maxLoc);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::gpu::minMaxLoc(loadMat(src, useRoi), &minVal, 0, 0, 0);
        cv::gpu::minMaxLoc(loadMat(src, useRoi), 0, &maxVal, 0, 0);
        cv::gpu::minMaxLoc(loadMat(src, useRoi), 0, 0, &minLoc, 0);
        cv::gpu::minMaxLoc(loadMat(src, useRoi), 0, 0, 0, &maxLoc);

        double minVal_gold, maxVal_gold;
        cv::Point minLoc_gold, maxLoc_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

        expectEqual(src, minLoc_gold, minLoc);
        expectEqual(src, maxLoc_gold, maxLoc);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, MinMaxLoc, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////
// CountNonZero

PARAM_TEST_CASE(CountNonZero, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

GPU_TEST_P(CountNonZero, Accuracy)
{
    cv::Mat srcBase = randomMat(size, CV_8U, 0.0, 1.5);
    cv::Mat src;
    srcBase.convertTo(src, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
    {
        try
        {
            cv::gpu::countNonZero(loadMat(src));
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        int val = cv::gpu::countNonZero(loadMat(src, useRoi));

        int val_gold = cv::countNonZero(src);

        ASSERT_EQ(val_gold, val);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Core, CountNonZero, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Reduce

CV_ENUM(ReduceCode, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)
#define ALL_REDUCE_CODES testing::Values(ReduceCode(CV_REDUCE_SUM), ReduceCode(CV_REDUCE_AVG), ReduceCode(CV_REDUCE_MAX), ReduceCode(CV_REDUCE_MIN))

PARAM_TEST_CASE(Reduce, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels, ReduceCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int channels;
    int reduceOp;
    bool useRoi;

    int type;
    int dst_depth;
    int dst_type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        reduceOp = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, channels);

        if (reduceOp == CV_REDUCE_MAX || reduceOp == CV_REDUCE_MIN)
            dst_depth = depth;
        else if (reduceOp == CV_REDUCE_SUM)
            dst_depth = depth == CV_8U ? CV_32S : depth < CV_64F ? CV_32F : depth;
        else
            dst_depth = depth < CV_32F ? CV_32F : depth;

        dst_type = CV_MAKE_TYPE(dst_depth, channels);
    }

};

GPU_TEST_P(Reduce, Rows)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(cv::Size(src.cols, 1), dst_type, useRoi);
    cv::gpu::reduce(loadMat(src, useRoi), dst, 0, reduceOp, dst_depth);

    cv::Mat dst_gold;
    cv::reduce(src, dst_gold, 0, reduceOp, dst_depth);

    EXPECT_MAT_NEAR(dst_gold, dst, dst_depth < CV_32F ? 0.0 : 0.02);
}

GPU_TEST_P(Reduce, Cols)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(cv::Size(src.rows, 1), dst_type, useRoi);
    cv::gpu::reduce(loadMat(src, useRoi), dst, 1, reduceOp, dst_depth);

    cv::Mat dst_gold;
    cv::reduce(src, dst_gold, 1, reduceOp, dst_depth);
    dst_gold.cols = dst_gold.rows;
    dst_gold.rows = 1;
    dst_gold.step = dst_gold.cols * dst_gold.elemSize();

    EXPECT_MAT_NEAR(dst_gold, dst, dst_depth < CV_32F ? 0.0 : 0.02);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Reduce, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32F),
                    MatDepth(CV_64F)),
    ALL_CHANNELS,
    ALL_REDUCE_CODES,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Normalize

PARAM_TEST_CASE(Normalize, cv::gpu::DeviceInfo, cv::Size, MatDepth, NormCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int norm_type;
    bool useRoi;

    double alpha;
    double beta;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        norm_type = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        alpha = 1;
        beta = 0;
    }

};

GPU_TEST_P(Normalize, WithOutMask)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::normalize(loadMat(src, useRoi), dst, alpha, beta, norm_type, type);

    cv::Mat dst_gold;
    cv::normalize(src, dst_gold, alpha, beta, norm_type, type);

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

GPU_TEST_P(Normalize, WithMask)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat mask = randomMat(size, CV_8UC1, 0, 2);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    dst.setTo(cv::Scalar::all(0));
    cv::gpu::normalize(loadMat(src, useRoi), dst, alpha, beta, norm_type, type, loadMat(mask, useRoi));

    cv::Mat dst_gold(size, type);
    dst_gold.setTo(cv::Scalar::all(0));
    cv::normalize(src, dst_gold, alpha, beta, norm_type, type, mask);

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Core, Normalize, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    testing::Values(NormCode(cv::NORM_L1), NormCode(cv::NORM_L2), NormCode(cv::NORM_INF), NormCode(cv::NORM_MINMAX)),
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
