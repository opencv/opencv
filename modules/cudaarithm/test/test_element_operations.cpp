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
// Add_Array

PARAM_TEST_CASE(Add_Array, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

CUDA_TEST_P(Add_Array, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::add(loadMat(mat1), loadMat(mat2), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::add(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, cv::cuda::GpuMat(), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::add(mat1, mat2, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Add_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

PARAM_TEST_CASE(Add_Array_Mask, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, 1);
        dtype = CV_MAKE_TYPE(depth.second, 1);
    }
};

CUDA_TEST_P(Add_Array_Mask, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    cv::Mat mask = randomMat(size, CV_8UC1, 0, 2);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::add(loadMat(mat1), loadMat(mat2), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::add(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::add(mat1, mat2, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Add_Array_Mask, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Add_Scalar

PARAM_TEST_CASE(Add_Scalar, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Add_Scalar, WithOutMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::add(loadMat(mat), val, dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::add(loadMat(mat, useRoi), val, dst, cv::cuda::GpuMat(), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::add(mat, val, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

CUDA_TEST_P(Add_Scalar, WithMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::add(loadMat(mat), val, dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::add(loadMat(mat, useRoi), val, dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::add(mat, val, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Add_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Add_Scalar_First

PARAM_TEST_CASE(Add_Scalar_First, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Add_Scalar_First, WithOutMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::add(val, loadMat(mat), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::add(val, loadMat(mat, useRoi), dst, cv::cuda::GpuMat(), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::add(val, mat, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

CUDA_TEST_P(Add_Scalar_First, WithMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::add(val, loadMat(mat), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::add(val, loadMat(mat, useRoi), dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::add(val, mat, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Add_Scalar_First, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Subtract_Array

PARAM_TEST_CASE(Subtract_Array, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

CUDA_TEST_P(Subtract_Array, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::subtract(loadMat(mat1), loadMat(mat2), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::subtract(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, cv::cuda::GpuMat(), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::subtract(mat1, mat2, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Subtract_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

PARAM_TEST_CASE(Subtract_Array_Mask, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, 1);
        dtype = CV_MAKE_TYPE(depth.second, 1);
    }
};

CUDA_TEST_P(Subtract_Array_Mask, Accuracy)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::subtract(loadMat(mat1), loadMat(mat2), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::subtract(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, dtype, cv::Scalar::all(0));
        cv::subtract(mat1, mat2, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Subtract_Array_Mask, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Subtract_Scalar

PARAM_TEST_CASE(Subtract_Scalar, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Subtract_Scalar, WithOutMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::subtract(loadMat(mat), val, dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::subtract(loadMat(mat, useRoi), val, dst, cv::cuda::GpuMat(), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::subtract(mat, val, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

CUDA_TEST_P(Subtract_Scalar, WithMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::subtract(loadMat(mat), val, dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::subtract(loadMat(mat, useRoi), val, dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::subtract(mat, val, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Subtract_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Subtract_Scalar_First

PARAM_TEST_CASE(Subtract_Scalar_First, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Subtract_Scalar_First, WithOutMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::subtract(val, loadMat(mat), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::subtract(val, loadMat(mat, useRoi), dst, cv::cuda::GpuMat(), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::subtract(val, mat, dst_gold, cv::noArray(), depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

CUDA_TEST_P(Subtract_Scalar_First, WithMask)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::subtract(val, loadMat(mat), dst, cv::cuda::GpuMat(), depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        dst.setTo(cv::Scalar::all(0));
        cv::cuda::subtract(val, loadMat(mat, useRoi), dst, loadMat(mask, useRoi), depth.second);

        cv::Mat dst_gold(size, depth.second, cv::Scalar::all(0));
        cv::subtract(val, mat, dst_gold, mask, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Subtract_Scalar_First, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Array

PARAM_TEST_CASE(Multiply_Array, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

CUDA_TEST_P(Multiply_Array, WithOutScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::multiply(loadMat(mat1), loadMat(mat2), dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        cv::cuda::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat1, mat2, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-2 : 0.0);
    }
}

CUDA_TEST_P(Multiply_Array, WithScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::multiply(loadMat(mat1), loadMat(mat2), dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        cv::cuda::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat1, mat2, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 2.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Multiply_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Array_Special

PARAM_TEST_CASE(Multiply_Array_Special, cv::cuda::DeviceInfo, cv::Size, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Multiply_Array_Special, Case_8UC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_8UC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1);

    cv::cuda::GpuMat dst = createMat(size, CV_8UC4, useRoi);
    cv::cuda::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

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

CUDA_TEST_P(Multiply_Array_Special, Case_16SC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_16SC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1);

    cv::cuda::GpuMat dst = createMat(size, CV_16SC4, useRoi);
    cv::cuda::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

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

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Multiply_Array_Special, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Scalar

PARAM_TEST_CASE(Multiply_Scalar, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Multiply_Scalar, WithOutScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::multiply(loadMat(mat), val, dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::cuda::multiply(loadMat(mat, useRoi), val, dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat, val, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
    }
}


CUDA_TEST_P(Multiply_Scalar, WithScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::multiply(loadMat(mat), val, dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::cuda::multiply(loadMat(mat, useRoi), val, dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::multiply(mat, val, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Multiply_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Multiply_Scalar_First

PARAM_TEST_CASE(Multiply_Scalar_First, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Multiply_Scalar_First, WithOutScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::multiply(val, loadMat(mat), dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::cuda::multiply(val, loadMat(mat, useRoi), dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::multiply(val, mat, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
    }
}


CUDA_TEST_P(Multiply_Scalar_First, WithScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(0, 255);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::multiply(val, loadMat(mat), dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::cuda::multiply(val, loadMat(mat, useRoi), dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::multiply(val, mat, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Multiply_Scalar_First, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Array

PARAM_TEST_CASE(Divide_Array, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        stype = CV_MAKE_TYPE(depth.first, channels);
        dtype = CV_MAKE_TYPE(depth.second, channels);
    }
};

CUDA_TEST_P(Divide_Array, WithOutScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype, 1.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::divide(loadMat(mat1), loadMat(mat2), dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        cv::cuda::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat1, mat2, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

CUDA_TEST_P(Divide_Array, WithScale)
{
    cv::Mat mat1 = randomMat(size, stype);
    cv::Mat mat2 = randomMat(size, stype, 1.0, 255.0);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::divide(loadMat(mat1), loadMat(mat2), dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dtype, useRoi);
        cv::cuda::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat1, mat2, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-2 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Divide_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    ALL_CHANNELS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Array_Special

PARAM_TEST_CASE(Divide_Array_Special, cv::cuda::DeviceInfo, cv::Size, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Divide_Array_Special, Case_8UC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_8UC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1, 1.0, 255.0);

    cv::cuda::GpuMat dst = createMat(size, CV_8UC4, useRoi);
    cv::cuda::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

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

CUDA_TEST_P(Divide_Array_Special, Case_16SC4x_32FC1)
{
    cv::Mat mat1 = randomMat(size, CV_16SC4);
    cv::Mat mat2 = randomMat(size, CV_32FC1, 1.0, 255.0);

    cv::cuda::GpuMat dst = createMat(size, CV_16SC4, useRoi);
    cv::cuda::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), dst);

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

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Divide_Array_Special, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Scalar

PARAM_TEST_CASE(Divide_Scalar, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Divide_Scalar, WithOutScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(1.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::divide(loadMat(mat), val, dst, 1, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::cuda::divide(loadMat(mat, useRoi), val, dst, 1, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat, val, dst_gold, 1, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

CUDA_TEST_P(Divide_Scalar, WithScale)
{
    cv::Mat mat = randomMat(size, depth.first);
    cv::Scalar val = randomScalar(1.0, 255.0);
    double scale = randomDouble(0.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::divide(loadMat(mat), val, dst, scale, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::cuda::divide(loadMat(mat, useRoi), val, dst, scale, depth.second);

        cv::Mat dst_gold;
        cv::divide(mat, val, dst_gold, scale, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-2 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Divide_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Divide_Scalar_First

PARAM_TEST_CASE(Divide_Scalar_First, cv::cuda::DeviceInfo, cv::Size, std::pair<MatDepth, MatDepth>, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    std::pair<MatDepth, MatDepth> depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Divide_Scalar_First, Accuracy)
{
    double scale = randomDouble(0.0, 255.0);
    cv::Mat mat = randomMat(size, depth.first, 1.0, 255.0);

    if ((depth.first == CV_64F || depth.second == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::divide(scale, loadMat(mat), dst, depth.second);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth.second, useRoi);
        cv::cuda::divide(scale, loadMat(mat, useRoi), dst, depth.second);

        cv::Mat dst_gold;
        cv::divide(scale, mat, dst_gold, depth.second);

        EXPECT_MAT_NEAR(dst_gold, dst, depth.first >= CV_32F || depth.second >= CV_32F ? 1e-4 : 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Divide_Scalar_First, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DEPTH_PAIRS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// AbsDiff

PARAM_TEST_CASE(AbsDiff, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(AbsDiff, Array)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::absdiff(loadMat(src1), loadMat(src2), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::absdiff(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

        cv::Mat dst_gold;
        cv::absdiff(src1, src2, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

CUDA_TEST_P(AbsDiff, Scalar)
{
    cv::Mat src = randomMat(size, depth);
    cv::Scalar val = randomScalar(0.0, 255.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::absdiff(loadMat(src), val, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::absdiff(loadMat(src, useRoi), val, dst);

        cv::Mat dst_gold;
        cv::absdiff(src, val, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, depth <= CV_32F ? 1.0 : 1e-5);
    }
}

CUDA_TEST_P(AbsDiff, Scalar_First)
{
    cv::Mat src = randomMat(size, depth);
    cv::Scalar val = randomScalar(0.0, 255.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::absdiff(val, loadMat(src), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::absdiff(val, loadMat(src, useRoi), dst);

        cv::Mat dst_gold;
        cv::absdiff(val, src, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, depth <= CV_32F ? 1.0 : 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, AbsDiff, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Abs

PARAM_TEST_CASE(Abs, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Abs, Accuracy)
{
    cv::Mat src = randomMat(size, depth);

    cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
    cv::cuda::abs(loadMat(src, useRoi), dst);

    cv::Mat dst_gold = cv::abs(src);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Abs, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_16S), MatDepth(CV_32F)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Sqr

PARAM_TEST_CASE(Sqr, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Sqr, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 0, depth == CV_8U ? 16 : 255);

    cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
    cv::cuda::sqr(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::multiply(src, src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Sqr, testing::Combine(
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

PARAM_TEST_CASE(Sqrt, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Sqrt, Accuracy)
{
    cv::Mat src = randomMat(size, depth);

    cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
    cv::cuda::sqrt(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    sqrtGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-5);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Sqrt, testing::Combine(
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

PARAM_TEST_CASE(Log, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Log, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 1.0, 255.0);

    cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
    cv::cuda::log(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    logGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-6);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Log, testing::Combine(
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

PARAM_TEST_CASE(Exp, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Exp, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 0.0, 10.0);

    cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
    cv::cuda::exp(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    expGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-2);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Exp, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U),
                    MatDepth(CV_16U),
                    MatDepth(CV_16S),
                    MatDepth(CV_32F)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Pow

PARAM_TEST_CASE(Pow, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Pow, Accuracy)
{
    cv::Mat src = randomMat(size, depth, 0.0, 10.0);
    double power = randomDouble(2.0, 4.0);

    if (src.depth() < CV_32F)
        power = static_cast<int>(power);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::pow(loadMat(src), power, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::pow(loadMat(src, useRoi), power, dst);

        cv::Mat dst_gold;
        cv::pow(src, power, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 0.0 : 1e-1);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Pow, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Compare_Array

CV_ENUM(CmpCode, cv::CMP_EQ, cv::CMP_GT, cv::CMP_GE, cv::CMP_LT, cv::CMP_LE, cv::CMP_NE)
#define ALL_CMP_CODES testing::Values(CmpCode(cv::CMP_EQ), CmpCode(cv::CMP_NE), CmpCode(cv::CMP_GT), CmpCode(cv::CMP_GE), CmpCode(cv::CMP_LT), CmpCode(cv::CMP_LE))

PARAM_TEST_CASE(Compare_Array, cv::cuda::DeviceInfo, cv::Size, MatDepth, CmpCode, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Compare_Array, Accuracy)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::compare(loadMat(src1), loadMat(src2), dst, cmp_code);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, CV_8UC1, useRoi);
        cv::cuda::compare(loadMat(src1, useRoi), loadMat(src2, useRoi), dst, cmp_code);

        cv::Mat dst_gold;
        cv::compare(src1, src2, dst_gold, cmp_code);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Compare_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    ALL_CMP_CODES,
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

PARAM_TEST_CASE(Compare_Scalar, cv::cuda::DeviceInfo, cv::Size, MatType, CmpCode, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Compare_Scalar, Accuracy)
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

    if (src.depth() == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::compare(loadMat(src), sc, dst, cmp_code);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, CV_MAKE_TYPE(CV_8U, src.channels()), useRoi);

        cv::cuda::compare(loadMat(src, useRoi), sc, dst, cmp_code);

        cv::Mat dst_gold;
        compareScalarGold(src, sc, dst_gold, cmp_code);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Compare_Scalar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    TYPES(CV_8U, CV_64F, 1, 4),
    ALL_CMP_CODES,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Bitwise_Array

PARAM_TEST_CASE(Bitwise_Array, cv::cuda::DeviceInfo, cv::Size, MatType)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;

    cv::Mat src1;
    cv::Mat src2;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());

        src1 = randomMat(size, type, 0.0, std::numeric_limits<int>::max());
        src2 = randomMat(size, type, 0.0, std::numeric_limits<int>::max());
    }
};

CUDA_TEST_P(Bitwise_Array, Not)
{
    cv::cuda::GpuMat dst;
    cv::cuda::bitwise_not(loadMat(src1), dst);

    cv::Mat dst_gold = ~src1;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(Bitwise_Array, Or)
{
    cv::cuda::GpuMat dst;
    cv::cuda::bitwise_or(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 | src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(Bitwise_Array, And)
{
    cv::cuda::GpuMat dst;
    cv::cuda::bitwise_and(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 & src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(Bitwise_Array, Xor)
{
    cv::cuda::GpuMat dst;
    cv::cuda::bitwise_xor(loadMat(src1), loadMat(src2), dst);

    cv::Mat dst_gold = src1 ^ src2;

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Bitwise_Array, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    TYPES(CV_8U, CV_32S, 1, 4)));

//////////////////////////////////////////////////////////////////////////////
// Bitwise_Scalar

PARAM_TEST_CASE(Bitwise_Scalar, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        src = randomMat(size, CV_MAKE_TYPE(depth, channels));
        cv::Scalar_<int> ival = randomScalar(0.0, std::numeric_limits<int>::max());
        val = ival;
    }
};

CUDA_TEST_P(Bitwise_Scalar, Or)
{
    cv::cuda::GpuMat dst;
    cv::cuda::bitwise_or(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_or(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(Bitwise_Scalar, And)
{
    cv::cuda::GpuMat dst;
    cv::cuda::bitwise_and(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_and(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(Bitwise_Scalar, Xor)
{
    cv::cuda::GpuMat dst;
    cv::cuda::bitwise_xor(loadMat(src), val, dst);

    cv::Mat dst_gold;
    cv::bitwise_xor(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Bitwise_Scalar, testing::Combine(
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

PARAM_TEST_CASE(RShift, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(RShift, Accuracy)
{
    int type = CV_MAKE_TYPE(depth, channels);
    cv::Mat src = randomMat(size, type);
    cv::Scalar_<int> val = randomScalar(0.0, 8.0);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    cv::cuda::rshift(loadMat(src, useRoi), val, dst);

    cv::Mat dst_gold;
    rhiftGold(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, RShift, testing::Combine(
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

PARAM_TEST_CASE(LShift, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(LShift, Accuracy)
{
    int type = CV_MAKE_TYPE(depth, channels);
    cv::Mat src = randomMat(size, type);
    cv::Scalar_<int> val = randomScalar(0.0, 8.0);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    cv::cuda::lshift(loadMat(src, useRoi), val, dst);

    cv::Mat dst_gold;
    lhiftGold(src, val, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, LShift, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32S)),
    IMAGE_CHANNELS,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Min

PARAM_TEST_CASE(Min, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Min, Array)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::min(loadMat(src1), loadMat(src2), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::min(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

        cv::Mat dst_gold = cv::min(src1, src2);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

CUDA_TEST_P(Min, Scalar)
{
    cv::Mat src = randomMat(size, depth);
    double val = randomDouble(0.0, 255.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::min(loadMat(src), val, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::min(loadMat(src, useRoi), val, dst);

        cv::Mat dst_gold = cv::min(src, val);

        EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Min, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Max

PARAM_TEST_CASE(Max, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Max, Array)
{
    cv::Mat src1 = randomMat(size, depth);
    cv::Mat src2 = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::max(loadMat(src1), loadMat(src2), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::max(loadMat(src1, useRoi), loadMat(src2, useRoi), dst);

        cv::Mat dst_gold = cv::max(src1, src2);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

CUDA_TEST_P(Max, Scalar)
{
    cv::Mat src = randomMat(size, depth);
    double val = randomDouble(0.0, 255.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::max(loadMat(src), val, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, depth, useRoi);
        cv::cuda::max(loadMat(src, useRoi), val, dst);

        cv::Mat dst_gold = cv::max(src, val);

        EXPECT_MAT_NEAR(dst_gold, dst, depth < CV_32F ? 1.0 : 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Max, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// AddWeighted

PARAM_TEST_CASE(AddWeighted, cv::cuda::DeviceInfo, cv::Size, MatDepth, MatDepth, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(AddWeighted, Accuracy)
{
    cv::Mat src1 = randomMat(size, depth1);
    cv::Mat src2 = randomMat(size, depth2);
    double alpha = randomDouble(-10.0, 10.0);
    double beta = randomDouble(-10.0, 10.0);
    double gamma = randomDouble(-10.0, 10.0);

    if ((depth1 == CV_64F || depth2 == CV_64F || dst_depth == CV_64F) && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::addWeighted(loadMat(src1), alpha, loadMat(src2), beta, gamma, dst, dst_depth);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, dst_depth, useRoi);
        cv::cuda::addWeighted(loadMat(src1, useRoi), alpha, loadMat(src2, useRoi), beta, gamma, dst, dst_depth);

        cv::Mat dst_gold;
        cv::addWeighted(src1, alpha, src2, beta, gamma, dst_gold, dst_depth);

        EXPECT_MAT_NEAR(dst_gold, dst, dst_depth < CV_32F ? 2.0 : 1e-3);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, AddWeighted, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    ALL_DEPTH,
    ALL_DEPTH,
    WHOLE_SUBMAT));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Threshold

CV_ENUM(ThreshOp, cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV)
#define ALL_THRESH_OPS testing::Values(ThreshOp(cv::THRESH_BINARY), ThreshOp(cv::THRESH_BINARY_INV), ThreshOp(cv::THRESH_TRUNC), ThreshOp(cv::THRESH_TOZERO), ThreshOp(cv::THRESH_TOZERO_INV))

PARAM_TEST_CASE(Threshold, cv::cuda::DeviceInfo, cv::Size, MatType, ThreshOp, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int threshOp;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        threshOp = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Threshold, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    double maxVal = randomDouble(20.0, 127.0);
    double thresh = randomDouble(0.0, maxVal);

    cv::cuda::GpuMat dst = createMat(src.size(), src.type(), useRoi);
    cv::cuda::threshold(loadMat(src, useRoi), dst, thresh, maxVal, threshOp);

    cv::Mat dst_gold;
    cv::threshold(src, dst_gold, thresh, maxVal, threshOp);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Threshold, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_16SC1), MatType(CV_32FC1)),
    ALL_THRESH_OPS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Magnitude

PARAM_TEST_CASE(Magnitude, cv::cuda::DeviceInfo, cv::Size, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Magnitude, NPP)
{
    cv::Mat src = randomMat(size, CV_32FC2);

    cv::cuda::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::cuda::magnitude(loadMat(src, useRoi), dst);

    cv::Mat arr[2];
    cv::split(src, arr);
    cv::Mat dst_gold;
    cv::magnitude(arr[0], arr[1], dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

CUDA_TEST_P(Magnitude, Sqr_NPP)
{
    cv::Mat src = randomMat(size, CV_32FC2);

    cv::cuda::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::cuda::magnitudeSqr(loadMat(src, useRoi), dst);

    cv::Mat arr[2];
    cv::split(src, arr);
    cv::Mat dst_gold;
    cv::magnitude(arr[0], arr[1], dst_gold);
    cv::multiply(dst_gold, dst_gold, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-1);
}

CUDA_TEST_P(Magnitude, Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::cuda::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::cuda::magnitude(loadMat(x, useRoi), loadMat(y, useRoi), dst);

    cv::Mat dst_gold;
    cv::magnitude(x, y, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

CUDA_TEST_P(Magnitude, Sqr_Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::cuda::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::cuda::magnitudeSqr(loadMat(x, useRoi), loadMat(y, useRoi), dst);

    cv::Mat dst_gold;
    cv::magnitude(x, y, dst_gold);
    cv::multiply(dst_gold, dst_gold, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-1);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Magnitude, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Phase

namespace
{
    IMPLEMENT_PARAM_CLASS(AngleInDegrees, bool)
}

PARAM_TEST_CASE(Phase, cv::cuda::DeviceInfo, cv::Size, AngleInDegrees, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    bool angleInDegrees;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        angleInDegrees = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Phase, Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::cuda::GpuMat dst = createMat(size, CV_32FC1, useRoi);
    cv::cuda::phase(loadMat(x, useRoi), loadMat(y, useRoi), dst, angleInDegrees);

    cv::Mat dst_gold;
    cv::phase(x, y, dst_gold, angleInDegrees);

    EXPECT_MAT_NEAR(dst_gold, dst, angleInDegrees ? 1e-2 : 1e-3);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Phase, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(AngleInDegrees(false), AngleInDegrees(true)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// CartToPolar

PARAM_TEST_CASE(CartToPolar, cv::cuda::DeviceInfo, cv::Size, AngleInDegrees, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    bool angleInDegrees;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        angleInDegrees = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CartToPolar, Accuracy)
{
    cv::Mat x = randomMat(size, CV_32FC1);
    cv::Mat y = randomMat(size, CV_32FC1);

    cv::cuda::GpuMat mag = createMat(size, CV_32FC1, useRoi);
    cv::cuda::GpuMat angle = createMat(size, CV_32FC1, useRoi);
    cv::cuda::cartToPolar(loadMat(x, useRoi), loadMat(y, useRoi), mag, angle, angleInDegrees);

    cv::Mat mag_gold;
    cv::Mat angle_gold;
    cv::cartToPolar(x, y, mag_gold, angle_gold, angleInDegrees);

    EXPECT_MAT_NEAR(mag_gold, mag, 1e-4);
    EXPECT_MAT_NEAR(angle_gold, angle, angleInDegrees ? 1e-2 : 1e-3);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, CartToPolar, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(AngleInDegrees(false), AngleInDegrees(true)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// polarToCart

PARAM_TEST_CASE(PolarToCart, cv::cuda::DeviceInfo, cv::Size, AngleInDegrees, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    bool angleInDegrees;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        angleInDegrees = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(PolarToCart, Accuracy)
{
    cv::Mat magnitude = randomMat(size, CV_32FC1);
    cv::Mat angle = randomMat(size, CV_32FC1);

    cv::cuda::GpuMat x = createMat(size, CV_32FC1, useRoi);
    cv::cuda::GpuMat y = createMat(size, CV_32FC1, useRoi);
    cv::cuda::polarToCart(loadMat(magnitude, useRoi), loadMat(angle, useRoi), x, y, angleInDegrees);

    cv::Mat x_gold;
    cv::Mat y_gold;
    cv::polarToCart(magnitude, angle, x_gold, y_gold, angleInDegrees);

    EXPECT_MAT_NEAR(x_gold, x, 1e-4);
    EXPECT_MAT_NEAR(y_gold, y, 1e-4);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, PolarToCart, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(AngleInDegrees(false), AngleInDegrees(true)),
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
