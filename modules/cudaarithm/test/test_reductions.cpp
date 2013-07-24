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
// Norm

PARAM_TEST_CASE(Norm, cv::cuda::DeviceInfo, cv::Size, MatDepth, NormCode, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Norm, Accuracy)
{
    cv::Mat src = randomMat(size, depth);
    cv::Mat mask = randomMat(size, CV_8UC1, 0, 2);

    cv::cuda::GpuMat d_buf;
    double val = cv::cuda::norm(loadMat(src, useRoi), normCode, loadMat(mask, useRoi), d_buf);

    double val_gold = cv::norm(src, normCode, mask);

    EXPECT_NEAR(val_gold, val, depth < CV_32F ? 0.0 : 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Norm, testing::Combine(
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

PARAM_TEST_CASE(NormDiff, cv::cuda::DeviceInfo, cv::Size, NormCode, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int normCode;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        normCode = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(NormDiff, Accuracy)
{
    cv::Mat src1 = randomMat(size, CV_8UC1);
    cv::Mat src2 = randomMat(size, CV_8UC1);

    double val = cv::cuda::norm(loadMat(src1, useRoi), loadMat(src2, useRoi), normCode);

    double val_gold = cv::norm(src1, src2, normCode);

    EXPECT_NEAR(val_gold, val, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, NormDiff, testing::Combine(
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

PARAM_TEST_CASE(Sum, cv::cuda::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        src = randomMat(size, type, -128.0, 128.0);
    }
};

CUDA_TEST_P(Sum, Simple)
{
    cv::Scalar val = cv::cuda::sum(loadMat(src, useRoi));

    cv::Scalar val_gold = cv::sum(src);

    EXPECT_SCALAR_NEAR(val_gold, val, CV_MAT_DEPTH(type) < CV_32F ? 0.0 : 0.5);
}

CUDA_TEST_P(Sum, Abs)
{
    cv::Scalar val = cv::cuda::absSum(loadMat(src, useRoi));

    cv::Scalar val_gold = absSumGold(src);

    EXPECT_SCALAR_NEAR(val_gold, val, CV_MAT_DEPTH(type) < CV_32F ? 0.0 : 0.5);
}

CUDA_TEST_P(Sum, Sqr)
{
    cv::Scalar val = cv::cuda::sqrSum(loadMat(src, useRoi));

    cv::Scalar val_gold = sqrSumGold(src);

    EXPECT_SCALAR_NEAR(val_gold, val, CV_MAT_DEPTH(type) < CV_32F ? 0.0 : 0.5);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Sum, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    TYPES(CV_8U, CV_64F, 1, 4),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MinMax

PARAM_TEST_CASE(MinMax, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

CUDA_TEST_P(MinMax, WithoutMask)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::cuda::minMax(loadMat(src), &minVal, &maxVal);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::cuda::minMax(loadMat(src, useRoi), &minVal, &maxVal);

        double minVal_gold, maxVal_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
    }
}

CUDA_TEST_P(MinMax, WithMask)
{
    cv::Mat src = randomMat(size, depth);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::cuda::minMax(loadMat(src), &minVal, &maxVal, loadMat(mask));
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::cuda::minMax(loadMat(src, useRoi), &minVal, &maxVal, loadMat(mask, useRoi));

        double minVal_gold, maxVal_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, 0, 0, mask);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
    }
}

CUDA_TEST_P(MinMax, NullPtr)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::cuda::minMax(loadMat(src), &minVal, 0);
            cv::cuda::minMax(loadMat(src), 0, &maxVal);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::cuda::minMax(loadMat(src, useRoi), &minVal, 0);
        cv::cuda::minMax(loadMat(src, useRoi), 0, &maxVal);

        double minVal_gold, maxVal_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, 0, 0);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, MinMax, testing::Combine(
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

PARAM_TEST_CASE(MinMaxLoc, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

CUDA_TEST_P(MinMaxLoc, WithoutMask)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::cuda::minMaxLoc(loadMat(src), &minVal, &maxVal, &minLoc, &maxLoc);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::cuda::minMaxLoc(loadMat(src, useRoi), &minVal, &maxVal, &minLoc, &maxLoc);

        double minVal_gold, maxVal_gold;
        cv::Point minLoc_gold, maxLoc_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

        expectEqual(src, minLoc_gold, minLoc);
        expectEqual(src, maxLoc_gold, maxLoc);
    }
}

CUDA_TEST_P(MinMaxLoc, WithMask)
{
    cv::Mat src = randomMat(size, depth);
    cv::Mat mask = randomMat(size, CV_8UC1, 0.0, 2.0);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::cuda::minMaxLoc(loadMat(src), &minVal, &maxVal, &minLoc, &maxLoc, loadMat(mask));
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::cuda::minMaxLoc(loadMat(src, useRoi), &minVal, &maxVal, &minLoc, &maxLoc, loadMat(mask, useRoi));

        double minVal_gold, maxVal_gold;
        cv::Point minLoc_gold, maxLoc_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold, mask);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

        expectEqual(src, minLoc_gold, minLoc);
        expectEqual(src, maxLoc_gold, maxLoc);
    }
}

CUDA_TEST_P(MinMaxLoc, NullPtr)
{
    cv::Mat src = randomMat(size, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::cuda::minMaxLoc(loadMat(src, useRoi), &minVal, 0, 0, 0);
            cv::cuda::minMaxLoc(loadMat(src, useRoi), 0, &maxVal, 0, 0);
            cv::cuda::minMaxLoc(loadMat(src, useRoi), 0, 0, &minLoc, 0);
            cv::cuda::minMaxLoc(loadMat(src, useRoi), 0, 0, 0, &maxLoc);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::cuda::minMaxLoc(loadMat(src, useRoi), &minVal, 0, 0, 0);
        cv::cuda::minMaxLoc(loadMat(src, useRoi), 0, &maxVal, 0, 0);
        cv::cuda::minMaxLoc(loadMat(src, useRoi), 0, 0, &minLoc, 0);
        cv::cuda::minMaxLoc(loadMat(src, useRoi), 0, 0, 0, &maxLoc);

        double minVal_gold, maxVal_gold;
        cv::Point minLoc_gold, maxLoc_gold;
        minMaxLocGold(src, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold);

        EXPECT_DOUBLE_EQ(minVal_gold, minVal);
        EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

        expectEqual(src, minLoc_gold, minLoc);
        expectEqual(src, maxLoc_gold, maxLoc);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, MinMaxLoc, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////
// CountNonZero

PARAM_TEST_CASE(CountNonZero, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
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

CUDA_TEST_P(CountNonZero, Accuracy)
{
    cv::Mat srcBase = randomMat(size, CV_8U, 0.0, 1.5);
    cv::Mat src;
    srcBase.convertTo(src, depth);

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::countNonZero(loadMat(src));
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        int val = cv::cuda::countNonZero(loadMat(src, useRoi));

        int val_gold = cv::countNonZero(src);

        ASSERT_EQ(val_gold, val);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, CountNonZero, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// Reduce

CV_ENUM(ReduceCode, cv::REDUCE_SUM, cv::REDUCE_AVG, cv::REDUCE_MAX, cv::REDUCE_MIN)
#define ALL_REDUCE_CODES testing::Values(ReduceCode(cv::REDUCE_SUM), ReduceCode(cv::REDUCE_AVG), ReduceCode(cv::REDUCE_MAX), ReduceCode(cv::REDUCE_MIN))

PARAM_TEST_CASE(Reduce, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, ReduceCode, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, channels);

        if (reduceOp == cv::REDUCE_MAX || reduceOp == cv::REDUCE_MIN)
            dst_depth = depth;
        else if (reduceOp == cv::REDUCE_SUM)
            dst_depth = depth == CV_8U ? CV_32S : depth < CV_64F ? CV_32F : depth;
        else
            dst_depth = depth < CV_32F ? CV_32F : depth;

        dst_type = CV_MAKE_TYPE(dst_depth, channels);
    }

};

CUDA_TEST_P(Reduce, Rows)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(cv::Size(src.cols, 1), dst_type, useRoi);
    cv::cuda::reduce(loadMat(src, useRoi), dst, 0, reduceOp, dst_depth);

    cv::Mat dst_gold;
    cv::reduce(src, dst_gold, 0, reduceOp, dst_depth);

    EXPECT_MAT_NEAR(dst_gold, dst, dst_depth < CV_32F ? 0.0 : 0.02);
}

CUDA_TEST_P(Reduce, Cols)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(cv::Size(src.rows, 1), dst_type, useRoi);
    cv::cuda::reduce(loadMat(src, useRoi), dst, 1, reduceOp, dst_depth);

    cv::Mat dst_gold;
    cv::reduce(src, dst_gold, 1, reduceOp, dst_depth);
    dst_gold.cols = dst_gold.rows;
    dst_gold.rows = 1;
    dst_gold.step = dst_gold.cols * dst_gold.elemSize();

    EXPECT_MAT_NEAR(dst_gold, dst, dst_depth < CV_32F ? 0.0 : 0.02);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Reduce, testing::Combine(
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

PARAM_TEST_CASE(Normalize, cv::cuda::DeviceInfo, cv::Size, MatDepth, NormCode, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
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

        cv::cuda::setDevice(devInfo.deviceID());

        alpha = 1;
        beta = 0;
    }

};

CUDA_TEST_P(Normalize, WithOutMask)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    cv::cuda::normalize(loadMat(src, useRoi), dst, alpha, beta, norm_type, type);

    cv::Mat dst_gold;
    cv::normalize(src, dst_gold, alpha, beta, norm_type, type);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-6);
}

CUDA_TEST_P(Normalize, WithMask)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat mask = randomMat(size, CV_8UC1, 0, 2);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    dst.setTo(cv::Scalar::all(0));
    cv::cuda::normalize(loadMat(src, useRoi), dst, alpha, beta, norm_type, type, loadMat(mask, useRoi));

    cv::Mat dst_gold(size, type);
    dst_gold.setTo(cv::Scalar::all(0));
    cv::normalize(src, dst_gold, alpha, beta, norm_type, type, mask);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-6);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Normalize, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    testing::Values(NormCode(cv::NORM_L1), NormCode(cv::NORM_L2), NormCode(cv::NORM_INF), NormCode(cv::NORM_MINMAX)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MeanStdDev

PARAM_TEST_CASE(MeanStdDev, cv::cuda::DeviceInfo, cv::Size, UseRoi)
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

CUDA_TEST_P(MeanStdDev, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    if (!supportFeature(devInfo, cv::cuda::FEATURE_SET_COMPUTE_13))
    {
        try
        {
            cv::Scalar mean;
            cv::Scalar stddev;
            cv::cuda::meanStdDev(loadMat(src, useRoi), mean, stddev);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::cuda::meanStdDev(loadMat(src, useRoi), mean, stddev);

        cv::Scalar mean_gold;
        cv::Scalar stddev_gold;
        cv::meanStdDev(src, mean_gold, stddev_gold);

        EXPECT_SCALAR_NEAR(mean_gold, mean, 1e-5);
        EXPECT_SCALAR_NEAR(stddev_gold, stddev, 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, MeanStdDev, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
