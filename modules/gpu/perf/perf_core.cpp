#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace {

#define ARITHM_MAT_DEPTH Values(CV_8U, CV_16U, CV_32F, CV_64F)

//////////////////////////////////////////////////////////////////////
// Merge

PERF_TEST_P(Sz_Depth_Cn, Core_Merge, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH, Values(2, 3, 4)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    std::vector<cv::Mat> src(channels);
    for (int i = 0; i < channels; ++i)
        src[i] = cv::Mat(size, depth, cv::Scalar::all(i));

    if (runOnGpu)
    {
        std::vector<cv::gpu::GpuMat> d_src(channels);
        for (int i = 0; i < channels; ++i)
            d_src[i].upload(src[i]);

        cv::gpu::GpuMat d_dst;

        cv::gpu::merge(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::merge(d_src, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::merge(src, dst);

        TEST_CYCLE()
        {
            cv::merge(src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Split

PERF_TEST_P(Sz_Depth_Cn, Core_Split, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH, Values(2, 3, 4)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    cv::Mat src(size, CV_MAKE_TYPE(depth, channels), cv::Scalar(1, 2, 3, 4));

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);

        std::vector<cv::gpu::GpuMat> d_dst;

        cv::gpu::split(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::split(d_src, d_dst);
        }
    }
    else
    {
        std::vector<cv::Mat> dst;

        cv::split(src, dst);

        TEST_CYCLE()
        {
            cv::split(src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// AddMat

PERF_TEST_P(Sz_Depth, Core_AddMat, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::add(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::add(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::add(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::add(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// AddScalar

PERF_TEST_P(Sz_Depth, Core_AddScalar, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    cv::Scalar s(1, 2, 3, 4);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::add(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::add(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::add(src, s, dst);

        TEST_CYCLE()
        {
            cv::add(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// SubtractMat

PERF_TEST_P(Sz_Depth, Core_SubtractMat, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::subtract(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::subtract(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::subtract(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::subtract(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// SubtractScalar

PERF_TEST_P(Sz_Depth, Core_SubtractScalar, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    cv::Scalar s(1, 2, 3, 4);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::subtract(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::subtract(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::subtract(src, s, dst);

        TEST_CYCLE()
        {
            cv::subtract(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MultiplyMat

PERF_TEST_P(Sz_Depth, Core_MultiplyMat, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::multiply(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::multiply(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::multiply(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::multiply(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MultiplyScalar

PERF_TEST_P(Sz_Depth, Core_MultiplyScalar, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    cv::Scalar s(1, 2, 3, 4);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::multiply(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::multiply(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::multiply(src, s, dst);

        TEST_CYCLE()
        {
            cv::multiply(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// DivideMat

PERF_TEST_P(Sz_Depth, Core_DivideMat, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::divide(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::divide(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::divide(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::divide(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// DivideScalar

PERF_TEST_P(Sz_Depth, Core_DivideScalar, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    cv::Scalar s(1, 2, 3, 4);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::divide(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::divide(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::divide(src, s, dst);

        TEST_CYCLE()
        {
            cv::divide(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// DivideScalarInv

PERF_TEST_P(Sz_Depth, Core_DivideScalarInv, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    double s = 100.0;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::divide(s, d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::divide(s, d_src, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::divide(s, src, dst);

        TEST_CYCLE()
        {
            cv::divide(s, src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// AbsDiffMat

PERF_TEST_P(Sz_Depth, Core_AbsDiffMat, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::absdiff(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::absdiff(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::absdiff(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::absdiff(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// AbsDiffScalar

PERF_TEST_P(Sz_Depth, Core_AbsDiffScalar, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    cv::Scalar s(1, 2, 3, 4);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::absdiff(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::absdiff(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::absdiff(src, s, dst);

        TEST_CYCLE()
        {
            cv::absdiff(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Abs

PERF_TEST_P(Sz_Depth, Core_Abs, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::abs(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::abs(d_src, d_dst);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// Sqr

PERF_TEST_P(Sz_Depth, Core_Sqr, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::sqr(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::sqr(d_src, d_dst);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// Sqrt

PERF_TEST_P(Sz_Depth, Core_Sqrt, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::sqrt(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::sqrt(d_src, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::sqrt(src, dst);

        TEST_CYCLE()
        {
            cv::sqrt(src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Log

PERF_TEST_P(Sz_Depth, Core_Log, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src, 1.0, 255.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::log(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::log(d_src, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::log(src, dst);

        TEST_CYCLE()
        {
            cv::log(src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Exp

PERF_TEST_P(Sz_Depth, Core_Exp, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src, 1.0, 10.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::exp(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::exp(d_src, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::exp(src, dst);

        TEST_CYCLE()
        {
            cv::exp(src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Pow

DEF_PARAM_TEST(Sz_Depth_Power, cv::Size, MatDepth, double);

PERF_TEST_P(Sz_Depth_Power, Core_Pow, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F), Values(0.3, 2.0, 2.4)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const double power = GET_PARAM(2);

    cv::Mat src(size, depth);
    fillRandom(src, 1.0, 10.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::pow(d_src, power, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::pow(d_src, power, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::pow(src, power, dst);

        TEST_CYCLE()
        {
            cv::pow(src, power, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// CompareMat

CV_ENUM(CmpCode, cv::CMP_EQ, cv::CMP_GT, cv::CMP_GE, cv::CMP_LT, cv::CMP_LE, cv::CMP_NE)
#define ALL_CMP_CODES ValuesIn(CmpCode::all())

DEF_PARAM_TEST(Sz_Depth_Code, cv::Size, MatDepth, CmpCode);

PERF_TEST_P(Sz_Depth_Code, Core_CompareMat, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH, ALL_CMP_CODES))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int cmp_code = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::compare(d_src1, d_src2, d_dst, cmp_code);

        TEST_CYCLE()
        {
            cv::gpu::compare(d_src1, d_src2, d_dst, cmp_code);
        }
    }
    else
    {
        cv::Mat dst;

        cv::compare(src1, src2, dst, cmp_code);

        TEST_CYCLE()
        {
            cv::compare(src1, src2, dst, cmp_code);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// CompareScalar

PERF_TEST_P(Sz_Depth_Code, Core_CompareScalar, Combine(GPU_TYPICAL_MAT_SIZES, ARITHM_MAT_DEPTH, ALL_CMP_CODES))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int cmp_code = GET_PARAM(2);

    cv::Mat src(size, depth);
    fillRandom(src);

    cv::Scalar s = cv::Scalar::all(100);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::compare(d_src, s, d_dst, cmp_code);

        TEST_CYCLE()
        {
            cv::gpu::compare(d_src, s, d_dst, cmp_code);
        }
    }
    else
    {
        cv::Mat dst;

        cv::compare(src, s, dst, cmp_code);

        TEST_CYCLE()
        {
            cv::compare(src, s, dst, cmp_code);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseNot

PERF_TEST_P(Sz_Depth, Core_BitwiseNot, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bitwise_not(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::bitwise_not(d_src, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bitwise_not(src, dst);

        TEST_CYCLE()
        {
            cv::bitwise_not(src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseAndMat

PERF_TEST_P(Sz_Depth, Core_BitwiseAndMat, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bitwise_and(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::bitwise_and(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bitwise_and(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::bitwise_and(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseAndScalar

PERF_TEST_P(Sz_Depth_Cn, Core_BitwiseAndScalar, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S), GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Scalar s = cv::Scalar::all(100);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bitwise_and(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::bitwise_and(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bitwise_and(src, s, dst);

        TEST_CYCLE()
        {
            cv::bitwise_and(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseOrMat

PERF_TEST_P(Sz_Depth, Core_BitwiseOrMat, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bitwise_or(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::bitwise_or(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bitwise_or(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::bitwise_or(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseOrScalar

PERF_TEST_P(Sz_Depth_Cn, Core_BitwiseOrScalar, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S), GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Scalar s = cv::Scalar::all(100);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bitwise_or(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::bitwise_or(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bitwise_or(src, s, dst);

        TEST_CYCLE()
        {
            cv::bitwise_or(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseXorMat

PERF_TEST_P(Sz_Depth, Core_BitwiseXorMat, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bitwise_xor(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::bitwise_xor(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bitwise_xor(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::bitwise_xor(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseXorScalar

PERF_TEST_P(Sz_Depth_Cn, Core_BitwiseXorScalar, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S), GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Scalar s = cv::Scalar::all(100);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bitwise_xor(d_src, s, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::bitwise_xor(d_src, s, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bitwise_xor(src, s, dst);

        TEST_CYCLE()
        {
            cv::bitwise_xor(src, s, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// RShift

PERF_TEST_P(Sz_Depth_Cn, Core_RShift, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S), GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    const cv::Scalar_<int> val = cv::Scalar_<int>::all(4);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::rshift(d_src, val, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::rshift(d_src, val, d_dst);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// LShift

PERF_TEST_P(Sz_Depth_Cn, Core_LShift, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32S), GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    const cv::Scalar_<int> val = cv::Scalar_<int>::all(4);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::lshift(d_src, val, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::lshift(d_src, val, d_dst);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// MinMat

PERF_TEST_P(Sz_Depth, Core_MinMat, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::min(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::min(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::min(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::min(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MinScalar

PERF_TEST_P(Sz_Depth, Core_MinScalar, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    const double val = 50.0;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::min(d_src, val, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::min(d_src, val, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::min(src, val, dst);

        TEST_CYCLE()
        {
            cv::min(src, val, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MaxMat

PERF_TEST_P(Sz_Depth, Core_MaxMat, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    fillRandom(src1);

    cv::Mat src2(size, depth);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::max(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::max(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::max(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::max(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MaxScalar

PERF_TEST_P(Sz_Depth, Core_MaxScalar, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    const double val = 50.0;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::max(d_src, val, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::max(d_src, val, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::max(src, val, dst);

        TEST_CYCLE()
        {
            cv::max(src, val, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// AddWeighted

DEF_PARAM_TEST(Sz_3Depth, cv::Size, MatDepth, MatDepth, MatDepth);

PERF_TEST_P(Sz_3Depth, Core_AddWeighted, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32F, CV_64F),
    Values(CV_8U, CV_16U, CV_32F, CV_64F),
    Values(CV_8U, CV_16U, CV_32F, CV_64F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth1 = GET_PARAM(1);
    const int depth2 = GET_PARAM(2);
    const int dst_depth = GET_PARAM(3);

    cv::Mat src1(size, depth1);
    fillRandom(src1);

    cv::Mat src2(size, depth2);
    fillRandom(src2);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::addWeighted(d_src1, 0.5, d_src2, 0.5, 10.0, d_dst, dst_depth);

        TEST_CYCLE()
        {
            cv::gpu::addWeighted(d_src1, 0.5, d_src2, 0.5, 10.0, d_dst, dst_depth);
        }
    }
    else
    {
        cv::Mat dst;

        cv::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);

        TEST_CYCLE()
        {
            cv::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// GEMM

CV_FLAGS(GemmFlags, 0, cv::GEMM_1_T, cv::GEMM_2_T, cv::GEMM_3_T)
#define ALL_GEMM_FLAGS Values(0, CV_GEMM_A_T, CV_GEMM_B_T, CV_GEMM_C_T, CV_GEMM_A_T | CV_GEMM_B_T, CV_GEMM_A_T | CV_GEMM_C_T, CV_GEMM_A_T | CV_GEMM_B_T | CV_GEMM_C_T)

DEF_PARAM_TEST(Sz_Type_Flags, cv::Size, MatType, GemmFlags);

PERF_TEST_P(Sz_Type_Flags, Core_GEMM, Combine(
    Values(cv::Size(512, 512), cv::Size(1024, 1024)),
    Values(CV_32FC1, CV_32FC2, CV_64FC1, CV_64FC2),
    ALL_GEMM_FLAGS))
{
    declare.time(5.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int flags = GET_PARAM(2);

    cv::Mat src1(size, type);
    fillRandom(src1);

    cv::Mat src2(size, type);
    fillRandom(src2);

    cv::Mat src3(size, type);
    fillRandom(src3);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_src3(src3);
        cv::gpu::GpuMat d_dst;

        cv::gpu::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst, flags);

        TEST_CYCLE()
        {
            cv::gpu::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst, flags);
        }
    }
    else
    {
        cv::Mat dst;

        cv::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);

        declare.time(50.0);

        TEST_CYCLE()
        {
            cv::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Transpose

PERF_TEST_P(Sz_Type, Core_Transpose, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8UC1, CV_8UC4, CV_16UC2, CV_16SC2, CV_32SC1, CV_32SC2, CV_64FC1)))
{
    cv::Size size = GET_PARAM(0);
    int type = GET_PARAM(1);

    cv::Mat src(size, type);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::transpose(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::transpose(d_src, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::transpose(src, dst);

        TEST_CYCLE()
        {
            cv::transpose(src, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Flip

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)
#define ALL_FLIP_CODES ValuesIn(FlipCode::all())

DEF_PARAM_TEST(Sz_Depth_Cn_Code, cv::Size, MatDepth, int, FlipCode);

PERF_TEST_P(Sz_Depth_Cn_Code, Core_Flip, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32F),
    GPU_CHANNELS_1_3_4,
    ALL_FLIP_CODES))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int channels = GET_PARAM(2);
    int flipCode = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::flip(d_src, d_dst, flipCode);

        TEST_CYCLE()
        {
            cv::gpu::flip(d_src, d_dst, flipCode);
        }
    }
    else
    {
        cv::Mat dst;

        cv::flip(src, dst, flipCode);

        TEST_CYCLE()
        {
            cv::flip(src, dst, flipCode);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// LutOneChannel

PERF_TEST_P(Sz_Type, Core_LutOneChannel, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8UC1, CV_8UC3)))
{
    cv::Size size = GET_PARAM(0);
    int type = GET_PARAM(1);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Mat lut(1, 256, CV_8UC1);
    fillRandom(lut);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::LUT(d_src, lut, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::LUT(d_src, lut, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::LUT(src, lut, dst);

        TEST_CYCLE()
        {
            cv::LUT(src, lut, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// LutMultiChannel

PERF_TEST_P(Sz_Type, Core_LutMultiChannel, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values<MatType>(CV_8UC3)))
{
    cv::Size size = GET_PARAM(0);
    int type = GET_PARAM(1);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Mat lut(1, 256, CV_MAKE_TYPE(CV_8U, src.channels()));
    fillRandom(lut);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::LUT(d_src, lut, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::LUT(d_src, lut, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::LUT(src, lut, dst);

        TEST_CYCLE()
        {
            cv::LUT(src, lut, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeComplex

PERF_TEST_P(Sz, Core_MagnitudeComplex, GPU_TYPICAL_MAT_SIZES)
{
    cv::Size size = GetParam();

    cv::Mat src(size, CV_32FC2);
    fillRandom(src, -100.0, 100.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::magnitude(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::magnitude(d_src, d_dst);
        }
    }
    else
    {
        cv::Mat xy[2];
        cv::split(src, xy);

        cv::Mat dst;

        cv::magnitude(xy[0], xy[1], dst);

        TEST_CYCLE()
        {
            cv::magnitude(xy[0], xy[1], dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeSqrComplex

PERF_TEST_P(Sz, Core_MagnitudeSqrComplex, GPU_TYPICAL_MAT_SIZES)
{
    cv::Size size = GetParam();

    cv::Mat src(size, CV_32FC2);
    fillRandom(src, -100.0, 100.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::magnitudeSqr(d_src, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::magnitudeSqr(d_src, d_dst);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// Magnitude

PERF_TEST_P(Sz, Core_Magnitude, GPU_TYPICAL_MAT_SIZES)
{
    cv::Size size = GetParam();

    cv::Mat src1(size, CV_32FC1);
    fillRandom(src1, -100.0, 100.0);

    cv::Mat src2(size, CV_32FC1);
    fillRandom(src2, -100.0, 100.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::magnitude(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::magnitude(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        cv::Mat dst;

        cv::magnitude(src1, src2, dst);

        TEST_CYCLE()
        {
            cv::magnitude(src1, src2, dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeSqr

PERF_TEST_P(Sz, Core_MagnitudeSqr, GPU_TYPICAL_MAT_SIZES)
{
    cv::Size size = GetParam();

    cv::Mat src1(size, CV_32FC1);
    fillRandom(src1, -100.0, 100.0);

    cv::Mat src2(size, CV_32FC1);
    fillRandom(src2, -100.0, 100.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::magnitudeSqr(d_src1, d_src2, d_dst);

        TEST_CYCLE()
        {
            cv::gpu::magnitudeSqr(d_src1, d_src2, d_dst);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// Phase

DEF_PARAM_TEST(Sz_AngleInDegrees, cv::Size, bool);

PERF_TEST_P(Sz_AngleInDegrees, Core_Phase, Combine(GPU_TYPICAL_MAT_SIZES, Bool()))
{
    cv::Size size = GET_PARAM(0);
    bool angleInDegrees = GET_PARAM(1);

    cv::Mat src1(size, CV_32FC1);
    fillRandom(src1, -100.0, 100.0);

    cv::Mat src2(size, CV_32FC1);
    fillRandom(src2, -100.0, 100.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        cv::gpu::phase(d_src1, d_src2, d_dst, angleInDegrees);

        TEST_CYCLE()
        {
            cv::gpu::phase(d_src1, d_src2, d_dst, angleInDegrees);
        }
    }
    else
    {
        cv::Mat dst;

        cv::phase(src1, src2, dst, angleInDegrees);

        TEST_CYCLE()
        {
            cv::phase(src1, src2, dst, angleInDegrees);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// CartToPolar

PERF_TEST_P(Sz_AngleInDegrees, Core_CartToPolar, Combine(GPU_TYPICAL_MAT_SIZES, Bool()))
{
    cv::Size size = GET_PARAM(0);
    bool angleInDegrees = GET_PARAM(1);

    cv::Mat src1(size, CV_32FC1);
    fillRandom(src1, -100.0, 100.0);

    cv::Mat src2(size, CV_32FC1);
    fillRandom(src2, -100.0, 100.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_magnitude;
        cv::gpu::GpuMat d_angle;

        cv::gpu::cartToPolar(d_src1, d_src2, d_magnitude, d_angle, angleInDegrees);

        TEST_CYCLE()
        {
            cv::gpu::cartToPolar(d_src1, d_src2, d_magnitude, d_angle, angleInDegrees);
        }
    }
    else
    {
        cv::Mat magnitude;
        cv::Mat angle;

        cv::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);

        TEST_CYCLE()
        {
            cv::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// PolarToCart

PERF_TEST_P(Sz_AngleInDegrees, Core_PolarToCart, Combine(GPU_TYPICAL_MAT_SIZES, Bool()))
{
    cv::Size size = GET_PARAM(0);
    bool angleInDegrees = GET_PARAM(1);

    cv::Mat magnitude(size, CV_32FC1);
    fillRandom(magnitude, 0.0, 100.0);

    cv::Mat angle(size, CV_32FC1);
    fillRandom(angle, 0.0, angleInDegrees ? 360.0 : 2 * CV_PI);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_magnitude(magnitude);
        cv::gpu::GpuMat d_angle(angle);
        cv::gpu::GpuMat d_x;
        cv::gpu::GpuMat d_y;

        cv::gpu::polarToCart(d_magnitude, d_angle, d_x, d_y, angleInDegrees);

        TEST_CYCLE()
        {
            cv::gpu::polarToCart(d_magnitude, d_angle, d_x, d_y, angleInDegrees);
        }
    }
    else
    {
        cv::Mat x;
        cv::Mat y;

        cv::polarToCart(magnitude, angle, x, y, angleInDegrees);

        TEST_CYCLE()
        {
            cv::polarToCart(magnitude, angle, x, y, angleInDegrees);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// MeanStdDev

PERF_TEST_P(Sz, Core_MeanStdDev, GPU_TYPICAL_MAT_SIZES)
{
    cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC1);
    fillRandom(src);

    cv::Scalar mean;
    cv::Scalar stddev;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        cv::gpu::meanStdDev(d_src, mean, stddev, d_buf);

        TEST_CYCLE()
        {
            cv::gpu::meanStdDev(d_src, mean, stddev, d_buf);
        }
    }
    else
    {
        cv::meanStdDev(src, mean, stddev);

        TEST_CYCLE()
        {
            cv::meanStdDev(src, mean, stddev);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Norm

DEF_PARAM_TEST(Sz_Depth_Norm, cv::Size, MatDepth, NormType);

PERF_TEST_P(Sz_Depth_Norm, Core_Norm, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32S, CV_32F),
    Values(NormType(cv::NORM_INF), NormType(cv::NORM_L1), NormType(cv::NORM_L2))))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int normType = GET_PARAM(2);

    cv::Mat src(size, depth);
    fillRandom(src);

    double dst;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        dst = cv::gpu::norm(d_src, normType, d_buf);

        TEST_CYCLE()
        {
            dst = cv::gpu::norm(d_src, normType, d_buf);
        }
    }
    else
    {
        dst = cv::norm(src, normType);

        TEST_CYCLE()
        {
            dst = cv::norm(src, normType);
        }
    }
    (void)dst;
}

//////////////////////////////////////////////////////////////////////
// NormDiff

DEF_PARAM_TEST(Sz_Norm, cv::Size, NormType);

PERF_TEST_P(Sz_Norm, Core_NormDiff, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(NormType(cv::NORM_INF), NormType(cv::NORM_L1), NormType(cv::NORM_L2))))
{
    cv::Size size = GET_PARAM(0);
    int normType = GET_PARAM(1);

    cv::Mat src1(size, CV_8UC1);
    fillRandom(src1);

    cv::Mat src2(size, CV_8UC1);
    fillRandom(src2);

    double dst;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);

        dst = cv::gpu::norm(d_src1, d_src2, normType);

        TEST_CYCLE()
        {
            dst = cv::gpu::norm(d_src1, d_src2, normType);
        }
    }
    else
    {
        dst = cv::norm(src1, src2, normType);

        TEST_CYCLE()
        {
            dst = cv::norm(src1, src2, normType);
        }
    }
    (void)dst;
}

//////////////////////////////////////////////////////////////////////
// Sum

PERF_TEST_P(Sz_Depth_Cn, Core_Sum, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32F),
    GPU_CHANNELS_1_3_4))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int channels = GET_PARAM(2);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Scalar dst;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        dst = cv::gpu::sum(d_src, d_buf);

        TEST_CYCLE()
        {
            dst = cv::gpu::sum(d_src, d_buf);
        }
    }
    else
    {
        dst = cv::sum(src);

        TEST_CYCLE()
        {
            dst = cv::sum(src);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// SumAbs

PERF_TEST_P(Sz_Depth_Cn, Core_SumAbs, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32F),
    GPU_CHANNELS_1_3_4))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int channels = GET_PARAM(2);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Scalar dst;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        dst = cv::gpu::absSum(d_src, d_buf);

        TEST_CYCLE()
        {
            dst = cv::gpu::absSum(d_src, d_buf);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// SumSqr

PERF_TEST_P(Sz_Depth_Cn, Core_SumSqr, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values<MatDepth>(CV_8U, CV_16U, CV_32F),
    GPU_CHANNELS_1_3_4))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int channels = GET_PARAM(2);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    cv::Scalar dst;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        dst = cv::gpu::sqrSum(d_src, d_buf);

        TEST_CYCLE()
        {
            dst = cv::gpu::sqrSum(d_src, d_buf);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// MinMax

PERF_TEST_P(Sz_Depth, Core_MinMax, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32F, CV_64F)))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    double minVal, maxVal;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        cv::gpu::minMax(d_src, &minVal, &maxVal, cv::gpu::GpuMat(), d_buf);

        TEST_CYCLE()
        {
            cv::gpu::minMax(d_src, &minVal, &maxVal, cv::gpu::GpuMat(), d_buf);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////////////////////
// MinMaxLoc

PERF_TEST_P(Sz_Depth, Core_MinMaxLoc, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32F, CV_64F)))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_valbuf, d_locbuf;

        cv::gpu::minMaxLoc(d_src, &minVal, &maxVal, &minLoc, &maxLoc, cv::gpu::GpuMat(), d_valbuf, d_locbuf);

        TEST_CYCLE()
        {
            cv::gpu::minMaxLoc(d_src, &minVal, &maxVal, &minLoc, &maxLoc, cv::gpu::GpuMat(), d_valbuf, d_locbuf);
        }
    }
    else
    {
        cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

        TEST_CYCLE()
        {
            cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// CountNonZero

PERF_TEST_P(Sz_Depth, Core_CountNonZero, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_32F, CV_64F)))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    int dst;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        dst = cv::gpu::countNonZero(d_src, d_buf);

        TEST_CYCLE()
        {
            dst = cv::gpu::countNonZero(d_src, d_buf);
        }
    }
    else
    {
        dst = cv::countNonZero(src);

        TEST_CYCLE()
        {
            dst = cv::countNonZero(src);
        }
    }
    (void)dst;
}

//////////////////////////////////////////////////////////////////////
// Reduce

CV_ENUM(ReduceCode, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)
#define ALL_REDUCE_CODES ValuesIn(ReduceCode::all())

enum {Rows = 0, Cols = 1};
CV_ENUM(ReduceDim, Rows, Cols)
#define ALL_REDUCE_DIMS ValuesIn(ReduceDim::all())

DEF_PARAM_TEST(Sz_Depth_Cn_Code_Dim, cv::Size, MatDepth, int, ReduceCode, ReduceDim);

PERF_TEST_P(Sz_Depth_Cn_Code_Dim, Core_Reduce, Combine(
    GPU_TYPICAL_MAT_SIZES,
    Values(CV_8U, CV_16U, CV_16S, CV_32F),
    Values(1, 2, 3, 4),
    ALL_REDUCE_CODES,
    ALL_REDUCE_DIMS))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int channels = GET_PARAM(2);
    int reduceOp = GET_PARAM(3);
    int dim = GET_PARAM(4);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::reduce(d_src, d_dst, dim, reduceOp);

        TEST_CYCLE()
        {
            cv::gpu::reduce(d_src, d_dst, dim, reduceOp);
        }
    }
    else
    {
        cv::Mat dst;

        cv::reduce(src, dst, dim, reduceOp);

        TEST_CYCLE()
        {
            cv::reduce(src, dst, dim, reduceOp);
        }
    }
}

} // namespace
