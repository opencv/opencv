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

    if (PERF_RUN_GPU())
    {
        std::vector<cv::gpu::GpuMat> d_src(channels);
        for (int i = 0; i < channels; ++i)
            d_src[i].upload(src[i]);

        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::merge(d_src, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-12);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::merge(src, dst);

        CPU_SANITY_CHECK(dst, 1e-12);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);

        std::vector<cv::gpu::GpuMat> d_dst;

        TEST_CYCLE() cv::gpu::split(d_src, d_dst);

        cv::gpu::GpuMat first = d_dst[0];
        GPU_SANITY_CHECK(first, 1e-12);
    }
    else
    {
        std::vector<cv::Mat> dst;

        TEST_CYCLE() cv::split(src, dst);

        CPU_SANITY_CHECK(dst, 1e-12);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::add(d_src1, d_src2, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::add(src1, src2, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::add(d_src, s, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::add(src, s, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::subtract(d_src1, d_src2, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::subtract(src1, src2, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::subtract(d_src, s, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::subtract(src, s, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::multiply(d_src1, d_src2, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::multiply(src1, src2, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::multiply(d_src, s, d_dst);

        TEST_CYCLE() cv::gpu::multiply(d_src, s, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::multiply(src, s, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::divide(d_src1, d_src2, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::divide(src1, src2, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::divide(d_src, s, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::divide(src, s, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::divide(s, d_src, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::divide(s, src, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::absdiff(d_src1, d_src2, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::absdiff(src1, src2, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::absdiff(d_src, s, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::absdiff(src, s, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::abs(d_src, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else FAIL_NO_CPU();
}

//////////////////////////////////////////////////////////////////////
// Sqr

PERF_TEST_P(Sz_Depth, Core_Sqr, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::sqr(d_src, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else FAIL_NO_CPU();
}

//////////////////////////////////////////////////////////////////////
// Sqrt

PERF_TEST_P(Sz_Depth, Core_Sqrt, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::sqrt(d_src, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::sqrt(src, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::log(d_src, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::log(src, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::exp(d_src, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() TEST_CYCLE() cv::exp(src, dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::pow(d_src, power, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::pow(src, power,dst);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::compare(d_src1, d_src2, d_dst, cmp_code);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::compare(src1, src2, dst, cmp_code);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::compare(d_src, s, d_dst, cmp_code);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::compare(src, s, dst, cmp_code);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::bitwise_not(d_src,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_not(src,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::bitwise_and(d_src1, d_src2,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_and(src1, src2,dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::bitwise_and(d_src, s,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_and(src, s,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::bitwise_or(d_src1, d_src2,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_or(src1, src2,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::bitwise_or(d_src, s,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_or(src, s,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::bitwise_xor(d_src1, d_src2,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_xor(src1, src2,dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::bitwise_xor(d_src, s,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_xor(src, s,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::rshift(d_src, val,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        FAIL_NO_CPU();
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::lshift(d_src, val,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        FAIL_NO_CPU();
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::min(d_src1, d_src2,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::min(src1, src2,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::min(d_src, val,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::min(src, val,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::max(d_src1, d_src2,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::max(src1, src2,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::max(d_src, val,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::max(src, val,dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::addWeighted(d_src1, 0.5, d_src2, 0.5, 10.0, d_dst, dst_depth);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_src3(src3);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst, flags);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        declare.time(50.0);

        TEST_CYCLE() cv::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::transpose(d_src,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::transpose(src,dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Flip

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)
#define ALL_FLIP_CODES ValuesIn(FlipCode::all())

DEF_PARAM_TEST(Sz_Depth_Cn_Code, cv::Size, MatDepth, MatCn, FlipCode);

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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::flip(d_src, d_dst, flipCode);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::flip(src, dst, flipCode);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::LUT(d_src, lut,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::LUT(src, lut, dst);

        CPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::LUT(d_src, lut,d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::LUT(src, lut, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeComplex

PERF_TEST_P(Sz, Core_MagnitudeComplex, GPU_TYPICAL_MAT_SIZES)
{
    cv::Size size = GetParam();

    cv::Mat src(size, CV_32FC2);
    fillRandom(src, -100.0, 100.0);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::magnitude(d_src,d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat xy[2];
        cv::split(src, xy);

        cv::Mat dst;

        TEST_CYCLE() cv::magnitude(xy[0], xy[1], dst);

        CPU_SANITY_CHECK(dst, 1e-8);
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeSqrComplex

PERF_TEST_P(Sz, Core_MagnitudeSqrComplex, GPU_TYPICAL_MAT_SIZES)
{
    cv::Size size = GetParam();

    cv::Mat src(size, CV_32FC2);
    fillRandom(src, -100.0, 100.0);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::magnitudeSqr(d_src, d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        FAIL_NO_CPU();
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::magnitude(d_src1, d_src2, d_dst);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::magnitude(src1, src2, dst);

        CPU_SANITY_CHECK(dst, 1e-8);

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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::magnitudeSqr(d_src1, d_src2, d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        FAIL_NO_CPU();
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::phase(d_src1, d_src2, d_dst, angleInDegrees);

        GPU_SANITY_CHECK(d_dst, 1e-8);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::phase(src1, src2, dst, angleInDegrees);

        CPU_SANITY_CHECK(dst, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_magnitude;
        cv::gpu::GpuMat d_angle;

        TEST_CYCLE() cv::gpu::cartToPolar(d_src1, d_src2, d_magnitude, d_angle, angleInDegrees);

        GPU_SANITY_CHECK(d_magnitude, 1e-8);
        GPU_SANITY_CHECK(d_angle, 1e-8);

    }
    else
    {
        cv::Mat magnitude;
        cv::Mat angle;

        TEST_CYCLE() cv::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);

        CPU_SANITY_CHECK(magnitude, 1e-8);
        CPU_SANITY_CHECK(angle, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_magnitude(magnitude);
        cv::gpu::GpuMat d_angle(angle);
        cv::gpu::GpuMat d_x;
        cv::gpu::GpuMat d_y;

        TEST_CYCLE() cv::gpu::polarToCart(d_magnitude, d_angle, d_x, d_y, angleInDegrees);

        GPU_SANITY_CHECK(d_x, 1e-8);
        GPU_SANITY_CHECK(d_y, 1e-8);
    }
    else
    {
        cv::Mat x;
        cv::Mat y;

        TEST_CYCLE() cv::polarToCart(magnitude, angle, x, y, angleInDegrees);

        CPU_SANITY_CHECK(x, 1e-8);
        CPU_SANITY_CHECK(y, 1e-8);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::meanStdDev(d_src, mean, stddev, d_buf);
    }
    else
    {
        TEST_CYCLE() cv::meanStdDev(src, mean, stddev);
    }

    GPU_SANITY_CHECK(stddev, 1e-6);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() dst = cv::gpu::norm(d_src, normType, cv::gpu::GpuMat(), d_buf);
    }
    else
    {
        TEST_CYCLE() dst = cv::norm(src, normType);
    }

    SANITY_CHECK(dst, 1e-6);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);

        TEST_CYCLE() dst = cv::gpu::norm(d_src1, d_src2, normType);

    }
    else
    {
        TEST_CYCLE() dst = cv::norm(src1, src2, normType);
    }

    SANITY_CHECK(dst, 1e-6);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() dst = cv::gpu::sum(d_src, cv::gpu::GpuMat(), d_buf);
    }
    else
    {
        TEST_CYCLE() dst = cv::sum(src);
    }

    double error = (depth == CV_32F) ? 3e+1 : 1e-6;
    SANITY_CHECK(dst,  error);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() dst = cv::gpu::absSum(d_src, cv::gpu::GpuMat(), d_buf);

        SANITY_CHECK(dst, 1e-6);
    }
    else
    {
        FAIL_NO_CPU();
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() dst = cv::gpu::sqrSum(d_src, cv::gpu::GpuMat(), d_buf);

        SANITY_CHECK(dst, 1e-6);
    }
    else
    {
        FAIL_NO_CPU();
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::minMax(d_src, &minVal, &maxVal, cv::gpu::GpuMat(), d_buf);

        SANITY_CHECK(minVal);
        SANITY_CHECK(maxVal);
    }
    else
    {
        FAIL_NO_CPU();
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_valbuf, d_locbuf;

        TEST_CYCLE() cv::gpu::minMaxLoc(d_src, &minVal, &maxVal, &minLoc, &maxLoc, cv::gpu::GpuMat(), d_valbuf, d_locbuf);
    }
    else
    {
        TEST_CYCLE() cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
    }

    SANITY_CHECK(minVal, 1e-12);
    SANITY_CHECK(maxVal, 1e-12);

    // unsupported by peft system
    //SANITY_CHECK(minLoc);
    //SANITY_CHECK(maxLoc);
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

    int dst = 0;

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() dst = cv::gpu::countNonZero(d_src, d_buf);
    }
    else
    {
        TEST_CYCLE() dst = cv::countNonZero(src);
    }

    SANITY_CHECK(dst);
}

//////////////////////////////////////////////////////////////////////
// Reduce

CV_ENUM(ReduceCode, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)
#define ALL_REDUCE_CODES ValuesIn(ReduceCode::all())

enum {Rows = 0, Cols = 1};
CV_ENUM(ReduceDim, Rows, Cols)
#define ALL_REDUCE_DIMS ValuesIn(ReduceDim::all())

DEF_PARAM_TEST(Sz_Depth_Cn_Code_Dim, cv::Size, MatDepth, MatCn, ReduceCode, ReduceDim);

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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        TEST_CYCLE() cv::gpu::reduce(d_src, d_dst, dim, reduceOp);

        GPU_SANITY_CHECK(d_dst, 1);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::reduce(src, dst, dim, reduceOp);

        CPU_SANITY_CHECK(dst, 1);
    }
}

} // namespace
