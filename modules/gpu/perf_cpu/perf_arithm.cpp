#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Transpose

GPU_PERF_TEST(Transpose, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::transpose(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Transpose, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_32SC1, CV_64FC1)));

//////////////////////////////////////////////////////////////////////
// Flip

GPU_PERF_TEST(Flip, cv::gpu::DeviceInfo, cv::Size, perf::MatType, FlipCode)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int flipCode = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::flip(src, dst, flipCode);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Flip, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                        testing::Values((int) HORIZONTAL_AXIS, (int) VERTICAL_AXIS, (int) BOTH_AXIS)));

//////////////////////////////////////////////////////////////////////
// LUT

GPU_PERF_TEST(LUT, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);
    cv::Mat lut(1, 256, CV_8UC1);

    declare.in(src, lut, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::LUT(src, lut, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, LUT, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3)));

//////////////////////////////////////////////////////////////////////
// CartToPolar

GPU_PERF_TEST(CartToPolar, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat x(size, CV_32FC1);
    cv::Mat y(size, CV_32FC1);

    fill(x, -100.0, 100.0);
    fill(y, -100.0, 100.0);

    cv::Mat magnitude;
    cv::Mat angle;

    TEST_CYCLE()
    {
        cv::cartToPolar(x, y, magnitude, angle);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// PolarToCart

GPU_PERF_TEST(PolarToCart, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat magnitude(size, CV_32FC1);
    cv::Mat angle(size, CV_32FC1);

    fill(magnitude, 0.0, 100.0);
    fill(angle, 0.0, 360.0);

    cv::Mat x;
    cv::Mat y;

    TEST_CYCLE()
    {
        cv::polarToCart(magnitude, angle, x, y, true);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// AddMat

GPU_PERF_TEST(AddMat, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);

    fill(src1, 0.0, 100.0);
    fill(src2, 0.0, 100.0);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::add(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, AddMat, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// AddScalar

GPU_PERF_TEST(AddScalar, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    fill(src, 0.0, 100.0);

    cv::Scalar s(1,2,3,4);
    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::add(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, AddScalar, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Exp

GPU_PERF_TEST(Exp, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_32FC1);

    fill(src, 0.0, 10.0);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::exp(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Exp, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Pow

GPU_PERF_TEST(Pow, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::pow(src, 0.5, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Pow, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(perf::MatType(CV_32FC1))));

//////////////////////////////////////////////////////////////////////
// Compare

GPU_PERF_TEST(Compare, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);

    declare.in(src1, src2, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::compare(src1, src2, dst, cv::CMP_EQ);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Compare, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// BitwiseNot

GPU_PERF_TEST(BitwiseNot, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::bitwise_not(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseNot, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32SC1)));

//////////////////////////////////////////////////////////////////////
// BitwiseAnd

GPU_PERF_TEST(BitwiseAnd, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);

    declare.in(src1, src2, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::bitwise_and(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseAnd, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32SC1)));

//////////////////////////////////////////////////////////////////////
// BitwiseScalarAnd

GPU_PERF_TEST(BitwiseScalarAnd, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;
    cv::Scalar sc = cv::Scalar(123, 123, 123, 123);

    TEST_CYCLE()
    {
        cv::bitwise_and(src, sc, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseScalarAnd, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32SC1, CV_32SC3, CV_32SC4)));

//////////////////////////////////////////////////////////////////////
// Min

GPU_PERF_TEST(Min, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);

    declare.in(src1, src2, WARMUP_RNG);

    cv::Mat dst(size, type);

    TEST_CYCLE()
    {
        cv::min(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Min, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32SC1)));

//////////////////////////////////////////////////////////////////////
// MeanStdDev

GPU_PERF_TEST(MeanStdDev, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);

    declare.in(src, WARMUP_RNG);

    cv::Scalar mean;
    cv::Scalar stddev;

    TEST_CYCLE()
    {
        cv::meanStdDev(src, mean, stddev);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, MeanStdDev, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Norm

GPU_PERF_TEST(Norm, cv::gpu::DeviceInfo, cv::Size, perf::MatType, NormType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int normType = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    double dst;

    TEST_CYCLE()
    {
        dst = cv::norm(src, normType);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Norm, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32SC1),
                        testing::Values((int) cv::NORM_INF, (int) cv::NORM_L1, (int) cv::NORM_L2)));

//////////////////////////////////////////////////////////////////////
// NormDiff

GPU_PERF_TEST(NormDiff, cv::gpu::DeviceInfo, cv::Size, NormType)
{
    cv::Size size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    cv::Mat src1(size, CV_8UC1);
    cv::Mat src2(size, CV_8UC1);

    declare.in(src1, src2, WARMUP_RNG);

    double dst;

    TEST_CYCLE()
    {
        dst = cv::norm(src1, src2, normType);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, NormDiff, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values((int) cv::NORM_INF, (int) cv::NORM_L1, (int) cv::NORM_L2)));

//////////////////////////////////////////////////////////////////////
// Sum

GPU_PERF_TEST(Sum, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Scalar dst;

    TEST_CYCLE()
    {
        dst = cv::sum(src);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Sum, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// MinMaxLoc

GPU_PERF_TEST(MinMaxLoc, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    TEST_CYCLE()
    {
        cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, MinMaxLoc, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// CountNonZero

GPU_PERF_TEST(CountNonZero, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    fill(src, 0.0, 1.0);

    int dst;

    TEST_CYCLE()
    {
        dst = cv::countNonZero(src);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, CountNonZero, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// AddWeighted

GPU_PERF_TEST(AddWeighted, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);

    fill(src1, 0.0, 100.0);
    fill(src2, 0.0, 100.0);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::addWeighted(src1, 0.5, src2, 0.5, 0.0, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, AddWeighted, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Reduce

GPU_PERF_TEST(Reduce, cv::gpu::DeviceInfo, cv::Size, perf::MatType, FlipCode)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int dim = GET_PARAM(3);

    cv::Mat src(size, type);

    fill(src, 0.0, 10.0);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::reduce(src, dst, dim, CV_REDUCE_MIN);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Reduce, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1),
                        testing::Values((int) HORIZONTAL_AXIS, (int) VERTICAL_AXIS)));

//////////////////////////////////////////////////////////////////////
// GEMM

GPU_PERF_TEST(GEMM, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src1(size, CV_32FC1);
    cv::Mat src2(size, CV_32FC1);
    cv::Mat src3(size, CV_32FC1);

    fill(src1, 0.0, 10.0);
    fill(src2, 0.0, 10.0);
    fill(src3, 0.0, 10.0);

    cv::Mat dst;

    declare.time(15.0);

    TEST_CYCLE()
    {
        cv::gemm(src1, src2, 1.0, src3, 1.0, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, GEMM, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(cv::Size(512, 512), cv::Size(1024, 1024), cv::Size(2048, 2048))));

#endif
