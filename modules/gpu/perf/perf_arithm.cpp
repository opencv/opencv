#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Transpose

GPU_PERF_TEST(Transpose, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::transpose(src, dst);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int flipCode = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::flip(src, dst, flipCode);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Flip, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_8UC4),
                        testing::Values((int) HORIZONTAL_AXIS, (int) VERTICAL_AXIS, (int) BOTH_AXIS)));

//////////////////////////////////////////////////////////////////////
// LUT

GPU_PERF_TEST(LUT, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);
    cv::Mat lut(1, 256, CV_8UC1);

    declare.in(src_host, lut, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::LUT(src, lut, dst);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat x_host(size, CV_32FC1);
    cv::Mat y_host(size, CV_32FC1);

    fill(x_host, -100.0, 100.0);
    fill(y_host, -100.0, 100.0);

    cv::gpu::GpuMat x(x_host);
    cv::gpu::GpuMat y(y_host);
    cv::gpu::GpuMat magnitude;
    cv::gpu::GpuMat angle;

    TEST_CYCLE()
    {
        cv::gpu::cartToPolar(x, y, magnitude, angle);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// PolarToCart

GPU_PERF_TEST(PolarToCart, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat magnitude_host(size, CV_32FC1);
    cv::Mat angle_host(size, CV_32FC1);

    fill(magnitude_host, 0.0, 100.0);
    fill(angle_host, 0.0, 360.0);

    cv::gpu::GpuMat magnitude(magnitude_host);
    cv::gpu::GpuMat angle(angle_host);
    cv::gpu::GpuMat x;
    cv::gpu::GpuMat y;

    TEST_CYCLE()
    {
        cv::gpu::polarToCart(magnitude, angle, x, y, true);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// AddMat

GPU_PERF_TEST(AddMat, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, type);
    cv::Mat src2_host(size, type);

    fill(src1_host, 0.0, 100.0);
    fill(src2_host, 0.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::add(src1, src2, dst);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s(1,2,3,4);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::add(src, s, dst);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_32FC1);

    fill(src_host, 0.0, 10.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::exp(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Exp, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Pow

GPU_PERF_TEST(Pow, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::pow(src, 0.5, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Pow, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Compare

GPU_PERF_TEST(Compare, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, type);
    cv::Mat src2_host(size, type);

    declare.in(src1_host, src2_host, WARMUP_RNG);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::compare(src1, src2, dst, cv::CMP_EQ);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::bitwise_not(src, dst);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, type);
    cv::Mat src2_host(size, type);

    declare.in(src1_host, src2_host, WARMUP_RNG);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::bitwise_and(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseAnd, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32SC1)));

//////////////////////////////////////////////////////////////////////
// Min

GPU_PERF_TEST(Min, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, type);
    cv::Mat src2_host(size, type);

    declare.in(src1_host, src2_host, WARMUP_RNG);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst(size, type);

    TEST_CYCLE()
    {
        cv::gpu::min(src1, src2, dst);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host); 
    cv::Scalar mean;
    cv::Scalar stddev;

    TEST_CYCLE()
    {
        cv::gpu::meanStdDev(src, mean, stddev);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, MeanStdDev, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Norm

GPU_PERF_TEST(Norm, cv::gpu::DeviceInfo, cv::Size, perf::MatType, NormType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int normType = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    double dst;
    cv::gpu::GpuMat buf;

    TEST_CYCLE()
    {
        dst = cv::gpu::norm(src, normType, buf);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, CV_8UC1);
    cv::Mat src2_host(size, CV_8UC1);

    declare.in(src1_host, src2_host, WARMUP_RNG);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    double dst;

    TEST_CYCLE()
    {
        dst = cv::gpu::norm(src1, src2, normType);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar dst;
    cv::gpu::GpuMat buf;

    TEST_CYCLE()
    {
        dst = cv::gpu::sum(src, buf);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, Sum, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// MinMax

GPU_PERF_TEST(MinMax, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    double minVal, maxVal;
    cv::gpu::GpuMat buf;

    TEST_CYCLE()
    {
        cv::gpu::minMax(src, &minVal, &maxVal, cv::gpu::GpuMat(), buf);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, MinMax, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// MinMaxLoc

GPU_PERF_TEST(MinMaxLoc, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::gpu::GpuMat valbuf, locbuf;

    TEST_CYCLE()
    {
        cv::gpu::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, cv::gpu::GpuMat(), valbuf, locbuf);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    fill(src_host, 0.0, 1.0);

    cv::gpu::GpuMat src(src_host);
    int dst;
    cv::gpu::GpuMat buf;

    TEST_CYCLE()
    {
        dst = cv::gpu::countNonZero(src, buf);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, type);
    cv::Mat src2_host(size, type);

    fill(src1_host, 0.0, 100.0);
    fill(src2_host, 0.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::addWeighted(src1, 0.5, src2, 0.5, 0.0, dst);
    }

    cv::Mat dst_host(dst);
}

INSTANTIATE_TEST_CASE_P(Arithm, AddWeighted, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Reduce

GPU_PERF_TEST(Reduce, cv::gpu::DeviceInfo, cv::Size, perf::MatType, FlipCode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int dim = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    fill(src_host, 0.0, 10.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::reduce(src, dst, dim, CV_REDUCE_MIN);
    }

    cv::Mat dst_host(dst);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, CV_32FC1);
    cv::Mat src2_host(size, CV_32FC1);
    cv::Mat src3_host(size, CV_32FC1);

    fill(src1_host, 0.0, 10.0);
    fill(src2_host, 0.0, 10.0);
    fill(src3_host, 0.0, 10.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat src3(src3_host);
    cv::gpu::GpuMat dst;

    declare.time(5.0);

    TEST_CYCLE()
    {
        cv::gpu::gemm(src1, src2, 1.0, src3, 1.0, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Arithm, GEMM, testing::Combine(
                        ALL_DEVICES, 
                        testing::Values(cv::Size(512, 512), cv::Size(1024, 1024), cv::Size(2048, 2048))));

#endif
