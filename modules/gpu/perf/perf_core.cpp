#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Merge

GPU_PERF_TEST(Merge, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    std::vector<cv::gpu::GpuMat> src(channels);
    for (int i = 0; i < channels; ++i)
        src[i] = cv::gpu::GpuMat(size, depth, cv::Scalar::all(i));

    cv::gpu::GpuMat dst;

    cv::gpu::merge(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::merge(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Merge, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F),
    testing::Values<Channels>(2, 3, 4)));

//////////////////////////////////////////////////////////////////////
// Split

GPU_PERF_TEST(Split, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    cv::gpu::GpuMat src(size, CV_MAKE_TYPE(depth, channels), cv::Scalar(1, 2, 3, 4));

    std::vector<cv::gpu::GpuMat> dst;

    cv::gpu::split(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::split(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Split, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F),
    testing::Values<Channels>(2, 3, 4)));

//////////////////////////////////////////////////////////////////////
// Add_Mat

GPU_PERF_TEST(Add_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0.0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::add(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::add(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Add_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Add_Scalar

GPU_PERF_TEST(Add_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s(1, 2, 3, 4);
    cv::gpu::GpuMat dst;

    cv::gpu::add(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::add(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Add_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Subtract_Mat

GPU_PERF_TEST(Subtract_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0.0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::subtract(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::subtract(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Subtract_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Subtract_Scalar

GPU_PERF_TEST(Subtract_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s(1, 2, 3, 4);
    cv::gpu::GpuMat dst;

    cv::gpu::subtract(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::subtract(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Subtract_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Multiply_Mat

GPU_PERF_TEST(Multiply_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0.0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::multiply(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::multiply(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Multiply_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Multiply_Scalar

GPU_PERF_TEST(Multiply_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s(1, 2, 3, 4);
    cv::gpu::GpuMat dst;

    cv::gpu::multiply(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::multiply(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Multiply_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Divide_Mat

GPU_PERF_TEST(Divide_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0.0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::divide(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::divide(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Divide_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Divide_Scalar

GPU_PERF_TEST(Divide_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s(1, 2, 3, 4);
    cv::gpu::GpuMat dst;

    cv::gpu::divide(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::divide(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Divide_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Divide_Scalar_Inv

GPU_PERF_TEST(Divide_Scalar_Inv, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    double scale = 100.0;
    cv::gpu::GpuMat dst;

    cv::gpu::divide(scale, src, dst);

    TEST_CYCLE()
    {
        cv::gpu::divide(scale, src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Divide_Scalar_Inv, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// AbsDiff_Mat

GPU_PERF_TEST(AbsDiff_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0.0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::absdiff(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::absdiff(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, AbsDiff_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// AbsDiff_Scalar

GPU_PERF_TEST(AbsDiff_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s(1, 2, 3, 4);
    cv::gpu::GpuMat dst;

    cv::gpu::absdiff(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::absdiff(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, AbsDiff_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Abs

GPU_PERF_TEST(Abs, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::abs(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::abs(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Abs, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_16S, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Sqr

GPU_PERF_TEST(Sqr, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::sqr(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::sqr(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Sqr, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16S, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Sqrt

GPU_PERF_TEST(Sqrt, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::sqrt(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::sqrt(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Sqrt, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16S, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Log

GPU_PERF_TEST(Log, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 1.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::log(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::log(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Log, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16S, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Exp

GPU_PERF_TEST(Exp, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 1.0, 10.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::exp(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::exp(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Exp, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16S, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Pow

GPU_PERF_TEST(Pow, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 1.0, 10.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::pow(src, 2.3, dst);

    TEST_CYCLE()
    {
        cv::gpu::pow(src, 2.3, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Pow, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16S, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Compare_Mat

CV_ENUM(CmpCode, cv::CMP_EQ, cv::CMP_GT, cv::CMP_GE, cv::CMP_LT, cv::CMP_LE, cv::CMP_NE)
#define ALL_CMP_CODES testing::Values(CmpCode(cv::CMP_EQ), CmpCode(cv::CMP_NE), CmpCode(cv::CMP_GT), CmpCode(cv::CMP_GE), CmpCode(cv::CMP_LT), CmpCode(cv::CMP_LE))

GPU_PERF_TEST(Compare_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth, CmpCode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int cmp_code = GET_PARAM(3);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::compare(src1, src2, dst, cmp_code);

    TEST_CYCLE()
    {
        cv::gpu::compare(src1, src2, dst, cmp_code);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Compare_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F),
    ALL_CMP_CODES));

//////////////////////////////////////////////////////////////////////
// Compare_Scalar

GPU_PERF_TEST(Compare_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth, CmpCode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int cmp_code = GET_PARAM(3);

    cv::Mat src_host(size, depth);
    fill(src_host, 0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s = cv::Scalar::all(50);
    cv::gpu::GpuMat dst;

    cv::gpu::compare(src, s, dst, cmp_code);

    TEST_CYCLE()
    {
        cv::gpu::compare(src, s, dst, cmp_code);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Compare_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F),
    ALL_CMP_CODES));

//////////////////////////////////////////////////////////////////////
// Bitwise_Not

GPU_PERF_TEST(Bitwise_Not, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::bitwise_not(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::bitwise_not(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_Not, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S)));

//////////////////////////////////////////////////////////////////////
// Bitwise_And_Mat

GPU_PERF_TEST(Bitwise_And_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::bitwise_and(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::bitwise_and(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_And_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S)));

//////////////////////////////////////////////////////////////////////
// Bitwise_And_Scalar

GPU_PERF_TEST(Bitwise_And_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s = cv::Scalar(50, 50, 50, 50);
    cv::gpu::GpuMat dst;

    cv::gpu::bitwise_and(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::bitwise_and(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_And_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S),
    testing::Values<Channels>(1, 3, 4)));

//////////////////////////////////////////////////////////////////////
// Bitwise_Or_Mat

GPU_PERF_TEST(Bitwise_Or_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::bitwise_or(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::bitwise_or(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_Or_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S)));

//////////////////////////////////////////////////////////////////////
// Bitwise_Or_Scalar

GPU_PERF_TEST(Bitwise_Or_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s = cv::Scalar(50, 50, 50, 50);
    cv::gpu::GpuMat dst;

    cv::gpu::bitwise_or(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::bitwise_or(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_Or_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S),
    testing::Values<Channels>(1, 3, 4)));

//////////////////////////////////////////////////////////////////////
// Bitwise_Xor_Mat

GPU_PERF_TEST(Bitwise_Xor_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0, 100.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::bitwise_xor(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::bitwise_xor(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_Xor_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S)));

//////////////////////////////////////////////////////////////////////
// Bitwise_Xor_Scalar

GPU_PERF_TEST(Bitwise_Xor_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar s = cv::Scalar(50, 50, 50, 50);
    cv::gpu::GpuMat dst;

    cv::gpu::bitwise_xor(src, s, dst);

    TEST_CYCLE()
    {
        cv::gpu::bitwise_xor(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_Xor_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S),
    testing::Values<Channels>(1, 3, 4)));

//////////////////////////////////////////////////////////////////////
// RShift

GPU_PERF_TEST(RShift, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar_<int> val = cv::Scalar_<int>::all(4);
    cv::gpu::GpuMat dst;

    cv::gpu::rshift(src, val, dst);

    TEST_CYCLE()
    {
        cv::gpu::rshift(src, val, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, RShift, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S),
    testing::Values<Channels>(1, 3, 4)));

//////////////////////////////////////////////////////////////////////
// LShift

GPU_PERF_TEST(LShift, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar_<int> val = cv::Scalar_<int>::all(4);
    cv::gpu::GpuMat dst;

    cv::gpu::lshift(src, val, dst);

    TEST_CYCLE()
    {
        cv::gpu::lshift(src, val, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, LShift, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S),
    testing::Values<Channels>(1, 3, 4)));

//////////////////////////////////////////////////////////////////////
// Min_Mat

GPU_PERF_TEST(Min_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0, 255.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0, 255.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::min(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::min(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Min_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Min_Scalar

GPU_PERF_TEST(Min_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0, 255.0);

    cv::gpu::GpuMat src(src_host);
    double val = 50.0;
    cv::gpu::GpuMat dst;

    cv::gpu::min(src, val, dst);

    TEST_CYCLE()
    {
        cv::gpu::min(src, val, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Min_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Max_Mat

GPU_PERF_TEST(Max_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1_host(size, depth);
    fill(src1_host, 0, 255.0);

    cv::Mat src2_host(size, depth);
    fill(src2_host, 0, 255.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::max(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::max(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Max_Mat, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F)));

//////////////////////////////////////////////////////////////////////
// Max_Scalar

GPU_PERF_TEST(Max_Scalar, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0, 255.0);

    cv::gpu::GpuMat src(src_host);
    double val = 50.0;
    cv::gpu::GpuMat dst;

    cv::gpu::max(src, val, dst);

    TEST_CYCLE()
    {
        cv::gpu::max(src, val, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Max_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F)));

//////////////////////////////////////////////////////////////////////
// AddWeighted

GPU_PERF_TEST(AddWeighted, cv::gpu::DeviceInfo, cv::Size, MatDepth, MatDepth, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth1 = GET_PARAM(2);
    int depth2 = GET_PARAM(3);
    int dst_depth = GET_PARAM(4);

    cv::Mat src1_host(size, depth1);
    fill(src1_host, 0, 100.0);

    cv::Mat src2_host(size, depth2);
    fill(src2_host, 0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);

    TEST_CYCLE()
    {
        cv::gpu::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);
    }
}

INSTANTIATE_TEST_CASE_P(Core, AddWeighted, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F),
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F),
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// GEMM

CV_FLAGS(GemmFlags, 0, cv::GEMM_1_T, cv::GEMM_2_T, cv::GEMM_3_T)
#define ALL_GEMM_FLAGS testing::Values(GemmFlags(0), GemmFlags(cv::GEMM_1_T), GemmFlags(cv::GEMM_2_T), GemmFlags(cv::GEMM_3_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_3_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T | cv::GEMM_3_T))

GPU_PERF_TEST(GEMM, cv::gpu::DeviceInfo, cv::Size, MatType, GemmFlags)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int flags = GET_PARAM(3);

    cv::Mat src1_host(size, type);
    fill(src1_host, 0.0, 10.0);

    cv::Mat src2_host(size, type);
    fill(src2_host, 0.0, 10.0);

    cv::Mat src3_host(size, type);
    fill(src3_host, 0.0, 10.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat src3(src3_host);
    cv::gpu::GpuMat dst;

    cv::gpu::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);

    declare.time(5.0);

    TEST_CYCLE()
    {
        cv::gpu::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);
    }
}

INSTANTIATE_TEST_CASE_P(Core, GEMM, testing::Combine(
    ALL_DEVICES,
    testing::Values(cv::Size(512, 512), cv::Size(1024, 1024)),
    testing::Values<MatType>(CV_32FC1, CV_32FC2, CV_64FC1, CV_64FC2),
    ALL_GEMM_FLAGS));

//////////////////////////////////////////////////////////////////////
// Transpose

GPU_PERF_TEST(Transpose, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::transpose(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::transpose(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Transpose, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC1, CV_8UC4, CV_16UC2, CV_16SC2, CV_32SC1, CV_32SC2, CV_64FC1)));

//////////////////////////////////////////////////////////////////////
// Flip

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)
#define ALL_FLIP_CODES testing::Values(FlipCode(FLIP_BOTH), FlipCode(FLIP_X), FlipCode(FLIP_Y))

GPU_PERF_TEST(Flip, cv::gpu::DeviceInfo, cv::Size, MatType, FlipCode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int flipCode = GET_PARAM(3);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::flip(src, dst, flipCode);

    TEST_CYCLE()
    {
        cv::gpu::flip(src, dst, flipCode);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Flip, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
    ALL_FLIP_CODES));

//////////////////////////////////////////////////////////////////////
// LUT_OneChannel

GPU_PERF_TEST(LUT_OneChannel, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 100.0);

    cv::Mat lut(1, 256, CV_8UC1);
    fill(lut, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::LUT(src, lut, dst);

    TEST_CYCLE()
    {
        cv::gpu::LUT(src, lut, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, LUT_OneChannel, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC1, CV_8UC3)));

//////////////////////////////////////////////////////////////////////
// LUT_MultiChannel

GPU_PERF_TEST(LUT_MultiChannel, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 100.0);

    cv::Mat lut(1, 256, CV_MAKE_TYPE(CV_8U, src_host.channels()));
    fill(lut, 0.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::LUT(src, lut, dst);

    TEST_CYCLE()
    {
        cv::gpu::LUT(src, lut, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, LUT_MultiChannel, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC3)));

//////////////////////////////////////////////////////////////////////
// Magnitude_Complex

GPU_PERF_TEST(Magnitude_Complex, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_32FC2);
    fill(src_host, -100.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::magnitude(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::magnitude(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Magnitude_Complex, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Magnitude_Sqr_Complex

GPU_PERF_TEST(Magnitude_Sqr_Complex, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_32FC2);
    fill(src_host, -100.0, 100.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::magnitudeSqr(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::magnitudeSqr(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Magnitude_Sqr_Complex, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Magnitude

GPU_PERF_TEST(Magnitude, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src1_host(size, CV_32FC1);
    fill(src1_host, -100.0, 100.0);

    cv::Mat src2_host(size, CV_32FC1);
    fill(src2_host, -100.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::magnitude(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::magnitude(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Magnitude, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Magnitude_Sqr

GPU_PERF_TEST(Magnitude_Sqr, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src1_host(size, CV_32FC1);
    fill(src1_host, -100.0, 100.0);

    cv::Mat src2_host(size, CV_32FC1);
    fill(src2_host, -100.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::magnitudeSqr(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::gpu::magnitudeSqr(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Magnitude_Sqr, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Phase

IMPLEMENT_PARAM_CLASS(AngleInDegrees, bool)

GPU_PERF_TEST(Phase, cv::gpu::DeviceInfo, cv::Size, AngleInDegrees)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    bool angleInDegrees = GET_PARAM(2);

    cv::Mat src1_host(size, CV_32FC1);
    fill(src1_host, -100.0, 100.0);

    cv::Mat src2_host(size, CV_32FC1);
    fill(src2_host, -100.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::phase(src1, src2, dst, angleInDegrees);

    TEST_CYCLE()
    {
        cv::gpu::phase(src1, src2, dst, angleInDegrees);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Phase, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<AngleInDegrees>(false, true)));

//////////////////////////////////////////////////////////////////////
// CartToPolar

GPU_PERF_TEST(CartToPolar, cv::gpu::DeviceInfo, cv::Size, AngleInDegrees)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    bool angleInDegrees = GET_PARAM(2);

    cv::Mat src1_host(size, CV_32FC1);
    fill(src1_host, -100.0, 100.0);

    cv::Mat src2_host(size, CV_32FC1);
    fill(src2_host, -100.0, 100.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat magnitude;
    cv::gpu::GpuMat angle;

    cv::gpu::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);

    TEST_CYCLE()
    {
        cv::gpu::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);
    }
}

INSTANTIATE_TEST_CASE_P(Core, CartToPolar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<AngleInDegrees>(false, true)));

//////////////////////////////////////////////////////////////////////
// PolarToCart

GPU_PERF_TEST(PolarToCart, cv::gpu::DeviceInfo, cv::Size, AngleInDegrees)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    bool angleInDegrees = GET_PARAM(2);

    cv::Mat magnitude_host(size, CV_32FC1);
    fill(magnitude_host, 0.0, 100.0);

    cv::Mat angle_host(size, CV_32FC1);
    fill(angle_host, 0.0, angleInDegrees ? 360.0 : 2 * CV_PI);

    cv::gpu::GpuMat magnitude(magnitude_host);
    cv::gpu::GpuMat angle(angle_host);
    cv::gpu::GpuMat x;
    cv::gpu::GpuMat y;

    cv::gpu::polarToCart(magnitude, angle, x, y, angleInDegrees);

    TEST_CYCLE()
    {
        cv::gpu::polarToCart(magnitude, angle, x, y, angleInDegrees);
    }
}

INSTANTIATE_TEST_CASE_P(Core, PolarToCart, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<AngleInDegrees>(false, true)));

//////////////////////////////////////////////////////////////////////
// MeanStdDev

GPU_PERF_TEST(MeanStdDev, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_8UC1);
    fill(src_host, 0.0, 255.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::gpu::GpuMat buf;

    cv::gpu::meanStdDev(src, mean, stddev, buf);

    TEST_CYCLE()
    {
        cv::gpu::meanStdDev(src, mean, stddev, buf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, MeanStdDev, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Norm

GPU_PERF_TEST(Norm, cv::gpu::DeviceInfo, cv::Size, MatDepth, NormType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int normType = GET_PARAM(3);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 255.0);

    cv::gpu::GpuMat src(src_host);
    double dst;
    cv::gpu::GpuMat buf;

    dst = cv::gpu::norm(src, normType, buf);

    TEST_CYCLE()
    {
        dst = cv::gpu::norm(src, normType, buf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Norm, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S, CV_32F),
    testing::Values(NormType(cv::NORM_INF), NormType(cv::NORM_L1), NormType(cv::NORM_L2))));

//////////////////////////////////////////////////////////////////////
// NormDiff

GPU_PERF_TEST(NormDiff, cv::gpu::DeviceInfo, cv::Size, NormType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    cv::Mat src1_host(size, CV_8UC1);
    fill(src1_host, 0.0, 255.0);

    cv::Mat src2_host(size, CV_8UC1);
    fill(src2_host, 0.0, 255.0);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    double dst;

    dst = cv::gpu::norm(src1, src2, normType);

    TEST_CYCLE()
    {
        dst = cv::gpu::norm(src1, src2, normType);
    }
}

INSTANTIATE_TEST_CASE_P(Core, NormDiff, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(NormType(cv::NORM_INF), NormType(cv::NORM_L1), NormType(cv::NORM_L2))));

//////////////////////////////////////////////////////////////////////
// Sum

GPU_PERF_TEST(Sum, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 255.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar dst;
    cv::gpu::GpuMat buf;

    dst = cv::gpu::sum(src, buf);

    TEST_CYCLE()
    {
        dst = cv::gpu::sum(src, buf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Sum, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

//////////////////////////////////////////////////////////////////////
// Sum_Abs

GPU_PERF_TEST(Sum_Abs, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 255.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar dst;
    cv::gpu::GpuMat buf;

    dst = cv::gpu::absSum(src, buf);

    TEST_CYCLE()
    {
        dst = cv::gpu::absSum(src, buf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Sum_Abs, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

//////////////////////////////////////////////////////////////////////
// Sum_Sqr

GPU_PERF_TEST(Sum_Sqr, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 255.0);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar dst;
    cv::gpu::GpuMat buf;

    dst = cv::gpu::sqrSum(src, buf);

    TEST_CYCLE()
    {
        dst = cv::gpu::sqrSum(src, buf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Sum_Sqr, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

//////////////////////////////////////////////////////////////////////
// MinMax

GPU_PERF_TEST(MinMax, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 255.0);

    cv::gpu::GpuMat src(src_host);
    double minVal, maxVal;
    cv::gpu::GpuMat buf;

    cv::gpu::minMax(src, &minVal, &maxVal, cv::gpu::GpuMat(), buf);

    TEST_CYCLE()
    {
        cv::gpu::minMax(src, &minVal, &maxVal, cv::gpu::GpuMat(), buf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, MinMax, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// MinMaxLoc

GPU_PERF_TEST(MinMaxLoc, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 255.0);

    cv::gpu::GpuMat src(src_host);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::gpu::GpuMat valbuf, locbuf;

    cv::gpu::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, cv::gpu::GpuMat(), valbuf, locbuf);

    TEST_CYCLE()
    {
        cv::gpu::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, cv::gpu::GpuMat(), valbuf, locbuf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, MinMaxLoc, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// CountNonZero

GPU_PERF_TEST(CountNonZero, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0.0, 1.5);

    cv::gpu::GpuMat src(src_host);
    int dst;
    cv::gpu::GpuMat buf;

    dst = cv::gpu::countNonZero(src, buf);

    TEST_CYCLE()
    {
        dst = cv::gpu::countNonZero(src, buf);
    }
}

INSTANTIATE_TEST_CASE_P(Core, CountNonZero, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Reduce

CV_ENUM(ReduceCode, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)
#define ALL_REDUCE_CODES testing::Values<ReduceCode>(CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)

enum {Rows = 0, Cols = 1};
CV_ENUM(ReduceDim, Rows, Cols)

GPU_PERF_TEST(Reduce, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels, ReduceCode, ReduceDim)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);
    int reduceOp = GET_PARAM(4);
    int dim = GET_PARAM(5);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src_host(size, type);
    fill(src_host, 0.0, 10.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::reduce(src, dst, dim, reduceOp);

    TEST_CYCLE()
    {
        cv::gpu::reduce(src, dst, dim, reduceOp);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Reduce, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_16S, CV_32F),
    testing::Values<Channels>(1, 2, 3, 4),
    ALL_REDUCE_CODES,
    testing::Values(ReduceDim(Rows), ReduceDim(Cols))));

#endif
