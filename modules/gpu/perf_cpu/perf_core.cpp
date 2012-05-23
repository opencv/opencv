#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Merge

GPU_PERF_TEST(Merge, cv::gpu::DeviceInfo, cv::Size, MatDepth, Channels)
{
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    std::vector<cv::Mat> src(channels);
    for (int i = 0; i < channels; ++i)
        src[i] = cv::Mat(size, depth, cv::Scalar::all(i));

    cv::Mat dst;

    cv::merge(src, dst);

    TEST_CYCLE()
    {
        cv::merge(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    cv::Mat src(size, CV_MAKE_TYPE(depth, channels), cv::Scalar(1, 2, 3, 4));

    std::vector<cv::Mat> dst;

    cv::split(src, dst);

    TEST_CYCLE()
    {
        cv::split(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0.0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0.0, 100.0);

    cv::Mat dst;

    cv::add(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::add(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 100.0);

    cv::Scalar s(1, 2, 3, 4);
    cv::Mat dst;

    cv::add(src, s, dst);

    TEST_CYCLE()
    {
        cv::add(src, s, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0.0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0.0, 100.0);

    cv::Mat dst;

    cv::subtract(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::subtract(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 100.0);

    cv::Scalar s(1, 2, 3, 4);
    cv::Mat dst;

    cv::subtract(src, s, dst);

    TEST_CYCLE()
    {
        cv::subtract(src, s, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0.0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0.0, 100.0);

    cv::Mat dst;

    cv::multiply(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::multiply(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 100.0);

    cv::Scalar s(1, 2, 3, 4);
    cv::Mat dst;

    cv::multiply(src, s, dst);

    TEST_CYCLE()
    {
        cv::multiply(src, s, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0.0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0.0, 100.0);

    cv::Mat dst;

    cv::divide(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::divide(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 100.0);

    cv::Scalar s(1, 2, 3, 4);
    cv::Mat dst;

    cv::divide(src, s, dst);

    TEST_CYCLE()
    {
        cv::divide(src, s, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 100.0);

    double scale = 100.0;
    cv::Mat dst;

    cv::divide(scale, src, dst);

    TEST_CYCLE()
    {
        cv::divide(scale, src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0.0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0.0, 100.0);

    cv::Mat dst;

    cv::absdiff(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::absdiff(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 100.0);

    cv::Scalar s(1, 2, 3, 4);
    cv::Mat dst;

    cv::absdiff(src, s, dst);

    TEST_CYCLE()
    {
        cv::absdiff(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, AbsDiff_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////
// Sqrt

GPU_PERF_TEST(Sqrt, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 100.0);

    cv::Mat dst;

    cv::sqrt(src, dst);

    TEST_CYCLE()
    {
        cv::sqrt(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 1.0, 100.0);

    cv::Mat dst;

    cv::log(src, dst);

    TEST_CYCLE()
    {
        cv::log(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 1.0, 10.0);

    cv::Mat dst;

    cv::exp(src, dst);

    TEST_CYCLE()
    {
        cv::exp(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 1.0, 10.0);

    cv::Mat dst;

    cv::pow(src, 2.3, dst);

    TEST_CYCLE()
    {
        cv::pow(src, 2.3, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int cmp_code = GET_PARAM(3);

    cv::Mat src1(size, depth);
    fill(src1, 0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0, 100.0);

    cv::Mat dst;

    cv::compare(src1, src2, dst, cmp_code);

    TEST_CYCLE()
    {
        cv::compare(src1, src2, dst, cmp_code);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int cmp_code = GET_PARAM(3);

    cv::Mat src(size, depth);
    fill(src, 0, 100.0);

    cv::Scalar s = cv::Scalar::all(50);
    cv::Mat dst;

    cv::compare(src, s, dst, cmp_code);

    TEST_CYCLE()
    {
        cv::compare(src, s, dst, cmp_code);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0, 100.0);

    cv::Mat dst;

    cv::bitwise_not(src, dst);

    TEST_CYCLE()
    {
        cv::bitwise_not(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0, 100.0);

    cv::Mat dst;

    cv::bitwise_and(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::bitwise_and(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fill(src, 0, 100.0);

    cv::Scalar s = cv::Scalar(50, 50, 50, 50);
    cv::Mat dst;

    cv::bitwise_and(src, s, dst);

    TEST_CYCLE()
    {
        cv::bitwise_and(src, s, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0, 100.0);

    cv::Mat dst;

    cv::bitwise_or(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::bitwise_or(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fill(src, 0, 100.0);

    cv::Scalar s = cv::Scalar(50, 50, 50, 50);
    cv::Mat dst;

    cv::bitwise_or(src, s, dst);

    TEST_CYCLE()
    {
        cv::bitwise_or(src, s, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0, 100.0);

    cv::Mat src2(size, depth);
    fill(src2, 0, 100.0);

    cv::Mat dst;

    cv::bitwise_xor(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::bitwise_xor(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fill(src, 0, 100.0);

    cv::Scalar s = cv::Scalar(50, 50, 50, 50);
    cv::Mat dst;

    cv::bitwise_xor(src, s, dst);

    TEST_CYCLE()
    {
        cv::bitwise_xor(src, s, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Bitwise_Xor_Scalar, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16U, CV_32S),
    testing::Values<Channels>(1, 3, 4)));

//////////////////////////////////////////////////////////////////////
// Min_Mat

GPU_PERF_TEST(Min_Mat, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0, 255.0);

    cv::Mat src2(size, depth);
    fill(src2, 0, 255.0);

    cv::Mat dst;

    cv::min(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::min(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0, 255.0);

    double val = 50.0;
    cv::Mat dst;

    cv::min(src, val, dst);

    TEST_CYCLE()
    {
        cv::min(src, val, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src1(size, depth);
    fill(src1, 0, 255.0);

    cv::Mat src2(size, depth);
    fill(src2, 0, 255.0);

    cv::Mat dst;

    cv::max(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::max(src1, src2, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0, 255.0);

    double val = 50.0;
    cv::Mat dst;

    cv::max(src, val, dst);

    TEST_CYCLE()
    {
        cv::max(src, val, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth1 = GET_PARAM(2);
    int depth2 = GET_PARAM(3);
    int dst_depth = GET_PARAM(4);

    cv::Mat src1(size, depth1);
    fill(src1, 0, 100.0);

    cv::Mat src2(size, depth2);
    fill(src2, 0, 100.0);

    cv::Mat dst;

    cv::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);

    TEST_CYCLE()
    {
        cv::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int flags = GET_PARAM(3);

    cv::Mat src1(size, type);
    fill(src1, 0.0, 10.0);

    cv::Mat src2(size, type);
    fill(src2, 0.0, 10.0);

    cv::Mat src3(size, type);
    fill(src3, 0.0, 10.0);

    cv::Mat dst;

    cv::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);

    declare.time(50.0);

    TEST_CYCLE()
    {
        cv::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);
    fill(src, 0.0, 100.0);

    cv::Mat dst;

    cv::transpose(src, dst);

    TEST_CYCLE()
    {
        cv::transpose(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int flipCode = GET_PARAM(3);

    cv::Mat src(size, type);
    fill(src, 0.0, 100.0);

    cv::Mat dst;

    cv::flip(src, dst, flipCode);

    TEST_CYCLE()
    {
        cv::flip(src, dst, flipCode);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);
    fill(src, 0.0, 100.0);

    cv::Mat lut(1, 256, CV_8UC1);
    fill(lut, 0.0, 100.0);

    cv::Mat dst;

    cv::LUT(src, lut, dst);

    TEST_CYCLE()
    {
        cv::LUT(src, lut, dst);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);
    fill(src, 0.0, 100.0);

    cv::Mat lut(1, 256, CV_MAKE_TYPE(CV_8U, src.channels()));
    fill(lut, 0.0, 100.0);

    cv::Mat dst;

    cv::LUT(src, lut, dst);

    TEST_CYCLE()
    {
        cv::LUT(src, lut, dst);
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
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_32FC2);
    fill(src, -100.0, 100.0);

    cv::Mat srcs[2];
    cv::split(src, srcs);

    cv::Mat dst;

    cv::magnitude(srcs[0], srcs[1], dst);

    TEST_CYCLE()
    {
        cv::magnitude(srcs[0], srcs[1], dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Magnitude_Complex, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Magnitude

GPU_PERF_TEST(Magnitude, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src1(size, CV_32FC1);
    fill(src1, -100.0, 100.0);

    cv::Mat src2(size, CV_32FC1);
    fill(src2, -100.0, 100.0);

    cv::Mat dst;

    cv::magnitude(src1, src2, dst);

    TEST_CYCLE()
    {
        cv::magnitude(src1, src2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Magnitude, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Phase

IMPLEMENT_PARAM_CLASS(AngleInDegrees, bool)

GPU_PERF_TEST(Phase, cv::gpu::DeviceInfo, cv::Size, AngleInDegrees)
{
    cv::Size size = GET_PARAM(1);
    bool angleInDegrees = GET_PARAM(2);

    cv::Mat src1(size, CV_32FC1);
    fill(src1, -100.0, 100.0);

    cv::Mat src2(size, CV_32FC1);
    fill(src2, -100.0, 100.0);

    cv::Mat dst;

    cv::phase(src1, src2, dst, angleInDegrees);

    TEST_CYCLE()
    {
        cv::phase(src1, src2, dst, angleInDegrees);
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
    cv::Size size = GET_PARAM(1);
    bool angleInDegrees = GET_PARAM(2);

    cv::Mat src1(size, CV_32FC1);
    fill(src1, -100.0, 100.0);

    cv::Mat src2(size, CV_32FC1);
    fill(src2, -100.0, 100.0);

    cv::Mat magnitude;
    cv::Mat angle;

    cv::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);

    TEST_CYCLE()
    {
        cv::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);
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
    cv::Size size = GET_PARAM(1);
    bool angleInDegrees = GET_PARAM(2);

    cv::Mat magnitude(size, CV_32FC1);
    fill(magnitude, 0.0, 100.0);

    cv::Mat angle(size, CV_32FC1);
    fill(angle, 0.0, angleInDegrees ? 360.0 : 2 * CV_PI);

    cv::Mat x;
    cv::Mat y;

    cv::polarToCart(magnitude, angle, x, y, angleInDegrees);

    TEST_CYCLE()
    {
        cv::polarToCart(magnitude, angle, x, y, angleInDegrees);
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
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);
    fill(src, 0.0, 255.0);

    cv::Scalar mean;
    cv::Scalar stddev;

    cv::meanStdDev(src, mean, stddev);

    TEST_CYCLE()
    {
        cv::meanStdDev(src, mean, stddev);
    }
}

INSTANTIATE_TEST_CASE_P(Core, MeanStdDev, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Norm

GPU_PERF_TEST(Norm, cv::gpu::DeviceInfo, cv::Size, MatDepth, NormType)
{
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int normType = GET_PARAM(3);

    cv::Mat src(size, depth);
    fill(src, 0.0, 255.0);

    double dst;
    cv::Mat buf;

    dst = cv::norm(src, normType);

    TEST_CYCLE()
    {
        dst = cv::norm(src, normType);
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
    cv::Size size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    cv::Mat src1(size, CV_8UC1);
    fill(src1, 0.0, 255.0);

    cv::Mat src2(size, CV_8UC1);
    fill(src2, 0.0, 255.0);

    double dst;

    dst = cv::norm(src1, src2, normType);

    TEST_CYCLE()
    {
        dst = cv::norm(src1, src2, normType);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);
    fill(src, 0.0, 255.0);

    cv::Scalar dst;

    dst = cv::sum(src);

    TEST_CYCLE()
    {
        dst = cv::sum(src);
    }
}

INSTANTIATE_TEST_CASE_P(Core, Sum, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatType>(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

//////////////////////////////////////////////////////////////////////
// MinMaxLoc

GPU_PERF_TEST(MinMaxLoc, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 255.0);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

    TEST_CYCLE()
    {
        cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0.0, 1.5);

    int dst;

    dst = cv::countNonZero(src);

    TEST_CYCLE()
    {
        dst = cv::countNonZero(src);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int channels = GET_PARAM(3);
    int reduceOp = GET_PARAM(4);
    int dim = GET_PARAM(5);

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fill(src, 0.0, 10.0);

    cv::Mat dst;

    cv::reduce(src, dst, dim, reduceOp);

    TEST_CYCLE()
    {
        cv::reduce(src, dst, dim, reduceOp);
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
