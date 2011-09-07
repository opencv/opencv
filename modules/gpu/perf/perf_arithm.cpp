#include "perf_precomp.hpp"

PERF_TEST_P(DevInfo_Size_MatType, transpose, testing::Combine(testing::ValuesIn(devices()), 
                                                              testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                              testing::Values(CV_8UC1, CV_32SC1, CV_64FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size.width, size.height, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        transpose(src, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_FlipCode, flip, testing::Combine(testing::ValuesIn(devices()), 
                                                                  testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                  testing::Values(CV_8UC1, CV_8UC4), 
                                                                  testing::Values((int)HORIZONTAL_AXIS, (int)VERTICAL_AXIS, (int)BOTH_AXIS)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int flipCode = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        flip(src, dst, flipCode);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, LUT, testing::Combine(testing::ValuesIn(devices()), 
                                                        testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                        testing::Values(CV_8UC1, CV_8UC3)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);
    Mat lut(1, 256, CV_8UC1);

    declare.in(src_host, lut, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        LUT(src, lut, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, cartToPolar, testing::Combine(testing::ValuesIn(devices()), 
                                                        testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat x_host(size, CV_32FC1);
    Mat y_host(size, CV_32FC1);

    declare.in(x_host, y_host, WARMUP_RNG);

    GpuMat x(x_host);
    GpuMat y(y_host);
    GpuMat magnitude(size, CV_32FC1);
    GpuMat angle(size, CV_32FC1);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        cartToPolar(x, y, magnitude, angle);
    }

    Mat magnitude_host = magnitude;
    Mat angle_host = angle;

    SANITY_CHECK(magnitude_host);
    SANITY_CHECK(angle_host);
}

PERF_TEST_P(DevInfo_Size, polarToCart, testing::Combine(testing::ValuesIn(devices()), 
                                                        testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat magnitude_host(size, CV_32FC1);
    Mat angle_host(size, CV_32FC1);

    declare.in(magnitude_host, angle_host, WARMUP_RNG);

    GpuMat magnitude(magnitude_host);
    GpuMat angle(angle_host);
    GpuMat x(size, CV_32FC1);
    GpuMat y(size, CV_32FC1);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        polarToCart(magnitude, angle, x, y);
    }

    Mat x_host = x;
    Mat y_host = angle;

    SANITY_CHECK(x_host);
    SANITY_CHECK(y_host);
}

PERF_TEST_P(DevInfo_Size_MatType, addMat, testing::Combine(testing::ValuesIn(devices()), 
                                                           testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                           testing::Values(CV_8UC1, CV_8UC4, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat a_host(size, type);
    Mat b_host(size, type);

    declare.in(a_host, b_host, WARMUP_RNG);

    GpuMat a(a_host);
    GpuMat b(b_host);
    GpuMat c(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        add(a, b, c);
    }

    Mat c_host = c;

    SANITY_CHECK(c_host);
}

PERF_TEST_P(DevInfo_Size_MatType, addScalar, testing::Combine(testing::ValuesIn(devices()), 
                                                              testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                              testing::Values(CV_32FC1, CV_32FC2)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat a_host(size, type);

    declare.in(a_host, WARMUP_RNG);

    GpuMat a(a_host);
    Scalar b(1,2,3,4);
    GpuMat c(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        add(a, b, c);
    }

    Mat c_host = c;

    SANITY_CHECK(c_host);
}

PERF_TEST_P(DevInfo_Size, exp, testing::Combine(testing::ValuesIn(devices()), 
                                                testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat a_host(size, CV_32FC1);

    declare.in(a_host, WARMUP_RNG);

    GpuMat a(a_host);
    GpuMat b(size, CV_32FC1);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        exp(a, b);
    }

    Mat b_host = b;

    SANITY_CHECK(b_host);
}

PERF_TEST_P(DevInfo_Size_MatType, pow, testing::Combine(testing::ValuesIn(devices()), 
                                                        testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                        testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        pow(src, 2.0, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_CmpOp, compare, testing::Combine(testing::ValuesIn(devices()), 
                                                                  testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                  testing::Values(CV_8UC4, CV_32FC1), 
                                                                  testing::Values((int)CMP_NE, (int)CMP_EQ)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int cmpop = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src1_host(size, type);
    Mat src2_host(size, type);

    declare.in(src1_host, src2_host, WARMUP_RNG);

    GpuMat src1(src1_host);
    GpuMat src2(src2_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        compare(src1, src2, dst, cmpop);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, bitwise_not, testing::Combine(testing::ValuesIn(devices()), 
                                                                testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                testing::Values(CV_8UC1, CV_16UC1, CV_32SC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        bitwise_not(src, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, bitwise_and, testing::Combine(testing::ValuesIn(devices()), 
                                                                testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                testing::Values(CV_8UC1, CV_16UC1, CV_32SC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src1_host(size, type);
    Mat src2_host(size, type);

    declare.in(src1_host, src2_host, WARMUP_RNG);

    GpuMat src1(src1_host);
    GpuMat src2(src2_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        bitwise_and(src1, src2, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, min, testing::Combine(testing::ValuesIn(devices()), 
                                                        testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                        testing::Values(CV_8UC1, CV_16UC1, CV_32SC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src1_host(size, type);
    Mat src2_host(size, type);

    declare.in(src1_host, src2_host, WARMUP_RNG);

    GpuMat src1(src1_host);
    GpuMat src2(src2_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        min(src1, src2, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}
