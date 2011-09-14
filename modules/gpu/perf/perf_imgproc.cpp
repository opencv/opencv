#include "perf_precomp.hpp"

PERF_TEST_P(DevInfo_Size_MatType_Interpolation_BorderMode, remap, testing::Combine(testing::ValuesIn(devices()), 
                                                                  testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                  testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_32FC1), 
                                                                  testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR, (int)INTER_CUBIC),
                                                                  testing::Values((int)BORDER_REFLECT101, (int)BORDER_REPLICATE, (int)BORDER_CONSTANT)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int interpolation = std::tr1::get<3>(GetParam());
    int borderMode = std::tr1::get<4>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    Mat xmap_host(size, CV_32FC1);
    Mat ymap_host(size, CV_32FC1);
    randu(xmap_host, -300, size.width + 300);
    randu(ymap_host, -300, size.height + 300);

    GpuMat xmap(xmap_host);
    GpuMat ymap(ymap_host);

    declare.time(3.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        remap(src, dst, xmap, ymap, interpolation, borderMode);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, meanShiftFiltering, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());
    
    Mat rgba;
    cvtColor(img, rgba, CV_BGR2BGRA);

    GpuMat src(rgba);
    GpuMat dst(src.size(), CV_8UC4);

    declare.time(5.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        meanShiftFiltering(src, dst, 50, 50);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, meanShiftProc, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());
    
    Mat rgba;
    cvtColor(img, rgba, CV_BGR2BGRA);

    GpuMat src(rgba);
    GpuMat dstr(src.size(), CV_8UC4);
    GpuMat dstsp(src.size(), CV_16SC2);

    declare.time(5.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        meanShiftProc(src, dstr, dstsp, 50, 50);
    }

    Mat dstr_host = dstr;
    Mat dstsp_host = dstsp;

    SANITY_CHECK(dstr_host);
    SANITY_CHECK(dstsp_host);
}

PERF_TEST_P(DevInfo, meanShiftSegmentation, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());
    
    Mat rgba;
    cvtColor(img, rgba, CV_BGR2BGRA);

    GpuMat src(rgba);
    Mat dst(src.size(), CV_8UC4);

    declare.time(5.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        meanShiftSegmentation(src, dst, 10, 10, 20);
    }

    SANITY_CHECK(dst);
}

PERF_TEST_P(DevInfo_Size_MatType, drawColorDisp, testing::Combine(testing::ValuesIn(devices()),
                                                                  testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                  testing::Values(CV_8UC1, CV_16SC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_8UC1);
    declare.in(src_host, WARMUP_RNG);
    src_host.convertTo(src_host, type);

    GpuMat src(src_host);
    GpuMat dst(size, CV_8UC4);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        drawColorDisp(src, dst, 255);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, reprojectImageTo3D, testing::Combine(testing::ValuesIn(devices()),
                                                                       testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                       testing::Values(CV_8UC1, CV_16SC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, CV_32FC4);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        reprojectImageTo3D(src, dst, Mat::ones(4, 4, CV_32FC1));
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_CvtColorInfo, cvtColor, testing::Combine(testing::ValuesIn(devices()), 
            testing::Values(GPU_TYPICAL_MAT_SIZES), 
            testing::Values(CV_8UC1, CV_16UC1, CV_32FC1),
            testing::Values(CvtColorInfo(4, 4, CV_RGBA2BGRA), CvtColorInfo(4, 1, CV_BGRA2GRAY), CvtColorInfo(1, 4, CV_GRAY2BGRA), 
                            CvtColorInfo(4, 4, CV_BGR2XYZ), CvtColorInfo(4, 4, CV_BGR2YCrCb), CvtColorInfo(4, 4, CV_YCrCb2BGR), 
                            CvtColorInfo(4, 4, CV_BGR2HSV), CvtColorInfo(4, 4, CV_HSV2BGR))))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    CvtColorInfo info = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_MAKETYPE(type, info.scn));

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, CV_MAKETYPE(type, info.dcn));

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        cvtColor(src, dst, info.code, info.dcn);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, threshold, testing::Combine(testing::ValuesIn(devices()),
                                                              testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                              testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)))
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
        threshold(src, dst, 100.0, 255.0, THRESH_BINARY);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_Interpolation_SizeCoeff, resize, testing::Combine(testing::ValuesIn(devices()),
                                                                                   testing::Values(szSXGA, sz1080p), 
                                                                                   testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_32FC1),
                                                                                   testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR, (int)INTER_CUBIC),
                                                                                   testing::Values(0.5, 2.0)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int interpolation = std::tr1::get<3>(GetParam());
    double f = std::tr1::get<4>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    declare.time(1.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        resize(src, dst, Size(), f, f, interpolation);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_Interpolation, warpAffine, testing::Combine(testing::ValuesIn(devices()),
                                                                             testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                             testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                                                                             testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR, (int)INTER_CUBIC)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int interpolation = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    static double reflect[2][3] = { {-1,  0, 0},
                                    { 0, -1, 0}};
    reflect[0][2] = size.width;
    reflect[1][2] = size.height;
    Mat M(2, 3, CV_64F, (void*)reflect); 

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        warpAffine(src, dst, M, size, interpolation);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_Interpolation, warpPerspective, testing::Combine(testing::ValuesIn(devices()),
                                                                                  testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                                  testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                                                                                  testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR, (int)INTER_CUBIC)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int interpolation = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    static double reflect[3][3] = { {-1,  0, 0},
                                    { 0, -1, 0},
                                    { 0, 0, 1}};
    reflect[0][2] = size.width;
    reflect[1][2] = size.height;
    Mat M(3, 3, CV_64F, (void*)reflect); 

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        warpPerspective(src, dst, M, size, interpolation);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, buildWarpPlaneMaps, testing::Combine(testing::ValuesIn(devices()),
                                                               testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    GpuMat map_x(size, CV_32FC1);
    GpuMat map_y(size, CV_32FC1);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        buildWarpPlaneMaps(size, Rect(0, 0, size.width, size.height), Mat::ones(3, 3, CV_32FC1), 1.0, 1.0, 1.0, map_x, map_y);
    }

    Mat map_x_host(map_x);
    Mat map_y_host(map_y);

    SANITY_CHECK(map_x_host);
    SANITY_CHECK(map_y_host);
}

PERF_TEST_P(DevInfo_Size, buildWarpCylindricalMaps, testing::Combine(testing::ValuesIn(devices()),
                                                               testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    GpuMat map_x(size, CV_32FC1);
    GpuMat map_y(size, CV_32FC1);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        buildWarpCylindricalMaps(size, Rect(0, 0, size.width, size.height), Mat::ones(3, 3, CV_32FC1), 1.0, 1.0, map_x, map_y);
    }

    Mat map_x_host(map_x);
    Mat map_y_host(map_y);

    SANITY_CHECK(map_x_host);
    SANITY_CHECK(map_y_host);
}

PERF_TEST_P(DevInfo_Size, buildWarpSphericalMaps, testing::Combine(testing::ValuesIn(devices()),
                                                               testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    GpuMat map_x(size, CV_32FC1);
    GpuMat map_y(size, CV_32FC1);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        buildWarpSphericalMaps(size, Rect(0, 0, size.width, size.height), Mat::ones(3, 3, CV_32FC1), 1.0, 1.0, map_x, map_y);
    }

    Mat map_x_host(map_x);
    Mat map_y_host(map_y);

    SANITY_CHECK(map_x_host);
    SANITY_CHECK(map_y_host);
}

PERF_TEST_P(DevInfo_Size_MatType_Interpolation, rotate, testing::Combine(testing::ValuesIn(devices()),
                                                                         testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                         testing::Values(CV_8UC1, CV_8UC4),
                                                                         testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR, (int)INTER_CUBIC)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int interpolation = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        rotate(src, dst, size, 30.0, 0, 0, interpolation);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, copyMakeBorder, testing::Combine(testing::ValuesIn(devices()),
                                                                         testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                         testing::Values(CV_8UC1, CV_8UC4, CV_32SC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        copyMakeBorder(src, dst, 5, 5, 5, 5);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, integralBuffered, testing::Combine(testing::ValuesIn(devices()),
                                                             testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;
    GpuMat buf;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        integralBuffered(src, dst, buf);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, integral, testing::Combine(testing::ValuesIn(devices()),
                                                     testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat sum, sqsum;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        integral(src, sum, sqsum);
    }

    Mat sum_host(sum);
    Mat sqsum_host(sqsum);

    SANITY_CHECK(sum_host);
    SANITY_CHECK(sqsum_host);
}

PERF_TEST_P(DevInfo_Size, sqrIntegral, testing::Combine(testing::ValuesIn(devices()),
                                                        testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        sqrIntegral(src, dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, columnSum, testing::Combine(testing::ValuesIn(devices()),
                                                      testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_32FC1);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        columnSum(src, dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_MatType, cornerHarris, testing::Combine(testing::ValuesIn(devices()),
                                                            testing::Values(CV_8UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());
    
    Mat img = readImage("gpu/stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    GpuMat src(img);
    GpuMat dst;
    GpuMat Dx;
    GpuMat Dy;

    int blockSize = 3;
    int ksize = 7;        
    double k = 0.5;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        cornerHarris(src, dst, Dx, Dy, blockSize, ksize, k);
    }

    Mat dst_host(dst);
    Mat Dx_host(Dx);
    Mat Dy_host(Dy);

    SANITY_CHECK(dst_host);
    SANITY_CHECK(Dx_host);
    SANITY_CHECK(Dy_host);
}

PERF_TEST_P(DevInfo_MatType, cornerMinEigenVal, testing::Combine(testing::ValuesIn(devices()),
                                                            testing::Values(CV_8UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());
    
    Mat img = readImage("gpu/stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    GpuMat src(img);
    GpuMat dst;
    GpuMat Dx;
    GpuMat Dy;

    int blockSize = 3;
    int ksize = 7; 

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        cornerMinEigenVal(src, dst, Dx, Dy, blockSize, ksize);
    }

    Mat dst_host(dst);
    Mat Dx_host(Dx);
    Mat Dy_host(Dy);

    SANITY_CHECK(dst_host);
    SANITY_CHECK(Dx_host);
    SANITY_CHECK(Dy_host);
}

PERF_TEST_P(DevInfo_Size_MatType, mulSpectrums, testing::Combine(testing::ValuesIn(devices()),
                                                                 testing::Values(GPU_TYPICAL_MAT_SIZES),
                                                                 testing::Values(CV_8UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat a_host(size, CV_32FC2);
    Mat b_host(size, CV_32FC2);

    declare.in(a_host, b_host, WARMUP_RNG);

    GpuMat a(a_host);
    GpuMat b(b_host);
    GpuMat dst;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        mulSpectrums(a, b, dst, 0);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, dft, testing::Combine(testing::ValuesIn(devices()),
                                                testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_32FC2);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    declare.time(2.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        dft(src, dst, size);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, convolve, testing::Combine(testing::ValuesIn(devices()),
                                                testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat image_host(size, CV_32FC1);
    Mat templ_host(size, CV_32FC1);

    declare.in(image_host, templ_host, WARMUP_RNG);

    GpuMat image(image_host);
    GpuMat templ(templ_host);
    GpuMat dst;
    ConvolveBuf buf;

    declare.time(2.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        convolve(image, templ, dst, false, buf);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, pyrDown, testing::Combine(testing::ValuesIn(devices()),
                                                            testing::Values(GPU_TYPICAL_MAT_SIZES),
                                                            testing::Values(CV_8UC1, CV_8UC4, CV_16SC3, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        pyrDown(src, dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, pyrUp, testing::Combine(testing::ValuesIn(devices()),
                                                          testing::Values(GPU_TYPICAL_MAT_SIZES),
                                                          testing::Values(CV_8UC1, CV_8UC4, CV_16SC3, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        pyrUp(src, dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, blendLinear, testing::Combine(testing::ValuesIn(devices()),
                                                                testing::Values(GPU_TYPICAL_MAT_SIZES),
                                                                testing::Values(CV_8UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat img1_host(size, type);
    Mat img2_host(size, type);

    declare.in(img1_host, img2_host, WARMUP_RNG);

    GpuMat img1(img1_host);
    GpuMat img2(img2_host);
    GpuMat weights1(size, CV_32FC1, Scalar::all(0.5));
    GpuMat weights2(size, CV_32FC1, Scalar::all(0.5));
    GpuMat dst;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        blendLinear(img1, img2, weights1, weights2, dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, Canny, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat image_host = readImage("gpu/perf/aloe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    ASSERT_FALSE(image_host.empty());

    GpuMat image(image_host);
    GpuMat dst;
    CannyBuf buf;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        Canny(image, buf, dst, 50.0, 100.0);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size, calcHist, testing::Combine(testing::ValuesIn(devices()),
                                                     testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat hist;
    GpuMat buf;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        calcHist(src, hist, buf);
    }

    Mat hist_host(hist);

    SANITY_CHECK(hist_host);
}

PERF_TEST_P(DevInfo_Size, equalizeHist, testing::Combine(testing::ValuesIn(devices()),
                                                         testing::Values(GPU_TYPICAL_MAT_SIZES)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;
    GpuMat hist;
    GpuMat buf;

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        equalizeHist(src, dst, hist, buf);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}
