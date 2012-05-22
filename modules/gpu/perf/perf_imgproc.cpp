#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Remap

GPU_PERF_TEST(Remap, cv::gpu::DeviceInfo, cv::Size, perf::MatType, Interpolation, BorderMode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);
    cv::Mat xmap_host(size, CV_32FC1);
    cv::Mat ymap_host(size, CV_32FC1);

    declare.in(src_host, xmap_host, ymap_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat xmap(xmap_host);
    cv::gpu::GpuMat ymap(ymap_host);
    cv::gpu::GpuMat dst;

    declare.time(3.0);

    TEST_CYCLE()
    {
        cv::gpu::remap(src, dst, xmap, ymap, interpolation, borderMode);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Remap, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_32FC1),
                        testing::Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC),
                        testing::Values((int) cv::BORDER_REFLECT101, (int) cv::BORDER_REPLICATE, (int) cv::BORDER_CONSTANT)));

//////////////////////////////////////////////////////////////////////
// MeanShiftFiltering

GPU_PERF_TEST_1(MeanShiftFiltering, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat src(rgba);
    cv::gpu::GpuMat dst;

    declare.time(5.0);

    TEST_CYCLE()
    {
        cv::gpu::meanShiftFiltering(src, dst, 50, 50);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftFiltering, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// MeanShiftProc

GPU_PERF_TEST_1(MeanShiftProc, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat src(rgba);
    cv::gpu::GpuMat dstr;
    cv::gpu::GpuMat dstsp;

    declare.time(5.0);

    TEST_CYCLE()
    {
        cv::gpu::meanShiftProc(src, dstr, dstsp, 50, 50);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftProc, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// MeanShiftSegmentation

GPU_PERF_TEST_1(MeanShiftSegmentation, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat src(rgba);
    cv::Mat dst;

    declare.time(5.0);

    TEST_CYCLE()
    {
        meanShiftSegmentation(src, dst, 10, 10, 20);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftSegmentation, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// DrawColorDisp

GPU_PERF_TEST(DrawColorDisp, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::drawColorDisp(src, dst, 255);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, DrawColorDisp, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16SC1)));

//////////////////////////////////////////////////////////////////////
// ReprojectImageTo3D

GPU_PERF_TEST(ReprojectImageTo3D, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
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
        cv::gpu::reprojectImageTo3D(src, dst, cv::Mat::ones(4, 4, CV_32FC1));
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ReprojectImageTo3D, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16SC1)));

//////////////////////////////////////////////////////////////////////
// CvtColor

GPU_PERF_TEST(CvtColor, cv::gpu::DeviceInfo, cv::Size, perf::MatType, CvtColorInfo)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    CvtColorInfo info = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_MAKETYPE(type, info.scn));

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::cvtColor(src, dst, info.code, info.dcn);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1),
                        testing::Values(
                            CvtColorInfo(4, 4, cv::COLOR_RGBA2BGRA), CvtColorInfo(4, 1, cv::COLOR_BGRA2GRAY), CvtColorInfo(1, 4, cv::COLOR_GRAY2BGRA),
                            CvtColorInfo(4, 4, cv::COLOR_BGR2XYZ), CvtColorInfo(4, 4, cv::COLOR_BGR2YCrCb), CvtColorInfo(4, 4, cv::COLOR_YCrCb2BGR),
                            CvtColorInfo(4, 4, cv::COLOR_BGR2HSV), CvtColorInfo(4, 4, cv::COLOR_HSV2BGR))));

//////////////////////////////////////////////////////////////////////
// SwapChannels

GPU_PERF_TEST(SwapChannels, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_8UC4);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);

    const int dstOrder[] = {2, 1, 0, 3};

    TEST_CYCLE()
    {
        cv::gpu::swapChannels(src, dstOrder);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, SwapChannels, testing::Combine(ALL_DEVICES, GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Threshold

GPU_PERF_TEST(Threshold, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst(size, type);

    TEST_CYCLE()
    {
        cv::gpu::threshold(src, dst, 100.0, 255.0, cv::THRESH_BINARY);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Threshold, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Resize

GPU_PERF_TEST(Resize, cv::gpu::DeviceInfo, cv::Size, perf::MatType, Interpolation, double)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    double f = GET_PARAM(4);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    declare.time(1.0);

    TEST_CYCLE()
    {
        cv::gpu::resize(src, dst, cv::Size(), f, f, interpolation);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Resize, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(perf::szSXGA, perf::sz1080p),
                        testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_32FC1),
                        testing::Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC),
                        testing::Values(0.5, 2.0)));

//////////////////////////////////////////////////////////////////////
// WarpAffine

GPU_PERF_TEST(WarpAffine, cv::gpu::DeviceInfo, cv::Size, perf::MatType, Interpolation)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    const double aplha = CV_PI / 4;
    double mat[2][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0}};
    cv::Mat M(2, 3, CV_64F, (void*) mat);

    TEST_CYCLE()
    {
        cv::gpu::warpAffine(src, dst, M, size, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpAffine, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        testing::Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC)));

//////////////////////////////////////////////////////////////////////
// WarpPerspective

GPU_PERF_TEST(WarpPerspective, cv::gpu::DeviceInfo, cv::Size, perf::MatType, Interpolation)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    const double aplha = CV_PI / 4;
    double mat[3][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0},
                         {0.0,              0.0,             1.0}};
    cv::Mat M(3, 3, CV_64F, (void*) mat);

    TEST_CYCLE()
    {
        cv::gpu::warpPerspective(src, dst, M, size, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpPerspective, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        testing::Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC)));

//////////////////////////////////////////////////////////////////////
// BuildWarpPlaneMaps

GPU_PERF_TEST(BuildWarpPlaneMaps, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::gpu::GpuMat map_x;
    cv::gpu::GpuMat map_y;

    TEST_CYCLE()
    {
        cv::gpu::buildWarpPlaneMaps(size, cv::Rect(0, 0, size.width, size.height), cv::Mat::eye(3, 3, CV_32FC1),
                                    cv::Mat::ones(3, 3, CV_32FC1), cv::Mat::zeros(1, 3, CV_32F), 1.0, map_x, map_y);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, BuildWarpPlaneMaps, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// BuildWarpCylindricalMaps

GPU_PERF_TEST(BuildWarpCylindricalMaps, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::gpu::GpuMat map_x;
    cv::gpu::GpuMat map_y;

    TEST_CYCLE()
    {
        cv::gpu::buildWarpCylindricalMaps(size, cv::Rect(0, 0, size.width, size.height), cv::Mat::eye(3, 3, CV_32FC1),
                                          cv::Mat::ones(3, 3, CV_32FC1), 1.0, map_x, map_y);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, BuildWarpCylindricalMaps, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// BuildWarpSphericalMaps

GPU_PERF_TEST(BuildWarpSphericalMaps, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::gpu::GpuMat map_x;
    cv::gpu::GpuMat map_y;

    TEST_CYCLE()
    {
        cv::gpu::buildWarpSphericalMaps(size, cv::Rect(0, 0, size.width, size.height), cv::Mat::eye(3, 3, CV_32FC1),
                                        cv::Mat::ones(3, 3, CV_32FC1), 1.0, map_x, map_y);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, BuildWarpSphericalMaps, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Rotate

GPU_PERF_TEST(Rotate, cv::gpu::DeviceInfo, cv::Size, perf::MatType, Interpolation)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::rotate(src, dst, size, 30.0, 0, 0, interpolation);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Rotate, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        testing::Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC)));

//////////////////////////////////////////////////////////////////////
// CopyMakeBorder

GPU_PERF_TEST(CopyMakeBorder, cv::gpu::DeviceInfo, cv::Size, perf::MatType, BorderMode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int borderType = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::copyMakeBorder(src, dst, 5, 5, 5, 5, borderType);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CopyMakeBorder, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                        testing::Values((int) cv::BORDER_REPLICATE, (int) cv::BORDER_REFLECT, (int) cv::BORDER_WRAP, (int) cv::BORDER_CONSTANT)));

//////////////////////////////////////////////////////////////////////
// Integral

GPU_PERF_TEST(Integral, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat buf;

    TEST_CYCLE()
    {
        cv::gpu::integralBuffered(src, dst, buf);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Integral, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// IntegralSqr

GPU_PERF_TEST(IntegralSqr, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::sqrIntegral(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, IntegralSqr, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// ColumnSum

GPU_PERF_TEST(ColumnSum, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_32FC1);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::columnSum(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ColumnSum, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// CornerHarris

GPU_PERF_TEST(CornerHarris, cv::gpu::DeviceInfo, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    int type = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::gpu::GpuMat src(img);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat Dx;
    cv::gpu::GpuMat Dy;

    int blockSize = 3;
    int ksize = 7;
    double k = 0.5;

    TEST_CYCLE()
    {
        cv::gpu::cornerHarris(src, dst, Dx, Dy, blockSize, ksize, k);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerHarris, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(CV_8UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// CornerMinEigenVal

GPU_PERF_TEST(CornerMinEigenVal, cv::gpu::DeviceInfo, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    int type = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::gpu::GpuMat src(img);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat Dx;
    cv::gpu::GpuMat Dy;

    int blockSize = 3;
    int ksize = 7;

    TEST_CYCLE()
    {
        cv::gpu::cornerMinEigenVal(src, dst, Dx, Dy, blockSize, ksize);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerMinEigenVal, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(CV_8UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// MulSpectrums

GPU_PERF_TEST(MulSpectrums, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat a_host(size, CV_32FC2);
    cv::Mat b_host(size, CV_32FC2);

    declare.in(a_host, b_host, WARMUP_RNG);

    cv::gpu::GpuMat a(a_host);
    cv::gpu::GpuMat b(b_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::mulSpectrums(a, b, dst, 0);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulSpectrums, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Dft

GPU_PERF_TEST(Dft, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_32FC2);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    declare.time(2.0);

    TEST_CYCLE()
    {
        cv::gpu::dft(src, dst, size);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Dft, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Convolve

GPU_PERF_TEST(Convolve, cv::gpu::DeviceInfo, cv::Size, int, bool)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int templ_size = GET_PARAM(2);
    bool ccorr = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::gpu::GpuMat image = cv::gpu::createContinuous(size, CV_32FC1);
    cv::gpu::GpuMat templ = cv::gpu::createContinuous(templ_size, templ_size, CV_32FC1);

    image.setTo(cv::Scalar(1.0));
    templ.setTo(cv::Scalar(1.0));

    cv::gpu::GpuMat dst;
    cv::gpu::ConvolveBuf buf;

    declare.time(2.0);

    TEST_CYCLE()
    {
        cv::gpu::convolve(image, templ, dst, ccorr, buf);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Convolve, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(3, 9, 27, 32, 64),
                        testing::Bool()));

//////////////////////////////////////////////////////////////////////
// PyrDown

GPU_PERF_TEST(PyrDown, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
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
        cv::gpu::pyrDown(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrDown, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_16SC3, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// PyrUp

GPU_PERF_TEST(PyrUp, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
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
        cv::gpu::pyrUp(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrUp, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_16SC3, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// BlendLinear

GPU_PERF_TEST(BlendLinear, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img1_host(size, type);
    cv::Mat img2_host(size, type);

    declare.in(img1_host, img2_host, WARMUP_RNG);

    cv::gpu::GpuMat img1(img1_host);
    cv::gpu::GpuMat img2(img2_host);
    cv::gpu::GpuMat weights1(size, CV_32FC1, cv::Scalar::all(0.5));
    cv::gpu::GpuMat weights2(size, CV_32FC1, cv::Scalar::all(0.5));
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::blendLinear(img1, img2, weights1, weights2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, BlendLinear, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// AlphaComp

GPU_PERF_TEST(AlphaComp, cv::gpu::DeviceInfo, cv::Size, perf::MatType, AlphaOp)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int alpha_op = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img1_host(size, type);
    cv::Mat img2_host(size, type);

    declare.in(img1_host, img2_host, WARMUP_RNG);

    cv::gpu::GpuMat img1(img1_host);
    cv::gpu::GpuMat img2(img2_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE()
    {
        cv::gpu::alphaComp(img1, img2, dst, alpha_op);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, AlphaComp, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC4, CV_16UC4, CV_32SC4, CV_32FC4),
                        testing::Values((int)cv::gpu::ALPHA_OVER, (int)cv::gpu::ALPHA_IN, (int)cv::gpu::ALPHA_OUT, (int)cv::gpu::ALPHA_ATOP, (int)cv::gpu::ALPHA_XOR, (int)cv::gpu::ALPHA_PLUS, (int)cv::gpu::ALPHA_OVER_PREMUL, (int)cv::gpu::ALPHA_IN_PREMUL, (int)cv::gpu::ALPHA_OUT_PREMUL, (int)cv::gpu::ALPHA_ATOP_PREMUL, (int)cv::gpu::ALPHA_XOR_PREMUL, (int)cv::gpu::ALPHA_PLUS_PREMUL, (int)cv::gpu::ALPHA_PREMUL)));

//////////////////////////////////////////////////////////////////////
// Canny

GPU_PERF_TEST_1(Canny, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat image_host = readImage("perf/1280x1024.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image_host.empty());

    cv::gpu::GpuMat image(image_host);
    cv::gpu::GpuMat dst;
    cv::gpu::CannyBuf buf;

    TEST_CYCLE()
    {
        cv::gpu::Canny(image, buf, dst, 50.0, 100.0);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Canny, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// CalcHist

GPU_PERF_TEST(CalcHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat hist;
    cv::gpu::GpuMat buf;

    TEST_CYCLE()
    {
        cv::gpu::calcHist(src, hist, buf);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CalcHist, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// EqualizeHist

GPU_PERF_TEST(EqualizeHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat hist;
    cv::gpu::GpuMat buf;

    TEST_CYCLE()
    {
        cv::gpu::equalizeHist(src, dst, hist, buf);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, EqualizeHist, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// ImagePyramid

GPU_PERF_TEST(ImagePyramid_build, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);

    cv::gpu::ImagePyramid pyr;

    TEST_CYCLE()
    {
        pyr.build(src, 5);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ImagePyramid_build, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

GPU_PERF_TEST(ImagePyramid_getLayer, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::ImagePyramid pyr(src, 3);

    TEST_CYCLE()
    {
        pyr.getLayer(dst, cv::Size(size.width / 2 + 10, size.height / 2 + 10));
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ImagePyramid_getLayer, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));



//////////////////////////////////////////////////////////////////////
// MulAndScaleSpectrums

GPU_PERF_TEST(MulAndScaleSpectrums, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
        
    cv::gpu::setDevice(devInfo.deviceID());

    int type = CV_32FC2;

    cv::Mat src1_host(size, type);
    cv::Mat src2_host(size, type);
    declare.in(src1_host, src2_host, WARMUP_RNG);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst(size, type);
    
    TEST_CYCLE()
    {        
        cv::gpu::mulSpectrums(src1, src2, dst, cv::DFT_ROWS, false);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulAndScaleSpectrums, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));



//////////////////////////////////////////////////////////////////////
// MulAndScaleSpectrumsScale


GPU_PERF_TEST(MulAndScaleSpectrumsScale, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);

    float scale = 1.f / size.area();
    int type = CV_32FC2;
    
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src1_host(size, type);
    cv::Mat src2_host(size, type);
    declare.in(src1_host, src2_host, WARMUP_RNG);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst(size, type);
    
    TEST_CYCLE()
    {        
        cv::gpu::mulAndScaleSpectrums(src1, src2, dst, cv::DFT_ROWS, scale, false);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulAndScaleSpectrumsScale, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));




#endif
