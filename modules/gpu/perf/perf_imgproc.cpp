#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Remap

GPU_PERF_TEST(Remap, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation, BorderMode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::Mat xmap_host(size, CV_32FC1);
    fill(xmap_host, 0, size.width);

    cv::Mat ymap_host(size, CV_32FC1);
    fill(ymap_host, 0, size.height);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat xmap(xmap_host);
    cv::gpu::GpuMat ymap(ymap_host);
    cv::gpu::GpuMat dst;

    cv::gpu::remap(src, dst, xmap, ymap, interpolation, borderMode);

    declare.time(3.0);

    TEST_CYCLE()
    {
        cv::gpu::remap(src, dst, xmap, ymap, interpolation, borderMode);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Remap, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
    testing::Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_CONSTANT), BorderMode(cv::BORDER_REFLECT), BorderMode(cv::BORDER_WRAP))));


//////////////////////////////////////////////////////////////////////
// Resize

IMPLEMENT_PARAM_CLASS(Scale, double)

GPU_PERF_TEST(Resize, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation, Scale)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    double f = GET_PARAM(4);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::resize(src, dst, cv::Size(), f, f, interpolation);

    declare.time(1.0);

    TEST_CYCLE()
    {
        cv::gpu::resize(src, dst, cv::Size(), f, f, interpolation);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Resize, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR),
                    Interpolation(cv::INTER_CUBIC),   Interpolation(cv::INTER_AREA)),
    testing::Values(Scale(0.5), Scale(0.3)/*, Scale(2.0)*/)));

//////////////////////////////////////////////////////////////////////
// WarpAffine

GPU_PERF_TEST(WarpAffine, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation, BorderMode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    const double aplha = CV_PI / 4;
    double mat[2][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0}};
    cv::Mat M(2, 3, CV_64F, (void*) mat);

    cv::gpu::warpAffine(src, dst, M, size, interpolation, borderMode);

    TEST_CYCLE()
    {
        cv::gpu::warpAffine(src, dst, M, size, interpolation, borderMode);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpAffine, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
    testing::Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_CONSTANT), BorderMode(cv::BORDER_REFLECT), BorderMode(cv::BORDER_WRAP))));

//////////////////////////////////////////////////////////////////////
// WarpPerspective

GPU_PERF_TEST(WarpPerspective, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation, BorderMode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    const double aplha = CV_PI / 4;
    double mat[3][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0},
                         {0.0,              0.0,             1.0}};
    cv::Mat M(3, 3, CV_64F, (void*) mat);

    cv::gpu::warpPerspective(src, dst, M, size, interpolation, borderMode);

    TEST_CYCLE()
    {
        cv::gpu::warpPerspective(src, dst, M, size, interpolation, borderMode);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpPerspective, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
    testing::Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_CONSTANT), BorderMode(cv::BORDER_REFLECT), BorderMode(cv::BORDER_WRAP))));

//////////////////////////////////////////////////////////////////////
// CopyMakeBorder

GPU_PERF_TEST(CopyMakeBorder, cv::gpu::DeviceInfo, cv::Size, MatType, BorderMode)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int borderType = GET_PARAM(3);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::copyMakeBorder(src, dst, 5, 5, 5, 5, borderType);

    TEST_CYCLE()
    {
        cv::gpu::copyMakeBorder(src, dst, 5, 5, 5, 5, borderType);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CopyMakeBorder, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_CONSTANT), BorderMode(cv::BORDER_REFLECT), BorderMode(cv::BORDER_WRAP))));

//////////////////////////////////////////////////////////////////////
// Threshold

CV_ENUM(ThreshOp, cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV)
#define ALL_THRESH_OPS testing::Values(ThreshOp(cv::THRESH_BINARY), ThreshOp(cv::THRESH_BINARY_INV), ThreshOp(cv::THRESH_TRUNC), ThreshOp(cv::THRESH_TOZERO), ThreshOp(cv::THRESH_TOZERO_INV))

GPU_PERF_TEST(Threshold, cv::gpu::DeviceInfo, cv::Size, MatDepth, ThreshOp)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int threshOp = GET_PARAM(3);

    cv::Mat src_host(size, depth);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::threshold(src, dst, 100.0, 255.0, threshOp);

    TEST_CYCLE()
    {
        cv::gpu::threshold(src, dst, 100.0, 255.0, threshOp);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Threshold, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32F), MatDepth(CV_64F)),
    ALL_THRESH_OPS));

//////////////////////////////////////////////////////////////////////
// Integral

GPU_PERF_TEST(Integral, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_8UC1);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat buf;

    cv::gpu::integralBuffered(src, dst, buf);

    TEST_CYCLE()
    {
        cv::gpu::integralBuffered(src, dst, buf);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Integral, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Integral_Sqr

GPU_PERF_TEST(Integral_Sqr, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_8UC1);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::sqrIntegral(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::sqrIntegral(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Integral_Sqr, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// HistEven_OneChannel

GPU_PERF_TEST(HistEven_OneChannel, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat hist;
    cv::gpu::GpuMat buf;

    cv::gpu::histEven(src, hist, buf, 30, 0, 180);

    TEST_CYCLE()
    {
        cv::gpu::histEven(src, hist, buf, 30, 0, 180);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, HistEven_OneChannel, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S))));

//////////////////////////////////////////////////////////////////////
// HistEven_FourChannel

GPU_PERF_TEST(HistEven_FourChannel, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, CV_MAKE_TYPE(depth, 4));
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat hist[4];
    cv::gpu::GpuMat buf;
    int histSize[] = {30, 30, 30, 30};
    int lowerLevel[] = {0, 0, 0, 0};
    int upperLevel[] = {180, 180, 180, 180};

    cv::gpu::histEven(src, hist, buf, histSize, lowerLevel, upperLevel);

    TEST_CYCLE()
    {
        cv::gpu::histEven(src, hist, buf, histSize, lowerLevel, upperLevel);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, HistEven_FourChannel, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S))));

//////////////////////////////////////////////////////////////////////
// CalcHist

GPU_PERF_TEST(CalcHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_8UC1);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat hist;
    cv::gpu::GpuMat buf;

    cv::gpu::calcHist(src, hist, buf);

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
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_8UC1);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat hist;
    cv::gpu::GpuMat buf;

    cv::gpu::equalizeHist(src, dst, hist, buf);

    TEST_CYCLE()
    {
        cv::gpu::equalizeHist(src, dst, hist, buf);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, EqualizeHist, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// ColumnSum

GPU_PERF_TEST(ColumnSum, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_32FC1);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::columnSum(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::columnSum(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ColumnSum, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Canny

IMPLEMENT_PARAM_CLASS(AppertureSize, int)
IMPLEMENT_PARAM_CLASS(L2gradient, bool)

GPU_PERF_TEST(Canny, cv::gpu::DeviceInfo, AppertureSize, L2gradient)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int apperture_size = GET_PARAM(1);
    bool useL2gradient = GET_PARAM(2);

    cv::Mat image_host = readImage("perf/1280x1024.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image_host.empty());

    cv::gpu::GpuMat image(image_host);
    cv::gpu::GpuMat dst;
    cv::gpu::CannyBuf buf;

    cv::gpu::Canny(image, buf, dst, 50.0, 100.0, apperture_size, useL2gradient);

    TEST_CYCLE()
    {
        cv::gpu::Canny(image, buf, dst, 50.0, 100.0, apperture_size, useL2gradient);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Canny, testing::Combine(
    ALL_DEVICES,
    testing::Values(AppertureSize(3), AppertureSize(5)),
    testing::Values(L2gradient(false), L2gradient(true))));

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

    cv::gpu::meanShiftFiltering(src, dst, 50, 50);

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

    cv::gpu::meanShiftProc(src, dstr, dstsp, 50, 50);

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

    meanShiftSegmentation(src, dst, 10, 10, 20);

    declare.time(5.0);

    TEST_CYCLE()
    {
        meanShiftSegmentation(src, dst, 10, 10, 20);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftSegmentation, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// BlendLinear

GPU_PERF_TEST(BlendLinear, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat img1_host(size, type);
    fill(img1_host, 0, 255);

    cv::Mat img2_host(size, type);
    fill(img2_host, 0, 255);

    cv::gpu::GpuMat img1(img1_host);
    cv::gpu::GpuMat img2(img2_host);
    cv::gpu::GpuMat weights1(size, CV_32FC1, cv::Scalar::all(0.5));
    cv::gpu::GpuMat weights2(size, CV_32FC1, cv::Scalar::all(0.5));
    cv::gpu::GpuMat dst;

    cv::gpu::blendLinear(img1, img2, weights1, weights2, dst);

    TEST_CYCLE()
    {
        cv::gpu::blendLinear(img1, img2, weights1, weights2, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, BlendLinear, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4))));

//////////////////////////////////////////////////////////////////////
// Convolve

IMPLEMENT_PARAM_CLASS(KSize, int)
IMPLEMENT_PARAM_CLASS(Ccorr, bool)

GPU_PERF_TEST(Convolve, cv::gpu::DeviceInfo, cv::Size, KSize, Ccorr)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int templ_size = GET_PARAM(2);
    bool ccorr = GET_PARAM(3);

    cv::gpu::GpuMat image = cv::gpu::createContinuous(size, CV_32FC1);
    image.setTo(cv::Scalar(1.0));

    cv::gpu::GpuMat templ = cv::gpu::createContinuous(templ_size, templ_size, CV_32FC1);
    templ.setTo(cv::Scalar(1.0));

    cv::gpu::GpuMat dst;
    cv::gpu::ConvolveBuf buf;

    cv::gpu::convolve(image, templ, dst, ccorr, buf);

    declare.time(2.0);

    TEST_CYCLE()
    {
        cv::gpu::convolve(image, templ, dst, ccorr, buf);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Convolve, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(KSize(3), KSize(9), KSize(17), KSize(27), KSize(32), KSize(64)),
    testing::Values(Ccorr(false), Ccorr(true))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_8U

CV_ENUM(TemplateMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)
#define ALL_TEMPLATE_METHODS testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_SQDIFF_NORMED), TemplateMethod(cv::TM_CCORR), TemplateMethod(cv::TM_CCORR_NORMED), TemplateMethod(cv::TM_CCOEFF), TemplateMethod(cv::TM_CCOEFF_NORMED))

IMPLEMENT_PARAM_CLASS(TemplateSize, cv::Size)

GPU_PERF_TEST(MatchTemplate_8U, cv::gpu::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    cv::Size templ_size = GET_PARAM(2);
    int cn = GET_PARAM(3);
    int method = GET_PARAM(4);

    cv::Mat image_host(size, CV_MAKE_TYPE(CV_8U, cn));
    fill(image_host, 0, 255);

    cv::Mat templ_host(templ_size, CV_MAKE_TYPE(CV_8U, cn));
    fill(templ_host, 0, 255);

    cv::gpu::GpuMat image(image_host);
    cv::gpu::GpuMat templ(templ_host);
    cv::gpu::GpuMat dst;

    cv::gpu::matchTemplate(image, templ, dst, method);

    TEST_CYCLE()
    {
        cv::gpu::matchTemplate(image, templ, dst, method);
    }
};

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate_8U, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    ALL_TEMPLATE_METHODS));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_32F

GPU_PERF_TEST(MatchTemplate_32F, cv::gpu::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    cv::Size templ_size = GET_PARAM(2);
    int cn = GET_PARAM(3);
    int method = GET_PARAM(4);

    cv::Mat image_host(size, CV_MAKE_TYPE(CV_32F, cn));
    fill(image_host, 0, 255);

    cv::Mat templ_host(templ_size, CV_MAKE_TYPE(CV_32F, cn));
    fill(templ_host, 0, 255);

    cv::gpu::GpuMat image(image_host);
    cv::gpu::GpuMat templ(templ_host);
    cv::gpu::GpuMat dst;

    cv::gpu::matchTemplate(image, templ, dst, method);

    TEST_CYCLE()
    {
        cv::gpu::matchTemplate(image, templ, dst, method);
    }
};

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate_32F, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))));

//////////////////////////////////////////////////////////////////////
// MulSpectrums

CV_FLAGS(DftFlags, 0, cv::DFT_INVERSE, cv::DFT_SCALE, cv::DFT_ROWS, cv::DFT_COMPLEX_OUTPUT, cv::DFT_REAL_OUTPUT)

GPU_PERF_TEST(MulSpectrums, cv::gpu::DeviceInfo, cv::Size, DftFlags)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int flag = GET_PARAM(2);

    cv::Mat a_host(size, CV_32FC2);
    fill(a_host, 0, 100);

    cv::Mat b_host(size, CV_32FC2);
    fill(b_host, 0, 100);

    cv::gpu::GpuMat a(a_host);
    cv::gpu::GpuMat b(b_host);
    cv::gpu::GpuMat dst;

    cv::gpu::mulSpectrums(a, b, dst, flag);

    TEST_CYCLE()
    {
        cv::gpu::mulSpectrums(a, b, dst, flag);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulSpectrums, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(DftFlags(0), DftFlags(cv::DFT_ROWS))));

//////////////////////////////////////////////////////////////////////
// MulAndScaleSpectrums

GPU_PERF_TEST(MulAndScaleSpectrums, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    float scale = 1.f / size.area();

    cv::Mat src1_host(size, CV_32FC2);
    fill(src1_host, 0, 100);

    cv::Mat src2_host(size, CV_32FC2);
    fill(src2_host, 0, 100);

    cv::gpu::GpuMat src1(src1_host);
    cv::gpu::GpuMat src2(src2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::mulAndScaleSpectrums(src1, src2, dst, cv::DFT_ROWS, scale, false);

    TEST_CYCLE()
    {
        cv::gpu::mulAndScaleSpectrums(src1, src2, dst, cv::DFT_ROWS, scale, false);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulAndScaleSpectrums, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Dft

GPU_PERF_TEST(Dft, cv::gpu::DeviceInfo, cv::Size, DftFlags)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int flag = GET_PARAM(2);

    cv::Mat src_host(size, CV_32FC2);
    fill(src_host, 0, 100);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::dft(src, dst, size, flag);

    declare.time(2.0);

    TEST_CYCLE()
    {
        cv::gpu::dft(src, dst, size, flag);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Dft, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(DftFlags(0), DftFlags(cv::DFT_ROWS), DftFlags(cv::DFT_INVERSE))));

//////////////////////////////////////////////////////////////////////
// CornerHarris

IMPLEMENT_PARAM_CLASS(BlockSize, int)
IMPLEMENT_PARAM_CLASS(ApertureSize, int)

GPU_PERF_TEST(CornerHarris, cv::gpu::DeviceInfo, MatType, BorderMode, BlockSize, ApertureSize)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int type = GET_PARAM(1);
    int borderType = GET_PARAM(2);
    int blockSize = GET_PARAM(3);
    int apertureSize = GET_PARAM(4);

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::gpu::GpuMat src(img);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat Dx;
    cv::gpu::GpuMat Dy;
    cv::gpu::GpuMat buf;

    double k = 0.5;

    cv::gpu::cornerHarris(src, dst, Dx, Dy, buf, blockSize, apertureSize, k, borderType);

    TEST_CYCLE()
    {
        cv::gpu::cornerHarris(src, dst, Dx, Dy, buf, blockSize, apertureSize, k, borderType);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerHarris, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    testing::Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_REFLECT)),
    testing::Values(BlockSize(3), BlockSize(5), BlockSize(7)),
    testing::Values(ApertureSize(0), ApertureSize(3), ApertureSize(5), ApertureSize(7))));

//////////////////////////////////////////////////////////////////////
// CornerMinEigenVal

GPU_PERF_TEST(CornerMinEigenVal, cv::gpu::DeviceInfo, MatType, BorderMode, BlockSize, ApertureSize)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int type = GET_PARAM(1);
    int borderType = GET_PARAM(2);
    int blockSize = GET_PARAM(3);
    int apertureSize = GET_PARAM(4);

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::gpu::GpuMat src(img);
    cv::gpu::GpuMat dst;
    cv::gpu::GpuMat Dx;
    cv::gpu::GpuMat Dy;
    cv::gpu::GpuMat buf;

    cv::gpu::cornerMinEigenVal(src, dst, Dx, Dy, buf, blockSize, apertureSize, borderType);

    TEST_CYCLE()
    {
        cv::gpu::cornerMinEigenVal(src, dst, Dx, Dy, buf, blockSize, apertureSize, borderType);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerMinEigenVal, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    testing::Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_REFLECT)),
    testing::Values(BlockSize(3), BlockSize(5), BlockSize(7)),
    testing::Values(ApertureSize(0), ApertureSize(3), ApertureSize(5), ApertureSize(7))));

//////////////////////////////////////////////////////////////////////
// BuildWarpPlaneMaps

GPU_PERF_TEST(BuildWarpPlaneMaps, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat R = cv::Mat::ones(3, 3, CV_32FC1);
    cv::Mat T = cv::Mat::zeros(1, 3, CV_32F);
    cv::gpu::GpuMat map_x;
    cv::gpu::GpuMat map_y;

    cv::gpu::buildWarpPlaneMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, T, 1.0, map_x, map_y);

    TEST_CYCLE()
    {
        cv::gpu::buildWarpPlaneMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, T, 1.0, map_x, map_y);
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
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat R = cv::Mat::ones(3, 3, CV_32FC1);
    cv::gpu::GpuMat map_x;
    cv::gpu::GpuMat map_y;

    cv::gpu::buildWarpCylindricalMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, 1.0, map_x, map_y);

    TEST_CYCLE()
    {
        cv::gpu::buildWarpCylindricalMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, 1.0, map_x, map_y);
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
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat R = cv::Mat::ones(3, 3, CV_32FC1);
    cv::gpu::GpuMat map_x;
    cv::gpu::GpuMat map_y;

    cv::gpu::buildWarpSphericalMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, 1.0, map_x, map_y);

    TEST_CYCLE()
    {
        cv::gpu::buildWarpSphericalMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, 1.0, map_x, map_y);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, BuildWarpSphericalMaps, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Rotate

GPU_PERF_TEST(Rotate, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::rotate(src, dst, size, 30.0, 0, 0, interpolation);

    TEST_CYCLE()
    {
        cv::gpu::rotate(src, dst, size, 30.0, 0, 0, interpolation);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Rotate, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC))));

//////////////////////////////////////////////////////////////////////
// PyrDown

GPU_PERF_TEST(PyrDown, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::pyrDown(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::pyrDown(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrDown, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4))));

//////////////////////////////////////////////////////////////////////
// PyrUp

GPU_PERF_TEST(PyrUp, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::pyrUp(src, dst);

    TEST_CYCLE()
    {
        cv::gpu::pyrUp(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrUp, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4))));

//////////////////////////////////////////////////////////////////////
// CvtColor

GPU_PERF_TEST(CvtColor, cv::gpu::DeviceInfo, cv::Size, MatDepth, CvtColorInfo)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    CvtColorInfo info = GET_PARAM(3);

    cv::Mat src_host(size, CV_MAKETYPE(depth, info.scn));
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::cvtColor(src, dst, info.code, info.dcn);

    TEST_CYCLE()
    {
        cv::gpu::cvtColor(src, dst, info.code, info.dcn);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32F)),
    testing::Values(CvtColorInfo(4, 4, cv::COLOR_RGBA2BGRA),
                    CvtColorInfo(4, 1, cv::COLOR_BGRA2GRAY),
                    CvtColorInfo(1, 4, cv::COLOR_GRAY2BGRA),
                    CvtColorInfo(3, 3, cv::COLOR_BGR2XYZ),
                    CvtColorInfo(3, 3, cv::COLOR_XYZ2BGR),
                    CvtColorInfo(3, 3, cv::COLOR_BGR2YCrCb),
                    CvtColorInfo(3, 3, cv::COLOR_YCrCb2BGR),
                    CvtColorInfo(3, 3, cv::COLOR_BGR2YUV),
                    CvtColorInfo(3, 3, cv::COLOR_YUV2BGR),
                    CvtColorInfo(3, 3, cv::COLOR_BGR2HSV),
                    CvtColorInfo(3, 3, cv::COLOR_HSV2BGR),
                    CvtColorInfo(3, 3, cv::COLOR_BGR2HLS),
                    CvtColorInfo(3, 3, cv::COLOR_HLS2BGR))));

//////////////////////////////////////////////////////////////////////
// SwapChannels

GPU_PERF_TEST(SwapChannels, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);

    cv::Mat src_host(size, CV_8UC4);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);

    const int dstOrder[] = {2, 1, 0, 3};

    cv::gpu::swapChannels(src, dstOrder);

    TEST_CYCLE()
    {
        cv::gpu::swapChannels(src, dstOrder);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, SwapChannels, testing::Combine(ALL_DEVICES, GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// AlphaComp

CV_ENUM(AlphaOp, cv::gpu::ALPHA_OVER, cv::gpu::ALPHA_IN, cv::gpu::ALPHA_OUT, cv::gpu::ALPHA_ATOP, cv::gpu::ALPHA_XOR, cv::gpu::ALPHA_PLUS, cv::gpu::ALPHA_OVER_PREMUL, cv::gpu::ALPHA_IN_PREMUL, cv::gpu::ALPHA_OUT_PREMUL, cv::gpu::ALPHA_ATOP_PREMUL, cv::gpu::ALPHA_XOR_PREMUL, cv::gpu::ALPHA_PLUS_PREMUL, cv::gpu::ALPHA_PREMUL)

GPU_PERF_TEST(AlphaComp, cv::gpu::DeviceInfo, cv::Size, MatType, AlphaOp)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int alpha_op = GET_PARAM(3);

    cv::Mat img1_host(size, type);
    fill(img1_host, 0, 255);

    cv::Mat img2_host(size, type);
    fill(img2_host, 0, 255);

    cv::gpu::GpuMat img1(img1_host);
    cv::gpu::GpuMat img2(img2_host);
    cv::gpu::GpuMat dst;

    cv::gpu::alphaComp(img1, img2, dst, alpha_op);

    TEST_CYCLE()
    {
        cv::gpu::alphaComp(img1, img2, dst, alpha_op);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, AlphaComp, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC4), MatType(CV_16UC4), MatType(CV_32SC4), MatType(CV_32FC4)),
    testing::Values(AlphaOp(cv::gpu::ALPHA_OVER),
                    AlphaOp(cv::gpu::ALPHA_IN),
                    AlphaOp(cv::gpu::ALPHA_OUT),
                    AlphaOp(cv::gpu::ALPHA_ATOP),
                    AlphaOp(cv::gpu::ALPHA_XOR),
                    AlphaOp(cv::gpu::ALPHA_PLUS),
                    AlphaOp(cv::gpu::ALPHA_OVER_PREMUL),
                    AlphaOp(cv::gpu::ALPHA_IN_PREMUL),
                    AlphaOp(cv::gpu::ALPHA_OUT_PREMUL),
                    AlphaOp(cv::gpu::ALPHA_ATOP_PREMUL),
                    AlphaOp(cv::gpu::ALPHA_XOR_PREMUL),
                    AlphaOp(cv::gpu::ALPHA_PLUS_PREMUL),
                    AlphaOp(cv::gpu::ALPHA_PREMUL))));

//////////////////////////////////////////////////////////////////////
// ImagePyramid

GPU_PERF_TEST(ImagePyramid_build, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);

    cv::gpu::ImagePyramid pyr;

    pyr.build(src, 5);

    TEST_CYCLE()
    {
        pyr.build(src, 5);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ImagePyramid_build, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4))));

GPU_PERF_TEST(ImagePyramid_getLayer, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::ImagePyramid pyr(src, 3);

    pyr.getLayer(dst, cv::Size(size.width / 2 + 10, size.height / 2 + 10));

    TEST_CYCLE()
    {
        pyr.getLayer(dst, cv::Size(size.width / 2 + 10, size.height / 2 + 10));
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ImagePyramid_getLayer, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4))));

#endif
