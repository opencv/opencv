#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Remap

GPU_PERF_TEST(Remap, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation, BorderMode)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::Mat src(size, type);
    fill(src, 0, 255);

    cv::Mat xmap(size, CV_32FC1);
    fill(xmap, 0, size.width);

    cv::Mat ymap(size, CV_32FC1);
    fill(ymap, 0, size.height);

    cv::Mat dst;

    cv::remap(src, dst, xmap, ymap, interpolation, borderMode);

    declare.time(20.0);

    TEST_CYCLE()
    {
        cv::remap(src, dst, xmap, ymap, interpolation, borderMode);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    double f = GET_PARAM(4);

    cv::Mat src(size, type);
    fill(src, 0, 255);

    cv::Mat dst;

    cv::resize(src, dst, cv::Size(), f, f, interpolation);

    declare.time(20.0);

    TEST_CYCLE()
    {
        cv::resize(src, dst, cv::Size(), f, f, interpolation);
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
    testing::Values(Scale(0.5), Scale(0.3), Scale(2.0))));

GPU_PERF_TEST(ResizeArea, cv::gpu::DeviceInfo, cv::Size, MatType, Scale)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = cv::INTER_AREA;
    double f = GET_PARAM(3);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::Mat src(src_host);
    cv::Mat dst;

    cv::resize(src, dst, cv::Size(), f, f, interpolation);

    declare.time(1.0);

    TEST_CYCLE()
    {
        cv::resize(src, dst, cv::Size(), f, f, interpolation);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ResizeArea, testing::Combine(
    ALL_DEVICES,
    testing::Values(perf::sz1080p, cv::Size(4096, 2048)),
    testing::Values(MatType(CV_8UC1)/*,  MatType(CV_8UC3), MatType(CV_8UC4),
                    MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4),
                    MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)*/),
    testing::Values(Scale(0.2),Scale(0.1),Scale(0.05))));

//////////////////////////////////////////////////////////////////////
// WarpAffine

GPU_PERF_TEST(WarpAffine, cv::gpu::DeviceInfo, cv::Size, MatType, Interpolation, BorderMode)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::Mat src(size, type);
    fill(src, 0, 255);

    cv::Mat dst;

    const double aplha = CV_PI / 4;
    double mat[2][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0}};
    cv::Mat M(2, 3, CV_64F, (void*) mat);

    cv::warpAffine(src, dst, M, size, interpolation, borderMode);

    declare.time(20.0);

    TEST_CYCLE()
    {
        cv::warpAffine(src, dst, M, size, interpolation, borderMode);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::Mat src(size, type);
    fill(src, 0, 255);

    cv::Mat dst;

    const double aplha = CV_PI / 4;
    double mat[3][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0},
                         {0.0,              0.0,             1.0}};
    cv::Mat M(3, 3, CV_64F, (void*) mat);

    cv::warpPerspective(src, dst, M, size, interpolation, borderMode);

    declare.time(20.0);

    TEST_CYCLE()
    {
        cv::warpPerspective(src, dst, M, size, interpolation, borderMode);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int borderType = GET_PARAM(3);

    cv::Mat src(size, type);
    fill(src, 0, 255);

    cv::Mat dst;

    cv::copyMakeBorder(src, dst, 5, 5, 5, 5, borderType);

    TEST_CYCLE()
    {
        cv::copyMakeBorder(src, dst, 5, 5, 5, 5, borderType);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    int threshOp = GET_PARAM(3);

    cv::Mat src(size, depth);
    fill(src, 0, 255);

    cv::Mat dst;

    cv::threshold(src, dst, 100.0, 255.0, threshOp);

    TEST_CYCLE()
    {
        cv::threshold(src, dst, 100.0, 255.0, threshOp);
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
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);
    fill(src, 0, 255);

    cv::Mat dst;

    cv::integral(src, dst);

    TEST_CYCLE()
    {
        cv::integral(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Integral, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// HistEven_OneChannel

GPU_PERF_TEST(HistEven_OneChannel, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 0, 255);

    int hbins = 30;
    float hranges[] = {0.0f, 180.0f};
    cv::Mat hist;
    int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    int channels[] = {0};

    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

    TEST_CYCLE()
    {
        cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, HistEven_OneChannel, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S))));

//////////////////////////////////////////////////////////////////////
// EqualizeHist

GPU_PERF_TEST(EqualizeHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);
    fill(src, 0, 255);

    cv::Mat dst;

    cv::equalizeHist(src, dst);

    TEST_CYCLE()
    {
        cv::equalizeHist(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, EqualizeHist, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Canny

IMPLEMENT_PARAM_CLASS(AppertureSize, int)
IMPLEMENT_PARAM_CLASS(L2gradient, bool)

GPU_PERF_TEST(Canny, cv::gpu::DeviceInfo, AppertureSize, L2gradient)
{
    int apperture_size = GET_PARAM(1);
    bool useL2gradient = GET_PARAM(2);

    cv::Mat image = readImage("perf/1280x1024.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat dst;

    cv::Canny(image, dst, 50.0, 100.0, apperture_size, useL2gradient);

    TEST_CYCLE()
    {
        cv::Canny(image, dst, 50.0, 100.0, apperture_size, useL2gradient);
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
    cv::Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());

    cv::Mat dst;

    cv::pyrMeanShiftFiltering(img, dst, 50, 50);

    declare.time(15.0);

    TEST_CYCLE()
    {
        cv::pyrMeanShiftFiltering(img, dst, 50, 50);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftFiltering, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// Convolve

IMPLEMENT_PARAM_CLASS(KSize, int)
IMPLEMENT_PARAM_CLASS(Ccorr, bool)

GPU_PERF_TEST(Convolve, cv::gpu::DeviceInfo, cv::Size, KSize, Ccorr)
{
    cv::Size size = GET_PARAM(1);
    int templ_size = GET_PARAM(2);
    bool ccorr = GET_PARAM(3);

    ASSERT_FALSE(ccorr);

    cv::Mat image(size, CV_32FC1);
    image.setTo(1.0);

    cv::Mat templ(templ_size, templ_size, CV_32FC1);
    templ.setTo(1.0);

    cv::Mat dst;

    cv::filter2D(image, dst, image.depth(), templ);

    declare.time(10.0);

    TEST_CYCLE()
    {
        cv::filter2D(image, dst, image.depth(), templ);
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
    cv::Size size = GET_PARAM(1);
    cv::Size templ_size = GET_PARAM(2);
    int cn = GET_PARAM(3);
    int method = GET_PARAM(4);

    cv::Mat image(size, CV_MAKE_TYPE(CV_8U, cn));
    fill(image, 0, 255);

    cv::Mat templ(templ_size, CV_MAKE_TYPE(CV_8U, cn));
    fill(templ, 0, 255);

    cv::Mat dst;

    cv::matchTemplate(image, templ, dst, method);

    TEST_CYCLE()
    {
        cv::matchTemplate(image, templ, dst, method);
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
    cv::Size size = GET_PARAM(1);
    cv::Size templ_size = GET_PARAM(2);
    int cn = GET_PARAM(3);
    int method = GET_PARAM(4);

    cv::Mat image(size, CV_MAKE_TYPE(CV_32F, cn));
    fill(image, 0, 255);

    cv::Mat templ(templ_size, CV_MAKE_TYPE(CV_32F, cn));
    fill(templ, 0, 255);

    cv::Mat dst;

    cv::matchTemplate(image, templ, dst, method);

    TEST_CYCLE()
    {
        cv::matchTemplate(image, templ, dst, method);
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
    cv::Size size = GET_PARAM(1);
    int flag = GET_PARAM(2);

    cv::Mat a(size, CV_32FC2);
    fill(a, 0, 100);

    cv::Mat b(size, CV_32FC2);
    fill(b, 0, 100);

    cv::Mat dst;

    cv::mulSpectrums(a, b, dst, flag);

    TEST_CYCLE()
    {
        cv::mulSpectrums(a, b, dst, flag);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulSpectrums, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(DftFlags(0), DftFlags(cv::DFT_ROWS))));

//////////////////////////////////////////////////////////////////////
// Dft

GPU_PERF_TEST(Dft, cv::gpu::DeviceInfo, cv::Size, DftFlags)
{
    cv::Size size = GET_PARAM(1);
    int flag = GET_PARAM(2);

    cv::Mat src(size, CV_32FC2);
    fill(src, 0, 100);

    cv::Mat dst;

    cv::dft(src, dst, flag);

    declare.time(10.0);

    TEST_CYCLE()
    {
        cv::dft(src, dst, flag);
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
    int type = GET_PARAM(1);
    int borderType = GET_PARAM(2);
    int blockSize = GET_PARAM(3);
    int apertureSize = GET_PARAM(4);

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::Mat dst;

    double k = 0.5;

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, borderType);

    TEST_CYCLE()
    {
        cv::cornerHarris(img, dst, blockSize, apertureSize, k, borderType);
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
    int type = GET_PARAM(1);
    int borderType = GET_PARAM(2);
    int blockSize = GET_PARAM(3);
    int apertureSize = GET_PARAM(4);

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::Mat dst;

    cv::cornerMinEigenVal(img, dst, blockSize, apertureSize, borderType);

    TEST_CYCLE()
    {
        cv::cornerMinEigenVal(img, dst, blockSize, apertureSize, borderType);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerMinEigenVal, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    testing::Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_REFLECT)),
    testing::Values(BlockSize(3), BlockSize(5), BlockSize(7)),
    testing::Values(ApertureSize(0), ApertureSize(3), ApertureSize(5), ApertureSize(7))));

//////////////////////////////////////////////////////////////////////
// PyrDown

GPU_PERF_TEST(PyrDown, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);
    fill(src, 0, 255);

    cv::Mat dst;

    cv::pyrDown(src, dst);

    TEST_CYCLE()
    {
        cv::pyrDown(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);
    fill(src, 0, 255);

    cv::Mat dst;

    cv::pyrUp(src, dst);

    TEST_CYCLE()
    {
        cv::pyrUp(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);
    CvtColorInfo info = GET_PARAM(3);

    cv::Mat src(size, CV_MAKETYPE(depth, info.scn));
    fill(src, 0, 255);

    cv::Mat dst;

    cv::cvtColor(src, dst, info.code, info.dcn);

    TEST_CYCLE()
    {
        cv::cvtColor(src, dst, info.code, info.dcn);
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

#endif
