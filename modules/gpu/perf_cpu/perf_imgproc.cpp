#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Remap

GPU_PERF_TEST(Remap, cv::gpu::DeviceInfo, cv::Size, perf::MatType, Interpolation, BorderMode)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    int borderMode = GET_PARAM(4);

    cv::Mat src(size, type);
    cv::Mat xmap(size, CV_32FC1);
    cv::Mat ymap(size, CV_32FC1);

    declare.in(src, xmap, ymap, WARMUP_RNG);

    cv::Mat dst;

    declare.time(10.0);

    TEST_CYCLE()
    {
        cv::remap(src, dst, xmap, ymap, interpolation, borderMode);
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
    cv::Mat img = readImage("gpu/meanshift/cones.png");
    ASSERT_FALSE(img.empty());

    cv::Mat dst;

    declare.time(100.0);

    TEST_CYCLE()
    {
        cv::pyrMeanShiftFiltering(img, dst, 50, 50);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftFiltering, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// ReprojectImageTo3D

GPU_PERF_TEST(ReprojectImageTo3D, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::reprojectImageTo3D(src, dst, cv::Mat::ones(4, 4, CV_32FC1));
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    CvtColorInfo info = GET_PARAM(3);

    cv::Mat src(size, CV_MAKETYPE(type, info.scn));

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::cvtColor(src, dst, info.code, info.dcn);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1),
                        testing::Values(
                            CvtColorInfo(4, 4, cv::COLOR_RGBA2BGRA), CvtColorInfo(4, 1, cv::COLOR_BGRA2GRAY), CvtColorInfo(1, 4, cv::COLOR_GRAY2BGRA),
                            CvtColorInfo(3, 3, cv::COLOR_BGR2XYZ), CvtColorInfo(3, 3, cv::COLOR_BGR2YCrCb), CvtColorInfo(3, 3, cv::COLOR_YCrCb2BGR),
                            CvtColorInfo(3, 3, cv::COLOR_BGR2HSV), CvtColorInfo(3, 3, cv::COLOR_HSV2BGR))));

//////////////////////////////////////////////////////////////////////
// Threshold

GPU_PERF_TEST(Threshold, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst(size, type);

    TEST_CYCLE()
    {
        cv::threshold(src, dst, 100.0, 255.0, cv::THRESH_BINARY);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Threshold, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Resize

GPU_PERF_TEST(Resize, cv::gpu::DeviceInfo, cv::Size, perf::MatType, Interpolation, double)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);
    double f = GET_PARAM(4);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    declare.time(1.0);

    TEST_CYCLE()
    {
        cv::resize(src, dst, cv::Size(), f, f, interpolation);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    const double aplha = CV_PI / 4;
    double mat[2][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0}};
    cv::Mat M(2, 3, CV_64F, (void*) mat);

    TEST_CYCLE()
    {
        cv::warpAffine(src, dst, M, size, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int interpolation = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    const double aplha = CV_PI / 4;
    double mat[3][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0},
                         {0.0,              0.0,             1.0}};
    cv::Mat M(3, 3, CV_64F, (void*) mat);

    TEST_CYCLE()
    {
        cv::warpPerspective(src, dst, M, size, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpPerspective, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        testing::Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC)));

//////////////////////////////////////////////////////////////////////
// CopyMakeBorder

GPU_PERF_TEST(CopyMakeBorder, cv::gpu::DeviceInfo, cv::Size, perf::MatType, BorderMode)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int borderType = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::copyMakeBorder(src, dst, 5, 5, 5, 5, borderType);
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
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::integral(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Integral, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// CornerHarris

GPU_PERF_TEST(CornerHarris, cv::gpu::DeviceInfo, perf::MatType)
{
    int type = GET_PARAM(1);

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::Mat dst;

    int blockSize = 3;
    int ksize = 7;
    double k = 0.5;

    TEST_CYCLE()
    {
        cv::cornerHarris(img, dst, blockSize, ksize, k);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerHarris, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(CV_8UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// CornerMinEigenVal

GPU_PERF_TEST(CornerMinEigenVal, cv::gpu::DeviceInfo, perf::MatType)
{
    int type = GET_PARAM(1);

    cv::Mat img = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    cv::Mat dst;

    int blockSize = 3;
    int ksize = 7;

    TEST_CYCLE()
    {
        cv::cornerMinEigenVal(img, dst, blockSize, ksize);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerMinEigenVal, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(CV_8UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// MulSpectrums

GPU_PERF_TEST(MulSpectrums, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat a(size, CV_32FC2);
    cv::Mat b(size, CV_32FC2);

    declare.in(a, b, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::mulSpectrums(a, b, dst, 0);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulSpectrums, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Dft

GPU_PERF_TEST(Dft, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_32FC2);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    declare.time(2.0);

    TEST_CYCLE()
    {
        cv::dft(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Dft, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// Convolve

GPU_PERF_TEST(Convolve, cv::gpu::DeviceInfo, cv::Size, int, bool)
{
    cv::Size size = GET_PARAM(1);
    int templ_size = GET_PARAM(2);

    cv::Mat image(size, CV_32FC1);
    cv::Mat templ(templ_size, templ_size, CV_32FC1);

    image.setTo(cv::Scalar(1.0));
    templ.setTo(cv::Scalar(1.0));

    cv::Mat dst;

    declare.time(2.0);

    TEST_CYCLE()
    {
        cv::filter2D(image, dst, image.depth(), templ);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::pyrDown(src, dst);
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
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::pyrUp(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrUp, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_16SC3, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Canny

GPU_PERF_TEST_1(Canny, cv::gpu::DeviceInfo)
{
    cv::Mat image = readImage("perf/1280x1024.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::Canny(image, dst, 50.0, 100.0);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, Canny, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// CalcHist

GPU_PERF_TEST(CalcHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);

    declare.in(src, WARMUP_RNG);

    cv::Mat hist;

    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    TEST_CYCLE()
    {
        cv::calcHist(&src, 1, 0, cv::noArray(), hist, 1, &histSize, &histRange);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, CalcHist, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));

//////////////////////////////////////////////////////////////////////
// EqualizeHist

GPU_PERF_TEST(EqualizeHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::equalizeHist(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, EqualizeHist, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));


//////////////////////////////////////////////////////////////////////
// MulAndScaleSpectrums


GPU_PERF_TEST(MulAndScaleSpectrums, cv::gpu::DeviceInfo, cv::Size)
{
    cv::Size size = GET_PARAM(1);       

    int type = CV_32FC2;

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    declare.in(src1, src2, WARMUP_RNG);   
    
    TEST_CYCLE()
    {        
        cv::mulSpectrums(src1, src2, dst, cv::DFT_ROWS, false);
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulAndScaleSpectrums, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES));


#endif
