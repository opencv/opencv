#include <stdexcept>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/photo/photo.hpp>

#include "performance.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

const Size typicalMatSizes[] =
{
    Size(1024, 768),
    Size(1280, 1024),
    Size(1280, 720),
    Size(1920, 1080)
};
const size_t sizeCount = sizeof(typicalMatSizes) / sizeof(typicalMatSizes[0]);

TEST(blur)
{
    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        blur(src, dst, Size(5, 5));

        CPU_ON;
        blur(src, dst, Size(5, 5));
        CPU_OFF;

        gpu::blur(d_src, d_dst, Size(5, 5));

        GPU_ON;
        gpu::blur(d_src, d_dst, Size(5, 5));
        GPU_OFF;
    }
}

TEST(GaussianBlur)
{
    Mat src, dst;
    GpuMat d_src, d_dst, d_buf;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        GaussianBlur(src, dst, Size(5, 5), 0);

        CPU_ON;
        GaussianBlur(src, dst, Size(5, 5), 0);
        CPU_OFF;

        gpu::GaussianBlur(d_src, d_dst, Size(5, 5), d_buf, 0);

        GPU_ON;
        gpu::GaussianBlur(d_src, d_dst, Size(5, 5), d_buf, 0);
        GPU_OFF;
    }
}

TEST(filter2D)
{
    const int ksize = 5;

    Mat kernel;
    gen(kernel, ksize, ksize, CV_32FC1, Scalar::all(0), Scalar::all(1.0));

    Mat src, dst;
    GpuMat d_src, d_dst, d_buf;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        filter2D(src, dst, -1, kernel);

        CPU_ON;
        filter2D(src, dst, -1, kernel);
        CPU_OFF;

        gpu::filter2D(d_src, d_dst, -1, kernel);

        GPU_ON;
        gpu::filter2D(d_src, d_dst, -1, kernel);
        GPU_OFF;
    }
}

TEST(Laplacian)
{
    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        Laplacian(src, dst, -1, 3);

        CPU_ON;
        Laplacian(src, dst, -1, 3);
        CPU_OFF;

        gpu::Laplacian(d_src, d_dst, -1, 3);

        GPU_ON;
        gpu::Laplacian(d_src, d_dst, -1, 3);
        GPU_OFF;
    }
}

TEST(gemm)
{
    Mat src1, src2, src3, dst;
    GpuMat d_src1, d_src2, d_src3, d_dst;

    for (int size = 512; size <= 1024; size *= 2)
    {
        SUBTEST << size << 'x' << size;

        gen(src1, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src2, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src3, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        d_src1.upload(src1);
        d_src2.upload(src2);
        d_src3.upload(src3);

        gemm(src1, src2, 1.0, src3, 1.0, dst);

        CPU_ON;
        gemm(src1, src2, 1.0, src3, 1.0, dst);
        CPU_OFF;

        gpu::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);

        GPU_ON;
        gpu::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        GPU_OFF;
    }
}

TEST(remap)
{
    const int interpolation = INTER_LINEAR;
    const int borderMode = BORDER_REPLICATE;

    Mat src, dst, xmap, ymap;
    GpuMat d_src, d_dst, d_xmap, d_ymap;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        xmap.create(size, CV_32F);
        ymap.create(size, CV_32F);
        for (int y = 0; y < size.height; ++y)
        {
            float* xmap_row = xmap.ptr<float>(y);
            float* ymap_row = ymap.ptr<float>(y);

            for (int x = 0; x < size.width; ++x)
            {
                xmap_row[x] = (x - size.width * 0.5f) * 0.75f + size.width * 0.5f;
                ymap_row[x] = (y - size.height * 0.5f) * 0.75f + size.height * 0.5f;
            }
        }
        d_xmap.upload(xmap);
        d_ymap.upload(ymap);

        remap(src, dst, xmap, ymap, interpolation, borderMode);

        CPU_ON;
        remap(src, dst, xmap, ymap, interpolation, borderMode);
        CPU_OFF;

        gpu::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);

        GPU_ON;
        gpu::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
        GPU_OFF;
    }
}

TEST(meanShift)
{
    const int sp = 10;
    const int sr = 10;

    Mat src, dst, temp;
    GpuMat d_src, d_dst;

    for (int size = 400; size <= 800; size *= 2)
    {
        SUBTEST << size << 'x' << size;

        gen(src, size, size, CV_8UC3, Scalar::all(0), Scalar::all(256));
        cvtColor(src, temp, COLOR_BGR2BGRA);
        d_src.upload(temp);

        pyrMeanShiftFiltering(src, dst, sp, sr);

        CPU_ON;
        pyrMeanShiftFiltering(src, dst, sp, sr);
        CPU_OFF;

        gpu::meanShiftFiltering(d_src, d_dst, sp, sr);

        GPU_ON;
        gpu::meanShiftFiltering(d_src, d_dst, sp, sr);
        GPU_OFF;
    }
}

TEST(cvtColor)
{
    const int width = 1920;
    const int height = 1080;

    Mat src, dst;
    GpuMat d_src, d_dst;

    gen(src, height, width, CV_8UC1, Scalar::all(0), Scalar::all(256));
    d_src.upload(src);

    SUBTEST << width << "x" << height << ", GRAY -> BGRA";

    cvtColor(src, dst, CV_GRAY2BGRA, 4);

    CPU_ON;
    cvtColor(src, dst, CV_GRAY2BGRA, 4);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_GRAY2BGRA, 4);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_GRAY2BGRA, 4);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", BGR -> YCrCb";

    cvtColor(src, dst, CV_BGR2YCrCb);

    CPU_ON;
    cvtColor(src, dst, CV_BGR2YCrCb);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_BGR2YCrCb, 4);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BGR2YCrCb, 4);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", YCrCb -> BGR";

    cvtColor(src, dst, CV_YCrCb2BGR, 4);

    CPU_ON;
    cvtColor(src, dst, CV_YCrCb2BGR, 4);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_YCrCb2BGR, 4);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_YCrCb2BGR, 4);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", BGR -> XYZ";

    cvtColor(src, dst, CV_BGR2XYZ);

    CPU_ON;
    cvtColor(src, dst, CV_BGR2XYZ);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_BGR2XYZ, 4);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BGR2XYZ, 4);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", XYZ -> BGR";

    cvtColor(src, dst, CV_XYZ2BGR, 4);

    CPU_ON;
    cvtColor(src, dst, CV_XYZ2BGR, 4);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_XYZ2BGR, 4);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_XYZ2BGR, 4);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", BGR -> HSV";

    cvtColor(src, dst, CV_BGR2HSV);

    CPU_ON;
    cvtColor(src, dst, CV_BGR2HSV);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_BGR2HSV, 4);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BGR2HSV, 4);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", HSV -> BGR";

    cvtColor(src, dst, CV_HSV2BGR);

    CPU_ON;
    cvtColor(src, dst, CV_HSV2BGR);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_HSV2BGR);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_HSV2BGR);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", BGR -> Lab";

    cvtColor(src, dst, CV_BGR2Lab);

    CPU_ON;
    cvtColor(src, dst, CV_BGR2Lab);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_BGR2Lab);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BGR2Lab);
    GPU_OFF;

    swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << width << "x" << height << ", Lab -> BGR";

    cvtColor(src, dst, CV_Lab2BGR);

    CPU_ON;
    cvtColor(src, dst, CV_Lab2BGR);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_Lab2BGR);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_Lab2BGR);
    GPU_OFF;

    gen(src, height, width, CV_8UC1, 0, 255);
    d_src.upload(src);

    SUBTEST << width << "x" << height << ", Bayer -> BGR";

    cvtColor(src, dst, CV_BayerBG2BGR);

    CPU_ON;
    cvtColor(src, dst, CV_BayerBG2BGR);
    CPU_OFF;

    gpu::cvtColor(d_src, d_dst, CV_BayerBG2BGR);

    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BayerBG2BGR);
    GPU_OFF;
}

TEST(resize)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height << ", x2";

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        resize(src, dst, Size(), 2.0, 2.0);

        CPU_ON;
        resize(src, dst, Size(), 2.0, 2.0);
        CPU_OFF;

        gpu::resize(d_src, d_dst, Size(), 2.0, 2.0);

        GPU_ON;
        gpu::resize(d_src, d_dst, Size(), 2.0, 2.0);
        GPU_OFF;
    }

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height << ", x0.5";

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        resize(src, dst, Size(), 0.5, 0.5);

        CPU_ON;
        resize(src, dst, Size(), 0.5, 0.5);
        CPU_OFF;

        gpu::resize(d_src, d_dst, Size(), 0.5, 0.5);

        GPU_ON;
        gpu::resize(d_src, d_dst, Size(), 0.5, 0.5);
        GPU_OFF;
    }
}

TEST(warpAffine)
{
    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        const double aplha = CV_PI / 4;
        double mat[2][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                             {std::sin(aplha),  std::cos(aplha), 0}};
        cv::Mat M(2, 3, CV_64F, (void*) mat);

        warpAffine(src, dst, M, src.size());

        CPU_ON;
        warpAffine(src, dst, M, src.size());
        CPU_OFF;

        gpu::warpAffine(d_src, d_dst, M, src.size());

        GPU_ON;
        gpu::warpAffine(d_src, d_dst, M, src.size());
        GPU_OFF;
    }
}

TEST(cornerHarris)
{
    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_32FC1, Scalar::all(0), Scalar::all(1));

        cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT101);
        d_src.upload(src);

        CPU_ON;
        cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT101);
        CPU_OFF;

        gpu::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT101);

        GPU_ON;
        gpu::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT101);
        GPU_OFF;
    }
}

TEST(dft)
{
    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_32FC2, Scalar::all(0), Scalar::all(1));
        createContinuous(src.size(), src.type(), d_src);
        d_src.upload(src);

        dft(src, dst);

        CPU_ON;
        dft(src, dst);
        CPU_OFF;

        gpu::dft(d_src, d_dst, size);

        GPU_ON;
        gpu::dft(d_src, d_dst, size);
        GPU_OFF;
    }
}

TEST(convolve)
{
    const int templ_size = 64;

    Mat src, templ, dst;
    GpuMat d_src, d_templ, d_dst;
    ConvolveBuf d_buf;

    gen(templ, templ_size, templ_size, CV_32FC1, Scalar::all(0), Scalar::all(1));
    d_templ.upload(templ);

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_32FC1, Scalar::all(0), Scalar::all(1));
        d_src.upload(src);

        filter2D(src, dst, src.depth(), templ);

        CPU_ON;
        filter2D(src, dst, src.depth(), templ);
        CPU_OFF;

        gpu::convolve(d_src, d_templ, d_dst, false, d_buf);

        GPU_ON;
        gpu::convolve(d_src, d_templ, d_dst, false, d_buf);
        GPU_OFF;
    }
}

TEST(matchTemplate)
{
    Mat src, templ, dst;
    gen(src, 1920, 1080, CV_32FC1, Scalar::all(0), Scalar::all(1));

    GpuMat d_src(src), d_templ, d_dst;

    for (int templ_size = 5; templ_size < 200; templ_size *= 5)
    {
        SUBTEST << src.cols << 'x' << src.rows << ", templ " << templ_size << 'x' << templ_size;

        gen(templ, templ_size, templ_size, CV_32FC1, Scalar::all(0), Scalar::all(1));
        d_templ.upload(templ);

        matchTemplate(src, templ, dst, CV_TM_CCORR);

        CPU_ON;
        matchTemplate(src, templ, dst, CV_TM_CCORR);
        CPU_OFF;

        gpu::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);

        GPU_ON;
        gpu::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
        GPU_OFF;
    }
}

TEST(pyrDown)
{
    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        pyrDown(src, dst);

        CPU_ON;
        pyrDown(src, dst);
        CPU_OFF;

        gpu::pyrDown(d_src, d_dst);

        GPU_ON;
        gpu::pyrDown(d_src, d_dst);
        GPU_OFF;
    }
}

TEST(pyrUp)
{
    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_8UC4, Scalar::all(0), Scalar::all(256));
        d_src.upload(src);

        pyrUp(src, dst);

        CPU_ON;
        pyrUp(src, dst);
        CPU_OFF;

        gpu::pyrUp(d_src, d_dst);

        GPU_ON;
        gpu::pyrUp(d_src, d_dst);
        GPU_OFF;
    }
}

TEST(bilateralFilter)
{
    const int kernel_size = 5;
    const float sigma_color = 7;
    const float sigma_spatial = 5;

    Mat src, dst;
    GpuMat d_src, d_dst;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_32FC1, Scalar::all(0), Scalar::all(1));
        d_src.upload(src);

        bilateralFilter(src, dst, kernel_size, sigma_color, sigma_spatial);

        CPU_ON;
        bilateralFilter(src, dst, kernel_size, sigma_color, sigma_spatial);
        CPU_OFF;

        gpu::bilateralFilter(d_src, d_dst, kernel_size, sigma_color, sigma_spatial);

        GPU_ON;
        gpu::bilateralFilter(d_src, d_dst, kernel_size, sigma_color, sigma_spatial);
        GPU_OFF;
    }
}

TEST(Canny)
{
    Mat img = imread(abspath("performance.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
    if (img.empty())
        throw runtime_error("can't open performance.jpg");

    SUBTEST << img.cols << 'x' << img.rows;

    Mat edges;

    Canny(img, edges, 50.0, 100.0);

    CPU_ON;
    Canny(img, edges, 50.0, 100.0);
    CPU_OFF;

    GpuMat d_img(img);
    GpuMat d_edges;
    CannyBuf d_buf;

    gpu::Canny(d_img, d_buf, d_edges, 50.0, 100.0);

    GPU_ON;
    gpu::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    GPU_OFF;
}

TEST(HoughLines)
{
    RNG rng(123456789);

    const float rho = 1.0f;
    const float theta = static_cast<float>(CV_PI / 180.0);
    const int threshold = 300;

    Mat src(1080, 1920, CV_8UC1, Scalar::all(0));
    const int numLines = 10;
    for (int i = 0; i < numLines; ++i)
    {
        Point p1(rng.uniform(0, src.cols), rng.uniform(0, src.rows));
        Point p2(rng.uniform(0, src.cols), rng.uniform(0, src.rows));
        line(src, p1, p2, Scalar::all(255), 2);
    }

    vector<Vec2f> lines;

    GpuMat d_src(src);
    GpuMat d_lines;
    cv::gpu::HoughLinesBuf d_buf;

    SUBTEST << src.cols << 'x' << src.rows;

    HoughLines(src, lines, rho, theta, threshold);

    CPU_ON;
    HoughLines(src, lines, rho, theta, threshold);
    CPU_OFF;

    gpu::HoughLines(d_src, d_lines, d_buf, rho, theta, threshold);

    GPU_ON;
    gpu::HoughLines(d_src, d_lines, d_buf, rho, theta, threshold);
    GPU_OFF;
}

TEST(HoughCircles)
{
    RNG rng(123456789);

    const int minRadius = 10;
    const int maxRadius = 30;
    const int cannyThreshold = 100;
    const int votesThreshold = 15;
    const float dp = 2;
    const float minDist = 1;

    Mat src(1080, 1920, CV_8UC1, Scalar::all(0));
    const int numCircles = 10;
    for (int i = 0; i < numCircles; ++i)
    {
        Point center(rng.uniform(0, src.cols), rng.uniform(0, src.rows));
        const int radius = rng.uniform(minRadius, maxRadius + 1);
        circle(src, center, radius, cv::Scalar::all(255), -1);
    }

    vector<Vec3f> circles;

    GpuMat d_src(src);
    GpuMat d_circles;
    HoughCirclesBuf d_buf;

    SUBTEST << src.cols << 'x' << src.rows;

    HoughCircles(src, circles, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

    CPU_ON;
    HoughCircles(src, circles, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);
    CPU_OFF;

    gpu::HoughCircles(d_src, d_circles, d_buf, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

    GPU_ON;
    gpu::HoughCircles(d_src, d_circles, d_buf, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);
    GPU_OFF;
}

TEST(norm)
{
    Mat src;
    GpuMat d_src, d_buf;

    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        gen(src, size.height, size.width, CV_32FC4, Scalar::all(0), Scalar::all(1));
        d_src.upload(src);

        norm(src, NORM_INF);

        CPU_ON;
        norm(src, NORM_INF);
        CPU_OFF;

        gpu::norm(d_src, NORM_INF, d_buf);

        GPU_ON;
        gpu::norm(d_src, NORM_INF, d_buf);
        GPU_OFF;
    }
}

TEST(reduce)
{
    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        Mat src;
        gen(src, size.height, size.width, CV_32FC1, Scalar::all(0), Scalar::all(1));

        Mat dst0;
        Mat dst1;

        GpuMat d_src(src);
        GpuMat d_dst0;
        GpuMat d_dst1;

        SUBTEST << size.width << 'x' << size.height << ", dim = 0";

        reduce(src, dst0, 0, CV_REDUCE_MIN);

        CPU_ON;
        reduce(src, dst0, 0, CV_REDUCE_MIN);
        CPU_OFF;

        gpu::reduce(d_src, d_dst0, 0, CV_REDUCE_MIN);

        GPU_ON;
        gpu::reduce(d_src, d_dst0, 0, CV_REDUCE_MIN);
        GPU_OFF;

        SUBTEST << size.width << 'x' << size.height << ", dim = 1";

        reduce(src, dst1, 1, CV_REDUCE_MIN);

        CPU_ON;
        reduce(src, dst1, 1, CV_REDUCE_MIN);
        CPU_OFF;

        gpu::reduce(d_src, d_dst1, 1, CV_REDUCE_MIN);

        GPU_ON;
        gpu::reduce(d_src, d_dst1, 1, CV_REDUCE_MIN);
        GPU_OFF;
    }
}

TEST(equalizeHist)
{
    for (int i = 0; i < sizeCount; ++i)
    {
        Size size = typicalMatSizes[i];

        SUBTEST << size.width << 'x' << size.height;

        Mat src, dst;
        gen(src, size.height, size.width, CV_8UC1, Scalar::all(0), Scalar::all(256));

        GpuMat d_src(src);
        GpuMat d_dst;
        GpuMat d_hist;
        GpuMat d_buf;

        equalizeHist(src, dst);

        CPU_ON;
        equalizeHist(src, dst);
        CPU_OFF;

        gpu::equalizeHist(d_src, d_dst, d_hist, d_buf);

        GPU_ON;
        gpu::equalizeHist(d_src, d_dst, d_hist, d_buf);
        GPU_OFF;
    }
}
