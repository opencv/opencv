#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "performance.h"

using namespace std;
using namespace cv;

// This code calls CUFFT DFT and initializes that lib
INIT(CUFFT_library)
{
    Mat src, templ;
    gen(src, 500, 500, CV_32F, 0, 1);
    gen(templ, 500, 500, CV_32F, 0, 1);

    gpu::GpuMat d_src(src);
    gpu::GpuMat d_templ(templ);
    gpu::GpuMat d_result;

    gpu::matchTemplate(d_src, d_templ, d_result, CV_TM_CCORR);
}


TEST(matchTemplate)
{
    Mat src, templ, result;
    gen(src, 3000, 3000, CV_32F, 0, 1);

    gpu::GpuMat d_image(src), d_templ, d_result;

    for (int templ_size = 5; templ_size <= 1000; templ_size *= 2)
    {
        SUBTEST << "src " << src.rows << ", templ " << templ_size << ", 32F, CCORR";

        gen(templ, templ_size, templ_size, CV_32F, 0, 1);

        CPU_ON;
        matchTemplate(src, templ, result, CV_TM_CCORR);
        CPU_OFF;

        d_templ = templ;

        GPU_ON;
        gpu::matchTemplate(d_image, d_templ, d_result, CV_TM_CCORR);
        GPU_OFF;
    }
}


TEST(minMaxLoc)
{
    Mat src;
    gpu::GpuMat d_src;

    double min_val, max_val;
    Point min_loc, max_loc;

    for (int size = 2000; size <= 8000; size *= 2)
    {
        SUBTEST << "src " << size << ", 32F, no mask";

        gen(src, size, size, CV_32F, 0, 1);

        CPU_ON;
        minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);
        CPU_OFF;

        d_src = src;

        GPU_ON;
        gpu::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
        GPU_OFF;
    }
}


TEST(remap)
{
    Mat src, dst, xmap, ymap;
    gpu::GpuMat d_src, d_dst, d_xmap, d_ymap;

    for (int size = 1000; size <= 8000; size *= 2)
    {
        SUBTEST << "src " << size << " and 8UC1, 32FC1 maps";

        gen(src, size, size, CV_8UC1, 0, 256);
        gen(xmap, size, size, CV_32F, 0, size);
        gen(ymap, size, size, CV_32F, 0, size);

        CPU_ON;
        remap(src, dst, xmap, ymap, INTER_LINEAR);
        CPU_OFF;

        d_src = src;
        d_xmap = xmap;
        d_ymap = ymap;

        GPU_ON;
        gpu::remap(d_src, d_dst, d_xmap, d_ymap);
        GPU_OFF;
    }
}


TEST(dft)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 8000; size *= 2)
    {
        SUBTEST << "size " << size << ", 32FC2, complex-to-complex";

        gen(src, size, size, CV_32FC2, Scalar::all(0), Scalar::all(1));

        CPU_ON;
        dft(src, dst);
        CPU_OFF;

        d_src = src;

        GPU_ON;
        gpu::dft(d_src, d_dst, Size(size, size));
        GPU_OFF;
    }
}


TEST(cornerHarris)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 2000; size <= 4000; size *= 2)
    {
        SUBTEST << "size " << size << ", 32FC1";

        gen(src, size, size, CV_32F, 0, 1);

        CPU_ON;
        cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT101);
        CPU_OFF;

        d_src = src;

        GPU_ON;
        gpu::cornerHarris(d_src, d_dst, 5, 7, 0.1);
        GPU_OFF;
    }
}


TEST(memoryAllocation)
{
    Mat mat;
    gpu::GpuMat d_mat;

    int begin = 100, end = 8000, step = 100;

    DESCRIPTION << "32F matrices from " << begin << " to " << end;

    CPU_ON;
    for (int size = begin; size <= end; size += step)
        mat.create(size, size, CV_32FC1);
    CPU_OFF;

    GPU_ON;
    for (int size = begin; size <= end; size += step)
        d_mat.create(size, size, CV_32FC1);
    GPU_OFF;
}
