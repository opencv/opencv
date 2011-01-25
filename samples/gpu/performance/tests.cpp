#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "performance.h"

using namespace std;
using namespace cv;

TEST(matchTemplate)
{
    Mat image, templ, result;
    gen(image, 3000, 3000, CV_32F);

    gpu::GpuMat d_image(image), d_templ, d_result;

    for (int templ_size = 5; templ_size <= 1000; templ_size *= 2)
    {
        SUBTEST << "img " << image.rows << ", templ " << templ_size << ", 32F, CCORR";

        gen(templ, templ_size, templ_size, CV_32F);

        CPU_ON;
        matchTemplate(image, templ, result, CV_TM_CCORR);
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
        SUBTEST << "img " << size << ", 32F, no mask";

        gen(src, size, size, CV_32F);

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
        SUBTEST << "img " << size << " and 8UC1, 32FC1 maps";

        gen(src, size, size, CV_8UC1);
        gen(xmap, size, size, CV_32FC1, 0, size);
        gen(ymap, size, size, CV_32FC1, 0, size);

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

    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32FC2, complex-to-complex";

        gen(src, size, size, CV_32FC2);

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

        gen(src, size, size, CV_32FC1);

        CPU_ON;
        cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT101);
        CPU_OFF;

        d_src = src;

        GPU_ON;
        gpu::cornerHarris(d_src, d_dst, 5, 7, 0.1);
        GPU_OFF;
    }
}