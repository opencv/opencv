#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "performance.h"

using namespace std;
using namespace cv;

INIT(matchTemplate)
{
    Mat src; gen(src, 500, 500, CV_32F, 0, 1);
    Mat templ; gen(templ, 500, 500, CV_32F, 0, 1);

    gpu::GpuMat d_src(src), d_templ(templ), d_dst;

    gpu::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
}


TEST(matchTemplate)
{
    Mat src, templ, dst;
    gen(src, 3000, 3000, CV_32F, 0, 1);

    gpu::GpuMat d_src(src), d_templ, d_dst;

    for (int templ_size = 5; templ_size < 200; templ_size *= 5)
    {
        SUBTEST << "src " << src.rows << ", templ " << templ_size << ", 32F, CCORR";

        gen(templ, templ_size, templ_size, CV_32F, 0, 1);
        dst.create(src.rows - templ.rows + 1, src.cols - templ.cols + 1, CV_32F);

        CPU_ON;
        matchTemplate(src, templ, dst, CV_TM_CCORR);
        CPU_OFF;

        d_templ = templ;
        d_dst.create(d_src.rows - d_templ.rows + 1, d_src.cols - d_templ.cols + 1, CV_32F);

        GPU_ON;
        gpu::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
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
        SUBTEST << "src " << size << " and 8U, 32F maps";

        gen(src, size, size, CV_8UC1, 0, 256);
        gen(xmap, size, size, CV_32F, 0, size);
        gen(ymap, size, size, CV_32F, 0, size);
        dst.create(xmap.size(), src.type());

        CPU_ON;
        remap(src, dst, xmap, ymap, INTER_LINEAR);
        CPU_OFF;

        d_src = src;
        d_xmap = xmap;
        d_ymap = ymap;
        d_dst.create(d_xmap.size(), d_src.type());

        GPU_ON;
        gpu::remap(d_src, d_dst, d_xmap, d_ymap);
        GPU_OFF;

        SUBTEST << "src " << size << " and 8U, 32F singular maps";

        gen(xmap, size, size, CV_32F, 0, 0);
        gen(ymap, size, size, CV_32F, 0, 0);

        CPU_ON;
        remap(src, dst, xmap, ymap, INTER_LINEAR);
        CPU_OFF;

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

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << "size " << size << ", 32FC2, complex-to-complex";

        gen(src, size, size, CV_32FC2, Scalar::all(0), Scalar::all(1));
        dst.create(src.size(), src.type());

        CPU_ON;
        dft(src, dst);
        CPU_OFF;

        d_src = src;
        d_dst.create(d_src.size(), d_src.type());

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
        SUBTEST << "size " << size << ", 32F";

        gen(src, size, size, CV_32F, 0, 1);
        dst.create(src.size(), src.type());

        CPU_ON;
        cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT101);
        CPU_OFF;

        d_src = src;
        d_dst.create(src.size(), src.type());

        GPU_ON;
        gpu::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT101);
        GPU_OFF;
    }
}


TEST(integral)
{
    Mat src, sum;
    gpu::GpuMat d_src, d_sum;

    for (int size = 1000; size <= 8000; size *= 2)
    {
        SUBTEST << "size " << size << ", 8U";

        gen(src, size, size, CV_8U, 0, 256);
        sum.create(size + 1, size + 1, CV_32S);

        CPU_ON;
        integral(src, sum);
        CPU_OFF;

        d_src = src;
        d_sum.create(size + 1, size + 1, CV_32S);

        GPU_ON;
        gpu::integral(d_src, d_sum);
        GPU_OFF;
    }
}


TEST(norm)
{
    Mat src;
    gpu::GpuMat d_src;

    for (int size = 1000; size <= 8000; size *= 2)
    {
        SUBTEST << "size " << size << ", 8U";

        gen(src, size, size, CV_8U, 0, 256);

        CPU_ON;
        norm(src);
        CPU_OFF;

        d_src = src;

        GPU_ON;
        gpu::norm(d_src);
        GPU_OFF;
    }
}
