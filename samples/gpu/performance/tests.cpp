#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "performance.h"

using namespace std;
using namespace cv;

TEST(matchTemplate)
{
    Mat image, templ, result;
    gen(image, 3000, 3000, CV_8U);

    gpu::GpuMat d_image(image), d_templ, d_result;

    for (int templ_size = 5; templ_size <= 1000; templ_size *= 2)
    {
        SUBTEST << "img 3000, templ " << templ_size << ", 8U, SQDIFF";

        gen(templ, templ_size, templ_size, CV_8U);

        CPU_ON;
        matchTemplate(image, templ, result, CV_TM_SQDIFF);
        CPU_OFF;

        d_templ = templ;

        GPU_ON;
        gpu::matchTemplate(d_image, d_templ, d_result, CV_TM_SQDIFF);
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