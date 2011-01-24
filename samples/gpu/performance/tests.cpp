#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "performance.h"

using namespace std;
using namespace cv;

TEST(matchTemplate)
{
    for (int templ_size = 5; templ_size <= 1000; templ_size *= 2)
    {
        SUBTEST << "img 3000, templ " << templ_size << ", 8U, SQDIFF";

        Mat image; gen(image, 3000, 3000, CV_8U);
        Mat templ; gen(templ, templ_size, templ_size, CV_8U);
        Mat result;

        CPU_ON;
        matchTemplate(image, templ, result, CV_TM_SQDIFF);
        CPU_OFF;

        gpu::GpuMat d_image(image);
        gpu::GpuMat d_templ(templ);
        gpu::GpuMat d_result;

        GPU_ON;
        gpu::matchTemplate(d_image, d_templ, d_result, CV_TM_SQDIFF);
        GPU_OFF;
    }
}
