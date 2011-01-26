#include <stdexcept>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

        xmap.create(size, size, CV_32F);
        ymap.create(size, size, CV_32F);
        for (int i = 0; i < size; ++i)
        {
            float* xmap_row = xmap.ptr<float>(i);
            float* ymap_row = ymap.ptr<float>(i);
            for (int j = 0; j < size; ++j)
            {
                xmap_row[j] = (j - size * 0.5f) * 0.75f + size * 0.5f;
                ymap_row[j] = (i - size * 0.5f) * 0.75f + size * 0.5f;
            }
        }

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


TEST(meanShift)
{
    int sp = 10, sr = 10;

    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 400; size <= 800; size *= 2)
    {
        SUBTEST << "size " << size << ", 8UC3 vs 8UC4";

        gen(src, size, size, CV_8UC3, Scalar::all(0), Scalar::all(256));
        dst.create(src.size(), src.type());

        CPU_ON;
        pyrMeanShiftFiltering(src, dst, sp, sr);
        CPU_OFF;

        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));

        d_src = src;
        d_dst.create(d_src.size(), d_src.type());

        GPU_ON;
        gpu::meanShiftFiltering(d_src, d_dst, sp, sr);
        GPU_OFF;
    }
}


TEST(SURF)
{
    Mat src1 = imread(abspath("bowlingL.png"), CV_LOAD_IMAGE_GRAYSCALE);
    Mat src2 = imread(abspath("bowlingR.png"), CV_LOAD_IMAGE_GRAYSCALE);
    if (src1.empty()) throw runtime_error("can't open bowlingL.png");
    if (src2.empty()) throw runtime_error("can't open bowlingR.png");

    gpu::GpuMat d_src1(src1);
    gpu::GpuMat d_src2(src2);

    SURF surf;
    vector<KeyPoint> keypoints1, keypoints2;

    CPU_ON;
    surf(src1, Mat(), keypoints1);
    surf(src2, Mat(), keypoints2);
    CPU_OFF;

    gpu::SURF_GPU d_surf;
    gpu::GpuMat d_keypoints1, d_keypoints2;
    gpu::GpuMat d_descriptors1, d_descriptors2;

    GPU_ON;
    d_surf(d_src1, gpu::GpuMat(), d_keypoints1);
    d_surf(d_src2, gpu::GpuMat(), d_keypoints2);
    GPU_OFF;
}


TEST(BruteForceMatcher)
{
    RNG rng(0);

    // Init CPU matcher

    int desc_len = 128;
    int num_trains = rng.uniform(1, 5);

    BruteForceMatcher< L2<float> > matcher;

    Mat query; 
    gen(query, rng.uniform(100, 300), desc_len, CV_32F, 0, 10);

    vector<Mat> trains(num_trains);
    for (int i = 0; i < num_trains; ++i)
    {
        Mat train; 
        gen(train, rng.uniform(100, 300), desc_len, CV_32F, 0, 10);
        trains[i] = train;
    }
    matcher.add(trains);

    // Init GPU matcher

    gpu::BruteForceMatcher_GPU< L2<float> > d_matcher;

    gpu::GpuMat d_query(query);

    vector<gpu::GpuMat> d_trains(num_trains);
    for (int i = 0; i < num_trains; ++i)
    {
        d_trains[i] = trains[i];
    }
    d_matcher.add(d_trains);

    // Output
    vector< vector<DMatch> > matches(1);
    vector< vector<DMatch> > d_matches(1);

    SUBTEST << "match";

    CPU_ON;
    matcher.match(query, matches[0]);
    CPU_OFF;

    GPU_ON;
    d_matcher.match(d_query, d_matches[0]);
    GPU_OFF;

    SUBTEST << "knnMatch";
    int knn = rng.uniform(3, 10);

    CPU_ON;
    matcher.knnMatch(query, matches, knn);
    CPU_OFF;

    GPU_ON;
    d_matcher.knnMatch(d_query, d_matches, knn);
    GPU_OFF;

    SUBTEST << "radiusMatch";
    float max_distance = rng.uniform(25.0f, 65.0f);

    CPU_ON;
    matcher.radiusMatch(query, matches, max_distance);
    CPU_OFF;

    GPU_ON;
    d_matcher.radiusMatch(d_query, d_matches, max_distance);
    GPU_OFF;
}