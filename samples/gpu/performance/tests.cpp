#include <stdexcept>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "performance.h"

using namespace std;
using namespace cv;

void InitMatchTemplate()
{
    Mat src; gen(src, 500, 500, CV_32F, 0, 1);
    Mat templ; gen(templ, 500, 500, CV_32F, 0, 1);
    gpu::GpuMat d_src(src), d_templ(templ), d_dst;
    gpu::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
}


TEST(matchTemplate)
{
    InitMatchTemplate();

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

        d_templ.upload(templ);
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

        d_src.upload(src);

        GPU_ON;
        gpu::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
        GPU_OFF;
    }
}


TEST(remap)
{
    Mat src, dst, xmap, ymap;
    gpu::GpuMat d_src, d_dst, d_xmap, d_ymap;
    
    int interpolation = INTER_LINEAR;
    int borderMode = BORDER_REPLICATE;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << "src " << size << ", 8UC1";

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
        remap(src, dst, xmap, ymap, interpolation, borderMode);
        CPU_OFF;

        d_src.upload(src);
        d_xmap.upload(xmap);
        d_ymap.upload(ymap);
        d_dst.create(d_xmap.size(), d_src.type());

        GPU_ON;
        gpu::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
        GPU_OFF;
    }

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << "src " << size << ", 8UC3";

        gen(src, size, size, CV_8UC3, 0, 256);

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
        remap(src, dst, xmap, ymap, interpolation, borderMode);
        CPU_OFF;

        d_src.upload(src);
        d_xmap.upload(xmap);
        d_ymap.upload(ymap);
        d_dst.create(d_xmap.size(), d_src.type());

        GPU_ON;
        gpu::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
        GPU_OFF;
    }

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << "src " << size << ", 8UC4";

        gen(src, size, size, CV_8UC4, 0, 256);

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
        remap(src, dst, xmap, ymap, interpolation, borderMode);
        CPU_OFF;

        d_src.upload(src);
        d_xmap.upload(xmap);
        d_ymap.upload(ymap);
        d_dst.create(d_xmap.size(), d_src.type());

        GPU_ON;
        gpu::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
        GPU_OFF;
    }

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << "src " << size << ", 16SC3";

        gen(src, size, size, CV_16SC3, 0, 256);

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
        remap(src, dst, xmap, ymap, interpolation, borderMode);
        CPU_OFF;

        d_src.upload(src);
        d_xmap.upload(xmap);
        d_ymap.upload(ymap);
        d_dst.create(d_xmap.size(), d_src.type());

        GPU_ON;
        gpu::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
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

        d_src.upload(src);
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

        d_src.upload(src);
        d_dst.create(src.size(), src.type());

        GPU_ON;
        gpu::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT101);
        GPU_OFF;
    }
}


TEST(integral)
{
    Mat src, sum;
    gpu::GpuMat d_src, d_sum, d_buf;

    int size = 4000;

    gen(src, size, size, CV_8U, 0, 256);
    sum.create(size + 1, size + 1, CV_32S);

    d_src.upload(src);
    d_sum.create(size + 1, size + 1, CV_32S);

    for (int i = 0; i < 5; ++i)
    {
        SUBTEST << "size " << size << ", 8U";

        CPU_ON;
        integral(src, sum);
        CPU_OFF;

        GPU_ON;
        gpu::integralBuffered(d_src, d_sum, d_buf);
        GPU_OFF;
    }
}


TEST(norm)
{
    Mat src;
    gpu::GpuMat d_src, d_buf;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32FC4, NORM_INF";

        gen(src, size, size, CV_32FC4, Scalar::all(0), Scalar::all(1));

        CPU_ON;
        for (int i = 0; i < 5; ++i)
            norm(src, NORM_INF);
        CPU_OFF;

        d_src.upload(src);

        GPU_ON;
        for (int i = 0; i < 5; ++i)
            gpu::norm(d_src, NORM_INF, d_buf);
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

        d_src.upload(src);
        d_dst.create(d_src.size(), d_src.type());

        GPU_ON;
        gpu::meanShiftFiltering(d_src, d_dst, sp, sr);
        GPU_OFF;
    }
}


TEST(SURF)
{
    Mat src = imread(abspath("aloeL.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
    if (src.empty()) throw runtime_error("can't open aloeL.jpg");

    SURF surf;
    vector<KeyPoint> keypoints;
    vector<float> descriptors;

    surf(src, Mat(), keypoints, descriptors);

    CPU_ON;
    surf(src, Mat(), keypoints, descriptors);
    CPU_OFF;

    gpu::SURF_GPU d_surf;
    gpu::GpuMat d_src(src);
    gpu::GpuMat d_keypoints;
    gpu::GpuMat d_descriptors;

    d_surf(d_src, gpu::GpuMat(), d_keypoints, d_descriptors);

    GPU_ON;
    d_surf(d_src, gpu::GpuMat(), d_keypoints, d_descriptors);
    GPU_OFF;
}


TEST(FAST)
{
    Mat src = imread(abspath("aloeL.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
    if (src.empty()) throw runtime_error("can't open aloeL.jpg");

    vector<KeyPoint> keypoints;

    FAST(src, keypoints, 20);

    CPU_ON;
    FAST(src, keypoints, 20);
    CPU_OFF;

    gpu::FAST_GPU d_FAST(20);
    gpu::GpuMat d_src(src);
    gpu::GpuMat d_keypoints;

    d_FAST(d_src, gpu::GpuMat(), d_keypoints);

    GPU_ON;
    d_FAST(d_src, gpu::GpuMat(), d_keypoints);
    GPU_OFF;
}


TEST(ORB)
{
    Mat src = imread(abspath("aloeL.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
    if (src.empty()) throw runtime_error("can't open aloeL.jpg");

    ORB orb(4000);
    vector<KeyPoint> keypoints;
    Mat descriptors;

    orb(src, Mat(), keypoints, descriptors);

    CPU_ON;
    orb(src, Mat(), keypoints, descriptors);
    CPU_OFF;

    gpu::ORB_GPU d_orb;
    gpu::GpuMat d_src(src);
    gpu::GpuMat d_keypoints;
    gpu::GpuMat d_descriptors;

    d_orb(d_src, gpu::GpuMat(), d_keypoints, d_descriptors);

    GPU_ON;
    d_orb(d_src, gpu::GpuMat(), d_keypoints, d_descriptors);
    GPU_OFF;
}


TEST(BruteForceMatcher)
{
    // Init CPU matcher

    int desc_len = 64;

    BruteForceMatcher< L2<float> > matcher;

    Mat query; 
    gen(query, 3000, desc_len, CV_32F, 0, 1);
    
    Mat train; 
    gen(train, 3000, desc_len, CV_32F, 0, 1);

    // Init GPU matcher

    gpu::BruteForceMatcher_GPU< L2<float> > d_matcher;

    gpu::GpuMat d_query(query);
    gpu::GpuMat d_train(train);

    // Output
    vector< vector<DMatch> > matches(2);
    gpu::GpuMat d_trainIdx, d_distance, d_allDist, d_nMatches;

    SUBTEST << "match";

    matcher.match(query, train, matches[0]);
    CPU_ON;
    matcher.match(query, train, matches[0]);
    CPU_OFF;

    d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
    GPU_ON;
    d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
    GPU_OFF;

    SUBTEST << "knnMatch, 2";

    matcher.knnMatch(query, train, matches, 2);
    CPU_ON;
    matcher.knnMatch(query, train, matches, 2);
    CPU_OFF;

    d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);
    GPU_ON;
    d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);
    GPU_OFF;

    SUBTEST << "knnMatch, 3";

    matcher.knnMatch(query, train, matches, 3);
    CPU_ON;
    matcher.knnMatch(query, train, matches, 3);
    CPU_OFF;

    d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 3);
    GPU_ON;
    d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 3);
    GPU_OFF;

    SUBTEST << "radiusMatch";
    float max_distance = 2.0f;

    matcher.radiusMatch(query, train, matches, max_distance);
    CPU_ON;
    matcher.radiusMatch(query, train, matches, max_distance);
    CPU_OFF;

    d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);
    GPU_ON;
    d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);
    GPU_OFF;
}


TEST(magnitude)
{
    Mat x, y, mag;
    gpu::GpuMat d_x, d_y, d_mag;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size;

        gen(x, size, size, CV_32F, 0, 1);
        gen(y, size, size, CV_32F, 0, 1);
        mag.create(size, size, CV_32F);

        CPU_ON;
        magnitude(x, y, mag);
        CPU_OFF;

        d_x.upload(x);
        d_y.upload(y);
        d_mag.create(size, size, CV_32F);

        GPU_ON;
        gpu::magnitude(d_x, d_y, d_mag);
        GPU_OFF;
    }
}


TEST(add)
{
    Mat src1, src2, dst;
    gpu::GpuMat d_src1, d_src2, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32F";

        gen(src1, size, size, CV_32F, 0, 1);
        gen(src2, size, size, CV_32F, 0, 1);
        dst.create(size, size, CV_32F);

        CPU_ON;
        add(src1, src2, dst);
        CPU_OFF;

        d_src1.upload(src1);
        d_src2.upload(src2);
        d_dst.create(size, size, CV_32F);

        GPU_ON;
        gpu::add(d_src1, d_src2, d_dst);
        GPU_OFF;
    }
}


TEST(log)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32F";

        gen(src, size, size, CV_32F, 1, 10);
        dst.create(size, size, CV_32F);

        CPU_ON;
        log(src, dst);
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size, size, CV_32F);

        GPU_ON;
        gpu::log(d_src, d_dst);
        GPU_OFF;
    }
}


TEST(exp)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32F";

        gen(src, size, size, CV_32F, 0, 1);
        dst.create(size, size, CV_32F);

        CPU_ON;
        exp(src, dst);
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size, size, CV_32F);

        GPU_ON;
        gpu::exp(d_src, d_dst);
        GPU_OFF;
    }
}


TEST(mulSpectrums)
{
    Mat src1, src2, dst;
    gpu::GpuMat d_src1, d_src2, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size;

        gen(src1, size, size, CV_32FC2, Scalar::all(0), Scalar::all(1));
        gen(src2, size, size, CV_32FC2, Scalar::all(0), Scalar::all(1));
        dst.create(size, size, CV_32FC2);

        CPU_ON;
        mulSpectrums(src1, src2, dst, 0, true);
        CPU_OFF;

        d_src1.upload(src1);
        d_src2.upload(src2);
        d_dst.create(size, size, CV_32FC2);

        GPU_ON;
        gpu::mulSpectrums(d_src1, d_src2, d_dst, 0, true);
        GPU_OFF;
    }
}


TEST(resize)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8UC1, up";

        gen(src, size, size, CV_8U, 0, 256);
        dst.create(size * 2, size * 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size * 2, size * 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8UC1, down";

        gen(src, size, size, CV_8U, 0, 256);
        dst.create(size / 2, size / 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size / 2, size / 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8UC3, up";

        gen(src, size, size, CV_8UC3, 0, 256);
        dst.create(size * 2, size * 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size * 2, size * 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8UC3, down";

        gen(src, size, size, CV_8UC3, 0, 256);
        dst.create(size / 2, size / 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size / 2, size / 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8UC4, up";

        gen(src, size, size, CV_8UC4, 0, 256);
        dst.create(size * 2, size * 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size * 2, size * 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8UC4, down";

        gen(src, size, size, CV_8UC4, 0, 256);
        dst.create(size / 2, size / 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size / 2, size / 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32FC1, up";

        gen(src, size, size, CV_32FC1, 0, 256);
        dst.create(size * 2, size * 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size * 2, size * 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32FC1, down";

        gen(src, size, size, CV_32FC1, 0, 256);
        dst.create(size / 2, size / 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size / 2, size / 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
}


TEST(cvtColor)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    gen(src, 4000, 4000, CV_8UC1, 0, 255);
    d_src.upload(src);
    
    SUBTEST << "size 4000, CV_GRAY2BGRA";
    
    dst.create(src.size(), CV_8UC4);

    CPU_ON;
    cvtColor(src, dst, CV_GRAY2BGRA, 4);
    CPU_OFF;
    
    d_dst.create(d_src.size(), CV_8UC4);
    
    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_GRAY2BGRA, 4);
    GPU_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "size 4000, CV_BGR2YCrCb";
    
    dst.create(src.size(), CV_8UC3);

    CPU_ON;
    cvtColor(src, dst, CV_BGR2YCrCb);
    CPU_OFF;
    
    d_dst.create(d_src.size(), CV_8UC4);
        
    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BGR2YCrCb, 4);
    GPU_OFF;
    
    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "size 4000, CV_YCrCb2BGR";
    
    dst.create(src.size(), CV_8UC4);

    CPU_ON;
    cvtColor(src, dst, CV_YCrCb2BGR, 4);
    CPU_OFF;
    
    d_dst.create(d_src.size(), CV_8UC4);
        
    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_YCrCb2BGR, 4);
    GPU_OFF;
    
    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "size 4000, CV_BGR2XYZ";
    
    dst.create(src.size(), CV_8UC3);

    CPU_ON;
    cvtColor(src, dst, CV_BGR2XYZ);
    CPU_OFF;
    
    d_dst.create(d_src.size(), CV_8UC4);
        
    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BGR2XYZ, 4);
    GPU_OFF;
    
    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "size 4000, CV_XYZ2BGR";
    
    dst.create(src.size(), CV_8UC4);

    CPU_ON;
    cvtColor(src, dst, CV_XYZ2BGR, 4);
    CPU_OFF;
    
    d_dst.create(d_src.size(), CV_8UC4);
        
    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_XYZ2BGR, 4);
    GPU_OFF;
    
    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "size 4000, CV_BGR2HSV";
    
    dst.create(src.size(), CV_8UC3);

    CPU_ON;
    cvtColor(src, dst, CV_BGR2HSV);
    CPU_OFF;
    
    d_dst.create(d_src.size(), CV_8UC4);
        
    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_BGR2HSV, 4);
    GPU_OFF;
    
    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "size 4000, CV_HSV2BGR";
    
    dst.create(src.size(), CV_8UC4);

    CPU_ON;
    cvtColor(src, dst, CV_HSV2BGR, 4);
    CPU_OFF;
    
    d_dst.create(d_src.size(), CV_8UC4);
        
    GPU_ON;
    gpu::cvtColor(d_src, d_dst, CV_HSV2BGR, 4);
    GPU_OFF;
    
    cv::swap(src, dst);
    d_src.swap(d_dst);
}


TEST(erode)
{
    Mat src, dst, ker;
    gpu::GpuMat d_src, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size;

        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));
        ker = getStructuringElement(MORPH_RECT, Size(3, 3));
        dst.create(src.size(), src.type());

        CPU_ON;
        erode(src, dst, ker);
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(d_src.size(), d_src.type());

        GPU_ON;
        gpu::erode(d_src, d_dst, ker);
        GPU_OFF;
    }
}

TEST(threshold)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8U, THRESH_BINARY";

        gen(src, size, size, CV_8U, 0, 100);
        dst.create(size, size, CV_8U);

        CPU_ON; 
        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size, size, CV_8U);

        GPU_ON;
        gpu::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        GPU_OFF;
    }

    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32F, THRESH_BINARY";

        gen(src, size, size, CV_32F, 0, 100);
        dst.create(size, size, CV_32F);

        CPU_ON; 
        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size, size, CV_32F);

        GPU_ON;
        gpu::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        GPU_OFF;
    }
}

TEST(pow)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32F";

        gen(src, size, size, CV_32F, 0, 100);
        dst.create(size, size, CV_32F);

        CPU_ON;
        pow(src, -2.0, dst);
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(size, size, CV_32F);

        GPU_ON;
        gpu::pow(d_src, -2.0, d_dst);
        GPU_OFF;
    }
}


TEST(projectPoints)
{
    Mat src;
    vector<Point2f> dst;
    gpu::GpuMat d_src, d_dst;

    Mat rvec; gen(rvec, 1, 3, CV_32F, 0, 1);
    Mat tvec; gen(tvec, 1, 3, CV_32F, 0, 1);
    Mat camera_mat; gen(camera_mat, 3, 3, CV_32F, 0, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    for (int size = (int)1e6, count = 0; size >= 1e5 && count < 5; size = int(size / 1.4), count++)
    {
        SUBTEST << "size " << size;

        gen(src, 1, size, CV_32FC3, Scalar::all(0), Scalar::all(10));
        dst.resize(size);

        CPU_ON;
        projectPoints(src, rvec, tvec, camera_mat, Mat::zeros(1, 8, CV_32F), dst);
        CPU_OFF;

        d_src.upload(src);
        d_dst.create(1, size, CV_32FC2);

        GPU_ON;
        gpu::projectPoints(d_src, rvec, tvec, camera_mat, Mat(), d_dst);
        GPU_OFF;
    }
}


void InitSolvePnpRansac()
{
    Mat object; gen(object, 1, 4, CV_32FC3, Scalar::all(0), Scalar::all(100));
    Mat image; gen(image, 1, 4, CV_32FC2, Scalar::all(0), Scalar::all(100));
    Mat rvec, tvec;
    gpu::solvePnPRansac(object, image, Mat::eye(3, 3, CV_32F), Mat(), rvec, tvec);
}


TEST(solvePnPRansac)
{
    InitSolvePnpRansac();

    for (int num_points = 5000; num_points <= 300000; num_points = int(num_points * 3.76))
    {
        SUBTEST << "num_points " << num_points;

        Mat object; gen(object, 1, num_points, CV_32FC3, Scalar::all(10), Scalar::all(100));
        Mat image; gen(image, 1, num_points, CV_32FC2, Scalar::all(10), Scalar::all(100));
        Mat camera_mat; gen(camera_mat, 3, 3, CV_32F, 0.5, 1);
        camera_mat.at<float>(0, 1) = 0.f;
        camera_mat.at<float>(1, 0) = 0.f;
        camera_mat.at<float>(2, 0) = 0.f;
        camera_mat.at<float>(2, 1) = 0.f;

        Mat rvec, tvec;
        const int num_iters = 200;
        const float max_dist = 2.0f;
        vector<int> inliers_cpu, inliers_gpu;

        CPU_ON;
        solvePnPRansac(object, image, camera_mat, Mat::zeros(1, 8, CV_32F), rvec, tvec, false, num_iters,
                       max_dist, int(num_points * 0.05), inliers_cpu);
        CPU_OFF;

        GPU_ON;
        gpu::solvePnPRansac(object, image, camera_mat, Mat::zeros(1, 8, CV_32F), rvec, tvec, false, num_iters,
                            max_dist, int(num_points * 0.05), &inliers_gpu);
        GPU_OFF;
    }
}


TEST(GaussianBlur)
{
    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << "8UC1, size " << size;

        Mat src; gen(src, size, size, CV_8UC1, 0, 256);
        Mat dst(src.size(), src.type());

        CPU_ON;
        GaussianBlur(src, dst, Size(3, 3), 1);
        CPU_OFF;

        gpu::GpuMat d_src(src);
        gpu::GpuMat d_dst(src.size(), src.type());
        gpu::GpuMat d_buf;
        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);

        GPU_ON;
        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);
        GPU_OFF;
    }

    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << "8UC4, size " << size;

        Mat src; gen(src, size, size, CV_8UC4, 0, 256);
        Mat dst(src.size(), src.type());

        CPU_ON;
        GaussianBlur(src, dst, Size(3, 3), 1);
        CPU_OFF;

        gpu::GpuMat d_src(src);
        gpu::GpuMat d_dst(src.size(), src.type());
        gpu::GpuMat d_buf;
        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);

        GPU_ON;
        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);
        GPU_OFF;
    }

    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << "32FC1, size " << size;

        Mat src; gen(src, size, size, CV_32FC1, 0, 1);
        Mat dst(src.size(), src.type());

        CPU_ON;
        GaussianBlur(src, dst, Size(3, 3), 1);
        CPU_OFF;

        gpu::GpuMat d_src(src);
        gpu::GpuMat d_dst(src.size(), src.type());
        gpu::GpuMat d_buf;
        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);

        GPU_ON;
        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);
        GPU_OFF;
    }
}

TEST(pyrDown)
{
    {
        for (int size = 4000; size >= 1000; size -= 1000)
        {
            SUBTEST << "8UC1, size " << size;

            Mat src; gen(src, size, size, CV_8UC1, 0, 256);
            Mat dst(Size(src.cols / 2, src.rows / 2), src.type());

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols / 2, src.rows / 2), src.type());

            GPU_ON;
            gpu::pyrDown(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 4000; size >= 1000; size -= 1000)
        {
            SUBTEST << "8UC3, size " << size;

            Mat src; gen(src, size, size, CV_8UC3, 0, 256);
            Mat dst(Size(src.cols / 2, src.rows / 2), src.type());

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols / 2, src.rows / 2), src.type());

            GPU_ON;
            gpu::pyrDown(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 4000; size >= 1000; size -= 1000)
        {
            SUBTEST << "8UC4, size " << size;

            Mat src; gen(src, size, size, CV_8UC4, 0, 256);
            Mat dst(Size(src.cols / 2, src.rows / 2), src.type());

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols / 2, src.rows / 2), src.type());

            GPU_ON;
            gpu::pyrDown(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 4000; size >= 1000; size -= 1000)
        {
            SUBTEST << "16SC3, size " << size;

            Mat src; gen(src, size, size, CV_16SC3, 0, 256);
            Mat dst(Size(src.cols / 2, src.rows / 2), src.type());

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols / 2, src.rows / 2), src.type());

            GPU_ON;
            gpu::pyrDown(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 4000; size >= 1000; size -= 1000)
        {
            SUBTEST << "32FC1, size " << size;

            Mat src; gen(src, size, size, CV_32FC1, 0, 256);
            Mat dst(Size(src.cols / 2, src.rows / 2), src.type());

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols / 2, src.rows / 2), src.type());

            GPU_ON;
            gpu::pyrDown(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 4000; size >= 1000; size -= 1000)
        {
            SUBTEST << "32FC3, size " << size;

            Mat src; gen(src, size, size, CV_32FC3, 0, 256);
            Mat dst(Size(src.cols / 2, src.rows / 2), src.type());

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols / 2, src.rows / 2), src.type());

            GPU_ON;
            gpu::pyrDown(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 4000; size >= 1000; size -= 1000)
        {
            SUBTEST << "32FC4, size " << size;

            Mat src; gen(src, size, size, CV_32FC4, 0, 256);
            Mat dst(Size(src.cols / 2, src.rows / 2), src.type());

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols / 2, src.rows / 2), src.type());

            GPU_ON;
            gpu::pyrDown(d_src, d_dst);
            GPU_OFF;
        }
    }
}

TEST(pyrUp)
{
    {
        for (int size = 2000; size >= 1000; size -= 1000)
        {
            SUBTEST << "8UC1, size " << size;

            Mat src; gen(src, size, size, CV_8UC1, 0, 256);
            Mat dst(Size(src.cols * 2, src.rows * 2), src.type());

            CPU_ON;
            pyrUp(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols * 2, src.rows * 2), src.type());

            GPU_ON;
            gpu::pyrUp(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 2000; size >= 1000; size -= 1000)
        {
            SUBTEST << "8UC3, size " << size;

            Mat src; gen(src, size, size, CV_8UC3, 0, 256);
            Mat dst(Size(src.cols * 2, src.rows * 2), src.type());

            CPU_ON;
            pyrUp(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols * 2, src.rows * 2), src.type());

            GPU_ON;
            gpu::pyrUp(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 2000; size >= 1000; size -= 1000)
        {
            SUBTEST << "8UC4, size " << size;

            Mat src; gen(src, size, size, CV_8UC4, 0, 256);
            Mat dst(Size(src.cols * 2, src.rows * 2), src.type());

            CPU_ON;
            pyrUp(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols * 2, src.rows * 2), src.type());

            GPU_ON;
            gpu::pyrUp(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 2000; size >= 1000; size -= 1000)
        {
            SUBTEST << "16SC3, size " << size;

            Mat src; gen(src, size, size, CV_16SC3, 0, 256);
            Mat dst(Size(src.cols * 2, src.rows * 2), src.type());

            CPU_ON;
            pyrUp(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols * 2, src.rows * 2), src.type());

            GPU_ON;
            gpu::pyrUp(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 2000; size >= 1000; size -= 1000)
        {
            SUBTEST << "32FC1, size " << size;

            Mat src; gen(src, size, size, CV_32FC1, 0, 256);
            Mat dst(Size(src.cols * 2, src.rows * 2), src.type());

            CPU_ON;
            pyrUp(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols * 2, src.rows * 2), src.type());

            GPU_ON;
            gpu::pyrUp(d_src, d_dst);
            GPU_OFF;
        }
    }
    {
        for (int size = 2000; size >= 1000; size -= 1000)
        {
            SUBTEST << "32FC3, size " << size;

            Mat src; gen(src, size, size, CV_32FC3, 0, 256);
            Mat dst(Size(src.cols * 2, src.rows * 2), src.type());

            CPU_ON;
            pyrUp(src, dst);
            CPU_OFF;

            gpu::GpuMat d_src(src);
            gpu::GpuMat d_dst(Size(src.cols * 2, src.rows * 2), src.type());

            GPU_ON;
            gpu::pyrUp(d_src, d_dst);
            GPU_OFF;
        }
    }
}


TEST(equalizeHist)
{
    for (int size = 1000; size < 4000; size += 1000)
    {
        SUBTEST << "size " << size;

        Mat src; gen(src, size, size, CV_8UC1, 0, 256);
        Mat dst(src.size(), src.type());

        CPU_ON;
        equalizeHist(src, dst);
        CPU_OFF;

        gpu::GpuMat d_src(src);
        gpu::GpuMat d_dst(src.size(), src.type());

        GPU_ON;
        gpu::equalizeHist(d_src, d_dst);
        GPU_OFF;
    }
}


TEST(Canny)
{
    Mat img = imread(abspath("aloeL.jpg"), CV_LOAD_IMAGE_GRAYSCALE);

    if (img.empty()) throw runtime_error("can't open aloeL.jpg");

    Mat edges(img.size(), CV_8UC1);

    CPU_ON;
    Canny(img, edges, 50.0, 100.0);
    CPU_OFF;
    
    gpu::GpuMat d_img(img);
    gpu::GpuMat d_edges(img.size(), CV_8UC1);
    gpu::CannyBuf d_buf(img.size());

    GPU_ON;
    gpu::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    GPU_OFF;
}


TEST(reduce)
{
    for (int size = 1000; size < 4000; size += 1000)
    {
        Mat src;
        gen(src, size, size, CV_32F, 0, 255);
        Mat dst0(1, src.cols, CV_32F);
        Mat dst1(src.rows, 1, CV_32F);

        gpu::GpuMat d_src(src);
        gpu::GpuMat d_dst0(1, src.cols, CV_32F);
        gpu::GpuMat d_dst1(1, src.rows, CV_32F);

        SUBTEST << "size " << size << ", dim = 0";

        CPU_ON;
        reduce(src, dst0, 0, CV_REDUCE_MIN);
        CPU_OFF;

        GPU_ON;
        gpu::reduce(d_src, d_dst0, 0, CV_REDUCE_MIN);
        GPU_OFF;

        SUBTEST << "size " << size << ", dim = 1";

        CPU_ON;
        reduce(src, dst1, 1, CV_REDUCE_MIN);
        CPU_OFF;

        GPU_ON;
        gpu::reduce(d_src, d_dst1, 1, CV_REDUCE_MIN);
        GPU_OFF;
    }
}


TEST(gemm)
{
    Mat src1, src2, src3, dst;
    gpu::GpuMat d_src1, d_src2, d_src3, d_dst;

    for (int size = 512; size <= 2048; size *= 2)
    {
        SUBTEST << "size " << size << ", 32FC1";

        gen(src1, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src2, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src3, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        dst.create(src1.size(), src1.type());

        CPU_ON;
        gemm(src1, src2, 1.0, src3, 1.0, dst);
        CPU_OFF;

        d_src1.upload(src1);
        d_src2.upload(src2);
        d_src3.upload(src3);
        d_dst.create(d_src1.size(), d_src1.type());

        GPU_ON;
        gpu::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        GPU_OFF;
    }
}
