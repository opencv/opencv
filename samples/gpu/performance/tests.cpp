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
    gpu::GpuMat d_src, d_sum, d_buf;

    int size = 4000;

    gen(src, size, size, CV_8U, 0, 256);
    sum.create(size + 1, size + 1, CV_32S);

    d_src = src;
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

        d_src = src;

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

        d_src = src;
        d_dst.create(d_src.size(), d_src.type());

        GPU_ON;
        gpu::meanShiftFiltering(d_src, d_dst, sp, sr);
        GPU_OFF;
    }
}


TEST(SURF)
{
    Mat src1 = imread(abspath("aloeL.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
    Mat src2 = imread(abspath("aloeR.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
    if (src1.empty()) throw runtime_error("can't open aloeL.jpg");
    if (src2.empty()) throw runtime_error("can't open aloeR.jpg");

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
    // Init CPU matcher

    int desc_len = 128;

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
    vector< vector<DMatch> > matches(1);
    vector< vector<DMatch> > d_matches(1);

    SUBTEST << "match";

    CPU_ON;
    matcher.match(query, train, matches[0]);
    CPU_OFF;

    GPU_ON;
    d_matcher.match(d_query, d_train, d_matches[0]);
    GPU_OFF;

    SUBTEST << "knnMatch";
    int knn = 10;

    CPU_ON;
    matcher.knnMatch(query, train, matches, knn);
    CPU_OFF;

    GPU_ON;
    d_matcher.knnMatch(d_query, d_train, d_matches, knn);
    GPU_OFF;

    SUBTEST << "radiusMatch";
    float max_distance = 3.8f;

    CPU_ON;
    matcher.radiusMatch(query, train, matches, max_distance);
    CPU_OFF;

    GPU_ON;
    d_matcher.radiusMatch(d_query, d_train, d_matches, max_distance);
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

        d_x = x;
        d_y = y;
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

        d_src1 = src1;
        d_src2 = src2;
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

        d_src = src;
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

        d_src = src;
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

        d_src1 = src1;
        d_src2 = src2;
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
        SUBTEST << "size " << size;

        gen(src, size, size, CV_8U, 0, 256);
        dst.create(size * 2, size * 2, CV_8U);

        CPU_ON;
        resize(src, dst, dst.size());
        CPU_OFF;

        d_src = src;
        d_dst.create(size * 2, size * 2, CV_8U);

        GPU_ON;
        gpu::resize(d_src, d_dst, d_dst.size());
        GPU_OFF;
    }
}


TEST(Sobel)
{
    Mat src, dst;
    gpu::GpuMat d_src, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32F";

        gen(src, size, size, CV_32F, 0, 1);
        dst.create(size, size, CV_32F);

        CPU_ON;
        Sobel(src, dst, dst.depth(), 1, 1);
        CPU_OFF;

        d_src = src;
        d_dst.create(size, size, CV_32F);

        GPU_ON;
        gpu::Sobel(d_src, d_dst, d_dst.depth(), 1, 1);
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

        d_src = src;
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

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8U, THRESH_TRUNC";

        gen(src, size, size, CV_8U, 0, 100);
        dst.create(size, size, CV_8U);

        CPU_ON; 
        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);
        CPU_OFF;

        d_src = src;
        d_dst.create(size, size, CV_8U);

        GPU_ON;
        gpu::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        GPU_OFF;
    }

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 8U, THRESH_BINARY";

        gen(src, size, size, CV_8U, 0, 100);
        dst.create(size, size, CV_8U);

        CPU_ON; 
        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);
        CPU_OFF;

        d_src = src;
        d_dst.create(size, size, CV_8U);

        GPU_ON;
        gpu::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        GPU_OFF;
    }

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << "size " << size << ", 32F, THRESH_TRUNC";

        gen(src, size, size, CV_32F, 0, 100);
        dst.create(size, size, CV_32F);

        CPU_ON; 
        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);
        CPU_OFF;

        d_src = src;
        d_dst.create(size, size, CV_32F);

        GPU_ON;
        gpu::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        GPU_OFF;
    }
}