#include <stdexcept>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video.hpp"
#include "opencv2/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudabgsegm.hpp"

#include "opencv2/legacy.hpp"
#include "performance.h"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_NONFREE
#include "opencv2/nonfree/cuda.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#endif

using namespace std;
using namespace cv;


TEST(matchTemplate)
{
    Mat src, templ, dst;
    gen(src, 3000, 3000, CV_32F, 0, 1);

    cuda::GpuMat d_src(src), d_templ, d_dst;

    Ptr<cuda::TemplateMatching> alg = cuda::createTemplateMatching(src.type(), TM_CCORR);

    for (int templ_size = 5; templ_size < 200; templ_size *= 5)
    {
        SUBTEST << src.cols << 'x' << src.rows << ", 32FC1" << ", templ " << templ_size << 'x' << templ_size << ", CCORR";

        gen(templ, templ_size, templ_size, CV_32F, 0, 1);
        matchTemplate(src, templ, dst, TM_CCORR);

        CPU_ON;
        matchTemplate(src, templ, dst, TM_CCORR);
        CPU_OFF;

        d_templ.upload(templ);
        alg->match(d_src, d_templ, d_dst);

        CUDA_ON;
        alg->match(d_src, d_templ, d_dst);
        CUDA_OFF;
    }
}


TEST(minMaxLoc)
{
    Mat src;
    cuda::GpuMat d_src;

    double min_val, max_val;
    Point min_loc, max_loc;

    for (int size = 2000; size <= 8000; size *= 2)
    {
        SUBTEST << size << 'x' << size << ", 32F";

        gen(src, size, size, CV_32F, 0, 1);

        CPU_ON;
        minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);
        CPU_OFF;

        d_src.upload(src);

        CUDA_ON;
        cuda::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
        CUDA_OFF;
    }
}


TEST(remap)
{
    Mat src, dst, xmap, ymap;
    cuda::GpuMat d_src, d_dst, d_xmap, d_ymap;

    int interpolation = INTER_LINEAR;
    int borderMode = BORDER_REPLICATE;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << ", 8UC4, INTER_LINEAR, BORDER_REPLICATE";

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

        remap(src, dst, xmap, ymap, interpolation, borderMode);

        CPU_ON;
        remap(src, dst, xmap, ymap, interpolation, borderMode);
        CPU_OFF;

        d_src.upload(src);
        d_xmap.upload(xmap);
        d_ymap.upload(ymap);

        cuda::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);

        CUDA_ON;
        cuda::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
        CUDA_OFF;
    }
}


TEST(dft)
{
    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << ", 32FC2, complex-to-complex";

        gen(src, size, size, CV_32FC2, Scalar::all(0), Scalar::all(1));

        dft(src, dst);

        CPU_ON;
        dft(src, dst);
        CPU_OFF;

        d_src.upload(src);

        cuda::dft(d_src, d_dst, Size(size, size));

        CUDA_ON;
        cuda::dft(d_src, d_dst, Size(size, size));
        CUDA_OFF;
    }
}


TEST(cornerHarris)
{
    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << ", 32FC1, BORDER_REFLECT101";

        gen(src, size, size, CV_32F, 0, 1);

        cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT101);

        CPU_ON;
        cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT101);
        CPU_OFF;

        d_src.upload(src);

        Ptr<cuda::CornernessCriteria> harris = cuda::createHarrisCorner(src.type(), 5, 7, 0.1, BORDER_REFLECT101);

        harris->compute(d_src, d_dst);

        CUDA_ON;
        harris->compute(d_src, d_dst);
        CUDA_OFF;
    }
}


TEST(integral)
{
    Mat src, sum;
    cuda::GpuMat d_src, d_sum, d_buf;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << ", 8UC1";

        gen(src, size, size, CV_8U, 0, 256);

        integral(src, sum);

        CPU_ON;
        integral(src, sum);
        CPU_OFF;

        d_src.upload(src);

        cuda::integralBuffered(d_src, d_sum, d_buf);

        CUDA_ON;
        cuda::integralBuffered(d_src, d_sum, d_buf);
        CUDA_OFF;
    }
}


TEST(norm)
{
    Mat src;
    cuda::GpuMat d_src, d_buf;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 32FC4, NORM_INF";

        gen(src, size, size, CV_32FC4, Scalar::all(0), Scalar::all(1));

        norm(src, NORM_INF);

        CPU_ON;
        norm(src, NORM_INF);
        CPU_OFF;

        d_src.upload(src);

        cuda::norm(d_src, NORM_INF, d_buf);

        CUDA_ON;
        cuda::norm(d_src, NORM_INF, d_buf);
        CUDA_OFF;
    }
}


TEST(meanShift)
{
    int sp = 10, sr = 10;

    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    for (int size = 400; size <= 800; size *= 2)
    {
        SUBTEST << size << 'x' << size << ", 8UC3 vs 8UC4";

        gen(src, size, size, CV_8UC3, Scalar::all(0), Scalar::all(256));

        pyrMeanShiftFiltering(src, dst, sp, sr);

        CPU_ON;
        pyrMeanShiftFiltering(src, dst, sp, sr);
        CPU_OFF;

        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));

        d_src.upload(src);

        cuda::meanShiftFiltering(d_src, d_dst, sp, sr);

        CUDA_ON;
        cuda::meanShiftFiltering(d_src, d_dst, sp, sr);
        CUDA_OFF;
    }
}

#ifdef HAVE_OPENCV_NONFREE

TEST(SURF)
{
    Mat src = imread(abspath("aloeL.jpg"), IMREAD_GRAYSCALE);
    if (src.empty()) throw runtime_error("can't open aloeL.jpg");

    SURF surf;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    surf(src, Mat(), keypoints, descriptors);

    CPU_ON;
    surf(src, Mat(), keypoints, descriptors);
    CPU_OFF;

    cuda::SURF_CUDA d_surf;
    cuda::GpuMat d_src(src);
    cuda::GpuMat d_keypoints;
    cuda::GpuMat d_descriptors;

    d_surf(d_src, cuda::GpuMat(), d_keypoints, d_descriptors);

    CUDA_ON;
    d_surf(d_src, cuda::GpuMat(), d_keypoints, d_descriptors);
    CUDA_OFF;
}

#endif


TEST(FAST)
{
    Mat src = imread(abspath("aloeL.jpg"), IMREAD_GRAYSCALE);
    if (src.empty()) throw runtime_error("can't open aloeL.jpg");

    vector<KeyPoint> keypoints;

    FAST(src, keypoints, 20);

    CPU_ON;
    FAST(src, keypoints, 20);
    CPU_OFF;

    cuda::FAST_CUDA d_FAST(20);
    cuda::GpuMat d_src(src);
    cuda::GpuMat d_keypoints;

    d_FAST(d_src, cuda::GpuMat(), d_keypoints);

    CUDA_ON;
    d_FAST(d_src, cuda::GpuMat(), d_keypoints);
    CUDA_OFF;
}


TEST(ORB)
{
    Mat src = imread(abspath("aloeL.jpg"), IMREAD_GRAYSCALE);
    if (src.empty()) throw runtime_error("can't open aloeL.jpg");

    ORB orb(4000);
    vector<KeyPoint> keypoints;
    Mat descriptors;

    orb(src, Mat(), keypoints, descriptors);

    CPU_ON;
    orb(src, Mat(), keypoints, descriptors);
    CPU_OFF;

    cuda::ORB_CUDA d_orb;
    cuda::GpuMat d_src(src);
    cuda::GpuMat d_keypoints;
    cuda::GpuMat d_descriptors;

    d_orb(d_src, cuda::GpuMat(), d_keypoints, d_descriptors);

    CUDA_ON;
    d_orb(d_src, cuda::GpuMat(), d_keypoints, d_descriptors);
    CUDA_OFF;
}


TEST(BruteForceMatcher)
{
    // Init CPU matcher

    int desc_len = 64;

    BFMatcher matcher(NORM_L2);

    Mat query;
    gen(query, 3000, desc_len, CV_32F, 0, 1);

    Mat train;
    gen(train, 3000, desc_len, CV_32F, 0, 1);

    // Init CUDA matcher

    cuda::BFMatcher_CUDA d_matcher(NORM_L2);

    cuda::GpuMat d_query(query);
    cuda::GpuMat d_train(train);

    // Output
    vector< vector<DMatch> > matches(2);
    cuda::GpuMat d_trainIdx, d_distance, d_allDist, d_nMatches;

    SUBTEST << "match";

    matcher.match(query, train, matches[0]);

    CPU_ON;
    matcher.match(query, train, matches[0]);
    CPU_OFF;

    d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);

    CUDA_ON;
    d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
    CUDA_OFF;

    SUBTEST << "knnMatch";

    matcher.knnMatch(query, train, matches, 2);

    CPU_ON;
    matcher.knnMatch(query, train, matches, 2);
    CPU_OFF;

    d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);

    CUDA_ON;
    d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);
    CUDA_OFF;

    SUBTEST << "radiusMatch";

    float max_distance = 2.0f;

    matcher.radiusMatch(query, train, matches, max_distance);

    CPU_ON;
    matcher.radiusMatch(query, train, matches, max_distance);
    CPU_OFF;

    d_trainIdx.release();

    d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);

    CUDA_ON;
    d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);
    CUDA_OFF;
}


TEST(magnitude)
{
    Mat x, y, mag;
    cuda::GpuMat d_x, d_y, d_mag;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 32FC1";

        gen(x, size, size, CV_32F, 0, 1);
        gen(y, size, size, CV_32F, 0, 1);

        magnitude(x, y, mag);

        CPU_ON;
        magnitude(x, y, mag);
        CPU_OFF;

        d_x.upload(x);
        d_y.upload(y);

        cuda::magnitude(d_x, d_y, d_mag);

        CUDA_ON;
        cuda::magnitude(d_x, d_y, d_mag);
        CUDA_OFF;
    }
}


TEST(add)
{
    Mat src1, src2, dst;
    cuda::GpuMat d_src1, d_src2, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 32FC1";

        gen(src1, size, size, CV_32F, 0, 1);
        gen(src2, size, size, CV_32F, 0, 1);

        add(src1, src2, dst);

        CPU_ON;
        add(src1, src2, dst);
        CPU_OFF;

        d_src1.upload(src1);
        d_src2.upload(src2);

        cuda::add(d_src1, d_src2, d_dst);

        CUDA_ON;
        cuda::add(d_src1, d_src2, d_dst);
        CUDA_OFF;
    }
}


TEST(log)
{
    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 32F";

        gen(src, size, size, CV_32F, 1, 10);

        log(src, dst);

        CPU_ON;
        log(src, dst);
        CPU_OFF;

        d_src.upload(src);

        cuda::log(d_src, d_dst);

        CUDA_ON;
        cuda::log(d_src, d_dst);
        CUDA_OFF;
    }
}


TEST(mulSpectrums)
{
    Mat src1, src2, dst;
    cuda::GpuMat d_src1, d_src2, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size;

        gen(src1, size, size, CV_32FC2, Scalar::all(0), Scalar::all(1));
        gen(src2, size, size, CV_32FC2, Scalar::all(0), Scalar::all(1));

        mulSpectrums(src1, src2, dst, 0, true);

        CPU_ON;
        mulSpectrums(src1, src2, dst, 0, true);
        CPU_OFF;

        d_src1.upload(src1);
        d_src2.upload(src2);

        cuda::mulSpectrums(d_src1, d_src2, d_dst, 0, true);

        CUDA_ON;
        cuda::mulSpectrums(d_src1, d_src2, d_dst, 0, true);
        CUDA_OFF;
    }
}


TEST(resize)
{
    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 8UC4, up";

        gen(src, size, size, CV_8UC4, 0, 256);

        resize(src, dst, Size(), 2.0, 2.0);

        CPU_ON;
        resize(src, dst, Size(), 2.0, 2.0);
        CPU_OFF;

        d_src.upload(src);

        cuda::resize(d_src, d_dst, Size(), 2.0, 2.0);

        CUDA_ON;
        cuda::resize(d_src, d_dst, Size(), 2.0, 2.0);
        CUDA_OFF;
    }

    for (int size = 1000; size <= 3000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 8UC4, down";

        gen(src, size, size, CV_8UC4, 0, 256);

        resize(src, dst, Size(), 0.5, 0.5);

        CPU_ON;
        resize(src, dst, Size(), 0.5, 0.5);
        CPU_OFF;

        d_src.upload(src);

        cuda::resize(d_src, d_dst, Size(), 0.5, 0.5);

        CUDA_ON;
        cuda::resize(d_src, d_dst, Size(), 0.5, 0.5);
        CUDA_OFF;
    }
}


TEST(cvtColor)
{
    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    gen(src, 4000, 4000, CV_8UC1, 0, 255);
    d_src.upload(src);

    SUBTEST << "4000x4000, 8UC1, COLOR_GRAY2BGRA";

    cvtColor(src, dst, COLOR_GRAY2BGRA, 4);

    CPU_ON;
    cvtColor(src, dst, COLOR_GRAY2BGRA, 4);
    CPU_OFF;

    cuda::cvtColor(d_src, d_dst, COLOR_GRAY2BGRA, 4);

    CUDA_ON;
    cuda::cvtColor(d_src, d_dst, COLOR_GRAY2BGRA, 4);
    CUDA_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "4000x4000, 8UC3 vs 8UC4, COLOR_BGR2YCrCb";

    cvtColor(src, dst, COLOR_BGR2YCrCb);

    CPU_ON;
    cvtColor(src, dst, COLOR_BGR2YCrCb);
    CPU_OFF;

    cuda::cvtColor(d_src, d_dst, COLOR_BGR2YCrCb, 4);

    CUDA_ON;
    cuda::cvtColor(d_src, d_dst, COLOR_BGR2YCrCb, 4);
    CUDA_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "4000x4000, 8UC4, COLOR_YCrCb2BGR";

    cvtColor(src, dst, COLOR_YCrCb2BGR, 4);

    CPU_ON;
    cvtColor(src, dst, COLOR_YCrCb2BGR, 4);
    CPU_OFF;

    cuda::cvtColor(d_src, d_dst, COLOR_YCrCb2BGR, 4);

    CUDA_ON;
    cuda::cvtColor(d_src, d_dst, COLOR_YCrCb2BGR, 4);
    CUDA_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "4000x4000, 8UC3 vs 8UC4, COLOR_BGR2XYZ";

    cvtColor(src, dst, COLOR_BGR2XYZ);

    CPU_ON;
    cvtColor(src, dst, COLOR_BGR2XYZ);
    CPU_OFF;

    cuda::cvtColor(d_src, d_dst, COLOR_BGR2XYZ, 4);

    CUDA_ON;
    cuda::cvtColor(d_src, d_dst, COLOR_BGR2XYZ, 4);
    CUDA_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "4000x4000, 8UC4, COLOR_XYZ2BGR";

    cvtColor(src, dst, COLOR_XYZ2BGR, 4);

    CPU_ON;
    cvtColor(src, dst, COLOR_XYZ2BGR, 4);
    CPU_OFF;

    cuda::cvtColor(d_src, d_dst, COLOR_XYZ2BGR, 4);

    CUDA_ON;
    cuda::cvtColor(d_src, d_dst, COLOR_XYZ2BGR, 4);
    CUDA_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "4000x4000, 8UC3 vs 8UC4, COLOR_BGR2HSV";

    cvtColor(src, dst, COLOR_BGR2HSV);

    CPU_ON;
    cvtColor(src, dst, COLOR_BGR2HSV);
    CPU_OFF;

    cuda::cvtColor(d_src, d_dst, COLOR_BGR2HSV, 4);

    CUDA_ON;
    cuda::cvtColor(d_src, d_dst, COLOR_BGR2HSV, 4);
    CUDA_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);

    SUBTEST << "4000x4000, 8UC4, COLOR_HSV2BGR";

    cvtColor(src, dst, COLOR_HSV2BGR, 4);

    CPU_ON;
    cvtColor(src, dst, COLOR_HSV2BGR, 4);
    CPU_OFF;

    cuda::cvtColor(d_src, d_dst, COLOR_HSV2BGR, 4);

    CUDA_ON;
    cuda::cvtColor(d_src, d_dst, COLOR_HSV2BGR, 4);
    CUDA_OFF;

    cv::swap(src, dst);
    d_src.swap(d_dst);
}


TEST(erode)
{
    Mat src, dst, ker;
    cuda::GpuMat d_src, d_buf, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size;

        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));
        ker = getStructuringElement(MORPH_RECT, Size(3, 3));

        erode(src, dst, ker);

        CPU_ON;
        erode(src, dst, ker);
        CPU_OFF;

        d_src.upload(src);

        Ptr<cuda::Filter> erode = cuda::createMorphologyFilter(MORPH_ERODE, d_src.type(), ker);

        erode->apply(d_src, d_dst);

        CUDA_ON;
        erode->apply(d_src, d_dst);
        CUDA_OFF;
    }
}

TEST(threshold)
{
    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 8UC1, THRESH_BINARY";

        gen(src, size, size, CV_8U, 0, 100);

        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);

        CPU_ON;
        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);
        CPU_OFF;

        d_src.upload(src);

        cuda::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);

        CUDA_ON;
        cuda::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        CUDA_OFF;
    }

    for (int size = 2000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 32FC1, THRESH_TRUNC [NPP]";

        gen(src, size, size, CV_32FC1, 0, 100);

        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);

        CPU_ON;
        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);
        CPU_OFF;

        d_src.upload(src);

        cuda::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);

        CUDA_ON;
        cuda::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        CUDA_OFF;
    }
}

TEST(pow)
{
    Mat src, dst;
    cuda::GpuMat d_src, d_dst;

    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 32F";

        gen(src, size, size, CV_32F, 0, 100);

        pow(src, -2.0, dst);

        CPU_ON;
        pow(src, -2.0, dst);
        CPU_OFF;

        d_src.upload(src);

        cuda::pow(d_src, -2.0, d_dst);

        CUDA_ON;
        cuda::pow(d_src, -2.0, d_dst);
        CUDA_OFF;
    }
}


TEST(projectPoints)
{
    Mat src;
    vector<Point2f> dst;
    cuda::GpuMat d_src, d_dst;

    Mat rvec; gen(rvec, 1, 3, CV_32F, 0, 1);
    Mat tvec; gen(tvec, 1, 3, CV_32F, 0, 1);
    Mat camera_mat; gen(camera_mat, 3, 3, CV_32F, 0, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    for (int size = (int)1e6, count = 0; size >= 1e5 && count < 5; size = int(size / 1.4), count++)
    {
        SUBTEST << size;

        gen(src, 1, size, CV_32FC3, Scalar::all(0), Scalar::all(10));

        projectPoints(src, rvec, tvec, camera_mat, Mat::zeros(1, 8, CV_32F), dst);

        CPU_ON;
        projectPoints(src, rvec, tvec, camera_mat, Mat::zeros(1, 8, CV_32F), dst);
        CPU_OFF;

        d_src.upload(src);

        cuda::projectPoints(d_src, rvec, tvec, camera_mat, Mat(), d_dst);

        CUDA_ON;
        cuda::projectPoints(d_src, rvec, tvec, camera_mat, Mat(), d_dst);
        CUDA_OFF;
    }
}


static void InitSolvePnpRansac()
{
    Mat object; gen(object, 1, 4, CV_32FC3, Scalar::all(0), Scalar::all(100));
    Mat image; gen(image, 1, 4, CV_32FC2, Scalar::all(0), Scalar::all(100));
    Mat rvec, tvec;
    cuda::solvePnPRansac(object, image, Mat::eye(3, 3, CV_32F), Mat(), rvec, tvec);
}


TEST(solvePnPRansac)
{
    InitSolvePnpRansac();

    for (int num_points = 5000; num_points <= 300000; num_points = int(num_points * 3.76))
    {
        SUBTEST << num_points;

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

        CUDA_ON;
        cuda::solvePnPRansac(object, image, camera_mat, Mat::zeros(1, 8, CV_32F), rvec, tvec, false, num_iters,
                            max_dist, int(num_points * 0.05), &inliers_gpu);
        CUDA_OFF;
    }
}

TEST(GaussianBlur)
{
    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 8UC4";

        Mat src, dst;

        gen(src, size, size, CV_8UC4, 0, 256);

        GaussianBlur(src, dst, Size(3, 3), 1);

        CPU_ON;
        GaussianBlur(src, dst, Size(3, 3), 1);
        CPU_OFF;

        cuda::GpuMat d_src(src);
        cuda::GpuMat d_dst(src.size(), src.type());
        cuda::GpuMat d_buf;

        cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(d_src.type(), -1, cv::Size(3, 3), 1);

        gauss->apply(d_src, d_dst);

        CUDA_ON;
        gauss->apply(d_src, d_dst);
        CUDA_OFF;
    }
}

TEST(filter2D)
{
    for (int size = 512; size <= 2048; size *= 2)
    {
        Mat src;
        gen(src, size, size, CV_8UC4, 0, 256);

        for (int ksize = 3; ksize <= 16; ksize += 2)
        {
            SUBTEST << "ksize = " << ksize << ", " << size << 'x' << size << ", 8UC4";

            Mat kernel;
            gen(kernel, ksize, ksize, CV_32FC1, 0.0, 1.0);

            Mat dst;
            cv::filter2D(src, dst, -1, kernel);

            CPU_ON;
            cv::filter2D(src, dst, -1, kernel);
            CPU_OFF;

            cuda::GpuMat d_src(src);
            cuda::GpuMat d_dst;

            Ptr<cuda::Filter> filter2D = cuda::createLinearFilter(d_src.type(), -1, kernel);
            filter2D->apply(d_src, d_dst);

            CUDA_ON;
            filter2D->apply(d_src, d_dst);
            CUDA_OFF;
        }
    }
}

TEST(pyrDown)
{
    for (int size = 4000; size >= 1000; size -= 1000)
    {
        SUBTEST << size << 'x' << size << ", 8UC4";

        Mat src, dst;
        gen(src, size, size, CV_8UC4, 0, 256);

        pyrDown(src, dst);

        CPU_ON;
        pyrDown(src, dst);
        CPU_OFF;

        cuda::GpuMat d_src(src);
        cuda::GpuMat d_dst;

        cuda::pyrDown(d_src, d_dst);

        CUDA_ON;
        cuda::pyrDown(d_src, d_dst);
        CUDA_OFF;
    }
}

TEST(pyrUp)
{
    for (int size = 2000; size >= 1000; size -= 1000)
    {
        SUBTEST << size << 'x' << size << ", 8UC4";

        Mat src, dst;

        gen(src, size, size, CV_8UC4, 0, 256);

        pyrUp(src, dst);

        CPU_ON;
        pyrUp(src, dst);
        CPU_OFF;

        cuda::GpuMat d_src(src);
        cuda::GpuMat d_dst;

        cuda::pyrUp(d_src, d_dst);

        CUDA_ON;
        cuda::pyrUp(d_src, d_dst);
        CUDA_OFF;
    }
}


TEST(equalizeHist)
{
    for (int size = 1000; size < 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size;

        Mat src, dst;

        gen(src, size, size, CV_8UC1, 0, 256);

        equalizeHist(src, dst);

        CPU_ON;
        equalizeHist(src, dst);
        CPU_OFF;

        cuda::GpuMat d_src(src);
        cuda::GpuMat d_dst;
        cuda::GpuMat d_buf;

        cuda::equalizeHist(d_src, d_dst, d_buf);

        CUDA_ON;
        cuda::equalizeHist(d_src, d_dst, d_buf);
        CUDA_OFF;
    }
}


TEST(Canny)
{
    Mat img = imread(abspath("aloeL.jpg"), IMREAD_GRAYSCALE);

    if (img.empty()) throw runtime_error("can't open aloeL.jpg");

    Mat edges(img.size(), CV_8UC1);

    CPU_ON;
    Canny(img, edges, 50.0, 100.0);
    CPU_OFF;

    cuda::GpuMat d_img(img);
    cuda::GpuMat d_edges;

    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(50.0, 100.0);

    canny->detect(d_img, d_edges);

    CUDA_ON;
    canny->detect(d_img, d_edges);
    CUDA_OFF;
}


TEST(reduce)
{
    for (int size = 1000; size < 4000; size += 1000)
    {
        Mat src;
        gen(src, size, size, CV_32F, 0, 255);

        Mat dst0;
        Mat dst1;

        cuda::GpuMat d_src(src);
        cuda::GpuMat d_dst0;
        cuda::GpuMat d_dst1;

        SUBTEST << size << 'x' << size << ", dim = 0";

        reduce(src, dst0, 0, REDUCE_MIN);

        CPU_ON;
        reduce(src, dst0, 0, REDUCE_MIN);
        CPU_OFF;

        cuda::reduce(d_src, d_dst0, 0, REDUCE_MIN);

        CUDA_ON;
        cuda::reduce(d_src, d_dst0, 0, REDUCE_MIN);
        CUDA_OFF;

        SUBTEST << size << 'x' << size << ", dim = 1";

        reduce(src, dst1, 1, REDUCE_MIN);

        CPU_ON;
        reduce(src, dst1, 1, REDUCE_MIN);
        CPU_OFF;

        cuda::reduce(d_src, d_dst1, 1, REDUCE_MIN);

        CUDA_ON;
        cuda::reduce(d_src, d_dst1, 1, REDUCE_MIN);
        CUDA_OFF;
    }
}


TEST(gemm)
{
    Mat src1, src2, src3, dst;
    cuda::GpuMat d_src1, d_src2, d_src3, d_dst;

    for (int size = 512; size <= 1024; size *= 2)
    {
        SUBTEST << size << 'x' << size;

        gen(src1, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src2, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src3, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));

        gemm(src1, src2, 1.0, src3, 1.0, dst);

        CPU_ON;
        gemm(src1, src2, 1.0, src3, 1.0, dst);
        CPU_OFF;

        d_src1.upload(src1);
        d_src2.upload(src2);
        d_src3.upload(src3);

        cuda::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);

        CUDA_ON;
        cuda::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        CUDA_OFF;
    }
}

TEST(GoodFeaturesToTrack)
{
    Mat src = imread(abspath("aloeL.jpg"), IMREAD_GRAYSCALE);
    if (src.empty()) throw runtime_error("can't open aloeL.jpg");

    vector<Point2f> pts;

    goodFeaturesToTrack(src, pts, 8000, 0.01, 0.0);

    CPU_ON;
    goodFeaturesToTrack(src, pts, 8000, 0.01, 0.0);
    CPU_OFF;

    Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(src.type(), 8000, 0.01, 0.0);

    cuda::GpuMat d_src(src);
    cuda::GpuMat d_pts;

    detector->detect(d_src, d_pts);

    CUDA_ON;
    detector->detect(d_src, d_pts);
    CUDA_OFF;
}

TEST(PyrLKOpticalFlow)
{
    Mat frame0 = imread(abspath("rubberwhale1.png"));
    if (frame0.empty()) throw runtime_error("can't open rubberwhale1.png");

    Mat frame1 = imread(abspath("rubberwhale2.png"));
    if (frame1.empty()) throw runtime_error("can't open rubberwhale2.png");

    Mat gray_frame;
    cvtColor(frame0, gray_frame, COLOR_BGR2GRAY);

    for (int points = 1000; points <= 8000; points *= 2)
    {
        SUBTEST << points;

        vector<Point2f> pts;
        goodFeaturesToTrack(gray_frame, pts, points, 0.01, 0.0);

        vector<Point2f> nextPts;
        vector<unsigned char> status;

        vector<float> err;

        calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);

        CPU_ON;
        calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);
        CPU_OFF;

        cuda::PyrLKOpticalFlow d_pyrLK;

        cuda::GpuMat d_frame0(frame0);
        cuda::GpuMat d_frame1(frame1);

        cuda::GpuMat d_pts;
        Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void*)&pts[0]);
        d_pts.upload(pts_mat);

        cuda::GpuMat d_nextPts;
        cuda::GpuMat d_status;
        cuda::GpuMat d_err;

        d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);

        CUDA_ON;
        d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);
        CUDA_OFF;
    }
}


TEST(FarnebackOpticalFlow)
{
    const string datasets[] = {"rubberwhale", "basketball"};
    for (size_t i = 0; i < sizeof(datasets)/sizeof(*datasets); ++i) {
    for (int fastPyramids = 0; fastPyramids < 2; ++fastPyramids) {
    for (int useGaussianBlur = 0; useGaussianBlur < 2; ++useGaussianBlur) {

    SUBTEST << "dataset=" << datasets[i] << ", fastPyramids=" << fastPyramids << ", useGaussianBlur=" << useGaussianBlur;
    Mat frame0 = imread(abspath(datasets[i] + "1.png"), IMREAD_GRAYSCALE);
    Mat frame1 = imread(abspath(datasets[i] + "2.png"), IMREAD_GRAYSCALE);
    if (frame0.empty()) throw runtime_error("can't open " + datasets[i] + "1.png");
    if (frame1.empty()) throw runtime_error("can't open " + datasets[i] + "2.png");

    cuda::FarnebackOpticalFlow calc;
    calc.fastPyramids = fastPyramids != 0;
    calc.flags |= useGaussianBlur ? OPTFLOW_FARNEBACK_GAUSSIAN : 0;

    cuda::GpuMat d_frame0(frame0), d_frame1(frame1), d_flowx, d_flowy;
    CUDA_ON;
    calc(d_frame0, d_frame1, d_flowx, d_flowy);
    CUDA_OFF;

    Mat flow;
    CPU_ON;
    calcOpticalFlowFarneback(frame0, frame1, flow, calc.pyrScale, calc.numLevels, calc.winSize, calc.numIters, calc.polyN, calc.polySigma, calc.flags);
    CPU_OFF;

    }}}
}

namespace cv
{
    template<> void DefaultDeleter<CvBGStatModel>::operator ()(CvBGStatModel* obj) const
    {
        cvReleaseBGStatModel(&obj);
    }
}

TEST(FGDStatModel)
{
    const std::string inputFile = abspath("768x576.avi");

    VideoCapture cap(inputFile);
    if (!cap.isOpened()) throw runtime_error("can't open 768x576.avi");

    Mat frame;
    cap >> frame;

    IplImage ipl_frame = frame;
    Ptr<CvBGStatModel> model(cvCreateFGDStatModel(&ipl_frame));

    while (!TestSystem::instance().stop())
    {
        cap >> frame;
        ipl_frame = frame;

        TestSystem::instance().cpuOn();

        cvUpdateBGStatModel(&ipl_frame, model);

        TestSystem::instance().cpuOff();
    }
    TestSystem::instance().cpuComplete();

    cap.open(inputFile);

    cap >> frame;

    cuda::GpuMat d_frame(frame), d_fgmask;
    Ptr<BackgroundSubtractor> d_fgd = cuda::createBackgroundSubtractorFGD();

    d_fgd->apply(d_frame, d_fgmask);

    while (!TestSystem::instance().stop())
    {
        cap >> frame;
        d_frame.upload(frame);

        TestSystem::instance().gpuOn();

        d_fgd->apply(d_frame, d_fgmask);

        TestSystem::instance().gpuOff();
    }
    TestSystem::instance().gpuComplete();
}

TEST(MOG)
{
    const std::string inputFile = abspath("768x576.avi");

    cv::VideoCapture cap(inputFile);
    if (!cap.isOpened()) throw runtime_error("can't open 768x576.avi");

    cv::Mat frame;
    cap >> frame;

    cv::Ptr<cv::BackgroundSubtractor> mog = cv::createBackgroundSubtractorMOG();
    cv::Mat foreground;

    mog->apply(frame, foreground, 0.01);

    while (!TestSystem::instance().stop())
    {
        cap >> frame;

        TestSystem::instance().cpuOn();

        mog->apply(frame, foreground, 0.01);

        TestSystem::instance().cpuOff();
    }
    TestSystem::instance().cpuComplete();

    cap.open(inputFile);

    cap >> frame;

    cv::cuda::GpuMat d_frame(frame);
    cv::Ptr<cv::BackgroundSubtractor> d_mog = cv::cuda::createBackgroundSubtractorMOG();
    cv::cuda::GpuMat d_foreground;

    d_mog->apply(d_frame, d_foreground, 0.01);

    while (!TestSystem::instance().stop())
    {
        cap >> frame;
        d_frame.upload(frame);

        TestSystem::instance().gpuOn();

        d_mog->apply(d_frame, d_foreground, 0.01);

        TestSystem::instance().gpuOff();
    }
    TestSystem::instance().gpuComplete();
}

TEST(MOG2)
{
    const std::string inputFile = abspath("768x576.avi");

    cv::VideoCapture cap(inputFile);
    if (!cap.isOpened()) throw runtime_error("can't open 768x576.avi");

    cv::Mat frame;
    cap >> frame;

    cv::Ptr<cv::BackgroundSubtractor> mog2 = cv::createBackgroundSubtractorMOG2();
    cv::Mat foreground;
    cv::Mat background;

    mog2->apply(frame, foreground);
    mog2->getBackgroundImage(background);

    while (!TestSystem::instance().stop())
    {
        cap >> frame;

        TestSystem::instance().cpuOn();

        mog2->apply(frame, foreground);
        mog2->getBackgroundImage(background);

        TestSystem::instance().cpuOff();
    }
    TestSystem::instance().cpuComplete();

    cap.open(inputFile);

    cap >> frame;

    cv::Ptr<cv::BackgroundSubtractor> d_mog2 = cv::cuda::createBackgroundSubtractorMOG2();
    cv::cuda::GpuMat d_frame(frame);
    cv::cuda::GpuMat d_foreground;
    cv::cuda::GpuMat d_background;

    d_mog2->apply(d_frame, d_foreground);
    d_mog2->getBackgroundImage(d_background);

    while (!TestSystem::instance().stop())
    {
        cap >> frame;
        d_frame.upload(frame);

        TestSystem::instance().gpuOn();

        d_mog2->apply(d_frame, d_foreground);
        d_mog2->getBackgroundImage(d_background);

        TestSystem::instance().gpuOff();
    }
    TestSystem::instance().gpuComplete();
}
