// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

struct Vec2fComparator
{
    bool operator()(const Vec2f& a, const Vec2f b) const
    {
        if(a[0] != b[0]) return a[0] < b[0];
        else return a[1] < b[1];
    }
};

/////////////////////////////// HoughLines ////////////////////////////////////

PARAM_TEST_CASE(HoughLines, double, double, int)
{
    double rhoStep, thetaStep;
    int threshold;

    Size src_size;
    Mat src, dst;
    UMat usrc, udst;

    virtual void SetUp()
    {
        rhoStep = GET_PARAM(0);
        thetaStep = GET_PARAM(1);
        threshold = GET_PARAM(2);
    }

    void generateTestData()
    {
        src_size = randomSize(500, 1920);
        src.create(src_size, CV_8UC1);
        src.setTo(Scalar::all(0));
        line(src, Point(0, 100), Point(100, 100), Scalar::all(255), 1);
        line(src, Point(0, 200), Point(100, 200), Scalar::all(255), 1);
        line(src, Point(0, 400), Point(100, 400), Scalar::all(255), 1);
        line(src, Point(100, 0), Point(100, 200), Scalar::all(255), 1);
        line(src, Point(200, 0), Point(200, 200), Scalar::all(255), 1);
        line(src, Point(400, 0), Point(400, 200), Scalar::all(255), 1);

        src.copyTo(usrc);
    }

    void readRealTestData()
    {
        Mat img = readImage("shared/pic5.png", IMREAD_GRAYSCALE);
        Canny(img, src, 100, 150, 3);

        src.copyTo(usrc);
    }

    void Near(double eps = 0.)
    {
        EXPECT_EQ(dst.size(), udst.size());

        if (dst.total() > 0)
        {
            Mat lines_cpu, lines_gpu;
            dst.copyTo(lines_cpu);
            udst.copyTo(lines_gpu);

            std::sort(lines_cpu.begin<Vec2f>(), lines_cpu.end<Vec2f>(), Vec2fComparator());
            std::sort(lines_gpu.begin<Vec2f>(), lines_gpu.end<Vec2f>(), Vec2fComparator());

            EXPECT_LE(TestUtils::checkNorm2(lines_cpu, lines_gpu), eps);
        }
    }
};

OCL_TEST_P(HoughLines, RealImage)
{
    readRealTestData();

    OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold));
    OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold));

    Near(1e-5);
}

OCL_TEST_P(HoughLines, GeneratedImage)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold));
        OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold));

        Near(1e-5);
    }
}

/////////////////////////////// HoughLinesP ///////////////////////////////////

PARAM_TEST_CASE(HoughLinesP, int, double, double)
{
    double rhoStep, thetaStep, minLineLength, maxGap;
    int threshold;

    Size src_size;
    Mat src, dst;
    UMat usrc, udst;

    virtual void SetUp()
    {
        rhoStep = 1.0;
        thetaStep = CV_PI / 180;
        threshold = GET_PARAM(0);
        minLineLength = GET_PARAM(1);
        maxGap = GET_PARAM(2);
    }

    void readRealTestData()
    {
        Mat img = readImage("shared/pic5.png", IMREAD_GRAYSCALE);
        Canny(img, src, 50, 200, 3);

        src.copyTo(usrc);
    }

    void Near(double eps = 0.)
    {
        Mat lines_gpu = udst.getMat(ACCESS_READ);

        if (dst.total() > 0 && lines_gpu.total() > 0)
        {
            Mat result_cpu(src.size(), CV_8UC1, Scalar::all(0));
            Mat result_gpu(src.size(), CV_8UC1, Scalar::all(0));

            MatConstIterator_<Vec4i> it = dst.begin<Vec4i>(), end = dst.end<Vec4i>();
            for ( ; it != end; it++)
            {
                Vec4i p = *it;
                line(result_cpu, Point(p[0], p[1]), Point(p[2], p[3]), Scalar(255));
            }

            it = lines_gpu.begin<Vec4i>(), end = lines_gpu.end<Vec4i>();
            for ( ; it != end; it++)
            {
                Vec4i p = *it;
                line(result_gpu, Point(p[0], p[1]), Point(p[2], p[3]), Scalar(255));
            }

            EXPECT_MAT_SIMILAR(result_cpu, result_gpu, eps);
        }
    }
};


OCL_TEST_P(HoughLinesP, RealImage)
{
    readRealTestData();

    OCL_OFF(cv::HoughLinesP(src, dst, rhoStep, thetaStep, threshold, minLineLength, maxGap));
    OCL_ON(cv::HoughLinesP(usrc, udst, rhoStep, thetaStep, threshold, minLineLength, maxGap));

    Near(0.25);
}

//////////////////////// HoughLinesUsingSetOfPoints ///////////////////////////
PARAM_TEST_CASE(HoughLinesUsingSetOfPoints, int, double, double, double, double, int)
{
    int point_cnt, polar_cnt;
    Point2f *point = NULL;
    HoughLinePolar *hough_polar = NULL;
    HoughDetectParam rho_param, theta_param;

    Size src_size;
    Mat src, dst;
    UMat usrc, udst;

    virtual void SetUp()
    {
        point_cnt = GET_PARAM(0);
        rho_param.min = GET_PARAM(1);
        rho_param.max = GET_PARAM(2);
        rho_param.step = (rho_param.max - rho_param.min) / 180.0f;
        theta_param.min = GET_PARAM(3);
        theta_param.max = GET_PARAM(4);
        theta_param.step = (theta_param.max - theta_param.min) / 256.0f;
        polar_cnt = GET_PARAM(5);
    }

    void detectEdge()
    {
        float edge = 0.0f;
        int point_cnt_temp = 0;
        Point2f *point_work = point;
        for (int y = 0; y < src.rows; y++)
        {
            for (int x = 0; x < src.cols; x++)
            {
                edge = src.at<float>(y, x);
                if ((point_cnt_temp < point_cnt) && (edge > 150))
                {
                    point_work->x = (float)x;
                    point_work->y = (float)y;
                    point_cnt_temp++;
                    // Pointer increment
                    point_work++;
                }
            }
        }
    }

    void mallocPoints()
    {
        if (point_cnt > 0)
        {
            point = (Point2f*)fastMalloc(sizeof(Point2f) * point_cnt);
        }
        else
        {
            point = NULL;
        }
    }

    void freePoints()
    {
        if (point != NULL)
        {
            fastFree(point);
        }
    }

    void mallocHoughPolar()
    {
        if (polar_cnt > 0)
        {
            hough_polar = (HoughLinePolar*)fastMalloc(sizeof(HoughLinePolar) * polar_cnt);
        }
        else
        {
            hough_polar = NULL;
        }
    }

    void freeHoughPolar()
    {
        if (hough_polar != NULL)
        {
            fastFree(hough_polar);
        }
    }

    void readTestData()
    {
        Mat img = readImage("shared/pic5.png", IMREAD_GRAYSCALE);
        Canny(img, src, 50, 200, 3);
        src.copyTo(usrc);
    }
};

OCL_TEST_P(HoughLinesUsingSetOfPoints, RealImage)
{
    readTestData();

    mallocPoints();

    mallocHoughPolar();

    detectEdge();

    OCL_OFF(cv::HoughLinesUsingSetOfPoints(point_cnt, point, &rho_param, &theta_param, polar_cnt, hough_polar));
    OCL_ON(cv::HoughLinesUsingSetOfPoints(point_cnt, point, &rho_param, &theta_param, polar_cnt, hough_polar));


    freePoints();
    freeHoughPolar();
}

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, HoughLines, Combine(Values(1, 0.5),                        // rhoStep
                                                         Values(CV_PI / 180.0, CV_PI / 360.0),  // thetaStep
                                                         Values(80, 150)));                     // threshold

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, HoughLinesP, Combine(Values(100, 150),                     // threshold
                                                          Values(50, 100),                      // minLineLength
                                                          Values(5, 10)));                      // maxLineGap

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, HoughLinesUsingSetOfPoints, Combine(Values(256, 512),               // pointCnt
                                                                         Values(0.0f, 0.0f),             // rhoMin
                                                                         Values(180.0f, 320.0f),         // rhoMax
                                                                         Values(0, 0),                   // thetaMin
                                                                         Values(CV_PI / 2.0f, CV_PI),    // thetaMax
                                                                         Values(10, 20)));               // polarCnt


} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
