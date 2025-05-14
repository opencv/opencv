/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_Affine3D_EstTest : public cvtest::BaseTest
{
public:
    CV_Affine3D_EstTest();
    ~CV_Affine3D_EstTest();
protected:
    void run(int);

    bool test4Points();
    bool testNPoints();
};

CV_Affine3D_EstTest::CV_Affine3D_EstTest()
{
}
CV_Affine3D_EstTest::~CV_Affine3D_EstTest() {}


float rngIn(float from, float to) { return from + (to-from) * (float)theRNG(); }


struct WrapAff
{
    const double *F;
    WrapAff(const Mat& aff) : F(aff.ptr<double>()) {}
    Point3f operator()(const Point3f& p)
    {
        return Point3f( (float)(p.x * F[0] + p.y * F[1] + p.z *  F[2] +  F[3]),
                        (float)(p.x * F[4] + p.y * F[5] + p.z *  F[6] +  F[7]),
                        (float)(p.x * F[8] + p.y * F[9] + p.z * F[10] + F[11])  );
    }
};

bool CV_Affine3D_EstTest::test4Points()
{
    Mat aff(3, 4, CV_64F);
    cv::randu(aff, Scalar(1), Scalar(3));

    // setting points that are no in the same line

    Mat fpts(1, 4, CV_32FC3);
    Mat tpts(1, 4, CV_32FC3);

    fpts.ptr<Point3f>()[0] = Point3f( rngIn(1,2), rngIn(1,2), rngIn(5, 6) );
    fpts.ptr<Point3f>()[1] = Point3f( rngIn(3,4), rngIn(3,4), rngIn(5, 6) );
    fpts.ptr<Point3f>()[2] = Point3f( rngIn(1,2), rngIn(3,4), rngIn(5, 6) );
    fpts.ptr<Point3f>()[3] = Point3f( rngIn(3,4), rngIn(1,2), rngIn(5, 6) );

    std::transform(fpts.ptr<Point3f>(), fpts.ptr<Point3f>() + 4, tpts.ptr<Point3f>(), WrapAff(aff));

    Mat aff_est;
    vector<uchar> outliers;
    estimateAffine3D(fpts, tpts, aff_est, outliers);

    const double thres = 1e-3;
    if (cvtest::norm(aff_est, aff, NORM_INF) > thres)
    {
        //cout << cvtest::norm(aff_est, aff, NORM_INF) << endl;
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

struct Noise
{
    float l;
    Noise(float level) : l(level) {}
    Point3f operator()(const Point3f& p)
    {
        RNG& rng = theRNG();
        return Point3f( p.x + l * (float)rng,  p.y + l * (float)rng,  p.z + l * (float)rng);
    }
};

bool CV_Affine3D_EstTest::testNPoints()
{
    Mat aff(3, 4, CV_64F);
    cv::randu(aff, Scalar(-2), Scalar(2));

    // setting points that are no in the same line

    const int n = 100;
    const int m = 3*n/5;
    const Point3f shift_outl = Point3f(15, 15, 15);
    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC3);
    Mat tpts(1, n, CV_32FC3);

    randu(fpts, Scalar::all(0), Scalar::all(100));
    std::transform(fpts.ptr<Point3f>(), fpts.ptr<Point3f>() + n, tpts.ptr<Point3f>(), WrapAff(aff));

    /* adding noise*/
    std::transform(tpts.ptr<Point3f>() + m, tpts.ptr<Point3f>() + n, tpts.ptr<Point3f>() + m,
        [=] (const Point3f& pt) -> Point3f { return Noise(noise_level)(pt + shift_outl); });

    Mat aff_est;
    vector<uchar> outl;
    int res = estimateAffine3D(fpts, tpts, aff_est, outl);

    if (!res)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }

    const double thres = 1e-4;
    if (cvtest::norm(aff_est, aff, NORM_INF) > thres)
    {
        cout << "aff est: " << aff_est << endl;
        cout << "aff ref: " << aff << endl;
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }

    bool outl_good = std::count(outl.begin(), outl.end(), 1) == m &&
        m == std::accumulate(outl.begin(), outl.begin() + m, 0);

    if (!outl_good)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}


void CV_Affine3D_EstTest::run( int /* start_from */)
{
    cvtest::DefaultRngAuto dra;

    if (!test4Points())
        return;

    if (!testNPoints())
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Calib3d_EstimateAffine3D, accuracy) { CV_Affine3D_EstTest test; test.safe_run(); }

TEST(Calib3d_EstimateAffine3D, regression_16007)
{
    std::vector<cv::Point3f> m1, m2;
    m1.push_back(Point3f(1.0f, 0.0f, 0.0f)); m2.push_back(Point3f(1.0f, 1.0f, 0.0f));
    m1.push_back(Point3f(1.0f, 0.0f, 1.0f)); m2.push_back(Point3f(1.0f, 1.0f, 1.0f));
    m1.push_back(Point3f(0.5f, 0.0f, 0.5f)); m2.push_back(Point3f(0.5f, 1.0f, 0.5f));
    m1.push_back(Point3f(2.5f, 0.0f, 2.5f)); m2.push_back(Point3f(2.5f, 1.0f, 2.5f));
    m1.push_back(Point3f(2.0f, 0.0f, 1.0f)); m2.push_back(Point3f(2.0f, 1.0f, 1.0f));

    cv::Mat m3D, inl;
    int res = cv::estimateAffine3D(m1, m2, m3D, inl);
    EXPECT_EQ(1, res);
}

TEST(Calib3d_EstimateAffine3D, umeyama_3_pt)
{
    std::vector<cv::Vec3d> points =   {{{0.80549149, 0.8225781, 0.79949521},
                                        {0.28906756, 0.57158557, 0.9864789},
                                        {0.58266182, 0.65474983, 0.25078834}}};
    cv::Mat R =   (cv::Mat_<double>(3,3) << 0.9689135, -0.0232753, 0.2463025,
                                            0.0236362,  0.9997195, 0.0014915,
                                            -0.2462682, 0.0043765, 0.9691918);
    cv::Vec3d t(1., 2., 3.);
    cv::Affine3d transform(R, t);
    std::vector<cv::Vec3d> transformed_points(points.size());
    std::transform(points.begin(), points.end(), transformed_points.begin(), [transform](const cv::Vec3d v){return transform * v;});
    double scale;
    cv::Mat trafo_est = estimateAffine3D(points, transformed_points, &scale);
    Mat R_est(trafo_est(Rect(0, 0, 3, 3)));
    EXPECT_LE(cvtest::norm(R_est, R, NORM_INF), 1e-6);
    Vec3d t_est = trafo_est.col(3);
    EXPECT_LE(cvtest::norm(t_est, t, NORM_INF), 1e-6);
    EXPECT_NEAR(scale, 1.0, 1e-6);
}

}} // namespace
