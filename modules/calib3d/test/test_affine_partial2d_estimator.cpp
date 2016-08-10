/*M///////////////////////////////////////////////////////////////////////////////////////
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

using namespace cv;
using namespace std;

#include <string>
#include <iostream>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <cmath>

class CV_AffinePartial2D_EstTest : public cvtest::BaseTest
{
public:
    CV_AffinePartial2D_EstTest();
    ~CV_AffinePartial2D_EstTest();
protected:
    void run(int);

    bool test2Points();
    bool testNPoints();
};

CV_AffinePartial2D_EstTest::CV_AffinePartial2D_EstTest()
{
}
CV_AffinePartial2D_EstTest::~CV_AffinePartial2D_EstTest() {}


static float rngIn(float from, float to) { return from + (to-from) * (float)theRNG(); }

// get random matrix of affine transformation limited to combinations of translation,
// rotation, and uniform scaling
static Mat rngPartialAffMat() {
    double theta = rngIn(0, (float)CV_PI*2.f);
    double scale = rngIn(0, 3);
    double tx = rngIn(-2, 2);
    double ty = rngIn(-2, 2);
    double aff[2*3] = { std::cos(theta) * scale, -std::sin(theta) * scale, tx,
                        std::sin(theta) * scale,  std::cos(theta) * scale, ty };
    return Mat(2, 3, CV_64F, aff).clone();
}

struct WrapAff2D
{
    const double *F;
    WrapAff2D(const Mat& aff) : F(aff.ptr<double>()) {}
    Point2f operator()(const Point2f& p)
    {
        return Point2f( (float)(p.x * F[0] + p.y * F[1] + F[2]),
                        (float)(p.x * F[3] + p.y * F[4] + F[5]) );
    }
};

bool CV_AffinePartial2D_EstTest::test2Points()
{
    Mat aff = rngPartialAffMat();

    // setting points that are no in the same line
    Mat fpts(1, 2, CV_32FC2);
    Mat tpts(1, 2, CV_32FC2);

    fpts.ptr<Point2f>()[0] = Point2f( rngIn(1,2), rngIn(5,6) );
    fpts.ptr<Point2f>()[1] = Point2f( rngIn(3,4), rngIn(3,4) );

    transform(fpts.begin<Point2f>(), fpts.end<Point2f>(), tpts.begin<Point2f>(), WrapAff2D(aff));

    Mat aff_est;
    vector<uchar> inliers;
    estimateAffinePartial2D(fpts, tpts, aff_est, inliers);

    const double thres = 1e-3;
    if (cvtest::norm(aff_est, aff, NORM_INF) > thres)
    {
        cout << "norm: " << cvtest::norm(aff_est, aff, NORM_INF) << endl;
        cout << "aff est: " << aff_est << endl;
        cout << "aff ref: " << aff << endl;
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return false;
    }
    return true;
}

struct Noise
{
    float l;
    Noise(float level) : l(level) {}
    Point2f operator()(const Point2f& p)
    {
        RNG& rng = theRNG();
        return Point2f( p.x + l * (float)rng,  p.y + l * (float)rng );
    }
};

bool CV_AffinePartial2D_EstTest::testNPoints()
{
    Mat aff = rngPartialAffMat();

    // setting points that are no in the same line
    const int n = 100;
    const int m = 2*n/5;
    const Point2f shift_outl = Point2f(15, 15);
    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC2);
    Mat tpts(1, n, CV_32FC2);

    randu(fpts, Scalar::all(0), Scalar::all(100));
    transform(fpts.ptr<Point2f>(), fpts.ptr<Point2f>() + n, tpts.ptr<Point2f>(), WrapAff2D(aff));

    /* adding noise*/
    transform(tpts.ptr<Point2f>() + m, tpts.ptr<Point2f>() + n, tpts.ptr<Point2f>() + m, bind2nd(plus<Point2f>(), shift_outl));
    transform(tpts.ptr<Point2f>() + m, tpts.ptr<Point2f>() + n, tpts.ptr<Point2f>() + m, Noise(noise_level));

    Mat aff_est;
    vector<uchar> outl;
    int res = estimateAffinePartial2D(fpts, tpts, aff_est, outl);

    if (!res)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return false;
    }

    const double thres = 1e-4;
    if (cvtest::norm(aff_est, aff, NORM_INF) > thres)
    {
        cout << "norm: " << cvtest::norm(aff_est, aff, NORM_INF) << endl;
        cout << "aff est: " << aff_est << endl;
        cout << "aff ref: " << aff << endl;
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return false;
    }

    bool outl_good = count(outl.begin(), outl.end(), 1) == m &&
        m == accumulate(outl.begin(), outl.begin() + m, 0);

    if (!outl_good)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return false;
    }
    return true;
}


void CV_AffinePartial2D_EstTest::run( int /* start_from */)
{
    cvtest::DefaultRngAuto dra;

    if (!test2Points())
        return;

    if (!testNPoints())
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Calib3d_EstimateAffinePartial2D, accuracy) { CV_AffinePartial2D_EstTest test; test.safe_run(); }
