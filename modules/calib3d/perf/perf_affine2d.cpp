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

#include "perf_precomp.hpp"
#include <algorithm>
#include <functional>

namespace cvtest
{

using std::tr1::tuple;
using std::tr1::get;
using namespace perf;
using namespace testing;
using namespace cv;

typedef tuple<int, double> AffineParams;
typedef TestBaseWithParam<AffineParams> EstimateAffine;

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

static float rngIn(float from, float to) { return from + (to-from) * (float)theRNG(); }

static Mat rngPartialAffMat() {
    double theta = rngIn(0, (float)CV_PI*2.f);
    double scale = rngIn(0, 3);
    double tx = rngIn(-2, 2);
    double ty = rngIn(-2, 2);
    double aff[2*3] = { std::cos(theta) * scale, -std::sin(theta) * scale, tx,
                        std::sin(theta) * scale,  std::cos(theta) * scale, ty };
    return Mat(2, 3, CV_64F, aff).clone();
}

PERF_TEST_P( EstimateAffine, EstimateAffine2D, Combine(Values(100000, 5000, 100), Values(0.99, 0.95, 0.9)) )
{
    AffineParams params = GetParam();
    const int n = get<0>(params);
    const double confidence = get<1>(params);

    Mat aff(2, 3, CV_64F);
    cv::randu(aff, Scalar(-2), Scalar(2));

    // setting points that are no in the same line
    const int m = 2*n/5;
    const Point2f shift_outl = Point2f(15, 15);
    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC2);
    Mat tpts(1, n, CV_32FC2);

    cv::randu(fpts, Scalar::all(0), Scalar::all(100));
    std::transform(fpts.ptr<Point2f>(), fpts.ptr<Point2f>() + n, tpts.ptr<Point2f>(), WrapAff2D(aff));

    /* adding noise*/
    std::transform(tpts.ptr<Point2f>() + m, tpts.ptr<Point2f>() + n, tpts.ptr<Point2f>() + m, bind2nd(std::plus<Point2f>(), shift_outl));
    std::transform(tpts.ptr<Point2f>() + m, tpts.ptr<Point2f>() + n, tpts.ptr<Point2f>() + m, Noise(noise_level));

    Mat aff_est(2, 3, CV_64F);

    warmup(aff_est, WARMUP_WRITE);
    warmup(fpts, WARMUP_READ);
    warmup(tpts, WARMUP_READ);

    TEST_CYCLE()
    {
        estimateAffine2D(fpts, tpts, aff_est, noArray(), 3, confidence);
    }

    SANITY_CHECK(aff_est, .01, ERROR_RELATIVE);
}

PERF_TEST_P( EstimateAffine, EstimateAffinePartial2D, Combine(Values(100000, 5000, 100), Values(0.99, 0.95, 0.9)) )
{
    AffineParams params = GetParam();
    const int n = get<0>(params);
    const double confidence = get<1>(params);

    Mat aff = rngPartialAffMat();

    // setting points that are no in the same line
    const int m = 2*n/5;
    const Point2f shift_outl = Point2f(15, 15);
    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC2);
    Mat tpts(1, n, CV_32FC2);

    cv::randu(fpts, Scalar::all(0), Scalar::all(100));
    std::transform(fpts.ptr<Point2f>(), fpts.ptr<Point2f>() + n, tpts.ptr<Point2f>(), WrapAff2D(aff));

    /* adding noise*/
    std::transform(tpts.ptr<Point2f>() + m, tpts.ptr<Point2f>() + n, tpts.ptr<Point2f>() + m, bind2nd(std::plus<Point2f>(), shift_outl));
    std::transform(tpts.ptr<Point2f>() + m, tpts.ptr<Point2f>() + n, tpts.ptr<Point2f>() + m, Noise(noise_level));

    Mat aff_est(2, 3, CV_64F);

    warmup(aff_est, WARMUP_WRITE);
    warmup(fpts, WARMUP_READ);
    warmup(tpts, WARMUP_READ);

    TEST_CYCLE()
    {
        estimateAffinePartial2D(fpts, tpts, aff_est, noArray(), 3, confidence);
    }

    SANITY_CHECK(aff_est, .01, ERROR_RELATIVE);
}

} // namespace cvtest
