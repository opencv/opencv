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

// this is test for a deprecated function. let's ignore deprecated warnings in this file
#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
    #pragma warning( disable : 4996)
#endif

namespace opencv_test { namespace {

class CV_RigidTransform_Test : public cvtest::BaseTest
{
public:
    CV_RigidTransform_Test();
    ~CV_RigidTransform_Test();
protected:
    void run(int);

    bool testNPoints(int);
    bool testImage();
};

CV_RigidTransform_Test::CV_RigidTransform_Test()
{
}
CV_RigidTransform_Test::~CV_RigidTransform_Test() {}

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

bool CV_RigidTransform_Test::testNPoints(int from)
{
    cv::RNG rng = cv::theRNG();

    int progress = 0;
    int k, ntests = 10000;

    for( k = from; k < ntests; k++ )
    {
        ts->update_context( this, k, true );
        progress = update_progress(progress, k, ntests, 0);

        Mat aff(2, 3, CV_64F);
        rng.fill(aff, RNG::UNIFORM, Scalar(-2), Scalar(2));

        int n = (unsigned)rng % 100 + 10;

        Mat fpts(1, n, CV_32FC2);
        Mat tpts(1, n, CV_32FC2);

        rng.fill(fpts, RNG::UNIFORM, Scalar(0,0), Scalar(10,10));
        std::transform(fpts.ptr<Point2f>(), fpts.ptr<Point2f>() + n, tpts.ptr<Point2f>(), WrapAff2D(aff));

        Mat noise(1, n, CV_32FC2);
        rng.fill(noise, RNG::NORMAL, Scalar::all(0), Scalar::all(0.001*(n<=7 ? 0 : n <= 30 ? 1 : 10)));
        tpts += noise;

        Mat aff_est = estimateRigidTransform(fpts, tpts, true);

        double thres = 0.1*cvtest::norm(aff, NORM_L2);
        double d = cvtest::norm(aff_est, aff, NORM_L2);
        if (d > thres)
        {
            double dB=0, nB=0;
            if (n <= 4)
            {
                Mat A = fpts.reshape(1, 3);
                Mat B = A - repeat(A.row(0), 3, 1), Bt = B.t();
                B = Bt*B;
                dB = cv::determinant(B);
                nB = cvtest::norm(B, NORM_L2);
                if( fabs(dB) < 0.01*nB )
                    continue;
            }
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf( cvtest::TS::LOG, "Threshold = %f, norm of difference = %f", thres, d );
            return false;
        }
    }
    return true;
}

bool CV_RigidTransform_Test::testImage()
{
    Mat img;
    Mat testImg = imread( string(ts->get_data_path()) + "shared/graffiti.png", 1);
    if (testImg.empty())
    {
       ts->printf( ts->LOG, "test image can not be read");
       ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
       return false;
    }
    pyrDown(testImg, img);

    Mat aff = cv::getRotationMatrix2D(Point(img.cols/2, img.rows/2), 1, 0.99);
    aff.ptr<double>()[2]+=3;
    aff.ptr<double>()[5]+=3;

    Mat rotated;
    warpAffine(img, rotated, aff, img.size());

    Mat aff_est = estimateRigidTransform(img, rotated, true);

    const double thres = 0.033;
    if (cvtest::norm(aff_est, aff, NORM_INF) > thres)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        ts->printf( cvtest::TS::LOG, "Threshold = %f, norm of difference = %f", thres,
            cvtest::norm(aff_est, aff, NORM_INF) );
        return false;
    }

    return true;
}

void CV_RigidTransform_Test::run( int start_from )
{
    cvtest::DefaultRngAuto dra;

    if (!testNPoints(start_from))
        return;

    if (!testImage())
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Video_RigidFlow, accuracy) { CV_RigidTransform_Test test; test.safe_run(); }

}} // namespace
