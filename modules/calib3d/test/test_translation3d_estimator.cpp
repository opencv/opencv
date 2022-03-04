// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Calib3d_EstimateTranslation3D, test4Points)
{
    Matx13d trans;
    cv::randu(trans, Scalar(1), Scalar(3));

    // setting points that are no in the same line

    Mat fpts(1, 4, CV_32FC3);
    Mat tpts(1, 4, CV_32FC3);

    RNG& rng = theRNG();
    fpts.at<Point3f>(0) = Point3f(rng.uniform(1.0f, 2.0f), rng.uniform(1.0f, 2.0f), rng.uniform(5.0f, 6.0f));
    fpts.at<Point3f>(1) = Point3f(rng.uniform(3.0f, 4.0f), rng.uniform(3.0f, 4.0f), rng.uniform(5.0f, 6.0f));
    fpts.at<Point3f>(2) = Point3f(rng.uniform(1.0f, 2.0f), rng.uniform(3.0f, 4.0f), rng.uniform(5.0f, 6.0f));
    fpts.at<Point3f>(3) = Point3f(rng.uniform(3.0f, 4.0f), rng.uniform(1.0f, 2.0f), rng.uniform(5.0f, 6.0f));

    std::transform(fpts.ptr<Point3f>(), fpts.ptr<Point3f>() + 4, tpts.ptr<Point3f>(),
        [&] (const Point3f& p) -> Point3f
        {
            return Point3f((float)(p.x + trans(0, 0)),
                           (float)(p.y + trans(0, 1)),
                           (float)(p.z + trans(0, 2)));
        }
    );

    Matx13d trans_est;
    vector<uchar> outliers;
    int res = estimateTranslation3D(fpts, tpts, trans_est, outliers);
    EXPECT_GT(res, 0);

    const double thres = 1e-3;

    EXPECT_LE(cvtest::norm(trans_est, trans, NORM_INF), thres)
        << "aff est: " << trans_est << endl
        << "aff ref: " << trans;
}

TEST(Calib3d_EstimateTranslation3D, testNPoints)
{
    Matx13d trans;
    cv::randu(trans, Scalar(-2), Scalar(2));

    // setting points that are no in the same line

    const int n = 100;
    const int m = 3*n/5;
    const Point3f shift_outl = Point3f(15, 15, 15);
    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC3);
    Mat tpts(1, n, CV_32FC3);

    randu(fpts, Scalar::all(0), Scalar::all(100));
    std::transform(fpts.ptr<Point3f>(), fpts.ptr<Point3f>() + n, tpts.ptr<Point3f>(),
        [&] (const Point3f& p) -> Point3f
        {
            return Point3f((float)(p.x + trans(0, 0)),
                           (float)(p.y + trans(0, 1)),
                           (float)(p.z + trans(0, 2)));
        }
    );

    /* adding noise*/
    std::transform(tpts.ptr<Point3f>() + m, tpts.ptr<Point3f>() + n, tpts.ptr<Point3f>() + m,
        [&] (const Point3f& pt) -> Point3f
        {
            Point3f p = pt + shift_outl;
            RNG& rng = theRNG();
            return Point3f(p.x + noise_level * (float)rng,
                           p.y + noise_level * (float)rng,
                           p.z + noise_level * (float)rng);
        }
    );

    Matx13d trans_est;
    vector<uchar> outl;
    int res = estimateTranslation3D(fpts, tpts, trans_est, outl);
    EXPECT_GT(res, 0);

    const double thres = 1e-4;
    EXPECT_LE(cvtest::norm(trans_est, trans, NORM_INF), thres)
        << "aff est: " << trans_est << endl
        << "aff ref: " << trans;

    bool outl_good = count(outl.begin(), outl.end(), 1) == m &&
        m == accumulate(outl.begin(), outl.begin() + m, 0);

    EXPECT_TRUE(outl_good);
}

}} // namespace
