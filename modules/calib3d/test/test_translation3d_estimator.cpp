// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_Translation3D_EstTest : public cvtest::BaseTest
{
public:
    CV_Translation3D_EstTest();
    ~CV_Translation3D_EstTest();
protected:
    void run(int);

    bool test4Points();
    bool testNPoints();
};

CV_Translation3D_EstTest::CV_Translation3D_EstTest()
{
}
CV_Translation3D_EstTest::~CV_Translation3D_EstTest() {}


float rngIn(float from, float to) { return from + (to-from) * (float)theRNG(); }


struct WrapTrans
{
    const Matx13d * F;
    WrapTrans(const Matx13d& trans) { F = &trans; }
    Point3f operator()(const Point3f& p)
    {
        return Point3f( (float)(p.x + (*F)(0, 0)),
                        (float)(p.y + (*F)(0, 1)),
                        (float)(p.z + (*F)(0, 2))  );
    }
};

bool CV_Translation3D_EstTest::test4Points()
{

    Matx13d trans;
    cv::randu(trans, Scalar(1), Scalar(3));

    // setting points that are no in the same line

    Mat fpts(1, 4, CV_32FC3);
    Mat tpts(1, 4, CV_32FC3);

    fpts.ptr<Point3f>()[0] = Point3f( rngIn(1,2), rngIn(1,2), rngIn(5, 6) );
    fpts.ptr<Point3f>()[1] = Point3f( rngIn(3,4), rngIn(3,4), rngIn(5, 6) );
    fpts.ptr<Point3f>()[2] = Point3f( rngIn(1,2), rngIn(3,4), rngIn(5, 6) );
    fpts.ptr<Point3f>()[3] = Point3f( rngIn(3,4), rngIn(1,2), rngIn(5, 6) );

    std::transform(fpts.ptr<Point3f>(), fpts.ptr<Point3f>() + 4, tpts.ptr<Point3f>(), WrapTrans(trans));

    Matx13d trans_est;
    vector<uchar> outliers;
    estimateTranslation3D(fpts, tpts, trans_est, outliers);

    const double thres = 1e-3;
    if (cvtest::norm(trans_est, trans, NORM_INF) > thres)
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

bool CV_Translation3D_EstTest::testNPoints()
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
    std::transform(fpts.ptr<Point3f>(), fpts.ptr<Point3f>() + n, tpts.ptr<Point3f>(), WrapTrans(trans));

    /* adding noise*/
#ifdef CV_CXX11
    std::transform(tpts.ptr<Point3f>() + m, tpts.ptr<Point3f>() + n, tpts.ptr<Point3f>() + m,
        [=] (const Point3f& pt) -> Point3f { return Noise(noise_level)(pt + shift_outl); });
#else
    std::transform(tpts.ptr<Point3f>() + m, tpts.ptr<Point3f>() + n, tpts.ptr<Point3f>() + m, std::bind2nd(std::plus<Point3f>(), shift_outl));
    std::transform(tpts.ptr<Point3f>() + m, tpts.ptr<Point3f>() + n, tpts.ptr<Point3f>() + m, Noise(noise_level));
#endif

    Matx13d trans_est;
    vector<uchar> outl;
    int res = estimateTranslation3D(fpts, tpts, trans_est, outl);

    if (!res)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }

    const double thres = 1e-4;
    if (cvtest::norm(trans_est, trans, NORM_INF) > thres)
    {
        cout << "aff est: " << trans_est << endl;
        cout << "aff ref: " << trans << endl;
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }

    bool outl_good = count(outl.begin(), outl.end(), 1) == m &&
        m == accumulate(outl.begin(), outl.begin() + m, 0);

    if (!outl_good)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}


void CV_Translation3D_EstTest::run( int /* start_from */)
{
    cvtest::DefaultRngAuto dra;

    if (!test4Points())
        return;

    if (!testNPoints())
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Calib3d_EstimateTranslation3D, accuracy) { CV_Translation3D_EstTest test; test.safe_run(); }

TEST(Calib3d_EstimateTranslation3D, regression_16007)
{
    std::vector<cv::Point3f> m1, m2;
    m1.push_back(Point3f(1.0f, 0.0f, 0.0f)); m2.push_back(Point3f(1.0f, 1.0f, 0.0f));
    m1.push_back(Point3f(1.0f, 0.0f, 1.0f)); m2.push_back(Point3f(1.0f, 1.0f, 1.0f));
    m1.push_back(Point3f(0.5f, 0.0f, 0.5f)); m2.push_back(Point3f(0.5f, 1.0f, 0.5f));
    m1.push_back(Point3f(2.5f, 0.0f, 2.5f)); m2.push_back(Point3f(2.5f, 1.0f, 2.5f));
    m1.push_back(Point3f(2.0f, 0.0f, 1.0f)); m2.push_back(Point3f(2.0f, 1.0f, 1.0f));

    cv::Mat m3D, inl;
    int res = cv::estimateTranslation3D(m1, m2, m3D, inl);
    EXPECT_EQ(1, res);
}

}} // namespace
