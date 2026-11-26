// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: imgproc functions on the accelerator devices of the build (CUDA, HIP), against the CPU.

#include "test_precomp.hpp"
#include "opencv2/core/accel.hpp"
#include "opencv2/core/private/accel_backend.hpp"

namespace opencv_test { namespace {

std::vector<accel::Device> devices()
{
    std::vector<accel::Device> d = accel::Device::enumerate();
    if (d.empty())
        throw SkipTestException("no accelerator device");
    return d;
}

//! a kernel wrote @p m (UMat::handle(ACCESS_WRITE)); the CPU path leaves the host copy current
bool writtenByDevice(const UMat& m)
{
    return m.u->hostCopyObsolete();
}

Mat toHost(const UMat& u)
{
    Mat m;
    u.copyTo(m);
    return m;
}

Mat randomImage(Size size, int type)
{
    Mat m(size, type);
    const int depth = CV_MAT_DEPTH(type);
    randu(m, depth == CV_16S ? -32768 : 0, depth == CV_16U ? 65536 : depth == CV_16S ? 32768 : 256);
    return m;
}

//! 8U and 16U are bit-exact; the floating-point ones round differently
double blurTolerance(int depth)
{
    return depth == CV_8U || depth == CV_16U ? 0 : depth == CV_16S ? 1 : 1e-3;
}

struct BlurCase { Size ksize; double sigmaX, sigmaY; };

TEST(Imgproc_Accel, GaussianBlur_matches_cpu)
{
    const int types[] = { CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC2, CV_16SC1, CV_32FC1, CV_32FC3 };
    const BlurCase cases[] = { { Size(3, 3), 0, 0 }, { Size(5, 5), 0, 0 }, { Size(7, 3), 1.3, 0.8 },
                               { Size(1, 9), 0, 2 }, { Size(0, 0), 2.5, 0 } };
    const int borders[] = { BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101 };
    for (const accel::Device& d : devices())
        for (int type : types)
        {
            const Mat src = randomImage(Size(641, 361), type);
            const UMat u = src.upload(d);
            for (const BlurCase& c : cases)
                for (int border : borders)
                {
                    UMat dst;
                    cv::GaussianBlur(u, dst, c.ksize, c.sigmaX, c.sigmaY, border);
                    EXPECT_EQ(d, accel::getDevice(dst));
                    EXPECT_TRUE(writtenByDevice(dst));
                    Mat expected;
                    cv::GaussianBlur(src, expected, c.ksize, c.sigmaX, c.sigmaY, border);
                    EXPECT_LE(cvtest::norm(expected, toHost(dst), NORM_INF), blurTolerance(CV_MAT_DEPTH(type)))
                        << d.name() << " " << typeToString(type) << " ksize " << c.ksize << " border " << border;
                }
        }
}

TEST(Imgproc_Accel, GaussianBlur_images_smaller_than_the_kernel)
{
    const Size sizes[] = { Size(1, 1), Size(7, 1), Size(1, 7), Size(3, 2), Size(2, 5) };
    const int borders[] = { BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101 };
    for (const accel::Device& d : devices())
        for (int type : { CV_8UC1, CV_32FC1 })
            for (Size sz : sizes)
                for (int border : borders)
                {
                    const Mat src = randomImage(sz, type);
                    UMat dst;
                    cv::GaussianBlur(src.upload(d), dst, Size(5, 7), 0, 0, border);
                    EXPECT_TRUE(writtenByDevice(dst));
                    Mat expected;
                    cv::GaussianBlur(src, expected, Size(5, 7), 0, 0, border);
                    EXPECT_LE(cvtest::norm(expected, toHost(dst), NORM_INF), blurTolerance(CV_MAT_DEPTH(type)))
                        << d.name() << " " << typeToString(type) << " " << sz << " border " << border;
                }
}

TEST(Imgproc_Accel, GaussianBlur_in_place_and_regions)
{
    for (const accel::Device& d : devices())
    {
        const Mat src = randomImage(Size(320, 240), CV_8UC3);
        const Rect r(17, 9, 200, 150);
        Mat expected;

        UMat u = src.upload(d);
        cv::GaussianBlur(u, u, Size(5, 5), 0);
        EXPECT_TRUE(writtenByDevice(u));
        cv::GaussianBlur(src, expected, Size(5, 5), 0);
        EXPECT_EQ(0, cvtest::norm(expected, toHost(u), NORM_INF)) << d.name() << " in place";

        // an isolated region is the whole input; otherwise the CPU reads its parent beyond its edges
        const UMat whole = src.upload(d);
        for (int isolated : { (int)BORDER_ISOLATED, 0 })
        {
            UMat dst;
            cv::GaussianBlur(whole(r), dst, Size(7, 7), 0, 0, BORDER_REFLECT_101 | isolated);
            EXPECT_EQ(d, accel::getDevice(dst));
            EXPECT_EQ(isolated != 0, writtenByDevice(dst));
            cv::GaussianBlur(src(r), expected, Size(7, 7), 0, 0, BORDER_REFLECT_101 | isolated);
            EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << d.name() << " isolated " << isolated;
        }
    }
}

/* The GPU computes coordinates as the CPU's SIMD body does, but the CPU's scalar tail, IPP and
   compilers that fuse y*M1 + M2 round them differently: a coordinate one ulp away moves a linear
   result a little and may flip a nearest one. */
double warpMismatches(const Mat& expected, const Mat& actual)
{
    const int depth = expected.depth();
    Mat diff;
    absdiff(expected, actual, diff);
    diff = diff.reshape(1);
    return (double)countNonZero(diff > (depth == CV_8U ? 1 : depth == CV_16U ? 8 : 0.05)) / diff.total();
}

TEST(Imgproc_Accel, warpAffine_matches_cpu)
{
    const int types[] = { CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC4, CV_32FC1, CV_32FC3 };
    const int inters[] = { INTER_NEAREST, INTER_LINEAR, INTER_AREA };
    const int borders[] = { BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101 };
    const Size ssize(641, 481), dsize(500, 401);
    Mat M = getRotationMatrix2D(Point2f(320.f, 240.f), 33., 0.83);
    M.at<double>(0, 2) += 41.5;
    M.at<double>(1, 2) -= 23.25;
    for (const accel::Device& d : devices())
        for (int type : types)
        {
            const Mat src = randomImage(ssize, type);
            const UMat u = src.upload(d);
            for (int inter : inters)
                for (int border : borders)
                    for (int inverse : { 0, (int)WARP_INVERSE_MAP })
                    {
                        const Scalar bval(10, 50, 100, 200);
                        UMat dst;
                        cv::warpAffine(u, dst, M, dsize, inter | inverse, border, bval);
                        EXPECT_EQ(d, accel::getDevice(dst));
                        EXPECT_TRUE(writtenByDevice(dst));
                        Mat expected, actual = toHost(dst);
                        cv::warpAffine(src, expected, M, dsize, inter | inverse, border, bval);
                        EXPECT_LE(warpMismatches(expected, actual), 1e-3)
                            << d.name() << " " << typeToString(type) << " inter " << inter << " border " << border
                            << " inverse " << inverse;
                    }
        }
}

//! gives @p backend its own tuning back when it goes out of scope
struct TuningReset
{
    explicit TuningReset(const String& backend) : backend_(backend) {}
    ~TuningReset() { accel::setTuning(backend_, String()); }
    String backend_;
};

TEST(Imgproc_Accel, kernels_match_cpu_in_every_tuning)
{
    const int blurTypes[] = { CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16SC2, CV_32FC1 };
    const int warpTypes[] = { CV_8UC1, CV_8UC4, CV_16UC1, CV_32FC3 };
    const char* blocks[] = { "32x8", "7x5", "64x2", "128x1", "16x16" };
    const Size size(203, 131);
    const Mat M = getRotationMatrix2D(Point2f(100.f, 65.f), 25., 0.9);
    int shapes = 0, n = 0;
    for (const accel::Device& d : devices())
        for (int px : { 1, 2, 4 })
            for (int py : { 1, 2, 4 })
            {
                const String spec = cv::format("blurRows:px=%d,py=%d,block=%s,loopy=%d;"
                                               "blurCols:px=%d,py=%d,block=%s;"
                                               "warpAffine:px=%d,py=%d,block=%s,loopy=3",
                                               px, py, blocks[n % 5], 1 + n % 3, px, py, blocks[(n + 1) % 5],
                                               px, py, blocks[(n + 2) % 5]);
                n++;
                TuningReset reset(d.backend());
                try
                {
                    accel::setTuning(d.backend(), spec);
                }
                catch (const cv::Exception& e)
                {
                    ASSERT_EQ(Error::StsBadArg, e.code) << e.what();   // a patch the build does not compile
                    continue;
                }
                shapes++;
                for (int type : blurTypes)
                    for (int border : { BORDER_CONSTANT, BORDER_REFLECT_101 })
                    {
                        const Mat src = randomImage(size, type);
                        UMat dst;
                        cv::GaussianBlur(src.upload(d), dst, Size(7, 5), 1.7, 1.1, border);
                        EXPECT_TRUE(writtenByDevice(dst));
                        Mat expected;
                        cv::GaussianBlur(src, expected, Size(7, 5), 1.7, 1.1, border);
                        EXPECT_LE(cvtest::norm(expected, toHost(dst), NORM_INF), blurTolerance(CV_MAT_DEPTH(type)))
                            << d.name() << " " << spec << " " << typeToString(type) << " border " << border;
                    }
                for (int type : warpTypes)
                    for (int inter : { INTER_NEAREST, INTER_LINEAR })
                    {
                        const Mat src = randomImage(size, type);
                        UMat dst;
                        cv::warpAffine(src.upload(d), dst, M, size, inter, BORDER_REFLECT);
                        EXPECT_TRUE(writtenByDevice(dst));
                        Mat expected;
                        cv::warpAffine(src, expected, M, size, inter, BORDER_REFLECT);
                        EXPECT_LE(warpMismatches(expected, toHost(dst)), 1e-3)
                            << d.name() << " " << spec << " " << typeToString(type) << " inter " << inter;
                    }
            }
    EXPECT_GE(shapes, 1);
}

TEST(Imgproc_Accel, tunings_are_checked)
{
    const char* bad[] = { "blurRows", "blurRows:px", "blurRows:px=0", "blurRows:px=3", "blurRows:pz=1",
                          "blurRows:block=64", "blurRows:block=64x32", "blurRows:loopy=-1", "nosuch:px=1",
                          "blurRows@8UC9:px=1", "blurRows@9U:px=1" };
    for (const accel::Device& d : devices())
    {
        for (const char* spec : bad)
            EXPECT_THROW(accel::setTuning(d.backend(), spec), cv::Exception) << d.name() << " " << spec;
        TuningReset reset(d.backend());
        accel::setTuning(d.backend(), " blurRows:block=64x4 ; blurRows@8UC1:px=4 ; blurCols@CV_32FC1:py=4,loopy=2 ;");
        const Mat src = randomImage(Size(99, 77), CV_8UC1);
        UMat dst;
        cv::GaussianBlur(src.upload(d), dst, Size(5, 5), 0);
        Mat expected;
        cv::GaussianBlur(src, expected, Size(5, 5), 0);
        EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << d.name();
    }
    EXPECT_THROW(accel::setTuning("nosuch", String()), cv::Exception);
}

TEST(Imgproc_Accel, warpAffine_in_place_region_and_fallbacks)
{
    const Mat M = getRotationMatrix2D(Point2f(100.f, 80.f), -20., 1.1);
    for (const accel::Device& d : devices())
    {
        const Mat src = randomImage(Size(200, 160), CV_8UC3);
        Mat expected;
        cv::warpAffine(src, expected, M, src.size());

        UMat u = src.upload(d);
        cv::warpAffine(u, u, M, u.size());
        EXPECT_TRUE(writtenByDevice(u));
        EXPECT_LE(warpMismatches(expected, toHost(u)), 1e-3) << d.name() << " in place";

        // a region of an output on the device is written there
        UMat big = Mat(Mat::zeros(300, 400, CV_8UC3)).upload(d);
        cv::warpAffine(src.upload(d), big(Rect(50, 70, 200, 160)), M, src.size());
        Mat bigExpected = Mat::zeros(300, 400, CV_8UC3);
        expected.copyTo(bigExpected(Rect(50, 70, 200, 160)));
        EXPECT_LE(warpMismatches(bigExpected, toHost(big)), 1e-3) << d.name() << " region";

        // no kernel for cubic interpolation: the CPU runs it, the output is still on the device
        UMat cubic;
        cv::warpAffine(src.upload(d), cubic, M, src.size(), INTER_CUBIC);
        EXPECT_EQ(d, accel::getDevice(cubic));
        EXPECT_FALSE(writtenByDevice(cubic));
        cv::warpAffine(src, expected, M, src.size(), INTER_CUBIC);
        EXPECT_EQ(0, cvtest::norm(expected, toHost(cubic), NORM_INF)) << d.name() << " cubic";

        // BORDER_TRANSPARENT keeps what the output held, wherever it is
        Mat keep(src.size(), CV_8UC3, Scalar(1, 2, 3));
        UMat transparent;
        keep.copyTo(transparent);
        cv::warpAffine(src.upload(d), transparent, M, src.size(), INTER_LINEAR, BORDER_TRANSPARENT);
        cv::warpAffine(src, keep, M, src.size(), INTER_LINEAR, BORDER_TRANSPARENT);
        EXPECT_EQ(0, cvtest::norm(keep, toHost(transparent), NORM_INF)) << d.name() << " transparent";
    }
}

}} // namespace
