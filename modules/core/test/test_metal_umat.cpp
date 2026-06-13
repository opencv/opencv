// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_METAL
#include "opencv2/core/metal.hpp"
#endif

namespace opencv_test { namespace {

#ifdef HAVE_METAL

TEST(Core_Metal_UMat, UploadDownload)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(33, 35, CV_8UC3);
    randu(src, 0, 255);

    UMat u;
    src.copyTo(u);

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 0);
}

TEST(Core_Metal_UMat, MapWriteDownload)
{
    if (!cv::metal::haveMetal())
        return;

    UMat u(32, 32, CV_8UC1);
    {
        Mat mapped = u.getMat(ACCESS_WRITE);
        mapped.setTo(Scalar::all(17));
    }

    Mat dst;
    u.copyTo(dst);

    Mat expected(dst.size(), dst.type(), Scalar::all(17));
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceCopy)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(31, 37, CV_32FC2);
    randu(src, -10.0f, 10.0f);

    UMat u1;
    src.copyTo(u1);

    UMat u2;
    u1.copyTo(u2);

    Mat dst;
    u2.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 0);
}

#endif // HAVE_METAL

}} // namespace
