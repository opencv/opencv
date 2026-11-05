// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {
TEST(Features2D_ORB, orientationCircularMoments)
{
    RNG rng(30388);
    Mat storage(260, 340, CV_8U);
    rng.fill(storage, RNG::UNIFORM, 0, 256);
    Mat image = storage(Rect(3, 2, 320, 240));
    Mat before = storage.clone();
    for (int patch : {7, 15, 31, 65})
    {
        const int radius = patch / 2;
        std::vector<int> umax(radius + 2);
        const int vmax = cvFloor(radius * std::sqrt(2.f) / 2 + 1);
        const int vmin = cvCeil(radius * std::sqrt(2.f) / 2);
        for (int v = 0; v <= vmax; ++v)
            umax[v] = cvRound(std::sqrt(double(radius*radius - v*v)));
        for (int v = radius, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1]) ++v0;
            umax[v] = v0++;
        }
        Ptr<ORB> orb = ORB::create(300, 1.2f, 1, patch + 8, 0, 2, ORB::FAST_SCORE, patch, 10);
        std::vector<KeyPoint> points;
        orb->detect(image, points);
        ASSERT_FALSE(points.empty());
        for (const KeyPoint& point : points)
        {
            const int px = cvRound(point.pt.x), py = cvRound(point.pt.y);
            int mx = 0, my = 0;
            for (int y = -radius; y <= radius; ++y)
            for (int x = -umax[std::abs(y)]; x <= umax[std::abs(y)]; ++x)
            {
                const int value = image.at<uchar>(py + y, px + x);
                mx += x * value;
                my += y * value;
            }
            EXPECT_FLOAT_EQ(fastAtan2(float(my), float(mx)), point.angle);
        }
    }
    EXPECT_EQ(0., cv::norm(before, storage, NORM_INF));
}
}} // namespace
