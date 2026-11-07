// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Imgproc_GetRectSubPix, floatVectorTailsAndRoi)
{
    RNG rng(30388);
    for (int cn : {1, 3})
    for (int width : {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129})
    {
        Mat storage(76, 180, CV_MAKETYPE(CV_32F, cn));
        rng.fill(storage, RNG::UNIFORM, -1000.f, 1000.f);
        Mat src = storage(Rect(3, 2, 170, 70));
        Mat before = storage.clone();
        for (Point2f center : {Point2f(84.375f, 34.625f), Point2f(84.f, 34.f), Point2f(0.25f, 0.75f)})
        {
            const Size size(width, 7);
            Mat outputStorage(9, width + 6, src.type(), Scalar::all(123));
            Mat actual = outputStorage(Rect(2, 1, width, 7));
            getRectSubPix(src, size, center, actual, CV_32F);
            const float x0 = center.x - (width - 1) * .5f, y0 = center.y - 3.f;
            const int ix = cvFloor(x0), iy = cvFloor(y0);
            const float a = x0 - ix, b = y0 - iy;
            for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
            for (int c = 0; c < cn; ++c)
            {
                const int sx0 = std::max(0, std::min(src.cols - 1, ix + x));
                const int sx1 = std::max(0, std::min(src.cols - 1, ix + x + 1));
                const int sy0 = std::max(0, std::min(src.rows - 1, iy + y));
                const int sy1 = std::max(0, std::min(src.rows - 1, iy + y + 1));
                const double expected = src.ptr<float>(sy0)[sx0*cn+c]*(1.f-a)*(1.f-b)
                    + src.ptr<float>(sy0)[sx1*cn+c]*a*(1.f-b)
                    + src.ptr<float>(sy1)[sx0*cn+c]*(1.f-a)*b
                    + src.ptr<float>(sy1)[sx1*cn+c]*a*b;
                EXPECT_NEAR(actual.ptr<float>(y)[x*cn+c], expected, 3e-4 + 3e-6*std::abs(expected));
            }
            for (int y = 0; y < outputStorage.rows; ++y)
            for (int x = 0; x < outputStorage.cols; ++x)
                if (y == 0 || y == 8 || x < 2 || x >= width + 2)
                {
                    for (int c = 0; c < cn; ++c)
                        EXPECT_EQ(123.f, outputStorage.ptr<float>(y)[x*cn+c]);
                }
        }
        EXPECT_EQ(0., cv::norm(before, storage, NORM_INF));
    }
}
}} // namespace
