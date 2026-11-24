// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

static Mat makeCloningImage(Size sz, uint64 seed)
{
    RNG rng(seed);
    Mat noise(sz, CV_8UC3);
    rng.fill(noise, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
    GaussianBlur(noise, noise, Size(0, 0), 3.0);
    Mat grad(sz, CV_8UC3);
    for (int y = 0; y < sz.height; y++)
    {
        uchar* p = grad.ptr<uchar>(y);
        for (int x = 0; x < sz.width; x++)
        {
            p[3 * x + 0] = (uchar)(x * 255 / std::max(1, sz.width - 1));
            p[3 * x + 1] = (uchar)(y * 255 / std::max(1, sz.height - 1));
            p[3 * x + 2] = (uchar)((x + y) * 255 / std::max(1, sz.width + sz.height - 2));
        }
    }
    Mat img;
    addWeighted(noise, 0.5, grad, 0.5, 0, img);
    return img;
}

static Mat makeCloningMask(Size sz)
{
    Mat mask = Mat::zeros(sz, CV_8UC1);
    ellipse(mask, Point(sz.width / 2, sz.height / 2),
            Size(sz.width / 2 - 8, sz.height / 2 - 8), 0, 0, 360, Scalar(255), FILLED);
    return mask;
}

typedef TestBaseWithParam<Size> Photo_Cloning_Size;
typedef TestBaseWithParam< tuple<Size, int> > Photo_Cloning_Size_Flag;

PERF_TEST_P(Photo_Cloning_Size_Flag, seamlessClone,
            testing::Combine(testing::Values(szQVGA, szVGA, Size(641, 481), sz720p, sz1080p),
                             testing::Values((int)NORMAL_CLONE, (int)MIXED_CLONE)))
{
    const Size sz = get<0>(GetParam());
    const int flags = get<1>(GetParam());

    Mat src = makeCloningImage(sz, 1);
    Mat dst = makeCloningImage(sz, 2);
    Mat mask = makeCloningMask(sz);
    Mat result(sz, CV_8UC3);
    const Point center(sz.width / 2, sz.height / 2);

    declare.in(src, dst, mask).out(result).time(60);

    TEST_CYCLE() seamlessClone(src, dst, mask, center, result, flags);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Photo_Cloning_Size, colorChange,
            testing::Values(szQVGA, szVGA, Size(641, 481), sz720p, sz1080p))
{
    const Size sz = GetParam();

    Mat src = makeCloningImage(sz, 3);
    Mat mask = makeCloningMask(sz);
    Mat result(sz, CV_8UC3);

    declare.in(src, mask).out(result).time(60);

    TEST_CYCLE() colorChange(src, mask, result, 1.5f, 0.5f, 0.5f);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Photo_Cloning_Size, illuminationChange,
            testing::Values(szQVGA, szVGA, Size(641, 481), sz720p, sz1080p))
{
    const Size sz = GetParam();

    Mat src = makeCloningImage(sz, 4);
    Mat mask = makeCloningMask(sz);
    Mat result(sz, CV_8UC3);

    declare.in(src, mask).out(result).time(60);

    TEST_CYCLE() illuminationChange(src, mask, result, 0.2f, 0.4f);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Photo_Cloning_Size, textureFlattening,
            testing::Values(szQVGA, szVGA, Size(641, 481), sz720p, sz1080p))
{
    const Size sz = GetParam();

    Mat src = makeCloningImage(sz, 5);
    Mat mask = makeCloningMask(sz);
    Mat result(sz, CV_8UC3);

    declare.in(src, mask).out(result).time(60);

    TEST_CYCLE() textureFlattening(src, mask, result, 30, 45, 3);

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
