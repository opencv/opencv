// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4)

typedef TestBaseWithParam< tuple<Size, MatType, MatType, InterType> > TestRemap;

PERF_TEST_P( TestRemap, Remap,
             Combine(
                Values( szVGA, sz1080p ),
                Values( CV_16UC1, CV_16SC1, CV_32FC1 ),
                Values( CV_16SC2, CV_32FC1, CV_32FC2 ),
                InterType::all()
             )
)
{
    Size sz;
    int src_type, map1_type, inter_type;

    sz         = get<0>(GetParam());
    src_type   = get<1>(GetParam());
    map1_type  = get<2>(GetParam());
    inter_type = get<3>(GetParam());

    Mat src(sz, src_type), dst(sz, src_type), map1(sz, map1_type), map2;
    if (map1_type == CV_32FC1)
        map2.create(sz, CV_32FC1);
    else if (inter_type != INTER_NEAREST && map1_type == CV_16SC2)
    {
        map2.create(sz, CV_16UC1);
        map2 = Scalar::all(0);
    }

    RNG rng;
    rng.fill(src, RNG::UNIFORM, 0, 256);

    for (int j = 0; j < map1.rows; ++j)
        for (int i = 0; i < map1.cols; ++i)
            switch (map1_type)
            {
                case CV_32FC1:
                    map1.at<float>(j, i) = static_cast<float>(src.cols - i - 1);
                    map2.at<float>(j, i) = static_cast<float>(j);
                    break;
                case CV_32FC2:
                    map1.at<Vec2f>(j, i)[0] = static_cast<float>(src.cols - i - 1);
                    map1.at<Vec2f>(j, i)[1] = static_cast<float>(j);
                    break;
                case CV_16SC2:
                    map1.at<Vec2s>(j, i)[0] = static_cast<short>(src.cols - i - 1);
                    map1.at<Vec2s>(j, i)[1] = static_cast<short>(j);
                    break;
                default:
                    CV_Assert(0);
            }


    declare.in(src, WARMUP_RNG).out(dst).time(20);

    int runs = (sz.width <= 640) ? 3 : 1;
    TEST_CYCLE_MULTIRUN(runs) remap(src, dst, map1, map2, inter_type);

    if (inter_type == INTER_CUBIC)
    {
        SANITY_CHECK_NOTHING();
    }
    else
    {
        SANITY_CHECK(dst);
    }
}

typedef TestBaseWithParam< tuple<Size, bool> > TestRemap32FC2;

PERF_TEST_P(TestRemap32FC2, linearFloatMaps, Combine(Values(szVGA, sz1080p), testing::Bool()))
{
    const Size size = get<0>(GetParam());
    const bool splitMap = get<1>(GetParam());
    Mat src(size, CV_32FC2), dst(size, CV_32FC2);
    Mat map(size, CV_32FC2);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x)
            map.at<Vec2f>(y, x) = Vec2f(x + 0.19f, y - 0.41f);

    Mat map1, map2;
    if (splitMap)
    {
        extractChannel(map, map1, 0);
        extractChannel(map, map2, 1);
    }
    else
        map1 = map;

    declare.in(src, WARMUP_RNG).out(dst);
    TEST_CYCLE() remap(src, dst, map1, map2, INTER_LINEAR);
    SANITY_CHECK_NOTHING();
}

} // namespace
