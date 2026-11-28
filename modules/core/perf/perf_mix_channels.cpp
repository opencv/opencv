// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef TestBaseWithParam<tuple<Size, MatType, bool, bool>> MixChannelsPermutation;

PERF_TEST_P(MixChannelsPermutation, interleaved,
            testing::Combine(testing::Values(Size(256, 256), Size(512, 512), Size(1024, 1024)),
                             testing::Values(CV_8UC2, CV_8UC3, CV_8UC4, CV_16UC2, CV_16UC3, CV_16UC4,
                                             CV_32FC2, CV_32FC3, CV_32FC4),
                             testing::Bool(), testing::Bool()))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());
    const bool roi = get<2>(GetParam());
    const bool swapOuter = get<3>(GetParam());
    const Rect region(roi ? 1 : 0, roi ? 1 : 0, size.width, size.height);
    const Size storage(region.x + size.width, region.y + size.height);
    Mat source(storage, type);
    Mat destination(storage, type);
    const Mat src = source(region);
    Mat dst = destination(region);
    declare.in(source, WARMUP_RNG).out(destination);

    const int rgb[] = {2, 0, swapOuter ? 1 : 0, 1, swapOuter ? 0 : 1, 2, 3, 3};
    const int twoChannels[] = {swapOuter ? 1 : 0, 0, swapOuter ? 0 : 1, 1};
    const int* fromTo = src.channels() == 2 ? twoChannels : rgb;
    TEST_CYCLE() mixChannels(&src, 1, &dst, 1, fromTo, src.channels());
    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam<tuple<MatType, bool>> MixChannelsWidePermutation;

PERF_TEST_P(MixChannelsWidePermutation, interleaved,
            testing::Combine(testing::Values(CV_8UC(5), CV_8UC(8), CV_8UC(12), CV_8UC(16),
                                             CV_32FC(32), CV_32FC(64), CV_32FC(CV_CN_MAX),
                                             CV_32SC(64), CV_32UC(64)),
                             testing::Bool()))
{
    const int type = get<0>(GetParam());
    const bool roi = get<1>(GetParam());
    const Rect region(roi ? 1 : 0, 0, 512, 512);
    Mat source(region.height, region.x + region.width, type);
    Mat destination(source.size(), type);
    const Mat src = source(region);
    Mat dst = destination(region);
    declare.in(source, WARMUP_RNG).out(destination);

    std::vector<int> fromTo;
    for (int channel = 0; channel < src.channels(); ++channel)
    {
        fromTo.push_back(src.channels() - 1 - channel);
        fromTo.push_back(channel);
    }
    TEST_CYCLE() mixChannels(&src, 1, &dst, 1, fromTo.data(), src.channels());
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
