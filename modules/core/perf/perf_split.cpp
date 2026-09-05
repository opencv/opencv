// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<Size, MatType, int> Size_Depth_Channels_t;
typedef perf::TestBaseWithParam<Size_Depth_Channels_t> Size_Depth_Channels;

PERF_TEST_P( Size_Depth_Channels, split,
             testing::Combine
             (
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F),
                 testing::Values(2, 3, 4)
             )
           )
{
    Size sz = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int channels = get<2>(GetParam());

    Mat m(sz, CV_MAKETYPE(depth, channels));
    randu(m, 0, 255);

    vector<Mat> mv;
    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) split(m, (vector<Mat>&)mv);

    SANITY_CHECK(mv, 2e-5);
}

typedef tuple<Size, int> Size_Channel_t;
typedef perf::TestBaseWithParam<Size_Channel_t> Size_Channel;

PERF_TEST_P( Size_Channel, extractChannel,
             testing::Combine
             (
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(0, 1)
             )
           )
{
    Size sz = get<0>(GetParam());
    int channel = get<1>(GetParam());

    Mat src(sz, CV_8UC2);
    Mat dst(sz, CV_8UC1);
    randu(src, 0, 255);

    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) extractChannel(src, dst, channel);

    SANITY_CHECK(dst);
}

} // namespace
