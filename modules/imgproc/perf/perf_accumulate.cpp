// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

#ifdef HAVE_OPENVX
PERF_TEST_P(Size_MatType, Accumulate,
    testing::Combine(
        testing::Values(::perf::szODD, ::perf::szQVGA, ::perf::szVGA, ::perf::sz1080p),
        testing::Values(CV_16SC1, CV_32FC1)
    )
)
#else
PERF_TEST_P( Size_MatType, Accumulate,
             testing::Combine(
                 testing::Values(::perf::szODD, ::perf::szQVGA, ::perf::szVGA, ::perf::sz1080p),
                 testing::Values(CV_32FC1)
             )
           )
#endif
{
    Size sz = get<0>(GetParam());
    int dstType = get<1>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, dstType);

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() accumulate(src, dst);

    SANITY_CHECK_NOTHING();
}

#ifdef HAVE_OPENVX
PERF_TEST_P(Size_MatType, AccumulateSquare,
    testing::Combine(
        testing::Values(::perf::szODD, ::perf::szQVGA, ::perf::szVGA, ::perf::sz1080p),
        testing::Values(CV_16SC1, CV_32FC1)
    )
)
#else
PERF_TEST_P( Size_MatType, AccumulateSquare,
             testing::Combine(
                 testing::Values(::perf::szODD, ::perf::szQVGA, ::perf::szVGA, ::perf::sz1080p),
                 testing::Values(CV_32FC1)
             )
           )
#endif
{
    Size sz = get<0>(GetParam());
    int dstType = get<1>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, dstType);

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() accumulateSquare(src, dst);

    SANITY_CHECK_NOTHING();
}

#ifdef HAVE_OPENVX
PERF_TEST_P(Size_MatType, AccumulateWeighted,
    testing::Combine(
        testing::Values(::perf::szODD, ::perf::szQVGA, ::perf::szVGA, ::perf::sz1080p),
        testing::Values(CV_8UC1, CV_32FC1)
    )
)
#else
PERF_TEST_P( Size_MatType, AccumulateWeighted,
             testing::Combine(
                 testing::Values(::perf::szODD, ::perf::szQVGA, ::perf::szVGA, ::perf::sz1080p),
                 testing::Values(CV_32FC1)
             )
           )
#endif
{
    Size sz = get<0>(GetParam());
    int dstType = get<1>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, dstType);

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() accumulateWeighted(src, dst, 0.314);

    SANITY_CHECK_NOTHING();
}

} // namespace
