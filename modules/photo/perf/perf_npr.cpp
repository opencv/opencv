#include "perf_precomp.hpp"

namespace opencv_test
{

CV_ENUM(NPRFlags, 1, 2)

typedef tuple<Size, NPRFlags> Size_NPRFlags_t;
typedef perf::TestBaseWithParam<Size_NPRFlags_t> Perf_EdgePreserving;

PERF_TEST_P(Perf_EdgePreserving, edgePreservingFilter,
    testing::Combine(
        testing::Values(::perf::szSmall64,
                        Size(128, 128),
                        ::perf::szVGA,
                        ::perf::sz720p,
                        ::perf::sz1080p),
        NPRFlags::all()))
{
    Size sz    = get<0>(GetParam());
    int  flags = get<1>(GetParam());

    Mat src(sz, CV_8UC3);
    randu(src, 0, 256);
    Mat dst(sz, CV_8UC3);

    declare.in(src).out(dst).time(120);

    TEST_CYCLE() cv::edgePreservingFilter(src, dst, flags, 60.0f, 0.4f);

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<Size> Perf_DetailEnhance;

PERF_TEST_P(Perf_DetailEnhance, detailEnhance,
    testing::Values(::perf::szSmall64,
                    Size(128, 128),
                    ::perf::szVGA,
                    ::perf::sz720p,
                    ::perf::sz1080p))
{
    Size sz = GetParam();

    Mat src(sz, CV_8UC3);
    randu(src, 0, 256);
    Mat dst(sz, CV_8UC3);

    declare.in(src).out(dst).time(120);

    TEST_CYCLE() cv::detailEnhance(src, dst, 10.0f, 0.15f);

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<Size> Perf_Stylization;

PERF_TEST_P(Perf_Stylization, stylization,
    testing::Values(::perf::szSmall64,
                    Size(128, 128),
                    ::perf::szVGA,
                    ::perf::sz720p,
                    ::perf::sz1080p))
{
    Size sz = GetParam();

    Mat src(sz, CV_8UC3);
    randu(src, 0, 256);
    Mat dst(sz, CV_8UC3);

    declare.in(src).out(dst).time(120);

    TEST_CYCLE() cv::stylization(src, dst, 60.0f, 0.45f);

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<Size> Perf_PencilSketch;

PERF_TEST_P(Perf_PencilSketch, pencilSketch,
    testing::Values(::perf::szSmall64,
                    Size(128, 128),
                    ::perf::szVGA,
                    ::perf::sz720p,
                    ::perf::sz1080p))
{
    Size sz = GetParam();

    Mat src(sz, CV_8UC3);
    randu(src, 0, 256);
    Mat sketch, color_sketch;

    declare.in(src).time(120);

    TEST_CYCLE() cv::pencilSketch(src, sketch, color_sketch, 60.0f, 0.07f, 0.02f);

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
