// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)
CV_ENUM(TargetDepth, CV_8U, CV_16S)

typedef tuple<Size, int, TargetDepth, BorderMode> LaplacianParams;
typedef perf::TestBaseWithParam<LaplacianParams> Perf_Laplacian;

PERF_TEST_P(Perf_Laplacian, Laplacian,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(1, 3, 5),                // ksize: 1, 3, 5
                TargetDepth::all(),                      // CV_8U and CV_16S
                BorderMode::all()
            ))
{
    Size sz        = get<0>(GetParam());
    int ksize      = get<1>(GetParam());
    int ddepth     = get<2>(GetParam());
    int borderMode = get<3>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, ddepth == CV_16S ? CV_16SC1 : CV_8UC1);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE()
    {
        cv::Laplacian(src, dst, ddepth, ksize, 1.0, 0.0, borderMode);
    }

    SANITY_CHECK(dst);
}

} // namespace opencv_test