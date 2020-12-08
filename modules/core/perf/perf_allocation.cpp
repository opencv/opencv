// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include <array>

using namespace perf;

#define ALLOC_MAT_SIZES ::perf::szSmall24, ::perf::szSmall32, ::perf::szSmall64, \
    ::perf::sz5MP, ::perf::sz2K, ::perf::szSmall128, ::perf::szODD, ::perf::szQVGA, \
    ::perf::szVGA, ::perf::szSVGA, ::perf::sz720p, ::perf::sz1080p, ::perf::sz2160p, \
    ::perf::sz4320p, ::perf::sz3MP, ::perf::szXGA, ::perf::szSXGA, ::perf::szWQHD, \
    ::perf::sznHD, ::perf::szqHD

namespace opencv_test
{

typedef perf::TestBaseWithParam<MatType> MatDepth_tb;

PERF_TEST_P(MatDepth_tb, DISABLED_Allocation_Aligned,
    testing::Values(CV_8UC1, CV_16SC1, CV_8UC3, CV_8UC4))
{
    const int matType = GetParam();
    const cv::Mat utility(1, 1, matType);
    const size_t elementBytes = utility.elemSize();

    const std::array<cv::Size, 20> sizes{ALLOC_MAT_SIZES};
    std::array<size_t, 20> bytes;
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        bytes[i] = sizes[i].width * sizes[i].height * elementBytes;
    }

    declare.time(60)
           .iterations(100);

    TEST_CYCLE()
    {
        for (int i = 0; i < 100000; ++i)
        {
            fastFree(fastMalloc(bytes[i % sizes.size()]));
        }
    }
    SANITY_CHECK_NOTHING();
}

};
