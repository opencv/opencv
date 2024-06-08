// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////// 3 channels Vs 4 ////////////////////////

enum
{
    Pure = 0, Split, Convert
};

CV_ENUM(Modes, Pure, Split, Convert)

typedef tuple <Size, MatType, Modes> _3vs4Params;
typedef TestBaseWithParam<_3vs4Params> _3vs4_Fixture;

OCL_PERF_TEST_P(_3vs4_Fixture, Resize,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC3, CV_32FC3), Modes::all()))
{
    _3vs4Params params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), depth = CV_MAT_DEPTH(type);
    const int mode = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (mode == Pure)
    {
        OCL_TEST_CYCLE() resize(src, dst, Size(), 0.5, 0.5, INTER_LINEAR_EXACT);
    }
    else if (mode == Split)
    {
        std::vector<UMat> srcs(3), dsts(3);

        for (int i = 0; i < 3; ++i)
        {
            dsts[i] = UMat(srcSize, depth);
            srcs[i] = UMat(srcSize, depth);
        }

        OCL_TEST_CYCLE()
        {
            split(src, srcs);

            for (size_t i = 0; i < srcs.size(); ++i)
                resize(srcs[i], dsts[i], Size(), 0.5, 0.5, INTER_LINEAR_EXACT);

            merge(dsts, dst);
        }
    }
    else if (mode == Convert)
    {
        int type4 = CV_MAKE_TYPE(depth, 4);
        UMat src4(srcSize, type4), dst4(srcSize, type4);

        OCL_TEST_CYCLE()
        {
            cvtColor(src, src4, COLOR_RGB2RGBA);
            resize(src4, dst4, Size(), 0.5, 0.5, INTER_LINEAR_EXACT);
            cvtColor(dst4, dst, COLOR_RGBA2RGB);
        }
    }

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(_3vs4_Fixture, Subtract,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC3, CV_32FC3), Modes::all()))
{
    _3vs4Params params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), depth = CV_MAT_DEPTH(type);
    const int mode = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Scalar s(14);
    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (mode == Pure)
    {
        OCL_TEST_CYCLE() subtract(src, s, dst);
    }
    else if (mode == Split)
    {
        std::vector<UMat> srcs(3), dsts(3);

        for (int i = 0; i < 3; ++i)
        {
            dsts[i] = UMat(srcSize, depth);
            srcs[i] = UMat(srcSize, depth);
        }

        OCL_TEST_CYCLE()
        {
            split(src, srcs);

            for (size_t i = 0; i < srcs.size(); ++i)
                subtract(srcs[i], s, dsts[i]);

            merge(dsts, dst);
        }
    }
    else if (mode == Convert)
    {
        int type4 = CV_MAKE_TYPE(depth, 4);
        UMat src4(srcSize, type4), dst4(srcSize, type4);

        OCL_TEST_CYCLE()
        {
            cvtColor(src, src4, COLOR_RGB2RGBA);
            subtract(src4, s, dst4);
            cvtColor(dst4, dst, COLOR_RGBA2RGB);
        }
    }

    SANITY_CHECK_NOTHING();
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
