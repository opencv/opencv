// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, all rights reserved.

#include "precomp.hpp"

#include "remap.simd.hpp"
#include "remap.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX512_ICL,...,BASELINE

namespace cv {

// Separate dispatch unit from warp_kernels. imgwarp.cpp already includes
// warp_kernels.simd_declarations.hpp, so these kernels cannot be dispatched
// with CV_CPU_DISPATCH in that translation unit.
// Callers cache the pointer returned here and invoke it directly; resolving on
// every row costs about 10%.

using RemapBilinearC1Fn = int (*)(int, const uchar*, size_t, uchar*, const short*, const ushort*, const float*, int, int, int);
using RemapWideC1Fn = int (*)(int, const uchar*, size_t, uchar*, const short*, const ushort*, const float*, int, int, unsigned, unsigned, int);

RemapBilinearC1Fn remapBilinearC1_fn();
RemapWideC1Fn remapBicubicC1wp_fn();
RemapWideC1Fn remapLanczos4C1_fn();

static RemapBilinearC1Fn resolveBilinear()
{
#if CV_TRY_AVX512_ICL
    if (cv::checkHardwareSupport(CV_CPU_AVX512_ICL))
        return opt_AVX512_ICL::remapBilinearC1_simd;
#endif
#if CV_TRY_AVX512_SKX
    if (cv::checkHardwareSupport(CV_CPU_AVX512_SKX))
        return opt_AVX512_SKX::remapBilinearC1_simd;
#endif
    return cpu_baseline::remapBilinearC1_simd;
}

static RemapWideC1Fn resolveBicubic()
{
#if CV_TRY_AVX512_ICL
    if (cv::checkHardwareSupport(CV_CPU_AVX512_ICL))
        return opt_AVX512_ICL::remapBicubicC1wp_simd;
#endif
#if CV_TRY_AVX512_SKX
    if (cv::checkHardwareSupport(CV_CPU_AVX512_SKX))
        return opt_AVX512_SKX::remapBicubicC1wp_simd;
#endif
    return cpu_baseline::remapBicubicC1wp_simd;
}

static RemapWideC1Fn resolveLanczos()
{
#if CV_TRY_AVX512_ICL
    if (cv::checkHardwareSupport(CV_CPU_AVX512_ICL))
        return opt_AVX512_ICL::remapLanczos4C1_simd;
#endif
#if CV_TRY_AVX512_SKX
    if (cv::checkHardwareSupport(CV_CPU_AVX512_SKX))
        return opt_AVX512_SKX::remapLanczos4C1_simd;
#endif
    return cpu_baseline::remapLanczos4C1_simd;
}

RemapBilinearC1Fn remapBilinearC1_fn()
{
    static RemapBilinearC1Fn fn = resolveBilinear();
    return fn;
}

RemapWideC1Fn remapBicubicC1wp_fn()
{
    static RemapWideC1Fn fn = resolveBicubic();
    return fn;
}

RemapWideC1Fn remapLanczos4C1_fn()
{
    static RemapWideC1Fn fn = resolveLanczos();
    return fn;
}

} // namespace cv
