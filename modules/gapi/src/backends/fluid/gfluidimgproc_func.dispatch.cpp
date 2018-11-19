// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "gfluidimgproc_func.hpp"
#include "gfluidimgproc_func.simd.hpp"
#include "backends/fluid/gfluidimgproc_func.simd_declarations.hpp"

#include "gfluidutils.hpp"

#include "opencv2/core/cvdef.h"
#include "opencv2/core/hal/intrin.hpp"

#include <cmath>
#include <cstdlib>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif

namespace cv {
namespace gapi {
namespace fluid {

//----------------------------------
//
// Fluid kernels: RGB2Gray, BGR2Gray
//
//----------------------------------

void run_rgb2gray_impl(uchar out[], const uchar in[], int width,
                       float coef_r, float coef_g, float coef_b)
{
    CV_CPU_DISPATCH(run_rgb2gray_impl,
        (out, in, width, coef_r, coef_g, coef_b),
        CV_CPU_DISPATCH_MODES_ALL);
}

//--------------------------------------
//
// Fluid kernels: RGB-to-YUV, YUV-to-RGB
//
//--------------------------------------

void run_rgb2yuv_impl(uchar out[], const uchar in[], int width, const float coef[5])
{
    CV_CPU_DISPATCH(run_rgb2yuv_impl, (out, in, width, coef), CV_CPU_DISPATCH_MODES_ALL);
}

void run_yuv2rgb_impl(uchar out[], const uchar in[], int width, const float coef[4])
{
    CV_CPU_DISPATCH(run_yuv2rgb_impl, (out, in, width, coef), CV_CPU_DISPATCH_MODES_ALL);
}

//---------------------
//
// Fluid kernels: Sobel
//
//---------------------

#define RUN_SOBEL_ROW(DST, SRC)                                          \
void run_sobel_row(DST out[], const SRC *in[], int width, int chan,      \
                   const float kx[], const float ky[], int border,       \
                   float scale, float delta, float *buf[],               \
                   int y, int y0)                                        \
{                                                                        \
    CV_CPU_DISPATCH(run_sobel_row,                                       \
        (out, in, width, chan, kx, ky, border, scale, delta, buf,y, y0), \
        CV_CPU_DISPATCH_MODES_ALL);                                      \
}

RUN_SOBEL_ROW(uchar , uchar )
RUN_SOBEL_ROW(ushort, ushort)
RUN_SOBEL_ROW( short, uchar )
RUN_SOBEL_ROW( short, ushort)
RUN_SOBEL_ROW( short,  short)
RUN_SOBEL_ROW( float, uchar )
RUN_SOBEL_ROW( float, ushort)
RUN_SOBEL_ROW( float,  short)
RUN_SOBEL_ROW( float,  float)

#undef RUN_SOBEL_ROW

} // namespace fliud
} // namespace gapi
} // namespace cv

#endif // !defined(GAPI_STANDALONE)
