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

#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/intrin.hpp>

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
// Fluid kernels: RGB-to-HSV
//
//--------------------------------------

void run_rgb2hsv_impl(uchar out[], const uchar in[], const int sdiv_table[],
                      const int hdiv_table[], int width)
{
    CV_CPU_DISPATCH(run_rgb2hsv_impl, (out, in, sdiv_table, hdiv_table, width), CV_CPU_DISPATCH_MODES_ALL);
}

//--------------------------------------
//
// Fluid kernels: RGB-to-BayerGR
//
//--------------------------------------

void run_bayergr2rgb_bg_impl(uchar out[], const uchar **in, int width)
{
    CV_CPU_DISPATCH(run_bayergr2rgb_bg_impl, (out, in, width), CV_CPU_DISPATCH_MODES_ALL);
}

void run_bayergr2rgb_gr_impl(uchar out[], const uchar **in, int width)
{
    CV_CPU_DISPATCH(run_bayergr2rgb_gr_impl, (out, in, width), CV_CPU_DISPATCH_MODES_ALL);
}

//--------------------------------------
//
// Fluid kernels: RGB-to-YUV, RGB-to-YUV422, YUV-to-RGB
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

void run_rgb2yuv422_impl(uchar out[], const uchar in[], int width)
{
    CV_CPU_DISPATCH(run_rgb2yuv422_impl, (out, in, width), CV_CPU_DISPATCH_MODES_ALL);
}

//-------------------------
//
// Fluid kernels: sepFilter
//
//-------------------------

#define RUN_SEPFILTER3X3_IMPL(DST, SRC)                                     \
void run_sepfilter3x3_impl(DST out[], const SRC *in[], int width, int chan, \
                           const float kx[], const float ky[], int border,  \
                           float scale, float delta,                        \
                           float *buf[], int y, int y0)                     \
{                                                                           \
    CV_CPU_DISPATCH(run_sepfilter3x3_impl,                                  \
        (out, in, width, chan, kx, ky, border, scale, delta, buf,y, y0),    \
        CV_CPU_DISPATCH_MODES_ALL);                                         \
}

RUN_SEPFILTER3X3_IMPL(uchar , uchar )
RUN_SEPFILTER3X3_IMPL( short, uchar )
RUN_SEPFILTER3X3_IMPL( float, uchar )
RUN_SEPFILTER3X3_IMPL(ushort, ushort)
RUN_SEPFILTER3X3_IMPL( short, ushort)
RUN_SEPFILTER3X3_IMPL( float, ushort)
RUN_SEPFILTER3X3_IMPL( short,  short)
RUN_SEPFILTER3X3_IMPL( float,  short)
RUN_SEPFILTER3X3_IMPL( float,  float)

#undef RUN_SEPFILTER3X3_IMPL

#define RUN_SEPFILTER5x5_IMPL(DST, SRC)                                     \
void run_sepfilter5x5_impl(DST out[], const SRC *in[], int width, int chan, \
                           const float kx[], const float ky[], int border,  \
                           float scale, float delta,                        \
                           float *buf[], int y, int y0)                     \
{                                                                           \
    CV_CPU_DISPATCH(run_sepfilter5x5_impl,                                  \
        (out, in, width, chan, kx, ky, border, scale, delta, buf,y, y0),    \
        CV_CPU_DISPATCH_MODES_ALL);                                         \
}

RUN_SEPFILTER5x5_IMPL(uchar, uchar)
RUN_SEPFILTER5x5_IMPL(short, uchar)
RUN_SEPFILTER5x5_IMPL(float, uchar)
RUN_SEPFILTER5x5_IMPL(ushort, ushort)
RUN_SEPFILTER5x5_IMPL(short, ushort)
RUN_SEPFILTER5x5_IMPL(float, ushort)
RUN_SEPFILTER5x5_IMPL(short, short)
RUN_SEPFILTER5x5_IMPL(float, short)
RUN_SEPFILTER5x5_IMPL(float, float)

#undef RUN_SEPFILTER5x5_IMPL
//-------------------------
//
// Fluid kernels: Filter 2D
//
//-------------------------

#define RUN_FILTER2D_3X3_IMPL(DST, SRC)                                     \
void run_filter2d_3x3_impl(DST out[], const SRC *in[], int width, int chan, \
                           const float kernel[], float scale, float delta)  \
{                                                                           \
    CV_CPU_DISPATCH(run_filter2d_3x3_impl,                                  \
        (out, in, width, chan, kernel, scale, delta),                       \
        CV_CPU_DISPATCH_MODES_ALL);                                         \
}

RUN_FILTER2D_3X3_IMPL(uchar , uchar )
RUN_FILTER2D_3X3_IMPL(ushort, ushort)
RUN_FILTER2D_3X3_IMPL( short,  short)
RUN_FILTER2D_3X3_IMPL( float, uchar )
RUN_FILTER2D_3X3_IMPL( float, ushort)
RUN_FILTER2D_3X3_IMPL( float,  short)
RUN_FILTER2D_3X3_IMPL( float,  float)

#undef RUN_FILTER2D_3X3_IMPL

//-----------------------------
//
// Fluid kernels: Erode, Dilate
//
//-----------------------------

#define RUN_MORPHOLOGY3X3_IMPL(T)                                        \
void run_morphology3x3_impl(T out[], const T *in[], int width, int chan, \
                            const uchar k[], MorphShape k_type,          \
                            Morphology morphology)                       \
{                                                                        \
    CV_CPU_DISPATCH(run_morphology3x3_impl,                              \
        (out, in, width, chan, k, k_type, morphology),                   \
        CV_CPU_DISPATCH_MODES_ALL);                                      \
}

RUN_MORPHOLOGY3X3_IMPL(uchar )
RUN_MORPHOLOGY3X3_IMPL(ushort)
RUN_MORPHOLOGY3X3_IMPL( short)
RUN_MORPHOLOGY3X3_IMPL( float)

#undef RUN_MORPHOLOGY3X3_IMPL

//---------------------------
//
// Fluid kernels: Median blur
//
//---------------------------

#define RUN_MEDBLUR3X3_IMPL(T)                                        \
void run_medblur3x3_impl(T out[], const T *in[], int width, int chan) \
{                                                                     \
    CV_CPU_DISPATCH(run_medblur3x3_impl, (out, in, width, chan),      \
        CV_CPU_DISPATCH_MODES_ALL);                                   \
}

RUN_MEDBLUR3X3_IMPL(uchar )
RUN_MEDBLUR3X3_IMPL(ushort)
RUN_MEDBLUR3X3_IMPL( short)
RUN_MEDBLUR3X3_IMPL( float)

#undef RUN_MEDBLUR3X3_IMPL

} // namespace fliud
} // namespace gapi
} // namespace cv

#endif // !defined(GAPI_STANDALONE)
