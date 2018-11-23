// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#pragma once

#if !defined(GAPI_STANDALONE)

#include "opencv2/core.hpp"

namespace cv {
namespace gapi {
namespace fluid {

//----------------------------------
//
// Fluid kernels: RGB2Gray, BGR2Gray
//
//----------------------------------

void run_rgb2gray_impl(uchar out[], const uchar in[], int width,
                       float coef_r, float coef_g, float coef_b);

//--------------------------------------
//
// Fluid kernels: RGB-to-YUV, YUV-to-RGB
//
//--------------------------------------

void run_rgb2yuv_impl(uchar out[], const uchar in[], int width, const float coef[5]);

void run_yuv2rgb_impl(uchar out[], const uchar in[], int width, const float coef[4]);

//---------------------
//
// Fluid kernels: Sobel
//
//---------------------

#define RUN_SOBEL_WITH_BUF 1  // 1=run_sobel_row() using buf[], 0=not using buf[]

#if RUN_SOBEL_WITH_BUF

#define RUN_SOBEL_ROW(DST, SRC)                                     \
void run_sobel_row(DST out[], const SRC *in[], int width, int chan, \
                   const float kx[], const float ky[], int border,  \
                   float scale, float delta, float *buf[],          \
                   int y, int y0);

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

#else  // if not RUN_SOBEL_WITH_BUF

#define RUN_SOBEL_ROW1(DST, SRC)                                     \
void run_sobel_row1(DST out[], const SRC *in[], int width, int chan, \
                    const float kx[], const float ky[], int border,  \
                    float scale, float delta);

RUN_SOBEL_ROW1(uchar , uchar )
RUN_SOBEL_ROW1(ushort, ushort)
RUN_SOBEL_ROW1( short, uchar )
RUN_SOBEL_ROW1( short, ushort)
RUN_SOBEL_ROW1( short,  short)
RUN_SOBEL_ROW1( float, uchar )
RUN_SOBEL_ROW1( float, ushort)
RUN_SOBEL_ROW1( float,  short)
RUN_SOBEL_ROW1( float,  float)

#undef RUN_SOBEL_ROW1

#endif  // RUN_SOBEL_WITH_BUF

//-------------------------
//
// Fluid kernels: sepFilter
//
//-------------------------

#define RUN_SEPFILTER3X3_IMPL(DST, SRC)                                     \
void run_sepfilter3x3_impl(DST out[], const SRC *in[], int width, int chan, \
                           const float kx[], const float ky[], int border,  \
                           float scale, float delta,                        \
                           float *buf[], int y, int y0);

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

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
