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

//---------------------
//
// Fluid kernels: Sobel
//
//---------------------

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

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
