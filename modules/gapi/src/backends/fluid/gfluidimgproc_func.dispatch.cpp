// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "gfluidimgproc_func.hpp"
#include "gfluidimgproc_func.simd.hpp"
#if 1
  // NB: workaround for CV_SIMD bug (or feature?):
  // - dynamic dispatcher assumes *.simd.hpp is directly in src dir
  #include "backends/fluid/gfluidimgproc_func.simd_declarations.hpp"
#else
  #include                "gfluidimgproc_func.simd_declarations.hpp"
#endif

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
