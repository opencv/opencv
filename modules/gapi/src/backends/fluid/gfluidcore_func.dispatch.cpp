// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "gfluidcore_func.hpp"
#include "gfluidcore_func.simd.hpp"

#include "backends/fluid/gfluidcore_func.simd_declarations.hpp"

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

#define DIV_SIMD(SRC, DST)                                                  \
int div_simd(const SRC in1[], const SRC in2[], DST out[],                   \
             const int length, double _scale)                               \
{                                                                           \
    CV_CPU_DISPATCH(div_simd, (in1, in2, out, length, _scale),              \
                    CV_CPU_DISPATCH_MODES_ALL);                             \
}


DIV_SIMD(uchar, uchar)
DIV_SIMD(ushort, uchar)
DIV_SIMD(short, uchar)
DIV_SIMD(float, uchar)
DIV_SIMD(short, short)
DIV_SIMD(ushort, short)
DIV_SIMD(uchar, short)
DIV_SIMD(float, short)
DIV_SIMD(ushort, ushort)
DIV_SIMD(uchar, ushort)
DIV_SIMD(short, ushort)
DIV_SIMD(float, ushort)
DIV_SIMD(uchar, float)
DIV_SIMD(ushort, float)
DIV_SIMD(short, float)
DIV_SIMD(float, float)

#undef DIV_SIMD


#define MUL_SIMD(SRC, DST)                                                  \
int mul_simd(const SRC in1[], const SRC in2[], DST out[],                   \
             const int length, double _scale)                               \
{                                                                           \
    CV_CPU_DISPATCH(mul_simd, (in1, in2, out, length, _scale),              \
                    CV_CPU_DISPATCH_MODES_ALL);                             \
}

MUL_SIMD(uchar, uchar)
MUL_SIMD(ushort, uchar)
MUL_SIMD(short, uchar)
MUL_SIMD(float, uchar)
MUL_SIMD(short, short)
MUL_SIMD(ushort, short)
MUL_SIMD(uchar, short)
MUL_SIMD(float, short)
MUL_SIMD(ushort, ushort)
MUL_SIMD(uchar, ushort)
MUL_SIMD(short, ushort)
MUL_SIMD(float, ushort)
MUL_SIMD(uchar, float)
MUL_SIMD(ushort, float)
MUL_SIMD(short, float)
MUL_SIMD(float, float)

#undef MUL_SIMD

#define ADDC_SIMD(SRC, DST)                                               \
int addc_simd(const SRC in[], const float scalar[], DST out[],            \
              const int length, const int chan)                           \
{                                                                         \
    CV_CPU_DISPATCH(addc_simd, (in, scalar, out, length, chan),           \
                    CV_CPU_DISPATCH_MODES_ALL);                           \
}

ADDC_SIMD(uchar, uchar)
ADDC_SIMD(ushort, uchar)
ADDC_SIMD(short, uchar)
ADDC_SIMD(float, uchar)
ADDC_SIMD(short, short)
ADDC_SIMD(ushort, short)
ADDC_SIMD(uchar, short)
ADDC_SIMD(float, short)
ADDC_SIMD(ushort, ushort)
ADDC_SIMD(uchar, ushort)
ADDC_SIMD(short, ushort)
ADDC_SIMD(float, ushort)
ADDC_SIMD(uchar, float)
ADDC_SIMD(ushort, float)
ADDC_SIMD(short, float)
ADDC_SIMD(float, float)

#undef ADDC_SIMD

#define SUBC_SIMD(SRC, DST)                                               \
int subc_simd(const SRC in[], const float scalar[], DST out[],            \
              const int length, const int chan)                           \
{                                                                         \
    CV_CPU_DISPATCH(subc_simd, (in, scalar, out, length, chan),           \
                    CV_CPU_DISPATCH_MODES_ALL);                           \
}

SUBC_SIMD(uchar, uchar)
SUBC_SIMD(ushort, uchar)
SUBC_SIMD(short, uchar)
SUBC_SIMD(float, uchar)
SUBC_SIMD(short, short)
SUBC_SIMD(ushort, short)
SUBC_SIMD(uchar, short)
SUBC_SIMD(float, short)
SUBC_SIMD(ushort, ushort)
SUBC_SIMD(uchar, ushort)
SUBC_SIMD(short, ushort)
SUBC_SIMD(float, ushort)
SUBC_SIMD(uchar, float)
SUBC_SIMD(ushort, float)
SUBC_SIMD(short, float)
SUBC_SIMD(float, float)

#undef SUBC_SIMD

#define SUBRC_SIMD(SRC, DST)                                              \
int subrc_simd(const float scalar[], const SRC in[], DST out[],           \
               const int length, const int chan)                          \
{                                                                         \
    CV_CPU_DISPATCH(subrc_simd, (scalar, in, out, length, chan),          \
                    CV_CPU_DISPATCH_MODES_ALL);                           \
}

SUBRC_SIMD(uchar, uchar)
SUBRC_SIMD(ushort, uchar)
SUBRC_SIMD(short, uchar)
SUBRC_SIMD(float, uchar)
SUBRC_SIMD(short, short)
SUBRC_SIMD(ushort, short)
SUBRC_SIMD(uchar, short)
SUBRC_SIMD(float, short)
SUBRC_SIMD(ushort, ushort)
SUBRC_SIMD(uchar, ushort)
SUBRC_SIMD(short, ushort)
SUBRC_SIMD(float, ushort)
SUBRC_SIMD(uchar, float)
SUBRC_SIMD(ushort, float)
SUBRC_SIMD(short, float)
SUBRC_SIMD(float, float)

#undef SUBRC_SIMD

#define MULC_SIMD(SRC, DST)                                               \
int mulc_simd(const SRC in[], const float scalar[], DST out[],            \
              const int length, const int chan, const float scale)        \
{                                                                         \
    CV_CPU_DISPATCH(mulc_simd, (in, scalar, out, length, chan, scale),    \
                    CV_CPU_DISPATCH_MODES_ALL);                           \
}

MULC_SIMD(uchar, uchar)
MULC_SIMD(ushort, uchar)
MULC_SIMD(short, uchar)
MULC_SIMD(float, uchar)
MULC_SIMD(short, short)
MULC_SIMD(ushort, short)
MULC_SIMD(uchar, short)
MULC_SIMD(float, short)
MULC_SIMD(ushort, ushort)
MULC_SIMD(uchar, ushort)
MULC_SIMD(short, ushort)
MULC_SIMD(float, ushort)
MULC_SIMD(uchar, float)
MULC_SIMD(ushort, float)
MULC_SIMD(short, float)
MULC_SIMD(float, float)

#undef MULC_SIMD

#define ABSDIFFC_SIMD(SRC)                                               \
int absdiffc_simd(const SRC in[], const float scalar[], SRC out[],       \
                  const int length, const int chan)                      \
{                                                                        \
    CV_CPU_DISPATCH(absdiffc_simd, (in, scalar, out, length, chan),      \
                    CV_CPU_DISPATCH_MODES_ALL);                          \
}

ABSDIFFC_SIMD(uchar)
ABSDIFFC_SIMD(short)
ABSDIFFC_SIMD(ushort)
ABSDIFFC_SIMD(float)

#undef ABSDIFFC_SIMD

} // namespace fluid
} // namespace gapi
} // namespace cv

#endif // !defined(GAPI_STANDALONE)
