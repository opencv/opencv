// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#pragma once

#if !defined(GAPI_STANDALONE) && (CV_SIMD || CV_SIMD_SCALABLE)

#include <opencv2/core.hpp>

namespace cv {
namespace gapi {
namespace fluid {

#define DIV_SIMD(SRC, DST)                                       \
int div_simd(const SRC in1[], const SRC in2[], DST out[],        \
             const int length, double _scale);

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

#define MUL_SIMD(SRC, DST)                                       \
int mul_simd(const SRC in1[], const SRC in2[], DST out[],        \
             const int length, double _scale);

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

#define ADDC_SIMD(SRC, DST)                                             \
int addc_simd(const SRC in[], const float scalar[], DST out[],          \
              const int length, const int chan);

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

#define SUBC_SIMD(SRC, DST)                                        \
int subc_simd(const SRC in[], const float scalar[], DST out[],     \
              const int length, const int chan);

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

#define SUBRC_SIMD(SRC, DST)                                                             \
int subrc_simd(const float scalar[], const SRC in[], DST out[],                          \
               const int length, const int chan);

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

#define MULC_SIMD(SRC, DST)                                                              \
int mulc_simd(const SRC in[], const float scalar[], DST out[],                           \
              const int length, const int chan, const float scale);

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

#define DIVC_SIMD(SRC, DST)                                                              \
int divc_simd(const SRC in[], const float scalar[], DST out[],                           \
              const int length, const int chan, const float scale,                       \
              const int set_mask_flag);

DIVC_SIMD(uchar, uchar)
DIVC_SIMD(ushort, uchar)
DIVC_SIMD(short, uchar)
DIVC_SIMD(float, uchar)
DIVC_SIMD(short, short)
DIVC_SIMD(ushort, short)
DIVC_SIMD(uchar, short)
DIVC_SIMD(float, short)
DIVC_SIMD(ushort, ushort)
DIVC_SIMD(uchar, ushort)
DIVC_SIMD(short, ushort)
DIVC_SIMD(float, ushort)
DIVC_SIMD(uchar, float)
DIVC_SIMD(ushort, float)
DIVC_SIMD(short, float)
DIVC_SIMD(float, float)

#undef DIVC_SIMD

#define ABSDIFFC_SIMD(T)                                            \
int absdiffc_simd(const T in[], const float scalar[], T out[],      \
                  const int length, const int chan);

ABSDIFFC_SIMD(uchar)
ABSDIFFC_SIMD(short)
ABSDIFFC_SIMD(ushort)
ABSDIFFC_SIMD(float)

#undef ABSDIFFC_SIMD

#define DIVRC_SIMD(SRC, DST)                                           \
int divrc_simd(const float scalar[], const SRC in[], DST out[],        \
               const int length, const int chan, const float scale);

DIVRC_SIMD(uchar, uchar)
DIVRC_SIMD(ushort, uchar)
DIVRC_SIMD(short, uchar)
DIVRC_SIMD(float, uchar)
DIVRC_SIMD(short, short)
DIVRC_SIMD(ushort, short)
DIVRC_SIMD(uchar, short)
DIVRC_SIMD(float, short)
DIVRC_SIMD(ushort, ushort)
DIVRC_SIMD(uchar, ushort)
DIVRC_SIMD(short, ushort)
DIVRC_SIMD(float, ushort)
DIVRC_SIMD(uchar, float)
DIVRC_SIMD(ushort, float)
DIVRC_SIMD(short, float)
DIVRC_SIMD(float, float)

#undef DIVRC_SIMD

int split3_simd(const uchar in[], uchar out1[], uchar out2[],
                uchar out3[], const int width);

int split4_simd(const uchar in[], uchar out1[], uchar out2[],
                uchar out3[], uchar out4[], const int width);

#define MERGE3_SIMD(T)                                          \
int merge3_simd(const T in1[], const T in2[], const T in3[],    \
                T out[], const int width);

MERGE3_SIMD(uchar)
MERGE3_SIMD(short)
MERGE3_SIMD(ushort)
MERGE3_SIMD(float)

#undef MERGE3_SIMD

int merge4_simd(const uchar in1[], const uchar in2[], const uchar in3[],
                const uchar in4[], uchar out[], const int width);

#define ADD_SIMD(SRC, DST)                                                     \
int add_simd(const SRC in1[], const SRC in2[], DST out[], const int length);

ADD_SIMD(uchar, uchar)
ADD_SIMD(ushort, uchar)
ADD_SIMD(short, uchar)
ADD_SIMD(float, uchar)
ADD_SIMD(short, short)
ADD_SIMD(ushort, short)
ADD_SIMD(uchar, short)
ADD_SIMD(float, short)
ADD_SIMD(ushort, ushort)
ADD_SIMD(uchar, ushort)
ADD_SIMD(short, ushort)
ADD_SIMD(float, ushort)
ADD_SIMD(uchar, float)
ADD_SIMD(ushort, float)
ADD_SIMD(short, float)
ADD_SIMD(float, float)

#undef ADD_SIMD

#define SUB_SIMD(SRC, DST)                                                     \
int sub_simd(const SRC in1[], const SRC in2[], DST out[], const int length);

SUB_SIMD(uchar, uchar)
SUB_SIMD(ushort, uchar)
SUB_SIMD(short, uchar)
SUB_SIMD(float, uchar)
SUB_SIMD(short, short)
SUB_SIMD(ushort, short)
SUB_SIMD(uchar, short)
SUB_SIMD(float, short)
SUB_SIMD(ushort, ushort)
SUB_SIMD(uchar, ushort)
SUB_SIMD(short, ushort)
SUB_SIMD(float, ushort)
SUB_SIMD(uchar, float)
SUB_SIMD(ushort, float)
SUB_SIMD(short, float)
SUB_SIMD(float, float)

#undef SUB_SIMD

#define CONVERTTO_NOCOEF_SIMD(SRC, DST)                                                    \
int convertto_simd(const SRC in[], DST out[], const int length);

CONVERTTO_NOCOEF_SIMD(ushort, uchar)
CONVERTTO_NOCOEF_SIMD(short, uchar)
CONVERTTO_NOCOEF_SIMD(float, uchar)
CONVERTTO_NOCOEF_SIMD(ushort, short)
CONVERTTO_NOCOEF_SIMD(uchar, short)
CONVERTTO_NOCOEF_SIMD(float, short)
CONVERTTO_NOCOEF_SIMD(uchar, ushort)
CONVERTTO_NOCOEF_SIMD(short, ushort)
CONVERTTO_NOCOEF_SIMD(float, ushort)
CONVERTTO_NOCOEF_SIMD(uchar, float)
CONVERTTO_NOCOEF_SIMD(ushort, float)
CONVERTTO_NOCOEF_SIMD(short, float)

#undef CONVERTTO_NOCOEF_SIMD

#define CONVERTTO_SCALED_SIMD(SRC, DST)                                     \
int convertto_scaled_simd(const SRC in[], DST out[], const float alpha,     \
                          const float beta, const int length);

CONVERTTO_SCALED_SIMD(uchar, uchar)
CONVERTTO_SCALED_SIMD(ushort, uchar)
CONVERTTO_SCALED_SIMD(short, uchar)
CONVERTTO_SCALED_SIMD(float, uchar)
CONVERTTO_SCALED_SIMD(short, short)
CONVERTTO_SCALED_SIMD(ushort, short)
CONVERTTO_SCALED_SIMD(uchar, short)
CONVERTTO_SCALED_SIMD(float, short)
CONVERTTO_SCALED_SIMD(ushort, ushort)
CONVERTTO_SCALED_SIMD(uchar, ushort)
CONVERTTO_SCALED_SIMD(short, ushort)
CONVERTTO_SCALED_SIMD(float, ushort)
CONVERTTO_SCALED_SIMD(uchar, float)
CONVERTTO_SCALED_SIMD(ushort, float)
CONVERTTO_SCALED_SIMD(short, float)
CONVERTTO_SCALED_SIMD(float, float)

#undef CONVERTTO_SCALED_SIMD

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
