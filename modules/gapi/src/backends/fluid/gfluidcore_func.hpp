// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#pragma once

#if !defined(GAPI_STANDALONE)

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

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
