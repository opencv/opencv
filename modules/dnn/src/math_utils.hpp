// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Code is borrowed from https://github.com/kaldi-asr/kaldi/blob/master/src/base/kaldi-math.h

// base/kaldi-math.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Yanmin Qian;
//                      Jan Silovsky;  Saarland University
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef __OPENCV_DNN_MATH_UTILS_HPP__
#define __OPENCV_DNN_MATH_UTILS_HPP__

#ifdef OS_QNX
#include <math.h>
#else
#include <cmath>
#endif

#include <limits>

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-7f
#endif

namespace cv { namespace dnn {

const float kNegativeInfinity = -std::numeric_limits<float>::infinity();

const float kMinLogDiffFloat = std::log(FLT_EPSILON);

#if !defined(_MSC_VER) || (_MSC_VER >= 1700)
inline float Log1p(float x) {  return log1pf(x); }
#else
inline float Log1p(float x) {
  const float cutoff = 1.0e-07;
  if (x < cutoff)
    return x - 2 * x * x;
  else
    return Log(1.0 + x);
}
#endif

inline float Exp(float x) { return expf(x); }

inline float LogAdd(float x, float y) {
  float diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffFloat) {
    float res;
    res = x + Log1p(Exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

}}  // namespace

#endif  // __OPENCV_DNN_MATH_UTILS_HPP__
