// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __ZXING_COMMON_DETECTOR_MATH_UTILS_HPP__
#define __ZXING_COMMON_DETECTOR_MATH_UTILS_HPP__
/*
 *  Copyright 2012 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || \
    defined(__MINGW32__) || defined(__BORLANDC__)
#include <emmintrin.h>
#endif

#include <cmath>
#if (defined __GNUC__ && defined __x86_64__ && defined __SSE2__ && !defined __APPLE__ && !defined __GXX_WEAK__)
#include <ammintrin.h>
#endif

#include <numeric>
#include <vector>
#include <algorithm>

namespace zxing {
namespace common {
namespace detector {

class MathUtils {
 private:
  MathUtils();
  ~MathUtils();
 public:

  /**
   * Ends up being a bit faster than {@link Math#round(float)}. This merely rounds its
   * argument to the nearest int, where x.5 rounds up to x+1.
   */
static inline int round(double value) {
// return (int) (d + 0.5f);

#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ && defined __SSE2__ && !defined __APPLE__ && !defined __GXX_WEAK__)
    __m128d t = _mm_set_sd(value);
    return _mm_cvtsd_si32(t);
#elif defined _MSC_VER && defined _M_IX86
    int t;
    __asm
    {
        fld value;
        fistp t;
    }
    return t;
#elif defined _MSC_VER && defined _M_ARM && defined HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND(value);
#elif defined HAVE_LRINT || defined CV_ICC || defined __GNUC__
#  ifdef HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND(value);
#  else
    return (int)lrint(value);
#  endif
#else
    double intpart, fractpart;
    fractpart = modf(value, &intpart);
    if ((fabs(fractpart) != 0.5) || ((((int)intpart) % 2) != 0))
        return (int)(value + (value >= 0 ? 0.5 : -0.5));
    else
        return (int)intpart;
#endif

}

static inline float distance(float aX, float aY, float bX, float bY) {
    float xDiff = aX - bX;
    float yDiff = aY - bY;
    return sqrt(float(xDiff * xDiff + yDiff * yDiff));
}

static inline float distance_4_int(int aX, int aY, int bX, int bY) {
    return sqrt(float((aX-bX)*(aX-bX)+(aY-bY)*(aY-bY)));
}

static inline void getRangeValues(int& minValue, int& maxValue, int min, int max) {
    int finalMinValue, finalMaxValue;

    if (minValue < maxValue)
    {
        finalMinValue = minValue;
        finalMaxValue = maxValue;
    } 
    else
    {
        finalMinValue = maxValue;
        finalMaxValue = minValue;
    }

    finalMinValue = finalMinValue > min ? finalMinValue : min;
    finalMaxValue = finalMaxValue < max ? finalMaxValue : max;

    minValue = finalMinValue;
    maxValue = finalMaxValue;
}

static inline bool isInRange(float x, float y, float width, float height)
{
    if ((x >= 0.0 && x <= (width - 1.0)) && (y >= 0.0 && y <= (height - 1.0)))
    {
        return true;
    } 
    else
    {
        return false;
    }
}

static inline float distance(int aX, int aY, int bX, int bY) {
    int xDiff = aX - bX;
    int yDiff = aY - bY;
    return sqrt(float(xDiff * xDiff + yDiff * yDiff));
}

static inline float vecCross(float* v1, float* v2)
{
    return v1[0] * v2[1] - v1[1] * v2[0];
}
  
static inline void  stddev(std::vector<float> & resultSet, float & avg, float & stddev)
{
    double sum = std::accumulate(resultSet.begin(), resultSet.end(), 0.0);
        avg = sum / resultSet.size(); 
        
        double accum = 0.0;
        for (size_t i = 0; i < resultSet.size(); i++)
        {
            accum += (resultSet[i] - avg)*(resultSet[i] - avg);
        }
    
        stddev = sqrt(accum / (resultSet.size()));
}

};

}
}
}

#endif // __ZXING_COMMON_DETECTOR_MATH_UTILS_HPP__
