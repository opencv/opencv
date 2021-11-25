// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OWN_SATURATE_HPP
#define OPENCV_GAPI_OWN_SATURATE_HPP

#include <math.h>

#include <limits>

#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/util/type_traits.hpp>

namespace cv { namespace gapi { namespace own {
//-----------------------------
//
// Numeric cast with saturation
//
//-----------------------------

template<typename DST, typename SRC,
         typename = cv::util::enable_if_t<!std::is_same<DST, SRC>::value &&
                                           std::is_integral<DST>::value  &&
                                           std::is_integral<SRC>::value>   >
static CV_ALWAYS_INLINE DST saturate(SRC x)
{
    if (sizeof(DST) > sizeof(SRC))
        return static_cast<DST>(x);

    // compiler must recognize this saturation,
    // so compile saturate<s16>(a + b) with adds
    // instruction (e.g.: _mm_adds_epi16 if x86)
    return x < std::numeric_limits<DST>::min()?
               std::numeric_limits<DST>::min():
           x > std::numeric_limits<DST>::max()?
               std::numeric_limits<DST>::max():
           static_cast<DST>(x);
}
template<typename T>
static CV_ALWAYS_INLINE T saturate(T x)
{
    return x;
}

template<typename DST, typename SRC, typename R,
         cv::util::enable_if_t<std::is_floating_point<DST>::value, bool> = true >
static CV_ALWAYS_INLINE DST saturate(SRC x, R)
{
    return static_cast<DST>(x);
}
template<typename DST, typename SRC, typename R,
         cv::util::enable_if_t<std::is_integral<DST>::value &&
                               std::is_integral<SRC>::value   , bool> = true >
static CV_ALWAYS_INLINE DST saturate(SRC x, R)
{
    return saturate<DST>(x);
}
// Note, that OpenCV rounds differently:
// - like std::round() for add, subtract
// - like std::rint() for multiply, divide
template<typename DST, typename SRC, typename R,
         cv::util::enable_if_t<std::is_integral<DST>::value &&
                               std::is_floating_point<SRC>::value, bool> = true >
static CV_ALWAYS_INLINE DST saturate(SRC x, R round)
{
    int ix = static_cast<int>(round(x));
    return saturate<DST>(ix);
}

// explicit suffix 'd' for double type
inline double  ceild(double x) { return ceil(x); }
inline double floord(double x) { return floor(x); }
inline double roundd(double x) { return round(x); }
inline double  rintd(double x) { return rint(x); }

} //namespace own
} //namespace gapi
} //namespace cv
#endif /* OPENCV_GAPI_OWN_SATURATE_HPP */
