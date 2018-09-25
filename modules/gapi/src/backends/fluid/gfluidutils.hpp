// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef GFLUIDUTILS_HPP
#define GFLUIDUTILS_HPP

#include <limits>
#include <type_traits>
#include <opencv2/gapi/util/compiler_hints.hpp> //UNUSED

namespace cv {
namespace gapi {
namespace fluid {

//-----------------------------
//
// Numeric cast with saturation
//
//-----------------------------

template<typename DST, typename SRC>
static inline DST saturate(SRC x)
{
    // only integral types please!
    GAPI_DbgAssert(std::is_integral<DST>::value &&
                   std::is_integral<SRC>::value);

    if (std::is_same<DST, SRC>::value)
        return static_cast<DST>(x);

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

// Note, that OpenCV rounds differently:
// - like std::round() for add, subtract
// - like std::rint() for multiply, divide
template<typename DST, typename SRC, typename R>
static inline DST saturate(SRC x, R round)
{
    if (std::is_floating_point<DST>::value)
    {
        return static_cast<DST>(x);
    }
    else if (std::is_integral<SRC>::value)
    {
        GAPI_DbgAssert(std::is_integral<DST>::value &&
                       std::is_integral<SRC>::value);
        return saturate<DST>(x);
    }
    else
    {
        GAPI_DbgAssert(std::is_integral<DST>::value &&
                 std::is_floating_point<SRC>::value);
#ifdef _WIN32
// Suppress warning about convering x to floating-point
// Note that x is already floating-point at this point
#pragma warning(disable: 4244)
#endif
        int ix = static_cast<int>(round(x));
#ifdef _WIN32
#pragma warning(default: 4244)
#endif
        return saturate<DST>(ix);
    }
}

// explicit suffix 'd' for double type
static inline double  ceild(double x) { return std::ceil(x); }
static inline double floord(double x) { return std::floor(x); }
static inline double roundd(double x) { return std::round(x); }
static inline double  rintd(double x) { return std::rint(x); }

//--------------------------------
//
// Macros for mappig of data types
//
//--------------------------------

#define UNARY_(DST, SRC, OP, ...)                         \
    if (cv::DataType<DST>::depth == dst.meta().depth &&   \
        cv::DataType<SRC>::depth == src.meta().depth)     \
    {                                                     \
        GAPI_DbgAssert(dst.length() == src.length());       \
        GAPI_DbgAssert(dst.meta().chan == src.meta().chan); \
                                                          \
        OP<DST, SRC>(__VA_ARGS__);                        \
        return;                                           \
    }

// especial unary operation: dst is always 8UC1 image
#define INRANGE_(DST, SRC, OP, ...)                       \
    if (cv::DataType<DST>::depth == dst.meta().depth &&   \
        cv::DataType<SRC>::depth == src.meta().depth)     \
    {                                                     \
        GAPI_DbgAssert(dst.length() == src.length());       \
        GAPI_DbgAssert(dst.meta().chan == 1);               \
                                                          \
        OP<DST, SRC>(__VA_ARGS__);                        \
        return;                                           \
    }

#define BINARY_(DST, SRC1, SRC2, OP, ...)                  \
    if (cv::DataType<DST>::depth == dst.meta().depth &&    \
        cv::DataType<SRC1>::depth == src1.meta().depth &&  \
        cv::DataType<SRC2>::depth == src2.meta().depth)    \
    {                                                      \
        GAPI_DbgAssert(dst.length() == src1.length());       \
        GAPI_DbgAssert(dst.length() == src2.length());       \
                                                           \
        GAPI_DbgAssert(dst.meta().chan == src1.meta().chan); \
        GAPI_DbgAssert(dst.meta().chan == src2.meta().chan); \
                                                           \
        OP<DST, SRC1, SRC2>(__VA_ARGS__);                  \
        return;                                            \
    }

// especial ternary operation: src3 has only one channel
#define SELECT_(DST, SRC1, SRC2, SRC3, OP, ...)            \
    if (cv::DataType<DST>::depth == dst.meta().depth &&    \
        cv::DataType<SRC1>::depth == src1.meta().depth &&  \
        cv::DataType<SRC2>::depth == src2.meta().depth &&  \
        cv::DataType<SRC3>::depth == src3.meta().depth)    \
    {                                                      \
        GAPI_DbgAssert(dst.length() == src1.length());       \
        GAPI_DbgAssert(dst.length() == src2.length());       \
        GAPI_DbgAssert(dst.length() == src3.length());       \
                                                           \
        GAPI_DbgAssert(dst.meta().chan == src1.meta().chan); \
        GAPI_DbgAssert(dst.meta().chan == src2.meta().chan); \
        GAPI_DbgAssert(              1 == src3.meta().chan); \
                                                           \
        OP<DST, SRC1, SRC2, SRC3>(__VA_ARGS__);            \
        return;                                            \
    }

} // namespace fluid
} // namespace gapi
} // namespace cv

#endif // GFLUIDUTILS_HPP
