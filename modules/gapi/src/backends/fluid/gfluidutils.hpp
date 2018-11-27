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
#include <opencv2/gapi/own/saturate.hpp>

namespace cv {
namespace gapi {
namespace fluid {

using cv::gapi::own::saturate;
using cv::gapi::own::ceild;
using cv::gapi::own::floord;
using cv::gapi::own::roundd;
using cv::gapi::own::rintd;

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
