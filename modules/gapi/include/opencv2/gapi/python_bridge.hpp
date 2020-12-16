// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation

#ifndef OPENCV_GAPI_PYTHON_BRIDGE_HPP
#define OPENCV_GAPI_PYTHON_BRIDGE_HPP

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <iostream>

namespace cv {
namespace gapi {

// NB: cv.gapi.CV_BOOL in python
// As cv::detail::OpaqueKind except: CV_UNKNOWN, CV_DRAW_PRIM, CV_UINT64
enum ArgType {
    CV_BOOL,
    CV_INT,
    CV_DOUBLE,
    CV_FLOAT,
    CV_STRING,
    CV_POINT,
    CV_POINT2F,
    CV_SIZE,
    CV_RECT,
    CV_SCALAR,
    CV_MAT,
};

} // namespace gapi

template <template <typename> class Wrapper, typename T>
struct WrapType { using type = Wrapper<T>; };

template <template <typename> class T, typename... Types>
using MakeVariantType = cv::util::variant<typename WrapType<T, Types>::type...>;

struct GAPI_EXPORTS_W_SIMPLE GOpaqueT
{
    using Storage = MakeVariantType<  cv::GOpaque
                                    , bool
                                    , int
                                    , double
                                    , float
                                    , std::string
                                    , cv::Point
                                    , cv::Point2f
                                    , cv::Size
                                    , cv::Rect>;
    template<typename T>
    GOpaqueT(cv::GOpaque<T> opaque) : arg(opaque) { };

    GAPI_WRAP GOpaqueT(gapi::ArgType type)
    {

#define HANDLE_CASE(T, K) case gapi::ArgType::CV_##T: \
                arg = cv::GOpaque<K>(); \
                break;

        switch (type)
        {
            HANDLE_CASE(BOOL,    bool);
            HANDLE_CASE(INT,     int);
            HANDLE_CASE(DOUBLE,  double);
            HANDLE_CASE(FLOAT,   float);
            HANDLE_CASE(STRING,  std::string);
            HANDLE_CASE(POINT,   cv::Point);
            HANDLE_CASE(POINT2F, cv::Point2f);
            HANDLE_CASE(SIZE,    cv::Size);
            HANDLE_CASE(RECT,    cv::Rect);
#undef HANDLE_CASE
            default:
                GAPI_Assert(false && "Unsupported type");
        }
    }

    cv::detail::GOpaqueU strip() {
#define HANDLE_CASE(T) case Storage::template index_of<cv::GOpaque<T>>(): \
        return cv::util::get<cv::GOpaque<T>>(arg).strip(); \

        switch (arg.index())
        {
            HANDLE_CASE(bool);
            HANDLE_CASE(int);
            HANDLE_CASE(double);
            HANDLE_CASE(float);
            HANDLE_CASE(std::string);
            HANDLE_CASE(cv::Point);
            HANDLE_CASE(cv::Point2f);
            HANDLE_CASE(cv::Size);
            HANDLE_CASE(cv::Rect);
#undef HANDLE_CASE
            default:
                GAPI_Assert(false && "Unsupported type");
        }
        GAPI_Assert(false);
    }

    Storage arg;
};

struct GAPI_EXPORTS_W_SIMPLE GArrayT
{
    using Storage = MakeVariantType<  cv::GArray
                                    , bool
                                    , int
                                    , double
                                    , float
                                    , std::string
                                    , cv::Point
                                    , cv::Point2f
                                    , cv::Size
                                    , cv::Rect
                                    , cv::Scalar
                                    , cv::Mat>;
    template<typename T>
    GArrayT(cv::GArray<T> opaque) : arg(opaque) { };

    GAPI_WRAP GArrayT(gapi::ArgType type)
    {

#define HANDLE_CASE(T, K) case gapi::ArgType::CV_##T: \
                arg = cv::GArray<K>(); \
                break;

        switch (type)
        {
            HANDLE_CASE(BOOL,    bool);
            HANDLE_CASE(INT,     int);
            HANDLE_CASE(DOUBLE,  double);
            HANDLE_CASE(FLOAT,   float);
            HANDLE_CASE(STRING,  std::string);
            HANDLE_CASE(POINT,   cv::Point);
            HANDLE_CASE(POINT2F, cv::Point2f);
            HANDLE_CASE(SIZE,    cv::Size);
            HANDLE_CASE(RECT,    cv::Rect);
            HANDLE_CASE(SCALAR,  cv::Scalar);
            HANDLE_CASE(MAT,     cv::Mat);
#undef HANDLE_CASE
            default:
                GAPI_Assert(false && "Unsupported type");
        }
    }

    cv::detail::GArrayU strip() {
#define HANDLE_CASE(T) case Storage::template index_of<cv::GArray<T>>(): \
        return cv::util::get<cv::GArray<T>>(arg).strip(); \

        switch (arg.index())
        {
            HANDLE_CASE(bool);
            HANDLE_CASE(int);
            HANDLE_CASE(double);
            HANDLE_CASE(float);
            HANDLE_CASE(std::string);
            HANDLE_CASE(cv::Point);
            HANDLE_CASE(cv::Point2f);
            HANDLE_CASE(cv::Size);
            HANDLE_CASE(cv::Rect);
            HANDLE_CASE(cv::Scalar);
            HANDLE_CASE(cv::Mat);
#undef HANDLE_CASE
            default:
                GAPI_Assert(false && "Unsupported type");
        }
        GAPI_Assert(false);
    }

    Storage arg;
};

} // namespace cv

#endif // OPENCV_GAPI_PYTHON_BRIDGE_HPP
