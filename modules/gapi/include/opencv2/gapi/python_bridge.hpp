// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_PYTHON_BRIDGE_HPP
#define OPENCV_GAPI_PYTHON_BRIDGE_HPP

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <iostream>

namespace cv {
namespace gapi {

// NB: cv.gapi.CV_BOOL in python
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
    CV_GMAT,
};

} // namespace gapi

namespace detail {

template <template <typename> class Wrapper, typename T>
struct WrapType { using type = Wrapper<T>; };

template <template <typename> class T, typename... Types>
using MakeVariantType = cv::util::variant<typename WrapType<T, Types>::type...>;

template<typename T> struct ArgTypeTraits;

template<> struct ArgTypeTraits<bool> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_BOOL;
};
template<> struct ArgTypeTraits<int> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_INT;
};
template<> struct ArgTypeTraits<double> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_DOUBLE;
};
template<> struct ArgTypeTraits<float> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_FLOAT;
};
template<> struct ArgTypeTraits<std::string> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_STRING;
};
template<> struct ArgTypeTraits<cv::Point> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_POINT;
};
template<> struct ArgTypeTraits<cv::Point2f> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_POINT2F;
};
template<> struct ArgTypeTraits<cv::Size> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_SIZE;
};
template<> struct ArgTypeTraits<cv::Rect> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_RECT;
};
template<> struct ArgTypeTraits<cv::Scalar> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_SCALAR;
};
template<> struct ArgTypeTraits<cv::Mat> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_MAT;
};
template<> struct ArgTypeTraits<cv::GMat> {
    static constexpr const cv::gapi::ArgType type = cv::gapi::ArgType::CV_GMAT;
};

} // namespace detail

class GAPI_EXPORTS_W_SIMPLE GOpaqueT
{
public:
    using Storage = detail::MakeVariantType<  cv::GOpaque
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
    GOpaqueT(cv::GOpaque<T> arg) : m_arg(arg) { };

    GAPI_WRAP GOpaqueT(gapi::ArgType type) : m_type(type)
    {

#define HANDLE_CASE(T, K) case gapi::ArgType::CV_##T: \
                m_arg = cv::GOpaque<K>(); \
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
        return cv::util::get<cv::GOpaque<T>>(m_arg).strip(); \

        switch (m_arg.index())
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

GAPI_WRAP gapi::ArgType type() { return m_type; }

private:
    gapi::ArgType m_type;
    Storage m_arg;
};

class GAPI_EXPORTS_W_SIMPLE GArrayT
{
public:
    using Storage = detail::MakeVariantType<  cv::GArray
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
                                            , cv::Mat
                                            , cv::GMat>;
    template<typename T>
    GArrayT(cv::GArray<T> arg) : m_arg(arg) { };

    GAPI_WRAP GArrayT(gapi::ArgType type) : m_type(type)
    {

#define HANDLE_CASE(T, K) case gapi::ArgType::CV_##T: \
                m_arg = cv::GArray<K>(); \
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
            HANDLE_CASE(GMAT,    cv::GMat);
#undef HANDLE_CASE
            default:
                GAPI_Assert(false && "Unsupported type");
        }
    }

    cv::detail::GArrayU strip() {
#define HANDLE_CASE(T) case Storage::template index_of<cv::GArray<T>>(): \
        return cv::util::get<cv::GArray<T>>(m_arg).strip(); \

        switch (m_arg.index())
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
            HANDLE_CASE(cv::GMat);
#undef HANDLE_CASE
            default:
                GAPI_Assert(false && "Unsupported type");
        }
        GAPI_Assert(false);
    }

    GAPI_WRAP gapi::ArgType type() { return m_type; }

private:
    gapi::ArgType m_type;
    Storage m_arg;
};

} // namespace cv

#endif // OPENCV_GAPI_PYTHON_BRIDGE_HPP
