// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_PYTHON_BRIDGE_HPP
#define OPENCV_GAPI_PYTHON_BRIDGE_HPP

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gopaque.hpp>

#define ID(T, E)  T
#define ID_(T, E) ID(T, E),

#define WRAP_ARGS(T, E, G) \
    G(T, E)

#define SWITCH(type, LIST_G, HC) \
    switch(type) { \
        LIST_G(HC, HC)  \
        default: \
            GAPI_Assert(false && "Unsupported type"); \
    }

#define GARRAY_TYPE_LIST_G(G, G2) \
WRAP_ARGS(bool        ,  cv::gapi::ArgType::CV_BOOL,    G) \
WRAP_ARGS(int         ,  cv::gapi::ArgType::CV_INT,     G) \
WRAP_ARGS(double      ,  cv::gapi::ArgType::CV_DOUBLE,  G) \
WRAP_ARGS(float       ,  cv::gapi::ArgType::CV_FLOAT,   G) \
WRAP_ARGS(std::string ,  cv::gapi::ArgType::CV_STRING,  G) \
WRAP_ARGS(cv::Point   ,  cv::gapi::ArgType::CV_POINT,   G) \
WRAP_ARGS(cv::Point2f ,  cv::gapi::ArgType::CV_POINT2F, G) \
WRAP_ARGS(cv::Size    ,  cv::gapi::ArgType::CV_SIZE,    G) \
WRAP_ARGS(cv::Rect    ,  cv::gapi::ArgType::CV_RECT,    G) \
WRAP_ARGS(cv::Scalar  ,  cv::gapi::ArgType::CV_SCALAR,  G) \
WRAP_ARGS(cv::Mat     ,  cv::gapi::ArgType::CV_MAT,     G) \
WRAP_ARGS(cv::GMat    ,  cv::gapi::ArgType::CV_GMAT,    G2)

#define GOPAQUE_TYPE_LIST_G(G, G2) \
WRAP_ARGS(bool        ,  cv::gapi::ArgType::CV_BOOL,    G)  \
WRAP_ARGS(int         ,  cv::gapi::ArgType::CV_INT,     G)  \
WRAP_ARGS(double      ,  cv::gapi::ArgType::CV_DOUBLE,  G)  \
WRAP_ARGS(float       ,  cv::gapi::ArgType::CV_FLOAT,   G)  \
WRAP_ARGS(std::string ,  cv::gapi::ArgType::CV_STRING,  G)  \
WRAP_ARGS(cv::Point   ,  cv::gapi::ArgType::CV_POINT,   G)  \
WRAP_ARGS(cv::Point2f ,  cv::gapi::ArgType::CV_POINT2F, G)  \
WRAP_ARGS(cv::Size    ,  cv::gapi::ArgType::CV_SIZE,    G)  \
WRAP_ARGS(cv::Rect    ,  cv::gapi::ArgType::CV_RECT,    G2) \

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

#define DEFINE_TYPE_TRAITS(T, E) \
template <> \
struct ArgTypeTraits<T> { \
    static constexpr const cv::gapi::ArgType type = E; \
}; \

GARRAY_TYPE_LIST_G(DEFINE_TYPE_TRAITS, DEFINE_TYPE_TRAITS)

} // namespace detail

class GAPI_EXPORTS_W_SIMPLE GOpaqueT
{
public:
    using Storage = cv::detail::MakeVariantType<cv::GOpaque, GOPAQUE_TYPE_LIST_G(ID_, ID)>;

    template<typename T>
    GOpaqueT(cv::GOpaque<T> arg) : m_type(cv::detail::ArgTypeTraits<T>::type), m_arg(arg) { };

    GAPI_WRAP GOpaqueT(gapi::ArgType type) : m_type(type)
    {

#define HC(T, K) case K: \
        m_arg = cv::GOpaque<T>(); \
        break;

        SWITCH(type, GOPAQUE_TYPE_LIST_G, HC)
#undef HC
    }

    cv::detail::GOpaqueU strip() {
#define HC(T, K) case Storage:: index_of<cv::GOpaque<T>>(): \
        return cv::util::get<cv::GOpaque<T>>(m_arg).strip(); \

        SWITCH(m_arg.index(), GOPAQUE_TYPE_LIST_G, HC)
#undef HC

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
    using Storage = cv::detail::MakeVariantType<cv::GArray, GARRAY_TYPE_LIST_G(ID_, ID)>;

    template<typename T>
    GArrayT(cv::GArray<T> arg) : m_type(cv::detail::ArgTypeTraits<T>::type), m_arg(arg) { };

    GAPI_WRAP GArrayT(gapi::ArgType type) : m_type(type)
    {

#define HC(T, K) case K: \
        m_arg = cv::GArray<T>(); \
        break;

        SWITCH(type, GARRAY_TYPE_LIST_G, HC)
#undef HC
    }

    cv::detail::GArrayU strip() {
#define HC(T, K) case Storage:: index_of<cv::GArray<T>>(): \
        return cv::util::get<cv::GArray<T>>(m_arg).strip(); \

        SWITCH(m_arg.index(), GARRAY_TYPE_LIST_G, HC)
#undef HC

        GAPI_Assert(false);
    }

    GAPI_WRAP gapi::ArgType type() { return m_type; }

private:
    gapi::ArgType m_type;
    Storage m_arg;
};

} // namespace cv

#endif // OPENCV_GAPI_PYTHON_BRIDGE_HPP
