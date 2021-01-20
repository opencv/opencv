// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_PYTHON_BRIDGE_HPP
#define OPENCV_GAPI_PYTHON_BRIDGE_HPP

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gopaque.hpp>

#define ID(X)       X
#define ID_(X)		ID(X),
#define NOTHING(X)
#define SPACE
#define COMMA       ,

#define TYPE_LIST_G(LIST, G, A1, L1, A2, L2) \
    LIST(G, SPACE, A1, L1, A2, L2)

#define TYPE_LIST2_G(LIST, G, A1, A2) \
    LIST(G, COMMA, A1, A1, A2, A2)

#define SWITCH(VAR, LIST, HANDLE) \
switch (VAR) \
{ \
    TYPE_LIST2_G(LIST, HANDLE, ID, ID) \
    default: \
        GAPI_Assert(false && "Unsupported type"); \
}

#define GARRAY_TYPE_LIST_G(G, X, A1, L1, A2, L2) \
G(A1(bool)        X  A2(cv::gapi::ArgType::CV_BOOL)) \
G(A1(int)         X  A2(cv::gapi::ArgType::CV_INT)) \
G(A1(double)      X  A2(cv::gapi::ArgType::CV_DOUBLE)) \
G(A1(float)       X  A2(cv::gapi::ArgType::CV_FLOAT)) \
G(A1(std::string) X  A2(cv::gapi::ArgType::CV_STRING)) \
G(A1(cv::Point)   X  A2(cv::gapi::ArgType::CV_POINT)) \
G(A1(cv::Point2f) X  A2(cv::gapi::ArgType::CV_POINT2F)) \
G(A1(cv::Size)    X  A2(cv::gapi::ArgType::CV_SIZE)) \
G(A1(cv::Rect)    X  A2(cv::gapi::ArgType::CV_RECT)) \
G(A1(cv::Scalar)  X  A2(cv::gapi::ArgType::CV_SCALAR)) \
G(A1(cv::Mat)     X  A2(cv::gapi::ArgType::CV_MAT)) \
G(L1(cv::GMat)    X  L2(cv::gapi::ArgType::CV_GMAT))

#define GOPAQUE_TYPE_LIST_G(G, X, A1, L1, A2, L2) \
G(A1(bool)        X  A2(cv::gapi::ArgType::CV_BOOL)) \
G(A1(int)         X  A2(cv::gapi::ArgType::CV_INT)) \
G(A1(double)      X  A2(cv::gapi::ArgType::CV_DOUBLE)) \
G(A1(float)       X  A2(cv::gapi::ArgType::CV_FLOAT)) \
G(A1(std::string) X  A2(cv::gapi::ArgType::CV_STRING)) \
G(A1(cv::Point)   X  A2(cv::gapi::ArgType::CV_POINT)) \
G(A1(cv::Point2f) X  A2(cv::gapi::ArgType::CV_POINT2F)) \
G(A1(cv::Size)    X  A2(cv::gapi::ArgType::CV_SIZE)) \
G(L1(cv::Rect)    X  L2(cv::gapi::ArgType::CV_RECT)) \

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

GARRAY_TYPE_LIST_G(ID(DEFINE_TYPE_TRAITS), COMMA, ID, ID, ID, ID)

} // namespace detail

class GAPI_EXPORTS_W_SIMPLE GOpaqueT
{
public:
    using Storage = cv::detail::MakeVariantType<cv::GOpaque,
          TYPE_LIST_G(GOPAQUE_TYPE_LIST_G, ID, ID_, ID, NOTHING, NOTHING)>;

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
    using Storage = cv::detail::MakeVariantType<cv::GArray,
          TYPE_LIST_G(GARRAY_TYPE_LIST_G, ID, ID_, ID, NOTHING, NOTHING)>;

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
