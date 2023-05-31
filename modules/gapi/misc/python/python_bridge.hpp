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
#include <opencv2/gapi/render/render_types.hpp> // Prim

#define ID(T, E)  T
#define ID_(T, E) ID(T, E),

#define WRAP_ARGS(T, E, G) \
    G(T, E)

#define SWITCH(type, LIST_G, HC) \
    switch(type) { \
        LIST_G(HC, HC)  \
        default: \
            GAPI_Error("Unsupported type"); \
    }

using cv::gapi::wip::draw::Prim;

#define GARRAY_TYPE_LIST_G(G, G2) \
WRAP_ARGS(bool        , cv::gapi::ArgType::CV_BOOL,      G)  \
WRAP_ARGS(int         , cv::gapi::ArgType::CV_INT,       G)  \
WRAP_ARGS(int64_t     , cv::gapi::ArgType::CV_INT64,     G)  \
WRAP_ARGS(double      , cv::gapi::ArgType::CV_DOUBLE,    G)  \
WRAP_ARGS(float       , cv::gapi::ArgType::CV_FLOAT,     G)  \
WRAP_ARGS(std::string , cv::gapi::ArgType::CV_STRING,    G)  \
WRAP_ARGS(cv::Point   , cv::gapi::ArgType::CV_POINT,     G)  \
WRAP_ARGS(cv::Point2f , cv::gapi::ArgType::CV_POINT2F,   G)  \
WRAP_ARGS(cv::Point3f , cv::gapi::ArgType::CV_POINT3F,   G)  \
WRAP_ARGS(cv::Size    , cv::gapi::ArgType::CV_SIZE,      G)  \
WRAP_ARGS(cv::Rect    , cv::gapi::ArgType::CV_RECT,      G)  \
WRAP_ARGS(cv::Scalar  , cv::gapi::ArgType::CV_SCALAR,    G)  \
WRAP_ARGS(cv::Mat     , cv::gapi::ArgType::CV_MAT,       G)  \
WRAP_ARGS(Prim        , cv::gapi::ArgType::CV_DRAW_PRIM, G)  \
WRAP_ARGS(cv::GArg    , cv::gapi::ArgType::CV_ANY,       G)  \
WRAP_ARGS(cv::GMat    , cv::gapi::ArgType::CV_GMAT,      G2) \

#define GOPAQUE_TYPE_LIST_G(G, G2) \
WRAP_ARGS(bool        , cv::gapi::ArgType::CV_BOOL,    G)  \
WRAP_ARGS(int         , cv::gapi::ArgType::CV_INT,     G)  \
WRAP_ARGS(int64_t     , cv::gapi::ArgType::CV_INT64,   G)  \
WRAP_ARGS(double      , cv::gapi::ArgType::CV_DOUBLE,  G)  \
WRAP_ARGS(float       , cv::gapi::ArgType::CV_FLOAT,   G)  \
WRAP_ARGS(std::string , cv::gapi::ArgType::CV_STRING,  G)  \
WRAP_ARGS(cv::Point   , cv::gapi::ArgType::CV_POINT,   G)  \
WRAP_ARGS(cv::Point2f , cv::gapi::ArgType::CV_POINT2F, G)  \
WRAP_ARGS(cv::Point3f , cv::gapi::ArgType::CV_POINT3F, G)  \
WRAP_ARGS(cv::Size    , cv::gapi::ArgType::CV_SIZE,    G)  \
WRAP_ARGS(cv::GArg    , cv::gapi::ArgType::CV_ANY,     G)  \
WRAP_ARGS(cv::Rect    , cv::gapi::ArgType::CV_RECT,    G2) \

namespace cv {
namespace gapi {

// NB: cv.gapi.CV_BOOL in python
enum ArgType {
    CV_BOOL,
    CV_INT,
    CV_INT64,
    CV_DOUBLE,
    CV_FLOAT,
    CV_STRING,
    CV_POINT,
    CV_POINT2F,
    CV_POINT3F,
    CV_SIZE,
    CV_RECT,
    CV_SCALAR,
    CV_MAT,
    CV_GMAT,
    CV_DRAW_PRIM,
    CV_ANY,
};

GAPI_EXPORTS_W inline cv::GInferOutputs infer(const String& name, const cv::GInferInputs& inputs)
{
    return infer<Generic>(name, inputs);
}

GAPI_EXPORTS_W inline GInferOutputs infer(const std::string& name,
                                          const cv::GOpaque<cv::Rect>& roi,
                                          const GInferInputs& inputs)
{
    return infer<Generic>(name, roi, inputs);
}

GAPI_EXPORTS_W inline GInferListOutputs infer(const std::string& name,
                                              const cv::GArray<cv::Rect>& rois,
                                              const GInferInputs& inputs)
{
    return infer<Generic>(name, rois, inputs);
}

GAPI_EXPORTS_W inline GInferListOutputs infer2(const std::string& name,
                                               const cv::GMat in,
                                               const GInferListInputs& inputs)
{
    return infer2<Generic>(name, in, inputs);
}

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
    GOpaqueT() = default;
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

            GAPI_Error("InternalError");
    }

    GAPI_WRAP gapi::ArgType type() { return m_type; }
    const Storage& arg() const     { return m_arg;  }

private:
    gapi::ArgType m_type;
    Storage m_arg;
};

class GAPI_EXPORTS_W_SIMPLE GArrayT
{
public:
    GArrayT() = default;
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

        GAPI_Error("InternalError");
    }

    GAPI_WRAP gapi::ArgType type() { return m_type; }
    const Storage& arg() const     { return m_arg;  }

private:
    gapi::ArgType m_type;
    Storage m_arg;
};

namespace gapi {
namespace wip {

class GAPI_EXPORTS_W_SIMPLE GOutputs
{
public:
    GOutputs() = default;
    GOutputs(const std::string& id, cv::GKernel::M outMeta, cv::GArgs &&ins);

    GAPI_WRAP cv::GMat     getGMat();
    GAPI_WRAP cv::GScalar  getGScalar();
    GAPI_WRAP cv::GArrayT  getGArray(cv::gapi::ArgType type);
    GAPI_WRAP cv::GOpaqueT getGOpaque(cv::gapi::ArgType type);

private:
    class Priv;
    std::shared_ptr<Priv> m_priv;
};

GOutputs op(const std::string& id, cv::GKernel::M outMeta, cv::GArgs&& args);

template <typename... T>
GOutputs op(const std::string& id, cv::GKernel::M outMeta, T&&... args)
{
    return op(id, outMeta, cv::GArgs{cv::GArg(std::forward<T>(args))... });
}

} // namespace wip
} // namespace gapi
} // namespace cv

cv::gapi::wip::GOutputs cv::gapi::wip::op(const std::string& id,
                                          cv::GKernel::M outMeta,
                                          cv::GArgs&& args)
{
    cv::gapi::wip::GOutputs outputs{id, outMeta, std::move(args)};
    return outputs;
}

class cv::gapi::wip::GOutputs::Priv
{
public:
    Priv(const std::string& id, cv::GKernel::M outMeta, cv::GArgs &&ins);

    cv::GMat     getGMat();
    cv::GScalar  getGScalar();
    cv::GArrayT  getGArray(cv::gapi::ArgType);
    cv::GOpaqueT getGOpaque(cv::gapi::ArgType);

private:
    int output = 0;
    std::unique_ptr<cv::GCall> m_call;
};

cv::gapi::wip::GOutputs::Priv::Priv(const std::string& id, cv::GKernel::M outMeta, cv::GArgs &&args)
{
    cv::GKinds kinds;
    kinds.reserve(args.size());
    std::transform(args.begin(), args.end(), std::back_inserter(kinds),
            [](const cv::GArg& arg) { return arg.opaque_kind; });

    m_call.reset(new cv::GCall{cv::GKernel{id, {}, outMeta, {}, std::move(kinds), {}}});
    m_call->setArgs(std::move(args));
}

cv::GMat cv::gapi::wip::GOutputs::Priv::getGMat()
{
    m_call->kernel().outShapes.push_back(cv::GShape::GMAT);
    // ...so _empty_ constructor is passed here.
    m_call->kernel().outCtors.emplace_back(cv::util::monostate{});
    return m_call->yield(output++);
}

cv::GScalar cv::gapi::wip::GOutputs::Priv::getGScalar()
{
    m_call->kernel().outShapes.push_back(cv::GShape::GSCALAR);
    // ...so _empty_ constructor is passed here.
    m_call->kernel().outCtors.emplace_back(cv::util::monostate{});
    return m_call->yieldScalar(output++);
}

cv::GArrayT cv::gapi::wip::GOutputs::Priv::getGArray(cv::gapi::ArgType type)
{
    m_call->kernel().outShapes.push_back(cv::GShape::GARRAY);
#define HC(T, K)                                                                                \
    case K:                                                                                     \
        m_call->kernel().outCtors.emplace_back(cv::detail::GObtainCtor<cv::GArray<T>>::get());  \
        return cv::GArrayT(m_call->yieldArray<T>(output++));                                    \

    SWITCH(type, GARRAY_TYPE_LIST_G, HC)
#undef HC
}

cv::GOpaqueT cv::gapi::wip::GOutputs::Priv::getGOpaque(cv::gapi::ArgType type)
{
    m_call->kernel().outShapes.push_back(cv::GShape::GOPAQUE);
#define HC(T, K)                                                                                \
    case K:                                                                                     \
        m_call->kernel().outCtors.emplace_back(cv::detail::GObtainCtor<cv::GOpaque<T>>::get()); \
        return cv::GOpaqueT(m_call->yieldOpaque<T>(output++));                                  \

    SWITCH(type, GOPAQUE_TYPE_LIST_G, HC)
#undef HC
}

cv::gapi::wip::GOutputs::GOutputs(const std::string& id,
                                  cv::GKernel::M outMeta,
                                  cv::GArgs &&ins) :
    m_priv(new cv::gapi::wip::GOutputs::Priv(id, outMeta, std::move(ins)))
{
}

cv::GMat cv::gapi::wip::GOutputs::getGMat()
{
    return m_priv->getGMat();
}

cv::GScalar cv::gapi::wip::GOutputs::getGScalar()
{
    return m_priv->getGScalar();
}

cv::GArrayT cv::gapi::wip::GOutputs::getGArray(cv::gapi::ArgType type)
{
    return m_priv->getGArray(type);
}

cv::GOpaqueT cv::gapi::wip::GOutputs::getGOpaque(cv::gapi::ArgType type)
{
    return m_priv->getGOpaque(type);
}

#endif // OPENCV_GAPI_PYTHON_BRIDGE_HPP
