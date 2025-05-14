// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_GARG_HPP
#define OPENCV_GAPI_GARG_HPP

#include <vector>
#include <unordered_map>
#include <type_traits>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/mat.hpp>
#include <opencv2/gapi/media.hpp>

#include <opencv2/gapi/util/util.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/util/variant.hpp>

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gframe.hpp>
#include <opencv2/gapi/gtype_traits.hpp>
#include <opencv2/gapi/gmetaarg.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/rmat.hpp>

namespace cv {

class GArg;

namespace detail {
    template<typename T>
    using is_garg = std::is_same<GArg, typename std::decay<T>::type>;
}

// Parameter holder class for a node
// Depending on platform capabilities, can either support arbitrary types
// (as `boost::any`) or a limited number of types (as `boot::variant`).
// FIXME: put into "details" as a user shouldn't use it in his code
class GAPI_EXPORTS GArg
{
public:
    GArg() {}

    template<typename T, typename std::enable_if<!detail::is_garg<T>::value, int>::type = 0>
    explicit GArg(const T &t)
        : kind(detail::GTypeTraits<T>::kind)
        , opaque_kind(detail::GOpaqueTraits<T>::kind)
        , value(detail::wrap_gapi_helper<T>::wrap(t))
    {
    }

    template<typename T, typename std::enable_if<!detail::is_garg<T>::value, int>::type = 0>
    explicit GArg(T &&t)
        : kind(detail::GTypeTraits<typename std::decay<T>::type>::kind)
        , opaque_kind(detail::GOpaqueTraits<typename std::decay<T>::type>::kind)
        , value(detail::wrap_gapi_helper<T>::wrap(t))
    {
    }

    template<typename T> inline T& get()
    {
        return util::any_cast<typename std::remove_reference<T>::type>(value);
    }

    template<typename T> inline const T& get() const
    {
        return util::any_cast<typename std::remove_reference<T>::type>(value);
    }

    template<typename T> inline T& unsafe_get()
    {
        return util::unsafe_any_cast<typename std::remove_reference<T>::type>(value);
    }

    template<typename T> inline const T& unsafe_get() const
    {
        return util::unsafe_any_cast<typename std::remove_reference<T>::type>(value);
    }

    detail::ArgKind kind = detail::ArgKind::OPAQUE_VAL;
    detail::OpaqueKind opaque_kind = detail::OpaqueKind::CV_UNKNOWN;

protected:
    util::any value;
};

using GArgs = std::vector<GArg>;

// FIXME: Express as M<GProtoArg...>::type
// FIXME: Move to a separate file!
using GRunArgBase  = util::variant<
#if !defined(GAPI_STANDALONE)
    cv::UMat,
#endif // !defined(GAPI_STANDALONE)
    cv::RMat,
    cv::gapi::wip::IStreamSource::Ptr,
    cv::Mat,
    cv::Scalar,
    cv::detail::VectorRef,
    cv::detail::OpaqueRef,
    cv::MediaFrame
    >;

namespace detail {
template<typename,typename>
struct in_variant;

template<typename T, typename... Types>
struct in_variant<T, util::variant<Types...> >
    : std::integral_constant<bool, cv::detail::contains<T, Types...>::value > {
};
} // namespace detail

struct GAPI_EXPORTS GRunArg: public GRunArgBase
{
    // Metadata information here
    using Meta = std::unordered_map<std::string, util::any>;
    Meta meta;

    // Mimic the old GRunArg semantics here, old of the times when
    // GRunArg was an alias to variant<>
    GRunArg();
    GRunArg(const cv::GRunArg &arg);
    GRunArg(cv::GRunArg &&arg);

    GRunArg& operator= (const GRunArg &arg);
    GRunArg& operator= (GRunArg &&arg);

    template <typename T>
    GRunArg(const T &t,
            const Meta &m = Meta{},
            typename std::enable_if< detail::in_variant<T, GRunArgBase>::value, int>::type = 0)
        : GRunArgBase(t)
        , meta(m)
    {
    }
    template <typename T>
    GRunArg(T &&t,
            const Meta &m = Meta{},
            typename std::enable_if< detail::in_variant<T, GRunArgBase>::value, int>::type = 0)
        : GRunArgBase(std::move(t))
        , meta(m)
    {
    }
    template <typename T> auto operator= (const T &t)
        -> typename std::enable_if< detail::in_variant<T, GRunArgBase>::value, cv::GRunArg>::type&
    {
        GRunArgBase::operator=(t);
        return *this;
    }
    template <typename T> auto operator= (T&& t)
        -> typename std::enable_if< detail::in_variant<T, GRunArgBase>::value, cv::GRunArg>::type&
    {
        GRunArgBase::operator=(std::move(t));
        return *this;
    }
};
using GRunArgs = std::vector<GRunArg>;

// TODO: Think about the addition operator
/**
 * @brief This operator allows to complement the input vector at runtime.
 *
 * It's an ordinary overload of addition assignment operator.
 *
 * Example of usage:
 * @snippet samples/cpp/tutorial_code/gapi/doc_snippets/dynamic_graph_snippets.cpp GRunArgs usage
 *
 */
inline GRunArgs& operator += (GRunArgs &lhs, const GRunArgs &rhs)
{
    lhs.reserve(lhs.size() + rhs.size());
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

namespace gapi
{
namespace wip
{
/**
 * @brief This aggregate type represents all types which G-API can
 * handle (via variant).
 *
 * It only exists to overcome C++ language limitations (where a
 * `using`-defined class can't be forward-declared).
 */
struct GAPI_EXPORTS Data: public GRunArg
{
    using GRunArg::GRunArg;
    template <typename T>
    Data& operator= (const T& t) { GRunArg::operator=(t); return *this; }
    template <typename T>
    Data& operator= (T&& t) { GRunArg::operator=(std::move(t)); return *this; }
};
} // namespace wip
} // namespace gapi

using GRunArgP = util::variant<
#if !defined(GAPI_STANDALONE)
    cv::UMat*,
#endif // !defined(GAPI_STANDALONE)
    cv::Mat*,
    cv::RMat*,
    cv::Scalar*,
    cv::MediaFrame*,
    cv::detail::VectorRef,
    cv::detail::OpaqueRef
    >;
using GRunArgsP = std::vector<GRunArgP>;

// TODO: Think about the addition operator
/**
 * @brief This operator allows to complement the output vector at runtime.
 *
 * It's an ordinary overload of addition assignment operator.
 *
 * Example of usage:
 * @snippet samples/cpp/tutorial_code/gapi/doc_snippets/dynamic_graph_snippets.cpp GRunArgsP usage
 *
 */
inline GRunArgsP& operator += (GRunArgsP &lhs, const GRunArgsP &rhs)
{
    lhs.reserve(lhs.size() + rhs.size());
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

namespace gapi
{
/**
 * \addtogroup gapi_serialization
 * @{
 *
 * @brief G-API functions and classes for serialization and deserialization.
 */

/** @brief Wraps deserialized output GRunArgs to GRunArgsP which can be used by GCompiled.
 *
 * Since it's impossible to get modifiable output arguments from deserialization
 * it needs to be wrapped by this function.
 *
 * Example of usage:
 * @snippet samples/cpp/tutorial_code/gapi/doc_snippets/api_ref_snippets.cpp bind after deserialization
 *
 * @param out_args deserialized GRunArgs.
 * @return the same GRunArgs wrapped in GRunArgsP.
 * @see deserialize
 */
GAPI_EXPORTS cv::GRunArgsP bind(cv::GRunArgs &out_args);

/** @brief Wraps output GRunArgsP available during graph execution to GRunArgs which can be serialized.
 *
 * GRunArgsP is pointer-to-value, so to be serialized they need to be binded to real values
 * which this function does.
 *
 * Example of usage:
 * @snippet samples/cpp/tutorial_code/gapi/doc_snippets/api_ref_snippets.cpp bind before serialization
 *
 * @param out output GRunArgsP available during graph execution.
 * @return the same GRunArgsP wrapped in serializable GRunArgs.
 * @see serialize
 */
GAPI_EXPORTS cv::GRunArg   bind(cv::GRunArgP &out);     // FIXME: think more about it
/** @} */
}

template<typename... Ts> inline GRunArgs gin(const Ts&... args)
{
    return GRunArgs{ GRunArg(detail::wrap_host_helper<Ts>::wrap_in(args))... };
}

template<typename... Ts> inline GRunArgsP gout(Ts&... args)
{
    return GRunArgsP{ GRunArgP(detail::wrap_host_helper<Ts>::wrap_out(args))... };
}

struct GTypeInfo;
using GTypesInfo = std::vector<GTypeInfo>;

// FIXME: Needed for python bridge, must be moved to more appropriate header
namespace detail {
struct ExtractArgsCallback
{
    cv::GRunArgs operator()(const cv::GTypesInfo& info) const { return c(info); }
    using CallBackT = std::function<cv::GRunArgs(const cv::GTypesInfo& info)>;
    CallBackT c;
};

struct ExtractMetaCallback
{
    cv::GMetaArgs operator()(const cv::GTypesInfo& info) const { return c(info); }
    using CallBackT = std::function<cv::GMetaArgs(const cv::GTypesInfo& info)>;
    CallBackT c;
};

void constructGraphOutputs(const cv::GTypesInfo &out_info,
                           cv::GRunArgs         &args,
                           cv::GRunArgsP        &outs);
} // namespace detail

} // namespace cv

#endif // OPENCV_GAPI_GARG_HPP
