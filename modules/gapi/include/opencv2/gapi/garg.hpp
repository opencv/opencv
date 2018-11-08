// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GARG_HPP
#define OPENCV_GAPI_GARG_HPP

#include <vector>
#include <type_traits>

#include <opencv2/gapi/opencv_includes.hpp>
#include "opencv2/gapi/own/mat.hpp"

#include "opencv2/gapi/util/any.hpp"
#include "opencv2/gapi/util/variant.hpp"

#include "opencv2/gapi/gmat.hpp"
#include "opencv2/gapi/gscalar.hpp"
#include "opencv2/gapi/garray.hpp"
#include "opencv2/gapi/gtype_traits.hpp"
#include "opencv2/gapi/gmetaarg.hpp"
#include "opencv2/gapi/own/scalar.hpp"

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
        , value(detail::wrap_gapi_helper<T>::wrap(t))
    {
    }

    template<typename T, typename std::enable_if<!detail::is_garg<T>::value, int>::type = 0>
    explicit GArg(T &&t)
        : kind(detail::GTypeTraits<typename std::decay<T>::type>::kind)
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

    detail::ArgKind kind = detail::ArgKind::OPAQUE;

protected:
    util::any value;
};

using GArgs = std::vector<GArg>;

// FIXME: Express as M<GProtoArg...>::type
// FIXME: Move to a separate file!
using GRunArg  = util::variant<
#if !defined(GAPI_STANDALONE)
    cv::Mat,
    cv::Scalar,
    cv::UMat,
#endif // !defined(GAPI_STANDALONE)
    cv::gapi::own::Mat,
    cv::gapi::own::Scalar,
    cv::detail::VectorRef
    >;
using GRunArgs = std::vector<GRunArg>;

using GRunArgP = util::variant<
#if !defined(GAPI_STANDALONE)
    cv::Mat*,
    cv::Scalar*,
    cv::UMat*,
#endif // !defined(GAPI_STANDALONE)
    cv::gapi::own::Mat*,
    cv::gapi::own::Scalar*,
    cv::detail::VectorRef
    >;
using GRunArgsP = std::vector<GRunArgP>;


template<typename... Ts> inline GRunArgs gin(const Ts&... args)
{
    return GRunArgs{ GRunArg(detail::wrap_host_helper<Ts>::wrap_in(args))... };
}

template<typename... Ts> inline GRunArgsP gout(Ts&... args)
{
    return GRunArgsP{ GRunArgP(detail::wrap_host_helper<Ts>::wrap_out(args))... };
}

} // namespace cv

#endif // OPENCV_GAPI_GARG_HPP
