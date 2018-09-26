// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMMON_HPP
#define OPENCV_GAPI_GCOMMON_HPP

#include <functional>   // std::hash
#include <vector>       // std::vector
#include <type_traits>  // decay

#include <opencv2/gapi/opencv_includes.hpp>

#include "opencv2/gapi/util/any.hpp"
#include "opencv2/gapi/own/exports.hpp"
#include "opencv2/gapi/own/assert.hpp"

namespace cv {

namespace detail
{
    // This is a trait-like structure to mark backend-specific compile arguments
    // with tags
    template<typename T> struct CompileArgTag;
    template<typename T> struct CompileArgTag
    {
        static const char* tag() { return ""; };
    };
}

// This definition is here because it is reused by both public(?) and internal
// modules. Keeping it here wouldn't expose public details (e.g., API-level)
// to components which are internal and operate on a lower-level entities
// (e.g., compiler, backends).
// FIXME: merge with ArgKind?
// FIXME: replace with variant[format desc]?
enum class GShape: int
{
    GMAT,
    GSCALAR,
    GARRAY,
};

struct GCompileArg;

namespace detail {
    template<typename T>
    using is_compile_arg = std::is_same<GCompileArg, typename std::decay<T>::type>;
}
// CompileArg is an unified interface over backend-specific compilation
// information
// FIXME: Move to a separate file?
struct GAPI_EXPORTS GCompileArg
{
public:
    std::string tag;

    // FIXME: use decay in GArg/other trait-based wrapper before leg is shot!
    template<typename T, typename std::enable_if<!detail::is_compile_arg<T>::value, int>::type = 0>
    explicit GCompileArg(T &&t)
        : tag(detail::CompileArgTag<typename std::decay<T>::type>::tag())
        , arg(t)
    {
    }

    template<typename T> T& get()
    {
        return util::any_cast<T>(arg);
    }

    template<typename T> const T& get() const
    {
        return util::any_cast<T>(arg);
    }

private:
    util::any arg;
};

using GCompileArgs = std::vector<GCompileArg>;

template<typename... Ts> GCompileArgs compile_args(Ts&&... args)
{
    return GCompileArgs{ GCompileArg(args)... };
}

struct graph_dump_path
{
    std::string m_dump_path;
};

namespace detail
{
    template<> struct CompileArgTag<cv::graph_dump_path>
    {
        static const char* tag() { return "gapi.graph_dump_path"; }
    };
}

} // namespace cv

// std::hash overload for GShape
namespace std
{
template<> struct hash<cv::GShape>
{
    size_t operator() (cv::GShape sh) const
    {
        return std::hash<int>()(static_cast<int>(sh));
    }
};
} // namespace std


#endif // OPENCV_GAPI_GCOMMON_HPP
