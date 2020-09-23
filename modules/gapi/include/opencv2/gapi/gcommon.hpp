// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GCOMMON_HPP
#define OPENCV_GAPI_GCOMMON_HPP

#include <functional>   // std::hash
#include <vector>       // std::vector
#include <type_traits>  // decay

#include <opencv2/gapi/opencv_includes.hpp>

#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/gapi/own/exports.hpp>
#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/render/render_types.hpp>

namespace cv {

class GMat; // FIXME: forward declaration for GOpaqueTraits

namespace detail
{
    // This is a trait-like structure to mark backend-specific compile arguments
    // with tags
    template<typename T> struct CompileArgTag;

    // These structures are tags which separate kernels and transformations
    struct KernelTag
    {};
    struct TransformTag
    {};

    // This enum is utilized mostly by GArray and GOpaque to store and recognize their internal data
    // types (aka Host type). Also it is widely used during serialization routine.
    enum class OpaqueKind: int
    {
        CV_UNKNOWN,    // Unknown, generic, opaque-to-GAPI data type unsupported in graph seriallization
        CV_BOOL,       // bool user G-API data
        CV_INT,        // int user G-API data
        CV_DOUBLE,     // double user G-API data
        CV_POINT,      // cv::Point user G-API data
        CV_SIZE,       // cv::Size user G-API data
        CV_RECT,       // cv::Rect user G-API data
        CV_SCALAR,     // cv::Scalar user G-API data
        CV_MAT,        // cv::Mat user G-API data
        CV_PRIM,       // cv::gapi::wip::draw::Prim user G-API data
    };

    // Type traits helper which simplifies the extraction of kind from type
    template<typename T> struct GOpaqueTraits;
    template<typename T> struct GOpaqueTraits    { static constexpr const OpaqueKind kind = OpaqueKind::CV_UNKNOWN; };
    template<> struct GOpaqueTraits<int>         { static constexpr const OpaqueKind kind = OpaqueKind::CV_INT; };
    template<> struct GOpaqueTraits<double>      { static constexpr const OpaqueKind kind = OpaqueKind::CV_DOUBLE; };
    template<> struct GOpaqueTraits<cv::Size>    { static constexpr const OpaqueKind kind = OpaqueKind::CV_SIZE; };
    template<> struct GOpaqueTraits<bool>        { static constexpr const OpaqueKind kind = OpaqueKind::CV_BOOL; };
    template<> struct GOpaqueTraits<cv::Scalar>  { static constexpr const OpaqueKind kind = OpaqueKind::CV_SCALAR; };
    template<> struct GOpaqueTraits<cv::Point>   { static constexpr const OpaqueKind kind = OpaqueKind::CV_POINT; };
    template<> struct GOpaqueTraits<cv::Mat>     { static constexpr const OpaqueKind kind = OpaqueKind::CV_MAT; };
    template<> struct GOpaqueTraits<cv::Rect>    { static constexpr const OpaqueKind kind = OpaqueKind::CV_RECT; };
    template<> struct GOpaqueTraits<cv::GMat>    { static constexpr const OpaqueKind kind = OpaqueKind::CV_MAT; };
    template<> struct GOpaqueTraits<cv::gapi::wip::draw::Prim>
                                                 { static constexpr const OpaqueKind kind = OpaqueKind::CV_PRIM; };
    // GArray is not supporting bool type for now due to difference in std::vector<bool> implementation
    using GOpaqueTraitsArrayTypes = std::tuple<int, double, cv::Size, cv::Scalar, cv::Point, cv::Mat, cv::Rect, cv::gapi::wip::draw::Prim>;
    // GOpaque is not supporting cv::Mat and cv::Scalar since there are GScalar and GMat types
    using GOpaqueTraitsOpaqueTypes = std::tuple<bool, int, double, cv::Size, cv::Point, cv::Rect, cv::gapi::wip::draw::Prim>;
} // namespace detail

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
    GOPAQUE,
    GFRAME,
};

struct GCompileArg;

namespace detail {
    template<typename T>
    using is_compile_arg = std::is_same<GCompileArg, typename std::decay<T>::type>;
} // namespace detail

// CompileArg is an unified interface over backend-specific compilation
// information
// FIXME: Move to a separate file?
/** \addtogroup gapi_compile_args
 * @{
 *
 * @brief Compilation arguments: data structures controlling the
 * compilation process
 *
 * G-API comes with a number of graph compilation options which can be
 * passed to cv::GComputation::apply() or
 * cv::GComputation::compile(). Known compilation options are listed
 * in this page, while extra backends may introduce their own
 * compilation options (G-API transparently accepts _everything_ which
 * can be passed to cv::compile_args(), it depends on underlying
 * backends if an option would be interpreted or not).
 *
 * For example, if an example computation is executed like this:
 *
 * @snippet modules/gapi/samples/api_ref_snippets.cpp graph_decl_apply
 *
 * Extra parameter specifying which kernels to compile with can be
 * passed like this:
 *
 * @snippet modules/gapi/samples/api_ref_snippets.cpp apply_with_param
 */

/**
 * @brief Represents an arbitrary compilation argument.
 *
 * Any value can be wrapped into cv::GCompileArg, but only known ones
 * (to G-API or its backends) can be interpreted correctly.
 *
 * Normally objects of this class shouldn't be created manually, use
 * cv::compile_args() function which automatically wraps everything
 * passed in (a variadic template parameter pack) into a vector of
 * cv::GCompileArg objects.
 */
struct GAPI_EXPORTS_W_SIMPLE GCompileArg
{
public:
    // NB: Required for pythnon bindings
    GCompileArg() = default;

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

/**
 * @brief Wraps a list of arguments (a parameter pack) into a vector of
 *        compilation arguments (cv::GCompileArg).
 */
template<typename... Ts> GCompileArgs compile_args(Ts&&... args)
{
    return GCompileArgs{ GCompileArg(args)... };
}

/**
 * @brief Retrieves particular compilation argument by its type from
 *        cv::GCompileArgs
 */
namespace gapi
{
template<typename T>
inline cv::util::optional<T> getCompileArg(const cv::GCompileArgs &args)
{
    for (auto &compile_arg : args)
    {
        if (compile_arg.tag == cv::detail::CompileArgTag<T>::tag())
        {
            return cv::util::optional<T>(compile_arg.get<T>());
        }
    }
    return cv::util::optional<T>();
}
} // namespace gapi

/**
 * @brief Ask G-API to dump compiled graph in Graphviz format under
 * the given file name.
 *
 * Specifies a graph dump path (path to .dot file to be generated).
 * G-API will dump a .dot file under specified path during a
 * compilation process if this flag is passed.
 */
struct graph_dump_path
{
    std::string m_dump_path;
};
/** @} */

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
