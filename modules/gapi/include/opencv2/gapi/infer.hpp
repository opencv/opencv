// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_INFER_HPP
#define OPENCV_GAPI_INFER_HPP

// FIXME: Inference API is currently only available in full mode
#if !defined(GAPI_STANDALONE)

#include <functional>
#include <string>  // string
#include <utility> // tuple
#include <type_traits> // is_same, false_type

#include <opencv2/gapi/util/any.hpp>  // any<>
#include <opencv2/gapi/gkernel.hpp>   // GKernelType[M], GBackend
#include <opencv2/gapi/garg.hpp>      // GArg
#include <opencv2/gapi/gcommon.hpp>   // CompileArgTag
#include <opencv2/gapi/gmetaarg.hpp>  // GMetaArg

namespace cv {

template<typename, typename> class GNetworkType;

namespace detail {
    template<typename, typename>
    struct valid_infer2_types;

    // Terminal case 1 (50/50 success)
    template<typename T>
    struct valid_infer2_types< std::tuple<cv::GMat>, std::tuple<T> > {
        // By default, Nets are limited to GMat argument types only
        // for infer2, every GMat argument may translate to either
        // GArray<GMat> or GArray<Rect>. GArray<> part is stripped
        // already at this point.
        static constexpr const auto value =
                std::is_same<typename std::decay<T>::type, cv::GMat>::value
             || std::is_same<typename std::decay<T>::type, cv::Rect>::value;
    };

    // Terminal case 2 (100% failure)
    template<typename... Ts>
    struct valid_infer2_types< std::tuple<>, std::tuple<Ts...> >
        : public std::false_type {
    };

    // Terminal case 3 (100% failure)
    template<typename... Ns>
    struct valid_infer2_types< std::tuple<Ns...>, std::tuple<> >
        : public std::false_type {
    };

    // Recursion -- generic
    template<typename... Ns, typename T, typename...Ts>
    struct valid_infer2_types< std::tuple<cv::GMat,Ns...>, std::tuple<T,Ts...> > {
        static constexpr const auto value =
               valid_infer2_types< std::tuple<cv::GMat>, std::tuple<T> >::value
            && valid_infer2_types< std::tuple<Ns...>, std::tuple<Ts...> >::value;
    };
} // namespace detail

// TODO: maybe tuple_wrap_helper from util.hpp may help with this.
// Multiple-return-value network definition (specialized base class)
template<typename K, typename... R, typename... Args>
class GNetworkType<K, std::function<std::tuple<R...>(Args...)> >
{
public:
    using InArgs  = std::tuple<Args...>;
    using OutArgs = std::tuple<R...>;

    using Result  = OutArgs;
    using API     = std::function<Result(Args...)>;

    using ResultL = std::tuple< cv::GArray<R>... >;
    using APIList = std::function<ResultL(cv::GArray<cv::Rect>, Args...)>;
};

// Single-return-value network definition (specialized base class)
template<typename K, typename R, typename... Args>
class GNetworkType<K, std::function<R(Args...)> >
{
public:
    using InArgs  = std::tuple<Args...>;
    using OutArgs = std::tuple<R>;

    using Result  = R;
    using API     = std::function<R(Args...)>;

    using ResultL = cv::GArray<R>;
    using APIList = std::function<ResultL(cv::GArray<cv::Rect>, Args...)>;
};

// APIList2 is also template to allow different calling options
// (GArray<cv::Rect> vs GArray<cv::GMat> per input)
template<class Net, class... Ts>
struct InferAPIList2 {
    using type = typename std::enable_if
        < cv::detail::valid_infer2_types< typename Net::InArgs
                                        , std::tuple<Ts...> >::value,
          std::function<typename Net::ResultL(cv::GMat, cv::GArray<Ts>...)>
        >::type;
};

// Base "Infer" kernel. Note - for whatever network, kernel ID
// is always the same. Different inference calls are distinguished by
// network _tag_ (an extra field in GCall)
//
// getOutMeta is a stub callback collected by G-API kernel subsystem
// automatically. This is a rare case when this callback is defined by
// a particular backend, not by a network itself.
struct GInferBase {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer";     // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                // One more universal stub
    }
};


// Base "Infer list" kernel.
// All notes from "Infer" kernel apply here as well.
struct GInferListBase {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer-roi";      // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                     // One more universal stub
    }
};

// Base "Infer list 2" kernel.
// All notes from "Infer" kernel apply here as well.
struct GInferList2Base {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer-roi-list"; // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                     // One more universal stub
    }
};

// A generic inference kernel. API (::on()) is fully defined by the Net
// template parameter.
// Acts as a regular kernel in graph (via KernelTypeMedium).
template<typename Net>
struct GInfer final
    : public GInferBase
    , public detail::KernelTypeMedium< GInfer<Net>
                                     , typename Net::API > {
    using GInferBase::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};

// A generic roi-list inference kernel. API (::on()) is derived from
// the Net template parameter (see more in infer<> overload).
template<typename Net>
struct GInferList final
    : public GInferListBase
    , public detail::KernelTypeMedium< GInferList<Net>
                                     , typename Net::APIList > {
    using GInferListBase::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};

// An even more generic roi-list inference kernel. API (::on()) is
// derived from the Net template parameter (see more in infer<>
// overload).
// Takes an extra variadic template list to reflect how this network
// was called (with Rects or GMats as array parameters)
template<typename Net, typename... Args>
struct GInferList2 final
    : public GInferList2Base
    , public detail::KernelTypeMedium< GInferList2<Net, Args...>
                                     , typename InferAPIList2<Net, Args...>::type > {
    using GInferList2Base::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};

} // namespace cv

// FIXME: Probably the <API> signature makes a function/tuple/function round-trip
#define G_API_NET(Class, API, Tag)                                      \
    struct Class final: public cv::GNetworkType<Class, std::function API> { \
        static constexpr const char * tag() { return Tag; }             \
    }

namespace cv {
namespace gapi {


/** @brief Calculates responses for the specified network (template
 *     parameter) for every region in the source image.
 *
 * @tparam A network type defined with G_API_NET() macro.
 * @param roi a list of rectangles describing regions of interest
 *   in the source image. Usually an output of object detector or tracker.
 * @param args network's input parameters as specified in G_API_NET() macro.
 *   NOTE: verified to work reliably with 1-input topologies only.
 * @return a list of objects of return type as defined in G_API_NET().
 *   If a network has multiple return values (defined with a tuple), a tuple of
 *   GArray<> objects is returned with the appropriate types inside.
 * @sa  G_API_NET()
 */
template<typename Net, typename... Args>
typename Net::ResultL infer(cv::GArray<cv::Rect> roi, Args&&... args) {
    return GInferList<Net>::on(roi, std::forward<Args>(args)...);
}

/** @brief Calculates responses for the specified network (template
 *     parameter) for every region in the source image, extended version.
 *
 * @tparam A network type defined with G_API_NET() macro.
 * @param image A source image containing regions of interest
 * @param args GArray<> objects of cv::Rect or cv::GMat, one per every
 * network input:
 * - If a cv::GArray<cv::Rect> is passed, the appropriate
 *   regions are taken from `image` and preprocessed to this particular
 *   network input;
 * - If a cv::GArray<cv::GMat> is passed, the underlying data traited
 *   as tensor (no automatic preprocessing happen).
 * @return a list of objects of return type as defined in G_API_NET().
 *   If a network has multiple return values (defined with a tuple), a tuple of
 *   GArray<> objects is returned with the appropriate types inside.
 * @sa  G_API_NET()
 */
template<typename Net, typename... Args>
typename Net::ResultL infer2(cv::GMat image, cv::GArray<Args>... args) {
    // FIXME: Declared as "2" because in the current form it steals
    // overloads from the regular infer
    return GInferList2<Net, Args...>::on(image, args...);
}

/**
 * @brief Calculates response for the specified network (template
 *     parameter) given the input data.
 *
 * @tparam A network type defined with G_API_NET() macro.
 * @param args network's input parameters as specified in G_API_NET() macro.
 * @return an object of return type as defined in G_API_NET().
 *   If a network has multiple return values (defined with a tuple), a tuple of
 *   objects of appropriate type is returned.
 * @sa  G_API_NET()
 */
template<typename Net, typename... Args>
typename Net::Result infer(Args&&... args) {
    return GInfer<Net>::on(std::forward<Args>(args)...);
}


} // namespace gapi
} // namespace cv

#endif // GAPI_STANDALONE

namespace cv {
namespace gapi {

// Note: the below code _is_ part of STANDALONE build,
// just to make our compiler code compileable.

// A type-erased form of network parameters.
// Similar to how a type-erased GKernel is represented and used.
struct GAPI_EXPORTS GNetParam {
    std::string tag;     // FIXME: const?
    GBackend backend;    // Specifies the execution model
    util::any params;    // Backend-interpreted parameter structure
};

/** \addtogroup gapi_compile_args
 * @{
 */
/**
 * @brief A container class for network configurations. Similar to
 * GKernelPackage.Use cv::gapi::networks() to construct this object.
 *
 * @sa cv::gapi::networks
 */
struct GAPI_EXPORTS GNetPackage {
    GNetPackage() : GNetPackage({}) {}
    explicit GNetPackage(std::initializer_list<GNetParam> &&ii);
    std::vector<GBackend> backends() const;
    std::vector<GNetParam> networks;
};
/** @} gapi_compile_args */
} // namespace gapi

namespace detail {
template<typename T>
gapi::GNetParam strip(T&& t) {
    return gapi::GNetParam { t.tag()
                           , t.backend()
                           , t.params()
                           };
}

template<> struct CompileArgTag<cv::gapi::GNetPackage> {
    static const char* tag() { return "gapi.net_package"; }
};

} // namespace cv::detail

namespace gapi {
template<typename... Args>
cv::gapi::GNetPackage networks(Args&&... args) {
    return cv::gapi::GNetPackage({ cv::detail::strip(args)... });
}
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_HPP
