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

#include <opencv2/gapi/util/any.hpp>  // any<>
#include <opencv2/gapi/gkernel.hpp>   // GKernelType[M], GBackend
#include <opencv2/gapi/garg.hpp>      // GArg
#include <opencv2/gapi/gcommon.hpp>   // CompileArgTag
#include <opencv2/gapi/gmetaarg.hpp>  // GMetaArg

namespace cv {

namespace detail {
    // This tiny class eliminates the semantic difference between
    // GKernelType and GKernelTypeM.
    // FIXME: Something similar can be reused for regular kernels
    template<typename, typename>
    struct KernelTypeMedium;

    template<class K, typename... R, typename... Args>
    struct KernelTypeMedium<K, std::function<std::tuple<R...>(Args...)> >:
        public GKernelTypeM<K, std::function<std::tuple<R...>(Args...)> > {};

    template<class K, typename R, typename... Args>
    struct KernelTypeMedium<K, std::function<R(Args...)> >:
        public GKernelType<K, std::function<R(Args...)> > {};

} // namespace detail

template<typename, typename> class GNetworkType;

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
        return "org.opencv.dnn.infer-roi"; // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                // One more universal stub
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
