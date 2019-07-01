// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_INFER_HPP
#define OPENCV_GAPI_INFER_HPP

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

struct GInferBase {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer";     // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                // One more universal stub
    }
};
struct GInferListBase {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer-roi"; // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                // One more universal stub
    }
};

template<typename Net>
struct GInfer final
    : public GInferBase
    , public detail::KernelTypeMedium< GInfer<Net>
                                     , typename Net::API > {
    using GInferBase::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};
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
#define GAPI_NETWORK(Class, API, Tag)                                   \
    struct Class final: public cv::GNetworkType<Class, std::function API> { \
        static constexpr const char * tag() { return Tag; }             \
    }

namespace cv {
namespace gapi {
// FIXME: (all) how to dictate infer API by its Net?

template<typename Net, typename... Args>
typename Net::ResultL infer(cv::GArray<cv::Rect> roi, Args&&... args) {
    return GInferList<Net>::on(roi, std::forward<Args>(args)...);
}

template<typename Net, typename... Args>
typename Net::Result infer(Args&&... args) {
    return GInfer<Net>::on(std::forward<Args>(args)...);
}

struct GAPI_EXPORTS GNetParam {
    std::string tag;     // FIXME: const?
    GBackend backend;    // Specifies the execution model
    util::any params;    // Backend-interpreted parameter structure
};

struct GAPI_EXPORTS GNetPackage {
    explicit GNetPackage(std::initializer_list<GNetParam> &&ii = {});
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
