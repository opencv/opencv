// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021 Intel Corporation


#ifndef OPENCV_GAPI_INFER_HPP
#define OPENCV_GAPI_INFER_HPP

// FIXME: Inference API is currently only available in full mode
#if !defined(GAPI_STANDALONE)

#include <functional>
#include <string>  // string
#include <utility> // tuple
#include <type_traits> // is_same, false_type

#include <opencv2/gapi/util/util.hpp> // all_satisfy
#include <opencv2/gapi/util/any.hpp>  // any<>
#include <opencv2/gapi/gkernel.hpp>   // GKernelType[M], GBackend
#include <opencv2/gapi/garg.hpp>      // GArg
#include <opencv2/gapi/gcommon.hpp>   // CompileArgTag
#include <opencv2/gapi/gmetaarg.hpp>  // GMetaArg

namespace cv {

template<typename, typename> class GNetworkType;

namespace detail {

// Infer ///////////////////////////////////////////////////////////////////////
template<typename T>
struct accepted_infer_types {
    static constexpr const auto value =
            std::is_same<typename std::decay<T>::type, cv::GMat>::value
         || std::is_same<typename std::decay<T>::type, cv::GFrame>::value;
};

template<typename... Ts>
using valid_infer_types = all_satisfy<accepted_infer_types, Ts...>;

// Infer2 //////////////////////////////////////////////////////////////////////

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

// Struct stores network input/output names.
// Used by infer<Generic>
struct InOutInfo
{
    std::vector<std::string> in_names;
    std::vector<std::string> out_names;
};

template <typename OutT>
class GInferOutputsTyped
{
public:
    GInferOutputsTyped() = default;
    GInferOutputsTyped(std::shared_ptr<cv::GCall> call)
        : m_priv(std::make_shared<Priv>(std::move(call)))
    {
    }

    OutT at(const std::string& name)
    {
        auto it = m_priv->blobs.find(name);
        if (it == m_priv->blobs.end()) {
            // FIXME: Avoid modifying GKernel
            auto shape = cv::detail::GTypeTraits<OutT>::shape;
            auto kind  = cv::detail::GTypeTraits<OutT>::op_kind;
            m_priv->call->kernel().outShapes.push_back(shape);
            m_priv->call->kernel().outCtors.emplace_back(cv::detail::GObtainCtor<OutT>::get());
            m_priv->call->kernel().outKinds.emplace_back(kind);
            auto out_idx = static_cast<int>(m_priv->blobs.size());
            it = m_priv->blobs.emplace(name,
                    cv::detail::Yield<OutT>::yield(*(m_priv->call), out_idx)).first;
            m_priv->info->out_names.push_back(name);
        }
        return it->second;
    }
private:
    struct Priv
    {
        Priv(std::shared_ptr<cv::GCall> c)
            : call(std::move(c)), info(cv::util::any_cast<InOutInfo>(&call->params()))
        {
        }

        std::shared_ptr<cv::GCall> call;
        InOutInfo* info = nullptr;
        std::unordered_map<std::string, OutT> blobs;
    };

    std::shared_ptr<Priv> m_priv;
};

template <typename... Ts>
class GInferInputsTyped
{
public:
    GInferInputsTyped()
        : m_priv(std::make_shared<Priv>())
    {
    }

    template <typename U>
    GInferInputsTyped<Ts...>& setInput(const std::string& name, U in)
    {
        m_priv->blobs.emplace(std::piecewise_construct,
                              std::forward_as_tuple(name),
                              std::forward_as_tuple(in));
        return *this;
    }

    using StorageT = cv::util::variant<Ts...>;
    StorageT& operator[](const std::string& name) {
        return m_priv->blobs[name];
    }

    using Map = std::unordered_map<std::string, StorageT>;
    const Map& getBlobs() const {
        return m_priv->blobs;
    }

private:
    struct Priv
    {
        std::unordered_map<std::string, StorageT> blobs;
    };

    std::shared_ptr<Priv> m_priv;
};

template<typename InferT>
std::shared_ptr<cv::GCall> makeCall(const std::string         &tag,
                                    std::vector<cv::GArg>    &&args,
                                    std::vector<std::string> &&names,
                                    cv::GKinds               &&kinds) {
    auto call = std::make_shared<cv::GCall>(GKernel{
                InferT::id(),
                tag,
                InferT::getOutMeta,
                {}, // outShape will be filled later
                std::move(kinds),
                {}, // outCtors will be filled later
                {}, // outKinds will be filled later
            });

    call->setArgs(std::move(args));
    call->params() = cv::detail::InOutInfo{std::move(names), {}};

    return call;
}

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
};

// InferAPI: Accepts either GMat or GFrame for very individual network's input
template<class Net, class... Ts>
struct InferAPI {
    using type = typename std::enable_if
        <    detail::valid_infer_types<Ts...>::value
          && std::tuple_size<typename Net::InArgs>::value == sizeof...(Ts)
        , std::function<typename Net::Result(Ts...)>
        >::type;
};

// InferAPIRoi: Accepts a rectangle and either GMat or GFrame
template<class Net, class T>
struct InferAPIRoi {
    using type = typename std::enable_if
        <    detail::valid_infer_types<T>::value
          && std::tuple_size<typename Net::InArgs>::value == 1u
          , std::function<typename Net::Result(cv::GOpaque<cv::Rect>, T)>
        >::type;
};

// InferAPIList: Accepts a list of rectangles and list of GMat/GFrames;
// crops every input.
template<class Net, class... Ts>
struct InferAPIList {
    using type = typename std::enable_if
        <    detail::valid_infer_types<Ts...>::value
          && std::tuple_size<typename Net::InArgs>::value == sizeof...(Ts)
        , std::function<typename Net::ResultL(cv::GArray<cv::Rect>, Ts...)>
        >::type;
};

// APIList2 is also template to allow different calling options
// (GArray<cv::Rect> vs GArray<cv::GMat> per input)
template<class Net, typename T, class... Ts>
struct InferAPIList2 {
    using type = typename std::enable_if
        < detail::valid_infer_types<T>::value &&
          cv::detail::valid_infer2_types< typename Net::InArgs
                                        , std::tuple<Ts...> >::value,
          std::function<typename Net::ResultL(T, cv::GArray<Ts>...)>
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
        return "org.opencv.dnn.infer";            // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                       // One more universal stub
    }
};

// Base "InferROI" kernel.
// All notes from "Infer" kernel apply here as well.
struct GInferROIBase {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer-roi";        // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                       // One more universal stub
    }
};

// Base "Infer list" kernel.
// All notes from "Infer" kernel apply here as well.
struct GInferListBase {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer-roi-list-1"; // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                       // One more universal stub
    }
};

// Base "Infer list 2" kernel.
// All notes from "Infer" kernel apply here as well.
struct GInferList2Base {
    static constexpr const char * id() {
        return "org.opencv.dnn.infer-roi-list-2"; // Universal stub
    }
    static GMetaArgs getOutMeta(const GMetaArgs &, const GArgs &) {
        return GMetaArgs{};                       // One more universal stub
    }
};

// A generic inference kernel. API (::on()) is fully defined by the Net
// template parameter.
// Acts as a regular kernel in graph (via KernelTypeMedium).
template<typename Net, typename... Args>
struct GInfer final
    : public GInferBase
    , public detail::KernelTypeMedium< GInfer<Net, Args...>
                                     , typename InferAPI<Net, Args...>::type > {
    using GInferBase::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};

// A specific roi-inference kernel. API (::on()) is fixed here and
// verified against Net.
template<typename Net, typename T>
struct GInferROI final
    : public GInferROIBase
    , public detail::KernelTypeMedium< GInferROI<Net, T>
                                     , typename InferAPIRoi<Net, T>::type > {
    using GInferROIBase::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};


// A generic roi-list inference kernel. API (::on()) is derived from
// the Net template parameter (see more in infer<> overload).
template<typename Net, typename... Args>
struct GInferList final
    : public GInferListBase
    , public detail::KernelTypeMedium< GInferList<Net, Args...>
                                     , typename InferAPIList<Net, Args...>::type > {
    using GInferListBase::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};

// An even more generic roi-list inference kernel. API (::on()) is
// derived from the Net template parameter (see more in infer<>
// overload).
// Takes an extra variadic template list to reflect how this network
// was called (with Rects or GMats as array parameters)
template<typename Net, typename T, typename... Args>
struct GInferList2 final
    : public GInferList2Base
    , public detail::KernelTypeMedium< GInferList2<Net, T, Args...>
                                     , typename InferAPIList2<Net, T, Args...>::type > {
    using GInferList2Base::getOutMeta; // FIXME: name lookup conflict workaround?

    static constexpr const char* tag() { return Net::tag(); }
};

/**
 * @brief G-API object used to collect network inputs
 */
using GInferInputs = cv::detail::GInferInputsTyped<cv::GMat, cv::GFrame>;

/**
 * @brief G-API object used to collect the list of network inputs
 */
using GInferListInputs = cv::detail::GInferInputsTyped<cv::GArray<cv::GMat>, cv::GArray<cv::Rect>>;

/**
 * @brief G-API object used to collect network outputs
 */
using GInferOutputs = cv::detail::GInferOutputsTyped<cv::GMat>;

/**
 * @brief G-API object used to collect the list of network outputs
 */
using GInferListOutputs = cv::detail::GInferOutputsTyped<cv::GArray<cv::GMat>>;

namespace detail {
void inline unpackBlobs(const cv::GInferInputs::Map& blobs,
                        std::vector<cv::GArg>& args,
                        std::vector<std::string>& names,
                        cv::GKinds& kinds)
{
    for (auto&& p : blobs) {
        names.emplace_back(p.first);
        switch (p.second.index()) {
            case cv::GInferInputs::StorageT::index_of<cv::GMat>():
                args.emplace_back(cv::util::get<cv::GMat>(p.second));
                kinds.emplace_back(cv::detail::OpaqueKind::CV_MAT);
                break;
            case cv::GInferInputs::StorageT::index_of<cv::GFrame>():
                args.emplace_back(cv::util::get<cv::GFrame>(p.second));
                kinds.emplace_back(cv::detail::OpaqueKind::CV_UNKNOWN);
                break;
            default:
                GAPI_Error("InternalError");
        }
    }
}

template <typename InferType>
struct InferROITraits;

template <>
struct InferROITraits<GInferROIBase>
{
    using outType = cv::GInferOutputs;
    using inType  = cv::GOpaque<cv::Rect>;
};

template <>
struct InferROITraits<GInferListBase>
{
    using outType = cv::GInferListOutputs;
    using inType  = cv::GArray<cv::Rect>;
};

template<typename InferType>
typename InferROITraits<InferType>::outType
inferGenericROI(const std::string& tag,
         const typename InferROITraits<InferType>::inType& in,
         const cv::GInferInputs& inputs)
{
    std::vector<cv::GArg> args;
    std::vector<std::string> names;
    cv::GKinds kinds;

    args.emplace_back(in);
    kinds.emplace_back(cv::detail::OpaqueKind::CV_RECT);

    unpackBlobs(inputs.getBlobs(), args, names, kinds);

    auto call = cv::detail::makeCall<InferType>(tag,
                                                std::move(args),
                                                std::move(names),
                                                std::move(kinds));

    return {std::move(call)};
}

} // namespace detail
} // namespace cv

// FIXME: Probably the <API> signature makes a function/tuple/function round-trip
#define G_API_NET(Class, API, Tag)                                      \
    struct Class final: public cv::GNetworkType<Class, std::function API> { \
        static constexpr const char * tag() { return Tag; }             \
    }

namespace cv {
namespace gapi {

/** @brief Calculates response for the specified network (template
 *     parameter) for the specified region in the source image.
 *     Currently expects a single-input network only.
 *
 * @tparam A network type defined with G_API_NET() macro.
 * @param in input image where to take ROI from.
 * @param roi an object describing the region of interest
 *   in the source image. May be calculated in the same graph dynamically.
 * @return an object of return type as defined in G_API_NET().
 *   If a network has multiple return values (defined with a tuple), a tuple of
 *   objects of appropriate type is returned.
 * @sa  G_API_NET()
 */
template<typename Net, typename T>
typename Net::Result infer(cv::GOpaque<cv::Rect> roi, T in) {
    return GInferROI<Net, T>::on(roi, in);
}

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
    return GInferList<Net, Args...>::on(roi, std::forward<Args>(args)...);
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

template<typename Net, typename T, typename... Args>
typename Net::ResultL infer2(T image, cv::GArray<Args>... args) {
    // FIXME: Declared as "2" because in the current form it steals
    // overloads from the regular infer
    return GInferList2<Net, T, Args...>::on(image, args...);
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
    return GInfer<Net, Args...>::on(std::forward<Args>(args)...);
}

/**
 * @brief Generic network type: input and output layers are configured dynamically at runtime
 *
 * Unlike the network types defined with G_API_NET macro, this one
 * doesn't fix number of network inputs and outputs at the compilation stage
 * thus providing user with an opportunity to program them in runtime.
 */
struct Generic { };

/**
 * @brief Calculates response for generic network
 *
 * @param tag a network tag
 * @param inputs networks's inputs
 * @return a GInferOutputs
 */
template<typename T = Generic> cv::GInferOutputs
infer(const std::string& tag, const cv::GInferInputs& inputs)
{
    std::vector<cv::GArg> args;
    std::vector<std::string> names;
    cv::GKinds kinds;

    cv::detail::unpackBlobs(inputs.getBlobs(), args, names, kinds);

    auto call = cv::detail::makeCall<GInferBase>(tag,
                                                 std::move(args),
                                                 std::move(names),
                                                 std::move(kinds));

    return cv::GInferOutputs{std::move(call)};
}

/** @brief Calculates response for the generic network
 *     for the specified region in the source image.
 *     Currently expects a single-input network only.
 *
 * @param tag a network tag
 * @param roi a an object describing the region of interest
 *   in the source image. May be calculated in the same graph dynamically.
 * @param inputs networks's inputs
 * @return a cv::GInferOutputs
 */
template<typename T = Generic> cv::GInferOutputs
infer(const std::string& tag, const cv::GOpaque<cv::Rect>& roi, const cv::GInferInputs& inputs)
{
    return cv::detail::inferGenericROI<GInferROIBase>(tag, roi, inputs);
}

/** @brief Calculates responses for the specified network
 *     for every region in the source image.
 *
 * @param tag a network tag
 * @param rois a list of rectangles describing regions of interest
 *   in the source image. Usually an output of object detector or tracker.
 * @param inputs networks's inputs
 * @return a cv::GInferListOutputs
 */
template<typename T = Generic> cv::GInferListOutputs
infer(const std::string& tag, const cv::GArray<cv::Rect>& rois, const cv::GInferInputs& inputs)
{
    return cv::detail::inferGenericROI<GInferListBase>(tag, rois, inputs);
}

/** @brief Calculates responses for the specified network
 *     for every region in the source image, extended version.
 *
 * @param tag a network tag
 * @param in a source image containing regions of interest.
 * @param inputs networks's inputs
 * @return a cv::GInferListOutputs
 */
template<typename T = Generic, typename Input>
typename std::enable_if<cv::detail::accepted_infer_types<Input>::value, cv::GInferListOutputs>::type
infer2(const std::string& tag,
       const Input& in,
       const cv::GInferListInputs& inputs)
{
    std::vector<cv::GArg> args;
    std::vector<std::string> names;
    cv::GKinds kinds;

    args.emplace_back(in);
    auto k = cv::detail::GOpaqueTraits<Input>::kind;
    kinds.emplace_back(k);

    for (auto&& p : inputs.getBlobs()) {
        names.emplace_back(p.first);
        switch (p.second.index()) {
            case cv::GInferListInputs::StorageT::index_of<cv::GArray<cv::GMat>>():
                args.emplace_back(cv::util::get<cv::GArray<cv::GMat>>(p.second));
                kinds.emplace_back(cv::detail::OpaqueKind::CV_MAT);
                break;
            case cv::GInferListInputs::StorageT::index_of<cv::GArray<cv::Rect>>():
                args.emplace_back(cv::util::get<cv::GArray<cv::Rect>>(p.second));
                kinds.emplace_back(cv::detail::OpaqueKind::CV_RECT);
                break;
            default:
                GAPI_Error("InternalError");
        }
    }

    auto call = cv::detail::makeCall<GInferList2Base>(tag,
                                                      std::move(args),
                                                      std::move(names),
                                                      std::move(kinds));

    return cv::GInferListOutputs{std::move(call)};
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
/// @private
struct GAPI_EXPORTS_W_SIMPLE GNetParam {
    std::string tag;     // FIXME: const?
    GBackend backend;    // Specifies the execution model
    util::any params;    // Backend-interpreted parameter structure
};

/** \addtogroup gapi_compile_args
 * @{
 */
/**
 * @brief A container class for network configurations. Similar to
 * GKernelPackage. Use cv::gapi::networks() to construct this object.
 *
 * @sa cv::gapi::networks
 */
struct GAPI_EXPORTS_W_SIMPLE GNetPackage {
    GAPI_WRAP GNetPackage() = default;
    GAPI_WRAP explicit GNetPackage(std::vector<GNetParam> nets);
    explicit GNetPackage(std::initializer_list<GNetParam> ii);
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

inline cv::gapi::GNetPackage& operator += (      cv::gapi::GNetPackage& lhs,
                                           const cv::gapi::GNetPackage& rhs) {
    lhs.networks.reserve(lhs.networks.size() + rhs.networks.size());
    lhs.networks.insert(lhs.networks.end(), rhs.networks.begin(), rhs.networks.end());
    return lhs;
}

} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_HPP
