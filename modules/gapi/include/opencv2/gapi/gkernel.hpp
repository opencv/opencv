// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GKERNEL_HPP
#define OPENCV_GAPI_GKERNEL_HPP

#include <functional>
#include <iostream>
#include <string>  // string
#include <type_traits> // false_type, true_type
#include <unordered_map> // map (for GKernelPackage)
#include <utility> // tuple
#include <vector>  // lookup order

#include <opencv2/gapi/gcommon.hpp> // CompileArgTag
#include <opencv2/gapi/util/util.hpp> // Seq
#include <opencv2/gapi/gcall.hpp>
#include <opencv2/gapi/garg.hpp>      // GArg
#include <opencv2/gapi/gmetaarg.hpp>  // GMetaArg
#include <opencv2/gapi/gtype_traits.hpp> // GTypeTraits
#include <opencv2/gapi/util/compiler_hints.hpp> //suppress_unused_warning


namespace cv {

using GShapes = std::vector<GShape>;

// GKernel describes kernel API to the system
// FIXME: add attributes of a kernel, (e.g. number and types
// of inputs, etc)
struct GAPI_EXPORTS GKernel
{
    using M = std::function<GMetaArgs(const GMetaArgs &, const GArgs &)>;

    const std::string name;       // kernel ID, defined by its API (signature)
    const M           outMeta;    // generic adaptor to API::outMeta(...)
    const GShapes     outShapes; // types (shapes) kernel's outputs
};

// GKernelImpl describes particular kernel implementation to the system
struct GAPI_EXPORTS GKernelImpl
{
    util::any         opaque;    // backend-specific opaque info
};

template<typename, typename> class GKernelTypeM;

namespace detail
{
    ////////////////////////////////////////////////////////////////////////////
    // yield() is used in graph construction time as a generic method to obtain
    // lazy "return value" of G-API operations
    //
    namespace
    {

        template<typename T> struct Yield;
        template<> struct Yield<cv::GMat>
        {
            static inline cv::GMat yield(cv::GCall &call, int i) { return call.yield(i); }
        };
        template<> struct Yield<cv::GScalar>
        {
            static inline cv::GScalar yield(cv::GCall &call, int i) { return call.yieldScalar(i); }
        };
        template<typename U> struct Yield<cv::GArray<U> >
        {
            static inline cv::GArray<U> yield(cv::GCall &call, int i) { return call.yieldArray<U>(i); }
        };
    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////
    // Helper classes which brings outputMeta() marshalling to kernel
    // implementations
    //
    // 1. MetaType establishes G#Type -> G#Meta mapping between G-API dynamic
    //    types and its metadata descriptor types.
    //    This mapping is used to transform types to call outMeta() callback.
    template<typename T> struct MetaType;
    template<> struct MetaType<cv::GMat>    { using type = GMatDesc; };
    template<> struct MetaType<cv::GScalar> { using type = GScalarDesc; };
    template<typename U> struct MetaType<cv::GArray<U> > { using type = GArrayDesc; };
    template<typename T> struct MetaType    { using type = T; }; // opaque args passed as-is

    // 2. Hacky test based on MetaType to check if we operate on G-* type or not
    template<typename T> using is_nongapi_type = std::is_same<T, typename MetaType<T>::type>;

    // 3. Two ways to transform input arguments to its meta - for G-* and non-G* types:
    template<typename T>
    typename std::enable_if<!is_nongapi_type<T>::value, typename MetaType<T>::type>
    ::type get_in_meta(const GMetaArgs &in_meta, const GArgs &, int idx)
    {
        return util::get<typename MetaType<T>::type>(in_meta.at(idx));
    }

    template<typename T>
    typename std::enable_if<is_nongapi_type<T>::value, T>
    ::type get_in_meta(const GMetaArgs &, const GArgs &in_args, int idx)
    {
        return in_args.at(idx).template get<T>();
    }

    // 4. The MetaHelper itself: an entity which generates outMeta() call
    //    based on kernel signature, with arguments properly substituted.
    // 4.1 - case for multiple return values
    // FIXME: probably can be simplified with std::apply or analogue.
    template<typename, typename, typename>
    struct MetaHelper;

    template<typename K, typename... Ins, typename... Outs>
    struct MetaHelper<K, std::tuple<Ins...>, std::tuple<Outs...> >
    {
        template<int... IIs, int... OIs>
        static GMetaArgs getOutMeta_impl(const GMetaArgs &in_meta,
                                         const GArgs &in_args,
                                         detail::Seq<IIs...>,
                                         detail::Seq<OIs...>)
        {
            // FIXME: decay?
            using R   = std::tuple<typename MetaType<Outs>::type...>;
            const R r = K::outMeta( get_in_meta<Ins>(in_meta, in_args, IIs)... );
            return GMetaArgs{ GMetaArg(std::get<OIs>(r))... };
        }
        // FIXME: help users identify how outMeta must look like (via default impl w/static_assert?)

        static GMetaArgs getOutMeta(const GMetaArgs &in_meta,
                                    const GArgs &in_args)
        {
            return getOutMeta_impl(in_meta,
                                   in_args,
                                   typename detail::MkSeq<sizeof...(Ins)>::type(),
                                   typename detail::MkSeq<sizeof...(Outs)>::type());
        }
    };

    // 4.1 - case for a single return value
    // FIXME: How to avoid duplication here?
    template<typename K, typename... Ins, typename Out>
    struct MetaHelper<K, std::tuple<Ins...>, Out >
    {
        template<int... IIs>
        static GMetaArgs getOutMeta_impl(const GMetaArgs &in_meta,
                                         const GArgs &in_args,
                                         detail::Seq<IIs...>)
        {
            // FIXME: decay?
            using R = typename MetaType<Out>::type;
            const R r = K::outMeta( get_in_meta<Ins>(in_meta, in_args, IIs)... );
            return GMetaArgs{ GMetaArg(r) };
        }
        // FIXME: help users identify how outMeta must look like (via default impl w/static_assert?)

        static GMetaArgs getOutMeta(const GMetaArgs &in_meta,
                                    const GArgs &in_args)
        {
            return getOutMeta_impl(in_meta,
                                   in_args,
                                   typename detail::MkSeq<sizeof...(Ins)>::type());
        }
    };

} // namespace detail

// GKernelType and GKernelTypeM are base classes which implement typed ::on()
// method based on kernel signature. GKernelTypeM stands for multiple-return-value kernels
//
// G_TYPED_KERNEL and G_TYPED_KERNEK_M macros inherit user classes from GKernelType and
// GKernelTypeM respectively.

template<typename K, typename... R, typename... Args>
class GKernelTypeM<K, std::function<std::tuple<R...>(Args...)> >:
        public detail::MetaHelper<K, std::tuple<Args...>, std::tuple<R...> >
{
    template<int... IIs>
    static std::tuple<R...> yield(cv::GCall &call, detail::Seq<IIs...>)
    {
        return std::make_tuple(detail::Yield<R>::yield(call, IIs)...);
    }

public:
    using InArgs  = std::tuple<Args...>;
    using OutArgs = std::tuple<R...>;

    static std::tuple<R...> on(Args... args)
    {
        cv::GCall call(GKernel{K::id(), &K::getOutMeta, {detail::GTypeTraits<R>::shape...}});
        call.pass(args...);
        return yield(call, typename detail::MkSeq<sizeof...(R)>::type());
    }
};

template<typename, typename> class GKernelType;

template<typename K, typename R, typename... Args>
class GKernelType<K, std::function<R(Args...)> >:
        public detail::MetaHelper<K, std::tuple<Args...>, R >
{
public:
    using InArgs  = std::tuple<Args...>;
    using OutArgs = std::tuple<R>;

    static R on(Args... args)
    {
        cv::GCall call(GKernel{K::id(), &K::getOutMeta, {detail::GTypeTraits<R>::shape}});
        call.pass(args...);
        return detail::Yield<R>::yield(call, 0);
    }
};

} // namespace cv


// FIXME: I don't know a better way so far. Feel free to suggest one
// The problem is that every typed kernel should have ::id() but body
// of the class is defined by user (with outMeta, other stuff)

#define G_ID_HELPER_CLASS(Class)  Class##IdHelper

#define G_ID_HELPER_BODY(Class, Id)                                         \
    namespace detail                                                        \
    {                                                                       \
        struct G_ID_HELPER_CLASS(Class)                                     \
        {                                                                   \
            static constexpr const char * id() {return Id;};                \
        };                                                                  \
    }

#define G_TYPED_KERNEL(Class, API, Id)                                      \
    G_ID_HELPER_BODY(Class, Id)                                             \
    struct Class final: public cv::GKernelType<Class, std::function API >,  \
                        public detail::G_ID_HELPER_CLASS(Class)
// {body} is to be defined by user

#define G_TYPED_KERNEL_M(Class, API, Id)                                    \
    G_ID_HELPER_BODY(Class, Id)                                             \
    struct Class final: public cv::GKernelTypeM<Class, std::function API >, \
                        public detail::G_ID_HELPER_CLASS(Class)             \
// {body} is to be defined by user

namespace cv
{
// Declare <unite> in cv:: namespace
enum class unite_policy
{
    REPLACE,
    KEEP
};

namespace gapi
{
    // Prework: model "Device" API before it gets to G-API headers.
    // FIXME: Don't mix with internal Backends class!
    class GAPI_EXPORTS GBackend
    {
    public:
        class Priv;

        // TODO: make it template (call `new` within??)
        GBackend();
        explicit GBackend(std::shared_ptr<Priv> &&p);

        Priv& priv();
        const Priv& priv() const;
        std::size_t hash() const;

        bool operator== (const GBackend &rhs) const;

    private:
        std::shared_ptr<Priv> m_priv;
    };

    inline bool operator != (const GBackend &lhs, const GBackend &rhs)
    {
        return !(lhs == rhs);
    }
} // namespace gapi
} // namespace cv

namespace std
{
    template<> struct hash<cv::gapi::GBackend>
    {
        std::size_t operator() (const cv::gapi::GBackend &b) const
        {
            return b.hash();
        }
    };
} // namespace std


namespace cv {
namespace gapi {
    // Lookup order is in fact a vector of Backends to traverse during look-up
    using GLookupOrder = std::vector<GBackend>;
    inline GLookupOrder lookup_order(std::initializer_list<GBackend> &&list)
    {
        return GLookupOrder(std::move(list));
    }

    // FIXME: Hide implementation
    class GAPI_EXPORTS GKernelPackage
    {
        using S = std::unordered_map<std::string, GKernelImpl>;
        using M = std::unordered_map<GBackend, S>;
        M m_backend_kernels;

    protected:
        // Check if package contains ANY implementation of a kernel API
        // by API textual id.
        bool includesAPI(const std::string &id) const;

    public:
        // Return total number of kernels (accross all backends)
        std::size_t size() const;

        // Check if particular kernel implementation exist in the package.
        // The key word here is _particular_ - i.e., from the specific backend.
        template<typename KImpl>
        bool includes() const
        {
            const auto set_iter = m_backend_kernels.find(KImpl::backend());
            return (set_iter != m_backend_kernels.end())
                ? (set_iter->second.count(KImpl::API::id()) > 0)
                : false;
        }

        // Removes all the kernels related to the given backend
        void remove(const GBackend& backend);

        // Check if package contains ANY implementation of a kernel API
        // by API type.
        template<typename KAPI>
        bool includesAPI() const
        {
            return includesAPI(KAPI::id());
        }

        // Lookup a kernel, given the look-up order. Returns Backend which
        // hosts kernel implementation. Throws if nothing found.
        //
        // If order is empty(), returns first suitable implementation.
        template<typename KAPI>
        GBackend lookup(const GLookupOrder &order = {}) const
        {
            return lookup(KAPI::id(), order).first;
        }

        std::pair<cv::gapi::GBackend, cv::GKernelImpl>
        lookup(const std::string &id, const GLookupOrder &order = {}) const;

        // Put a new kernel implementation into package
        // FIXME: No overwrites allowed?
        template<typename KImpl> void include()
        {
            auto backend     = KImpl::backend();
            auto kernel_id   = KImpl::API::id();
            auto kernel_impl = GKernelImpl{KImpl::kernel()};
            m_backend_kernels[backend][kernel_id] = std::move(kernel_impl);
        }

        // Lists all backends which are included into package
        std::vector<GBackend> backends() const;

        friend GAPI_EXPORTS GKernelPackage combine(const GKernelPackage  &,
                                                 const GKernelPackage  &,
                                                 const cv::unite_policy);
    };

    template<typename... KK> GKernelPackage kernels()
    {
        GKernelPackage pkg;

        // For those who wonder - below is a trick to call a number of
        // methods based on parameter pack (zeroes just help hiding these
        // calls into a sequence which helps to expand this parameter pack).
        // Just note that `f(),a` always equals to `a` (with f() called!)
        // and parentheses are used to hide function call in the expanded sequence.
        // Leading 0 helps to handle case when KK is an empty list (kernels<>()).

        int unused[] = { 0, (pkg.include<KK>(), 0)... };
        cv::util::suppress_unused_warning(unused);
        return pkg;
    };

    // Return a new package based on `lhs` and `rhs`,
    // with unity policy defined by `policy`.
    GAPI_EXPORTS GKernelPackage combine(const GKernelPackage  &lhs,
                                      const GKernelPackage  &rhs,
                                      const cv::unite_policy policy);
} // namespace gapi

namespace detail
{
    template<> struct CompileArgTag<cv::gapi::GKernelPackage>
    {
        static const char* tag() { return "gapi.kernel_package"; }
    };
    template<> struct CompileArgTag<cv::gapi::GLookupOrder>
    {
        static const char* tag() { return "gapi.lookup_order"; }
    };
} // namespace detail
} // namespace cv

#endif // OPENCV_GAPI_GKERNEL_HPP
