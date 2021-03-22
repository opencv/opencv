// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_GKERNEL_HPP
#define OPENCV_GAPI_GKERNEL_HPP

#include <functional>
#include <iostream>
#include <string>  // string
#include <type_traits> // false_type, true_type
#include <unordered_map> // map (for GKernelPackage)
#include <utility> // tuple

#include <opencv2/gapi/gcommon.hpp> // CompileArgTag
#include <opencv2/gapi/util/util.hpp> // Seq
#include <opencv2/gapi/gcall.hpp>
#include <opencv2/gapi/garg.hpp>      // GArg
#include <opencv2/gapi/gmetaarg.hpp>  // GMetaArg
#include <opencv2/gapi/gtype_traits.hpp> // GTypeTraits
#include <opencv2/gapi/util/compiler_hints.hpp> //suppress_unused_warning
#include <opencv2/gapi/gtransform.hpp>

namespace cv {

struct GTypeInfo
{
    GShape                 shape;
    cv::detail::OpaqueKind kind;
    detail::HostCtor       ctor;
};

using GShapes    = std::vector<GShape>;
using GKinds     = std::vector<cv::detail::OpaqueKind>;
using GCtors     = std::vector<detail::HostCtor>;
using GTypesInfo = std::vector<GTypeInfo>;

// GKernel describes kernel API to the system
// FIXME: add attributes of a kernel, (e.g. number and types
// of inputs, etc)
struct GAPI_EXPORTS GKernel
{
    using M = std::function<GMetaArgs(const GMetaArgs &, const GArgs &)>;

    std::string name;       // kernel ID, defined by its API (signature)
    std::string tag;        // some (implementation-specific) tag
    M           outMeta;    // generic adaptor to API::outMeta(...)
    GShapes     outShapes;  // types (shapes) kernel's outputs
    GKinds      inKinds;    // kinds of kernel's inputs (fixme: below)
    GCtors      outCtors;   // captured constructors for template output types
};
// TODO: It's questionable if inKinds should really be here. Instead,
// this information could come from meta.

// GKernelImpl describes particular kernel implementation to the system
struct GAPI_EXPORTS GKernelImpl
{
    util::any         opaque;    // backend-specific opaque info
    GKernel::M        outMeta;   // for deserialized graphs, the outMeta is taken here
};

template<typename, typename> class GKernelTypeM;

namespace detail
{
    ////////////////////////////////////////////////////////////////////////////
    // yield() is used in graph construction time as a generic method to obtain
    // lazy "return value" of G-API operations
    //
    template<typename T> struct Yield;
    template<> struct Yield<cv::GMat>
    {
        static inline cv::GMat yield(cv::GCall &call, int i) { return call.yield(i); }
    };
    template<> struct Yield<cv::GMatP>
    {
        static inline cv::GMatP yield(cv::GCall &call, int i) { return call.yieldP(i); }
    };
    template<> struct Yield<cv::GScalar>
    {
        static inline cv::GScalar yield(cv::GCall &call, int i) { return call.yieldScalar(i); }
    };
    template<typename U> struct Yield<cv::GArray<U> >
    {
        static inline cv::GArray<U> yield(cv::GCall &call, int i) { return call.yieldArray<U>(i); }
    };
    template<typename U> struct Yield<cv::GOpaque<U> >
    {
        static inline cv::GOpaque<U> yield(cv::GCall &call, int i) { return call.yieldOpaque<U>(i); }
    };
    template<> struct Yield<GFrame>
    {
        static inline cv::GFrame yield(cv::GCall &call, int i) { return call.yieldFrame(i); }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Helper classes which brings outputMeta() marshalling to kernel
    // implementations
    //
    // 1. MetaType establishes G#Type -> G#Meta mapping between G-API dynamic
    //    types and its metadata descriptor types.
    //    This mapping is used to transform types to call outMeta() callback.
    template<typename T> struct MetaType;
    template<> struct MetaType<cv::GMat>    { using type = GMatDesc; };
    template<> struct MetaType<cv::GMatP>   { using type = GMatDesc; };
    template<> struct MetaType<cv::GFrame>  { using type = GFrameDesc; };
    template<> struct MetaType<cv::GScalar> { using type = GScalarDesc; };
    template<typename U> struct MetaType<cv::GArray<U> >  { using type = GArrayDesc; };
    template<typename U> struct MetaType<cv::GOpaque<U> > { using type = GOpaqueDesc; };
    template<typename T> struct MetaType    { using type = T; }; // opaque args passed as-is
    // FIXME: Move it to type traits?

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

    ////////////////////////////////////////////////////////////////////////////
    // Helper class to introduce tags to calls. By default there's no tag
    struct NoTag {
        static constexpr const char *tag() { return ""; }
    };

} // namespace detail

// GKernelType and GKernelTypeM are base classes which implement typed ::on()
// method based on kernel signature. GKernelTypeM stands for multiple-return-value kernels
//
// G_TYPED_KERNEL and G_TYPED_KERNEL_M macros inherit user classes from GKernelType and
// GKernelTypeM respectively.

template<typename K, typename... R, typename... Args>
class GKernelTypeM<K, std::function<std::tuple<R...>(Args...)> >
    : public detail::MetaHelper<K, std::tuple<Args...>, std::tuple<R...>>
    , public detail::NoTag
{
    template<int... IIs>
    static std::tuple<R...> yield(cv::GCall &call, detail::Seq<IIs...>)
    {
        return std::make_tuple(detail::Yield<R>::yield(call, IIs)...);
    }

public:
    using InArgs  = std::tuple<Args...>;
    using OutArgs = std::tuple<R...>;

    // TODO: Args&&... here?
    static std::tuple<R...> on(Args... args)
    {
        cv::GCall call(GKernel{ K::id()
                              , K::tag()
                              , &K::getOutMeta
                              , {detail::GTypeTraits<R>::shape...}
                              , {detail::GTypeTraits<Args>::op_kind...}
                              , {detail::GObtainCtor<R>::get()...}});
        call.pass(args...); // TODO: std::forward() here?
        return yield(call, typename detail::MkSeq<sizeof...(R)>::type());
    }
};

template<typename, typename> class GKernelType;

template<typename K, typename R, typename... Args>
class GKernelType<K, std::function<R(Args...)> >
    : public detail::MetaHelper<K, std::tuple<Args...>, R>
    , public detail::NoTag
{
public:
    using InArgs  = std::tuple<Args...>;
    using OutArgs = std::tuple<R>;

    static R on(Args... args)
    {
        cv::GCall call(GKernel{ K::id()
                              , K::tag()
                              , &K::getOutMeta
                              , {detail::GTypeTraits<R>::shape}
                              , {detail::GTypeTraits<Args>::op_kind...}
                              , {detail::GObtainCtor<R>::get()}});
        call.pass(args...);
        return detail::Yield<R>::yield(call, 0);
    }
};

namespace detail {
// This tiny class eliminates the semantic difference between
// GKernelType and GKernelTypeM.
template<typename, typename> class KernelTypeMedium;

template<typename K, typename... R, typename... Args>
class KernelTypeMedium<K, std::function<std::tuple<R...>(Args...)>> :
    public cv::GKernelTypeM<K, std::function<std::tuple<R...>(Args...)>> {};

template<typename K, typename R, typename... Args>
class KernelTypeMedium<K, std::function<R(Args...)>> :
    public cv::GKernelType<K, std::function<R(Args...)>> {};
} // namespace detail

} // namespace cv


// FIXME: I don't know a better way so far. Feel free to suggest one
// The problem is that every typed kernel should have ::id() but body
// of the class is defined by user (with outMeta, other stuff)

//! @cond IGNORED
#define G_ID_HELPER_CLASS(Class)  Class##IdHelper

#define G_ID_HELPER_BODY(Class, Id)                                         \
    struct G_ID_HELPER_CLASS(Class)                                         \
    {                                                                       \
        static constexpr const char * id() {return Id;}                     \
    };                                                                      \
//! @endcond

#define GET_G_TYPED_KERNEL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, NAME, ...) NAME
#define COMBINE_SIGNATURE(...) __VA_ARGS__
// Ensure correct __VA_ARGS__ expansion on Windows
#define __WRAP_VAARGS(x) x

/**
 * Helper for G_TYPED_KERNEL declares a new G-API Operation. See [Kernel API](@ref gapi_kernel_api)
 * for more details.
 *
 * @param Class type name for this operation.
 * @param API an `std::function<>`-like signature for the operation;
 *        return type is a single value or a tuple of multiple values.
 * @param Id string identifier for the operation. Must be unique.
 */
#define G_TYPED_KERNEL_HELPER(Class, API, Id)                                               \
    G_ID_HELPER_BODY(Class, Id)                                                             \
    struct Class final: public cv::detail::KernelTypeMedium<Class, std::function API >,     \
                        public G_ID_HELPER_CLASS(Class)
// {body} is to be defined by user

#define G_TYPED_KERNEL_HELPER_2(Class, _1, _2, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2), Id)

#define G_TYPED_KERNEL_HELPER_3(Class, _1, _2, _3, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3), Id)

#define G_TYPED_KERNEL_HELPER_4(Class, _1, _2, _3, _4, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3, _4), Id)

#define G_TYPED_KERNEL_HELPER_5(Class, _1, _2, _3, _4, _5, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3, _4, _5), Id)

#define G_TYPED_KERNEL_HELPER_6(Class, _1, _2, _3, _4, _5, _6, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3, _4, _5, _6), Id)

#define G_TYPED_KERNEL_HELPER_7(Class, _1, _2, _3, _4, _5, _6, _7, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3, _4, _5, _6, _7), Id)

#define G_TYPED_KERNEL_HELPER_8(Class, _1, _2, _3, _4, _5, _6, _7, _8, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3, _4, _5, _6, _7, _8), Id)

#define G_TYPED_KERNEL_HELPER_9(Class, _1, _2, _3, _4, _5, _6, _7, _8, _9, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3, _4, _5, _6, _7, _8, _9), Id)

#define G_TYPED_KERNEL_HELPER_10(Class, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, Id) \
G_TYPED_KERNEL_HELPER(Class, COMBINE_SIGNATURE(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10), Id)

/**
 * Declares a new G-API Operation. See [Kernel API](@ref gapi_kernel_api)
 * for more details.
 *
 * @param Class type name for this operation.
 */
#define G_TYPED_KERNEL(Class, ...) __WRAP_VAARGS(GET_G_TYPED_KERNEL(__VA_ARGS__, \
                                                 G_TYPED_KERNEL_HELPER_10, \
                                                 G_TYPED_KERNEL_HELPER_9, \
                                                 G_TYPED_KERNEL_HELPER_8, \
                                                 G_TYPED_KERNEL_HELPER_7, \
                                                 G_TYPED_KERNEL_HELPER_6, \
                                                 G_TYPED_KERNEL_HELPER_5, \
                                                 G_TYPED_KERNEL_HELPER_4, \
                                                 G_TYPED_KERNEL_HELPER_3, \
                                                 G_TYPED_KERNEL_HELPER_2, \
                                                 G_TYPED_KERNEL_HELPER)(Class, __VA_ARGS__)) \

/**
 * Declares a new G-API Operation. See [Kernel API](@ref gapi_kernel_api) for more details.
 *
 * @deprecated This macro is deprecated in favor of `G_TYPED_KERNEL` that is used for declaring any
 * G-API Operation.
 *
 * @param Class type name for this operation.
 */
#define G_TYPED_KERNEL_M G_TYPED_KERNEL

#define G_API_OP   G_TYPED_KERNEL
#define G_API_OP_M G_API_OP

namespace cv
{
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
    class GFunctor
    {
    public:
        virtual cv::GKernelImpl impl()       const = 0;
        virtual cv::gapi::GBackend backend() const = 0;
        const char* id()                     const { return m_id; }

        virtual ~GFunctor() = default;
    protected:
        GFunctor(const char* id) : m_id(id) { };
    private:
        const char* m_id;
    };

    /** \addtogroup gapi_compile_args
     * @{
     */

    // FIXME: Hide implementation
    /**
     * @brief A container class for heterogeneous kernel
     * implementation collections and graph transformations.
     *
     * GKernelPackage is a special container class which stores kernel
     * _implementations_ and graph _transformations_. Objects of this class
     * are created and passed to cv::GComputation::compile() to specify
     * which kernels to use and which transformations to apply in the
     * compiled graph. GKernelPackage may contain kernels of
     * different backends, e.g. be heterogeneous.
     *
     * The most easy way to create a kernel package is to use function
     * cv::gapi::kernels(). This template functions takes kernel
     * implementations in form of type list (variadic template) and
     * generates a kernel package atop of that.
     *
     * Kernel packages can be also generated programmatically, starting
     * with an empty package (created with the default constructor)
     * and then by populating it with kernels via call to
     * GKernelPackage::include(). Note this method is also a template
     * one since G-API kernel and transformation implementations are _types_,
     * not objects.
     *
     * Finally, two kernel packages can be combined into a new one
     * with function cv::gapi::combine().
     */
    class GAPI_EXPORTS_W_SIMPLE GKernelPackage
    {

        /// @private
        using M = std::unordered_map<std::string, std::pair<GBackend, GKernelImpl>>;

        /// @private
        M m_id_kernels;

        /// @private
        std::vector<GTransform> m_transformations;

    protected:
        /// @private
        // Remove ALL implementations of the given API (identified by ID)
        void removeAPI(const std::string &id);

        /// @private
        // Partial include() specialization for kernels
        template <typename KImpl>
        typename std::enable_if<(std::is_base_of<cv::detail::KernelTag, KImpl>::value), void>::type
        includeHelper()
        {
            auto backend     = KImpl::backend();
            auto kernel_id   = KImpl::API::id();
            auto kernel_impl = GKernelImpl{KImpl::kernel(), &KImpl::API::getOutMeta};
            removeAPI(kernel_id);

            m_id_kernels[kernel_id] = std::make_pair(backend, kernel_impl);
        }

        /// @private
        // Partial include() specialization for transformations
        template <typename TImpl>
        typename std::enable_if<(std::is_base_of<cv::detail::TransformTag, TImpl>::value), void>::type
        includeHelper()
        {
            m_transformations.emplace_back(TImpl::transformation());
        }

    public:
        void include(const GFunctor& functor)
        {
            m_id_kernels[functor.id()] = std::make_pair(functor.backend(), functor.impl());
        }
        /**
         * @brief Returns total number of kernels
         * in the package (across all backends included)
         *
         * @return a number of kernels in the package
         */
        std::size_t size() const;

        /**
         * @brief Returns vector of transformations included in the package
         *
         * @return vector of transformations included in the package
         */
        const std::vector<GTransform>& get_transformations() const;

        /**
         * @brief Returns vector of kernel ids included in the package
         *
         * @return vector of kernel ids included in the package
         */
        std::vector<std::string> get_kernel_ids() const;

        /**
         * @brief Test if a particular kernel _implementation_ KImpl is
         * included in this kernel package.
         *
         * @sa includesAPI()
         *
         * @note cannot be applied to transformations
         *
         * @return true if there is such kernel, false otherwise.
         */
        template<typename KImpl>
        bool includes() const
        {
            static_assert(std::is_base_of<cv::detail::KernelTag, KImpl>::value,
                          "includes() can be applied to kernels only");

            auto kernel_it = m_id_kernels.find(KImpl::API::id());
            return kernel_it != m_id_kernels.end() &&
                   kernel_it->second.first == KImpl::backend();
        }

        /**
         * @brief Remove all kernels associated with the given backend
         * from the package.
         *
         * Does nothing if there's no kernels of this backend in the package.
         *
         * @param backend backend which kernels to remove
         */
        void remove(const GBackend& backend);

        /**
         * @brief Remove all kernels implementing the given API from
         * the package.
         *
         * Does nothing if there's no kernels implementing the given interface.
         */
        template<typename KAPI>
        void remove()
        {
            removeAPI(KAPI::id());
        }

        // FIXME: Rename to includes() and distinguish API/impl case by
        //     statically?
        /**
         * Check if package contains ANY implementation of a kernel API
         * by API type.
         */
        template<typename KAPI>
        bool includesAPI() const
        {
            return includesAPI(KAPI::id());
        }

        /// @private
        bool includesAPI(const std::string &id) const;

        // FIXME: The below comment is wrong, and who needs this function?
        /**
         * @brief Find a kernel (by its API)
         *
         * Returns implementation corresponding id.
         * Throws if nothing found.
         *
         * @return Backend which hosts matching kernel implementation.
         *
         */
        template<typename KAPI>
        GBackend lookup() const
        {
            return lookup(KAPI::id()).first;
        }

        /// @private
        std::pair<cv::gapi::GBackend, cv::GKernelImpl>
        lookup(const std::string &id) const;

        // FIXME: No overwrites allowed?
        /**
         * @brief Put a new kernel implementation or a new transformation
         * KImpl into the package.
         */
        template<typename KImpl>
        void include()
        {
            includeHelper<KImpl>();
        }

        /**
         * @brief Adds a new kernel based on it's backend and id into the kernel package
         *
         * @param backend backend associated with the kernel
         * @param kernel_id a name/id of the kernel
         */
        void include(const cv::gapi::GBackend& backend, const std::string& kernel_id)
        {
            removeAPI(kernel_id);
            m_id_kernels[kernel_id] = std::make_pair(backend, GKernelImpl{{}, {}});
        }

        /**
         * @brief Lists all backends which are included into package
         *
         * @return vector of backends
         */
        std::vector<GBackend> backends() const;

        // TODO: Doxygen bug -- it wants me to place this comment
        // here, not below.
        /**
         * @brief Create a new package based on `lhs` and `rhs`.
         *
         * @param lhs "Left-hand-side" package in the process
         * @param rhs "Right-hand-side" package in the process
         * @return a new kernel package.
         */
        friend GAPI_EXPORTS GKernelPackage combine(const GKernelPackage  &lhs,
                                                   const GKernelPackage  &rhs);
    };

    /**
     * @brief Create a kernel package object containing kernels
     * and transformations specified in variadic template argument.
     *
     * In G-API, kernel implementations and transformations are _types_.
     * Every backend has its own kernel API (like GAPI_OCV_KERNEL() and
     * GAPI_FLUID_KERNEL()) but all of that APIs define a new type for
     * each kernel implementation.
     *
     * Use this function to pass kernel implementations (defined in
     * either way) and transformations to the system. Example:
     *
     * @snippet modules/gapi/samples/api_ref_snippets.cpp kernels_snippet
     *
     * Note that kernels() itself is a function returning object, not
     * a type, so having `()` at the end is important -- it must be a
     * function call.
     */
    template<typename... KK> GKernelPackage kernels()
    {
        // FIXME: currently there is no check that transformations' signatures are unique
        // and won't be any intersection in graph compilation stage
        static_assert(cv::detail::all_unique<typename KK::API...>::value, "Kernels API must be unique");

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

    template<typename... FF>
    GKernelPackage kernels(FF&... functors)
    {
        GKernelPackage pkg;
        int unused[] = { 0, (pkg.include(functors), 0)... };
        cv::util::suppress_unused_warning(unused);
        return pkg;
    };

    /** @} */

    // FYI - this function is already commented above
    GAPI_EXPORTS GKernelPackage combine(const GKernelPackage  &lhs,
                                        const GKernelPackage  &rhs);

    /**
     * @brief Combines multiple G-API kernel packages into one
     *
     * @overload
     *
     * This function successively combines the passed kernel packages using a right fold.
     * Calling `combine(a, b, c)` is equal to `combine(a, combine(b, c))`.
     *
     * @return The resulting kernel package
     */
    template<typename... Ps>
    GKernelPackage combine(const GKernelPackage &a, const GKernelPackage &b, Ps&&... rest)
    {
        return combine(a, combine(b, rest...));
    }

    /** \addtogroup gapi_compile_args
     * @{
     */
    /**
     * @brief cv::use_only() is a special combinator which hints G-API to use only
     * kernels specified in cv::GComputation::compile() (and not to extend kernels available by
     * default with that package).
     */
    struct GAPI_EXPORTS use_only
    {
        GKernelPackage pkg;
    };
    /** @} */

} // namespace gapi

namespace detail
{
    template<> struct CompileArgTag<cv::gapi::GKernelPackage>
    {
        static const char* tag() { return "gapi.kernel_package"; }
    };

    template<> struct CompileArgTag<cv::gapi::use_only>
    {
        static const char* tag() { return "gapi.use_only"; }
    };
} // namespace detail

} // namespace cv

#endif // OPENCV_GAPI_GKERNEL_HPP
