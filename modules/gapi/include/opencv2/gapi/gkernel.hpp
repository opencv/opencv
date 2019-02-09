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
    /** \addtogroup gapi_compile_args
     * @{
     */

    // Lookup order is in fact a vector of Backends to traverse during look-up
    /**
     * @brief Priority list of backends to use during kernel
     *   resolution process.
     *
     * Priority is descending -- the first backend in the list has the
     * top priority, and the last one has the lowest priority.
     *
     * If there's multiple implementations available for a kernel at
     * the moment of graph compilation, a kernel (and thus a backend)
     * will be selected according to this order (if the parameter is passed).
     *
     * Default order is not specified (and by default, only
     * CPU(OpenCV) backend is involved in graph compilation).
     */
    using GLookupOrder = std::vector<GBackend>;
    /**
     * @brief Create a backend lookup order -- priority list of
     * backends to use during graph compilation process.
     *
     * @sa GLookupOrder, @ref gapi_std_backends
     */
    inline GLookupOrder lookup_order(std::initializer_list<GBackend> &&list)
    {
        return GLookupOrder(std::move(list));
    }

    // FIXME: Hide implementation
    /**
     * @brief A container class for heterogeneous kernel
     * implementation collections.
     *
     * GKernelPackage is a special container class which stores kernel
     * _implementations_. Objects of this class are created and passed
     * to cv::GComputation::compile() to specify which kernels to use
     * in the compiled graph. GKernelPackage may contain kernels of
     * different backends, e.g. be heterogeneous.
     *
     * The most easy way to create a kernel package is to use function
     * cv::gapi::kernels(). This template functions takes kernel
     * implementations in form of type list (variadic template) and
     * generates a kernel package atop of that.
     *
     * Kernel packages can be also generated programatically, starting
     * with an empty package (created with the default constructor)
     * and then by populating it with kernels via call to
     * GKernelPackage::include(). Note this method is also a template
     * one since G-API kernel implementations are _types_, not objects.
     *
     * Finally, two kernel packages can be combined into a new one
     * with function cv::gapi::combine(). There are different rules
     * apply to this process, see also cv::gapi::unite_policy for
     * details.
     */
    class GAPI_EXPORTS GKernelPackage
    {
        /// @private
        using S = std::unordered_map<std::string, GKernelImpl>;

        /// @private
        using M = std::unordered_map<GBackend, S>;

        /// @private
        M m_backend_kernels;

    protected:
        /// @private
        // Check if package contains ANY implementation of a kernel API
        // by API textual id.
        bool includesAPI(const std::string &id) const;

        /// @private
        // Remove ALL implementations of the given API (identified by ID)
        void removeAPI(const std::string &id);

    public:
        /**
         * @brief Returns total number of kernels in the package
         * (across all backends included)
         *
         * @return a number of kernels in the package
         */
        std::size_t size() const;

        /**
         * @brief Test if a particular kernel _implementation_ KImpl is
         * included in this kernel package.
         *
         * @sa includesAPI()
         *
         * @return true if there is such kernel, false otherwise.
         */
        template<typename KImpl>
        bool includes() const
        {
            const auto set_iter = m_backend_kernels.find(KImpl::backend());
            return (set_iter != m_backend_kernels.end())
                ? (set_iter->second.count(KImpl::API::id()) > 0)
                : false;
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

        /**
         * @brief Find a kernel (by its API), given the look-up order.
         *
         * If order is empty, returns first suitable implementation.
         * Throws if nothing found.
         *
         * @return Backend which hosts matching kernel implementation.
         *
         * @sa cv::gapi::lookup_order
         */
        template<typename KAPI>
        GBackend lookup(const GLookupOrder &order = {}) const
        {
            return lookup(KAPI::id(), order).first;
        }

        /// @private
        std::pair<cv::gapi::GBackend, cv::GKernelImpl>
        lookup(const std::string &id, const GLookupOrder &order = {}) const;

        // FIXME: No overwrites allowed?
        /**
         * @brief Put a new kernel implementation KImpl into package.
         *
         * @param up unite policy to use. If the package has already
         * implementation for this kernel (probably from another
         * backend), and cv::unite_policy::KEEP is passed, the
         * existing implementation remains in package; on
         * cv::unite_policy::REPLACE all other existing
         * implementations are first dropped from the package.
         */
        template<typename KImpl>
        void include(const cv::unite_policy up = cv::unite_policy::KEEP)
        {
            auto backend     = KImpl::backend();
            auto kernel_id   = KImpl::API::id();
            auto kernel_impl = GKernelImpl{KImpl::kernel()};
            if (up == cv::unite_policy::REPLACE) removeAPI(kernel_id);
            else GAPI_Assert(up == cv::unite_policy::KEEP);

            // Regardless of the policy, store new impl in its storage slot.
            m_backend_kernels[backend][kernel_id] = std::move(kernel_impl);
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
         * @brief Create a new package based on `lhs` and `rhs`,
         * with unity policy defined by `policy`.
         *
         * @param lhs "Left-hand-side" package in the process
         * @param rhs "Right-hand-side" package in the process
         * @param policy Unite policy which is used in case of conflicts
         * -- when the same kernel API is implemented in both packages by
         * different backends; cv::unite_policy::KEEP keeps both
         * implementation in the resulting package, while
         * cv::unite_policy::REPLACE gives precedence two kernels from
         * "Right-hand-side".
         *
         * @return a new kernel package.
         */
        friend GAPI_EXPORTS GKernelPackage combine(const GKernelPackage  &lhs,
                                                   const GKernelPackage  &rhs,
                                                   const cv::unite_policy policy);
    };

    /**
     * @brief Create a kernel package object containing kernels
     * specified in variadic template argument.
     *
     * In G-API, kernel implementations are _types_. Every backend has
     * its own kernel API (like GAPI_OCV_KERNEL() and
     * GAPI_FLUID_KERNEL()) but all of that APIs define a new type for
     * each kernel implementation.
     *
     * Use this function to pass kernel implementations (defined in
     * either way) to the system. Example:
     *
     * @snippet modules/gapi/samples/api_ref_snippets.cpp kernels_snippet
     *
     * Note that kernels() itself is a function returning object, not
     * a type, so having `()` at the end is important -- it must be a
     * function call.
     */
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

    /** @} */

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
