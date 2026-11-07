// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2022 Intel Corporation


#ifndef OPENCV_GAPI_GCPUKERNEL_HPP
#define OPENCV_GAPI_GCPUKERNEL_HPP

#if defined _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4702)  // "Unreachable code" on postprocess(...) call inside OCVCallHelper
#endif

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gmetaarg.hpp>
#include <opencv2/gapi/util/compiler_hints.hpp> //suppress_unused_warning
#include <opencv2/gapi/util/util.hpp>

// FIXME: namespace scheme for backends?
namespace cv {

namespace gimpl
{
    // Forward-declare an internal class
    class GCPUExecutable;
} // namespace gimpl

namespace gapi
{
/**
 * @brief This namespace contains G-API CPU backend functions,
 * structures, and symbols.
 */
namespace cpu
{
    /**
     * \addtogroup gapi_std_backends
     * @{
     *
     * @brief G-API backends available in this OpenCV version
     *
     * G-API backends play a corner stone role in G-API execution
     * stack. Every backend is hardware-oriented and thus can run its
     * kernels efficiently on the target platform.
     *
     * Backends are usually "black boxes" for G-API users -- on the API
     * side, all backends are represented as different objects of the
     * same class cv::gapi::GBackend.
     * User can manipulate with backends by specifying which kernels to use.
     *
     * @sa @ref gapi_hld
     */

    /**
     * @brief Get a reference to CPU (OpenCV) backend.
     *
     * This is the default backend in G-API at the moment, providing
     * broader functional coverage but losing some graph model
     * advantages. Provided mostly for reference and prototyping
     * purposes.
     *
     * @sa gapi_std_backends
     */
    GAPI_EXPORTS cv::gapi::GBackend backend();
    /** @} */

    class GOCVFunctor;

    //! @cond IGNORED
    template<typename K, typename Callable>
    GOCVFunctor ocv_kernel(const Callable& c);

    template<typename K, typename Callable>
    GOCVFunctor ocv_kernel(Callable& c);
    //! @endcond

} // namespace cpu
} // namespace gapi

// Represents arguments which are passed to a wrapped CPU function
// FIXME: put into detail?
class GAPI_EXPORTS GCPUContext
{
public:
    // Generic accessor API
    template<typename T>
    const T& inArg(int input) { return m_args.at(input).get<T>(); }

    // Syntax sugar
    const cv::Mat&   inMat(int input);
    cv::Mat&         outMatR(int output); // FIXME: Avoid cv::Mat m = ctx.outMatR()

    const cv::Scalar& inVal(int input);
    cv::Scalar& outValR(int output); // FIXME: Avoid cv::Scalar s = ctx.outValR()
    cv::MediaFrame& outFrame(int output);
    template<typename T> std::vector<T>& outVecR(int output) // FIXME: the same issue
    {
        return outVecRef(output).wref<T>();
    }
    template<typename T> T& outOpaqueR(int output) // FIXME: the same issue
    {
        return outOpaqueRef(output).wref<T>();
    }

    GArg state()
    {
        return m_state;
    }

protected:
    detail::VectorRef& outVecRef(int output);
    detail::OpaqueRef& outOpaqueRef(int output);

    std::vector<GArg> m_args;
    GArg m_state;

    //FIXME: avoid conversion of arguments from internal representation to OpenCV one on each call
    //to OCV kernel. (This can be achieved by a two single time conversions in GCPUExecutable::run,
    //once on enter for input and output arguments, and once before return for output arguments only
    std::unordered_map<std::size_t, GRunArgP> m_results;

    friend class gimpl::GCPUExecutable;
};

class GAPI_EXPORTS GCPUKernel
{
public:
    // This function is a kernel's execution entry point (does the processing work)
    using RunF = std::function<void(GCPUContext &)>;
    // This function is a stateful kernel's setup routine (configures state)
    using SetupF = std::function<void(const GMetaArgs &, const GArgs &,
                                      GArg &, const GCompileArgs &)>;

    GCPUKernel();
    GCPUKernel(const RunF& runF, const SetupF& setupF = nullptr);

    RunF m_runF = nullptr;
    SetupF m_setupF = nullptr;

    bool m_isStateful = false;
};

// FIXME: This is an ugly ad-hoc implementation. TODO: refactor

namespace detail
{
template<class T> struct get_in;
template<> struct get_in<cv::GMat>
{
    static cv::Mat    get(GCPUContext &ctx, int idx) { return ctx.inMat(idx); }
};
template<> struct get_in<cv::GMatP>
{
    static cv::Mat    get(GCPUContext &ctx, int idx) { return get_in<cv::GMat>::get(ctx, idx); }
};
template<> struct get_in<cv::GFrame>
{
    static cv::MediaFrame get(GCPUContext &ctx, int idx) { return ctx.inArg<cv::MediaFrame>(idx); }
};
template<> struct get_in<cv::GScalar>
{
    static cv::Scalar get(GCPUContext &ctx, int idx) { return ctx.inVal(idx); }
};
template<typename U> struct get_in<cv::GArray<U> >
{
    static const std::vector<U>& get(GCPUContext &ctx, int idx) { return ctx.inArg<VectorRef>(idx).rref<U>(); }
};
template<typename U> struct get_in<cv::GOpaque<U> >
{
    static const U& get(GCPUContext &ctx, int idx) { return ctx.inArg<OpaqueRef>(idx).rref<U>(); }
};

//FIXME(dm): GArray<Mat>/GArray<GMat> conversion should be done more gracefully in the system
template<> struct get_in<cv::GArray<cv::GMat> >: public get_in<cv::GArray<cv::Mat> >
{
};

//FIXME(dm): GArray<Scalar>/GArray<GScalar> conversion should be done more gracefully in the system
template<> struct get_in<cv::GArray<cv::GScalar> >: public get_in<cv::GArray<cv::Scalar> >
{
};

// FIXME(dm): GArray<vector<U>>/GArray<GArray<U>> conversion should be done more gracefully in the system
template<typename U> struct get_in<cv::GArray<cv::GArray<U>> >: public get_in<cv::GArray<std::vector<U>> >
{
};

//FIXME(dm): GOpaque<Mat>/GOpaque<GMat> conversion should be done more gracefully in the system
template<> struct get_in<cv::GOpaque<cv::GMat> >: public get_in<cv::GOpaque<cv::Mat> >
{
};

//FIXME(dm): GOpaque<Scalar>/GOpaque<GScalar> conversion should be done more gracefully in the system
template<> struct get_in<cv::GOpaque<cv::GScalar> >: public get_in<cv::GOpaque<cv::Mat> >
{
};

template<class T> struct get_in
{
    static T get(GCPUContext &ctx, int idx) { return ctx.inArg<T>(idx); }
};

struct tracked_cv_mat{
    tracked_cv_mat(cv::Mat& m) : r{m}, original_data{m.data} {}
    cv::Mat r;
    uchar* original_data;

    operator cv::Mat& (){ return r;}
    void validate() const{
        if (r.data != original_data)
        {
            util::throw_error
                (std::logic_error
                 ("OpenCV kernel output parameter was reallocated. \n"
                  "Incorrect meta data was provided ?"));
        }
    }
};

template<typename... Outputs>
void postprocess(Outputs&... outs)
{
    struct
    {
        void operator()(tracked_cv_mat* bm) { bm->validate();  }
        void operator()(...)                {                  }

    } validate;
    //dummy array to unfold parameter pack
    int dummy[] = { 0, (validate(&outs), 0)... };
    cv::util::suppress_unused_warning(dummy);
}

template<class T> struct get_out;
template<> struct get_out<cv::GMat>
{
    static tracked_cv_mat get(GCPUContext &ctx, int idx)
    {
        auto& r = ctx.outMatR(idx);
        return {r};
    }
};
template<> struct get_out<cv::GMatP>
{
    static tracked_cv_mat get(GCPUContext &ctx, int idx)
    {
        return get_out<cv::GMat>::get(ctx, idx);
    }
};
template<> struct get_out<cv::GScalar>
{
    static cv::Scalar& get(GCPUContext &ctx, int idx)
    {
        return ctx.outValR(idx);
    }
};
template<> struct get_out<cv::GFrame>
{
    static cv::MediaFrame& get(GCPUContext &ctx, int idx)
    {
        return ctx.outFrame(idx);
    }
};
template<typename U> struct get_out<cv::GArray<U>>
{
    static std::vector<U>& get(GCPUContext &ctx, int idx)
    {
        return ctx.outVecR<U>(idx);
    }
};

//FIXME(dm): GArray<Mat>/GArray<GMat> conversion should be done more gracefully in the system
template<> struct get_out<cv::GArray<cv::GMat> >: public get_out<cv::GArray<cv::Mat> >
{
};

// FIXME(dm): GArray<vector<U>>/GArray<GArray<U>> conversion should be done more gracefully in the system
template<typename U> struct get_out<cv::GArray<cv::GArray<U>> >: public get_out<cv::GArray<std::vector<U>> >
{
};

template<typename U> struct get_out<cv::GOpaque<U>>
{
    static U& get(GCPUContext &ctx, int idx)
    {
        return ctx.outOpaqueR<U>(idx);
    }
};

template<typename, typename>
struct OCVSetupHelper;

template<typename Impl, typename... Ins>
struct OCVSetupHelper<Impl, std::tuple<Ins...>>
{
    // Using 'auto' return type and 'decltype' specifier in both 'setup_impl' versions
    // to check existence of required 'Impl::setup' functions.
    // While 'decltype' specifier accepts expression we pass expression with 'comma-operator'
    // where first operand of comma-operator is call attempt to desired 'Impl::setup' and
    // the second operand is 'void()' expression.
    //
    // SFINAE for 'Impl::setup' which accepts compile arguments.
    template<int... IIs>
    static auto setup_impl(const GMetaArgs &metaArgs, const GArgs &args,
                           GArg &state, const GCompileArgs &compileArgs,
                           detail::Seq<IIs...>) ->
        decltype(Impl::setup(detail::get_in_meta<Ins>(metaArgs, args, IIs)...,
                             std::declval<typename std::add_lvalue_reference<
                                              std::shared_ptr<typename Impl::State>
                                                                            >::type
                                         >(),
                            compileArgs)
                 , void())
    {
        // TODO: unique_ptr <-> shared_ptr conversion ?
        // To check: Conversion is possible only if the state which should be passed to
        // 'setup' user callback isn't required to have previous value
        std::shared_ptr<typename Impl::State> stPtr;
        Impl::setup(detail::get_in_meta<Ins>(metaArgs, args, IIs)..., stPtr, compileArgs);
        state = GArg(stPtr);
    }

    // SFINAE for 'Impl::setup' which doesn't accept compile arguments.
    template<int... IIs>
    static auto setup_impl(const GMetaArgs &metaArgs, const GArgs &args,
                           GArg &state, const GCompileArgs &/* compileArgs */,
                           detail::Seq<IIs...>) ->
        decltype(Impl::setup(detail::get_in_meta<Ins>(metaArgs, args, IIs)...,
                             std::declval<typename std::add_lvalue_reference<
                                              std::shared_ptr<typename Impl::State>
                                                                            >::type
                                         >()
                            )
                 , void())
    {
        // The same comment as in 'setup' above.
        std::shared_ptr<typename Impl::State> stPtr;
        Impl::setup(detail::get_in_meta<Ins>(metaArgs, args, IIs)..., stPtr);
        state = GArg(stPtr);
    }

    static void setup(const GMetaArgs &metaArgs, const GArgs &args,
                      GArg& state, const GCompileArgs &compileArgs)
    {
        setup_impl(metaArgs, args, state, compileArgs,
                   typename detail::MkSeq<sizeof...(Ins)>::type());
    }
};

// OCVCallHelper is a helper class to call stateless OCV kernels and OCV kernel functors.
template<typename, typename, typename>
struct OCVCallHelper;

// FIXME: probably can be simplified with std::apply or analogue.
template<typename Impl, typename... Ins, typename... Outs>
struct OCVCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...>>
{
    template<typename... Inputs>
    struct call_and_postprocess
    {
        template<typename... Outputs>
        static void call(Inputs&&... ins, Outputs&&... outs)
        {
            //not using a std::forward on outs is deliberate in order to
            //cause compilation error, by trying to bind rvalue references to lvalue references
            Impl::run(std::forward<Inputs>(ins)..., outs...);
            postprocess(outs...);
        }

        template<typename... Outputs>
        static void call(Impl& impl, Inputs&&... ins, Outputs&&... outs)
        {
            impl(std::forward<Inputs>(ins)..., outs...);
        }
    };

    template<int... IIs, int... OIs>
    static void call_impl(GCPUContext &ctx, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        //Make sure that OpenCV kernels do not reallocate memory for output parameters
        //by comparing it's state (data ptr) before and after the call.
        //This is done by converting each output Mat into tracked_cv_mat object, and binding
        //them to parameters of ad-hoc function
        call_and_postprocess<decltype(get_in<Ins>::get(ctx, IIs))...>
            ::call(get_in<Ins>::get(ctx, IIs)..., get_out<Outs>::get(ctx, OIs)...);
    }

    template<int... IIs, int... OIs>
    static void call_impl(cv::GCPUContext &ctx, Impl& impl,
                          detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        call_and_postprocess<decltype(get_in<Ins>::get(ctx, IIs))...>
            ::call(impl, get_in<Ins>::get(ctx, IIs)..., get_out<Outs>::get(ctx, OIs)...);
    }

    static void call(GCPUContext &ctx)
    {
        call_impl(ctx,
                  typename detail::MkSeq<sizeof...(Ins)>::type(),
                  typename detail::MkSeq<sizeof...(Outs)>::type());
    }

    // NB: Same as call but calling the object
    // This necessary for kernel implementations that have a state
    // and are represented as an object
    static void callFunctor(cv::GCPUContext &ctx, Impl& impl)
    {
        call_impl(ctx, impl,
                  typename detail::MkSeq<sizeof...(Ins)>::type(),
                  typename detail::MkSeq<sizeof...(Outs)>::type());
    }
};

// OCVStCallHelper is a helper class to call stateful OCV kernels.
template<typename, typename, typename>
struct OCVStCallHelper;

template<typename Impl, typename... Ins, typename... Outs>
struct OCVStCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...>> :
    OCVCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...>>
{
    template<typename... Inputs>
    struct call_and_postprocess
    {
        template<typename... Outputs>
        static void call(typename Impl::State& st, Inputs&&... ins, Outputs&&... outs)
        {
            Impl::run(std::forward<Inputs>(ins)..., outs..., st);
            postprocess(outs...);
        }
    };

    template<int... IIs, int... OIs>
    static void call_impl(GCPUContext &ctx, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        auto state_ptr = ctx.state().get<std::shared_ptr<typename Impl::State>>();
        if (state_ptr == nullptr) {
            CV_Error(cv::Error::StsNullPtr, "Stateful kernel's state is not initialized. "
                     "Make sure the setup() function properly initializes the state.");
        }
        auto& st = *state_ptr;
        call_and_postprocess<decltype(get_in<Ins>::get(ctx, IIs))...>
            ::call(st, get_in<Ins>::get(ctx, IIs)..., get_out<Outs>::get(ctx, OIs)...);
    }

    static void call(GCPUContext &ctx)
    {
        call_impl(ctx,
                  typename detail::MkSeq<sizeof...(Ins)>::type(),
                  typename detail::MkSeq<sizeof...(Outs)>::type());
    }
};

} // namespace detail

template<class Impl, class K>
class GCPUKernelImpl: public cv::detail::KernelTag
{
    using CallHelper = cv::detail::OCVCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

public:
    using API = K;

    static cv::gapi::GBackend backend() { return cv::gapi::cpu::backend(); }
    static cv::GCPUKernel      kernel() { return GCPUKernel(&CallHelper::call); }
};

template<class Impl, class K, class S>
class GCPUStKernelImpl: public cv::detail::KernelTag
{
    using StSetupHelper = detail::OCVSetupHelper<Impl, typename K::InArgs>;
    using StCallHelper  = detail::OCVStCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

public:
    using API = K;
    using State = S;

    static cv::gapi::GBackend backend() { return cv::gapi::cpu::backend(); }
    static cv::GCPUKernel     kernel()  { return GCPUKernel(&StCallHelper::call,
                                                            &StSetupHelper::setup); }
};

#define GAPI_OCV_KERNEL(Name, API) struct Name: public cv::GCPUKernelImpl<Name, API>

// TODO: Reuse Anatoliy's logic for support of types with commas in macro.
//       Retrieve the common part from Anatoliy's logic to the separate place.
#define GAPI_OCV_KERNEL_ST(Name, API, State)                   \
    struct Name: public cv::GCPUStKernelImpl<Name, API, State> \

/// @private
class gapi::cpu::GOCVFunctor : public gapi::GFunctor
{
public:
    using Impl = std::function<void(GCPUContext &)>;
    using Meta = cv::GKernel::M;

    GOCVFunctor(const char* id, const Meta &meta, const Impl& impl)
        : gapi::GFunctor(id), impl_{GCPUKernel(impl), meta}
    {
    }

    GKernelImpl    impl()    const override { return impl_;                }
    gapi::GBackend backend() const override { return gapi::cpu::backend(); }

private:
    GKernelImpl impl_;
};

//! @cond IGNORED
template<typename K, typename Callable>
gapi::cpu::GOCVFunctor gapi::cpu::ocv_kernel(Callable& c)
{
    using P = cv::detail::OCVCallHelper<Callable, typename K::InArgs, typename K::OutArgs>;
    return GOCVFunctor{ K::id()
                      , &K::getOutMeta
                      , std::bind(&P::callFunctor, std::placeholders::_1, std::ref(c))
                      };
}

template<typename K, typename Callable>
gapi::cpu::GOCVFunctor gapi::cpu::ocv_kernel(const Callable& c)
{
    using P = cv::detail::OCVCallHelper<Callable, typename K::InArgs, typename K::OutArgs>;
    return GOCVFunctor{ K::id()
                      , &K::getOutMeta
                      , std::bind(&P::callFunctor, std::placeholders::_1, c)
                      };
}
//! @endcond

} // namespace cv

#if defined _MSC_VER
#pragma warning(pop)
#endif

#endif // OPENCV_GAPI_GCPUKERNEL_HPP
