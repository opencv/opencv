// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GOCLKERNEL_HPP
#define OPENCV_GAPI_GOCLKERNEL_HPP

#include <vector>
#include <functional>
#include <map>
#include <unordered_map>

#include <opencv2/core/mat.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>

// FIXME: namespace scheme for backends?
namespace cv {

namespace gimpl
{
    // Forward-declare an internal class
    class GOCLExecutable;
} // namespace gimpl

namespace gapi
{
/**
 * @brief This namespace contains G-API OpenCL backend functions, structures, and symbols.
 */
namespace ocl
{
    /**
     * \addtogroup gapi_std_backends G-API Standard Backends
     * @{
     */
    /**
     * @brief Get a reference to OCL backend.
     *
     * At the moment, the OCL backend is built atop of OpenCV
     * "Transparent API" (T-API), see cv::UMat for details.
     *
     * @sa gapi_std_backends
     */
    GAPI_EXPORTS cv::gapi::GBackend backend();
    /** @} */
} // namespace ocl
} // namespace gapi


// Represents arguments which are passed to a wrapped OCL function
// FIXME: put into detail?
class GAPI_EXPORTS GOCLContext
{
public:
    // Generic accessor API
    template<typename T>
    const T& inArg(int input) { return m_args.at(input).get<T>(); }

    // Syntax sugar
    const cv::UMat&  inMat(int input);
    cv::UMat&  outMatR(int output); // FIXME: Avoid cv::Mat m = ctx.outMatR()

    const cv::Scalar& inVal(int input);
    cv::Scalar& outValR(int output); // FIXME: Avoid cv::Scalar s = ctx.outValR()
    template<typename T> std::vector<T>& outVecR(int output) // FIXME: the same issue
    {
        return outVecRef(output).wref<T>();
    }
    template<typename T> T& outOpaqueR(int output) // FIXME: the same issue
    {
        return outOpaqueRef(output).wref<T>();
    }

protected:
    detail::VectorRef& outVecRef(int output);
    detail::OpaqueRef& outOpaqueRef(int output);

    std::vector<GArg> m_args;
    std::unordered_map<std::size_t, GRunArgP> m_results;


    friend class gimpl::GOCLExecutable;
};

class GAPI_EXPORTS GOCLKernel
{
public:
    // This function is kernel's execution entry point (does the processing work)
    using F = std::function<void(GOCLContext &)>;

    GOCLKernel();
    explicit GOCLKernel(const F& f);

    void apply(GOCLContext &ctx);

protected:
    F m_f;
};

// FIXME: This is an ugly ad-hoc implementation. TODO: refactor

namespace detail
{
template<class T> struct ocl_get_in;
template<> struct ocl_get_in<cv::GMat>
{
    static cv::UMat    get(GOCLContext &ctx, int idx) { return ctx.inMat(idx); }
};
template<> struct ocl_get_in<cv::GScalar>
{
    static cv::Scalar get(GOCLContext &ctx, int idx) { return ctx.inVal(idx); }
};
template<typename U> struct ocl_get_in<cv::GArray<U> >
{
    static const std::vector<U>& get(GOCLContext &ctx, int idx) { return ctx.inArg<VectorRef>(idx).rref<U>(); }
};
template<typename U> struct ocl_get_in<cv::GOpaque<U> >
{
    static const U& get(GOCLContext &ctx, int idx) { return ctx.inArg<OpaqueRef>(idx).rref<U>(); }
};
template<class T> struct ocl_get_in
{
    static T get(GOCLContext &ctx, int idx) { return ctx.inArg<T>(idx); }
};

struct tracked_cv_umat{
    //TODO Think if T - API could reallocate UMat to a proper size - how do we handle this ?
    //tracked_cv_umat(cv::UMat& m) : r{(m)}, original_data{m.getMat(ACCESS_RW).data} {}
    tracked_cv_umat(cv::UMat& m) : r(m), original_data{ nullptr } {}
    cv::UMat &r; // FIXME: It was a value (not a reference) before.
                 // Actually OCL backend should allocate its internal data!
    uchar* original_data;

    operator cv::UMat& (){ return r;}
    void validate() const{
        //if (r.getMat(ACCESS_RW).data != original_data)
        //{
        //    util::throw_error
        //        (std::logic_error
        //         ("OpenCV kernel output parameter was reallocated. \n"
        //          "Incorrect meta data was provided ?"));
        //}

    }
};

template<typename... Outputs>
void postprocess_ocl(Outputs&... outs)
{
    struct
    {
        void operator()(tracked_cv_umat* bm) { bm->validate(); }
        void operator()(...) {                  }

    } validate;
    //dummy array to unfold parameter pack
    int dummy[] = { 0, (validate(&outs), 0)... };
    cv::util::suppress_unused_warning(dummy);
}

template<class T> struct ocl_get_out;
template<> struct ocl_get_out<cv::GMat>
{
    static tracked_cv_umat get(GOCLContext &ctx, int idx)
    {
        auto& r = ctx.outMatR(idx);
        return{ r };
    }
};
template<> struct ocl_get_out<cv::GScalar>
{
    static cv::Scalar& get(GOCLContext &ctx, int idx)
    {
        return ctx.outValR(idx);
    }
};
template<typename U> struct ocl_get_out<cv::GArray<U> >
{
    static std::vector<U>& get(GOCLContext &ctx, int idx) { return ctx.outVecR<U>(idx);  }
};
template<typename U> struct ocl_get_out<cv::GOpaque<U> >
{
    static U& get(GOCLContext &ctx, int idx) { return ctx.outOpaqueR<U>(idx);  }
};

template<typename, typename, typename>
struct OCLCallHelper;

// FIXME: probably can be simplified with std::apply or analogue.
template<typename Impl, typename... Ins, typename... Outs>
struct OCLCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...> >
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

            postprocess_ocl(outs...);
        }
    };

    template<int... IIs, int... OIs>
    static void call_impl(GOCLContext &ctx, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        //TODO: Make sure that OpenCV kernels do not reallocate memory for output parameters
        //by comparing it's state (data ptr) before and after the call.
        //Convert own::Scalar to cv::Scalar before call kernel and run kernel
        //convert cv::Scalar to own::Scalar after call kernel and write back results
        call_and_postprocess<decltype(ocl_get_in<Ins>::get(ctx, IIs))...>::call(ocl_get_in<Ins>::get(ctx, IIs)..., ocl_get_out<Outs>::get(ctx, OIs)...);
    }

    static void call(GOCLContext &ctx)
    {
        call_impl(ctx,
            typename detail::MkSeq<sizeof...(Ins)>::type(),
            typename detail::MkSeq<sizeof...(Outs)>::type());
    }
};

} // namespace detail

template<class Impl, class K>
class GOCLKernelImpl: public cv::detail::OCLCallHelper<Impl, typename K::InArgs, typename K::OutArgs>,
                      public cv::detail::KernelTag
{
    using P = detail::OCLCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

public:
    using API = K;

    static cv::gapi::GBackend backend()  { return cv::gapi::ocl::backend(); }
    static cv::GOCLKernel     kernel()   { return GOCLKernel(&P::call);     }
};

#define GAPI_OCL_KERNEL(Name, API) struct Name: public cv::GOCLKernelImpl<Name, API>

} // namespace cv

#endif // OPENCV_GAPI_GOCLKERNEL_HPP
