// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_GS11NKERNEL_HPP
#define OPENCV_GAPI_GS11NKERNEL_HPP

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/util/compiler_hints.hpp> //suppress_unused_warning
#include <opencv2/gapi/util/util.hpp>

// FIXME: namespace scheme for backends?
namespace opencv_test {

    namespace s11n
    {
        namespace impl
        {
            // Forward-declare an internal class
            class GS11NExecutable;
        } // namespace impl
    } // namespace serialization

    namespace s11n
    {
        namespace impl
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

            class GS11NFunctor;

            //! @cond IGNORED
            template<typename K, typename Callable>
            GS11NFunctor s11n_kernel(const Callable& c);

            template<typename K, typename Callable>
            GS11NFunctor s11n_kernel(Callable& c);
            //! @endcond

        } // namespace impl
    } // namespace s11n

      // Represents arguments which are passed to a wrapped CPU function
      // FIXME: put into detail?
    class GAPI_EXPORTS GS11NContext
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
        template<typename T> std::vector<T>& outVecR(int output) // FIXME: the same issue
        {
            return outVecRef(output).wref<T>();
        }
        template<typename T> T& outOpaqueR(int output) // FIXME: the same issue
        {
            return outOpaqueRef(output).wref<T>();
        }

    protected:
        cv::detail::VectorRef& outVecRef(int output);
        cv::detail::OpaqueRef& outOpaqueRef(int output);

        std::vector<cv::GArg> m_args;

        //FIXME: avoid conversion of arguments from internal representation to OpenCV one on each call
        //to OCV kernel. (This can be achieved by a two single time conversions in GCPUExecutable::run,
        //once on enter for input and output arguments, and once before return for output arguments only
        std::unordered_map<std::size_t, cv::GRunArgP> m_results;

        //friend class gimpl::GS11NExecutable;
        friend class opencv_test::s11n::impl::GS11NExecutable;
    };

    class GAPI_EXPORTS GS11NKernel
    {
    public:
        // This function is kernel's execution entry point (does the processing work)
        using F = std::function<void(GS11NContext &)>;

        GS11NKernel();
        explicit GS11NKernel(const F& f);

        void apply(GS11NContext &ctx);

    protected:
        F m_f;
    };

    // FIXME: This is an ugly ad-hoc implementation. TODO: refactor

    namespace detail
    {
        template<class T> struct s11n_get_in;
        template<> struct s11n_get_in<cv::GMat>
        {
            static cv::Mat    get(GS11NContext &ctx, int idx) { return ctx.inMat(idx); }
        };
        template<> struct s11n_get_in<cv::GMatP>
        {
            static cv::Mat    get(GS11NContext &ctx, int idx) { return s11n_get_in<cv::GMat>::get(ctx, idx); }
        };
        template<> struct s11n_get_in<cv::GFrame>
        {
            static cv::Mat    get(GS11NContext &ctx, int idx) { return s11n_get_in<cv::GMat>::get(ctx, idx); }
        };
        template<> struct s11n_get_in<cv::GScalar>
        {
            static cv::Scalar get(GS11NContext &ctx, int idx) { return ctx.inVal(idx); }
        };
        template<typename U> struct s11n_get_in<cv::GArray<U> >
        {
            static const std::vector<U>& get(GS11NContext &ctx, int idx) { return ctx.inArg<cv::detail::VectorRef>(idx).rref<U>(); }
        };
        template<typename U> struct s11n_get_in<cv::GOpaque<U> >
        {
            static const U& get(GS11NContext &ctx, int idx) { return ctx.inArg<cv::detail::OpaqueRef>(idx).rref<U>(); }
        };

        //FIXME(dm): GArray<Mat>/GArray<GMat> conversion should be done more gracefully in the system
        template<> struct s11n_get_in<cv::GArray<cv::GMat> > : public s11n_get_in<cv::GArray<cv::Mat> >
        {
        };

        //FIXME(dm): GArray<Scalar>/GArray<GScalar> conversion should be done more gracefully in the system
        template<> struct s11n_get_in<cv::GArray<cv::GScalar> > : public s11n_get_in<cv::GArray<cv::Scalar> >
        {
        };

        //FIXME(dm): GOpaque<Mat>/GOpaque<GMat> conversion should be done more gracefully in the system
        template<> struct s11n_get_in<cv::GOpaque<cv::GMat> > : public s11n_get_in<cv::GOpaque<cv::Mat> >
        {
        };

        //FIXME(dm): GOpaque<Scalar>/GOpaque<GScalar> conversion should be done more gracefully in the system
        template<> struct s11n_get_in<cv::GOpaque<cv::GScalar> > : public s11n_get_in<cv::GOpaque<cv::Mat> >
        {
        };

        template<class T> struct s11n_get_in
        {
            static T get(GS11NContext &ctx, int idx) { return ctx.inArg<T>(idx); }
        };

        struct tracked_s11n_mat {
            tracked_s11n_mat(cv::Mat& m) : r{ m }, original_data{ m.data } {}
            cv::Mat r;
            uchar* original_data;

            operator cv::Mat& () { return r; }
            void validate() const {
                if (r.data != original_data)
                {
                    cv::util::throw_error
                    (std::logic_error
                    ("OpenCV kernel output parameter was reallocated. \n"
                        "Incorrect meta data was provided ?"));
                }
            }
        };

        template<typename... Outputs>
        void postprocess_s11n(Outputs&... outs)
        {
            struct
            {
                void operator()(tracked_s11n_mat* bm) { bm->validate(); }
                void operator()(...) {                  }

            } validate;
            //dummy array to unfold parameter pack
            int dummy[] = { 0, (validate(&outs), 0)... };
            cv::util::suppress_unused_warning(dummy);
        }

        template<class T> struct s11n_get_out;
        template<> struct s11n_get_out<cv::GMat>
        {
            static tracked_s11n_mat get(GS11NContext &ctx, int idx)
            {
                auto& r = ctx.outMatR(idx);
                return{ r };
            }
        };
        template<> struct s11n_get_out<cv::GMatP>
        {
            static tracked_s11n_mat get(GS11NContext &ctx, int idx)
            {
                return s11n_get_out<cv::GMat>::get(ctx, idx);
            }
        };
        template<> struct s11n_get_out<cv::GScalar>
        {
            static cv::Scalar& get(GS11NContext &ctx, int idx)
            {
                return ctx.outValR(idx);
            }
        };
        template<typename U> struct s11n_get_out<cv::GArray<U>>
        {
            static std::vector<U>& get(GS11NContext &ctx, int idx)
            {
                return ctx.outVecR<U>(idx);
            }
        };
        template<typename U> struct s11n_get_out<cv::GOpaque<U>>
        {
            static U& get(GS11NContext &ctx, int idx)
            {
                return ctx.outOpaqueR<U>(idx);
            }
        };

        template<typename, typename, typename>
        struct S11NCallHelper;

        // FIXME: probably can be simplified with std::apply or analogue.
        template<typename Impl, typename... Ins, typename... Outs>
        struct S11NCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...> >
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
                    postprocess_s11n(outs...);
                }

                template<typename... Outputs>
                static void call(Impl& impl, Inputs&&... ins, Outputs&&... outs)
                {
                    impl(std::forward<Inputs>(ins)..., outs...);
                }
            };

            template<int... IIs, int... OIs>
            static void call_impl(GS11NContext &ctx, cv::detail::Seq<IIs...>, cv::detail::Seq<OIs...>)
            {
                //Make sure that OpenCV kernels do not reallocate memory for output parameters
                //by comparing it's state (data ptr) before and after the call.
                //This is done by converting each output Mat into tracked_cv_mat object, and binding
                //them to parameters of ad-hoc function
                //Convert own::Scalar to cv::Scalar before call kernel and run kernel
                //convert cv::Scalar to own::Scalar after call kernel and write back results
                call_and_postprocess<decltype(s11n_get_in<Ins>::get(ctx, IIs))...>
                    ::call(s11n_get_in<Ins>::get(ctx, IIs)...,
                        s11n_get_out<Outs>::get(ctx, OIs)...);
            }

            template<int... IIs, int... OIs>
            static void call_impl(opencv_test::GS11NContext &ctx, Impl& impl, cv::detail::Seq<IIs...>, cv::detail::Seq<OIs...>)
            {
                call_and_postprocess<decltype(opencv_test::detail::s11n_get_in<Ins>::get(ctx, IIs))...>
                    ::call(impl, opencv_test::detail::s11n_get_in<Ins>::get(ctx, IIs)...,
                        opencv_test::detail::s11n_get_out<Outs>::get(ctx, OIs)...);
            }

            static void call(GS11NContext &ctx)
            {
                call_impl(ctx,
                    typename cv::detail::MkSeq<sizeof...(Ins)>::type(),
                    typename cv::detail::MkSeq<sizeof...(Outs)>::type());
            }

            // NB: Same as call but calling the object
            // This necessary for kernel implementations that have a state
            // and are represented as an object
            static void callFunctor(opencv_test::GS11NContext &ctx, Impl& impl)
            {
                call_impl(ctx, impl,
                    typename cv::detail::MkSeq<sizeof...(Ins)>::type(),
                    typename cv::detail::MkSeq<sizeof...(Outs)>::type());
            }
        };

    } // namespace detail

    template<class Impl, class K>
    class GS11NKernelImpl : public opencv_test::detail::S11NCallHelper<Impl, typename K::InArgs, typename K::OutArgs>,
        public cv::detail::KernelTag
    {
        using P = detail::S11NCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

    public:
        using API = K;

        static cv::gapi::GBackend backend() { return opencv_test::s11n::impl::backend(); }
        static opencv_test::GS11NKernel     kernel() { return GS11NKernel(&P::call); }
    };

#define GAPI_S11N_KERNEL(Name, API) struct Name: public opencv_test::GS11NKernelImpl<Name, API>

    class s11n::impl::GS11NFunctor : public cv::gapi::GFunctor
    {
    public:
        using Impl = std::function<void(GS11NContext &)>;

        GS11NFunctor(const char* id, const Impl& impl)
            : cv::gapi::GFunctor(id), impl_{ GS11NKernel(impl) }
        {
        }

        cv::GKernelImpl    impl()    const override { return impl_; }
        cv::gapi::GBackend backend() const override { return s11n::impl::backend(); }

    private:
        cv::GKernelImpl impl_;
    };

    //! @cond IGNORED
    template<typename K, typename Callable>
    s11n::impl::GS11NFunctor s11n::impl::s11n_kernel(Callable& c)
    {
        using P = detail::S11NCallHelper<Callable, typename K::InArgs, typename K::OutArgs>;
        return GS11NFunctor(K::id(), std::bind(&P::callFunctor, std::placeholders::_1, std::ref(c)));
    }

    template<typename K, typename Callable>
    s11n::impl::GS11NFunctor s11n::impl::s11n_kernel(const Callable& c)
    {
        using P = detail::S11NCallHelper<Callable, typename K::InArgs, typename K::OutArgs>;
        return GS11NFunctor(K::id(), std::bind(&P::callFunctor, std::placeholders::_1, c));
    }
    //! @endcond

} // namespace opencv_test

#endif // OPENCV_GAPI_GS11NKERNEL_HPP
