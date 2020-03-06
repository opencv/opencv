// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_FLUID_KERNEL_HPP
#define OPENCV_GAPI_FLUID_KERNEL_HPP

#include <vector>
#include <functional>
#include <map>
#include <unordered_map>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>

#include <opencv2/gapi/fluid/gfluidbuffer.hpp>

// FIXME: namespace scheme for backends?
namespace cv {

namespace gapi
{
namespace fluid
{
    /**
     * \addtogroup gapi_std_backends G-API Standard Backends
     * @{
     */
    /**
     * @brief Get a reference to Fluid backend.
     *
     * @sa gapi_std_backends
     */
    GAPI_EXPORTS cv::gapi::GBackend backend();
    /** @} */
} // namespace fluid
} // namespace gapi


class GAPI_EXPORTS GFluidKernel
{
public:
    enum class Kind
    {
        Filter,
        Resize,
        YUV420toRGB //Color conversion of 4:2:0 chroma sub-sampling formats (NV12, I420 ..etc) to RGB
    };

    // This function is a generic "doWork" callback
    using F = std::function<void(const cv::GArgs&, const std::vector<gapi::fluid::Buffer*> &)>;

    // This function is a generic "initScratch" callback
    using IS = std::function<void(const cv::GMetaArgs &, const cv::GArgs&, gapi::fluid::Buffer &)>;

    // This function is a generic "resetScratch" callback
    using RS = std::function<void(gapi::fluid::Buffer &)>;

    // This function describes kernel metadata inference rule.
    using M = std::function<GMetaArgs(const GMetaArgs &, const GArgs &)>;

    // This function is a generic "getBorder" callback (extracts border-related data from kernel's input parameters)
    using B = std::function<gapi::fluid::BorderOpt(const GMetaArgs&, const GArgs&)>;

    // This function is a generic "getWindow" callback (extracts window-related data from kernel's input parameters)
    using GW = std::function<int(const GMetaArgs&, const GArgs&)>;

    // FIXME: move implementations out of header file
    GFluidKernel() {}
    GFluidKernel(Kind k, int l, bool scratch, const F& f, const IS &is, const RS &rs, const B& b, const GW& win)
        : m_kind(k)
        , m_lpi(l)
        , m_scratch(scratch)
        , m_f(f)
        , m_is(is)
        , m_rs(rs)
        , m_b(b)
        , m_gw(win) {}

    Kind m_kind;
    const int  m_lpi     = -1;
    const bool m_scratch = false;

    const F    m_f;
    const IS   m_is;
    const RS   m_rs;
    const B    m_b;
    const GW   m_gw;
};

// FIXME!!!
// This is the temporary and experimental API
// which should be replaced by runtime roi-based scheduling
/** \addtogroup gapi_compile_args
 * @{
 */
/**
 * @brief This structure allows to control the output image region
 * which Fluid backend will produce in the graph.
 *
 * This feature is useful for external tiling and parallelism, but
 * will be deprecated in the future releases.
 */
struct GFluidOutputRois
{
    std::vector<cv::Rect> rois;
};

/**
 * @brief This structure forces Fluid backend to generate multiple
 * parallel output regions in the graph. These regions execute in parallel.
 *
 * This feature may be deprecated in the future releases.
 */
struct GFluidParallelOutputRois
{
    std::vector<GFluidOutputRois> parallel_rois;
};

/**
 * @brief This structure allows to customize the way how Fluid executes
 * parallel regions.
 *
 * For example, user can utilize his own threading runtime via this parameter.
 * The `parallel_for` member functor is called by the Fluid runtime with the
 * following arguments:
 *
 * @param size Size of the parallel range to process
 * @param f A function which should be called for every integer index
 *   in this range by the specified parallel_for implementation.
 *
 * This feature may be deprecated in the future releases.
 */
struct GFluidParallelFor
{
    //this function accepts:
    // - size of the "parallel" range as the first argument
    // - and a function to be called on the range items, designated by item index
    std::function<void(std::size_t size, std::function<void(std::size_t index)>)> parallel_for;
};
/** @} gapi_compile_args */

namespace detail
{
template<> struct CompileArgTag<GFluidOutputRois>
{
    static const char* tag() { return "gapi.fluid.outputRois"; }
};

template<> struct CompileArgTag<GFluidParallelFor>
{
    static const char* tag() { return "gapi.fluid.parallelFor"; }
};

template<> struct CompileArgTag<GFluidParallelOutputRois>
{
    static const char* tag() { return "gapi.fluid.parallelOutputRois"; }
};

} // namespace detail

namespace detail
{
template<class T> struct fluid_get_in;
template<> struct fluid_get_in<cv::GMat>
{
    static const cv::gapi::fluid::View& get(const cv::GArgs &in_args, int idx)
    {
        return in_args[idx].unsafe_get<cv::gapi::fluid::View>();
    }
};

template<> struct fluid_get_in<cv::GScalar>
{
    // FIXME: change to return by reference when moved to own::Scalar
    static const cv::Scalar get(const cv::GArgs &in_args, int idx)
    {
        return in_args[idx].unsafe_get<cv::Scalar>();
    }
};

template<typename U> struct fluid_get_in<cv::GArray<U>>
{
    static const std::vector<U>& get(const cv::GArgs &in_args, int idx)
    {
        return in_args.at(idx).unsafe_get<cv::detail::VectorRef>().rref<U>();
    }
};

template<typename U> struct fluid_get_in<cv::GOpaque<U>>
{
    static const U& get(const cv::GArgs &in_args, int idx)
    {
        return in_args.at(idx).unsafe_get<cv::detail::OpaqueRef>().rref<U>();
    }
};

template<class T> struct fluid_get_in
{
    static const T& get(const cv::GArgs &in_args, int idx)
    {
        return in_args[idx].unsafe_get<T>();
    }
};

template<bool, typename Impl, typename... Ins>
struct scratch_helper;

template<typename Impl, typename... Ins>
struct scratch_helper<true, Impl, Ins...>
{
    // Init
    template<int... IIs>
    static void help_init_impl(const cv::GMetaArgs &metas,
                               const cv::GArgs     &in_args,
                               gapi::fluid::Buffer &scratch_buf,
                               detail::Seq<IIs...>)
    {
        Impl::initScratch(get_in_meta<Ins>(metas, in_args, IIs)..., scratch_buf);
    }

    static void help_init(const cv::GMetaArgs &metas,
                          const cv::GArgs     &in_args,
                          gapi::fluid::Buffer &b)
    {
        help_init_impl(metas, in_args, b, typename detail::MkSeq<sizeof...(Ins)>::type());
    }

    // Reset
    static void help_reset(gapi::fluid::Buffer &b)
    {
        Impl::resetScratch(b);
    }
};

template<typename Impl, typename... Ins>
struct scratch_helper<false, Impl, Ins...>
{
    static void help_init(const cv::GMetaArgs &,
                          const cv::GArgs     &,
                          gapi::fluid::Buffer &)
    {
        GAPI_Assert(false);
    }
    static void help_reset(gapi::fluid::Buffer &)
    {
        GAPI_Assert(false);
    }
};

template<typename T> struct is_gmat_type
{
    static const constexpr bool value = std::is_same<cv::GMat, T>::value;
};

template<bool CallCustomGetBorder, typename Impl, typename... Ins>
struct get_border_helper;

template<typename Impl, typename... Ins>
struct get_border_helper<true, Impl, Ins...>
{
    template<int... IIs>
    static gapi::fluid::BorderOpt get_border_impl(const GMetaArgs &metas,
                                                  const cv::GArgs &in_args,
                                                  cv::detail::Seq<IIs...>)
    {
        return util::make_optional(Impl::getBorder(cv::detail::get_in_meta<Ins>(metas, in_args, IIs)...));
    }

    static gapi::fluid::BorderOpt help(const GMetaArgs &metas,
                                       const cv::GArgs &in_args)
    {
        return get_border_impl(metas, in_args, typename detail::MkSeq<sizeof...(Ins)>::type());
    }
};

template<typename Impl, typename... Ins>
struct get_border_helper<false, Impl, Ins...>
{
    static gapi::fluid::BorderOpt help(const cv::GMetaArgs &,
                                       const cv::GArgs     &)
    {
        return {};
    }
};

template<bool CallCustomGetWindow, typename, typename... Ins>
struct get_window_helper;

template<typename Impl, typename... Ins>
struct get_window_helper<true, Impl, Ins...>
{
    template<int... IIs>
    static int get_window_impl(const GMetaArgs &metas,
                               const cv::GArgs &in_args,
                               cv::detail::Seq<IIs...>)
    {
        return Impl::getWindow(cv::detail::get_in_meta<Ins>(metas, in_args, IIs)...);
    }

    static int help(const GMetaArgs &metas, const cv::GArgs &in_args)
    {
        return get_window_impl(metas, in_args, typename detail::MkSeq<sizeof...(Ins)>::type());
    }
};

template<typename Impl, typename... Ins>
struct get_window_helper<false, Impl, Ins...>
{
    static int help(const cv::GMetaArgs &,
                    const cv::GArgs     &)
    {
        return Impl::Window;
    }
};

template<typename C, typename T>
struct has_Window
{
private:
    template<class U>
    static constexpr auto Check(U*) -> typename std::is_same<decltype(U::Window), T>::type;

    template<typename>
    static constexpr std::false_type Check(...);

    typedef decltype(Check<C>(0)) Result;

public:
    static constexpr bool value = Result::value;
};

template<bool hasWindow, typename Impl>
struct callCustomGetBorder;

template<typename Impl>
struct callCustomGetBorder<true, Impl>
{
    static constexpr bool value = (Impl::Window != 1);
};

template<typename Impl>
struct callCustomGetBorder<false, Impl>
{
    static constexpr bool value = true;
};

template<typename, typename, typename, bool UseScratch>
struct FluidCallHelper;

template<typename Impl, typename... Ins, typename... Outs, bool UseScratch>
struct FluidCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...>, UseScratch>
{
    static_assert(all_satisfy<is_gmat_type, Outs...>::value, "return type must be GMat");
    static_assert(contains<GMat, Ins...>::value, "input must contain at least one GMat");

    // Execution dispatcher ////////////////////////////////////////////////////
    template<int... IIs, int... OIs>
    static void call_impl(const cv::GArgs &in_args,
                          const std::vector<gapi::fluid::Buffer*> &out_bufs,
                          detail::Seq<IIs...>,
                          detail::Seq<OIs...>)
    {
        Impl::run(fluid_get_in<Ins>::get(in_args, IIs)..., *out_bufs[OIs]...);
    }

    static void call(const cv::GArgs &in_args,
                     const std::vector<gapi::fluid::Buffer*> &out_bufs)
    {
        constexpr int numOuts = (sizeof...(Outs)) + (UseScratch ? 1 : 0);
        call_impl(in_args, out_bufs,
                  typename detail::MkSeq<sizeof...(Ins)>::type(),
                  typename detail::MkSeq<numOuts>::type());
    }

    // Scratch buffer initialization dispatcher ////////////////////////////////
    static void init_scratch(const GMetaArgs &metas,
                             const cv::GArgs &in_args,
                             gapi::fluid::Buffer &b)
    {
        scratch_helper<UseScratch, Impl, Ins...>::help_init(metas, in_args, b);
    }

    // Scratch buffer reset dispatcher /////////////////////////////////////////
    static void reset_scratch(gapi::fluid::Buffer &scratch_buf)
    {
        scratch_helper<UseScratch, Impl, Ins...>::help_reset(scratch_buf);
    }

    static gapi::fluid::BorderOpt getBorder(const GMetaArgs &metas, const cv::GArgs &in_args)
    {
        constexpr bool hasWindow = has_Window<Impl, const int>::value;

        // User must provide "init" callback if Window != 1
        // TODO: move to constexpr if when we enable C++17
        return get_border_helper<callCustomGetBorder<hasWindow, Impl>::value, Impl, Ins...>::help(metas, in_args);
    }

    static int getWindow(const GMetaArgs &metas, const cv::GArgs &in_args)
    {
        constexpr bool callCustomGetWindow = !(has_Window<Impl, const int>::value);
        return get_window_helper<callCustomGetWindow, Impl, Ins...>::help(metas, in_args);
    }
};
} // namespace detail


template<class Impl, class K, bool UseScratch>
class GFluidKernelImpl : public cv::detail::KernelTag
{
    static const int LPI = 1;
    static const auto Kind = GFluidKernel::Kind::Filter;
    using P = detail::FluidCallHelper<Impl, typename K::InArgs, typename K::OutArgs, UseScratch>;

public:
    using API = K;

    static GFluidKernel kernel()
    {
        // FIXME: call() and getOutMeta() needs to be renamed so it is clear these
        // functions are internal wrappers, not user API
        return GFluidKernel(Impl::Kind, Impl::LPI,
                            UseScratch,
                            &P::call, &P::init_scratch, &P::reset_scratch, &P::getBorder, &P::getWindow);
    }

    static cv::gapi::GBackend backend() { return cv::gapi::fluid::backend(); }
};

#define GAPI_FLUID_KERNEL(Name, API, Scratch) struct Name: public cv::GFluidKernelImpl<Name, API, Scratch>

} // namespace cv

#endif // OPENCV_GAPI_GCPUKERNEL_HPP
