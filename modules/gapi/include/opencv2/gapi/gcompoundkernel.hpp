// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPOUNDKERNEL_HPP
#define OPENCV_GAPI_GCOMPOUNDKERNEL_HPP

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>

namespace cv {
namespace gapi
{
namespace compound
{
    // FIXME User does not need to know about this function
    // Needs that user may define compound kernels(as cpu kernels)
    GAPI_EXPORTS cv::gapi::GBackend backend();
} // namespace compound
} // namespace gapi

namespace detail
{

struct GCompoundContext
{
    explicit GCompoundContext(const GArgs& in_args);
    template<typename T>
    const T& inArg(int input) { return m_args.at(input).get<T>(); }

    GArgs m_args;
    GArgs m_results;
};

class GAPI_EXPORTS GCompoundKernel
{
// Compound kernel must use all of it's inputs
public:
    using F = std::function<void(GCompoundContext& ctx)>;

    explicit GCompoundKernel(const F& f);
    void apply(GCompoundContext& ctx);

protected:
    F m_f;
};

template<typename T> struct get_compound_in
{
    static T get(GCompoundContext &ctx, int idx) { return ctx.inArg<T>(idx); }
};

template<typename U> struct get_compound_in<cv::GArray<U>>
{
    static cv::GArray<U> get(GCompoundContext &ctx, int idx)
    {
        auto array = cv::GArray<U>();
        ctx.m_args[idx] = GArg(array);
        return array;
    }
};

template<typename U> struct get_compound_in<cv::GOpaque<U>>
{
    static cv::GOpaque<U> get(GCompoundContext &ctx, int idx)
    {
        auto opaq = cv::GOpaque<U>();
        ctx.m_args[idx] = GArg(opaq);
        return opaq;
    }
};

template<typename, typename, typename>
struct GCompoundCallHelper;

template<typename Impl, typename... Ins, typename... Outs>
struct GCompoundCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...> >
{
    template<int... IIs, int... OIs>
    static void expand_impl(GCompoundContext &ctx, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        auto result = Impl::expand(get_compound_in<Ins>::get(ctx, IIs)...);
        auto tuple_return = tuple_wrap_helper<decltype(result)>::get(std::move(result));
        ctx.m_results = { cv::GArg(std::get<OIs>(tuple_return))... };
    }

    static void expand(GCompoundContext &ctx)
    {
        expand_impl(ctx,
                    typename detail::MkSeq<sizeof...(Ins)>::type(),
                    typename detail::MkSeq<sizeof...(Outs)>::type());
    }
};

template<class Impl, class K>
class GCompoundKernelImpl: public cv::detail::GCompoundCallHelper<Impl, typename K::InArgs, typename K::OutArgs>,
                           public cv::detail::KernelTag
{
    using P = cv::detail::GCompoundCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

public:
    using API = K;

    static cv::gapi::GBackend backend() { return cv::gapi::compound::backend(); }
    static GCompoundKernel    kernel()  { return GCompoundKernel(&P::expand);   }
};

} // namespace detail


/**
 * Declares a new compound kernel. See this
 * [documentation chapter](@ref gapi_kernel_compound)
 * on compound kernels for more details.
 *
 * @param Name type name for new kernel
 * @param API the interface this kernel implements
 */
#define GAPI_COMPOUND_KERNEL(Name, API) \
    struct Name: public cv::detail::GCompoundKernelImpl<Name, API>

} // namespace cv

#endif // OPENCV_GAPI_GCOMPOUNDKERNEL_HPP
