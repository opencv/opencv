// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation
//


#ifndef OPENCV_GAPI_OVCV_LAYER_HPP
#define OPENCV_GAPI_OVCV_LAYER_HPP

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>

#include "backends/ov/ovdef.hpp"

#ifdef HAVE_OPENVINO_2_0

namespace cv
{
namespace gimpl
{
namespace ovcv
{

cv::gapi::GBackend backend();

} // namespace ovcv
} // namespace gimpl

struct GOVCVContext
{
    // Generic accessor API
    template<typename T>
    const T& inArg(int input) { return m_args.at(input).get<T>(); }

    // Syntax sugar
    const ov::Output<ov::Node>& inTensor(int input)
    {
        return inArg<ov::Output<ov::Node>>(input);
    }

    ov::Output<ov::Node>& outTensor(int output)
    {
        return *(m_results.at(output).get<ov::Output<ov::Node>*>());
    }

    std::vector<GArg> m_args;
    std::unordered_map<std::size_t, GArg> m_results;
};

class GOVCVKernel
{
public:
    using F = std::function<void(GOVCVContext &)>;

    GOVCVKernel() = default;
    explicit GOVCVKernel(const F& f) : m_f(f) {}

    void apply(GOVCVContext &ctx) const
    {
        GAPI_Assert(m_f);
        m_f(ctx);
    }

protected:
    F m_f;
};

namespace detail
{

template<class T> struct ovcv_get_in;
template<> struct ovcv_get_in<cv::GMat>
{
    static const ov::Output<ov::Node>& get(GOVCVContext& ctx, int idx)
    {
        return ctx.inTensor(idx);
    }
};

template<class T> struct ovcv_get_in
{
    static T get(GOVCVContext &ctx, int idx) { return ctx.inArg<T>(idx); }
};

template<class T> struct ovcv_get_out;
template<> struct ovcv_get_out<cv::GMat>
{
    static ov::Output<ov::Node>& get(GOVCVContext& ctx, int idx)
    {
        return ctx.outTensor(idx);
    }
};

template<typename, typename, typename>
struct OVCVCallHelper;

template<typename Impl, typename... Ins, typename... Outs>
struct OVCVCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...> >
{
    template<int... IIs, int... OIs>
    static void call_impl(GOVCVContext &ctx, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        Impl::run(ovcv_get_in<Ins>::get(ctx, IIs)..., ovcv_get_out<Outs>::get(ctx, OIs)...);
    }

    static void call(GOVCVContext& ctx)
    {
        call_impl(ctx,
                  typename detail::MkSeq<sizeof...(Ins)>::type(),
                  typename detail::MkSeq<sizeof...(Outs)>::type());
    }
};

} // namespace detail

template<class Impl, class K>
class GOVCVKernelImpl: public cv::detail::OVCVCallHelper<Impl, typename K::InArgs, typename K::OutArgs>,
                          public cv::detail::KernelTag
{
    using P = detail::OVCVCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

public:
    using API = K;

    static cv::gapi::GBackend backend()  { return cv::gimpl::ovcv::backend();   }
    static cv::GOVCVKernel kernel()      { return GOVCVKernel(&P::call);        }
};

#define GAPI_OVCV_KERNEL(Name, API) struct Name: public cv::GOVCVKernelImpl<Name, API>

} // namespace cv

#endif // HAVE_OPENVINO_2_0
#endif // OPENCV_GAPI_OVCV_LAYER_HPP
