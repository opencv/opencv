// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation
//


#ifndef OPENCV_GAPI_GPLAIDMLKERNEL_HPP
#define OPENCV_GAPI_GPLAIDMLKERNEL_HPP

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>

namespace plaidml
{
namespace edsl
{
    class Tensor;
} // namespace edsl
} // namespace plaidml

namespace cv
{
namespace gapi
{
namespace plaidml
{

GAPI_EXPORTS cv::gapi::GBackend backend();

} // namespace plaidml
} // namespace gapi

struct GPlaidMLContext
{
    // Generic accessor API
    template<typename T>
    const T& inArg(int input) { return m_args.at(input).get<T>(); }

    // Syntax sugar
    const plaidml::edsl::Tensor& inTensor(int input)
    {
        return inArg<plaidml::edsl::Tensor>(input);
    }

    plaidml::edsl::Tensor& outTensor(int output)
    {
        return *(m_results.at(output).get<plaidml::edsl::Tensor*>());
    }

    std::vector<GArg> m_args;
    std::unordered_map<std::size_t, GArg> m_results;
};

class GAPI_EXPORTS GPlaidMLKernel
{
public:
    using F = std::function<void(GPlaidMLContext &)>;

    GPlaidMLKernel() = default;
    explicit GPlaidMLKernel(const F& f) : m_f(f) {};

    void apply(GPlaidMLContext &ctx) const
    {
        GAPI_Assert(m_f);
        m_f(ctx);
    }

protected:
    F m_f;
};


namespace detail
{

template<class T> struct plaidml_get_in;
template<> struct plaidml_get_in<cv::GMat>
{
    static const plaidml::edsl::Tensor& get(GPlaidMLContext& ctx, int idx)
    {
        return ctx.inTensor(idx);
    }
};

template<class T> struct plaidml_get_in
{
    static T get(GPlaidMLContext &ctx, int idx) { return ctx.inArg<T>(idx); }
};

template<class T> struct plaidml_get_out;
template<> struct plaidml_get_out<cv::GMat>
{
    static plaidml::edsl::Tensor& get(GPlaidMLContext& ctx, int idx)
    {
        return ctx.outTensor(idx);
    }
};

template<typename, typename, typename>
struct PlaidMLCallHelper;

template<typename Impl, typename... Ins, typename... Outs>
struct PlaidMLCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...> >
{
    template<int... IIs, int... OIs>
    static void call_impl(GPlaidMLContext &ctx, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        Impl::run(plaidml_get_in<Ins>::get(ctx, IIs)..., plaidml_get_out<Outs>::get(ctx, OIs)...);
    }

    static void call(GPlaidMLContext& ctx)
    {
        call_impl(ctx,
                  typename detail::MkSeq<sizeof...(Ins)>::type(),
                  typename detail::MkSeq<sizeof...(Outs)>::type());
    }
};

} // namespace detail

template<class Impl, class K>
class GPlaidMLKernelImpl: public cv::detail::PlaidMLCallHelper<Impl, typename K::InArgs, typename K::OutArgs>,
                          public cv::detail::KernelTag
{
    using P = detail::PlaidMLCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

public:
    using API = K;

    static cv::gapi::GBackend backend()  { return cv::gapi::plaidml::backend(); }
    static cv::GPlaidMLKernel kernel()   { return GPlaidMLKernel(&P::call);     }
};

#define GAPI_PLAIDML_KERNEL(Name, API) struct Name: public cv::GPlaidMLKernelImpl<Name, API>

} // namespace cv

#endif // OPENCV_GAPI_GPLAIDMLKERNEL_HPP
