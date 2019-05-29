// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GTRANSFORM_HPP
#define OPENCV_GAPI_GTRANSFORM_HPP

#include <functional>
#include <type_traits>
#include <utility>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/util.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gtype_traits.hpp>
#include <opencv2/gapi/util/compiler_hints.hpp>

namespace cv
{

struct GAPI_EXPORTS GTransform
{
    using F = std::function<GArgs(const GArgs &)>;

    const char *description;
    F pattern;
    F substitute;

    explicit GTransform(const char *d, const F &p, const F &s) : description(d), pattern(p), substitute(s){};
};

template <class Impl>
GTransform transformation()
{
    return GTransform(Impl::descr(), &Impl::get_pattern, &Impl::get_substitute);
}

////////////////////////////////////////////////////////////////////

template <typename, typename, typename>
struct TransHelper;

// FIXME: code duplication:
// consider better approach like in compound kernels with tuple wrappers and context class

template <typename K, typename... Ins, typename Out>
struct TransHelper<K, std::tuple<Ins...>, Out>
{
    // FIXME: code duplication
    template <int... IIs>
    static GArgs get_pattern_impl(const GArgs &in_args, detail::Seq<IIs...>)
    {
        const Out r = K::pattern(in_args.at(IIs).template get<Ins>()...);
        return GArgs{GArg(r)};
    }

    template <int... IIs>
    static GArgs get_substitute_impl(const GArgs &in_args, detail::Seq<IIs...>)
    {
        const Out r = K::pattern(in_args.at(IIs).template get<Ins>()...);
        return GArgs{GArg(r)};
    }

    static GArgs get_pattern(const GArgs &in_args)
    {
        return get_pattern_impl(in_args, typename detail::MkSeq<sizeof...(Ins)>::type());
    }
    static GArgs get_substitute(const GArgs &in_args)
    {
        return get_substitute_impl(in_args, typename detail::MkSeq<sizeof...(Ins)>::type());
    }
};

template <typename K, typename... Ins, typename... Outs>
struct TransHelper<K, std::tuple<Ins...>, std::tuple<Outs...>>
{
    // FIXME: code duplication
    template <int... IIs, int... OIs>
    static GArgs get_pattern_impl(const GArgs &in_args, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        using R = std::tuple<Outs...>;
        const R r = K::pattern(in_args.at(IIs).template get<Ins>()...);
        return GArgs{GArg(std::get<OIs>(r))...};
    }

    template <int... IIs, int... OIs>
    static GArgs get_substitute_impl(const GArgs &in_args, detail::Seq<IIs...>, detail::Seq<OIs...>)
    {
        using R = std::tuple<Outs...>;
        const R r = K::pattern(in_args.at(IIs).template get<Ins>()...);
        return GArgs{GArg(std::get<OIs>(r))...};
    }

    static GArgs get_pattern(const GArgs &in_args)
    {
        return get_pattern_impl(in_args, typename detail::MkSeq<sizeof...(Ins)>::type(),
                                typename detail::MkSeq<sizeof...(Outs)>::type());
    }
    static GArgs get_substitute(const GArgs &in_args)
    {
        return get_substitute_impl(in_args, typename detail::MkSeq<sizeof...(Ins)>::type(),
                                   typename detail::MkSeq<sizeof...(Outs)>::type());
    }
};

/////////////////////////////////////////////////////////////////////

template <typename, typename>
class GTransformImpl;

template <typename K, typename R, typename... Args>
class GTransformImpl<K, std::function<R(Args...)>> : public TransHelper<K, std::tuple<Args...>, R>,
                                                     public cv::detail::TransformTag
{
public:
    // FIXME: currently there is no check that transformations' signatures are unique
    // and won't be any intersection in graph compilation stage
    using API = K;
};

template <typename, typename>
class GTransformImplM;

template <typename K, typename... R, typename... Args>
class GTransformImplM<K, std::function<std::tuple<R...>(Args...)>> : public TransHelper<K, std::tuple<Args...>, std::tuple<R...>>,
                                                                     public cv::detail::TransformTag
{
public:
    // FIXME: currently there is no check that transformations' signatures are unique
    // and won't be any intersection in graph compilation stage
    using API = K;
};

//////////////////////////////////////////////////////////////////////

#define G_DESCR_HELPER_CLASS(Class) Class##DescrHelper

#define G_DESCR_HELPER_BODY(Class, Descr)                       \
    namespace detail                                            \
    {                                                           \
    struct G_DESCR_HELPER_CLASS(Class)                          \
    {                                                           \
        static constexpr const char *descr() { return Descr; }; \
    };                                                          \
    }

#define GAPI_TRANSFORM(Class, API, Descr)                                     \
    G_DESCR_HELPER_BODY(Class, Descr)                                         \
    struct Class final : public cv::GTransformImpl<Class, std::function API>, \
                         public detail::G_DESCR_HELPER_CLASS(Class)

#define GAPI_TRANSFORM_M(Class, API, Descr)                                    \
    G_DESCR_HELPER_BODY(Class, Descr)                                          \
    struct Class final : public cv::GTransformImplM<Class, std::function API>, \
                         public detail::G_DESCR_HELPER_CLASS(Class)

} // namespace cv

#endif // OPENCV_GAPI_GTRANSFORM_HPP
