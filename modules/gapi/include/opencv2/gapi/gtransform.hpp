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

    std::string description;
    F pattern;
    F substitute;

    GTransform(const std::string& d, const F &p, const F &s) : description(d), pattern(p), substitute(s){};
};

namespace detail
{

template <typename, typename, typename>
struct TransHelper;

template <typename K, typename... Ins, typename Out>
struct TransHelper<K, std::tuple<Ins...>, Out>
{
    template <typename Callable, int... IIs, int... OIs>
    static GArgs invoke(Callable f, const GArgs &in_args, Seq<IIs...>, Seq<OIs...>)
    {
        const auto r = tuple_wrap_helper<Out>::get(f(in_args.at(IIs).template get<Ins>()...));
        return GArgs{GArg(std::get<OIs>(r))...};
    }

    static GArgs get_pattern(const GArgs &in_args)
    {
        return invoke(K::pattern, in_args, typename MkSeq<sizeof...(Ins)>::type(),
                      typename MkSeq<std::tuple_size<typename tuple_wrap_helper<Out>::type>::value>::type());
    }
    static GArgs get_substitute(const GArgs &in_args)
    {
        return invoke(K::substitute, in_args, typename MkSeq<sizeof...(Ins)>::type(),
                      typename MkSeq<std::tuple_size<typename tuple_wrap_helper<Out>::type>::value>::type());
    }
};
} // namespace detail

template <typename, typename>
class GTransformImpl;

template <typename K, typename R, typename... Args>
class GTransformImpl<K, std::function<R(Args...)>> : public cv::detail::TransHelper<K, std::tuple<Args...>, R>,
                                                     public cv::detail::TransformTag
{
public:
    // FIXME: currently there is no check that transformations' signatures are unique
    // and won't be any intersection in graph compilation stage
    using API = K;

    static GTransform transformation()
    {
        return GTransform(K::descr(), &K::get_pattern, &K::get_substitute);
    }
};
} // namespace cv

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

#endif // OPENCV_GAPI_GTRANSFORM_HPP
