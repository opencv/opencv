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
#include <opencv2/gapi/gcomputation.hpp>

namespace cv
{

struct GAPI_EXPORTS GTransform
{
    // FIXME: consider another simplified
    // class instead of GComputation
    using F = std::function<GComputation()>;

    std::string description;
    F pattern;
    F substitute;

    GTransform(const std::string& d, const F &p, const F &s) : description(d), pattern(p), substitute(s) {}
};

namespace detail
{

template <typename, typename, typename>
struct TransHelper;

template <typename K, typename... Ins, typename Out>
struct TransHelper<K, std::tuple<Ins...>, Out>
{
    template <typename Callable, int... IIs, int... OIs>
    static GComputation invoke(Callable f, Seq<IIs...>, Seq<OIs...>)
    {
        const std::tuple<Ins...> ins;
        const auto r = tuple_wrap_helper<Out>::get(f(std::get<IIs>(ins)...));
        return GComputation(cv::GIn(std::get<IIs>(ins)...),
                            cv::GOut(std::get<OIs>(r)...));
    }

    static GComputation get_pattern()
    {
        return invoke(K::pattern, typename MkSeq<sizeof...(Ins)>::type(),
                      typename MkSeq<std::tuple_size<typename tuple_wrap_helper<Out>::type>::value>::type());
    }
    static GComputation get_substitute()
    {
        return invoke(K::substitute, typename MkSeq<sizeof...(Ins)>::type(),
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
