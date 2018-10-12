// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPUTATION_HPP
#define OPENCV_GAPI_GCOMPUTATION_HPP

#include <functional>

#include "opencv2/gapi/util/util.hpp"
#include "opencv2/gapi/gcommon.hpp"
#include "opencv2/gapi/gproto.hpp"
#include "opencv2/gapi/garg.hpp"
#include "opencv2/gapi/gcompiled.hpp"

namespace cv {

namespace detail
{
    // FIXME: move to algorithm, cover with separate tests
    // FIXME: replace with O(1) version (both memory and compilation time)
    template<typename...>
    struct last_type;

    template<typename T>
    struct last_type<T> { using type = T;};

    template<typename T, typename... Ts>
    struct last_type<T, Ts...> { using type = typename last_type<Ts...>::type; };

    template<typename... Ts>
    using last_type_t = typename last_type<Ts...>::type;
}

class GAPI_EXPORTS GComputation
{
public:
    class Priv;
    typedef std::function<GComputation()> Generator;

    // Various constructors enable different ways to define a computation: /////
    // 1. Generic constructors
    GComputation(const Generator& gen);                // Generator overload
    GComputation(GProtoInputArgs &&ins,
                 GProtoOutputArgs &&outs);             // Arg-to-arg overload

    // 2. Syntax sugar and compatibility overloads
    GComputation(GMat in, GMat out);                   // Unary overload
    GComputation(GMat in, GScalar out);                // Unary overload (scalar)
    GComputation(GMat in1, GMat in2, GMat out);        // Binary overload
    GComputation(GMat in1, GMat in2, GScalar out);     // Binary overload (scalar)
    GComputation(const std::vector<GMat> &ins,         // Compatibility overload
                 const std::vector<GMat> &outs);

    // Various versions of apply(): ////////////////////////////////////////////
    // 1. Generic apply()
    void apply(GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args = {});       // Arg-to-arg overload
    void apply(const std::vector<cv::gapi::own::Mat>& ins,                        // Compatibility overload
               const std::vector<cv::gapi::own::Mat>& outs,
               GCompileArgs &&args = {});

    // 2. Syntax sugar and compatibility overloads
#if !defined(GAPI_STANDALONE)
    void apply(cv::Mat in, cv::Mat &out, GCompileArgs &&args = {});               // Unary overload
    void apply(cv::Mat in, cv::Scalar &out, GCompileArgs &&args = {});            // Unary overload (scalar)
    void apply(cv::Mat in1, cv::Mat in2, cv::Mat &out, GCompileArgs &&args = {}); // Binary overload
    void apply(cv::Mat in1, cv::Mat in2, cv::Scalar &out, GCompileArgs &&args = {}); // Binary overload (scalar)
    void apply(const std::vector<cv::Mat>& ins,         // Compatibility overload
               const std::vector<cv::Mat>& outs,
               GCompileArgs &&args = {});
#endif // !defined(GAPI_STANDALONE)
    // Various versions of compile(): //////////////////////////////////////////
    // 1. Generic compile() - requires metas to be passed as vector
    GCompiled compile(GMetaArgs &&in_metas, GCompileArgs &&args = {});

    // 2. Syntax sugar - variadic list of metas, no extra compile args
    template<typename... Ts>
    auto compile(const Ts&... metas) ->
        typename std::enable_if<detail::are_meta_descrs<Ts...>::value, GCompiled>::type
    {
        return compile(GMetaArgs{GMetaArg(metas)...}, GCompileArgs());
    }

    // 3. Syntax sugar - variadic list of metas, extra compile args
    // (seems optional parameters don't work well when there's an variadic template
    // comes first)
    //
    // Ideally it should look like:
    //
    //     template<typename... Ts>
    //     GCompiled compile(const Ts&... metas, GCompileArgs &&args)
    //
    // But not all compilers can hande this (and seems they shouldn't be able to).
    template<typename... Ts>
    auto compile(const Ts&... meta_and_compile_args) ->
        typename std::enable_if<detail::are_meta_descrs_but_last<Ts...>::value
                                && std::is_same<GCompileArgs, detail::last_type_t<Ts...> >::value,
                                GCompiled>::type
    {
        //FIXME: wrapping meta_and_compile_args into a tuple to unwrap them inside a helper function is the overkill
        return compile(std::make_tuple(meta_and_compile_args...),
                       typename detail::MkSeq<sizeof...(Ts)-1>::type());
    }

    // Internal use only
    Priv& priv();
    const Priv& priv() const;

protected:

    // 4. Helper method for (3)
    template<typename... Ts, int... IIs>
    GCompiled compile(const std::tuple<Ts...> &meta_and_compile_args, detail::Seq<IIs...>)
    {
        GMetaArgs meta_args = {GMetaArg(std::get<IIs>(meta_and_compile_args))...};
        GCompileArgs comp_args = std::get<sizeof...(Ts)-1>(meta_and_compile_args);
        return compile(std::move(meta_args), std::move(comp_args));
    }

    std::shared_ptr<Priv> m_priv;
};

namespace gapi
{
    // Declare an Island tagged with `name` and defined from `ins` to `outs`
    // (exclusively, as ins/outs are data objects, and regioning is done on
    // operations level).
    // Throws if any operation between `ins` and `outs` are already assigned
    // to another island.
    void GAPI_EXPORTS island(const std::string &name,
                           GProtoInputArgs  &&ins,
                           GProtoOutputArgs &&outs);
} // namespace gapi

} // namespace cv
#endif // OPENCV_GAPI_GCOMPUTATION_HPP
