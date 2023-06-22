// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_GSTREAMING_META_HPP
#define OPENCV_GAPI_GSTREAMING_META_HPP

#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gcall.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

namespace cv {
namespace gapi {
namespace streaming {

// FIXME: the name is debatable
namespace meta_tag {
static constexpr const char * timestamp = "org.opencv.gapi.meta.timestamp";
static constexpr const char * seq_id    = "org.opencv.gapi.meta.seq_id";
} // namespace meta_tag

namespace detail {
struct GMeta {
    static const char *id() {
        return "org.opencv.streaming.meta";
    }
    // A universal yield for meta(), same as in GDesync
    template<typename... R, int... IIs>
    static std::tuple<R...> yield(cv::GCall &call, cv::detail::Seq<IIs...>) {
        return std::make_tuple(cv::detail::Yield<R>::yield(call, IIs)...);
    }
    // Also a universal outMeta stub here
    static GMetaArgs getOutMeta(const GMetaArgs &args, const GArgs &) {
        return args;
    }
};
} // namespace detail

template<typename T, typename G>
cv::GOpaque<T> meta(G g, const std::string &tag) {
    using O = cv::GOpaque<T>;
    cv::GKernel k{
          detail::GMeta::id()                    // kernel id
        , tag                                    // kernel tag. Use meta tag here
        , &detail::GMeta::getOutMeta             // outMeta callback
        , {cv::detail::GTypeTraits<O>::shape}    // output Shape
        , {cv::detail::GTypeTraits<G>::op_kind}  // input data kinds
        , {cv::detail::GObtainCtor<O>::get()}    // output template ctors
        , {cv::detail::GTypeTraits<O>::op_kind}  // output data kind
    };
    cv::GCall call(std::move(k));
    call.pass(g);
    return std::get<0>(detail::GMeta::yield<O>(call, cv::detail::MkSeq<1>::type()));
}

template<typename G>
cv::GOpaque<int64_t> timestamp(G g) {
    return meta<int64_t>(g, meta_tag::timestamp);
}

template<typename G>
cv::GOpaque<int64_t> seq_id(G g) {
    return meta<int64_t>(g, meta_tag::seq_id);
}

template<typename G>
cv::GOpaque<int64_t> seqNo(G g) {
    // Old name, compatibility only
    return seq_id(g);
}

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_META_HPP
