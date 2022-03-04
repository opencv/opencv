// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation


#ifndef OPENCV_GAPI_GSTREAMING_DESYNC_HPP
#define OPENCV_GAPI_GSTREAMING_DESYNC_HPP

#include <tuple>

#include <opencv2/gapi/util/util.hpp>
#include <opencv2/gapi/gtype_traits.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gcall.hpp>
#include <opencv2/gapi/gkernel.hpp>

namespace cv {
namespace gapi {
namespace streaming {

namespace detail {
struct GDesync {
    static const char *id() {
        return "org.opencv.streaming.desync";
    }

    // An universal yield for desync.
    // Yields output objects according to the input Types...
    // Reuses gkernel machinery.
    // FIXME: This function can be generic and declared in gkernel.hpp
    //        (it is there already, but a part of GKernelType[M]
    template<typename... R, int... IIs>
    static std::tuple<R...> yield(cv::GCall &call, cv::detail::Seq<IIs...>) {
        return std::make_tuple(cv::detail::Yield<R>::yield(call, IIs)...);
    }
};

template<typename G>
G desync(const G &g) {
    cv::GKernel k{
          GDesync::id()                                     // kernel id
        , ""                                                // kernel tag
        , [](const GMetaArgs &a, const GArgs &) {return a;} // outMeta callback
        , {cv::detail::GTypeTraits<G>::shape}               // output Shape
        , {cv::detail::GTypeTraits<G>::op_kind}             // input data kinds
        , {cv::detail::GObtainCtor<G>::get()}               // output template ctors
    };
    cv::GCall call(std::move(k));
    call.pass(g);
    return std::get<0>(GDesync::yield<G>(call, cv::detail::MkSeq<1>::type()));
}
} // namespace detail

/**
 * @brief Starts a desynchronized branch in the graph.
 *
 * This operation takes a single G-API data object and returns a
 * graph-level "duplicate" of this object.
 *
 * Operations which use this data object can be desynchronized
 * from the rest of the graph.
 *
 * This operation has no effect when a GComputation is compiled with
 * regular cv::GComputation::compile(), since cv::GCompiled objects
 * always produce their full output vectors.
 *
 * This operation only makes sense when a GComputation is compiled in
 * straming mode with cv::GComputation::compileStreaming(). If this
 * operation is used and there are desynchronized outputs, the user
 * should use a special version of cv::GStreamingCompiled::pull()
 * which produces an array of cv::util::optional<> objects.
 *
 * @note This feature is highly experimental now and is currently
 * limited to a single GMat/GFrame argument only.
 */
GAPI_EXPORTS GMat desync(const GMat &g);
GAPI_EXPORTS GFrame desync(const GFrame &f);

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_DESYNC_HPP
