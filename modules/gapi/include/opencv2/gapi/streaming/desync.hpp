// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_GSTREAMING_DESYNC_HPP
#define OPENCV_GAPI_GSTREAMING_DESYNC_HPP

#include <vector>

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

        template<typename... R, int... IIs>
        static std::tuple<R...> yield(cv::GCall &call, cv::detail::Seq<IIs...>) {
            // An universal yield for desync.
            // Yields output objects according to the input Types...
            // Reuses gkernel machinery.
            // FIXME: This function can be generic and declared in gkernel.hpp
            //        (it is there already, but a part of GKenrelType[M]
            return std::make_tuple(cv::detail::Yield<R>::yield(call, IIs)...);
        }
    };
} // namespace detail

/**
 * @brief Starts a desynchronized branch in the graph.
 *
 * This operation takes an arbitrary number of G-API data objects
 * and returns a tuple of duplicates of that objects.
 *
 * Operations which use these data objects now may run in a desynchronized
 * fashion from the rest of the graph.
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
 */
template<typename... Args>
std::tuple<typename std::decay<Args>::type...> desync(Args&&... args)
{
    cv::GKernel k{
          detail::GDesync::id() // kernel id
        , ""                    // kernel tag
        , [](const GMetaArgs &a, const GArgs &) {return a;} // outMeta callback
        , {cv::detail::GTypeTraits<typename std::decay<Args>::type>::shape...} // out Shapes
    };
    cv::GCall call(std::move(k));
    call.pass(args...);
    return detail::GDesync::yield<typename std::decay<Args>::type...>
        (call, typename cv::detail::MkSeq<sizeof...(Args)>::type());
}

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_DESYNC_HPP
