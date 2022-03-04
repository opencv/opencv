// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_ITT_HPP
#define OPENCV_GAPI_ITT_HPP

// for GAPI_ITT_NAMED_TRACE_GUARD
#include <type_traits>
#include <memory>

#include <opencv2/gapi/util/compiler_hints.hpp>

// NOTE: OPENCV_WITH_ITT is only defined if ITT dependecy is built by OpenCV infrastructure.
//       There will not be such define in G-API standalone mode.
// TODO: Consider using OpenCV's trace.hpp
#if defined(OPENCV_WITH_ITT)
#include <ittnotify.h>

namespace cv {
namespace util {
    template< class T >
    using remove_reference_t = typename std::remove_reference<T>::type;

    // Home brew ScopeGuard
    // D will be called automatically with p as argument when ScopeGuard goes out of scope.
    // call release() on the ScopeGuard object to revoke guard action
    template<typename T, typename D>
    auto make_ptr_guard(T* p, D&& d) -> std::unique_ptr<T, util::remove_reference_t<D>> {
        return {p, std::forward<D>(d)};
    }
}  // namespace util

namespace gimpl {
    extern __itt_domain* gapi_itt_domain;
    namespace {
        auto make_itt_guard = [](__itt_string_handle* h) {
           __itt_task_begin(gapi_itt_domain, __itt_null, __itt_null, (h));
           return util::make_ptr_guard(reinterpret_cast<int*>(1),
                                       [](int* ){ __itt_task_end(gapi_itt_domain); });
        };
    }  // namespace
} // namespace gimpl
} // namespace cv

#define GAPI_ITT_NAMED_TRACE_GUARD(name, h)      auto name = cv::gimpl::make_itt_guard(h); \
                                                 cv::util::suppress_unused_warning(name)
#define GAPI_ITT_STATIC_LOCAL_HANDLE_IMPL(n, h)  static __itt_string_handle* n = \
                                                 __itt_string_handle_create(h)
#define GAPI_ITT_DYNAMIC_LOCAL_HANDLE_IMPL(n, h) __itt_string_handle* n = \
                                                 __itt_string_handle_create(h)
#else // OPENCV_WITH_ITT

namespace cv {
namespace gimpl {
struct dumb_guard { void reset() { } };
} // namespace gimpl
} // namespace cv

#define GAPI_ITT_NAMED_TRACE_GUARD(name, h)      cv::gimpl::dumb_guard name; \
                                                 cv::util::suppress_unused_warning(name); \
                                                 cv::util::suppress_unused_warning(h)
#define GAPI_ITT_STATIC_LOCAL_HANDLE_IMPL(n, h)  static auto n = h
#define GAPI_ITT_DYNAMIC_LOCAL_HANDLE_IMPL(n, h) auto n = h

#endif // OPENCV_WITH_ITT

#define GAPI_ITT_AUTO_TRACE_GUARD_IMPL_(LINE, h) GAPI_ITT_NAMED_TRACE_GUARD( \
                                                    itt_trace_guard_##LINE, h)
#define GAPI_ITT_AUTO_TRACE_GUARD_IMPL(LINE, h)  GAPI_ITT_AUTO_TRACE_GUARD_IMPL_(LINE, h)
#define GAPI_ITT_AUTO_TRACE_GUARD(h)             GAPI_ITT_AUTO_TRACE_GUARD_IMPL(__LINE__, h)

#define GAPI_ITT_STATIC_LOCAL_HANDLE(n, h)       GAPI_ITT_STATIC_LOCAL_HANDLE_IMPL(n, h)
#define GAPI_ITT_DYNAMIC_LOCAL_HANDLE(n, h)      GAPI_ITT_DYNAMIC_LOCAL_HANDLE_IMPL(n, h)

#endif // OPENCV_GAPI_ITT_HPP
