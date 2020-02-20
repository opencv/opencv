#ifndef OPENCV_GAPI_GAPI_ITT_HPP
#define OPENCV_GAPI_GAPI_ITT_HPP

//for ITT_NAMED_TRACE_GUARD
#include <type_traits>
#include <memory>

#ifdef OPENCV_WITH_ITT
#include <ittnotify.h>
#endif

#include <opencv2/gapi/util/compiler_hints.hpp>
#include <iostream>
namespace cv {
namespace util {
    template< class T >
    using remove_reference_t = typename std::remove_reference<T>::type;

    //Home brew ScopeGuard
    //D will be called automatically with p as argument when ScopeGuard goes out of scope.
    //call release() on the ScopeGuard object to revoke guard action
    template<typename T, typename D>
    auto make_ptr_guard(T* p, D&& d) -> std::unique_ptr<T, util::remove_reference_t<D>>
    {
        return {p, std::forward<D>(d)};
    }
}  //namespace util

namespace gimpl { namespace parallel {
    #ifdef OPENCV_WITH_ITT
    extern const __itt_domain* gapi_itt_domain;

    namespace {
        auto make_itt_guard = [](__itt_string_handle* h){
           __itt_task_begin(gapi_itt_domain, __itt_null, __itt_null, (h));
           return util::make_ptr_guard(reinterpret_cast<int*>(1), [](int* ){ __itt_task_end(gapi_itt_domain); });
        };
    }  // namespace

    #define ITT_NAMED_TRACE_GUARD(name, h)  auto name =  cv::gimpl::parallel::make_itt_guard(h); cv::util::suppress_unused_warning(name)
    #else
    struct dumb_guard {void reset(){}};
    #define ITT_NAMED_TRACE_GUARD(name, h)  cv::gimpl::parallel::dumb_guard name; cv::util::suppress_unused_warning(name)
    #endif

    #define ITT_AUTO_TRACE_GUARD_IMPL_(LINE, h)        ITT_NAMED_TRACE_GUARD(itt_trace_guard_##LINE , h)
    #define ITT_AUTO_TRACE_GUARD_IMPL(LINE, h)         ITT_AUTO_TRACE_GUARD_IMPL_(LINE, h)
    #define ITT_AUTO_TRACE_GUARD(h)                    ITT_AUTO_TRACE_GUARD_IMPL(__LINE__ , h)
}} //gimpl::parallel
}  //namespace cv



#endif /* OPENCV_GAPI_GAPI_ITT_HPP */
