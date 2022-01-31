// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_FP_CONTROL_UTILS_HPP
#define OPENCV_CORE_FP_CONTROL_UTILS_HPP

namespace cv {

namespace details {

struct FPDenormalsModeState
{
    uint32_t reserved[16];  // 64-bytes
};  // FPDenormalsModeState

CV_EXPORTS void setFPDenormalsIgnoreHint(bool ignore, CV_OUT FPDenormalsModeState& state);
CV_EXPORTS int saveFPDenormalsState(CV_OUT FPDenormalsModeState& state);
CV_EXPORTS bool restoreFPDenormalsState(const FPDenormalsModeState& state);

class FPDenormalsIgnoreHintScope
{
public:
    inline explicit FPDenormalsIgnoreHintScope(bool ignore = true)
    {
        details::setFPDenormalsIgnoreHint(ignore, saved_state);
    }

    inline explicit FPDenormalsIgnoreHintScope(const FPDenormalsModeState& state)
    {
        details::saveFPDenormalsState(saved_state);
        details::restoreFPDenormalsState(state);
    }

    inline ~FPDenormalsIgnoreHintScope()
    {
        details::restoreFPDenormalsState(saved_state);
    }

protected:
    FPDenormalsModeState saved_state;
};  // FPDenormalsIgnoreHintScope

class FPDenormalsIgnoreHintScopeNOOP
{
public:
    inline FPDenormalsIgnoreHintScopeNOOP(bool ignore = true) { CV_UNUSED(ignore); }
    inline FPDenormalsIgnoreHintScopeNOOP(const FPDenormalsModeState& state) { CV_UNUSED(state); }
    inline ~FPDenormalsIgnoreHintScopeNOOP() { }
};  // FPDenormalsIgnoreHintScopeNOOP

}  // namespace details


// Should depend on target compilation architecture only
// Note: previously added archs should NOT be removed to preserve ABI compatibility
#if defined(OPENCV_SUPPORTS_FP_DENORMALS_HINT)
  // preserve configuration overloading through ports
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_X64) || defined(_X86_)
typedef details::FPDenormalsIgnoreHintScope FPDenormalsIgnoreHintScope;
#define OPENCV_SUPPORTS_FP_DENORMALS_HINT 1
#else
#define OPENCV_SUPPORTS_FP_DENORMALS_HINT 0
typedef details::FPDenormalsIgnoreHintScopeNOOP FPDenormalsIgnoreHintScope;
#endif

}  // namespace cv

#endif // OPENCV_CORE_FP_CONTROL_UTILS_HPP
