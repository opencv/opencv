// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#ifndef OPENCV_CORE_UTILS_INSTRUMENTATION_PRIVATE_HPP
#define OPENCV_CORE_UTILS_INSTRUMENTATION_PRIVATE_HPP

#include "opencv2/core/utils/instrumentation.hpp"

// Region-based function-instrumentation primitives.
//
// Only meaningful when the build was configured with ENABLE_INSTRUMENTATION;
// otherwise this header contributes nothing and each includer provides its own
// no-op fallbacks. The IntrumentationRegion / getCurrentNode() definitions are
// compiled into and exported (CV_EXPORTS) from the core library, so the HAL only
// needs these declarations and resolves the symbols at link time.

#ifdef ENABLE_INSTRUMENTATION

namespace cv { namespace instr {

// Scoped region: records one instrumentation node on construction and closes it
// on destruction. Defined in modules/core/src/system.cpp.
class CV_EXPORTS IntrumentationRegion
{
public:
    IntrumentationRegion(const char* funName, const char* fileName, int lineNum, void *retAddress,
                         bool alwaysExpand, TYPE instrType = TYPE_GENERAL, IMPL implType = IMPL_PLAIN);
    ~IntrumentationRegion();

private:
    bool    m_disabled; // region status
    uint64  m_regionTicks;
};

CV_EXPORTS InstrNode*   getCurrentNode();

}} // namespace cv::instr

// Instrumentation information marker
#define CV_INSTRUMENT_MARK_META(IMPL, NAME, ...) {::cv::instr::IntrumentationRegion __instr_mark__(NAME, __FILE__, __LINE__, NULL, false, ::cv::instr::TYPE_MARKER, IMPL);}

// Instrument functions with non-void return type
#define CV_INSTRUMENT_FUN_RT_META(TYPE, IMPL, ERROR_COND, FUN, ...) ([&]()                              \
{                                                                                                       \
    if(::cv::instr::useInstrumentation()){                                                              \
        ::cv::instr::IntrumentationRegion __instr__(#FUN, __FILE__, __LINE__, NULL, false, TYPE, IMPL); \
        try{                                                                                            \
            auto instrStatus = ((FUN)(__VA_ARGS__));                                                    \
            if(ERROR_COND){                                                                             \
                ::cv::instr::getCurrentNode()->m_payload.m_funError = true;                             \
                CV_INSTRUMENT_MARK_META(IMPL, #FUN " - BadExit");                                       \
            }                                                                                           \
            return instrStatus;                                                                         \
        }catch(...){                                                                                    \
            ::cv::instr::getCurrentNode()->m_payload.m_funError = true;                                 \
            CV_INSTRUMENT_MARK_META(IMPL, #FUN " - BadExit");                                           \
            throw;                                                                                      \
        }                                                                                               \
    }else{                                                                                              \
        return ((FUN)(__VA_ARGS__));                                                                    \
    }                                                                                                   \
}())
// Instrument functions with void return type
#define CV_INSTRUMENT_FUN_RV_META(TYPE, IMPL, FUN, ...) ([&]()                                          \
{                                                                                                       \
    if(::cv::instr::useInstrumentation()){                                                              \
        ::cv::instr::IntrumentationRegion __instr__(#FUN, __FILE__, __LINE__, NULL, false, TYPE, IMPL); \
        try{                                                                                            \
            (FUN)(__VA_ARGS__);                                                                         \
        }catch(...){                                                                                    \
            ::cv::instr::getCurrentNode()->m_payload.m_funError = true;                                 \
            CV_INSTRUMENT_MARK_META(IMPL, #FUN " - BadExit");                                           \
            throw;                                                                                      \
        }                                                                                               \
    }else{                                                                                              \
        (FUN)(__VA_ARGS__);                                                                             \
    }                                                                                                   \
}())

// IPP function instrumentation macros
#define CV_INSTRUMENT_FUN_IPP(FUN, ...)     CV_INSTRUMENT_FUN_RT_META(::cv::instr::TYPE_FUN, ::cv::instr::IMPL_IPP, instrStatus < 0, FUN, __VA_ARGS__)
#define CV_INSTRUMENT_MARK_IPP(NAME)        CV_INSTRUMENT_MARK_META(::cv::instr::IMPL_IPP, NAME)

#endif // ENABLE_INSTRUMENTATION

#endif // OPENCV_CORE_UTILS_INSTRUMENTATION_PRIVATE_HPP
