// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#ifndef __IPP_HAL_UTILS_HPP__
#define __IPP_HAL_UTILS_HPP__

#include "ippversion.h"
#ifndef IPP_VERSION_UPDATE // prior to 7.1
#define IPP_VERSION_UPDATE 0
#endif

#define IPP_VERSION_X100 (IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR*10 + IPP_VERSION_UPDATE)

#ifdef HAVE_IPP_ICV
# define ICV_BASE
#if IPP_VERSION_X100 >= 201700
# include "ippicv.h"
#else
# include "ipp.h"
#endif
#else
# include "ipp.h"
#endif

// IPP call instrumentation for the standalone IPP HAL.
//
// The HAL is built as a separate static library that does not include the
// internal core/private.hpp, so by default its IPP calls never appear in the
// cv::instr trace (reported IPP weight = 0%). When the parent build enables
// ENABLE_INSTRUMENTATION, the shared instrumentation.private.hpp provides the
// exact same CV_INSTRUMENT_FUN_IPP macro used by core (backed by symbols exported
// from libopencv_core, resolved at link time), so HAL IPP calls are traced
// identically to in-module ones. Otherwise the no-op passthrough is used.
#if !defined(CV_INSTRUMENT_FUN_IPP)
#if defined(ENABLE_INSTRUMENTATION)
#include "opencv2/core/utils/instrumentation.private.hpp"
#else
#define CV_INSTRUMENT_FUN_IPP(FUN, ...) ((FUN)(__VA_ARGS__))
#endif // defined(ENABLE_INSTRUMENTATION)
#endif // !defined(CV_INSTRUMENT_FUN_IPP)

#define CV_HAL_CHECK_USE_IPP() if(!cv::ipp::useIPP()) return CV_HAL_ERROR_NOT_IMPLEMENTED;

#endif
