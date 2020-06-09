// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


// FIXME: OpenCV license header

#include "test_precomp.hpp"

#if defined(USE_GAPI_TBB_EXECUTOR)
#include "tbb/parallel_invoke.h"
#endif

bool opencv_test::isExactExceptionPropogationEnabled(){
#if defined(USE_GAPI_TBB_EXECUTOR)
    struct custom_exception {};
    auto detect_exact_exception_propogation = []()->bool {
        try {
            auto throw_exception = []{ throw custom_exception{};};
            tbb::parallel_invoke(throw_exception, throw_exception);
            CV_Assert(false);
            return false;
        }
        catch (custom_exception) {
            return true;
        }
        catch (...) {
            return false;
        }
    };
    static bool enabled = detect_exact_exception_propogation();
    return enabled;
#else
    return true;
#endif
//    return false;
}
CV_TEST_MAIN("gapi")
