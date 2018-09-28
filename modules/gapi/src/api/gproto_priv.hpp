// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GPROTO_PRIV_HPP
#define OPENCV_GAPI_GPROTO_PRIV_HPP

#include "opencv2/gapi/gproto.hpp"
#include "opencv2/gapi/garg.hpp"

#include "api/gapi_priv.hpp"

namespace cv {
namespace gimpl {
namespace proto {

// These methods are used by GModelBuilder only
// FIXME: Document semantics

// FIXME: GAPI_EXPORTS because of tests only!
// FIXME: Possible dangling reference alert!!!
GAPI_EXPORTS const GOrigin& origin_of (const GProtoArg &arg);
GAPI_EXPORTS const GOrigin& origin_of (const GArg      &arg);

bool           is_dynamic(const GArg      &arg);
GProtoArg      rewrap    (const GArg      &arg);

} // proto
} // gimpl
} // cv

#endif // OPENCV_GAPI_GPROTO_PRIV_HPP
