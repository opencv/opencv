// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifdef HAVE_ONEVPL

#ifndef VPP_PREPROC_ENGINE
#define VPP_PREPROC_ENGINE
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
struct vpp_pp_params {
    mfxSession handle;
    mfxFrameInfo info;
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // VPP_PREPROC_ENGINE
#endif // HAVE_ONEVPL
