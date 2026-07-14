// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifdef HAVE_ONEVPL

#ifndef VPP_PREPROC_ENGINE
#define VPP_PREPROC_ENGINE
#include "streaming/onevpl/onevpl_export.hpp"
#include "streaming/onevpl/engine/engine_session.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct vpp_pp_params {
    vpp_pp_params() : handle(), info(), reserved() {}
    vpp_pp_params(mfxSession s, mfxFrameInfo i, void *r = nullptr) :
        handle(s), info(i), reserved(r) {}
    mfxSession handle;
    mfxFrameInfo info;
    void *reserved = nullptr;
};

struct vpp_pp_session {
    vpp_pp_session() : handle(), reserved() {}
    vpp_pp_session(std::shared_ptr<EngineSession> h, void *r = nullptr) :
        handle(h), reserved(r) {}
    std::shared_ptr<EngineSession> handle;
    void *reserved = nullptr;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // VPP_PREPROC_ENGINE
#endif // HAVE_ONEVPL
