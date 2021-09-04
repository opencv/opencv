// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <algorithm>
#include <sstream>

#include "streaming/onevpl/source_priv.hpp"
#include "logger.hpp"

#ifndef HAVE_ONEVPL
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
bool GSource::Priv::pull(cv::gapi::wip::Data&) {
    return true;
}
GMetaArg GSource::Priv::descr_of() const {
    return {};
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#else // HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
GSource::Priv::Priv() :
    mfx_handle(MFXLoad())
{
    GAPI_LOG_INFO(nullptr, "Initialized MFX handle: " << mfx_handle);
    description_is_valid = false;
}

GSource::Priv::Priv(std::shared_ptr<IDataProvider>, const std::vector<CfgParam>&) :
    GSource::Priv()
{
}

GSource::Priv::~Priv()
{
    GAPI_LOG_INFO(nullptr, "Unload MFX handle: " << mfx_handle);
    MFXUnload(mfx_handle);
}

bool GSource::Priv::pull(cv::gapi::wip::Data&)
{
    return false;
}

GMetaArg GSource::Priv::descr_of() const
{
    return {};
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
