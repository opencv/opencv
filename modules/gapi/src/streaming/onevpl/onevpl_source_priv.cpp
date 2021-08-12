// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <algorithm>
#include <sstream>

#include "streaming/onevpl/onevpl_source_priv.hpp"
#include "logger.hpp"

#ifndef HAVE_ONEVPL
namespace cv {
namespace gapi {
namespace wip {
bool OneVPLSource::Priv::pull(cv::gapi::wip::Data&) {
    return true;
}
GMetaArg OneVPLSource::Priv::descr_of() const {
    return {};
}
} // namespace wip
} // namespace gapi
} // namespace cv

#else // HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
OneVPLSource::Priv::Priv() :
    mfx_handle(MFXLoad())
{
    GAPI_LOG_INFO(nullptr, "Initialized MFX handle: " << mfx_handle);
    description_is_valid = false;
}

OneVPLSource::Priv::Priv(std::shared_ptr<IDataProvider>, const std::vector<oneVPL_cfg_param>&) :
    OneVPLSource::Priv()
{
}

OneVPLSource::Priv::~Priv()
{
    GAPI_LOG_INFO(nullptr, "Unload MFX handle: " << mfx_handle);
    MFXUnload(mfx_handle);
}

bool OneVPLSource::Priv::pull(cv::gapi::wip::Data&)
{
    return false;
}

GMetaArg OneVPLSource::Priv::descr_of() const
{
    return {};
}
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
