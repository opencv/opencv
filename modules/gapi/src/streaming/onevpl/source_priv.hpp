// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_PRIV_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_PRIV_HPP

#include <stdio.h>

#include <memory>
#include <string>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#include "streaming/onevpl/engine/processing_engine_base.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct VPLAccelerationPolicy;
class ProcessingEngineBase;

struct GSource::Priv
{
    explicit Priv(std::shared_ptr<IDataProvider> provider,
                  const std::vector<CfgParam>& params,
                  std::shared_ptr<IDeviceSelector> selector);
    ~Priv();

    static const std::vector<CfgParam>& getDefaultCfgParams();
    const std::vector<CfgParam>& getCfgParams() const;

    bool pull(cv::gapi::wip::Data& data);
    GMetaArg descr_of() const;
private:
    Priv();
    std::unique_ptr<VPLAccelerationPolicy> initializeHWAccel(std::shared_ptr<IDeviceSelector> selector);

    // TODO not it is global variable. Waiting for FIX issue with CloneSession
    // mfxLoader mfx_handle;
    mfxImplDescription *mfx_impl_description;
    std::vector<mfxConfig> mfx_handle_configs;
    std::vector<CfgParam> cfg_params;

    mfxSession mfx_session;

    cv::GFrameDesc description;
    bool description_is_valid;

    std::unique_ptr<ProcessingEngineBase> engine;

    size_t consumed_frames_count;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#else // HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct GSource::Priv final
{
    bool pull(cv::gapi::wip::Data&);
    GMetaArg descr_of() const;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_PRIV_HPP
