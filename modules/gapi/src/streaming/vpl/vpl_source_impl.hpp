// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP
#define OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP

#include <stdio.h>

#include <memory>
#include <string>

#include "streaming/onevpl_priv_interface.hpp"

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif // MFX_VERSION

#include <vpl/mfx.h>

#include "streaming/vpl/vpl_source_engine.hpp"
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

struct VPLAccelerationPolicy;
class VPLDecodeEngine;
class VPLSourceImpl : public OneVPLCapture::IPriv
{
public:
    explicit VPLSourceImpl(const std::string& filePath,
                           const std::vector<oneVPL_cfg_param>& params);

    ~VPLSourceImpl();

    static const std::vector<oneVPL_cfg_param>& getDefaultCfgParams();
    const std::vector<oneVPL_cfg_param>& getCfgParams() const;

private:
    
    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;

    VPLSourceImpl();

    DecoderParams create_decoder_from_file(const oneVPL_cfg_param& decoder, FILE* source_ptr);
    std::unique_ptr<VPLAccelerationPolicy> initializeHWAccel();
    
    mfxLoader mfx_handle;
    mfxImplDescription *mfx_impl_desription;
    std::vector<mfxConfig> mfx_handle_configs;
    std::vector<oneVPL_cfg_param> cfg_params;

    mfxSession mfx_session;

    cv::GFrameDesc description;
    bool description_is_valid;
private:
    std::unique_ptr<VPLProcessingEngine> engine;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP
