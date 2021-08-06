// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_VPL_ENGINE_HPP
#define OPENCV_GAPI_STREAMING_VPL_ENGINE_HPP
#include <memory>

#include "streaming/engine/base_engine.hpp"

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

class DecodeSession;
struct IDataProvider;

class VPLDecodeEngine : public VPLProcessingEngine {
public:
    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;

    VPLDecodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel);
    void initialize_session(mfxSession mfx_session,
                            DecoderParams&& decoder_param,
                            std::shared_ptr<IDataProvider> provider) override;

private:
    ExecutionStatus execute_op(operation_t& op, EngineSession& sess) override;
    ExecutionStatus process_error(mfxStatus status, DecodeSession& sess);

    void on_frame_ready(DecodeSession& sess);
};


class DecodeSession : public EngineSession
{
public:
    friend class VPLDecodeEngine;

    DecodeSession(mfxSession sess, mfxBitstream&& str, std::shared_ptr<IDataProvider> provider);
    using EngineSession::EngineSession;

    std::shared_ptr<IDataProvider> data_provider;
    bool stop_processing;
    mfxFrameSurface1 *dec_surface_out;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_VPL_ENGINE_HPP
