#ifndef OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP
#define OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP

#include <stdio.h>

#include <memory>
#include <string>

#include "streaming/onevpl_priv_interface.hpp"
#include "streaming/vpl/vpl_source_engine.hpp"

#ifdef HAVE_ONEVPL
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
                           const CFGParams& params);

    ~VPLSourceImpl();

    static const CFGParams& getDefaultCfgParams();
    const CFGParams& getCfgParams() const;

private:
    
    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;

    VPLSourceImpl();

    DecoderParams create_decoder_from_file(const CFGParamValue& decoder, FILE* source_ptr);
    std::unique_ptr<VPLAccelerationPolicy> initializeHWAccel(mfxSession mfx_session);
    
    mfxLoader mfx_handle;
    mfxImplDescription *mfx_impl_desription;
    std::vector<mfxConfig> mfx_handle_configs;
    CFGParams cfg_params;

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
