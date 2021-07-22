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

    void initializeHWAccel();
private:
    
    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;

    VPLSourceImpl();

    mfxBitstream create_decoder_from_file(const CFGParamValue& decoder, FILE* source_ptr);

    mfxLoader mfx_handle;
    std::vector<mfxConfig> mfx_handle_configs;
    CFGParams cfg_params;

    mfxSession mfx_session;
    std::unique_ptr<VPLAccelerationPolicy> accel_policy;


    cv::Mat first_frame;
private:
    std::unique_ptr<VPLDecodeEngine> engine;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP
