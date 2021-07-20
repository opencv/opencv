#ifndef OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP
#define OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP

#include <stdio.h>

#include <memory>
#include <string>

#include "streaming/onevpl_priv_interface.hpp"

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>


namespace cv {
namespace gapi {
namespace wip {

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
    mfxLoader mfx_handle;
    std::vector<mfxConfig> mfx_handle_configs;
    mfxSession mfx_session;

    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;
    file_ptr source_handle;
    
    std::string filePath;
    CFGParams cfg_params;

    
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_ONEVPLSOURCE_IMPL_HPP
