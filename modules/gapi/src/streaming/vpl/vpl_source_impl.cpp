#ifdef HAVE_ONEVPL

#include <sstream>

#ifdef HAVE_DIRECTX
//#include "directx.inc.hpp"
    #ifdef HAVE_D3D11
        #define D3D11_NO_HELPERS
        #include <d3d11.h>
        #include <codecvt>
        #include "opencv2/core/directx.hpp"
        #ifdef HAVE_OPENCL
            #include <CL/cl_d3d11.h>
        #endif
    #endif // HAVE_D3D11

#endif // HAVE_DIRECTX


#include "streaming/vpl/vpl_source_impl.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "streaming/vpl/vpl_dx11_accel.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {


VPLSourceImpl::VPLSourceImpl() :
    mfx_handle(MFXLoad()),
    source_handle(nullptr, nullptr)
    
{
    GAPI_LOG_INFO(nullptr, "Initialized MFX handle: " << mfx_handle);
}

VPLSourceImpl::~VPLSourceImpl()
{
    GAPI_LOG_INFO(nullptr, "Unload MFX handle: " << mfx_handle);
    MFXUnload(mfx_handle);
}

VPLSourceImpl::VPLSourceImpl(const std::string& file_path, const CFGParams& params) :
    VPLSourceImpl()
{
    this->filePath = file_path;

    // Enable Config
    if (params.empty())
    {
        GAPI_LOG_INFO(nullptr, "No special cfg params requested - use default");
        this->cfg_params = getDefaultCfgParams();
    }
    else
    {
         this->cfg_params = params;
    }

    try {
        source_handle = file_ptr(fopen(filePath.c_str(), "rb"), &fclose);
        if (!source_handle) {
            throw std::runtime_error("Cannot open source file: " + filePath);
        }

        GAPI_LOG_INFO(nullptr, "Requested cfg params count: " << cfg_params.size());
        this->mfx_handle_configs.resize(cfg_params.size());

        // Set handle config params
        auto cfg_param_it = cfg_params.begin();
        for (mfxConfig& cfg_inst : mfx_handle_configs) {
            cfg_inst = MFXCreateConfig(mfx_handle);
            if (!cfg_inst) {
                throw std::runtime_error("MFXCreateConfig failed");
            }

            mfxStatus sts = MFXSetConfigFilterProperty(cfg_inst,
                                                       (mfxU8 *)cfg_param_it->first.c_str(),
                                                       cfg_param_it->second);
            if (sts != MFX_ERR_NONE )
            {
                throw std::runtime_error("MFXSetConfigFilterProperty failed, error: " + std::to_string(sts) +
                                         " - for \"" + cfg_param_it->first + "\"");
            }

            ++cfg_param_it;
        }

        GAPI_LOG_INFO(nullptr, "Find MFX better implementation satisfying requested params, handle: " << mfx_handle);
        int i = 0;
        mfxImplDescription *idesc = nullptr;
        std::map<size_t/*matches count*/, int /*impl index*/> matches_count;
        while (MFX_ERR_NONE == MFXEnumImplementations(mfx_handle,
                                                      i,
                                                      MFX_IMPLCAPS_IMPLDESCSTRUCTURE,
                                                      reinterpret_cast<mfxHDL *>(&idesc))) {

            std::stringstream ss;
            mfxHDL hImplPath = nullptr;
            if (MFX_ERR_NONE == MFXEnumImplementations(mfx_handle, i, MFX_IMPLCAPS_IMPLPATH, &hImplPath)) {
                if (hImplPath) {
                    ss << "Implementation path: "  <<  reinterpret_cast<mfxChar *>(hImplPath) << std::endl;
                    MFXDispReleaseImplDescription(mfx_handle, hImplPath);
                }
            }
            ss << *idesc << std::endl;

            GAPI_LOG_INFO(nullptr, "Implementation index: " << i);
            GAPI_LOG_INFO(nullptr, ss.str());

            MFXDispReleaseImplDescription(mfx_handle, idesc);

            // find intersection
            CFGParams matched_params = get_params_from_string(ss.str());
            GAPI_LOG_INFO/*DEBUG*/(nullptr, "Equal param intersection count: " << matched_params.size());
        
            matches_count.emplace(matches_count.size(), i++);
        }

        //Get max matched
        auto max_match_it = matches_count.rbegin();
        if (max_match_it == matches_count.rend()) {
            throw std::logic_error("Cannot find matched MFX implementation for requested configuration");
        }

        // create session
        mfxStatus sts = MFXCreateSession(mfx_handle, max_match_it->second, &mfx_session);
        if (MFX_ERR_NONE != sts)
        {
            throw std::logic_error("Cannot create MFX Session for implementation index:" +
                               std::to_string(max_match_it->second));
        }

        //open file
        //TODO
        
    } catch (const std::exception& ex) {
        std::stringstream ss;
        ss << "Cannot create VPL source Impl, error: " << ex.what() << ". Unload MFX handle: " << mfx_handle;
        const std::string str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);

        MFXUnload(mfx_handle);
        util::throw_error(std::logic_error(str));
    }
}

void VPLSourceImpl::initializeHWAccel()
{
    auto accel_mode_it = cfg_params.find(CFGParamName("mfxImplDescription.AccelerationMode"));
    if (accel_mode_it == cfg_params.end())
    {
        GAPI_LOG_INFO/*DEBUG*/(nullptr, "No HW Accel requested");
        return;
    }

    GAPI_LOG_INFO/*DEBUG*/(nullptr, "Add HW acceleration support");
    try {
        switch(accel_mode_it->second.Data.U32) {
            case MFX_ACCEL_MODE_VIA_D3D11:
            {
                std::unique_ptr<VPLDX11AccelerationPolicy> cand(new VPLDX11AccelerationPolicy(mfx_session));
                accel_policy = std::move(cand);
                break;
            }
            default:
                throw std::logic_error("invalid type: " +
                                               std::to_string(accel_mode_it->second.Data.U32));
                break;
        }
    } catch (const std::exception& ex) {
         util::throw_error(
                std::logic_error(std::string("Cannot initialize HW Accel, error: ") + ex.what()));
    }
}

#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

const CFGParams& VPLSourceImpl::getDefaultCfgParams()
{
    static const CFGParams def_params{
                {
                    CFGParamName("mfxImplDescription.Impl"),
                    create_cfg_value_u32(MFX_IMPL_TYPE_HARDWARE)},
                {
                    CFGParamName("mfxImplDescription.AccelerationMode"),
                    create_cfg_value_u32(MFX_ACCEL_MODE_VIA_D3D11)}/*,
                {
                    CFGParamName("mfxImplDescription.mfxDecoderDescription.decoder.CodecID"),
                    create_cfg_value_u32(MFX_CODEC_HEVC)},
                {
                    CFGParamName("mfxImplDescription.ApiVersion.Version"),
                    create_cfg_value_u32(MAJOR_API_VERSION_REQUIRED << 16 | MINOR_API_VERSION_REQUIRED)}*/
            };
    return def_params;
}
const CFGParams& VPLSourceImpl::getCfgParams() const
{
    return cfg_params;
}
    
bool VPLSourceImpl::pull(cv::gapi::wip::Data& data)
{
    (void)data;
    return true;
}

GMetaArg VPLSourceImpl::descr_of() const
{
    return GMetaArg{};
}
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
