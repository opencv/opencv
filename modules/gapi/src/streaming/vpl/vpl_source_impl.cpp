#ifdef HAVE_ONEVPL

#include <algorithm>
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

#include "streaming/vpl/vpl_source_engine.hpp"
#include "streaming/vpl/vpl_source_impl.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "streaming/vpl/vpl_dx11_accel.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {

VPLSourceImpl::VPLSourceImpl() :
    mfx_handle(MFXLoad())
{
    GAPI_LOG_INFO(nullptr, "Initialized MFX handle: " << mfx_handle);
}

VPLSourceImpl::~VPLSourceImpl()
{
    GAPI_LOG_INFO(nullptr, "Close Decode for session: " << mfx_session);
    MFXVideoDECODE_Close(mfx_session);

    GAPI_LOG_INFO(nullptr, "Close session: " << mfx_session);
    MFXClose(mfx_session);

    GAPI_LOG_INFO(nullptr, "Unload MFX handle: " << mfx_handle);
    MFXUnload(mfx_handle);
}

VPLSourceImpl::VPLSourceImpl(const std::string& file_path, const CFGParams& params) :
    VPLSourceImpl()
{
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
        VPLDecodeEngine::file_ptr source_handle(fopen(file_path.c_str(), "rb"), &fclose);
        if (!source_handle) {
            throw std::runtime_error("Cannot open source file: " + file_path);
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
            CFGParams impl_params = get_params_from_string(ss.str());

            CFGParams matched_params;
            std::set_intersection(impl_params.begin(), impl_params.end(), cfg_params.begin(), cfg_params.end(),
            std::inserter(matched_params, matched_params.end()),
                          [] (typename CFGParams::value_type& lhs, typename CFGParams::value_type& rhs) -> bool
            {
                if (lhs.first != rhs.first) {
                    return false;
                }
                if(lhs.second.Type != rhs.second.Type) {
                    return false;
                }
                if(!memcmp(&lhs.second.Data, &rhs.second.Data, sizeof(rhs.second.Data))) {
                    return false;
                }
                return true;
            });
            GAPI_LOG_INFO/*DEBUG*/(nullptr, "Equal param intersection count: " << matched_params.size());
        
            matches_count.emplace(matches_count.size(), i++);
        }

        //Get max matched
        int impl_number = 0;
        auto max_match_it = matches_count.rbegin();
        if (max_match_it == matches_count.rend()) {
            throw std::logic_error("Cannot find matched MFX implementation for requested configuration");
        }

        // create session
        GAPI_LOG_INFO(nullptr, "Chosen implementation index: " << impl_number);
        mfxStatus sts = MFXCreateSession(mfx_handle, impl_number, &mfx_session);
        if (MFX_ERR_NONE != sts)
        {
            throw std::logic_error("Cannot create MFX Session for implementation index:" +
                                   std::to_string(max_match_it->second));
        }

        GAPI_LOG_INFO(nullptr, "Initialized MFX session: " << mfx_session);

        // initialize decoder
        try {
            // Find codec ID from config
            auto dec_it = cfg_params.find(CFGParamName("mfxImplDescription.mfxDecoderDescription.decoder.CodecID"));
            if (dec_it == cfg_params.end()) {
                throw std::logic_error("Cannot determine DecoderID from oneVPL config. Abort");
            }

            mfxBitstream mfx_session_bitstream = create_decoder_from_file(dec_it->second, source_handle.get());

            if (!engine) {
                engine.reset(new VPLDecodeEngine(std::move(source_handle)));
            }
            engine->initialize_session(mfx_session, std::move(mfx_session_bitstream));
        } catch(const std::exception& ex) {
            std::stringstream ss;
            ss << ex.what() << ". Unload VPL session: " << mfx_session;
            const std::string str = ss.str();
            GAPI_LOG_WARNING(nullptr, str);
            MFXClose(mfx_session);
            util::throw_error(std::logic_error(str));
        }
    } catch (const std::exception& ex) {
        std::stringstream ss;
        ss << "Cannot create VPL source Impl, error: " << ex.what() << ". Unload MFX handle: " << mfx_handle;
        const std::string str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        MFXUnload(mfx_handle);
        util::throw_error(std::logic_error(str));
    }

    //prepare
    engine->process(mfx_session);
}

mfxBitstream VPLSourceImpl::create_decoder_from_file(const CFGParamValue& decoder, FILE* source_ptr)
{
    if (!source_ptr) {
        throw std::runtime_error("Cannot create decoder, source is nullptr");
    }

    mfxBitstream bitstream{};
    const int BITSTREAM_BUFFER_SIZE = 2000000;
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data      = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    if(!bitstream.Data) {
        throw std::runtime_error("Cannot allocate bitstream.Data bytes: " +
                             std::to_string(bitstream.MaxLength * sizeof(mfxU8)));
    }
    bitstream.CodecId = decoder.Data.U32;

    mfxStatus sts = ReadEncodedStream(bitstream, source_ptr);
    if(MFX_ERR_NONE != sts) {
        throw std::runtime_error("Error reading bitstream, error: " + std::to_string(sts));
    }

    // Retrieve the frame information from input stream
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = decoder.Data.U32;
    mfxDecParams.IOPattern   = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;//MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    sts                      = MFXVideoDECODE_DecodeHeader(mfx_session, &bitstream, &mfxDecParams);
    if(MFX_ERR_NONE != sts) {
        throw std::runtime_error("Error decoding header, error: " + std::to_string(sts));
    }

    // Input parameters finished, now initialize decode
    sts = MFXVideoDECODE_Init(mfx_session, &mfxDecParams);
    if (MFX_ERR_NONE != sts) {
        throw std::runtime_error("Error initializing Decode, error: " + std::to_string(sts));
    }
    return bitstream;
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
/*
    mfxStatus sts = ReadEncodedStream(mfx_session_bitstream, source);
    if (sts != MFX_ERR_NONE) // No more data to read, start decode draining mode
        isDrainingDec = true;
    }
    sts = MFXVideoDECODE_DecodeFrameAsync(mfx_session,
                                          (isDrainingDec == true) ? nullptr : &mfx_session_bitstream,
                                          nullptr,
                                          &dec_surface_out,
                                          &syncp);
    switch (sts) {
        case MFX_ERR_NONE: // Got 1 decoded frame
            do {
                sts = MFXVideoCORE_SyncOperation(mfx_session, syncp, 100);
                if (MFX_ERR_NONE == sts) {

                    // -S- TODO Consume Data
                    framenum++;
                }
            } while (sts == MFX_WRN_IN_EXECUTION);
            break;
        case MFX_ERR_MORE_DATA: // The function requires more bitstream at input before decoding can proceed
            if (isDrainingDec == true)
                isDrainingEnc =
                    true; // No more data to drain from decoder, start encode draining mode
            else
                continue; // read more data
            break;
        case MFX_ERR_MORE_SURFACE:
            // The function requires more frame surface at output before decoding can proceed.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            nIndex = GetFreeSurfaceIndex(decSurfPool, decRequest.NumFrameSuggested);
            break;
        case MFX_ERR_DEVICE_LOST:
            // For non-CPU implementations,
            // Cleanup if device is lost
            break;
        case MFX_WRN_DEVICE_BUSY:
            // For non-CPU implementations,
            // Wait a few milliseconds then try again
            break;
        case MFX_WRN_VIDEO_PARAM_CHANGED:
            // The decoder detected a new sequence header in the bitstream.
            // Video parameters may have changed.
            // In external memory allocation case, might need to reallocate the output surface
            break;
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
            // The function detected that video parameters provided by the application
            // are incompatible with initialization parameters.
            // The application should close the component and then reinitialize it
            break;
        case MFX_ERR_REALLOC_SURFACE:
            // Bigger surface_work required. May be returned only if
            // mfxInfoMFX::EnableReallocRequest was set to ON during initialization.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            break;
        default:
            printf("unknown status %d\n", sts);
            isStillGoing = false;
            break;
    }
    */

    engine->process(mfx_session);
    return true;
}

GMetaArg VPLSourceImpl::descr_of() const
{
    GAPI_Assert(!first_frame.empty());
    return cv::GMetaArg{cv::descr_of(first_frame)};
}
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
