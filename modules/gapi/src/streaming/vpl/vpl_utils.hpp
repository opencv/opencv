#ifndef OPENCV_GAPI_VPL_UTILS_HPP
#define OPENCV_GAPI_VPL_UTILS_HPP

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include <map>
#include <string>

using CFGParamName = std::string;
using CFGParamValue = mfxVariant;
using CFGParams = std::map<CFGParamName, CFGParamValue>;




namespace cv {
namespace gapi {
namespace wip {


inline const char* mfx_impl_to_cstr(const mfxIMPL impl) {
    switch (impl) {
        case MFX_IMPL_TYPE_SOFTWARE:
            return "MFX_IMPL_TYPE_SOFTWARE";
        case MFX_IMPL_TYPE_HARDWARE:
            return "MFX_IMPL_TYPE_HARDWARE";
        default:
            return "unknown mfxIMPL";
    }
}

inline mfxIMPL cstr_to_mfx_impl(const char* cstr) {
    if (!strcmp(cstr, "MFX_IMPL_TYPE_SOFTWARE")) {
        return MFX_IMPL_TYPE_SOFTWARE;
    } else if (!strcmp(cstr, "MFX_IMPL_TYPE_HARDWARE")) {
         return MFX_IMPL_TYPE_HARDWARE;
    }

    abort(); // TODO
    return MFX_IMPL_TYPE_SOFTWARE;
}

inline const char* mfx_accel_mode_to_cstr (const mfxAccelerationMode mode) {
    switch (mode) {
        case MFX_ACCEL_MODE_NA:	return "MFX_ACCEL_MODE_NA";
        case MFX_ACCEL_MODE_VIA_D3D9:	return "MFX_ACCEL_MODE_VIA_D3D9";
        case MFX_ACCEL_MODE_VIA_D3D11:	return "MFX_ACCEL_MODE_VIA_D3D11";
        case MFX_ACCEL_MODE_VIA_VAAPI:	return "MFX_ACCEL_MODE_VIA_VAAPI";
        case MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET:	return "MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET";
        case MFX_ACCEL_MODE_VIA_VAAPI_GLX:	return "MFX_ACCEL_MODE_VIA_VAAPI_GLX";
        case MFX_ACCEL_MODE_VIA_VAAPI_X11:	return "MFX_ACCEL_MODE_VIA_VAAPI_X11";
        case MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND:	return "MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND";
        case MFX_ACCEL_MODE_VIA_HDDLUNITE:	return "MFX_ACCEL_MODE_VIA_HDDLUNITE";
        default:
            return "unknown mfxAccelerationMode";
    }
}

inline mfxAccelerationMode cstr_to_mfx_accel_mode(const char* cstr) {
    if (!strcmp(cstr, "MFX_ACCEL_MODE_NA")) {
        return MFX_ACCEL_MODE_NA;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_D3D9")) {
         return MFX_ACCEL_MODE_VIA_D3D9;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_D3D11")) {
         return MFX_ACCEL_MODE_VIA_D3D11;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_VAAPI")) {
         return MFX_ACCEL_MODE_VIA_VAAPI;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET")) {
         return MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_VAAPI_GLX")) {
         return MFX_ACCEL_MODE_VIA_VAAPI_GLX;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_VAAPI_X11")) {
         return MFX_ACCEL_MODE_VIA_VAAPI_X11;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND")) {
         return MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND;
    } else if (!strcmp(cstr, "MFX_ACCEL_MODE_VIA_HDDLUNITE")) {
         return MFX_ACCEL_MODE_VIA_HDDLUNITE;
    }
    
    abort(); // TODO
    return MFX_ACCEL_MODE_NA;
}

inline const char* mfx_resource_type_to_cstr (const mfxResourceType type) {
    switch (type) {
        case MFX_RESOURCE_SYSTEM_SURFACE:	return "MFX_RESOURCE_SYSTEM_SURFACE";
        case MFX_RESOURCE_VA_SURFACE:	return "MFX_RESOURCE_VA_SURFACE";
        case MFX_RESOURCE_VA_BUFFER:	return "MFX_RESOURCE_VA_BUFFER";
        case MFX_RESOURCE_DX9_SURFACE:	return "MFX_RESOURCE_DX9_SURFACE";
        case MFX_RESOURCE_DX11_TEXTURE:	return "MFX_RESOURCE_DX11_TEXTURE";
        case MFX_RESOURCE_DX12_RESOURCE:	return "MFX_RESOURCE_DX12_RESOURCE";
        case MFX_RESOURCE_DMA_RESOURCE:	return "MFX_RESOURCE_DMA_RESOURCE";
        case MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY:	return "MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY";
        default:
            return "unknown mfxResourceType";
    }
}

inline mfxResourceType cstr_to_mfx_resource_type(const char* cstr) {
     if (!strcmp(cstr, "MFX_RESOURCE_SYSTEM_SURFACE")) {
        return MFX_RESOURCE_SYSTEM_SURFACE;
    } else if (!strcmp(cstr, "MFX_RESOURCE_VA_SURFACE")) {
         return MFX_RESOURCE_VA_SURFACE;
    } else if (!strcmp(cstr, "MFX_RESOURCE_VA_BUFFER")) {
         return MFX_RESOURCE_VA_BUFFER;
    } else if (!strcmp(cstr, "MFX_RESOURCE_DX9_SURFACE")) {
         return MFX_RESOURCE_DX9_SURFACE;
    } else if (!strcmp(cstr, "MFX_RESOURCE_DX11_TEXTURE")) {
         return MFX_RESOURCE_DX11_TEXTURE;
    } else if (!strcmp(cstr, "MFX_RESOURCE_DX12_RESOURCE")) {
         return MFX_RESOURCE_DX12_RESOURCE;
    } else if (!strcmp(cstr, "MFX_RESOURCE_DMA_RESOURCE")) {
         return MFX_RESOURCE_DMA_RESOURCE;
    } else if (!strcmp(cstr, "MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY")) {
         return MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY;
    }
    
    abort(); // TODO
    return MFX_RESOURCE_SYSTEM_SURFACE;
}

inline const char* mfx_codec_type_to_cstr(const mfxU32 fourcc, const mfxU32 type) {
    switch (fourcc) {
        case MFX_CODEC_JPEG: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;
                case MFX_PROFILE_JPEG_BASELINE:	return "MFX_PROFILE_JPEG_BASELINE"; break;

                default:
                    return "<unknown MFX_CODEC_JPEG profile";
            }
        }

        case MFX_CODEC_AVC: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;

                case MFX_PROFILE_AVC_BASELINE:	return "MFX_PROFILE_AVC_BASELINE"; break;
                case MFX_PROFILE_AVC_MAIN:	return "MFX_PROFILE_AVC_MAIN"; break;
                case MFX_PROFILE_AVC_EXTENDED:	return "MFX_PROFILE_AVC_EXTENDED"; break;
                case MFX_PROFILE_AVC_HIGH:	return "MFX_PROFILE_AVC_HIGH"; break;
                case MFX_PROFILE_AVC_HIGH10:	return "MFX_PROFILE_AVC_HIGH10"; break;
                case MFX_PROFILE_AVC_HIGH_422:	return "MFX_PROFILE_AVC_HIGH_422"; break;
                case MFX_PROFILE_AVC_CONSTRAINED_BASELINE:	return "MFX_PROFILE_AVC_CONSTRAINED_BASELINE"; break;
                case MFX_PROFILE_AVC_CONSTRAINED_HIGH:	return "MFX_PROFILE_AVC_CONSTRAINED_HIGH"; break;
                case MFX_PROFILE_AVC_PROGRESSIVE_HIGH:	return "MFX_PROFILE_AVC_PROGRESSIVE_HIGH"; break;

                default:
                    return "<unknown MFX_CODEC_AVC profile";
            }
        }

        case MFX_CODEC_HEVC: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;
                case MFX_PROFILE_HEVC_MAIN:	return "MFX_PROFILE_HEVC_MAIN"; break;
                case MFX_PROFILE_HEVC_MAIN10:	return "MFX_PROFILE_HEVC_MAIN10"; break;
                case MFX_PROFILE_HEVC_MAINSP:	return "MFX_PROFILE_HEVC_MAINSP"; break;
                case MFX_PROFILE_HEVC_REXT:	return "MFX_PROFILE_HEVC_REXT"; break;
                case MFX_PROFILE_HEVC_SCC:	return "MFX_PROFILE_HEVC_SCC"; break;

                default:
                    return "<unknown MFX_CODEC_HEVC profile";
            }
        }

        case MFX_CODEC_MPEG2: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;
                case MFX_PROFILE_MPEG2_SIMPLE:	return "MFX_PROFILE_MPEG2_SIMPLE"; break;
                case MFX_PROFILE_MPEG2_MAIN:	return "MFX_PROFILE_MPEG2_MAIN"; break;
                case MFX_LEVEL_MPEG2_HIGH:	return "MFX_LEVEL_MPEG2_HIGH"; break;
                case MFX_LEVEL_MPEG2_HIGH1440:	return "MFX_LEVEL_MPEG2_HIGH1440"; break;

                default:
                    return "<unknown MFX_CODEC_MPEG2 profile";
            }
        }

        case MFX_CODEC_VP8: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;
                case MFX_PROFILE_VP8_0:	return "MFX_PROFILE_VP8_0"; break;
                case MFX_PROFILE_VP8_1:	return "MFX_PROFILE_VP8_1"; break;
                case MFX_PROFILE_VP8_2:	return "MFX_PROFILE_VP8_2"; break;
                case MFX_PROFILE_VP8_3:	return "MFX_PROFILE_VP8_3"; break;

                default:
                    return "<unknown MFX_CODEC_VP9 profile";
            }
        }

        case MFX_CODEC_VC1: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;
                case MFX_PROFILE_VC1_SIMPLE:	return "MFX_PROFILE_VC1_SIMPLE"; break;
                case MFX_PROFILE_VC1_MAIN:	return "MFX_PROFILE_VC1_MAIN"; break;
                case MFX_PROFILE_VC1_ADVANCED:	return "MFX_PROFILE_VC1_ADVANCED"; break;

                default:
                    return "<unknown MFX_CODEC_VC1 profile";
            }
        }

        case MFX_CODEC_VP9: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;
                case MFX_PROFILE_VP9_0:	return "MFX_PROFILE_VP9_0"; break;
                case MFX_PROFILE_VP9_1:	return "MFX_PROFILE_VP9_1"; break;
                case MFX_PROFILE_VP9_2:	return "MFX_PROFILE_VP9_2"; break;
                case MFX_PROFILE_VP9_3:	return "MFX_PROFILE_VP9_3"; break;

                default:
                    return "<unknown MFX_CODEC_VP9 profile";
            }
        }

        case MFX_CODEC_AV1: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:	return "MFX_PROFILE_UNKNOWN"; break;
                case MFX_PROFILE_AV1_MAIN:	return "MFX_PROFILE_AV1_MAIN"; break;
                case MFX_PROFILE_AV1_HIGH:	return "MFX_PROFILE_AV1_HIGH"; break;
                case MFX_PROFILE_AV1_PRO:	return "MFX_PROFILE_AV1_PRO"; break;

                default:
                    return "<unknown MFX_CODEC_AV1 profile";
            }
        }

        default:
            return "unknown codec type :";
    }
}

inline std::tuple<mfxU32/*fourcc*/, mfxU32/*type*/> mfx_codec_type_to_cstr(const char* cstr)
{
    (void)cstr;
    return std::make_tuple(MFX_CODEC_HEVC, MFX_PROFILE_HEVC_MAIN);
}
    
inline std::ostream& operator<< (std::ostream& out, const mfxImplDescription& idesc)
{
    out << "mfxImplDescription.Version: " << static_cast<int>(idesc.Version.Major)
        << "." << static_cast<int>(idesc.Version.Minor) << std::endl;
    out << "mfxImplDescription.Impl: " << mfx_impl_to_cstr(idesc.Impl) << std::endl;
    out << "mfxImplDescription.AccelerationMode: " << mfx_accel_mode_to_cstr(idesc.AccelerationMode) << std::endl;
    out << "mfxImplDescription.ApiVersion: " << idesc.ApiVersion.Major << "." << idesc.ApiVersion.Minor << std::endl;
    out << "mfxImplDescription.ImplName: " << idesc.ImplName << std::endl;
    out << "mfxImplDescription.License: " << idesc.License << std::endl;
    out << "mfxImplDescription.Keywords: " << idesc.Keywords << std::endl;
    out << "mfxImplDescription.VendorID: " << idesc.VendorID << std::endl;
    out << "mfxImplDescription.VendorImplID: " << idesc.VendorImplID << std::endl;

    const mfxAccelerationModeDescription &accel = idesc.AccelerationModeDescription;
    out << "mfxImplDescription.mfxAccelerationMode.Version: " << static_cast<int>(accel.Version.Major)
        << "." << static_cast<int>(accel.Version.Minor) << std::endl;
    for (int mode = 0; mode < accel.NumAccelerationModes; mode++) {
        out << "mfxImplDescription.mfxAccelerationMode.Mode: " << mfx_accel_mode_to_cstr(accel.Mode[mode]) << std::endl;
    }

    const mfxDeviceDescription &dev = idesc.Dev;
    out << "mfxImplDescription.mfxDeviceDescription.Version: " << static_cast<int>(dev.Version.Major)
        << "." << static_cast<int>(dev.Version.Minor) << std::endl;
    out << "mfxImplDescription.mfxDeviceDescription.DeviceID: " << dev.DeviceID << std::endl;
    for (int subdevice = 0; subdevice < dev.NumSubDevices; subdevice++) {
        out << "mfxImplDescription.mfxDeviceDescription.subdevices.Index: " <<     dev.SubDevices[subdevice].Index << std::endl;
        out << "mfxImplDescription.mfxDeviceDescription.subdevices.SubDeviceID: " <<  dev.SubDevices[subdevice].SubDeviceID << std::endl;
    }

        /* mfxDecoderDescription */
    const mfxDecoderDescription &dec = idesc.Dec;
    out << "mfxImplDescription.mfxDecoderDescription.Version: " << static_cast<int>(dec.Version.Major)
        << "." << static_cast<int>(dec.Version.Minor) << std::endl;
    for (int codec = 0; codec < dec.NumCodecs; codec++) {
        auto cid = dec.Codecs[codec].CodecID;
        out << "mfxImplDescription.mfxDecoderDescription.decoder.CodecID: " << cid;//(cid & 0xff) << "." << (cid >> 8 & 0xff) << "." << (cid >> 16 & 0xff) << "." << (cid >> 24 & 0xff)  << std::endl;
        out << "mfxImplDescription.mfxDecoderDescription.decoder.MaxcodecLevel: " << dec.Codecs[codec].MaxcodecLevel << std::endl;
        for (int profile = 0; profile < dec.Codecs[codec].NumProfiles; profile++) {
            out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles: " << mfx_codec_type_to_cstr(dec.Codecs[codec].CodecID,
                                                               dec.Codecs[codec].Profiles[profile].Profile) << std::endl;
            for (int memtype = 0; memtype < dec.Codecs[codec].Profiles[profile].NumMemTypes; memtype++) {
                out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles.MemDesc.MemHandleType: "
                    << mfx_resource_type_to_cstr(dec.Codecs[codec].Profiles[profile].MemDesc[memtype].MemHandleType) << std::endl;
                out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles.MemDesc.Width.Min: " 
                    << dec.Codecs[codec].Profiles[profile].MemDesc[memtype].Width.Min << std::endl;
                out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles.MemDesc.Width.Max: "
                    << dec.Codecs[codec].Profiles[profile].MemDesc[memtype].Width.Max << std::endl;
                out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles.MemDesc.Width.Step: "
                    << dec.Codecs[codec].Profiles[profile].MemDesc[memtype].Width.Step << std::endl;
                out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles.MemDesc.Height.Min: "
                    << dec.Codecs[codec].Profiles[profile].MemDesc[memtype].Height.Min << std::endl;
                out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles.MemDesc.Height.Max: "
                    << dec.Codecs[codec].Profiles[profile].MemDesc[memtype].Height.Max << std::endl;
                out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles.MemDesc.Height.Step: "
                    << dec.Codecs[codec].Profiles[profile].MemDesc[memtype].Height.Step << std::endl;
            }
        }
    }

    /* TODO mfxEncoderDescription */
    /* TODO mfxVPPDescription */

    out << "mfxImplDescription.NumExtParam: " << idesc.NumExtParam << std::endl;
    return out;
}


inline CFGParams get_params_from_string(const std::string& str)
{
    CFGParams ret;
    std::string::size_type pos = 0;
    std::string::size_type endline_pos = std::string::npos;
    do
    {
        endline_pos = str.find_first_of("\r\n", pos);
        std::string line = str.substr(pos, endline_pos == std::string::npos ? std::string::npos : endline_pos - pos);
        if (line.empty()) break;

        std::string::size_type name_endline_pos = line.find(':');
        //TODO
        if (name_endline_pos == std::string::npos) { abort(); }

        std::string name = line.substr(0, name_endline_pos);
        std::string value = line.substr(name_endline_pos + 2);

        CFGParamValue candidate_value;
        if (name == "mfxImplDescription.Impl") {
            candidate_value.Type = MFX_VARIANT_TYPE_U32;
            candidate_value.Data.U32 = cstr_to_mfx_impl(value.c_str());

            ret.emplace(name, candidate_value);
            
        } else if (name == "mfxImplDescription.mfxDecoderDescription.decoder.CodecID") {
            candidate_value.Type = MFX_VARIANT_TYPE_U32;
            candidate_value.Data.U32 = static_cast<mfxU32>(std::strtoll(value.c_str(), nullptr, 10));

            ret.emplace(name, candidate_value);
        } else if (name == "mfxImplDescription.AccelerationMode") {
            candidate_value.Type = MFX_VARIANT_TYPE_U32;
            candidate_value.Data.U32 = cstr_to_mfx_accel_mode(value.c_str());

            ret.emplace(name, candidate_value);
        }//TODO

        pos = endline_pos + 1;
    }
    while (endline_pos != std::string::npos);

    return ret;
}

inline CFGParamValue create_cfg_value_u32(mfxU32 value)
{
    CFGParamValue ret;
    ret.Type = MFX_VARIANT_TYPE_U32;
    ret.Data.U32 = value;
    return ret;
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_VPL_UTILS_HPP
