// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <stdio.h>

#include <algorithm>
#include <sstream>

#ifdef HAVE_ONEVPL

#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

const char* mfx_impl_to_cstr(const mfxIMPL impl) {
    switch (impl) {
        case MFX_IMPL_TYPE_SOFTWARE:
            return "MFX_IMPL_TYPE_SOFTWARE";
        case MFX_IMPL_TYPE_HARDWARE:
            return "MFX_IMPL_TYPE_HARDWARE";
        default:
            return "unknown mfxIMPL";
    }
}

mfxIMPL cstr_to_mfx_impl(const char* cstr) {
    if (!strcmp(cstr, "MFX_IMPL_TYPE_SOFTWARE")) {
        return MFX_IMPL_TYPE_SOFTWARE;
    } else if (!strcmp(cstr, "MFX_IMPL_TYPE_HARDWARE")) {
        return MFX_IMPL_TYPE_HARDWARE;
    }

    throw std::logic_error(std::string("Invalid \"mfxImplDescription.Impl\":") + cstr);
}

const char* mfx_accel_mode_to_cstr (const mfxAccelerationMode mode) {
    switch (mode) {
        case MFX_ACCEL_MODE_NA:
            return "MFX_ACCEL_MODE_NA";
        case MFX_ACCEL_MODE_VIA_D3D9:
            return "MFX_ACCEL_MODE_VIA_D3D9";
        case MFX_ACCEL_MODE_VIA_D3D11:
            return "MFX_ACCEL_MODE_VIA_D3D11";
        case MFX_ACCEL_MODE_VIA_VAAPI:
            return "MFX_ACCEL_MODE_VIA_VAAPI";
        case MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET:
            return "MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET";
        case MFX_ACCEL_MODE_VIA_VAAPI_GLX:
            return "MFX_ACCEL_MODE_VIA_VAAPI_GLX";
        case MFX_ACCEL_MODE_VIA_VAAPI_X11:
            return "MFX_ACCEL_MODE_VIA_VAAPI_X11";
        case MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND:
            return "MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND";
        case MFX_ACCEL_MODE_VIA_HDDLUNITE:
            return "MFX_ACCEL_MODE_VIA_HDDLUNITE";
        default:
            return "unknown mfxAccelerationMode";
    }
    return "unknown mfxAccelerationMode";
}

mfxAccelerationMode cstr_to_mfx_accel_mode(const char* cstr) {
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
    throw std::logic_error(std::string("Invalid \"mfxImplDescription.AccelerationMode\":") + cstr);
}

const char* mfx_resource_type_to_cstr (const mfxResourceType type) {
    switch (type) {
        case MFX_RESOURCE_SYSTEM_SURFACE:
            return "MFX_RESOURCE_SYSTEM_SURFACE";
        case MFX_RESOURCE_VA_SURFACE:
            return "MFX_RESOURCE_VA_SURFACE";
        case MFX_RESOURCE_VA_BUFFER:
            return "MFX_RESOURCE_VA_BUFFER";
        case MFX_RESOURCE_DX9_SURFACE:
            return "MFX_RESOURCE_DX9_SURFACE";
        case MFX_RESOURCE_DX11_TEXTURE:
            return "MFX_RESOURCE_DX11_TEXTURE";
        case MFX_RESOURCE_DX12_RESOURCE:
            return "MFX_RESOURCE_DX12_RESOURCE";
        case MFX_RESOURCE_DMA_RESOURCE:
            return "MFX_RESOURCE_DMA_RESOURCE";
        case MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY:
            return "MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY";
        default:
            return "unknown mfxResourceType";
    }
}

mfxResourceType cstr_to_mfx_resource_type(const char* cstr) {
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
    throw std::logic_error(std::string("Invalid \"decoder.Profiles.MemDesc.MemHandleType\":") + cstr);
}

mfxU32 cstr_to_mfx_codec_id(const char* cstr) {
    if (!strcmp(cstr, "MFX_CODEC_AVC")) {
        return MFX_CODEC_AVC;
    } else if (!strcmp(cstr, "MFX_CODEC_HEVC")) {
        return MFX_CODEC_HEVC;
    } else if (!strcmp(cstr, "MFX_CODEC_MPEG2")) {
        return MFX_CODEC_MPEG2;
    } else if (!strcmp(cstr, "MFX_CODEC_VC1")) {
        return MFX_CODEC_VC1;
    } else if (!strcmp(cstr, "MFX_CODEC_CAPTURE")) {
        return MFX_CODEC_CAPTURE;
    } else if (!strcmp(cstr, "MFX_CODEC_VP9")) {
        return MFX_CODEC_VP9;
    } else if (!strcmp(cstr, "MFX_CODEC_AV1")) {
        return MFX_CODEC_AV1;
    }
    throw std::logic_error(std::string("Cannot parse \"mfxImplDescription.mfxDecoderDescription.decoder.CodecID\" value: ") + cstr);
}

const char* mfx_codec_type_to_cstr(const mfxU32 fourcc, const mfxU32 type) {
    switch (fourcc) {
        case MFX_CODEC_JPEG: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_JPEG_BASELINE:
                    return "MFX_PROFILE_JPEG_BASELINE";
                default:
                    return "<unknown MFX_CODEC_JPEG profile";
            }
        }

        case MFX_CODEC_AVC: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_AVC_BASELINE:
                    return "MFX_PROFILE_AVC_BASELINE";
                case MFX_PROFILE_AVC_MAIN:
                    return "MFX_PROFILE_AVC_MAIN";
                case MFX_PROFILE_AVC_EXTENDED:
                    return "MFX_PROFILE_AVC_EXTENDED";
                case MFX_PROFILE_AVC_HIGH:
                    return "MFX_PROFILE_AVC_HIGH";
                case MFX_PROFILE_AVC_HIGH10:
                    return "MFX_PROFILE_AVC_HIGH10";
                case MFX_PROFILE_AVC_HIGH_422:
                    return "MFX_PROFILE_AVC_HIGH_422";
                case MFX_PROFILE_AVC_CONSTRAINED_BASELINE:
                    return "MFX_PROFILE_AVC_CONSTRAINED_BASELINE";
                case MFX_PROFILE_AVC_CONSTRAINED_HIGH:
                    return "MFX_PROFILE_AVC_CONSTRAINED_HIGH";
                case MFX_PROFILE_AVC_PROGRESSIVE_HIGH:
                    return "MFX_PROFILE_AVC_PROGRESSIVE_HIGH";
                default:
                    return "<unknown MFX_CODEC_AVC profile";
            }
        }

        case MFX_CODEC_HEVC: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_HEVC_MAIN:
                    return "MFX_PROFILE_HEVC_MAIN";
                case MFX_PROFILE_HEVC_MAIN10:
                    return "MFX_PROFILE_HEVC_MAIN10";
                case MFX_PROFILE_HEVC_MAINSP:
                    return "MFX_PROFILE_HEVC_MAINSP";
                case MFX_PROFILE_HEVC_REXT:
                    return "MFX_PROFILE_HEVC_REXT";
                case MFX_PROFILE_HEVC_SCC:
                    return "MFX_PROFILE_HEVC_SCC";
                default:
                    return "<unknown MFX_CODEC_HEVC profile";
            }
        }

        case MFX_CODEC_MPEG2: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_MPEG2_SIMPLE:
                    return "MFX_PROFILE_MPEG2_SIMPLE";
                case MFX_PROFILE_MPEG2_MAIN:
                    return "MFX_PROFILE_MPEG2_MAIN";
                case MFX_LEVEL_MPEG2_HIGH:
                    return "MFX_LEVEL_MPEG2_HIGH";
                case MFX_LEVEL_MPEG2_HIGH1440:
                    return "MFX_LEVEL_MPEG2_HIGH1440";
                default:
                    return "<unknown MFX_CODEC_MPEG2 profile";
            }
        }

        case MFX_CODEC_VP8: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_VP8_0:
                    return "MFX_PROFILE_VP8_0";
                case MFX_PROFILE_VP8_1:
                    return "MFX_PROFILE_VP8_1";
                case MFX_PROFILE_VP8_2:
                    return "MFX_PROFILE_VP8_2";
                case MFX_PROFILE_VP8_3:
                    return "MFX_PROFILE_VP8_3";
                default:
                    return "<unknown MFX_CODEC_VP8 profile";
            }
        }

        case MFX_CODEC_VC1: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_VC1_SIMPLE:
                    return "MFX_PROFILE_VC1_SIMPLE";
                case MFX_PROFILE_VC1_MAIN:
                    return "MFX_PROFILE_VC1_MAIN";
                case MFX_PROFILE_VC1_ADVANCED:
                    return "MFX_PROFILE_VC1_ADVANCED";
                default:
                    return "<unknown MFX_CODEC_VC1 profile";
            }
        }

        case MFX_CODEC_VP9: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_VP9_0:
                    return "MFX_PROFILE_VP9_0";
                case MFX_PROFILE_VP9_1:
                    return "MFX_PROFILE_VP9_1";
                case MFX_PROFILE_VP9_2:
                    return "MFX_PROFILE_VP9_2";
                case MFX_PROFILE_VP9_3:
                    return "MFX_PROFILE_VP9_3";
                default:
                    return "<unknown MFX_CODEC_VP9 profile";
            }
        }

        case MFX_CODEC_AV1: {
            switch (type) {
                case MFX_PROFILE_UNKNOWN:
                    return "MFX_PROFILE_UNKNOWN";
                case MFX_PROFILE_AV1_MAIN:
                    return "MFX_PROFILE_AV1_MAIN";
                case MFX_PROFILE_AV1_HIGH:
                    return "MFX_PROFILE_AV1_HIGH";
                case MFX_PROFILE_AV1_PRO:
                    return "MFX_PROFILE_AV1_PRO";

                default:
                    return "<unknown MFX_CODEC_AV1 profile";
            }
        }

        default:
            return "unknown codec type :";
    }
}

mfxU32 cstr_to_mfx_version(const char* cstr) {
    if (!cstr) {
        return std::numeric_limits<mfxU32>::max();
    }

    const char* delim = strchr(cstr, '.');
    if (!delim) {
        // in digital form - return as is
        return std::stoul(cstr, nullptr, 10);
    }
    std::string major (cstr, delim - cstr);
    std::string minor (delim + 1);
    mfxU32 major_val = std::stoul(major, nullptr, 10);
    mfxU32 minor_val = std::stoul(minor, nullptr, 10);

    // pack to digital form
    return {major_val << 16 | minor_val};
}

std::ostream& operator<< (std::ostream& out, const mfxImplDescription& idesc)
{
    out << "mfxImplDescription.Version: " << static_cast<int>(idesc.Version.Major)
        << "." << static_cast<int>(idesc.Version.Minor) << std::endl;
    out << "mfxImplDescription.Impl: " << mfx_impl_to_cstr(idesc.Impl) << std::endl;
    out << "(*)mfxImplDescription.AccelerationMode: " << mfx_accel_mode_to_cstr(idesc.AccelerationMode) << std::endl;
    out << "mfxImplDescription.ApiVersion: " << idesc.ApiVersion.Major << "." << idesc.ApiVersion.Minor << std::endl;
    out << "(*)mfxImplDescription.ApiVersion.Version: " << idesc.ApiVersion.Version << std::endl;
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
        out << "(*)mfxImplDescription.mfxDecoderDescription.decoder.CodecID: " << cid;//(cid & 0xff) << "." << (cid >> 8 & 0xff) << "." << (cid >> 16 & 0xff) << "." << (cid >> 24 & 0xff)  << std::endl;
        out << "mfxImplDescription.mfxDecoderDescription.decoder.MaxcodecLevel: " << dec.Codecs[codec].MaxcodecLevel << std::endl;
        for (int profile = 0; profile < dec.Codecs[codec].NumProfiles; profile++) {
            out << "mfxImplDescription.mfxDecoderDescription.decoder.Profiles: "
                << mfx_codec_type_to_cstr(dec.Codecs[codec].CodecID,
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

    out << "mfxImplDescription.NumExtParam: " << idesc.NumExtParam << std::endl;

    out << "\n(*) - configurable params" << std::endl;
    return out;
}

std::string mfxstatus_to_string(mfxStatus err) {
    switch(err)
    {
        case MFX_ERR_NONE:
            return "MFX_ERR_NONE";
        case MFX_ERR_UNKNOWN:
            return "MFX_ERR_UNKNOWN";
        case MFX_ERR_NULL_PTR:
            return "MFX_ERR_NULL_PTR";
        case MFX_ERR_UNSUPPORTED:
            return "MFX_ERR_UNSUPPORTED";
        case MFX_ERR_MEMORY_ALLOC:
            return "MFX_ERR_MEMORY_ALLOC";
        case MFX_ERR_NOT_ENOUGH_BUFFER:
            return "MFX_ERR_NOT_ENOUGH_BUFFER";
        case MFX_ERR_INVALID_HANDLE:
            return "MFX_ERR_INVALID_HANDLE";
        case MFX_ERR_LOCK_MEMORY:
            return "MFX_ERR_LOCK_MEMORY";
        case MFX_ERR_NOT_INITIALIZED:
            return "MFX_ERR_NOT_INITIALIZED";
        case MFX_ERR_NOT_FOUND:
            return "MFX_ERR_NOT_FOUND";
        case MFX_ERR_MORE_DATA:
            return "MFX_ERR_MORE_DATA";
        case MFX_ERR_MORE_SURFACE:
            return "MFX_ERR_MORE_SURFACE";
        case MFX_ERR_ABORTED:
            return "MFX_ERR_ABORTED";
        case MFX_ERR_DEVICE_LOST:
            return "MFX_ERR_DEVICE_LOST";
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
            return "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM";
        case MFX_ERR_INVALID_VIDEO_PARAM:
            return "MFX_ERR_INVALID_VIDEO_PARAM";
        case MFX_ERR_UNDEFINED_BEHAVIOR:
            return "MFX_ERR_UNDEFINED_BEHAVIOR";
        case MFX_ERR_DEVICE_FAILED:
            return "MFX_ERR_DEVICE_FAILED";
        case MFX_ERR_MORE_BITSTREAM:
            return "MFX_ERR_MORE_BITSTREAM";
        case MFX_ERR_GPU_HANG:
            return "MFX_ERR_GPU_HANG";
        case MFX_ERR_REALLOC_SURFACE:
            return "MFX_ERR_REALLOC_SURFACE";
        case MFX_ERR_RESOURCE_MAPPED:
            return "MFX_ERR_RESOURCE_MAPPED";
        case MFX_ERR_NOT_IMPLEMENTED:
            return "MFX_ERR_NOT_IMPLEMENTED";
        case MFX_WRN_DEVICE_BUSY:
            return "MFX_WRN_DEVICE_BUSY";
        case MFX_WRN_VIDEO_PARAM_CHANGED:
            return "MFX_WRN_VIDEO_PARAM_CHANGED";
        case MFX_WRN_IN_EXECUTION:
            return "MFX_WRN_IN_EXECUTION";

        default:
            break;
    }

    std::string ret("<unknown ");
    ret += std::to_string(err) + ">";
    return ret;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
