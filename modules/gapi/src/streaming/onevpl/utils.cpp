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

#define ONEVPL_STRINGIFY_CASE(value)                                           \
    case value: return #value;

#define ONEVPL_STRINGIFY_IF(value)                                             \
    if (!strcmp(cstr, #value)) {                                               \
        return value;                                                          \
    }

#define APPEND_STRINGIFY_MASK_N_ERASE(value, pref, mask)                       \
    if (value & mask) { ss << pref << #mask; value ^= mask; }

#define DUMP_MEMBER(stream, object, member)                                    \
    stream << #member << ": " << object.member << "\n";

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

const char* mfx_impl_to_cstr(const mfxIMPL impl) {
    switch (impl) {
        ONEVPL_STRINGIFY_CASE(MFX_IMPL_TYPE_SOFTWARE);
        ONEVPL_STRINGIFY_CASE(MFX_IMPL_TYPE_HARDWARE);
        default: return "unknown mfxIMPL";
    }
}

mfxIMPL cstr_to_mfx_impl(const char* cstr) {
    ONEVPL_STRINGIFY_IF(MFX_IMPL_TYPE_SOFTWARE)
    ONEVPL_STRINGIFY_IF(MFX_IMPL_TYPE_HARDWARE)
    throw std::logic_error(std::string("Invalid \"") + CfgParam::implementation_name() +
                           "\":" + cstr);
}

const char* mfx_accel_mode_to_cstr (const mfxAccelerationMode mode) {
    switch (mode) {
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_NA)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_D3D9)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_D3D11)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_VAAPI)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_VAAPI_GLX)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_VAAPI_X11)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND)
        ONEVPL_STRINGIFY_CASE(MFX_ACCEL_MODE_VIA_HDDLUNITE)
        default: return "unknown mfxAccelerationMode";
    }
}

mfxAccelerationMode cstr_to_mfx_accel_mode(const char* cstr) {
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_NA)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_D3D9)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_D3D11)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_VAAPI)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_VAAPI_GLX)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_VAAPI_X11)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND)
    ONEVPL_STRINGIFY_IF(MFX_ACCEL_MODE_VIA_HDDLUNITE)
    throw std::logic_error(std::string("Invalid \"") +
                           CfgParam::acceleration_mode_name() +
                           "\":" + cstr);
}

const char* mfx_resource_type_to_cstr (const mfxResourceType type) {
    switch (type) {
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_SYSTEM_SURFACE)
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_VA_SURFACE)
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_VA_BUFFER)
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_DX9_SURFACE)
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_DX11_TEXTURE)
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_DX12_RESOURCE)
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_DMA_RESOURCE)
        ONEVPL_STRINGIFY_CASE(MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY)
        default: return "unknown mfxResourceType";
    }
}

mfxResourceType cstr_to_mfx_resource_type(const char* cstr) {
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_SYSTEM_SURFACE)
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_VA_SURFACE)
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_VA_BUFFER)
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_DX9_SURFACE)
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_DX11_TEXTURE)
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_DX12_RESOURCE)
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_DMA_RESOURCE)
    ONEVPL_STRINGIFY_IF(MFX_RESOURCE_HDDLUNITE_REMOTE_MEMORY)
    throw std::logic_error(std::string("Invalid \"decoder.Profiles.MemDesc.MemHandleType\":") + cstr);
}

mfxU32 cstr_to_mfx_codec_id(const char* cstr) {
    ONEVPL_STRINGIFY_IF(MFX_CODEC_AVC)
    ONEVPL_STRINGIFY_IF(MFX_CODEC_HEVC)
    ONEVPL_STRINGIFY_IF(MFX_CODEC_MPEG2)
    ONEVPL_STRINGIFY_IF(MFX_CODEC_VC1)
    ONEVPL_STRINGIFY_IF(MFX_CODEC_CAPTURE)
    ONEVPL_STRINGIFY_IF(MFX_CODEC_VP9)
    ONEVPL_STRINGIFY_IF(MFX_CODEC_AV1)
    throw std::logic_error(std::string("Cannot parse \"") + CfgParam::decoder_id_name() +
                           "\" value: " + cstr);
}

const char* mfx_codec_id_to_cstr(mfxU32 mfx_id) {
    switch(mfx_id) {
        ONEVPL_STRINGIFY_CASE(MFX_CODEC_AVC)
        ONEVPL_STRINGIFY_CASE(MFX_CODEC_HEVC)
        ONEVPL_STRINGIFY_CASE(MFX_CODEC_MPEG2)
        ONEVPL_STRINGIFY_CASE(MFX_CODEC_VC1)
        ONEVPL_STRINGIFY_CASE(MFX_CODEC_VP9)
        ONEVPL_STRINGIFY_CASE(MFX_CODEC_AV1)
        ONEVPL_STRINGIFY_CASE(MFX_CODEC_JPEG)
        default:
            return "<unsupported>";
    }
}

const std::set<mfxU32>& get_supported_mfx_codec_ids()
{
    static std::set<mfxU32> supported_codecs({MFX_CODEC_AVC,
                                              MFX_CODEC_HEVC,
                                              MFX_CODEC_MPEG2,
                                              MFX_CODEC_VC1,
                                              MFX_CODEC_VP9,
                                              MFX_CODEC_AV1,
                                              MFX_CODEC_JPEG});
    return supported_codecs;
}

const char* mfx_codec_type_to_cstr(const mfxU32 fourcc, const mfxU32 type) {
    switch (fourcc) {
        case MFX_CODEC_JPEG: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_JPEG_BASELINE)
                default: return "<unknown MFX_CODEC_JPEG profile";
            }
        }
        case MFX_CODEC_AVC: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_BASELINE)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_MAIN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_EXTENDED)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_HIGH)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_HIGH10)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_HIGH_422)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_CONSTRAINED_BASELINE)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_CONSTRAINED_HIGH)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AVC_PROGRESSIVE_HIGH)
                default: return "<unknown MFX_CODEC_AVC profile";
            }
        }
        case MFX_CODEC_HEVC: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_HEVC_MAIN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_HEVC_MAIN10)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_HEVC_MAINSP)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_HEVC_REXT)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_HEVC_SCC)
                default: return "<unknown MFX_CODEC_HEVC profile";
            }
        }

        case MFX_CODEC_MPEG2: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_MPEG2_SIMPLE)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_MPEG2_MAIN)
                ONEVPL_STRINGIFY_CASE(MFX_LEVEL_MPEG2_HIGH)
                ONEVPL_STRINGIFY_CASE(MFX_LEVEL_MPEG2_HIGH1440)
                default: return "<unknown MFX_CODEC_MPEG2 profile";
            }
        }

        case MFX_CODEC_VP8: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP8_0)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP8_1)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP8_2)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP8_3)
                default: return "<unknown MFX_CODEC_VP8 profile";
            }
        }

        case MFX_CODEC_VC1: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VC1_SIMPLE)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VC1_MAIN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VC1_ADVANCED)
                default: return "<unknown MFX_CODEC_VC1 profile";
            }
        }

        case MFX_CODEC_VP9: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP9_0)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP9_1)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP9_2)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_VP9_3)
                default: return "<unknown MFX_CODEC_VP9 profile";
            }
        }

        case MFX_CODEC_AV1: {
            switch (type) {
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_UNKNOWN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AV1_MAIN)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AV1_HIGH)
                ONEVPL_STRINGIFY_CASE(MFX_PROFILE_AV1_PRO)
                default: return "<unknown MFX_CODEC_AV1 profile";
            }
        }

        default: return "unknown codec type :";
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
    out << "(*)" << CfgParam::implementation_name() << ": " << mfx_impl_to_cstr(idesc.Impl) << std::endl;
    out << "(*)" << CfgParam::acceleration_mode_name() << ": " << mfx_accel_mode_to_cstr(idesc.AccelerationMode) << std::endl;
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
        out << "(*)" << CfgParam::decoder_id_name() << ": " << cid;//(cid & 0xff) << "." << (cid >> 8 & 0xff) << "." << (cid >> 16 & 0xff) << "." << (cid >> 24 & 0xff)  << std::endl;
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

std::string mfxstatus_to_string(int64_t err) {
    return mfxstatus_to_string(static_cast<mfxStatus>(err));
}

std::string mfxstatus_to_string(mfxStatus err) {
    switch(err) {
        ONEVPL_STRINGIFY_CASE(MFX_ERR_NONE)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_UNKNOWN)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_NULL_PTR)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_UNSUPPORTED)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_MEMORY_ALLOC)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_NOT_ENOUGH_BUFFER)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_INVALID_HANDLE)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_LOCK_MEMORY)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_NOT_INITIALIZED)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_NOT_FOUND)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_MORE_DATA)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_MORE_SURFACE)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_ABORTED)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_DEVICE_LOST)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_INCOMPATIBLE_VIDEO_PARAM)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_INVALID_VIDEO_PARAM)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_UNDEFINED_BEHAVIOR)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_DEVICE_FAILED)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_MORE_BITSTREAM)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_GPU_HANG)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_REALLOC_SURFACE)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_RESOURCE_MAPPED)
        ONEVPL_STRINGIFY_CASE(MFX_ERR_NOT_IMPLEMENTED)
        ONEVPL_STRINGIFY_CASE(MFX_WRN_DEVICE_BUSY)
        ONEVPL_STRINGIFY_CASE(MFX_WRN_VIDEO_PARAM_CHANGED)
        ONEVPL_STRINGIFY_CASE(MFX_WRN_IN_EXECUTION)
        default: break;
    }

    std::string ret("<unknown ");
    ret += std::to_string(static_cast<size_t>(err)) + ">";
    return ret;
}

std::string mfx_frame_info_to_string(const mfxFrameInfo &info) {
    std::stringstream ss;
    DUMP_MEMBER(ss, info, FrameRateExtN)
    DUMP_MEMBER(ss, info, FrameRateExtD)
    DUMP_MEMBER(ss, info, AspectRatioW)
    DUMP_MEMBER(ss, info, AspectRatioH)
    DUMP_MEMBER(ss, info, CropX)
    DUMP_MEMBER(ss, info, CropY)
    DUMP_MEMBER(ss, info, CropW)
    DUMP_MEMBER(ss, info, CropH)
    DUMP_MEMBER(ss, info, ChannelId)
    DUMP_MEMBER(ss, info, BitDepthLuma)
    DUMP_MEMBER(ss, info, BitDepthChroma)
    DUMP_MEMBER(ss, info, Shift)
    DUMP_MEMBER(ss, info, FourCC)
    DUMP_MEMBER(ss, info, Width)
    DUMP_MEMBER(ss, info, Height)
    DUMP_MEMBER(ss, info, BufferSize)
    DUMP_MEMBER(ss, info, PicStruct)
    DUMP_MEMBER(ss, info, ChromaFormat);
    return ss.str();
}

static int compare(const mfxFrameInfo &lhs, const mfxFrameInfo &rhs) {
    //NB: mfxFrameInfo is a `packed` struct declared in VPL
    return memcmp(&lhs, &rhs, sizeof(mfxFrameInfo));
}

bool operator< (const mfxFrameInfo &lhs, const mfxFrameInfo &rhs) {
    return (compare(lhs, rhs) < 0);
}

bool operator== (const mfxFrameInfo &lhs, const mfxFrameInfo &rhs) {
    return (compare(lhs, rhs) == 0);
}

std::string ext_mem_frame_type_to_cstr(int type) {
    std::stringstream ss;
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_DXVA2_DECODER_TARGET);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_DXVA2_PROCESSOR_TARGET);
    // NB: according to VPL source the commented MFX_* constane below are belong to the
    // same actual integral value as condition abobe. So it is impossible
    // to distinct them in condition branch.  Just put this comment and possible
    // constans here...
    //APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET);
    //APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_SYSTEM_MEMORY);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_RESERVED1);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_FROM_ENCODE);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_FROM_DECODE);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_FROM_VPPIN);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_FROM_VPPOUT);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_FROM_ENC);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_INTERNAL_FRAME);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_EXTERNAL_FRAME);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_EXPORT_FRAME);
    //APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_SHARED_RESOURCE);
    APPEND_STRINGIFY_MASK_N_ERASE(type, "|", MFX_MEMTYPE_VIDEO_MEMORY_ENCODER_TARGET);

    if (type != 0) {
        ss << "(rest: " << std::to_string(type) << ")";
    }
    return ss.str();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
