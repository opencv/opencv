// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <stdio.h>

#include <algorithm>
#include <sstream>

#ifdef HAVE_ONEVPL

#include "streaming/vpl/vpl_utils.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {

std::ostream& operator<< (std::ostream& out, const mfxImplDescription& idesc)
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

template <>
struct ParamCreator<oneVPL_cfg_param> {
    template<typename ValueType>
    oneVPL_cfg_param create (const std::string& name, ValueType&& value) {
        return oneVPL_cfg_param::create(name, std::forward<ValueType>(value), is_major_flag);
    }
    bool is_major_flag = false;
};

template <>
struct ParamCreator<mfxVariant> {
    template<typename ValueType>
    mfxVariant create (const std::string& name, mfxU32 value) {
        return create_impl(name, value);
    }
private:
    mfxVariant create_impl(const std::string&, mfxU32 value) {
        mfxVariant ret;
        ret.Type = MFX_VARIANT_TYPE_U32;
        ret.Data.U32 = value;
        return ret;
    }
};

template<typename ValueType>
std::vector<ValueType> get_params_from_string(const std::string& str) {
    std::vector<ValueType> ret;
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

        ParamCreator<ValueType> creator;
        if (name == "mfxImplDescription.Impl") {
            ret.push_back(creator.create<mfxU32>(name, cstr_to_mfx_impl(value.c_str())));
        } else if (name == "mfxImplDescription.mfxDecoderDescription.decoder.CodecID") {
            ret.push_back(creator.create<mfxU32>(name, cstr_to_mfx_codec_id(value.c_str())));
        } else if (name == "mfxImplDescription.AccelerationMode") {
            ret.push_back(creator.create<mfxU32>(name, cstr_to_mfx_accel_mode(value.c_str())));
        } else {
            GAPI_LOG_DEBUG(nullptr, "Cannot parse configuration param, name: " << name <<
                                    ", value: " << value);
        }

        pos = endline_pos + 1;
    }
    while (endline_pos != std::string::npos);

    return ret;
}

template
std::vector<oneVPL_cfg_param> get_params_from_string(const std::string& str);
template
std::vector<mfxVariant> get_params_from_string(const std::string& str);

mfxVariant cfg_param_to_mfx_variant(const oneVPL_cfg_param& cfg_val) {
    const oneVPL_cfg_param::name_t& name = cfg_val.get_name();
    mfxVariant ret;
    cv::util::visit(cv::util::overload_lambdas(
            [&ret](uint8_t value)   { ret.Type = MFX_VARIANT_TYPE_U8;   ret.Data.U8 = value;    },
            [&ret](int8_t value)    { ret.Type = MFX_VARIANT_TYPE_I8;   ret.Data.I8 = value;    },
            [&ret](uint16_t value)  { ret.Type = MFX_VARIANT_TYPE_U16;  ret.Data.U16 = value;   },
            [&ret](int16_t value)   { ret.Type = MFX_VARIANT_TYPE_I16;  ret.Data.I16 = value;   },
            [&ret](uint32_t value)  { ret.Type = MFX_VARIANT_TYPE_U32;  ret.Data.U32 = value;   },
            [&ret](int32_t value)   { ret.Type = MFX_VARIANT_TYPE_I32;  ret.Data.I32 = value;   },
            [&ret](uint64_t value)  { ret.Type = MFX_VARIANT_TYPE_U64;  ret.Data.U64 = value;   },
            [&ret](int64_t value)   { ret.Type = MFX_VARIANT_TYPE_I64;  ret.Data.I64 = value;   },
            [&ret](float_t value)   { ret.Type = MFX_VARIANT_TYPE_F32;  ret.Data.F32 = value;   },
            [&ret](double_t value)  { ret.Type = MFX_VARIANT_TYPE_F64;  ret.Data.F64 = value;   },
            [&ret](void* value)     { ret.Type = MFX_VARIANT_TYPE_PTR;  ret.Data.Ptr = value;   },
            [&ret, &name] (const std::string& value) {
                auto parsed = get_params_from_string<mfxVariant>(name + ": " + value + "\n");
                if (parsed.empty()) {
                    throw std::logic_error("Unsupported parameter, name: " + name + ", value: " + value);
                }
                ret = *parsed.begin();
            }), cfg_val.get_value());
    return ret;
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
        
        
        default:
            break;
    }

    std::string ret("<unknown ");
    ret += std::to_string(err) + ">";
    return ret;
}

// Read encoded stream from file
mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f) {

    if (!f) {
        return MFX_ERR_MORE_DATA;
    }

    mfxU8 *p0 = bs.Data;
    mfxU8 *p1 = bs.Data + bs.DataOffset;
    if (bs.DataOffset > bs.MaxLength - 1) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }
    if (bs.DataLength + bs.DataOffset > bs.MaxLength) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }
    for (mfxU32 i = 0; i < bs.DataLength; i++) {
        *(p0++) = *(p1++);
    }
    bs.DataOffset = 0;
    bs.DataLength += (mfxU32)fread(bs.Data + bs.DataLength, 1, bs.MaxLength - bs.DataLength, f);
    if (bs.DataLength == 0)
        return MFX_ERR_MORE_DATA;

    return MFX_ERR_NONE;
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
