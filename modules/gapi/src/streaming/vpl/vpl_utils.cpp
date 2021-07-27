#include <stdio.h>

#include <algorithm>
#include <sstream>

#ifdef HAVE_ONEVPL

#include "streaming/vpl/vpl_utils.hpp"

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


CFGParams get_params_from_string(const std::string& str)
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

CFGParamValue create_cfg_value_u32(mfxU32 value)
{
    CFGParamValue ret;
    ret.Type = MFX_VARIANT_TYPE_U32;
    ret.Data.U32 = value;
    return ret;
}

std::string mfxstatus_to_string(mfxStatus err) {
    switch(err)
    {
        case MFX_ERR_NONE:
            return "MFX_ERR_NONE";
        case MFX_ERR_MORE_DATA:
            return "MFX_ERR_MORE_DATA";
        case MFX_ERR_MORE_SURFACE:
            return "MFX_ERR_MORE_SURFACE";
        case MFX_ERR_DEVICE_LOST:
            return "MFX_ERR_DEVICE_LOST";
        case MFX_WRN_DEVICE_BUSY:
            return "MFX_WRN_DEVICE_BUSY";
        case MFX_WRN_VIDEO_PARAM_CHANGED:
            return "MFX_WRN_VIDEO_PARAM_CHANGED";
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
            return "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM";
        case MFX_ERR_REALLOC_SURFACE:
            return "MFX_ERR_REALLOC_SURFACE";
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
