// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_VPL_UTILS_HPP
#define OPENCV_GAPI_VPL_UTILS_HPP

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif // MFX_VERSION

#include <vpl/mfx.h>
#include <vpl/mfxvideo.h>
#include <map>
#include <string>

#include <opencv2/gapi/streaming/onevpl_cap.hpp>


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

inline mfxU32 cstr_to_mfx_codec_id(const char* cstr) {
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

std::string mfxstatus_to_string(mfxStatus err);

inline std::tuple<mfxU32/*fourcc*/, mfxU32/*type*/> mfx_codec_type_to_cstr(const char* cstr)
{
    (void)cstr;
    return std::make_tuple(MFX_CODEC_HEVC, MFX_PROFILE_HEVC_MAIN);
}
    
std::ostream& operator<< (std::ostream& out, const mfxImplDescription& idesc);

template<typename ValueType>
std::vector<ValueType> get_params_from_string(const std::string& str);

template <typename ReturnType>
struct ParamCreator {
    template<typename ValueType>
    ReturnType create(const std::string& name, ValueType&& value);
};

mfxVariant cfg_param_to_mfx_variant(const CFGParamValue& value);
mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f);
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_VPL_UTILS_HPP
