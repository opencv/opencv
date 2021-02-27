// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include "opencv2/videoio.hpp"
#if defined(__OPENCV_BUILD) || defined(OPENCV_HAVE_CVCONFIG_H)  // TODO Properly detect and add D3D11 / LIBVA dependencies for standalone plugins
#include "cvconfig.h"
#endif
#include <sstream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#ifdef _WIN32
#define FFMPEG_DECODE_ACCELERATION_TYPES    "d3d11va"
#define FFMPEG_ENCODE_ACCELERATION_TYPES    "qsv"
#define FFMPEG_DECODE_DISABLE_CODECS        "none"
#define FFMPEG_ENCODE_DISABLE_CODECS        "mjpeg_qsv"
#else
#define FFMPEG_DECODE_ACCELERATION_TYPES    "vaapi.iHD"
#define FFMPEG_ENCODE_ACCELERATION_TYPES    "qsv.iHD,vaapi.iHD"
#define FFMPEG_DECODE_DISABLE_CODECS        "av1.vaapi,av1_qsv,vp8.vaapi,vp8_qsv"
#define FFMPEG_ENCODE_DISABLE_CODECS        "mjpeg_vaapi,mjpeg_qsv,vp8_vaapi"
#endif

#define HW_DEFAULT_POOL_SIZE    32
#define HW_DEFAULT_SW_FORMAT    AV_PIX_FMT_NV12

using namespace cv;

AVCodec *hw_find_codec(AVCodecID id, AVHWDeviceType hw_type, int (*check_category)(const AVCodec *),
                       const char *disabled_codecs, AVPixelFormat *hw_pix_fmt);
AVBufferRef* hw_create_device(AVHWDeviceType hw_type, int hw_device, const std::string& device_subname);
AVBufferRef* hw_create_frames(struct AVCodecContext* ctx, AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat hw_format);
AVPixelFormat hw_get_format_callback(struct AVCodecContext *ctx, const enum AVPixelFormat * fmt);
VideoAccelerationType hw_type_to_va_type(AVHWDeviceType hw_type);

#ifdef HAVE_D3D11
#define D3D11_NO_HELPERS
#include <d3d11.h>
#include <codecvt>
#endif

#ifdef HAVE_VA
#include <va/va_backend.h>
#endif

extern "C" {
#include <libavutil/hwcontext.h>
#ifdef HAVE_D3D11
#include <libavutil/hwcontext_d3d11va.h>
#endif
#ifdef HAVE_VA
#include <libavutil/hwcontext_vaapi.h>
#endif
}

static bool hw_check_device(AVBufferRef* ctx, AVHWDeviceType hw_type, const std::string& device_subname) {
    if (!ctx)
        return false;
    AVHWDeviceContext* hw_device_ctx = (AVHWDeviceContext*)ctx->data;
    if (!hw_device_ctx->hwctx)
        return false;
    const char *hw_name = av_hwdevice_get_type_name(hw_type);
    if (hw_type == AV_HWDEVICE_TYPE_QSV)
        hw_name = "MFX";
    bool ret = true;
    std::string device_name;
#if defined(HAVE_D3D11)
    if (hw_device_ctx->type == AV_HWDEVICE_TYPE_D3D11VA) {
        ID3D11Device* device = ((AVD3D11VADeviceContext*)hw_device_ctx->hwctx)->device;
        IDXGIDevice* dxgiDevice = nullptr;
        if (device && SUCCEEDED(device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice)))) {
            IDXGIAdapter* adapter = nullptr;
            if (SUCCEEDED(dxgiDevice->GetAdapter(&adapter))) {
                DXGI_ADAPTER_DESC desc;
                if (SUCCEEDED(adapter->GetDesc(&desc))) {
                    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
                    device_name = conv.to_bytes(desc.Description);
                }
                adapter->Release();
            }
            dxgiDevice->Release();
        }
    }
#endif
    if (hw_device_ctx->type == AV_HWDEVICE_TYPE_VAAPI) {
#if defined(HAVE_VA) && (VA_MAJOR_VERSION >= 1)
        VADisplay display = ((AVVAAPIDeviceContext *) hw_device_ctx->hwctx)->display;
        if (display) {
            VADriverContext *va_ctx = ((VADisplayContext *) display)->pDriverContext;
            device_name = va_ctx->str_vendor;
            if (hw_type == AV_HWDEVICE_TYPE_QSV) {
                // Workaround for issue fixed in MediaSDK 21.x https://github.com/Intel-Media-SDK/MediaSDK/issues/2595
                // Checks VAAPI driver for support of VideoProc operation required by MediaSDK
                ret = false;
                int n_entrypoints = va_ctx->max_entrypoints;
                std::vector<VAEntrypoint> entrypoints(n_entrypoints);
                if (va_ctx->vtable->vaQueryConfigEntrypoints(va_ctx, VAProfileNone, entrypoints.data(), &n_entrypoints) == VA_STATUS_SUCCESS) {
                    for (int i = 0; i < n_entrypoints; i++) {
                        if (entrypoints[i] == VAEntrypointVideoProc) {
                            ret = true;
                            break;
                        }
                    }
                }
                if (!ret)
                    CV_LOG_INFO(NULL, "FFMPEG: Skipping MFX video acceleration as entrypoint VideoProc not found in: " << device_name);
            }
        }
#else
        ret = (hw_type != AV_HWDEVICE_TYPE_QSV); // disable MFX if we can't check VAAPI for VideoProc entrypoint
#endif
    }
    if (ret) {
        if (!device_subname.empty() && device_name.find(device_subname) == std::string::npos) {
            CV_LOG_INFO(NULL, "FFMPEG: Skipping " << hw_name <<
                " video acceleration on the following device name as not matching substring '" << device_subname << "': " << device_name);
        } else if (!device_name.empty()) {
            CV_LOG_INFO(NULL, "FFMPEG: Using " << hw_name << " video acceleration on GPU device: " << device_name);
        } else {
            CV_LOG_INFO(NULL, "FFMPEG: Using " << hw_name << " video acceleration");
        }
    }
    return ret;
}

AVBufferRef* hw_create_device(AVHWDeviceType hw_type, int hw_device, const std::string& device_subname) {
    if (AV_HWDEVICE_TYPE_NONE == hw_type)
        return NULL;

    AVHWDeviceType child_type = hw_type;
    if (hw_type == AV_HWDEVICE_TYPE_QSV) {
#ifdef _WIN32
        child_type = AV_HWDEVICE_TYPE_DXVA2;
#else
        child_type = AV_HWDEVICE_TYPE_VAAPI;
#endif
    }

    AVBufferRef* hw_device_ctx = NULL;
    char device[128] = "";
    char* pdevice = NULL;
    if (hw_device >= 0 && hw_device < 100000) {
        if (child_type == AV_HWDEVICE_TYPE_VAAPI) {
            snprintf(device, sizeof(device), "/dev/dri/renderD%d", 128 + hw_device);
        } else {
            snprintf(device, sizeof(device), "%d", hw_device);
        }
        pdevice = device;
    }
    av_hwdevice_ctx_create(&hw_device_ctx, child_type, pdevice, NULL, 0);
    if (hw_device_ctx) {
        if (!hw_check_device(hw_device_ctx, hw_type, device_subname)) {
            av_buffer_unref(&hw_device_ctx);
            return NULL;
        }
        if (hw_type != child_type) {
            AVBufferRef *derived_ctx = NULL;
            av_hwdevice_ctx_create_derived(&derived_ctx, hw_type, hw_device_ctx, 0);
            if (!derived_ctx) {
                const char *hw_name = av_hwdevice_get_type_name(hw_type);
                CV_LOG_INFO(NULL, "FFMPEG: Failed to create derived video acceleration (av_hwdevice_ctx_create_derived) for " << hw_name);
            }
            av_buffer_unref(&hw_device_ctx);
            return derived_ctx;
        } else {
            return hw_device_ctx;
        }
    } else {
        const char *hw_name = av_hwdevice_get_type_name(child_type);
        const char *device_name = pdevice ? pdevice : "'default'";
        CV_LOG_INFO(NULL, "FFMPEG: Failed to create " << hw_name << " video acceleration (av_hwdevice_ctx_create) on device " << device_name);
        return NULL;
    }
}

AVBufferRef* hw_create_frames(struct AVCodecContext* ctx, AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat hw_format)
{
    AVBufferRef *hw_frames_ref = nullptr;
    if (ctx) {
        avcodec_get_hw_frames_parameters(ctx, hw_device_ctx, hw_format, &hw_frames_ref);
    }
    if (!hw_frames_ref) {
        hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    }
    if (!hw_frames_ref) {
        CV_LOG_INFO(NULL, "FFMPEG: Failed to create HW frame context (av_hwframe_ctx_alloc)");
        return NULL;
    }
    AVHWFramesContext *frames_ctx = (AVHWFramesContext *)(hw_frames_ref->data);
    frames_ctx->width = width;
    frames_ctx->height = height;
    if (frames_ctx->format == AV_PIX_FMT_NONE)
        frames_ctx->format = hw_format;
    if (frames_ctx->sw_format == AV_PIX_FMT_NONE)
        frames_ctx->sw_format = HW_DEFAULT_SW_FORMAT;
    if (frames_ctx->initial_pool_size == 0)
        frames_ctx->initial_pool_size = HW_DEFAULT_POOL_SIZE;
    if (av_hwframe_ctx_init(hw_frames_ref) < 0) {
        CV_LOG_INFO(NULL, "FFMPEG: Failed to initialize HW frame context (av_hwframe_ctx_init)");
        av_buffer_unref(&hw_frames_ref);
        return NULL;
    }
    return hw_frames_ref;
}

static bool hw_check_codec(AVCodec* codec, AVHWDeviceType hw_type, const char *disabled_codecs) {
    if (!disabled_codecs)
        disabled_codecs = av_codec_is_decoder(codec) ? FFMPEG_DECODE_DISABLE_CODECS : FFMPEG_ENCODE_DISABLE_CODECS;
    std::string hw_name = std::string(".") + av_hwdevice_get_type_name(hw_type);
    std::stringstream s_stream(disabled_codecs);
    while (s_stream.good()) {
        std::string name;
        getline(s_stream, name, ',');
        if (name == codec->name || name == hw_name || name == codec->name + hw_name || name == "hw") {
            CV_LOG_INFO(NULL, "FFMPEG: skipping codec " << codec->name << hw_name);
            return false;
        }
    }
    return true;
}

AVCodec *hw_find_codec(AVCodecID id, AVHWDeviceType hw_type, int (*check_category)(const AVCodec *), const char *disabled_codecs, AVPixelFormat *hw_pix_fmt) {
    AVCodec *c = 0;
    void *opaque = 0;

    while (NULL != (c = (AVCodec*)av_codec_iterate(&opaque)))
    {
        if (!check_category(c))
            continue;
        if (c->id != id)
            continue;
        if (c->capabilities & AV_CODEC_CAP_EXPERIMENTAL)
            continue;
        if (hw_type != AV_HWDEVICE_TYPE_NONE) {
            AVPixelFormat hw_native_fmt = AV_PIX_FMT_NONE;
#if LIBAVUTIL_BUILD < AV_VERSION_INT(56, 51, 100) // VAAPI encoders support avcodec_get_hw_config() starting ffmpeg 4.3
            if (hw_type == AV_HWDEVICE_TYPE_VAAPI)
                hw_native_fmt = AV_PIX_FMT_VAAPI_VLD;
#endif
            if (hw_type == AV_HWDEVICE_TYPE_CUDA) // CUDA encoders don't support avcodec_get_hw_config()
                hw_native_fmt = AV_PIX_FMT_CUDA;
            if (av_codec_is_encoder(c) && hw_native_fmt != AV_PIX_FMT_NONE && c->pix_fmts) {
                for (int i = 0; c->pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
                    if (c->pix_fmts[i] == hw_native_fmt) {
                        *hw_pix_fmt = hw_native_fmt;
                        if (hw_check_codec(c, hw_type, disabled_codecs))
                            return c;
                    }
                }
            }
            for (int i = 0;; i++) {
                const AVCodecHWConfig *hw_config = avcodec_get_hw_config(c, i);
                if (!hw_config)
                    break;
                if (hw_config->device_type == hw_type) {
                    *hw_pix_fmt = hw_config->pix_fmt;
                    if (hw_check_codec(c, hw_type, disabled_codecs))
                        return c;
                }
            }
        } else {
            return c;
        }
    }

    return NULL;
}

// Callback to select hardware pixel format (not software format) and allocate frame pool (hw_frames_ctx)
AVPixelFormat hw_get_format_callback(struct AVCodecContext *ctx, const enum AVPixelFormat * fmt) {
    if (!ctx->hw_device_ctx)
        return fmt[0];
    AVHWDeviceType hw_type = ((AVHWDeviceContext*)ctx->hw_device_ctx->data)->type;
    for (int j = 0;; j++) {
        const AVCodecHWConfig *hw_config = avcodec_get_hw_config(ctx->codec, j);
        if (!hw_config)
            break;
        if (hw_config->device_type == hw_type) {
            for (int i = 0; fmt[i] != AV_PIX_FMT_NONE; i++) {
                if (fmt[i] == hw_config->pix_fmt) {
                    if (hw_config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX) {
                        ctx->sw_pix_fmt = HW_DEFAULT_SW_FORMAT;
                        ctx->hw_frames_ctx = hw_create_frames(ctx, ctx->hw_device_ctx, ctx->width, ctx->height, fmt[i]);
                        if (ctx->hw_frames_ctx) {
                            //ctx->sw_pix_fmt = ((AVHWFramesContext *)(ctx->hw_frames_ctx->data))->sw_format;
                            return fmt[i];
                        }
                    }
                }
            }
        }
    }
    return fmt[0];
}

VideoAccelerationType hw_type_to_va_type(AVHWDeviceType hw_type) {
    struct HWTypeFFMPEG {
        AVHWDeviceType hw_type;
        VideoAccelerationType va_type;
    } known_hw_types[] = {
            { AV_HWDEVICE_TYPE_D3D11VA, VIDEO_ACCELERATION_D3D11 },
            { AV_HWDEVICE_TYPE_VAAPI, VIDEO_ACCELERATION_VAAPI },
            { AV_HWDEVICE_TYPE_QSV, VIDEO_ACCELERATION_MFX },
            { AV_HWDEVICE_TYPE_CUDA, (VideoAccelerationType)(1 << 11) },
    };
    for (const HWTypeFFMPEG& hw : known_hw_types) {
        if (hw_type == hw.hw_type)
            return hw.va_type;
    }
    return VIDEO_ACCELERATION_NONE;
}

class AccelStringIterator {
public:
    AccelStringIterator(std::string accel_list) :
        _s_stream(accel_list.empty() ? "" : accel_list + ",") // add no-acceleration case to the end of the list
    {
    }
    bool good() const {
        return _s_stream.good();
    }
    void parse_next() {
        std::string hw_type_string;
        getline(_s_stream, hw_type_string, ',');
        size_t index = hw_type_string.find('.');
        if (index != std::string::npos) {
            _device_subname = hw_type_string.substr(index + 1);
            hw_type_string = hw_type_string.substr(0, index);
        } else {
            _device_subname.clear();
        }
        _hw_type = av_hwdevice_find_type_by_name(hw_type_string.c_str());
    }
    AVHWDeviceType hw_type() {
        return _hw_type;
    }
    std::string device_subname() {
        return _device_subname;
    }
private:
    std::stringstream _s_stream;
    AVHWDeviceType _hw_type;
    std::string _device_subname;
};
