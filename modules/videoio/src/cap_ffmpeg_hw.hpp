// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include "opencv2/videoio.hpp"
#include <list>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#ifdef __cplusplus
}
#endif

#define HW_DEFAULT_POOL_SIZE    32
#define HW_DEFAULT_SW_FORMAT    AV_PIX_FMT_NV12

using namespace cv;

AVBufferRef* hw_create_device(VideoAccelerationType va_type, int hw_device);
AVBufferRef* hw_create_frames(struct AVCodecContext* ctx, AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat hw_format);
AVCodec *hw_find_codec(AVCodecID id, AVBufferRef *hw_device_ctx, int (*check_category)(const AVCodec *), AVPixelFormat *hw_pix_fmt = NULL);
AVPixelFormat hw_get_format_callback(struct AVCodecContext *ctx, const enum AVPixelFormat * fmt);

#if LIBAVUTIL_VERSION_MAJOR >= 56 // FFMPEG 4.0+

#ifdef __cplusplus
extern "C" {
#endif
#include <libavutil/hwcontext.h>
#ifdef __cplusplus
}
#endif

static AVHWDeviceType VideoAccelerationTypeToFFMPEG(VideoAccelerationType va_type) {
    struct HWTypeFFMPEG {
        VideoAccelerationType va_type;
        AVHWDeviceType ffmpeg_type;
    } ffmpeg_hw_types[] = {
        { VIDEO_ACCELERATION_D3D11, AV_HWDEVICE_TYPE_D3D11VA },
        { VIDEO_ACCELERATION_VAAPI, AV_HWDEVICE_TYPE_VAAPI },
        { VIDEO_ACCELERATION_MFX, AV_HWDEVICE_TYPE_QSV }
    };
    for (const HWTypeFFMPEG& hw : ffmpeg_hw_types) {
        if (va_type == hw.va_type)
            return hw.ffmpeg_type;
    }
    return AV_HWDEVICE_TYPE_NONE;
}

AVBufferRef* hw_create_device(VideoAccelerationType va_type, int hw_device) {
    AVHWDeviceType hw_type = VideoAccelerationTypeToFFMPEG(va_type);
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
#ifdef _WIN32
        snprintf(device, sizeof(device), "%d", hw_device);
#else
        snprintf(device, sizeof(device), "/dev/dri/renderD%d", 128 + hw_device);
#endif
        pdevice = device;
    }
    av_hwdevice_ctx_create(&hw_device_ctx, child_type, pdevice, NULL, 0);
    if (hw_type != child_type) {
        AVBufferRef *derived_ctx = NULL;
        av_hwdevice_ctx_create_derived(&derived_ctx, hw_type, hw_device_ctx, 0);
        av_buffer_unref(&hw_device_ctx);
        return derived_ctx;
    } else {
        return hw_device_ctx;
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
        CV_LOG_DEBUG(NULL, "Failed to create HW frame context");
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
        CV_LOG_DEBUG(NULL, "Failed to initialize HW frame context");
        av_buffer_unref(&hw_frames_ref);
        return NULL;
    }
    return hw_frames_ref;
}

AVCodec *hw_find_codec(AVCodecID id, AVBufferRef *hw_device_ctx, int (*check_category)(const AVCodec *), AVPixelFormat *hw_pix_fmt) {
    AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
    AVCodec *c;
    void *opaque = 0;

    if (hw_device_ctx)
        hw_type = ((AVHWDeviceContext *) hw_device_ctx->data)->type;

    while ((c = (AVCodec*)av_codec_iterate(&opaque))) {
        if (!check_category(c))
            continue;
        if (c->id != id)
            continue;
        if (hw_type != AV_HWDEVICE_TYPE_NONE) {
#if LIBAVUTIL_BUILD < AV_VERSION_INT(56, 51, 100) // ffmpeg 4.3
            // Workaround for VAAPI encoders in ffmpeg 4.0-4.2 (ex, in Ubuntu 20.04)
            // Workaround not needed for ffmpeg 4.3+ as VAAPI encoders started reporting HW configs via avcodec_get_hw_config
            if (c->pix_fmts && AV_PIX_FMT_VAAPI == c->pix_fmts[0] && av_codec_is_encoder(c)) {
                *hw_pix_fmt = c->pix_fmts[0];
                return c;
            }
#endif
            for (int i = 0;; i++) {
                const AVCodecHWConfig *hw_config = avcodec_get_hw_config(c, i);
                if (!hw_config)
                    break;
                if (hw_config->device_type == hw_type) {
                    int m = hw_config->methods;
                    if ((m & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX) && hw_pix_fmt) {
                        // codec requires frame pool (hw_frames_ctx) created by application
                        *hw_pix_fmt = hw_config->pix_fmt;
                    }
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
                        ctx->hw_frames_ctx = hw_create_frames(ctx, ctx->hw_device_ctx, ctx->width, ctx->height, fmt[i]);
                        if (ctx->hw_frames_ctx) {
                            ctx->sw_pix_fmt = ((AVHWFramesContext *)(ctx->hw_frames_ctx->data))->sw_format;
                            return fmt[i];
                        }
                    }
                }
            }
        }
    }
    return fmt[0];
}

#else

AVBufferRef* hw_create_device(VideoAccelerationType, int) {
    return NULL;
}
AVBufferRef* hw_create_frames(AVBufferRef *, int , int , AVPixelFormat , AVPixelFormat , int ) {
    return NULL;
}
AVCodec *hw_find_codec(AVCodecID , AVBufferRef *, int (*)(const AVCodec *), AVPixelFormat *) {
    return NULL;
}
AVPixelFormat hw_get_format_callback(struct AVCodecContext *, const enum AVPixelFormat * fmt) {
    return fmt[0];
}

#endif
