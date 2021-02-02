/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/videoio.hpp"
#include <list>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#ifdef __cplusplus
}
#endif

#define HW_FRAMES_POOL_SIZE     20

using namespace cv;

AVBufferRef* hw_create_device(VideoAccelerationType va_type, int hw_device);
AVBufferRef* hw_create_frames(AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat hw_format, AVPixelFormat sw_format, int pool_size = HW_FRAMES_POOL_SIZE);
AVCodec *hw_find_codec(AVCodecID id, AVBufferRef *hw_device_ctx, int (*check_category)(const AVCodec *), AVPixelFormat *hw_pix_fmt = NULL);
AVPixelFormat hw_get_format_callback(struct AVCodecContext *ctx, const enum AVPixelFormat * fmt);

static AVHWDeviceType VideoAccelerationTypeToFFMPEG(VideoAccelerationType va_type) {
    struct HWTypeFFMPEG {
        VideoAccelerationType va_type;
        AVHWDeviceType ffmpeg_type;
    } ffmpeg_hw_types[] = {
        { VIDEO_ACCELERATION_D3D11, AV_HWDEVICE_TYPE_D3D11VA },
        { VIDEO_ACCELERATION_VAAPI, AV_HWDEVICE_TYPE_VAAPI },
        { VIDEO_ACCELERATION_QSV, AV_HWDEVICE_TYPE_QSV }
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

    AVBufferRef* hw_device_ctx = NULL;
    char device[128] = "";
    char* pdevice = NULL;
    AVDictionary* options = NULL;
    if (hw_device >= 0 && hw_device < 100000) {
#ifdef _WIN32
        snprintf(device, sizeof(device), "%d", hw_device);
#else
        snprintf(device, sizeof(device), "/dev/dri/renderD%d", 128 + hw_device);
#endif
        if (hw_type == AV_HWDEVICE_TYPE_QSV) {
            av_dict_set(&options, "child_device", device, 0);
        }
        else {
            pdevice = device;
        }
    }
    av_hwdevice_ctx_create(&hw_device_ctx, hw_type, pdevice, options, 0);
    if (options)
        av_dict_free(&options);
    return hw_device_ctx;
}

AVBufferRef* hw_create_frames(AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat hw_format, AVPixelFormat sw_format, int pool_size)
{
    AVBufferRef *hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ref) {
        fprintf(stderr, "Failed to create HW frame context\n");
        return NULL;
    }
    AVHWFramesContext *frames_ctx = (AVHWFramesContext *)(hw_frames_ref->data);
    frames_ctx->format    = hw_format;
    frames_ctx->sw_format = sw_format;
    frames_ctx->width     = width;
    frames_ctx->height    = height;
    frames_ctx->initial_pool_size = pool_size;
    if (av_hwframe_ctx_init(hw_frames_ref) < 0) {
        fprintf(stderr, "Failed to initialize HW frame context\n");
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
            for (int i = 0;; i++) {
                const AVCodecHWConfig *hw_config = avcodec_get_hw_config(c, i);
                if (!hw_config)
                    break;
                if (hw_config->device_type == hw_type) {
                    int m = hw_config->methods;
                    if (!(m & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && (m & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX) && hw_pix_fmt) {
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
                    if (hw_config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
                        return fmt[i];
                    }
                    if (hw_config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX) {
                        ctx->hw_frames_ctx = hw_create_frames(ctx->hw_device_ctx, ctx->width, ctx->height, fmt[i], AV_PIX_FMT_NV12);
                        if (ctx->hw_frames_ctx)
                            return fmt[i];
                    }
                }
            }
        }
    }
    return fmt[0];
}
