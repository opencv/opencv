/*
 * The Video Decode and Presentation API for UNIX (VDPAU) is used for
 * hardware-accelerated decoding of MPEG-1/2, H.264 and VC-1.
 *
 * Copyright (C) 2008 NVIDIA
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVCODEC_VDPAU_H
#define AVCODEC_VDPAU_H

/**
 * @file
 * @ingroup lavc_codec_hwaccel_vdpau
 * Public libavcodec VDPAU header.
 */


/**
 * @defgroup lavc_codec_hwaccel_vdpau VDPAU Decoder and Renderer
 * @ingroup lavc_codec_hwaccel
 *
 * VDPAU hardware acceleration has two modules
 * - VDPAU decoding
 * - VDPAU presentation
 *
 * The VDPAU decoding module parses all headers using FFmpeg
 * parsing mechanisms and uses VDPAU for the actual decoding.
 *
 * As per the current implementation, the actual decoding
 * and rendering (API calls) are done as part of the VDPAU
 * presentation (vo_vdpau.c) module.
 *
 * @{
 */

#include <vdpau/vdpau.h>
#include <vdpau/vdpau_x11.h>
#include "libavutil/avconfig.h"

union FFVdpPictureInfo {
    VdpPictureInfoH264        h264;
    VdpPictureInfoMPEG1Or2    mpeg;
    VdpPictureInfoVC1          vc1;
    VdpPictureInfoMPEG4Part2 mpeg4;
};

/**
 * This structure is used to share data between the libavcodec library and
 * the client video application.
 * The user shall zero-allocate the structure and make it available as
 * AVCodecContext.hwaccel_context. Members can be set by the user once
 * during initialization or through each AVCodecContext.get_buffer()
 * function call. In any case, they must be valid prior to calling
 * decoding functions.
 */
typedef struct AVVDPAUContext {
    /**
     * VDPAU decoder handle
     *
     * Set by user.
     */
    VdpDecoder decoder;

    /**
     * VDPAU decoder render callback
     *
     * Set by the user.
     */
    VdpDecoderRender *render;

    /**
     * VDPAU picture information
     *
     * Set by libavcodec.
     */
    union FFVdpPictureInfo info;

    /**
     * Allocated size of the bitstream_buffers table.
     *
     * Set by libavcodec.
     */
    int bitstream_buffers_allocated;

    /**
     * Useful bitstream buffers in the bitstream buffers table.
     *
     * Set by libavcodec.
     */
    int bitstream_buffers_used;

   /**
     * Table of bitstream buffers.
     * The user is responsible for freeing this buffer using av_freep().
     *
     * Set by libavcodec.
     */
    VdpBitstreamBuffer *bitstream_buffers;
} AVVDPAUContext;


/** @brief The videoSurface is used for rendering. */
#define FF_VDPAU_STATE_USED_FOR_RENDER 1

/**
 * @brief The videoSurface is needed for reference/prediction.
 * The codec manipulates this.
 */
#define FF_VDPAU_STATE_USED_FOR_REFERENCE 2

/**
 * @brief This structure is used as a callback between the FFmpeg
 * decoder (vd_) and presentation (vo_) module.
 * This is used for defining a video frame containing surface,
 * picture parameter, bitstream information etc which are passed
 * between the FFmpeg decoder and its clients.
 */
struct vdpau_render_state {
    VdpVideoSurface surface; ///< Used as rendered surface, never changed.

    int state; ///< Holds FF_VDPAU_STATE_* values.

#if AV_HAVE_INCOMPATIBLE_LIBAV_ABI
    /** picture parameter information for all supported codecs */
    union FFVdpPictureInfo info;
#endif

    /** Describe size/location of the compressed video data.
        Set to 0 when freeing bitstream_buffers. */
    int bitstream_buffers_allocated;
    int bitstream_buffers_used;
    /** The user is responsible for freeing this buffer using av_freep(). */
    VdpBitstreamBuffer *bitstream_buffers;

#if !AV_HAVE_INCOMPATIBLE_LIBAV_ABI
    /** picture parameter information for all supported codecs */
    union FFVdpPictureInfo info;
#endif
};

/* @}*/

#endif /* AVCODEC_VDPAU_H */
