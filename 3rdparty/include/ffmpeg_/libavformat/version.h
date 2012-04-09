/*
 * Version macros.
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

#ifndef AVFORMAT_VERSION_H
#define AVFORMAT_VERSION_H

/**
 * @file
 * @ingroup libavf
 * Libavformat version macros
 */

#include "libavutil/avutil.h"

#define LIBAVFORMAT_VERSION_MAJOR 53
#define LIBAVFORMAT_VERSION_MINOR 32
#define LIBAVFORMAT_VERSION_MICRO 100

#define LIBAVFORMAT_VERSION_INT AV_VERSION_INT(LIBAVFORMAT_VERSION_MAJOR, \
                                               LIBAVFORMAT_VERSION_MINOR, \
                                               LIBAVFORMAT_VERSION_MICRO)
#define LIBAVFORMAT_VERSION     AV_VERSION(LIBAVFORMAT_VERSION_MAJOR,   \
                                           LIBAVFORMAT_VERSION_MINOR,   \
                                           LIBAVFORMAT_VERSION_MICRO)
#define LIBAVFORMAT_BUILD       LIBAVFORMAT_VERSION_INT

#define LIBAVFORMAT_IDENT       "Lavf" AV_STRINGIFY(LIBAVFORMAT_VERSION)

/**
 * Those FF_API_* defines are not part of public API.
 * They may change, break or disappear at any time.
 */
#ifndef FF_API_OLD_METADATA2
#define FF_API_OLD_METADATA2           (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_OLD_AVIO
#define FF_API_OLD_AVIO                (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_DUMP_FORMAT
#define FF_API_DUMP_FORMAT             (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_PARSE_DATE
#define FF_API_PARSE_DATE              (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_FIND_INFO_TAG
#define FF_API_FIND_INFO_TAG           (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_PKT_DUMP
#define FF_API_PKT_DUMP                (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_GUESS_IMG2_CODEC
#define FF_API_GUESS_IMG2_CODEC        (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_SDP_CREATE
#define FF_API_SDP_CREATE              (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_ALLOC_OUTPUT_CONTEXT
#define FF_API_ALLOC_OUTPUT_CONTEXT    (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_FORMAT_PARAMETERS
#define FF_API_FORMAT_PARAMETERS       (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_FLAG_RTP_HINT
#define FF_API_FLAG_RTP_HINT           (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_AVSTREAM_QUALITY
#define FF_API_AVSTREAM_QUALITY        (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_LOOP_INPUT
#define FF_API_LOOP_INPUT              (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_LOOP_OUTPUT
#define FF_API_LOOP_OUTPUT             (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_TIMESTAMP
#define FF_API_TIMESTAMP               (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_FILESIZE
#define FF_API_FILESIZE                (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_MUXRATE
#define FF_API_MUXRATE                 (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_RTSP_URL_OPTIONS
#define FF_API_RTSP_URL_OPTIONS        (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_NEW_STREAM
#define FF_API_NEW_STREAM              (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_PRELOAD
#define FF_API_PRELOAD                 (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_STREAM_COPY
#define FF_API_STREAM_COPY             (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_SEEK_PUBLIC
#define FF_API_SEEK_PUBLIC             (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_REORDER_PRIVATE
#define FF_API_REORDER_PRIVATE         (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_OLD_INTERRUPT_CB
#define FF_API_OLD_INTERRUPT_CB        (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_SET_PTS_INFO
#define FF_API_SET_PTS_INFO            (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_CLOSE_INPUT_FILE
#define FF_API_CLOSE_INPUT_FILE        (LIBAVFORMAT_VERSION_MAJOR < 55)
#endif

#endif /* AVFORMAT_VERSION_H */
