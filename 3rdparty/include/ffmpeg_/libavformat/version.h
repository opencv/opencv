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

#define LIBAVFORMAT_VERSION_MAJOR 55
#define LIBAVFORMAT_VERSION_MINOR 12
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
 * FF_API_* defines may be placed below to indicate public API that will be
 * dropped at a future version bump. The defines themselves are not part of
 * the public API and may change, break or disappear at any time.
 */

#ifndef FF_API_OLD_AVIO
#define FF_API_OLD_AVIO                (LIBAVFORMAT_VERSION_MAJOR < 55)
#endif
#ifndef FF_API_PKT_DUMP
#define FF_API_PKT_DUMP                (LIBAVFORMAT_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_ALLOC_OUTPUT_CONTEXT
#define FF_API_ALLOC_OUTPUT_CONTEXT    (LIBAVFORMAT_VERSION_MAJOR < 56)
#endif
#ifndef FF_API_FORMAT_PARAMETERS
#define FF_API_FORMAT_PARAMETERS       (LIBAVFORMAT_VERSION_MAJOR < 56)
#endif
#ifndef FF_API_NEW_STREAM
#define FF_API_NEW_STREAM              (LIBAVFORMAT_VERSION_MAJOR < 56)
#endif
#ifndef FF_API_SET_PTS_INFO
#define FF_API_SET_PTS_INFO            (LIBAVFORMAT_VERSION_MAJOR < 56)
#endif
#ifndef FF_API_CLOSE_INPUT_FILE
#define FF_API_CLOSE_INPUT_FILE        (LIBAVFORMAT_VERSION_MAJOR < 56)
#endif
#ifndef FF_API_READ_PACKET
#define FF_API_READ_PACKET             (LIBAVFORMAT_VERSION_MAJOR < 56)
#endif
#ifndef FF_API_ASS_SSA
#define FF_API_ASS_SSA                 (LIBAVFORMAT_VERSION_MAJOR < 56)
#endif
#ifndef FF_API_R_FRAME_RATE
#define FF_API_R_FRAME_RATE            1
#endif
#endif /* AVFORMAT_VERSION_H */
