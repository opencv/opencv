/*
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

#ifndef AVCODEC_VERSION_H
#define AVCODEC_VERSION_H

#define LIBAVCODEC_VERSION_MAJOR 53
#define LIBAVCODEC_VERSION_MINOR  7
#define LIBAVCODEC_VERSION_MICRO  0

#define LIBAVCODEC_VERSION_INT  AV_VERSION_INT(LIBAVCODEC_VERSION_MAJOR, \
                                               LIBAVCODEC_VERSION_MINOR, \
                                               LIBAVCODEC_VERSION_MICRO)
#define LIBAVCODEC_VERSION      AV_VERSION(LIBAVCODEC_VERSION_MAJOR,    \
                                           LIBAVCODEC_VERSION_MINOR,    \
                                           LIBAVCODEC_VERSION_MICRO)
#define LIBAVCODEC_BUILD        LIBAVCODEC_VERSION_INT

#define LIBAVCODEC_IDENT        "Lavc" AV_STRINGIFY(LIBAVCODEC_VERSION)

/**
 * Those FF_API_* defines are not part of public API.
 * They may change, break or disappear at any time.
 */
#ifndef FF_API_PALETTE_CONTROL
#define FF_API_PALETTE_CONTROL  (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_OLD_SAMPLE_FMT
#define FF_API_OLD_SAMPLE_FMT   (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_OLD_AUDIOCONVERT
#define FF_API_OLD_AUDIOCONVERT (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_ANTIALIAS_ALGO
#define FF_API_ANTIALIAS_ALGO   (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_REQUEST_CHANNELS
#define FF_API_REQUEST_CHANNELS (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_OPT_H
#define FF_API_OPT_H            (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_THREAD_INIT
#define FF_API_THREAD_INIT      (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_OLD_FF_PICT_TYPES
#define FF_API_OLD_FF_PICT_TYPES (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_FLAC_GLOBAL_OPTS
#define FF_API_FLAC_GLOBAL_OPTS (LIBAVCODEC_VERSION_MAJOR < 54)
#endif
#ifndef FF_API_GET_PIX_FMT_NAME
#define FF_API_GET_PIX_FMT_NAME (LIBAVCODEC_VERSION_MAJOR < 54)
#endif

#endif /* AVCODEC_VERSION_H */
