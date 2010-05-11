/*
 * RTSP definitions
 * copyright (c) 2002 Fabrice Bellard
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

#ifndef AVFORMAT_RTSPCODES_H
#define AVFORMAT_RTSPCODES_H

/** RTSP handling */
enum RTSPStatusCode {
RTSP_STATUS_OK              =200, /**< OK */
RTSP_STATUS_METHOD          =405, /**< Method Not Allowed */
RTSP_STATUS_BANDWIDTH       =453, /**< Not Enough Bandwidth */
RTSP_STATUS_SESSION         =454, /**< Session Not Found */
RTSP_STATUS_STATE           =455, /**< Method Not Valid in This State */
RTSP_STATUS_AGGREGATE       =459, /**< Aggregate operation not allowed */
RTSP_STATUS_ONLY_AGGREGATE  =460, /**< Only aggregate operation allowed */
RTSP_STATUS_TRANSPORT       =461, /**< Unsupported transport */
RTSP_STATUS_INTERNAL        =500, /**< Internal Server Error */
RTSP_STATUS_SERVICE         =503, /**< Service Unavailable */
RTSP_STATUS_VERSION         =505, /**< RTSP Version not supported */
};

#endif /* AVFORMAT_RTSPCODES_H */
