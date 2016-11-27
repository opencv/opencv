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

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(WIN32) || defined(__MINGW32__)
// some versions of FFMPEG assume a C99 compiler, and don't define INT64_C
#include <stdint.h>

// some versions of FFMPEG assume a C99 compiler, and don't define INT64_C
#ifndef INT64_C
#define INT64_C(c) (c##LL)
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

#include <errno.h>
#endif

#include <libavformat/avformat.h>

#ifdef __cplusplus
}
#endif

#ifndef MKTAG
#define MKTAG(a,b,c,d) (a | (b << 8) | (c << 16) | (d << 24))
#endif

// required to look up the correct codec ID depending on the FOURCC code,
// this is just a snipped from the file riff.c from ffmpeg/libavformat
typedef struct AVCodecTag {
    int id;
    unsigned int tag;
} AVCodecTag;

#if (LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(54, 51, 100))
#define AV_CODEC_ID_H264 CODEC_ID_H264
#define AV_CODEC_ID_H263 CODEC_ID_H263
#define AV_CODEC_ID_H263P CODEC_ID_H263P
#define AV_CODEC_ID_H263I CODEC_ID_H263I
#define AV_CODEC_ID_H261 CODEC_ID_H261
#define AV_CODEC_ID_MPEG4 CODEC_ID_MPEG4
#define AV_CODEC_ID_MSMPEG4V3 CODEC_ID_MSMPEG4V3
#define AV_CODEC_ID_MSMPEG4V2 CODEC_ID_MSMPEG4V2
#define AV_CODEC_ID_MSMPEG4V1 CODEC_ID_MSMPEG4V1
#define AV_CODEC_ID_WMV1 CODEC_ID_WMV1
#define AV_CODEC_ID_WMV2 CODEC_ID_WMV1
#define AV_CODEC_ID_DVVIDEO CODEC_ID_DVVIDEO
#define AV_CODEC_ID_MPEG1VIDEO CODEC_ID_MPEG1VIDEO
#define AV_CODEC_ID_MPEG2VIDEO CODEC_ID_MPEG2VIDEO
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
#define AV_CODEC_ID_LJPEG CODEC_ID_LJPEG
#define AV_CODEC_ID_HUFFYUV CODEC_ID_HUFFYUV
#define AV_CODEC_ID_FFVHUFF CODEC_ID_FFVHUFF
#define AV_CODEC_ID_CYUV CODEC_ID_CYUV
#define AV_CODEC_ID_RAWVIDEO CODEC_ID_RAWVIDEO
#define AV_CODEC_ID_INDEO3 CODEC_ID_INDEO3
#define AV_CODEC_ID_VP3 CODEC_ID_VP3
#define AV_CODEC_ID_ASV1 CODEC_ID_ASV1
#define AV_CODEC_ID_ASV2 CODEC_ID_ASV2
#define AV_CODEC_ID_VCR1 CODEC_ID_VCR1
#define AV_CODEC_ID_FFV1 CODEC_ID_FFV1
#define AV_CODEC_ID_XAN_WC4 CODEC_ID_XAN_WC4
#define AV_CODEC_ID_MSRLE CODEC_ID_MSRLE
#define AV_CODEC_ID_MSVIDEO1 CODEC_ID_MSVIDEO1
#define AV_CODEC_ID_CINEPAK CODEC_ID_CINEPAK
#define AV_CODEC_ID_TRUEMOTION1 CODEC_ID_TRUEMOTION1
#define AV_CODEC_ID_MSZH CODEC_ID_MSZH
#define AV_CODEC_ID_ZLIB CODEC_ID_ZLIB
#define AV_CODEC_ID_SNOW CODEC_ID_SNOW
#define AV_CODEC_ID_4XM CODEC_ID_4XM
#define AV_CODEC_ID_FLV1 CODEC_ID_FLV1
#define AV_CODEC_ID_SVQ1 CODEC_ID_SVQ1
#define AV_CODEC_ID_TSCC CODEC_ID_TSCC
#define AV_CODEC_ID_ULTI CODEC_ID_ULTI
#define AV_CODEC_ID_VIXL CODEC_ID_VIXL
#define AV_CODEC_ID_QPEG CODEC_ID_QPEG
#define AV_CODEC_ID_WMV3 CODEC_ID_WMV3
#define AV_CODEC_ID_LOCO CODEC_ID_LOCO
#define AV_CODEC_ID_THEORA CODEC_ID_THEORA
#define AV_CODEC_ID_WNV1 CODEC_ID_WNV1
#define AV_CODEC_ID_AASC CODEC_ID_AASC
#define AV_CODEC_ID_INDEO2 CODEC_ID_INDEO2
#define AV_CODEC_ID_FRAPS CODEC_ID_FRAPS
#define AV_CODEC_ID_TRUEMOTION2 CODEC_ID_TRUEMOTION2
#define AV_CODEC_ID_FLASHSV CODEC_ID_FLASHSV
#define AV_CODEC_ID_JPEGLS CODEC_ID_JPEGLS
#define AV_CODEC_ID_VC1 CODEC_ID_VC1
#define AV_CODEC_ID_CSCD CODEC_ID_CSCD
#define AV_CODEC_ID_ZMBV CODEC_ID_ZMBV
#define AV_CODEC_ID_KMVC CODEC_ID_KMVC
#define AV_CODEC_ID_VP5 CODEC_ID_VP5
#define AV_CODEC_ID_VP6 CODEC_ID_VP6
#define AV_CODEC_ID_VP6F CODEC_ID_VP6F
#define AV_CODEC_ID_JPEG2000 CODEC_ID_JPEG2000
#define AV_CODEC_ID_VMNC CODEC_ID_VMNC
#define AV_CODEC_ID_TARGA CODEC_ID_TARGA
#define AV_CODEC_ID_NONE CODEC_ID_NONE
#endif

const AVCodecTag codec_bmp_tags[] = {
    { AV_CODEC_ID_H264, MKTAG('H', '2', '6', '4') },
    { AV_CODEC_ID_H264, MKTAG('h', '2', '6', '4') },
    { AV_CODEC_ID_H264, MKTAG('X', '2', '6', '4') },
    { AV_CODEC_ID_H264, MKTAG('x', '2', '6', '4') },
    { AV_CODEC_ID_H264, MKTAG('a', 'v', 'c', '1') },
    { AV_CODEC_ID_H264, MKTAG('V', 'S', 'S', 'H') },

    { AV_CODEC_ID_H263, MKTAG('H', '2', '6', '3') },
    { AV_CODEC_ID_H263P, MKTAG('H', '2', '6', '3') },
    { AV_CODEC_ID_H263I, MKTAG('I', '2', '6', '3') }, /* intel h263 */
    { AV_CODEC_ID_H261, MKTAG('H', '2', '6', '1') },

    /* added based on MPlayer */
    { AV_CODEC_ID_H263P, MKTAG('U', '2', '6', '3') },
    { AV_CODEC_ID_H263P, MKTAG('v', 'i', 'v', '1') },

    { AV_CODEC_ID_MPEG4, MKTAG('F', 'M', 'P', '4') },
    { AV_CODEC_ID_MPEG4, MKTAG('D', 'I', 'V', 'X') },
    { AV_CODEC_ID_MPEG4, MKTAG('D', 'X', '5', '0') },
    { AV_CODEC_ID_MPEG4, MKTAG('X', 'V', 'I', 'D') },
    { AV_CODEC_ID_MPEG4, MKTAG('M', 'P', '4', 'S') },
    { AV_CODEC_ID_MPEG4, MKTAG('M', '4', 'S', '2') },
    { AV_CODEC_ID_MPEG4, MKTAG(0x04, 0, 0, 0) }, /* some broken avi use this */

    /* added based on MPlayer */
    { AV_CODEC_ID_MPEG4, MKTAG('D', 'I', 'V', '1') },
    { AV_CODEC_ID_MPEG4, MKTAG('B', 'L', 'Z', '0') },
    { AV_CODEC_ID_MPEG4, MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_MPEG4, MKTAG('U', 'M', 'P', '4') },
    { AV_CODEC_ID_MPEG4, MKTAG('W', 'V', '1', 'F') },
    { AV_CODEC_ID_MPEG4, MKTAG('S', 'E', 'D', 'G') },

    { AV_CODEC_ID_MPEG4, MKTAG('R', 'M', 'P', '4') },

    { AV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '3') }, /* default signature when using MSMPEG4 */
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('M', 'P', '4', '3') },

    /* added based on MPlayer */
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('M', 'P', 'G', '3') },
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '5') },
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '6') },
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '4') },
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('A', 'P', '4', '1') },
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('C', 'O', 'L', '1') },
    { AV_CODEC_ID_MSMPEG4V3, MKTAG('C', 'O', 'L', '0') },

    { AV_CODEC_ID_MSMPEG4V2, MKTAG('M', 'P', '4', '2') },

    /* added based on MPlayer */
    { AV_CODEC_ID_MSMPEG4V2, MKTAG('D', 'I', 'V', '2') },

    { AV_CODEC_ID_MSMPEG4V1, MKTAG('M', 'P', 'G', '4') },

    { AV_CODEC_ID_WMV1, MKTAG('W', 'M', 'V', '1') },

    /* added based on MPlayer */
    { AV_CODEC_ID_WMV2, MKTAG('W', 'M', 'V', '2') },
    { AV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', 's', 'd') },
    { AV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', 'h', 'd') },
    { AV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', 's', 'l') },
    { AV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', '2', '5') },
    { AV_CODEC_ID_MPEG1VIDEO, MKTAG('m', 'p', 'g', '1') },
    { AV_CODEC_ID_MPEG1VIDEO, MKTAG('m', 'p', 'g', '2') },
    { AV_CODEC_ID_MPEG2VIDEO, MKTAG('m', 'p', 'g', '2') },
    { AV_CODEC_ID_MPEG2VIDEO, MKTAG('M', 'P', 'E', 'G') },
    { AV_CODEC_ID_MPEG1VIDEO, MKTAG('P', 'I', 'M', '1') },
    { AV_CODEC_ID_MPEG1VIDEO, MKTAG('V', 'C', 'R', '2') },
    { AV_CODEC_ID_MPEG1VIDEO, 0x10000001 },
    { AV_CODEC_ID_MPEG2VIDEO, 0x10000002 },
    { AV_CODEC_ID_MPEG2VIDEO, MKTAG('D', 'V', 'R', ' ') },
    { AV_CODEC_ID_MPEG2VIDEO, MKTAG('M', 'M', 'E', 'S') },
    { AV_CODEC_ID_MJPEG, MKTAG('M', 'J', 'P', 'G') },
    { AV_CODEC_ID_MJPEG, MKTAG('L', 'J', 'P', 'G') },
    { AV_CODEC_ID_LJPEG, MKTAG('L', 'J', 'P', 'G') },
    { AV_CODEC_ID_MJPEG, MKTAG('J', 'P', 'G', 'L') }, /* Pegasus lossless JPEG */
    { AV_CODEC_ID_MJPEG, MKTAG('M', 'J', 'L', 'S') }, /* JPEG-LS custom FOURCC for avi - decoder */
    { AV_CODEC_ID_MJPEG, MKTAG('j', 'p', 'e', 'g') },
    { AV_CODEC_ID_MJPEG, MKTAG('I', 'J', 'P', 'G') },
    { AV_CODEC_ID_MJPEG, MKTAG('A', 'V', 'R', 'n') },
    { AV_CODEC_ID_HUFFYUV, MKTAG('H', 'F', 'Y', 'U') },
    { AV_CODEC_ID_FFVHUFF, MKTAG('F', 'F', 'V', 'H') },
    { AV_CODEC_ID_CYUV, MKTAG('C', 'Y', 'U', 'V') },
    { AV_CODEC_ID_RAWVIDEO, 0 },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('I', '4', '2', '0') },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('Y', 'U', 'Y', '2') },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('Y', '4', '2', '2') },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('Y', 'V', '1', '2') },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('U', 'Y', 'V', 'Y') },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('I', 'Y', 'U', 'V') },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('Y', '8', '0', '0') },
    { AV_CODEC_ID_RAWVIDEO, MKTAG('H', 'D', 'Y', 'C') },
    { AV_CODEC_ID_INDEO3, MKTAG('I', 'V', '3', '1') },
    { AV_CODEC_ID_INDEO3, MKTAG('I', 'V', '3', '2') },
    { AV_CODEC_ID_VP3, MKTAG('V', 'P', '3', '1') },
    { AV_CODEC_ID_VP3, MKTAG('V', 'P', '3', '0') },
    { AV_CODEC_ID_ASV1, MKTAG('A', 'S', 'V', '1') },
    { AV_CODEC_ID_ASV2, MKTAG('A', 'S', 'V', '2') },
    { AV_CODEC_ID_VCR1, MKTAG('V', 'C', 'R', '1') },
    { AV_CODEC_ID_FFV1, MKTAG('F', 'F', 'V', '1') },
    { AV_CODEC_ID_XAN_WC4, MKTAG('X', 'x', 'a', 'n') },
    { AV_CODEC_ID_MSRLE, MKTAG('m', 'r', 'l', 'e') },
    { AV_CODEC_ID_MSRLE, MKTAG(0x1, 0x0, 0x0, 0x0) },
    { AV_CODEC_ID_MSVIDEO1, MKTAG('M', 'S', 'V', 'C') },
    { AV_CODEC_ID_MSVIDEO1, MKTAG('m', 's', 'v', 'c') },
    { AV_CODEC_ID_MSVIDEO1, MKTAG('C', 'R', 'A', 'M') },
    { AV_CODEC_ID_MSVIDEO1, MKTAG('c', 'r', 'a', 'm') },
    { AV_CODEC_ID_MSVIDEO1, MKTAG('W', 'H', 'A', 'M') },
    { AV_CODEC_ID_MSVIDEO1, MKTAG('w', 'h', 'a', 'm') },
    { AV_CODEC_ID_CINEPAK, MKTAG('c', 'v', 'i', 'd') },
    { AV_CODEC_ID_TRUEMOTION1, MKTAG('D', 'U', 'C', 'K') },
    { AV_CODEC_ID_MSZH, MKTAG('M', 'S', 'Z', 'H') },
    { AV_CODEC_ID_ZLIB, MKTAG('Z', 'L', 'I', 'B') },
    { AV_CODEC_ID_4XM, MKTAG('4', 'X', 'M', 'V') },
    { AV_CODEC_ID_FLV1, MKTAG('F', 'L', 'V', '1') },
    { AV_CODEC_ID_SVQ1, MKTAG('s', 'v', 'q', '1') },
    { AV_CODEC_ID_TSCC, MKTAG('t', 's', 'c', 'c') },
    { AV_CODEC_ID_ULTI, MKTAG('U', 'L', 'T', 'I') },
    { AV_CODEC_ID_VIXL, MKTAG('V', 'I', 'X', 'L') },
    { AV_CODEC_ID_QPEG, MKTAG('Q', 'P', 'E', 'G') },
    { AV_CODEC_ID_QPEG, MKTAG('Q', '1', '.', '0') },
    { AV_CODEC_ID_QPEG, MKTAG('Q', '1', '.', '1') },
    { AV_CODEC_ID_WMV3, MKTAG('W', 'M', 'V', '3') },
    { AV_CODEC_ID_LOCO, MKTAG('L', 'O', 'C', 'O') },
    { AV_CODEC_ID_THEORA, MKTAG('t', 'h', 'e', 'o') },
#if LIBAVCODEC_VERSION_INT>0x000409
    { AV_CODEC_ID_WNV1, MKTAG('W', 'N', 'V', '1') },
    { AV_CODEC_ID_AASC, MKTAG('A', 'A', 'S', 'C') },
    { AV_CODEC_ID_INDEO2, MKTAG('R', 'T', '2', '1') },
    { AV_CODEC_ID_FRAPS, MKTAG('F', 'P', 'S', '1') },
    { AV_CODEC_ID_TRUEMOTION2, MKTAG('T', 'M', '2', '0') },
#endif
#if LIBAVCODEC_VERSION_INT>((50<<16)+(1<<8)+0)
    { AV_CODEC_ID_FLASHSV, MKTAG('F', 'S', 'V', '1') },
    { AV_CODEC_ID_JPEGLS,MKTAG('M', 'J', 'L', 'S') }, /* JPEG-LS custom FOURCC for avi - encoder */
    { AV_CODEC_ID_VC1, MKTAG('W', 'V', 'C', '1') },
    { AV_CODEC_ID_VC1, MKTAG('W', 'M', 'V', 'A') },
    { AV_CODEC_ID_CSCD, MKTAG('C', 'S', 'C', 'D') },
    { AV_CODEC_ID_ZMBV, MKTAG('Z', 'M', 'B', 'V') },
    { AV_CODEC_ID_KMVC, MKTAG('K', 'M', 'V', 'C') },
#endif
#if LIBAVCODEC_VERSION_INT>((51<<16)+(11<<8)+0)
    { AV_CODEC_ID_VP5, MKTAG('V', 'P', '5', '0') },
    { AV_CODEC_ID_VP6, MKTAG('V', 'P', '6', '0') },
    { AV_CODEC_ID_VP6, MKTAG('V', 'P', '6', '1') },
    { AV_CODEC_ID_VP6, MKTAG('V', 'P', '6', '2') },
    { AV_CODEC_ID_VP6F, MKTAG('V', 'P', '6', 'F') },
    { AV_CODEC_ID_JPEG2000, MKTAG('M', 'J', '2', 'C') },
    { AV_CODEC_ID_VMNC, MKTAG('V', 'M', 'n', 'c') },
#endif
#if LIBAVCODEC_VERSION_INT>=((51<<16)+(49<<8)+0)
// this tag seems not to exist in older versions of FFMPEG
    { AV_CODEC_ID_TARGA, MKTAG('t', 'g', 'a', ' ') },
#endif
    { AV_CODEC_ID_NONE, 0 },
};
