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

// if the header path is not specified explicitly, let's deduce it
#if !defined HAVE_FFMPEG_AVCODEC_H && !defined HAVE_LIBAVCODEC_AVCODEC_H

#if defined(HAVE_GENTOO_FFMPEG)
  #define HAVE_LIBAVFORMAT_AVFORMAT_H 1
#elif defined HAVE_FFMPEG
  #define HAVE_FFMPEG_AVFORMAT_H 1
#endif

#if defined(HAVE_FFMPEG_AVFORMAT_H)
  #include <ffmpeg/avformat.h>
#endif

#if defined(HAVE_LIBAVFORMAT_AVFORMAT_H) || defined(WIN32)
  #include <libavformat/avformat.h>
#endif

#endif

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

const AVCodecTag codec_bmp_tags[] = {
    { CODEC_ID_H264, MKTAG('H', '2', '6', '4') },
    { CODEC_ID_H264, MKTAG('h', '2', '6', '4') },
    { CODEC_ID_H264, MKTAG('X', '2', '6', '4') },
    { CODEC_ID_H264, MKTAG('x', '2', '6', '4') },
    { CODEC_ID_H264, MKTAG('a', 'v', 'c', '1') },
    { CODEC_ID_H264, MKTAG('V', 'S', 'S', 'H') },

    { CODEC_ID_H263, MKTAG('H', '2', '6', '3') },
    { CODEC_ID_H263P, MKTAG('H', '2', '6', '3') },
    { CODEC_ID_H263I, MKTAG('I', '2', '6', '3') }, /* intel h263 */
    { CODEC_ID_H261, MKTAG('H', '2', '6', '1') },

    /* added based on MPlayer */
    { CODEC_ID_H263P, MKTAG('U', '2', '6', '3') },
    { CODEC_ID_H263P, MKTAG('v', 'i', 'v', '1') },

    { CODEC_ID_MPEG4, MKTAG('F', 'M', 'P', '4') },
    { CODEC_ID_MPEG4, MKTAG('D', 'I', 'V', 'X') },
    { CODEC_ID_MPEG4, MKTAG('D', 'X', '5', '0') },
    { CODEC_ID_MPEG4, MKTAG('X', 'V', 'I', 'D') },
    { CODEC_ID_MPEG4, MKTAG('M', 'P', '4', 'S') },
    { CODEC_ID_MPEG4, MKTAG('M', '4', 'S', '2') },
    { CODEC_ID_MPEG4, MKTAG(0x04, 0, 0, 0) }, /* some broken avi use this */

    /* added based on MPlayer */
    { CODEC_ID_MPEG4, MKTAG('D', 'I', 'V', '1') },
    { CODEC_ID_MPEG4, MKTAG('B', 'L', 'Z', '0') },
    { CODEC_ID_MPEG4, MKTAG('m', 'p', '4', 'v') },
    { CODEC_ID_MPEG4, MKTAG('U', 'M', 'P', '4') },
    { CODEC_ID_MPEG4, MKTAG('W', 'V', '1', 'F') },
    { CODEC_ID_MPEG4, MKTAG('S', 'E', 'D', 'G') },

    { CODEC_ID_MPEG4, MKTAG('R', 'M', 'P', '4') },

    { CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '3') }, /* default signature when using MSMPEG4 */
    { CODEC_ID_MSMPEG4V3, MKTAG('M', 'P', '4', '3') },

    /* added based on MPlayer */
    { CODEC_ID_MSMPEG4V3, MKTAG('M', 'P', 'G', '3') },
    { CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '5') },
    { CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '6') },
    { CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '4') },
    { CODEC_ID_MSMPEG4V3, MKTAG('A', 'P', '4', '1') },
    { CODEC_ID_MSMPEG4V3, MKTAG('C', 'O', 'L', '1') },
    { CODEC_ID_MSMPEG4V3, MKTAG('C', 'O', 'L', '0') },

    { CODEC_ID_MSMPEG4V2, MKTAG('M', 'P', '4', '2') },

    /* added based on MPlayer */
    { CODEC_ID_MSMPEG4V2, MKTAG('D', 'I', 'V', '2') },

    { CODEC_ID_MSMPEG4V1, MKTAG('M', 'P', 'G', '4') },

    { CODEC_ID_WMV1, MKTAG('W', 'M', 'V', '1') },

    /* added based on MPlayer */
    { CODEC_ID_WMV2, MKTAG('W', 'M', 'V', '2') },
    { CODEC_ID_DVVIDEO, MKTAG('d', 'v', 's', 'd') },
    { CODEC_ID_DVVIDEO, MKTAG('d', 'v', 'h', 'd') },
    { CODEC_ID_DVVIDEO, MKTAG('d', 'v', 's', 'l') },
    { CODEC_ID_DVVIDEO, MKTAG('d', 'v', '2', '5') },
    { CODEC_ID_MPEG1VIDEO, MKTAG('m', 'p', 'g', '1') },
    { CODEC_ID_MPEG1VIDEO, MKTAG('m', 'p', 'g', '2') },
    { CODEC_ID_MPEG2VIDEO, MKTAG('m', 'p', 'g', '2') },
    { CODEC_ID_MPEG2VIDEO, MKTAG('M', 'P', 'E', 'G') },
    { CODEC_ID_MPEG1VIDEO, MKTAG('P', 'I', 'M', '1') },
    { CODEC_ID_MPEG1VIDEO, MKTAG('V', 'C', 'R', '2') },
    { CODEC_ID_MPEG1VIDEO, 0x10000001 },
    { CODEC_ID_MPEG2VIDEO, 0x10000002 },
    { CODEC_ID_MPEG2VIDEO, MKTAG('D', 'V', 'R', ' ') },
    { CODEC_ID_MPEG2VIDEO, MKTAG('M', 'M', 'E', 'S') },
    { CODEC_ID_MJPEG, MKTAG('M', 'J', 'P', 'G') },
    { CODEC_ID_MJPEG, MKTAG('L', 'J', 'P', 'G') },
    { CODEC_ID_LJPEG, MKTAG('L', 'J', 'P', 'G') },
    { CODEC_ID_MJPEG, MKTAG('J', 'P', 'G', 'L') }, /* Pegasus lossless JPEG */
    { CODEC_ID_MJPEG, MKTAG('M', 'J', 'L', 'S') }, /* JPEG-LS custom FOURCC for avi - decoder */
    { CODEC_ID_MJPEG, MKTAG('j', 'p', 'e', 'g') },
    { CODEC_ID_MJPEG, MKTAG('I', 'J', 'P', 'G') },
    { CODEC_ID_MJPEG, MKTAG('A', 'V', 'R', 'n') },
    { CODEC_ID_HUFFYUV, MKTAG('H', 'F', 'Y', 'U') },
    { CODEC_ID_FFVHUFF, MKTAG('F', 'F', 'V', 'H') },
    { CODEC_ID_CYUV, MKTAG('C', 'Y', 'U', 'V') },
    { CODEC_ID_RAWVIDEO, 0 },
    { CODEC_ID_RAWVIDEO, MKTAG('I', '4', '2', '0') },
    { CODEC_ID_RAWVIDEO, MKTAG('Y', 'U', 'Y', '2') },
    { CODEC_ID_RAWVIDEO, MKTAG('Y', '4', '2', '2') },
    { CODEC_ID_RAWVIDEO, MKTAG('Y', 'V', '1', '2') },
    { CODEC_ID_RAWVIDEO, MKTAG('U', 'Y', 'V', 'Y') },
    { CODEC_ID_RAWVIDEO, MKTAG('I', 'Y', 'U', 'V') },
    { CODEC_ID_RAWVIDEO, MKTAG('Y', '8', '0', '0') },
    { CODEC_ID_RAWVIDEO, MKTAG('H', 'D', 'Y', 'C') },
    { CODEC_ID_INDEO3, MKTAG('I', 'V', '3', '1') },
    { CODEC_ID_INDEO3, MKTAG('I', 'V', '3', '2') },
    { CODEC_ID_VP3, MKTAG('V', 'P', '3', '1') },
    { CODEC_ID_VP3, MKTAG('V', 'P', '3', '0') },
    { CODEC_ID_ASV1, MKTAG('A', 'S', 'V', '1') },
    { CODEC_ID_ASV2, MKTAG('A', 'S', 'V', '2') },
    { CODEC_ID_VCR1, MKTAG('V', 'C', 'R', '1') },
    { CODEC_ID_FFV1, MKTAG('F', 'F', 'V', '1') },
    { CODEC_ID_XAN_WC4, MKTAG('X', 'x', 'a', 'n') },
    { CODEC_ID_MSRLE, MKTAG('m', 'r', 'l', 'e') },
    { CODEC_ID_MSRLE, MKTAG(0x1, 0x0, 0x0, 0x0) },
    { CODEC_ID_MSVIDEO1, MKTAG('M', 'S', 'V', 'C') },
    { CODEC_ID_MSVIDEO1, MKTAG('m', 's', 'v', 'c') },
    { CODEC_ID_MSVIDEO1, MKTAG('C', 'R', 'A', 'M') },
    { CODEC_ID_MSVIDEO1, MKTAG('c', 'r', 'a', 'm') },
    { CODEC_ID_MSVIDEO1, MKTAG('W', 'H', 'A', 'M') },
    { CODEC_ID_MSVIDEO1, MKTAG('w', 'h', 'a', 'm') },
    { CODEC_ID_CINEPAK, MKTAG('c', 'v', 'i', 'd') },
    { CODEC_ID_TRUEMOTION1, MKTAG('D', 'U', 'C', 'K') },
    { CODEC_ID_MSZH, MKTAG('M', 'S', 'Z', 'H') },
    { CODEC_ID_ZLIB, MKTAG('Z', 'L', 'I', 'B') },
    { CODEC_ID_SNOW, MKTAG('S', 'N', 'O', 'W') },
    { CODEC_ID_4XM, MKTAG('4', 'X', 'M', 'V') },
    { CODEC_ID_FLV1, MKTAG('F', 'L', 'V', '1') },
    { CODEC_ID_SVQ1, MKTAG('s', 'v', 'q', '1') },
    { CODEC_ID_TSCC, MKTAG('t', 's', 'c', 'c') },
    { CODEC_ID_ULTI, MKTAG('U', 'L', 'T', 'I') },
    { CODEC_ID_VIXL, MKTAG('V', 'I', 'X', 'L') },
    { CODEC_ID_QPEG, MKTAG('Q', 'P', 'E', 'G') },
    { CODEC_ID_QPEG, MKTAG('Q', '1', '.', '0') },
    { CODEC_ID_QPEG, MKTAG('Q', '1', '.', '1') },
    { CODEC_ID_WMV3, MKTAG('W', 'M', 'V', '3') },
    { CODEC_ID_LOCO, MKTAG('L', 'O', 'C', 'O') },
    { CODEC_ID_THEORA, MKTAG('t', 'h', 'e', 'o') },
#if LIBAVCODEC_VERSION_INT>0x000409
    { CODEC_ID_WNV1, MKTAG('W', 'N', 'V', '1') },
    { CODEC_ID_AASC, MKTAG('A', 'A', 'S', 'C') },
    { CODEC_ID_INDEO2, MKTAG('R', 'T', '2', '1') },
    { CODEC_ID_FRAPS, MKTAG('F', 'P', 'S', '1') },
    { CODEC_ID_TRUEMOTION2, MKTAG('T', 'M', '2', '0') },
#endif
#if LIBAVCODEC_VERSION_INT>((50<<16)+(1<<8)+0)
    { CODEC_ID_FLASHSV, MKTAG('F', 'S', 'V', '1') },
    { CODEC_ID_JPEGLS,MKTAG('M', 'J', 'L', 'S') }, /* JPEG-LS custom FOURCC for avi - encoder */
    { CODEC_ID_VC1, MKTAG('W', 'V', 'C', '1') },
    { CODEC_ID_VC1, MKTAG('W', 'M', 'V', 'A') },
    { CODEC_ID_CSCD, MKTAG('C', 'S', 'C', 'D') },
    { CODEC_ID_ZMBV, MKTAG('Z', 'M', 'B', 'V') },
    { CODEC_ID_KMVC, MKTAG('K', 'M', 'V', 'C') },
#endif
#if LIBAVCODEC_VERSION_INT>((51<<16)+(11<<8)+0)
    { CODEC_ID_VP5, MKTAG('V', 'P', '5', '0') },
    { CODEC_ID_VP6, MKTAG('V', 'P', '6', '0') },
    { CODEC_ID_VP6, MKTAG('V', 'P', '6', '1') },
    { CODEC_ID_VP6, MKTAG('V', 'P', '6', '2') },
    { CODEC_ID_VP6F, MKTAG('V', 'P', '6', 'F') },
    { CODEC_ID_JPEG2000, MKTAG('M', 'J', '2', 'C') },
    { CODEC_ID_VMNC, MKTAG('V', 'M', 'n', 'c') },
#endif
#if LIBAVCODEC_VERSION_INT>=((51<<16)+(49<<8)+0)
// this tag seems not to exist in older versions of FFMPEG
    { CODEC_ID_TARGA, MKTAG('t', 'g', 'a', ' ') },
#endif
    { CODEC_ID_NONE, 0 },
};
