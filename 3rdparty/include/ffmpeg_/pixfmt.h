/*
 * copyright (c) 2006 Michael Niedermayer <michaelni@gmx.at>
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

#ifndef AVUTIL_PIXFMT_H
#define AVUTIL_PIXFMT_H

/**
 * @file libavutil/pixfmt.h
 * pixel format definitions
 *
 * @warning This file has to be considered an internal but installed
 * header, so it should not be directly included in your projects.
 */

/**
 * Pixel format. Notes:
 *
 * PIX_FMT_RGB32 is handled in an endian-specific manner. An RGBA
 * color is put together as:
 *  (A << 24) | (R << 16) | (G << 8) | B
 * This is stored as BGRA on little-endian CPU architectures and ARGB on
 * big-endian CPUs.
 *
 * When the pixel format is palettized RGB (PIX_FMT_PAL8), the palettized
 * image data is stored in AVFrame.data[0]. The palette is transported in
 * AVFrame.data[1], is 1024 bytes long (256 4-byte entries) and is
 * formatted the same as in PIX_FMT_RGB32 described above (i.e., it is
 * also endian-specific). Note also that the individual RGB palette
 * components stored in AVFrame.data[1] should be in the range 0..255.
 * This is important as many custom PAL8 video codecs that were designed
 * to run on the IBM VGA graphics adapter use 6-bit palette components.
 *
 * For all the 8bit per pixel formats, an RGB32 palette is in data[1] like
 * for pal8. This palette is filled in automatically by the function
 * allocating the picture.
 *
 * Note, make sure that all newly added big endian formats have pix_fmt&1==1
 *       and that all newly added little endian formats have pix_fmt&1==0
 *       this allows simpler detection of big vs little endian.
 */
enum PixelFormat {
    PIX_FMT_NONE= -1,
    PIX_FMT_YUV420P,   ///< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
    PIX_FMT_YUYV422,   ///< packed YUV 4:2:2, 16bpp, Y0 Cb Y1 Cr
    PIX_FMT_RGB24,     ///< packed RGB 8:8:8, 24bpp, RGBRGB...
    PIX_FMT_BGR24,     ///< packed RGB 8:8:8, 24bpp, BGRBGR...
    PIX_FMT_YUV422P,   ///< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
    PIX_FMT_YUV444P,   ///< planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
    PIX_FMT_RGB32,     ///< packed RGB 8:8:8, 32bpp, (msb)8A 8R 8G 8B(lsb), in CPU endianness
    PIX_FMT_YUV410P,   ///< planar YUV 4:1:0,  9bpp, (1 Cr & Cb sample per 4x4 Y samples)
    PIX_FMT_YUV411P,   ///< planar YUV 4:1:1, 12bpp, (1 Cr & Cb sample per 4x1 Y samples)
    PIX_FMT_RGB565,    ///< packed RGB 5:6:5, 16bpp, (msb)   5R 6G 5B(lsb), in CPU endianness
    PIX_FMT_RGB555,    ///< packed RGB 5:5:5, 16bpp, (msb)1A 5R 5G 5B(lsb), in CPU endianness, most significant bit to 0
    PIX_FMT_GRAY8,     ///<        Y        ,  8bpp
    PIX_FMT_MONOWHITE, ///<        Y        ,  1bpp, 0 is white, 1 is black
    PIX_FMT_MONOBLACK, ///<        Y        ,  1bpp, 0 is black, 1 is white
    PIX_FMT_PAL8,      ///< 8 bit with PIX_FMT_RGB32 palette
    PIX_FMT_YUVJ420P,  ///< planar YUV 4:2:0, 12bpp, full scale (JPEG)
    PIX_FMT_YUVJ422P,  ///< planar YUV 4:2:2, 16bpp, full scale (JPEG)
    PIX_FMT_YUVJ444P,  ///< planar YUV 4:4:4, 24bpp, full scale (JPEG)
    PIX_FMT_XVMC_MPEG2_MC,///< XVideo Motion Acceleration via common packet passing
    PIX_FMT_XVMC_MPEG2_IDCT,
    PIX_FMT_UYVY422,   ///< packed YUV 4:2:2, 16bpp, Cb Y0 Cr Y1
    PIX_FMT_UYYVYY411, ///< packed YUV 4:1:1, 12bpp, Cb Y0 Y1 Cr Y2 Y3
    PIX_FMT_BGR32,     ///< packed RGB 8:8:8, 32bpp, (msb)8A 8B 8G 8R(lsb), in CPU endianness
    PIX_FMT_BGR565,    ///< packed RGB 5:6:5, 16bpp, (msb)   5B 6G 5R(lsb), in CPU endianness
    PIX_FMT_BGR555,    ///< packed RGB 5:5:5, 16bpp, (msb)1A 5B 5G 5R(lsb), in CPU endianness, most significant bit to 1
    PIX_FMT_BGR8,      ///< packed RGB 3:3:2,  8bpp, (msb)2B 3G 3R(lsb)
    PIX_FMT_BGR4,      ///< packed RGB 1:2:1,  4bpp, (msb)1B 2G 1R(lsb)
    PIX_FMT_BGR4_BYTE, ///< packed RGB 1:2:1,  8bpp, (msb)1B 2G 1R(lsb)
    PIX_FMT_RGB8,      ///< packed RGB 3:3:2,  8bpp, (msb)2R 3G 3B(lsb)
    PIX_FMT_RGB4,      ///< packed RGB 1:2:1,  4bpp, (msb)1R 2G 1B(lsb)
    PIX_FMT_RGB4_BYTE, ///< packed RGB 1:2:1,  8bpp, (msb)1R 2G 1B(lsb)
    PIX_FMT_NV12,      ///< planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 for UV
    PIX_FMT_NV21,      ///< as above, but U and V bytes are swapped

    PIX_FMT_RGB32_1,   ///< packed RGB 8:8:8, 32bpp, (msb)8R 8G 8B 8A(lsb), in CPU endianness
    PIX_FMT_BGR32_1,   ///< packed RGB 8:8:8, 32bpp, (msb)8B 8G 8R 8A(lsb), in CPU endianness

    PIX_FMT_GRAY16BE,  ///<        Y        , 16bpp, big-endian
    PIX_FMT_GRAY16LE,  ///<        Y        , 16bpp, little-endian
    PIX_FMT_YUV440P,   ///< planar YUV 4:4:0 (1 Cr & Cb sample per 1x2 Y samples)
    PIX_FMT_YUVJ440P,  ///< planar YUV 4:4:0 full scale (JPEG)
    PIX_FMT_YUVA420P,  ///< planar YUV 4:2:0, 20bpp, (1 Cr & Cb sample per 2x2 Y & A samples)
    PIX_FMT_VDPAU_H264,///< H.264 HW decoding with VDPAU, data[0] contains a vdpau_render_state struct which contains the bitstream of the slices as well as various fields extracted from headers
    PIX_FMT_VDPAU_MPEG1,///< MPEG-1 HW decoding with VDPAU, data[0] contains a vdpau_render_state struct which contains the bitstream of the slices as well as various fields extracted from headers
    PIX_FMT_VDPAU_MPEG2,///< MPEG-2 HW decoding with VDPAU, data[0] contains a vdpau_render_state struct which contains the bitstream of the slices as well as various fields extracted from headers
    PIX_FMT_VDPAU_WMV3,///< WMV3 HW decoding with VDPAU, data[0] contains a vdpau_render_state struct which contains the bitstream of the slices as well as various fields extracted from headers
    PIX_FMT_VDPAU_VC1, ///< VC-1 HW decoding with VDPAU, data[0] contains a vdpau_render_state struct which contains the bitstream of the slices as well as various fields extracted from headers
    PIX_FMT_RGB48BE,   ///< packed RGB 16:16:16, 48bpp, 16R, 16G, 16B, big-endian
    PIX_FMT_RGB48LE,   ///< packed RGB 16:16:16, 48bpp, 16R, 16G, 16B, little-endian
    PIX_FMT_VAAPI_MOCO, ///< HW acceleration through VA API at motion compensation entry-point, Picture.data[0] contains a vaapi_render_state struct which contains macroblocks as well as various fields extracted from headers
    PIX_FMT_VAAPI_IDCT, ///< HW acceleration through VA API at IDCT entry-point, Picture.data[0] contains a vaapi_render_state struct which contains fields extracted from headers
    PIX_FMT_VAAPI_VLD,  ///< HW decoding through VA API, Picture.data[0] contains a vaapi_render_state struct which contains the bitstream of the slices as well as various fields extracted from headers
    PIX_FMT_NB,        ///< number of pixel formats, DO NOT USE THIS if you want to link with shared libav* because the number of formats might differ between versions
};

#ifdef WORDS_BIGENDIAN
#define PIX_FMT_RGBA PIX_FMT_RGB32_1
#define PIX_FMT_BGRA PIX_FMT_BGR32_1
#define PIX_FMT_ARGB PIX_FMT_RGB32
#define PIX_FMT_ABGR PIX_FMT_BGR32
#define PIX_FMT_GRAY16 PIX_FMT_GRAY16BE
#define PIX_FMT_RGB48 PIX_FMT_RGB48BE
#else
#define PIX_FMT_RGBA PIX_FMT_BGR32
#define PIX_FMT_BGRA PIX_FMT_RGB32
#define PIX_FMT_ARGB PIX_FMT_BGR32_1
#define PIX_FMT_ABGR PIX_FMT_RGB32_1
#define PIX_FMT_GRAY16 PIX_FMT_GRAY16LE
#define PIX_FMT_RGB48 PIX_FMT_RGB48LE
#endif

#if LIBAVUTIL_VERSION_INT < (50<<16)
#define PIX_FMT_UYVY411 PIX_FMT_UYYVYY411
#define PIX_FMT_RGBA32  PIX_FMT_RGB32
#define PIX_FMT_YUV422  PIX_FMT_YUYV422
#endif

#endif /* AVUTIL_PIXFMT_H */
