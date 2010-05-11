/*
 *  software RGB to RGB converter
 *  pluralize by Software PAL8 to RGB converter
 *               Software YUV to YUV converter
 *               Software YUV to RGB converter
 *  Written by Nick Kurshev.
 *  palette & YUV & runtime CPU stuff by Michael (michaelni@gmx.at)
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

#ifndef SWSCALE_RGB2RGB_H
#define SWSCALE_RGB2RGB_H

#include <inttypes.h>

/* A full collection of RGB to RGB(BGR) converters */
extern void (*rgb24tobgr32)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb24tobgr16)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb24tobgr15)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb32tobgr24)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb32to16)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb32to15)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb15to16)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb15tobgr24)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb15to32)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb16to15)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb16tobgr24)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb16to32)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb24tobgr24)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb24to16)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb24to15)   (const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb32tobgr32)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb32tobgr16)(const uint8_t *src, uint8_t *dst, long src_size);
extern void (*rgb32tobgr15)(const uint8_t *src, uint8_t *dst, long src_size);

void rgb24to32   (const uint8_t *src, uint8_t *dst, long src_size);
void rgb32to24   (const uint8_t *src, uint8_t *dst, long src_size);
void rgb16tobgr32(const uint8_t *src, uint8_t *dst, long src_size);
void rgb16to24   (const uint8_t *src, uint8_t *dst, long src_size);
void rgb16tobgr16(const uint8_t *src, uint8_t *dst, long src_size);
void rgb16tobgr15(const uint8_t *src, uint8_t *dst, long src_size);
void rgb15tobgr32(const uint8_t *src, uint8_t *dst, long src_size);
void rgb15to24   (const uint8_t *src, uint8_t *dst, long src_size);
void rgb15tobgr16(const uint8_t *src, uint8_t *dst, long src_size);
void rgb15tobgr15(const uint8_t *src, uint8_t *dst, long src_size);
void bgr8torgb8  (const uint8_t *src, uint8_t *dst, long src_size);


void palette8topacked32(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette);
void palette8topacked24(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette);
void palette8torgb16(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette);
void palette8tobgr16(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette);
void palette8torgb15(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette);
void palette8tobgr15(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette);

/**
 * Height should be a multiple of 2 and width should be a multiple of 16.
 * (If this is a problem for anyone then tell me, and I will fix it.)
 * Chrominance data is only taken from every second line, others are ignored.
 * FIXME: Write high quality version.
 */
//void uyvytoyv12(const uint8_t *src, uint8_t *ydst, uint8_t *udst, uint8_t *vdst,

/**
 * Height should be a multiple of 2 and width should be a multiple of 16.
 * (If this is a problem for anyone then tell me, and I will fix it.)
 */
extern void (*yv12toyuy2)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc, uint8_t *dst,
                          long width, long height,
                          long lumStride, long chromStride, long dstStride);

/**
 * Width should be a multiple of 16.
 */
extern void (*yuv422ptoyuy2)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc, uint8_t *dst,
                             long width, long height,
                             long lumStride, long chromStride, long dstStride);

/**
 * Height should be a multiple of 2 and width should be a multiple of 16.
 * (If this is a problem for anyone then tell me, and I will fix it.)
 */
extern void (*yuy2toyv12)(const uint8_t *src, uint8_t *ydst, uint8_t *udst, uint8_t *vdst,
                          long width, long height,
                          long lumStride, long chromStride, long srcStride);

/**
 * Height should be a multiple of 2 and width should be a multiple of 16.
 * (If this is a problem for anyone then tell me, and I will fix it.)
 */
extern void (*yv12touyvy)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc, uint8_t *dst,
                          long width, long height,
                          long lumStride, long chromStride, long dstStride);

/**
 * Width should be a multiple of 16.
 */
extern void (*yuv422ptouyvy)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc, uint8_t *dst,
                             long width, long height,
                             long lumStride, long chromStride, long dstStride);

/**
 * Height should be a multiple of 2 and width should be a multiple of 2.
 * (If this is a problem for anyone then tell me, and I will fix it.)
 * Chrominance data is only taken from every second line, others are ignored.
 * FIXME: Write high quality version.
 */
extern void (*rgb24toyv12)(const uint8_t *src, uint8_t *ydst, uint8_t *udst, uint8_t *vdst,
                           long width, long height,
                           long lumStride, long chromStride, long srcStride);
extern void (*planar2x)(const uint8_t *src, uint8_t *dst, long width, long height,
                        long srcStride, long dstStride);

extern void (*interleaveBytes)(uint8_t *src1, uint8_t *src2, uint8_t *dst,
                               long width, long height, long src1Stride,
                               long src2Stride, long dstStride);

extern void (*vu9_to_vu12)(const uint8_t *src1, const uint8_t *src2,
                           uint8_t *dst1, uint8_t *dst2,
                           long width, long height,
                           long srcStride1, long srcStride2,
                           long dstStride1, long dstStride2);

extern void (*yvu9_to_yuy2)(const uint8_t *src1, const uint8_t *src2, const uint8_t *src3,
                            uint8_t *dst,
                            long width, long height,
                            long srcStride1, long srcStride2,
                            long srcStride3, long dstStride);

void sws_rgb2rgb_init(int flags);

#endif /* SWSCALE_RGB2RGB_H */
