/*
 * pixel format descriptor
 * Copyright (c) 2009 Michael Niedermayer <michaelni@gmx.at>
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

#ifndef AVUTIL_PIXDESC_H
#define AVUTIL_PIXDESC_H

#include <inttypes.h>
#include "pixfmt.h"

typedef struct AVComponentDescriptor{
    uint16_t plane        :2;            ///< which of the 4 planes contains the component

    /**
     * Number of elements between 2 horizontally consecutive pixels minus 1.
     * Elements are bits for bitstream formats, bytes otherwise.
     */
    uint16_t step_minus1  :3;

    /**
     * Number of elements before the component of the first pixel plus 1.
     * Elements are bits for bitstream formats, bytes otherwise.
     */
    uint16_t offset_plus1 :3;
    uint16_t shift        :3;            ///< number of least significant bits that must be shifted away to get the value
    uint16_t depth_minus1 :4;            ///< number of bits in the component minus 1
}AVComponentDescriptor;

/**
 * Descriptor that unambiguously describes how the bits of a pixel are
 * stored in the up to 4 data planes of an image. It also stores the
 * subsampling factors and number of components.
 *
 * @note This is separate of the colorspace (RGB, YCbCr, YPbPr, JPEG-style YUV
 *       and all the YUV variants) AVPixFmtDescriptor just stores how values
 *       are stored not what these values represent.
 */
typedef struct AVPixFmtDescriptor{
    const char *name;
    uint8_t nb_components;      ///< The number of components each pixel has, (1-4)

    /**
     * Amount to shift the luma width right to find the chroma width.
     * For YV12 this is 1 for example.
     * chroma_width = -((-luma_width) >> log2_chroma_w)
     * The note above is needed to ensure rounding up.
     * This value only refers to the chroma components.
     */
    uint8_t log2_chroma_w;      ///< chroma_width = -((-luma_width )>>log2_chroma_w)

    /**
     * Amount to shift the luma height right to find the chroma height.
     * For YV12 this is 1 for example.
     * chroma_height= -((-luma_height) >> log2_chroma_h)
     * The note above is needed to ensure rounding up.
     * This value only refers to the chroma components.
     */
    uint8_t log2_chroma_h;
    uint8_t flags;

    /**
     * Parameters that describe how pixels are packed. If the format
     * has chroma components, they must be stored in comp[1] and
     * comp[2].
     */
    AVComponentDescriptor comp[4];
}AVPixFmtDescriptor;

#define PIX_FMT_BE        1 ///< Pixel format is big-endian.
#define PIX_FMT_PAL       2 ///< Pixel format has a palette in data[1], values are indexes in this palette.
#define PIX_FMT_BITSTREAM 4 ///< All values of a component are bit-wise packed end to end.
#define PIX_FMT_HWACCEL   8 ///< Pixel format is an HW accelerated format.

/**
 * The array of all the pixel format descriptors.
 */
extern const AVPixFmtDescriptor av_pix_fmt_descriptors[];

/**
 * Read a line from an image, and write the values of the
 * pixel format component c to dst.
 *
 * @param data the array containing the pointers to the planes of the image
 * @param linesize the array containing the linesizes of the image
 * @param desc the pixel format descriptor for the image
 * @param x the horizontal coordinate of the first pixel to read
 * @param y the vertical coordinate of the first pixel to read
 * @param w the width of the line to read, that is the number of
 * values to write to dst
 * @param read_pal_component if not zero and the format is a paletted
 * format writes the values corresponding to the palette
 * component c in data[1] to dst, rather than the palette indexes in
 * data[0]. The behavior is undefined if the format is not paletted.
 */
void av_read_image_line(uint16_t *dst, const uint8_t *data[4], const int linesize[4],
                        const AVPixFmtDescriptor *desc, int x, int y, int c, int w, int read_pal_component);

/**
 * Write the values from src to the pixel format component c of an
 * image line.
 *
 * @param src array containing the values to write
 * @param data the array containing the pointers to the planes of the
 * image to write into. It is supposed to be zeroed.
 * @param linesize the array containing the linesizes of the image
 * @param desc the pixel format descriptor for the image
 * @param x the horizontal coordinate of the first pixel to write
 * @param y the vertical coordinate of the first pixel to write
 * @param w the width of the line to write, that is the number of
 * values to write to the image line
 */
void av_write_image_line(const uint16_t *src, uint8_t *data[4], const int linesize[4],
                         const AVPixFmtDescriptor *desc, int x, int y, int c, int w);

/**
 * Return the pixel format corresponding to name.
 *
 * If there is no pixel format with name name, then looks for a
 * pixel format with the name corresponding to the native endian
 * format of name.
 * For example in a little-endian system, first looks for "gray16",
 * then for "gray16le".
 *
 * Finally if no pixel format has been found, returns PIX_FMT_NONE.
 */
enum PixelFormat av_get_pix_fmt(const char *name);

/**
 * Return the short name for a pixel format, NULL in case pix_fmt is
 * unknown.
 *
 * @see av_get_pix_fmt(), av_get_pix_fmt_string()
 */
const char *av_get_pix_fmt_name(enum PixelFormat pix_fmt);

/**
 * Print in buf the string corresponding to the pixel format with
 * number pix_fmt, or an header if pix_fmt is negative.
 *
 * @param buf the buffer where to write the string
 * @param buf_size the size of buf
 * @param pix_fmt the number of the pixel format to print the
 * corresponding info string, or a negative value to print the
 * corresponding header.
 */
char *av_get_pix_fmt_string (char *buf, int buf_size, enum PixelFormat pix_fmt);

/**
 * Return the number of bits per pixel used by the pixel format
 * described by pixdesc.
 *
 * The returned number of bits refers to the number of bits actually
 * used for storing the pixel information, that is padding bits are
 * not counted.
 */
int av_get_bits_per_pixel(const AVPixFmtDescriptor *pixdesc);

#endif /* AVUTIL_PIXDESC_H */
