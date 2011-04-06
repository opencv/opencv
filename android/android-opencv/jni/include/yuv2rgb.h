/* YUV-> RGB conversion code.
 *
 * Copyright (C) 2008-9 Robin Watts (robin@wss.co.uk) for Pinknoise
 * Productions Ltd.
 *
 * Licensed under the GNU GPL. If you need it under another license, contact
 * me and ask.
 *
 *  This program is free software ; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation ; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY ; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program ; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#ifndef YUV2RGB_H

#define YUV2RGB_H

/* Define these to something appropriate in your build */
typedef unsigned int   uint32_t;
typedef signed   int   int32_t;
typedef unsigned short uint16_t;
typedef unsigned char  uint8_t;

extern const uint32_t yuv2rgb565_table[];
extern const uint32_t yuv2bgr565_table[];

void yuv420_2_rgb565(uint8_t  *dst_ptr,
               const uint8_t  *y_ptr,
               const uint8_t  *u_ptr,
               const uint8_t  *v_ptr,
                     int32_t   width,
                     int32_t   height,
                     int32_t   y_span,
                     int32_t   uv_span,
                     int32_t   dst_span,
               const uint32_t *tables,
                     int32_t   dither);

void yuv422_2_rgb565(uint8_t  *dst_ptr,
               const uint8_t  *y_ptr,
               const uint8_t  *u_ptr,
               const uint8_t  *v_ptr,
                     int32_t   width,
                     int32_t   height,
                     int32_t   y_span,
                     int32_t   uv_span,
                     int32_t   dst_span,
               const uint32_t *tables,
                     int32_t   dither);

void yuv444_2_rgb565(uint8_t  *dst_ptr,
               const uint8_t  *y_ptr,
               const uint8_t  *u_ptr,
               const uint8_t  *v_ptr,
                     int32_t   width,
                     int32_t   height,
                     int32_t   y_span,
                     int32_t   uv_span,
                     int32_t   dst_span,
               const uint32_t *tables,
                     int32_t   dither);

void yuv420_2_rgb888(uint8_t  *dst_ptr,
               const uint8_t  *y_ptr,
               const uint8_t  *u_ptr,
               const uint8_t  *v_ptr,
                     int32_t   width,
                     int32_t   height,
                     int32_t   y_span,
                     int32_t   uv_span,
                     int32_t   dst_span,
               const uint32_t *tables,
                     int32_t   dither);

void yuv422_2_rgb888(uint8_t  *dst_ptr,
               const uint8_t  *y_ptr,
               const uint8_t  *u_ptr,
               const uint8_t  *v_ptr,
                     int32_t   width,
                     int32_t   height,
                     int32_t   y_span,
                     int32_t   uv_span,
                     int32_t   dst_span,
               const uint32_t *tables,
                     int32_t   dither);

void yuv444_2_rgb888(uint8_t  *dst_ptr,
               const uint8_t  *y_ptr,
               const uint8_t  *u_ptr,
               const uint8_t  *v_ptr,
                     int32_t   width,
                     int32_t   height,
                     int32_t   y_span,
                     int32_t   uv_span,
                     int32_t   dst_span,
               const uint32_t *tables,
                     int32_t   dither);

void yuv420_2_rgb8888(uint8_t  *dst_ptr,
                const uint8_t  *y_ptr,
                const uint8_t  *u_ptr,
                const uint8_t  *v_ptr,
                      int32_t   width,
                      int32_t   height,
                      int32_t   y_span,
                      int32_t   uv_span,
                      int32_t   dst_span,
                const uint32_t *tables,
                      int32_t   dither);

void yuv422_2_rgb8888(uint8_t  *dst_ptr,
                const uint8_t  *y_ptr,
                const uint8_t  *u_ptr,
                const uint8_t  *v_ptr,
                      int32_t   width,
                      int32_t   height,
                      int32_t   y_span,
                      int32_t   uv_span,
                      int32_t   dst_span,
                const uint32_t *tables,
                      int32_t   dither);

void yuv444_2_rgb8888(uint8_t  *dst_ptr,
                const uint8_t  *y_ptr,
                const uint8_t  *u_ptr,
                const uint8_t  *v_ptr,
                      int32_t   width,
                      int32_t   height,
                      int32_t   y_span,
                      int32_t   uv_span,
                      int32_t   dst_span,
                const uint32_t *tables,
                      int32_t   dither);


#endif /* YUV2RGB_H */
