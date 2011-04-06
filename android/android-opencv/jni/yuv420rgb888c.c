/* YUV-> RGB conversion code. (YUV420 to RGB565)
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
 *
 * The algorithm used here is based heavily on one created by Sophie Wilson
 * of Acorn/e-14/Broadcomm. Many thanks.
 *
 * Additional tweaks (in the fast fixup code) are from Paul Gardiner.
 *
 * The old implementation of YUV -> RGB did:
 *
 * R = CLAMP((Y-16)*1.164 +           1.596*V)
 * G = CLAMP((Y-16)*1.164 - 0.391*U - 0.813*V)
 * B = CLAMP((Y-16)*1.164 + 2.018*U          )
 *
 * We're going to bend that here as follows:
 *
 * R = CLAMP(y +           1.596*V)
 * G = CLAMP(y - 0.383*U - 0.813*V)
 * B = CLAMP(y + 1.976*U          )
 *
 * where y = 0               for       Y <=  16,
 *       y = (  Y-16)*1.164, for  16 < Y <= 239,
 *       y = (239-16)*1.164, for 239 < Y
 *
 * i.e. We clamp Y to the 16 to 239 range (which it is supposed to be in
 * anyway). We then pick the B_U factor so that B never exceeds 511. We then
 * shrink the G_U factor in line with that to avoid a colour shift as much as
 * possible.
 *
 * We're going to use tables to do it faster, but rather than doing it using
 * 5 tables as as the above suggests, we're going to do it using just 3.
 *
 * We do this by working in parallel within a 32 bit word, and using one
 * table each for Y U and V.
 *
 * Source Y values are    0 to 255, so    0.. 260 after scaling
 * Source U values are -128 to 127, so  -49.. 49(G), -253..251(B) after
 * Source V values are -128 to 127, so -204..203(R), -104..103(G) after
 *
 * So total summed values:
 * -223 <= R <= 481, -173 <= G <= 431, -253 <= B < 511
 *
 * We need to pack R G and B into a 32 bit word, and because of Bs range we
 * need 2 bits above the valid range of B to detect overflow, and another one
 * to detect the sense of the overflow. We therefore adopt the following
 * representation:
 *
 * osGGGGGgggggosBBBBBbbbosRRRRRrrr
 *
 * Each such word breaks down into 3 ranges.
 *
 * osGGGGGggggg   osBBBBBbbb   osRRRRRrrr
 *
 * Thus we have 8 bits for each B and R table entry, and 10 bits for G (good
 * as G is the most noticable one). The s bit for each represents the sign,
 * and o represents the overflow.
 *
 * For R and B we pack the table by taking the 11 bit representation of their
 * values, and toggling bit 10 in the U and V tables.
 *
 * For the green case we calculate 4*G (thus effectively using 10 bits for the
 * valid range) truncate to 12 bits. We toggle bit 11 in the Y table.
 */

#include "yuv2rgb.h"

enum
{
    FLAGS         = 0x40080100
};

#define READUV(U,V) (tables[256 + (U)] + tables[512 + (V)])
#define READY(Y)    tables[Y]
#define FIXUP(Y)                 \
do {                             \
    int tmp = (Y) & FLAGS;       \
    if (tmp != 0)                \
    {                            \
        tmp  -= tmp>>8;          \
        (Y)  |= tmp;             \
        tmp   = FLAGS & ~(Y>>1); \
        (Y)  += tmp>>8;          \
    }                            \
} while (0 == 1)

#define STORE(Y,DSTPTR)           \
do {                              \
    uint32_t Y2       = (Y);      \
    uint8_t  *DSTPTR2 = (DSTPTR); \
    (DSTPTR2)[0] = (Y2);          \
    (DSTPTR2)[1] = (Y2)>>22;      \
    (DSTPTR2)[2] = (Y2)>>11;      \
} while (0 == 1)

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
                     int32_t   dither)
{
    height -= 1;
    while (height > 0)
    {
        height -= width<<16;
        height += 1<<16;
        while (height < 0)
        {
            /* Do 2 column pairs */
            uint32_t uv, y0, y1;

            uv  = READUV(*u_ptr++,*v_ptr++);
            y1  = uv + READY(y_ptr[y_span]);
            y0  = uv + READY(*y_ptr++);
            FIXUP(y1);
            FIXUP(y0);
            STORE(y1, &dst_ptr[dst_span]);
            STORE(y0, dst_ptr);
            dst_ptr += 3;
            y1  = uv + READY(y_ptr[y_span]);
            y0  = uv + READY(*y_ptr++);
            FIXUP(y1);
            FIXUP(y0);
            STORE(y1, &dst_ptr[dst_span]);
            STORE(y0, dst_ptr);
            dst_ptr += 3;
            height += (2<<16);
        }
        if ((height>>16) == 0)
        {
            /* Trailing column pair */
            uint32_t uv, y0, y1;

            uv = READUV(*u_ptr,*v_ptr);
            y1 = uv + READY(y_ptr[y_span]);
            y0 = uv + READY(*y_ptr++);
            FIXUP(y1);
            FIXUP(y0);
            STORE(y0, &dst_ptr[dst_span]);
            STORE(y1, dst_ptr);
            dst_ptr += 3;
        }
        dst_ptr += dst_span*2-width*3;
        y_ptr   += y_span*2-width;
        u_ptr   += uv_span-(width>>1);
        v_ptr   += uv_span-(width>>1);
        height = (height<<16)>>16;
        height -= 2;
    }
    if (height == 0)
    {
        /* Trail row */
        height -= width<<16;
        height += 1<<16;
        while (height < 0)
        {
            /* Do a row pair */
            uint32_t uv, y0, y1;

            uv  = READUV(*u_ptr++,*v_ptr++);
            y1  = uv + READY(*y_ptr++);
            y0  = uv + READY(*y_ptr++);
            FIXUP(y1);
            FIXUP(y0);
            STORE(y1, dst_ptr);
            dst_ptr += 3;
            STORE(y0, dst_ptr);
            dst_ptr += 3;
            height += (2<<16);
        }
        if ((height>>16) == 0)
        {
            /* Trailing pix */
            uint32_t uv, y0;

            uv = READUV(*u_ptr++,*v_ptr++);
            y0 = uv + READY(*y_ptr++);
            FIXUP(y0);
            STORE(y0, dst_ptr);
            dst_ptr += 3;
        }
    }
}
