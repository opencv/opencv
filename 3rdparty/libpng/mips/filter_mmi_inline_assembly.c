/* filter_mmi_intrinsics.c - MMI optimized filter functions
 *
 * Copyright (c) 2024 Cosmin Truta
 * Written by zhanglixia and guxiwei, 2023
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED

#if PNG_MIPS_MMI_IMPLEMENTATION == 2 /* Inline Assembly */

/* Functions in this file look at most 3 pixels (a,b,c) to predict the 4th (d).
 * They're positioned like this:
 *    prev:  c b
 *    row:   a d
 * The Sub filter predicts d=a, Avg d=(a+b)/2, and Paeth predicts d to be
 * whichever of a, b, or c is closest to p=a+b-c.
 */

void png_read_filter_row_up_mmi(png_row_infop row_info, png_bytep row,
                                png_const_bytep prev_row)
{
   int istop = row_info->rowbytes;
   double rp,pp;
   __asm__ volatile (
       "1:                                          \n\t"
       "ldc1   %[rp],       0x00(%[row])            \n\t"
       "ldc1   %[pp],       0x00(%[prev_row])       \n\t"
       "paddb  %[rp],       %[rp],            %[pp] \n\t"
       "sdc1   %[rp],       0x00(%[row])            \n\t"

       "daddiu %[row],      %[row],           0x08  \n\t"
       "daddiu %[prev_row], %[prev_row],      0x08  \n\t"
       "daddiu %[istop],    %[istop],        -0x08  \n\t"
       "bgtz   %[istop],    1b                      \n\t"
       : [rp]"=&f"(rp), [pp]"=&f"(pp)
       : [row]"r"(row), [prev_row]"r"(prev_row),
         [istop]"r"(istop)
       : "memory"
   );
}

void png_read_filter_row_sub3_mmi(png_row_infop row_info, png_bytep row,
   png_const_bytep prev)
{
   int istop = row_info->rowbytes;
   double rp, pp, dest;
   double eight, sixteen, twenty_four, forty_eight;
   double tmp0;
   double ftmp[2];

   __asm__ volatile (
        "li         %[tmp0],    0x08                          \n\t"
        "dmtc1      %[tmp0],    %[eight]                      \n\t"
        "li         %[tmp0],    0x10                          \n\t"
        "dmtc1      %[tmp0],    %[sixteen]                    \n\t"
        "li         %[tmp0],    0x18                          \n\t"
        "dmtc1      %[tmp0],    %[twenty_four]                \n\t"
        "li         %[tmp0],    0x30                          \n\t"
        "dmtc1      %[tmp0],    %[forty_eight]                \n\t"
        "xor        %[dest],    %[dest],       %[dest]        \n\t"

        "1:                                                   \n\t"
        "gsldrc1    %[rp],      0x00(%[row])                  \n\t"
        "gsldlc1    %[rp],      0x07(%[row])                  \n\t"
        "gsldrc1    %[pp],      0x08(%[row])                  \n\t"
        "gsldlc1    %[pp],      0x0f(%[row])                  \n\t"

        "paddb      %[ftmp0],   %[dest],      %[rp]           \n\t"
        "swc1       %[ftmp0],   0x00(%[row])                  \n\t"

        "dsrl       %[ftmp1],   %[rp],        %[twenty_four]  \n\t"
        "paddb      %[dest],    %[ftmp1],     %[ftmp0]        \n\t"
        "gsswrc1    %[dest],    0x03(%[row])                  \n\t"
        "gsswlc1    %[dest],    0x06(%[row])                  \n\t"

        "dsrl       %[ftmp0],   %[rp],        %[forty_eight]  \n\t"
        "dsll       %[ftmp1],   %[pp],        %[sixteen]      \n\t"
        "or         %[ftmp0],   %[ftmp0],     %[ftmp1]        \n\t"
        "paddb      %[dest],    %[dest],      %[ftmp0]        \n\t"
        "gsswrc1    %[dest],    0x06(%[row])                  \n\t"
        "gsswlc1    %[dest],    0x09(%[row])                  \n\t"

        "dsrl       %[ftmp0],   %[pp],        %[eight]        \n\t"
        "paddb      %[dest],    %[dest],      %[ftmp0]        \n\t"
        "gsswrc1    %[dest],    0x09(%[row])                  \n\t"
        "daddiu     %[row],     %[row],       0x0c            \n\t"
        "daddiu     %[istop],   %[istop],    -0x0c            \n\t"
        "bgtz       %[istop],   1b                            \n\t"
        : [rp]"=&f"(rp), [pp]"=&f"(pp), [dest]"=&f"(dest),
          [tmp0]"=&r"(tmp0), [ftmp0]"=&f"(ftmp[0]),
          [ftmp1]"=&f"(ftmp[1]), [eight]"=&f"(eight),
          [sixteen]"=&f"(sixteen), [twenty_four]"=&f"(twenty_four),
          [forty_eight]"=&f"(forty_eight)
        : [row]"r"(row), [istop]"r"(istop)
        : "memory"
   );

   PNG_UNUSED(prev)
}

void png_read_filter_row_sub4_mmi(png_row_infop row_info, png_bytep row,
   png_const_bytep prev)
{
   /* The Sub filter predicts each pixel as the previous pixel, a.
    * There is no pixel to the left of the first pixel.  It's encoded directly.
    * That works with our main loop if we just say that left pixel was zero.
    */
   int istop = row_info->rowbytes;
   double rp,pp;

   __asm__ volatile (
        "1:                                          \n\t"
        "lwc1   %[pp],       0x00(%[row])            \n\t"
        "lwc1   %[rp],       0x04(%[row])            \n\t"
        "paddb  %[rp],       %[rp],       %[pp]      \n\t"
        "swc1   %[rp],       0x04(%[row])            \n\t"

        "daddiu %[row],      %[row],      0x04       \n\t"
        "daddiu %[istop],    %[istop],   -0x04       \n\t"
        "bgtz   %[istop],    1b                      \n\t"
        : [rp]"=&f"(rp), [pp]"=&f"(pp)
        : [row]"r"(row), [istop]"r"(istop)
        : "memory"
   );

   PNG_UNUSED(prev)
}

void png_read_filter_row_avg3_mmi(png_row_infop row_info, png_bytep row,
   png_const_bytep prev)
{
   int istop = row_info->rowbytes;
   double rp, pp, rp1, pp1;
   double tmp0;
   double ftmp[3];
   double one, dest;
   double eight, sixteen, twenty_four, forty_eight;

   __asm__ volatile (
        "li         %[tmp0],    0x08                          \n\t"
        "dmtc1      %[tmp0],    %[eight]                      \n\t"
        "li         %[tmp0],    0x10                          \n\t"
        "dmtc1      %[tmp0],    %[sixteen]                    \n\t"
        "li         %[tmp0],    0x18                          \n\t"
        "dmtc1      %[tmp0],    %[twenty_four]                \n\t"
        "li         %[tmp0],    0x30                          \n\t"
        "dmtc1      %[tmp0],    %[forty_eight]                \n\t"
        "xor        %[dest],    %[dest],       %[dest]        \n\t"

        "li         %[tmp0],   0x01                           \n\t"
        "ins        %[tmp0],   %[tmp0],        8,   8         \n\t"
        "dmtc1      %[tmp0],   %[one]                         \n\t"
        "pshufh     %[one],    %[one],         %[dest]        \n\t"

        "1:                                                   \n\t"
        "gsldrc1    %[rp],      0x00(%[row])                  \n\t"
        "gsldlc1    %[rp],      0x07(%[row])                  \n\t"
        "gsldrc1    %[pp],      0x00(%[prev])                 \n\t"
        "gsldlc1    %[pp],      0x07(%[prev])                 \n\t"
        "gsldrc1    %[rp1],     0x08(%[row])                  \n\t"
        "gsldlc1    %[rp1],     0x0f(%[row])                  \n\t"
        "gsldrc1    %[pp1],     0x08(%[prev])                 \n\t"
        "gsldlc1    %[pp1],     0x0f(%[prev])                 \n\t"

        "xor        %[ftmp0],   %[pp],         %[dest]        \n\t"
        "pavgb      %[ftmp1],   %[pp],         %[dest]        \n\t"
        "and        %[ftmp0],   %[ftmp0],      %[one]         \n\t"
        "psubb      %[ftmp1],   %[ftmp1],      %[ftmp0]       \n\t"
        "paddb      %[dest],    %[rp],         %[ftmp1]       \n\t"
        "swc1       %[dest],    0x00(%[row])                  \n\t"

        "dsrl       %[ftmp0],   %[rp],         %[twenty_four] \n\t"
        "dsrl       %[ftmp1],   %[pp],         %[twenty_four] \n\t"

        "xor        %[ftmp2],   %[ftmp1],      %[dest]        \n\t"
        "pavgb      %[ftmp1],   %[ftmp1],      %[dest]        \n\t"
        "and        %[ftmp2],   %[ftmp2],      %[one]         \n\t"
        "psubb      %[ftmp1],   %[ftmp1],      %[ftmp2]       \n\t"
        "paddb      %[dest],    %[ftmp0],      %[ftmp1]       \n\t"
        "gsswrc1    %[dest],    0x03(%[row])                  \n\t"
        "gsswlc1    %[dest],    0x06(%[row])                  \n\t"

        "dsrl       %[ftmp0],   %[rp],         %[forty_eight] \n\t"
        "dsll       %[ftmp1],   %[rp1],        %[sixteen]     \n\t"
        "or         %[ftmp0],   %[ftmp0],      %[ftmp1]       \n\t"
        "dsrl       %[ftmp2],   %[pp],         %[forty_eight] \n\t"
        "dsll       %[ftmp1],   %[pp1],        %[sixteen]     \n\t"
        "or         %[ftmp1],   %[ftmp2],      %[ftmp1]       \n\t"

        "xor        %[ftmp2],   %[ftmp1],      %[dest]        \n\t"
        "pavgb      %[ftmp1],   %[ftmp1],      %[dest]        \n\t"
        "and        %[ftmp2],   %[ftmp2],      %[one]         \n\t"
        "psubb      %[ftmp1],   %[ftmp1],      %[ftmp2]       \n\t"
        "paddb      %[dest],    %[ftmp0],      %[ftmp1]       \n\t"
        "gsswrc1    %[dest],    0x06(%[row])                  \n\t"
        "gsswlc1    %[dest],    0x09(%[row])                  \n\t"

        "dsrl       %[ftmp0],   %[rp1],        %[eight]       \n\t"
        "dsrl       %[ftmp1],   %[pp1],        %[eight]       \n\t"

        "xor        %[ftmp2],   %[ftmp1],      %[dest]        \n\t"
        "pavgb      %[ftmp1],   %[ftmp1],      %[dest]        \n\t"
        "and        %[ftmp2],   %[ftmp2],      %[one]         \n\t"
        "psubb      %[ftmp1],   %[ftmp1],      %[ftmp2]       \n\t"
        "paddb      %[dest],    %[ftmp0],      %[ftmp1]       \n\t"
        "gsswrc1    %[dest],    0x09(%[row])                  \n\t"
        "daddiu     %[row],     %[row],        0x0c           \n\t"
        "daddiu     %[prev],    %[prev],       0x0c           \n\t"
        "daddiu     %[istop],   %[istop],     -0x0c           \n\t"
        "bgtz       %[istop],   1b                            \n\t"
        : [rp]"=&f"(rp), [pp]"=&f"(pp), [rp1]"=&f"(rp1),
          [pp1]"=&f"(pp1), [tmp0]"=&r"(tmp0), [ftmp0]"=&f"(ftmp[0]),
          [ftmp1]"=&f"(ftmp[1]), [ftmp2]"=&f"(ftmp[2]), [one]"=&f"(one),
          [dest]"=&f"(dest), [eight]"=&f"(eight), [sixteen]"=&f"(sixteen),
          [twenty_four]"=&f"(twenty_four), [forty_eight]"=&f"(forty_eight)
        : [row]"r"(row), [prev]"r"(prev), [istop]"r"(istop)
        : "memory"
   );
}

void png_read_filter_row_avg4_mmi(png_row_infop row_info, png_bytep row,
   png_const_bytep prev)
{
   int istop = row_info->rowbytes;
   double rp,pp;
   double dest;
   double ftmp[2];
   double tmp;

   __asm__ volatile (
        "xor        %[dest],   %[dest],       %[dest]  \n\t"
        "li         %[tmp],    0x01                    \n\t"
        "ins        %[tmp],    %[tmp],        8,  8    \n\t"
        "dmtc1      %[tmp],    %[ftmp1]                \n\t"
        "pshufh     %[ftmp1],  %[ftmp1],      %[dest]  \n\t"

        "1:                                            \n\t"
        "lwc1       %[rp],     0x00(%[row])            \n\t"
        "lwc1       %[pp],     0x00(%[prev])           \n\t"
        "xor        %[ftmp0],  %[pp],         %[dest]  \n\t"
        "pavgb      %[pp],     %[pp],         %[dest]  \n\t"
        "and        %[ftmp0],  %[ftmp0],      %[ftmp1] \n\t"
        "psubb      %[pp],     %[pp],         %[ftmp0] \n\t"
        "paddb      %[dest],   %[rp],         %[pp]    \n\t"
        "swc1       %[dest],   0x00(%[row])            \n\t"
        "daddiu     %[row],    %[row],        0x04     \n\t"
        "daddiu     %[prev],   %[prev],       0x04     \n\t"
        "daddiu     %[istop],  %[istop],     -0x04     \n\t"
        "bgtz       %[istop],  1b                      \n\t"
        : [rp]"=&f"(rp), [pp]"=&f"(pp), [ftmp0]"=&f"(ftmp[0]),
          [ftmp1]"=&f"(ftmp[1]), [dest]"=&f"(dest), [tmp]"=&r"(tmp)
        : [row]"r"(row), [prev]"r"(prev), [istop]"r"(istop)
        : "memory"
   );
}

void png_read_filter_row_paeth3_mmi(png_row_infop row_info, png_bytep row,
   png_const_bytep prev)
{
   /* Paeth tries to predict pixel d using the pixel to the left of it, a,
    * and two pixels from the previous row, b and c:
    *   prev: c b
    *   row:  a d
    * The Paeth function predicts d to be whichever of a, b, or c is nearest to
    * p=a+b-c.
    *
    * The first pixel has no left context, and so uses an Up filter, p = b.
    * This works naturally with our main loop's p = a+b-c if we force a and c
    * to zero.
    * Here we zero b and d, which become c and a respectively at the start of
    * the loop.
    */
   int istop = row_info->rowbytes;
   double rp, pp, rp1, pp1, zero;
   double a, b, c, d, pa, pb, pc;
   double tmp0;
   double ftmp[3];
   double eight, sixteen, twenty_four, forty_eight;

   __asm__ volatile (
        "xor        %[a],      %[a],           %[a]           \n\t"
        "xor        %[c],      %[c],           %[c]           \n\t"
        "xor        %[zero],   %[zero],        %[zero]        \n\t"
        "li         %[tmp0],    0x08                          \n\t"
        "dmtc1      %[tmp0],    %[eight]                      \n\t"
        "li         %[tmp0],    0x10                          \n\t"
        "dmtc1      %[tmp0],    %[sixteen]                    \n\t"
        "li         %[tmp0],    0x18                          \n\t"
        "dmtc1      %[tmp0],    %[twenty_four]                \n\t"
        "li         %[tmp0],    0x30                          \n\t"
        "dmtc1      %[tmp0],    %[forty_eight]                \n\t"

        "1:                                                   \n\t"
        "gsldrc1    %[rp],      0x00(%[row])                  \n\t"
        "gsldlc1    %[rp],      0x07(%[row])                  \n\t"
        "gsldrc1    %[pp],      0x00(%[prev])                 \n\t"
        "gsldlc1    %[pp],      0x07(%[prev])                 \n\t"
        "gsldrc1    %[rp1],     0x08(%[row])                  \n\t"
        "gsldlc1    %[rp1],     0x0f(%[row])                  \n\t"
        "gsldrc1    %[pp1],     0x08(%[prev])                 \n\t"
        "gsldlc1    %[pp1],     0x0f(%[prev])                 \n\t"

        "punpcklbh  %[b],      %[pp],          %[zero]        \n\t"
        "punpcklbh  %[d],      %[rp],          %[zero]        \n\t"
        "packushb   %[ftmp0],  %[c],           %[c]           \n\t"
        "packushb   %[ftmp1],  %[a],           %[a]           \n\t"
        "pasubub    %[pa],     %[pp],          %[ftmp0]       \n\t"
        "pasubub    %[pb],     %[ftmp1],       %[ftmp0]       \n\t"
        "psubh      %[ftmp0],  %[b],           %[c]           \n\t"
        "psubh      %[ftmp1],  %[a],           %[c]           \n\t"
        "paddh      %[pc],     %[ftmp0],       %[ftmp1]       \n\t"
        "pcmpgth    %[ftmp0],  %[zero],        %[pc]          \n\t"
        "xor        %[pc],     %[pc],          %[ftmp0]       \n\t"
        "psubh      %[pc],     %[pc],          %[ftmp0]       \n\t"
        "punpcklbh  %[pa],     %[pa],          %[zero]        \n\t"
        "punpcklbh  %[pb],     %[pb],          %[zero]        \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pb]          \n\t"
        "and        %[ftmp1],  %[b],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "pminsh     %[pa],     %[pa],          %[pb]          \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pc]          \n\t"
        "and        %[ftmp1],  %[c],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "paddb      %[a],      %[a],           %[d]           \n\t"
        "packushb   %[d],      %[a],           %[a]           \n\t"
        "punpcklbh  %[c],      %[pp],          %[zero]        \n\t"
        "swc1       %[d],      0x00(%[row])                   \n\t"

        "dsrl       %[ftmp0],  %[rp],          %[twenty_four] \n\t"
        "dsrl       %[ftmp2],  %[pp],          %[twenty_four] \n\t"

        "punpcklbh  %[b],      %[ftmp2],       %[zero]        \n\t"
        "punpcklbh  %[d],      %[ftmp0],       %[zero]        \n\t"
        "packushb   %[ftmp0],  %[c],           %[c]           \n\t"
        "packushb   %[ftmp1],  %[a],           %[a]           \n\t"
        "pasubub    %[pa],     %[ftmp2],       %[ftmp0]       \n\t"
        "pasubub    %[pb],     %[ftmp1],       %[ftmp0]       \n\t"
        "psubh      %[ftmp0],  %[b],           %[c]           \n\t"
        "psubh      %[ftmp1],  %[a],           %[c]           \n\t"
        "paddh      %[pc],     %[ftmp0],       %[ftmp1]       \n\t"
        "pcmpgth    %[ftmp0],  %[zero],        %[pc]          \n\t"
        "xor        %[pc],     %[pc],          %[ftmp0]       \n\t"
        "psubh      %[pc],     %[pc],          %[ftmp0]       \n\t"
        "punpcklbh  %[pa],     %[pa],          %[zero]        \n\t"
        "punpcklbh  %[pb],     %[pb],          %[zero]        \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pb]          \n\t"
        "and        %[ftmp1],  %[b],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "pminsh     %[pa],     %[pa],          %[pb]          \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pc]          \n\t"
        "and        %[ftmp1],  %[c],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "paddb      %[a],      %[a],           %[d]           \n\t"
        "packushb   %[d],      %[a],           %[a]           \n\t"
        "punpcklbh  %[c],      %[ftmp2],       %[zero]        \n\t"
        "gsswrc1    %[d],      0x03(%[row])                   \n\t"
        "gsswlc1    %[d],      0x06(%[row])                   \n\t"

        "dsrl       %[ftmp0],  %[rp],          %[forty_eight] \n\t"
        "dsll       %[ftmp1],  %[rp1],         %[sixteen]     \n\t"
        "or         %[ftmp0],  %[ftmp0],       %[ftmp1]       \n\t"
        "dsrl       %[ftmp2],  %[pp],          %[forty_eight] \n\t"
        "dsll       %[ftmp1],  %[pp1],         %[sixteen]     \n\t"
        "or         %[ftmp2],  %[ftmp2],       %[ftmp1]       \n\t"

        "punpcklbh  %[b],      %[ftmp2],       %[zero]        \n\t"
        "punpcklbh  %[d],      %[ftmp0],       %[zero]        \n\t"
        "packushb   %[ftmp0],  %[c],           %[c]           \n\t"
        "packushb   %[ftmp1],  %[a],           %[a]           \n\t"
        "pasubub    %[pa],     %[ftmp2],       %[ftmp0]       \n\t"
        "pasubub    %[pb],     %[ftmp1],       %[ftmp0]       \n\t"
        "psubh      %[ftmp0],  %[b],           %[c]           \n\t"
        "psubh      %[ftmp1],  %[a],           %[c]           \n\t"
        "paddh      %[pc],     %[ftmp0],       %[ftmp1]       \n\t"
        "pcmpgth    %[ftmp0],  %[zero],        %[pc]          \n\t"
        "xor        %[pc],     %[pc],          %[ftmp0]       \n\t"
        "psubh      %[pc],     %[pc],          %[ftmp0]       \n\t"
        "punpcklbh  %[pa],     %[pa],          %[zero]        \n\t"
        "punpcklbh  %[pb],     %[pb],          %[zero]        \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pb]          \n\t"
        "and        %[ftmp1],  %[b],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "pminsh     %[pa],     %[pa],          %[pb]          \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pc]          \n\t"
        "and        %[ftmp1],  %[c],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "paddb      %[a],      %[a],           %[d]           \n\t"
        "packushb   %[d],      %[a],           %[a]           \n\t"
        "punpcklbh  %[c],      %[ftmp2],       %[zero]        \n\t"
        "gsswrc1    %[d],      0x06(%[row])                   \n\t"
        "gsswlc1    %[d],      0x09(%[row])                   \n\t"

        "dsrl       %[ftmp0],   %[rp1],        %[eight]       \n\t"
        "dsrl       %[ftmp2],   %[pp1],        %[eight]       \n\t"

        "punpcklbh  %[b],      %[ftmp2],       %[zero]        \n\t"
        "punpcklbh  %[d],      %[ftmp0],       %[zero]        \n\t"
        "packushb   %[ftmp0],  %[c],           %[c]           \n\t"
        "packushb   %[ftmp1],  %[a],           %[a]           \n\t"
        "pasubub    %[pa],     %[ftmp2],       %[ftmp0]       \n\t"
        "pasubub    %[pb],     %[ftmp1],       %[ftmp0]       \n\t"
        "psubh      %[ftmp0],  %[b],           %[c]           \n\t"
        "psubh      %[ftmp1],  %[a],           %[c]           \n\t"
        "paddh      %[pc],     %[ftmp0],       %[ftmp1]       \n\t"
        "pcmpgth    %[ftmp0],  %[zero],        %[pc]          \n\t"
        "xor        %[pc],     %[pc],          %[ftmp0]       \n\t"
        "psubh      %[pc],     %[pc],          %[ftmp0]       \n\t"
        "punpcklbh  %[pa],     %[pa],          %[zero]        \n\t"
        "punpcklbh  %[pb],     %[pb],          %[zero]        \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pb]          \n\t"
        "and        %[ftmp1],  %[b],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "pminsh     %[pa],     %[pa],          %[pb]          \n\t"
        "pcmpgth    %[ftmp0],  %[pa],          %[pc]          \n\t"
        "and        %[ftmp1],  %[c],           %[ftmp0]       \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]           \n\t"
        "or         %[a],      %[a],           %[ftmp1]       \n\t"
        "paddb      %[a],      %[a],           %[d]           \n\t"
        "packushb   %[d],      %[a],           %[a]           \n\t"
        "punpcklbh  %[c],      %[ftmp2],       %[zero]        \n\t"
        "gsswrc1    %[d],      0x09(%[row])                   \n\t"

        "daddiu     %[row],    %[row],         0x0c           \n\t"
        "daddiu     %[prev],   %[prev],        0x0c           \n\t"
        "daddiu     %[istop],  %[istop],      -0x0c           \n\t"
        "bgtz       %[istop],  1b                             \n\t"
        : [rp]"=&f"(rp), [pp]"=&f"(pp), [rp1]"=&f"(rp1), [pp1]"=&f"(pp1),
          [zero]"=&f"(zero), [a]"=&f"(a),[b]"=&f"(b), [c]"=&f"(c),
          [d]"=&f"(d), [pa]"=&f"(pa), [pb]"=&f"(pb), [pc]"=&f"(pc),
          [tmp0]"=&r"(tmp0), [ftmp0]"=&f"(ftmp[0]), [ftmp1]"=&f"(ftmp[1]),
          [ftmp2]"=&f"(ftmp[2]), [eight]"=&f"(eight), [sixteen]"=&f"(sixteen),
          [twenty_four]"=&f"(twenty_four), [forty_eight]"=&f"(forty_eight)
        : [row]"r"(row), [prev]"r"(prev), [istop]"r"(istop)
        : "memory"
   );
}

void png_read_filter_row_paeth4_mmi(png_row_infop row_info, png_bytep row,
   png_const_bytep prev)
{
   /* Paeth tries to predict pixel d using the pixel to the left of it, a,
    * and two pixels from the previous row, b and c:
    *   prev: c b
    *   row:  a d
    * The Paeth function predicts d to be whichever of a, b, or c is nearest to
    * p=a+b-c.
    *
    * The first pixel has no left context, and so uses an Up filter, p = b.
    * This works naturally with our main loop's p = a+b-c if we force a and c
    * to zero.
    * Here we zero b and d, which become c and a respectively at the start of
    * the loop.
    */
   int istop = row_info->rowbytes;
   double rp, pp, zero;
   double a, b, c, d, pa, pb, pc;
   double ftmp[2];

   __asm__ volatile (
        "xor        %[a],      %[a],           %[a]     \n\t"
        "xor        %[c],      %[c],           %[c]     \n\t"
        "xor        %[zero],   %[zero],        %[zero]  \n\t"

        "1:                                             \n\t"
        "lwc1       %[rp],     0x00(%[row])             \n\t"
        "lwc1       %[pp],     0x00(%[prev])            \n\t"
        "punpcklbh  %[b],      %[pp],          %[zero]  \n\t"
        "punpcklbh  %[d],      %[rp],          %[zero]  \n\t"

        "packushb   %[ftmp0],  %[c],           %[c]     \n\t"
        "packushb   %[ftmp1],  %[a],           %[a]     \n\t"
        "pasubub    %[pa],     %[pp],          %[ftmp0] \n\t"
        "pasubub    %[pb],     %[ftmp1],       %[ftmp0] \n\t"
        "psubh      %[ftmp0],  %[b],           %[c]     \n\t"
        "psubh      %[ftmp1],  %[a],           %[c]     \n\t"
        "paddh      %[pc],     %[ftmp0],       %[ftmp1] \n\t"
        "pcmpgth    %[ftmp0],  %[zero],        %[pc]    \n\t"
        "xor        %[pc],     %[pc],          %[ftmp0] \n\t"
        "psubh      %[pc],     %[pc],          %[ftmp0] \n\t"

        "punpcklbh  %[pa],     %[pa],           %[zero] \n\t"
        "punpcklbh  %[pb],     %[pb],           %[zero] \n\t"

        "pcmpgth    %[ftmp0],  %[pa],          %[pb]    \n\t"
        "and        %[ftmp1],  %[b],           %[ftmp0] \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]     \n\t"
        "or         %[a],      %[a],           %[ftmp1] \n\t"
        "pminsh     %[pa],     %[pa],          %[pb]    \n\t"

        "pcmpgth    %[ftmp0],  %[pa],          %[pc]    \n\t"
        "and        %[ftmp1],  %[c],           %[ftmp0] \n\t"
        "pandn      %[a],      %[ftmp0],       %[a]     \n\t"
        "or         %[a],      %[a],           %[ftmp1] \n\t"
        "paddb      %[a],      %[a],           %[d]     \n\t"
        "packushb   %[d],      %[a],           %[a]     \n\t"
        "swc1       %[d],      0x00(%[row])             \n\t"
        "punpcklbh  %[c],      %[pp],          %[zero]  \n\t"
        "daddiu     %[row],    %[row],         0x04     \n\t"
        "daddiu     %[prev],   %[prev],        0x04     \n\t"
        "daddiu     %[istop],  %[istop],      -0x04     \n\t"
        "bgtz       %[istop],  1b                       \n\t"
        : [rp]"=&f"(rp), [pp]"=&f"(pp), [zero]"=&f"(zero),
          [a]"=&f"(a), [b]"=&f"(b), [c]"=&f"(c), [d]"=&f"(d),
          [pa]"=&f"(pa), [pb]"=&f"(pb), [pc]"=&f"(pc),
          [ftmp0]"=&f"(ftmp[0]), [ftmp1]"=&f"(ftmp[1])
        : [row]"r"(row), [prev]"r"(prev), [istop]"r"(istop)
        : "memory"
   );
}

#endif /* PNG_MIPS_MMI_IMPLEMENTATION > 0 */
#endif /* READ */
