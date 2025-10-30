/* filter_msa_intrinsics.c - MSA optimised filter functions
 *
 * Copyright (c) 2018-2024 Cosmin Truta
 * Copyright (c) 2016 Glenn Randers-Pehrson
 * Written by Mandar Sahastrabuddhe, August 2016
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include <stdio.h>
#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED

/* This code requires -mfpu=msa on the command line: */
#if PNG_MIPS_MSA_IMPLEMENTATION == 1 /* intrinsics code from pngpriv.h */

#include <msa.h>
#include <stdint.h>

/* libpng row pointers are not necessarily aligned to any particular boundary,
 * however this code will only work with appropriate alignment. mips/mips_init.c
 * checks for this (and will not compile unless it is done). This code uses
 * variants of png_aligncast to avoid compiler warnings.
 */
#define png_ptr(type,pointer) png_aligncast(type *,pointer)
#define png_ptrc(type,pointer) png_aligncastconst(const type *,pointer)

/* The following relies on a variable 'temp_pointer' being declared with type
 * 'type'.  This is written this way just to hide the GCC strict aliasing
 * warning; note that the code is safe because there never is an alias between
 * the input and output pointers.
 */
#define png_ldr(type,pointer)\
   (temp_pointer = png_ptr(type,pointer), *temp_pointer)

#if PNG_MIPS_MSA_OPT > 0

#ifdef CLANG_BUILD
   #define MSA_SRLI_B(a, b)   __msa_srli_b((v16i8) a, b)

   #define LW(psrc)                              \
   ( {                                           \
       uint8_t *psrc_lw_m = (uint8_t *) (psrc);  \
       uint32_t val_m;                           \
                                                 \
       __asm__ volatile (                        \
           "lw  %[val_m],  %[psrc_lw_m]  \n\t"   \
                                                 \
           : [val_m] "=r" (val_m)                \
           : [psrc_lw_m] "m" (*psrc_lw_m)        \
       );                                        \
                                                 \
       val_m;                                    \
   } )

   #define SH(val, pdst)                         \
   {                                             \
       uint8_t *pdst_sh_m = (uint8_t *) (pdst);  \
       uint16_t val_m = (val);                   \
                                                 \
       __asm__ volatile (                        \
           "sh  %[val_m],  %[pdst_sh_m]  \n\t"   \
                                                 \
           : [pdst_sh_m] "=m" (*pdst_sh_m)       \
           : [val_m] "r" (val_m)                 \
       );                                        \
   }

   #define SW(val, pdst)                         \
   {                                             \
       uint8_t *pdst_sw_m = (uint8_t *) (pdst);  \
       uint32_t val_m = (val);                   \
                                                 \
       __asm__ volatile (                        \
           "sw  %[val_m],  %[pdst_sw_m]  \n\t"   \
                                                 \
           : [pdst_sw_m] "=m" (*pdst_sw_m)       \
           : [val_m] "r" (val_m)                 \
       );                                        \
   }

   #if __mips == 64
        #define SD(val, pdst)                         \
        {                                             \
            uint8_t *pdst_sd_m = (uint8_t *) (pdst);  \
            uint64_t val_m = (val);                   \
                                                      \
            __asm__ volatile (                        \
                "sd  %[val_m],  %[pdst_sd_m]  \n\t"   \
                                                      \
                : [pdst_sd_m] "=m" (*pdst_sd_m)       \
                : [val_m] "r" (val_m)                 \
            );                                        \
        }
   #else
        #define SD(val, pdst)                                          \
        {                                                              \
            uint8_t *pdst_sd_m = (uint8_t *) (pdst);                   \
            uint32_t val0_m, val1_m;                                   \
                                                                       \
            val0_m = (uint32_t) ((val) & 0x00000000FFFFFFFF);          \
            val1_m = (uint32_t) (((val) >> 32) & 0x00000000FFFFFFFF);  \
                                                                       \
            SW(val0_m, pdst_sd_m);                                     \
            SW(val1_m, pdst_sd_m + 4);                                 \
        }
   #endif /* __mips == 64 */
#else
   #define MSA_SRLI_B(a, b)   (a >> b)

#if __mips_isa_rev >= 6
   #define LW(psrc)                              \
   ( {                                           \
       uint8_t *psrc_lw_m = (uint8_t *) (psrc);  \
       uint32_t val_m;                           \
                                                 \
       __asm__ volatile (                        \
           "lw  %[val_m],  %[psrc_lw_m]  \n\t"   \
                                                 \
           : [val_m] "=r" (val_m)                \
           : [psrc_lw_m] "m" (*psrc_lw_m)        \
       );                                        \
                                                 \
       val_m;                                    \
   } )

   #define SH(val, pdst)                         \
   {                                             \
       uint8_t *pdst_sh_m = (uint8_t *) (pdst);  \
       uint16_t val_m = (val);                   \
                                                 \
       __asm__ volatile (                        \
           "sh  %[val_m],  %[pdst_sh_m]  \n\t"   \
                                                 \
           : [pdst_sh_m] "=m" (*pdst_sh_m)       \
           : [val_m] "r" (val_m)                 \
       );                                        \
   }

   #define SW(val, pdst)                         \
   {                                             \
       uint8_t *pdst_sw_m = (uint8_t *) (pdst);  \
       uint32_t val_m = (val);                   \
                                                 \
       __asm__ volatile (                        \
           "sw  %[val_m],  %[pdst_sw_m]  \n\t"   \
                                                 \
           : [pdst_sw_m] "=m" (*pdst_sw_m)       \
           : [val_m] "r" (val_m)                 \
       );                                        \
   }

   #if __mips == 64
        #define SD(val, pdst)                         \
        {                                             \
            uint8_t *pdst_sd_m = (uint8_t *) (pdst);  \
            uint64_t val_m = (val);                   \
                                                      \
            __asm__ volatile (                        \
                "sd  %[val_m],  %[pdst_sd_m]  \n\t"   \
                                                      \
                : [pdst_sd_m] "=m" (*pdst_sd_m)       \
                : [val_m] "r" (val_m)                 \
            );                                        \
        }
   #else
        #define SD(val, pdst)                                          \
        {                                                              \
            uint8_t *pdst_sd_m = (uint8_t *) (pdst);                   \
            uint32_t val0_m, val1_m;                                   \
                                                                       \
            val0_m = (uint32_t) ((val) & 0x00000000FFFFFFFF);          \
            val1_m = (uint32_t) (((val) >> 32) & 0x00000000FFFFFFFF);  \
                                                                       \
            SW(val0_m, pdst_sd_m);                                     \
            SW(val1_m, pdst_sd_m + 4);                                 \
        }
   #endif /* __mips == 64 */
#else
   #define LW(psrc)                              \
   ( {                                           \
       uint8_t *psrc_lw_m = (uint8_t *) (psrc);  \
       uint32_t val_m;                           \
                                                 \
       __asm__ volatile (                        \
           "ulw  %[val_m],  %[psrc_lw_m]  \n\t"  \
                                                 \
           : [val_m] "=r" (val_m)                \
           : [psrc_lw_m] "m" (*psrc_lw_m)        \
       );                                        \
                                                 \
       val_m;                                    \
   } )

   #define SH(val, pdst)                         \
   {                                             \
       uint8_t *pdst_sh_m = (uint8_t *) (pdst);  \
       uint16_t val_m = (val);                   \
                                                 \
       __asm__ volatile (                        \
           "ush  %[val_m],  %[pdst_sh_m]  \n\t"  \
                                                 \
           : [pdst_sh_m] "=m" (*pdst_sh_m)       \
           : [val_m] "r" (val_m)                 \
       );                                        \
   }

   #define SW(val, pdst)                         \
   {                                             \
       uint8_t *pdst_sw_m = (uint8_t *) (pdst);  \
       uint32_t val_m = (val);                   \
                                                 \
       __asm__ volatile (                        \
           "usw  %[val_m],  %[pdst_sw_m]  \n\t"  \
                                                 \
           : [pdst_sw_m] "=m" (*pdst_sw_m)       \
           : [val_m] "r" (val_m)                 \
       );                                        \
   }

   #define SD(val, pdst)                                           \
    {                                                              \
        uint8_t *pdst_sd_m = (uint8_t *) (pdst);                   \
        uint32_t val0_m, val1_m;                                   \
                                                                   \
        val0_m = (uint32_t) ((val) & 0x00000000FFFFFFFF);          \
        val1_m = (uint32_t) (((val) >> 32) & 0x00000000FFFFFFFF);  \
                                                                   \
        SW(val0_m, pdst_sd_m);                                     \
        SW(val1_m, pdst_sd_m + 4);                                 \
    }

    #define SW_ZERO(pdst)                      \
    {                                          \
        uint8_t *pdst_m = (uint8_t *) (pdst);  \
                                               \
        __asm__ volatile (                     \
            "usw  $0,  %[pdst_m]  \n\t"        \
                                               \
            : [pdst_m] "=m" (*pdst_m)          \
            :                                  \
        );                                     \
    }
#endif /* __mips_isa_rev >= 6 */
#endif

#define LD_B(RTYPE, psrc) *((RTYPE *) (psrc))
#define LD_UB(...) LD_B(v16u8, __VA_ARGS__)
#define LD_B2(RTYPE, psrc, stride, out0, out1)  \
{                                               \
    out0 = LD_B(RTYPE, (psrc));                 \
    out1 = LD_B(RTYPE, (psrc) + stride);        \
}
#define LD_UB2(...) LD_B2(v16u8, __VA_ARGS__)
#define LD_B4(RTYPE, psrc, stride, out0, out1, out2, out3)   \
{                                                            \
    LD_B2(RTYPE, (psrc), stride, out0, out1);                \
    LD_B2(RTYPE, (psrc) + 2 * stride , stride, out2, out3);  \
}
#define LD_UB4(...) LD_B4(v16u8, __VA_ARGS__)

#define ST_B(RTYPE, in, pdst) *((RTYPE *) (pdst)) = (in)
#define ST_UB(...) ST_B(v16u8, __VA_ARGS__)
#define ST_B2(RTYPE, in0, in1, pdst, stride)  \
{                                             \
    ST_B(RTYPE, in0, (pdst));                 \
    ST_B(RTYPE, in1, (pdst) + stride);        \
}
#define ST_UB2(...) ST_B2(v16u8, __VA_ARGS__)
#define ST_B4(RTYPE, in0, in1, in2, in3, pdst, stride)    \
{                                                         \
    ST_B2(RTYPE, in0, in1, (pdst), stride);               \
    ST_B2(RTYPE, in2, in3, (pdst) + 2 * stride, stride);  \
}
#define ST_UB4(...) ST_B4(v16u8, __VA_ARGS__)

#define ADD2(in0, in1, in2, in3, out0, out1)  \
{                                             \
    out0 = in0 + in1;                         \
    out1 = in2 + in3;                         \
}
#define ADD3(in0, in1, in2, in3, in4, in5,  \
             out0, out1, out2)              \
{                                           \
    ADD2(in0, in1, in2, in3, out0, out1);   \
    out2 = in4 + in5;                       \
}
#define ADD4(in0, in1, in2, in3, in4, in5, in6, in7,  \
             out0, out1, out2, out3)                  \
{                                                     \
    ADD2(in0, in1, in2, in3, out0, out1);             \
    ADD2(in4, in5, in6, in7, out2, out3);             \
}

#define ILVR_B2(RTYPE, in0, in1, in2, in3, out0, out1)      \
{                                                           \
    out0 = (RTYPE) __msa_ilvr_b((v16i8) in0, (v16i8) in1);  \
    out1 = (RTYPE) __msa_ilvr_b((v16i8) in2, (v16i8) in3);  \
}
#define ILVR_B2_SH(...) ILVR_B2(v8i16, __VA_ARGS__)

#define HSUB_UB2(RTYPE, in0, in1, out0, out1)                 \
{                                                             \
    out0 = (RTYPE) __msa_hsub_u_h((v16u8) in0, (v16u8) in0);  \
    out1 = (RTYPE) __msa_hsub_u_h((v16u8) in1, (v16u8) in1);  \
}
#define HSUB_UB2_SH(...) HSUB_UB2(v8i16, __VA_ARGS__)

#define SLDI_B2_0(RTYPE, in0, in1, out0, out1, slide_val)                 \
{                                                                         \
    v16i8 zero_m = { 0 };                                                 \
    out0 = (RTYPE) __msa_sldi_b((v16i8) zero_m, (v16i8) in0, slide_val);  \
    out1 = (RTYPE) __msa_sldi_b((v16i8) zero_m, (v16i8) in1, slide_val);  \
}
#define SLDI_B2_0_UB(...) SLDI_B2_0(v16u8, __VA_ARGS__)

#define SLDI_B3_0(RTYPE, in0, in1, in2, out0, out1, out2,  slide_val)     \
{                                                                         \
    v16i8 zero_m = { 0 };                                                 \
    SLDI_B2_0(RTYPE, in0, in1, out0, out1, slide_val);                    \
    out2 = (RTYPE) __msa_sldi_b((v16i8) zero_m, (v16i8) in2, slide_val);  \
}
#define SLDI_B3_0_UB(...) SLDI_B3_0(v16u8, __VA_ARGS__)

#define ILVEV_W2(RTYPE, in0, in1, in2, in3, out0, out1)      \
{                                                            \
    out0 = (RTYPE) __msa_ilvev_w((v4i32) in1, (v4i32) in0);  \
    out1 = (RTYPE) __msa_ilvev_w((v4i32) in3, (v4i32) in2);  \
}
#define ILVEV_W2_UB(...) ILVEV_W2(v16u8, __VA_ARGS__)

#define ADD_ABS_H3(RTYPE, in0, in1, in2, out0, out1, out2)  \
{                                                           \
    RTYPE zero = {0};                                       \
                                                            \
    out0 = __msa_add_a_h((v8i16) zero, in0);                \
    out1 = __msa_add_a_h((v8i16) zero, in1);                \
    out2 = __msa_add_a_h((v8i16) zero, in2);                \
}
#define ADD_ABS_H3_SH(...) ADD_ABS_H3(v8i16, __VA_ARGS__)

#define VSHF_B2(RTYPE, in0, in1, in2, in3, mask0, mask1, out0, out1)       \
{                                                                          \
    out0 = (RTYPE) __msa_vshf_b((v16i8) mask0, (v16i8) in1, (v16i8) in0);  \
    out1 = (RTYPE) __msa_vshf_b((v16i8) mask1, (v16i8) in3, (v16i8) in2);  \
}
#define VSHF_B2_UB(...) VSHF_B2(v16u8, __VA_ARGS__)

#define CMP_AND_SELECT(inp0, inp1, inp2, inp3, inp4, inp5, out0)              \
{                                                                             \
   v8i16 _sel_h0, _sel_h1;                                                    \
   v16u8 _sel_b0, _sel_b1;                                                    \
   _sel_h0 = (v8i16) __msa_clt_u_h((v8u16) inp1, (v8u16) inp0);               \
   _sel_b0 = (v16u8) __msa_pckev_b((v16i8) _sel_h0, (v16i8) _sel_h0);         \
   inp0 = (v8i16) __msa_bmnz_v((v16u8) inp0, (v16u8) inp1, (v16u8) _sel_h0);  \
   inp4 = (v16u8) __msa_bmnz_v(inp3, inp4, _sel_b0);                          \
   _sel_h1 = (v8i16) __msa_clt_u_h((v8u16) inp2, (v8u16) inp0);               \
   _sel_b1 = (v16u8) __msa_pckev_b((v16i8) _sel_h1, (v16i8) _sel_h1);         \
   inp4 = (v16u8) __msa_bmnz_v(inp4, inp5, _sel_b1);                          \
   out0 += inp4;                                                              \
}

void png_read_filter_row_up_msa(png_row_infop row_info, png_bytep row,
                                png_const_bytep prev_row)
{
   size_t i, cnt, cnt16, cnt32;
   size_t istop = row_info->rowbytes;
   png_bytep rp = row;
   png_const_bytep pp = prev_row;
   v16u8 src0, src1, src2, src3, src4, src5, src6, src7;

   for (i = 0; i < (istop >> 6); i++)
   {
      LD_UB4(rp, 16, src0, src1, src2, src3);
      LD_UB4(pp, 16, src4, src5, src6, src7);
      pp += 64;

      ADD4(src0, src4, src1, src5, src2, src6, src3, src7,
           src0, src1, src2, src3);

      ST_UB4(src0, src1, src2, src3, rp, 16);
      rp += 64;
   }

   if (istop & 0x3F)
   {
      cnt32 = istop & 0x20;
      cnt16 = istop & 0x10;
      cnt = istop & 0xF;

      if(cnt32)
      {
         if (cnt16 && cnt)
         {
            LD_UB4(rp, 16, src0, src1, src2, src3);
            LD_UB4(pp, 16, src4, src5, src6, src7);

            ADD4(src0, src4, src1, src5, src2, src6, src3, src7,
                 src0, src1, src2, src3);

            ST_UB4(src0, src1, src2, src3, rp, 16);
            rp += 64;
         }
         else if (cnt16 || cnt)
         {
            LD_UB2(rp, 16, src0, src1);
            LD_UB2(pp, 16, src4, src5);
            pp += 32;
            src2 = LD_UB(rp + 32);
            src6 = LD_UB(pp);

            ADD3(src0, src4, src1, src5, src2, src6, src0, src1, src2);

            ST_UB2(src0, src1, rp, 16);
            rp += 32;
            ST_UB(src2, rp);
            rp += 16;
         }
         else
         {
            LD_UB2(rp, 16, src0, src1);
            LD_UB2(pp, 16, src4, src5);

            ADD2(src0, src4, src1, src5, src0, src1);

            ST_UB2(src0, src1, rp, 16);
            rp += 32;
         }
      }
      else if (cnt16 && cnt)
      {
         LD_UB2(rp, 16, src0, src1);
         LD_UB2(pp, 16, src4, src5);

         ADD2(src0, src4, src1, src5, src0, src1);

         ST_UB2(src0, src1, rp, 16);
         rp += 32;
      }
      else if (cnt16 || cnt)
      {
         src0 = LD_UB(rp);
         src4 = LD_UB(pp);
         pp += 16;

         src0 += src4;

         ST_UB(src0, rp);
         rp += 16;
      }
   }
}

void png_read_filter_row_sub4_msa(png_row_infop row_info, png_bytep row,
                                  png_const_bytep prev_row)
{
   size_t count;
   size_t istop = row_info->rowbytes;
   png_bytep src = row;
   png_bytep nxt = row + 4;
   int32_t inp0;
   v16u8 src0, src1, src2, src3, src4;
   v16u8 dst0, dst1;
   v16u8 zero = { 0 };

   istop -= 4;

   inp0 = LW(src);
   src += 4;
   src0 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp0);

   for (count = 0; count < istop; count += 16)
   {
      src1 = LD_UB(src);
      src += 16;

      src2 = (v16u8) __msa_sldi_b((v16i8) zero, (v16i8) src1, 4);
      src3 = (v16u8) __msa_sldi_b((v16i8) zero, (v16i8) src1, 8);
      src4 = (v16u8) __msa_sldi_b((v16i8) zero, (v16i8) src1, 12);
      src1 += src0;
      src2 += src1;
      src3 += src2;
      src4 += src3;
      src0 = src4;
      ILVEV_W2_UB(src1, src2, src3, src4, dst0, dst1);
      dst0 = (v16u8) __msa_pckev_d((v2i64) dst1, (v2i64) dst0);

      ST_UB(dst0, nxt);
      nxt += 16;
   }
}

void png_read_filter_row_sub3_msa(png_row_infop row_info, png_bytep row,
                                  png_const_bytep prev_row)
{
   size_t count;
   size_t istop = row_info->rowbytes;
   png_bytep src = row;
   png_bytep nxt = row + 3;
   int64_t out0;
   int32_t inp0, out1;
   v16u8 src0, src1, src2, src3, src4, dst0, dst1;
   v16u8 zero = { 0 };
   v16i8 mask0 = { 0, 1, 2, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   v16i8 mask1 = { 0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0 };

   istop -= 3;

   inp0 = LW(src);
   src += 3;
   src0 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp0);

   for (count = 0; count < istop; count += 12)
   {
      src1 = LD_UB(src);
      src += 12;

      src2 = (v16u8) __msa_sldi_b((v16i8) zero, (v16i8) src1, 3);
      src3 = (v16u8) __msa_sldi_b((v16i8) zero, (v16i8) src1, 6);
      src4 = (v16u8) __msa_sldi_b((v16i8) zero, (v16i8) src1, 9);
      src1 += src0;
      src2 += src1;
      src3 += src2;
      src4 += src3;
      src0 = src4;
      VSHF_B2_UB(src1, src2, src3, src4, mask0, mask0, dst0, dst1);
      dst0 = (v16u8) __msa_vshf_b(mask1, (v16i8) dst1, (v16i8) dst0);
      out0 = __msa_copy_s_d((v2i64) dst0, 0);
      out1 = __msa_copy_s_w((v4i32) dst0, 2);

      SD(out0, nxt);
      nxt += 8;
      SW(out1, nxt);
      nxt += 4;
   }
}

void png_read_filter_row_avg4_msa(png_row_infop row_info, png_bytep row,
                                  png_const_bytep prev_row)
{
   size_t i;
   png_bytep src = row;
   png_bytep nxt = row;
   png_const_bytep pp = prev_row;
   size_t istop = row_info->rowbytes - 4;
   int32_t inp0, inp1, out0;
   v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, dst0, dst1;
   v16u8 zero = { 0 };

   inp0 = LW(pp);
   pp += 4;
   inp1 = LW(src);
   src += 4;
   src0 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp0);
   src1 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp1);
   src0 = (v16u8) MSA_SRLI_B(src0, 1);
   src1 += src0;
   out0 = __msa_copy_s_w((v4i32) src1, 0);
   SW(out0, nxt);
   nxt += 4;

   for (i = 0; i < istop; i += 16)
   {
      src2 = LD_UB(pp);
      pp += 16;
      src6 = LD_UB(src);
      src += 16;

      SLDI_B2_0_UB(src2, src6, src3, src7, 4);
      SLDI_B2_0_UB(src2, src6, src4, src8, 8);
      SLDI_B2_0_UB(src2, src6, src5, src9, 12);
      src2 = __msa_ave_u_b(src2, src1);
      src6 += src2;
      src3 = __msa_ave_u_b(src3, src6);
      src7 += src3;
      src4 = __msa_ave_u_b(src4, src7);
      src8 += src4;
      src5 = __msa_ave_u_b(src5, src8);
      src9 += src5;
      src1 = src9;
      ILVEV_W2_UB(src6, src7, src8, src9, dst0, dst1);
      dst0 = (v16u8) __msa_pckev_d((v2i64) dst1, (v2i64) dst0);

      ST_UB(dst0, nxt);
      nxt += 16;
   }
}

void png_read_filter_row_avg3_msa(png_row_infop row_info, png_bytep row,
                                  png_const_bytep prev_row)
{
   size_t i;
   png_bytep src = row;
   png_bytep nxt = row;
   png_const_bytep pp = prev_row;
   size_t istop = row_info->rowbytes - 3;
   int64_t out0;
   int32_t inp0, inp1, out1;
   int16_t out2;
   v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, dst0, dst1;
   v16u8 zero = { 0 };
   v16i8 mask0 = { 0, 1, 2, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   v16i8 mask1 = { 0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0 };

   inp0 = LW(pp);
   pp += 3;
   inp1 = LW(src);
   src += 3;
   src0 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp0);
   src1 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp1);
   src0 = (v16u8) MSA_SRLI_B(src0, 1);
   src1 += src0;
   out2 = __msa_copy_s_h((v8i16) src1, 0);
   SH(out2, nxt);
   nxt += 2;
   nxt[0] = src1[2];
   nxt++;

   for (i = 0; i < istop; i += 12)
   {
      src2 = LD_UB(pp);
      pp += 12;
      src6 = LD_UB(src);
      src += 12;

      SLDI_B2_0_UB(src2, src6, src3, src7, 3);
      SLDI_B2_0_UB(src2, src6, src4, src8, 6);
      SLDI_B2_0_UB(src2, src6, src5, src9, 9);
      src2 = __msa_ave_u_b(src2, src1);
      src6 += src2;
      src3 = __msa_ave_u_b(src3, src6);
      src7 += src3;
      src4 = __msa_ave_u_b(src4, src7);
      src8 += src4;
      src5 = __msa_ave_u_b(src5, src8);
      src9 += src5;
      src1 = src9;
      VSHF_B2_UB(src6, src7, src8, src9, mask0, mask0, dst0, dst1);
      dst0 = (v16u8) __msa_vshf_b(mask1, (v16i8) dst1, (v16i8) dst0);
      out0 = __msa_copy_s_d((v2i64) dst0, 0);
      out1 = __msa_copy_s_w((v4i32) dst0, 2);

      SD(out0, nxt);
      nxt += 8;
      SW(out1, nxt);
      nxt += 4;
   }
}

void png_read_filter_row_paeth4_msa(png_row_infop row_info,
                                    png_bytep row,
                                    png_const_bytep prev_row)
{
   int32_t count, rp_end;
   png_bytep nxt;
   png_const_bytep prev_nxt;
   int32_t inp0, inp1, res0;
   v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9;
   v16u8 src10, src11, src12, src13, dst0, dst1;
   v8i16 vec0, vec1, vec2;
   v16u8 zero = { 0 };

   nxt = row;
   prev_nxt = prev_row;

   inp0 = LW(nxt);
   inp1 = LW(prev_nxt);
   prev_nxt += 4;
   src0 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp0);
   src1 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp1);

   src1 += src0;
   res0 = __msa_copy_s_w((v4i32) src1, 0);

   SW(res0, nxt);
   nxt += 4;

   /* Remainder */
   rp_end = row_info->rowbytes - 4;

   for (count = 0; count < rp_end; count += 16)
   {
      src2 = LD_UB(prev_nxt);
      prev_nxt += 16;
      src6 = LD_UB(prev_row);
      prev_row += 16;
      src10 = LD_UB(nxt);

      SLDI_B3_0_UB(src2, src6, src10, src3, src7, src11, 4);
      SLDI_B3_0_UB(src2, src6, src10, src4, src8, src12, 8);
      SLDI_B3_0_UB(src2, src6, src10, src5, src9, src13, 12);
      ILVR_B2_SH(src2, src6, src1, src6, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src1, src2, src6, src10);
      ILVR_B2_SH(src3, src7, src10, src7, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src10, src3, src7, src11);
      ILVR_B2_SH(src4, src8, src11, src8, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src11, src4, src8, src12);
      ILVR_B2_SH(src5, src9, src12, src9, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src12, src5, src9, src13);
      src1 = src13;
      ILVEV_W2_UB(src10, src11, src12, src1, dst0, dst1);
      dst0 = (v16u8) __msa_pckev_d((v2i64) dst1, (v2i64) dst0);

      ST_UB(dst0, nxt);
      nxt += 16;
   }
}

void png_read_filter_row_paeth3_msa(png_row_infop row_info,
                                    png_bytep row,
                                    png_const_bytep prev_row)
{
   int32_t count, rp_end;
   png_bytep nxt;
   png_const_bytep prev_nxt;
   int64_t out0;
   int32_t inp0, inp1, out1;
   int16_t out2;
   v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, dst0, dst1;
   v16u8 src10, src11, src12, src13;
   v8i16 vec0, vec1, vec2;
   v16u8 zero = { 0 };
   v16i8 mask0 = { 0, 1, 2, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   v16i8 mask1 = { 0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0 };

   nxt = row;
   prev_nxt = prev_row;

   inp0 = LW(nxt);
   inp1 = LW(prev_nxt);
   prev_nxt += 3;
   src0 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp0);
   src1 = (v16u8) __msa_insert_w((v4i32) zero, 0, inp1);

   src1 += src0;
   out2 = __msa_copy_s_h((v8i16) src1, 0);

   SH(out2, nxt);
   nxt += 2;
   nxt[0] = src1[2];
   nxt++;

   /* Remainder */
   rp_end = row_info->rowbytes - 3;

   for (count = 0; count < rp_end; count += 12)
   {
      src2 = LD_UB(prev_nxt);
      prev_nxt += 12;
      src6 = LD_UB(prev_row);
      prev_row += 12;
      src10 = LD_UB(nxt);

      SLDI_B3_0_UB(src2, src6, src10, src3, src7, src11, 3);
      SLDI_B3_0_UB(src2, src6, src10, src4, src8, src12, 6);
      SLDI_B3_0_UB(src2, src6, src10, src5, src9, src13, 9);
      ILVR_B2_SH(src2, src6, src1, src6, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src1, src2, src6, src10);
      ILVR_B2_SH(src3, src7, src10, src7, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src10, src3, src7, src11);
      ILVR_B2_SH(src4, src8, src11, src8, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src11, src4, src8, src12);
      ILVR_B2_SH(src5, src9, src12, src9, vec0, vec1);
      HSUB_UB2_SH(vec0, vec1, vec0, vec1);
      vec2 = vec0 + vec1;
      ADD_ABS_H3_SH(vec0, vec1, vec2, vec0, vec1, vec2);
      CMP_AND_SELECT(vec0, vec1, vec2, src12, src5, src9, src13);
      src1 = src13;
      VSHF_B2_UB(src10, src11, src12, src13, mask0, mask0, dst0, dst1);
      dst0 = (v16u8) __msa_vshf_b(mask1, (v16i8) dst1, (v16i8) dst0);
      out0 = __msa_copy_s_d((v2i64) dst0, 0);
      out1 = __msa_copy_s_w((v4i32) dst0, 2);

      SD(out0, nxt);
      nxt += 8;
      SW(out1, nxt);
      nxt += 4;
   }
}

#endif /* PNG_MIPS_MSA_OPT > 0 */
#endif /* PNG_MIPS_MSA_IMPLEMENTATION == 1 (intrinsics) */
#endif /* READ */
