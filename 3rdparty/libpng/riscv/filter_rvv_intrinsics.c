/* filter_rvv_intrinsics.c - RISC-V Vector optimized filter functions
 *
 * Copyright (c) 2023 Google LLC
 * Written by Manfred SCHLAEGL, 2022
 *            Dragoș Tiselice <dtiselice@google.com>, May 2023.
 *            Filip Wasil     <f.wasil@samsung.com>, March 2025.
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED

#if PNG_RISCV_RVV_IMPLEMENTATION == 1 /* intrinsics code from pngpriv.h */

#include <riscv_vector.h>

void
png_read_filter_row_up_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   for (size_t vl; len > 0; len -= vl, row += vl, prev_row += vl)
   {
      vl = __riscv_vsetvl_e8m8(len);

      vuint8m8_t prev_vals = __riscv_vle8_v_u8m8(prev_row, vl);
      vuint8m8_t row_vals = __riscv_vle8_v_u8m8(row, vl);

      row_vals = __riscv_vadd_vv_u8m8(row_vals, prev_vals, vl);

      __riscv_vse8_v_u8m8(row, row_vals, vl);
   }
}

static inline void
png_read_filter_row_sub_rvv(size_t len, size_t bpp, unsigned char* row)
{
   png_bytep rp_end = row + len;

   /*
    * row:      | a | x |
    *
    * a = a + x
    *
    * a .. [v0](e8)
    * x .. [v8](e8)
    */

   size_t vl = __riscv_vsetvl_e8m1(bpp);

   /* a = *row */
   vuint8m1_t a = __riscv_vle8_v_u8m1(row, vl);
   row += bpp;

   while (row < rp_end)
   {
      /* x = *row */
      vuint8m1_t x = __riscv_vle8_v_u8m1(row, vl);

      /* a = a + x */
      a = __riscv_vadd_vv_u8m1(a, x, vl);

      /* *row = a */
      __riscv_vse8_v_u8m1(row, a, vl);
      row += bpp;
   }
}

void
png_read_filter_row_sub3_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_sub_rvv(len, 3, row);

   PNG_UNUSED(prev_row)
}

void
png_read_filter_row_sub4_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_sub_rvv(len, 4, row);

   PNG_UNUSED(prev_row)
}

static inline void
png_read_filter_row_avg_rvv(size_t len, size_t bpp, unsigned char* row,
    const unsigned char* prev_row)
{
   png_bytep rp_end = row + len;

   /*
    * row:      | a | x |
    * prev_row: |   | b |
    *
    * a ..   [v2](e8)
    * b ..   [v4](e8)
    * x ..   [v8](e8)
    * tmp .. [v12-v13](e16)
    */

   /* first pixel */

   size_t vl = __riscv_vsetvl_e8m1(bpp);

   /* b = *prev_row */
   vuint8m1_t b = __riscv_vle8_v_u8m1(prev_row, vl);
   prev_row += bpp;

   /* x = *row */
   vuint8m1_t x = __riscv_vle8_v_u8m1(row, vl);

   /* b = b / 2 */
   b = __riscv_vsrl_vx_u8m1(b, 1, vl);

   /* a = x + b */
   vuint8m1_t a = __riscv_vadd_vv_u8m1(b, x, vl);

   /* *row = a */
   __riscv_vse8_v_u8m1(row, a, vl);
   row += bpp;

   /* remaining pixels */
   while (row < rp_end)
   {
      /* b = *prev_row */
      b = __riscv_vle8_v_u8m1(prev_row, vl);
      prev_row += bpp;

      /* x = *row */
      x = __riscv_vle8_v_u8m1(row, vl);

      /* tmp = a + b */
      vuint16m2_t tmp = __riscv_vwaddu_vv_u16m2(a, b, vl);

      /* a = tmp/2 */
      a = __riscv_vnsrl_wx_u8m1(tmp, 1, vl);

      /* a += x */
      a = __riscv_vadd_vv_u8m1(a, x, vl);

      /* *row = a */
      __riscv_vse8_v_u8m1(row, a, vl);
      row += bpp;
   }
}

void
png_read_filter_row_avg3_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_avg_rvv(len, 3, row, prev_row);

   PNG_UNUSED(prev_row)
}

void
png_read_filter_row_avg4_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_avg_rvv(len, 4, row, prev_row);

   PNG_UNUSED(prev_row)
}

#define MIN_CHUNK_LEN 256
#define MAX_CHUNK_LEN 2048

static inline vuint8m1_t
prefix_sum(vuint8m1_t chunk, unsigned char* carry, size_t vl,
    size_t max_chunk_len)
{
   size_t r;

   for (r = 1; r < MIN_CHUNK_LEN; r <<= 1)
   {
      vbool8_t shift_mask = __riscv_vmsgeu_vx_u8m1_b8(__riscv_vid_v_u8m1(vl), r, vl);
      chunk = __riscv_vadd_vv_u8m1_mu(shift_mask, chunk, chunk, __riscv_vslideup_vx_u8m1(__riscv_vundefined_u8m1(), chunk, r, vl), vl);
   }

   for (r = MIN_CHUNK_LEN; r < MAX_CHUNK_LEN && r < max_chunk_len; r <<= 1)
   {
      vbool8_t shift_mask = __riscv_vmsgeu_vx_u8m1_b8(__riscv_vid_v_u8m1(vl), r, vl);
      chunk = __riscv_vadd_vv_u8m1_mu(shift_mask, chunk, chunk, __riscv_vslideup_vx_u8m1(__riscv_vundefined_u8m1(), chunk, r, vl), vl);
   }

   chunk = __riscv_vadd_vx_u8m1(chunk, *carry, vl);
   *carry = __riscv_vmv_x_s_u8m1_u8(__riscv_vslidedown_vx_u8m1(chunk, vl - 1, vl));

   return chunk;
}

static inline vint16m1_t
abs_diff(vuint16m1_t a, vuint16m1_t b, size_t vl)
{
   vint16m1_t diff = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vv_u16m1(a, b, vl));
   vbool16_t mask = __riscv_vmslt_vx_i16m1_b16(diff, 0, vl);
   return __riscv_vrsub_vx_i16m1_m(mask, diff, 0, vl);
}

static inline vint16m1_t
abs_sum(vint16m1_t a, vint16m1_t b, size_t vl)
{
   return __riscv_vadd_vv_i16m1(a, b, vl);
}

static inline void
png_read_filter_row_paeth_rvv(size_t len, size_t bpp, unsigned char* row,
    const unsigned char* prev)
{
   png_bytep rp_end = row + len;

   /*
    * row:  | a | x |
    * prev: | c | b |
    *
    * a .. [v2](e8)
    * b .. [v4](e8)
    * c .. [v6](e8)
    * x .. [v8](e8)
    * p .. [v12-v13](e16)
    * pa, pb, pc .. [v16-v17, v20-v21, v24-v25](e16)
    */

   /* first pixel */

   size_t vl = __riscv_vsetvl_e8m1(bpp);

   /* a = *row */
   vuint8m1_t a = __riscv_vle8_v_u8m1(row, vl);

   /* c = *prev */
   vuint8m1_t c = __riscv_vle8_v_u8m1(prev, vl);

   /* a += c */
   a = __riscv_vadd_vv_u8m1(a, c, vl);

   /* *row = a */
   __riscv_vse8_v_u8m1(row, a, vl);
   row += bpp;
   prev += bpp;

   /* remaining pixels */

   while (row < rp_end)
   {
      /* b = *prev */
      vuint8m1_t b = __riscv_vle8_v_u8m1(prev, vl);
      prev += bpp;

      /* x = *row */
      vuint8m1_t x = __riscv_vle8_v_u8m1(row, vl);

      /* Calculate p = b - c and pc = a - c using widening subtraction */
      vuint16m2_t p_wide = __riscv_vwsubu_vv_u16m2(b, c, vl);
      vuint16m2_t pc_wide = __riscv_vwsubu_vv_u16m2(a, c, vl);

      /* Convert to signed for easier manipulation */
      size_t vl16 = __riscv_vsetvl_e16m2(bpp);
      vint16m2_t p = __riscv_vreinterpret_v_u16m2_i16m2(p_wide);
      vint16m2_t pc = __riscv_vreinterpret_v_u16m2_i16m2(pc_wide);

      /* pa = |p| */
      vbool8_t p_neg_mask = __riscv_vmslt_vx_i16m2_b8(p, 0, vl16);
      vint16m2_t pa = __riscv_vrsub_vx_i16m2_m(p_neg_mask, p, 0, vl16);

      /* pb = |pc| */
      vbool8_t pc_neg_mask = __riscv_vmslt_vx_i16m2_b8(pc, 0, vl16);
      vint16m2_t pb = __riscv_vrsub_vx_i16m2_m(pc_neg_mask, pc, 0, vl16);

      /* pc = |p + pc| */
      vint16m2_t p_plus_pc = __riscv_vadd_vv_i16m2(p, pc, vl16);
      vbool8_t p_plus_pc_neg_mask = __riscv_vmslt_vx_i16m2_b8(p_plus_pc, 0, vl16);
      pc = __riscv_vrsub_vx_i16m2_m(p_plus_pc_neg_mask, p_plus_pc, 0, vl16);

      /*
       * The key insight is that we want the minimum of pa, pb, pc.
       * - If pa <= pb and pa <= pc, use a
       * - Else if pb <= pc, use b
       * - Else use c
       */

      /* Find which predictor to use based on minimum absolute difference */
      vbool8_t pa_le_pb = __riscv_vmsle_vv_i16m2_b8(pa, pb, vl16);
      vbool8_t pa_le_pc = __riscv_vmsle_vv_i16m2_b8(pa, pc, vl16);
      vbool8_t pb_le_pc = __riscv_vmsle_vv_i16m2_b8(pb, pc, vl16);

      /* use_a = pa <= pb && pa <= pc */
      vbool8_t use_a = __riscv_vmand_mm_b8(pa_le_pb, pa_le_pc, vl16);

      /* use_b = !use_a && pb <= pc */
      vbool8_t not_use_a = __riscv_vmnot_m_b8(use_a, vl16);
      vbool8_t use_b = __riscv_vmand_mm_b8(not_use_a, pb_le_pc, vl16);

      /* Switch back to e8m1 for final operations */
      vl = __riscv_vsetvl_e8m1(bpp);

      /* Start with a, then conditionally replace with b or c */
      vuint8m1_t result = a;
      result = __riscv_vmerge_vvm_u8m1(result, b, use_b, vl);

      /* use_c = !use_a && !use_b */
      vbool8_t use_c = __riscv_vmnand_mm_b8(__riscv_vmor_mm_b8(use_a, use_b, vl), __riscv_vmor_mm_b8(use_a, use_b, vl), vl);
      result = __riscv_vmerge_vvm_u8m1(result, c, use_c, vl);

      /* a = result + x */
      a = __riscv_vadd_vv_u8m1(result, x, vl);

      /* *row = a */
      __riscv_vse8_v_u8m1(row, a, vl);
      row += bpp;

      /* c = b for next iteration */
      c = b;
   }
}
void
png_read_filter_row_paeth3_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_paeth_rvv(len, 3, row, prev_row);
}

void
png_read_filter_row_paeth4_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_paeth_rvv(len, 4, row, prev_row);
}

#endif /* PNG_RISCV_RVV_IMPLEMENTATION == 1 */
#endif /* PNG_READ_SUPPORTED */
