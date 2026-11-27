/*
 * Merged Upsampling/Color Conversion (64-bit MMI)
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2015, 2019, 2025, D. R. Commander.
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 * Authors:  ZhangLixia <zhanglixia-hf@loongson.cn>
 *
 * Based on the x86 SIMD extension for IJG JPEG library
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* This file is included by jdmerge-mmi.c */


#if RGB_RED == 0
#define mmA  re
#define mmB  ro
#elif RGB_GREEN == 0
#define mmA  ge
#define mmB  go
#elif RGB_BLUE == 0
#define mmA  be
#define mmB  bo
#else
#define mmA  xe
#define mmB  xo
#endif

#if RGB_RED == 1
#define mmC  re
#define mmD  ro
#elif RGB_GREEN == 1
#define mmC  ge
#define mmD  go
#elif RGB_BLUE == 1
#define mmC  be
#define mmD  bo
#else
#define mmC  xe
#define mmD  xo
#endif

#if RGB_RED == 2
#define mmE  re
#define mmF  ro
#elif RGB_GREEN == 2
#define mmE  ge
#define mmF  go
#elif RGB_BLUE == 2
#define mmE  be
#define mmF  bo
#else
#define mmE  xe
#define mmF  xo
#endif

#if RGB_RED == 3
#define mmG  re
#define mmH  ro
#elif RGB_GREEN == 3
#define mmG  ge
#define mmH  go
#elif RGB_BLUE == 3
#define mmG  be
#define mmH  bo
#else
#define mmG  xe
#define mmH  xo
#endif


HIDDEN void
jsimd_h2v1_merged_upsample_mmi(JDIMENSION output_width, JSAMPIMAGE input_buf,
                               JDIMENSION in_row_group_ctr,
                               JSAMPARRAY output_buf)
{
  JSAMPROW outptr, inptr0, inptr1, inptr2;
  int num_cols, col;
  __m64 ythise, ythiso, ythis, ynexte, ynexto, ynext, yl, y;
  __m64 cbl, cbl2, cbh, cbh2, cb, crl, crl2, crh, crh2, cr;
  __m64 rle, rlo, rl, rhe, rho, rh, re, ro;
  __m64 ga, gb, gle, glo, gl, gc, gd, ghe, gho, gh, ge, go;
  __m64 ble, blo, bl, bhe, bho, bh, be, bo, xe = 0.0, xo = 0.0;
  __m64 decenter, mask, zero = 0.0;
#if RGB_PIXELSIZE == 4
  __m64 mm8, mm9;
#endif

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  for (num_cols = output_width >> 1; num_cols > 0; num_cols -= 8,
       inptr0 += 16, inptr1 += 8, inptr2 += 8) {

    cb = _mm_load_si64((__m64 *)inptr1);
    cr = _mm_load_si64((__m64 *)inptr2);
    ythis = _mm_load_si64((__m64 *)inptr0);
    ynext = _mm_load_si64((__m64 *)inptr0 + 1);

    mask = decenter = 0.0;
    mask = _mm_cmpeq_pi16(mask, mask);
    decenter = _mm_cmpeq_pi16(decenter, decenter);
    mask = _mm_srli_pi16(mask, BYTE_BIT);     /* { 0xFF 0x00 0xFF 0x00 .. } */
    decenter = _mm_slli_pi16(decenter, 7);  /* { 0xFF80 0xFF80 0xFF80 0xFF80 } */

    cbl = _mm_unpacklo_pi8(cb, zero);         /* Cb(0123) */
    cbh = _mm_unpackhi_pi8(cb, zero);         /* Cb(4567) */
    crl = _mm_unpacklo_pi8(cr, zero);         /* Cr(0123) */
    crh = _mm_unpackhi_pi8(cr, zero);         /* Cr(4567) */
    cbl = _mm_add_pi16(cbl, decenter);
    cbh = _mm_add_pi16(cbh, decenter);
    crl = _mm_add_pi16(crl, decenter);
    crh = _mm_add_pi16(crh, decenter);

    /* (Original)
     * R = Y                + 1.40200 * Cr
     * G = Y - 0.34414 * Cb - 0.71414 * Cr
     * B = Y + 1.77200 * Cb
     *
     * (This implementation)
     * R = Y                + 0.40200 * Cr + Cr
     * G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
     * B = Y - 0.22800 * Cb + Cb + Cb
     */

    cbl2 = _mm_add_pi16(cbl, cbl);            /* 2 * CbL */
    cbh2 = _mm_add_pi16(cbh, cbh);            /* 2 * CbH */
    crl2 = _mm_add_pi16(crl, crl);            /* 2 * CrL */
    crh2 = _mm_add_pi16(crh, crh);            /* 2 * CrH */

    bl = _mm_mulhi_pi16(cbl2, PW_MF0228);     /* (2 * CbL * -FIX(0.22800) */
    bh = _mm_mulhi_pi16(cbh2, PW_MF0228);     /* (2 * CbH * -FIX(0.22800) */
    rl = _mm_mulhi_pi16(crl2, PW_F0402);      /* (2 * CrL * FIX(0.40200)) */
    rh = _mm_mulhi_pi16(crh2, PW_F0402);      /* (2 * CrH * FIX(0.40200)) */

    bl = _mm_add_pi16(bl, PW_ONE);
    bh = _mm_add_pi16(bh, PW_ONE);
    bl = _mm_srai_pi16(bl, 1);                /* (CbL * -FIX(0.22800)) */
    bh = _mm_srai_pi16(bh, 1);                /* (CbH * -FIX(0.22800)) */
    rl = _mm_add_pi16(rl, PW_ONE);
    rh = _mm_add_pi16(rh, PW_ONE);
    rl = _mm_srai_pi16(rl, 1);                /* (CrL * FIX(0.40200)) */
    rh = _mm_srai_pi16(rh, 1);                /* (CrH * FIX(0.40200)) */

    bl = _mm_add_pi16(bl, cbl);
    bh = _mm_add_pi16(bh, cbh);
    bl = _mm_add_pi16(bl, cbl);           /* (CbL * FIX(1.77200)) = (B - Y)L */
    bh = _mm_add_pi16(bh, cbh);           /* (CbH * FIX(1.77200)) = (B - Y)H */
    rl = _mm_add_pi16(rl, crl);           /* (CrL * FIX(1.40200)) = (R - Y)L */
    rh = _mm_add_pi16(rh, crh);           /* (CrH * FIX(1.40200)) = (R - Y)H */

    ga = _mm_unpacklo_pi16(cbl, crl);
    gb = _mm_unpackhi_pi16(cbl, crl);
    ga = _mm_madd_pi16(ga, PW_MF0344_F0285);
    gb = _mm_madd_pi16(gb, PW_MF0344_F0285);
    gc = _mm_unpacklo_pi16(cbh, crh);
    gd = _mm_unpackhi_pi16(cbh, crh);
    gc = _mm_madd_pi16(gc, PW_MF0344_F0285);
    gd = _mm_madd_pi16(gd, PW_MF0344_F0285);

    ga = _mm_add_pi32(ga, PD_ONEHALF);
    gb = _mm_add_pi32(gb, PD_ONEHALF);
    ga = _mm_srai_pi32(ga, SCALEBITS);
    gb = _mm_srai_pi32(gb, SCALEBITS);
    gc = _mm_add_pi32(gc, PD_ONEHALF);
    gd = _mm_add_pi32(gd, PD_ONEHALF);
    gc = _mm_srai_pi32(gc, SCALEBITS);
    gd = _mm_srai_pi32(gd, SCALEBITS);

    gl = _mm_packs_pi32(ga, gb);     /* CbL * -FIX(0.344) + CrL * FIX(0.285) */
    gh = _mm_packs_pi32(gc, gd);     /* CbH * -FIX(0.344) + CrH * FIX(0.285) */
    gl = _mm_sub_pi16(gl, crl);  /* CbL * -FIX(0.344) + CrL * -FIX(0.714) = (G - Y)L */
    gh = _mm_sub_pi16(gh, crh);  /* CbH * -FIX(0.344) + CrH * -FIX(0.714) = (G - Y)H */

    ythise = _mm_and_si64(mask, ythis);       /* Y(0246) */
    ythiso = _mm_srli_pi16(ythis, BYTE_BIT);  /* Y(1357) */
    ynexte = _mm_and_si64(mask, ynext);       /* Y(8ACE) */
    ynexto = _mm_srli_pi16(ynext, BYTE_BIT);  /* Y(9BDF) */

    rle = _mm_add_pi16(rl, ythise);           /* (R0 R2 R4 R6) */
    rlo = _mm_add_pi16(rl, ythiso);           /* (R1 R3 R5 R7) */
    rhe = _mm_add_pi16(rh, ynexte);           /* (R8 RA RC RE) */
    rho = _mm_add_pi16(rh, ynexto);           /* (R9 RB RD RF) */
    re = _mm_packs_pu16(rle, rhe);            /* (R0 R2 R4 R6 R8 RA RC RE) */
    ro = _mm_packs_pu16(rlo, rho);            /* (R1 R3 R5 R7 R9 RB RD RF) */

    gle = _mm_add_pi16(gl, ythise);           /* (G0 G2 G4 G6) */
    glo = _mm_add_pi16(gl, ythiso);           /* (G1 G3 G5 G7) */
    ghe = _mm_add_pi16(gh, ynexte);           /* (G8 GA GC GE) */
    gho = _mm_add_pi16(gh, ynexto);           /* (G9 GB GD GF) */
    ge = _mm_packs_pu16(gle, ghe);            /* (G0 G2 G4 G6 G8 GA GC GE) */
    go = _mm_packs_pu16(glo, gho);            /* (G1 G3 G5 G7 G9 GB GD GF) */

    ble = _mm_add_pi16(bl, ythise);           /* (B0 B2 B4 B6) */
    blo = _mm_add_pi16(bl, ythiso);           /* (B1 B3 B5 B7) */
    bhe = _mm_add_pi16(bh, ynexte);           /* (B8 BA BC BE) */
    bho = _mm_add_pi16(bh, ynexto);           /* (B9 BB BD BF) */
    be = _mm_packs_pu16(ble, bhe);            /* (B0 B2 B4 B6 B8 BA BC BE) */
    bo = _mm_packs_pu16(blo, bho);            /* (B1 B3 B5 B7 B9 BB BD BF) */

#if RGB_PIXELSIZE == 3

    /* NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
     * mapping of components A, B, and C to red, green, and blue.
     *
     * mmA = (A0 A2 A4 A6 A8 Aa Ac Ae) = AE
     * mmB = (A1 A3 A5 A7 A9 Ab Ad Af) = AO
     * mmC = (B0 B2 B4 B6 B8 Ba Bc Be) = BE
     * mmD = (B1 B3 B5 B7 B9 Bb Bd Bf) = BO
     * mmE = (C0 C2 C4 C6 C8 Ca Cc Ce) = CE
     * mmF = (C1 C3 C5 C7 C9 Cb Cd Cf) = CO
     * mmG = (** ** ** ** ** ** ** **)
     * mmH = (** ** ** ** ** ** ** **)
     */
    mmG = _mm_unpacklo_pi8(mmA, mmC);         /* (A0 B0 A2 B2 A4 B4 A6 B6) */
    mmA = _mm_unpackhi_pi8(mmA, mmC);         /* (A8 B8 Aa Ba Ac Bc Ae Be) */
    mmH = _mm_unpacklo_pi8(mmE, mmB);         /* (C0 A1 C2 A3 C4 A5 C6 A7) */
    mmE = _mm_unpackhi_pi8(mmE, mmB);         /* (C8 A9 Ca Ab Cc Ad Ce Af) */
    mmC = _mm_unpacklo_pi8(mmD, mmF);         /* (B1 C1 B3 C3 B5 C5 B7 C7) */
    mmD = _mm_unpackhi_pi8(mmD, mmF);         /* (B9 C9 Bb Cb Bd Cd Bf Cf) */

    mmB = _mm_unpacklo_pi16(mmG, mmA);        /* (A0 B0 A8 B8 A2 B2 Aa Ba) */
    mmA = _mm_unpackhi_pi16(mmG, mmA);        /* (A4 B4 Ac Bc A6 B6 Ae Be) */
    mmF = _mm_unpacklo_pi16(mmH, mmE);        /* (C0 A1 C8 A9 C2 A3 Ca Ab) */
    mmE = _mm_unpackhi_pi16(mmH, mmE);        /* (C4 A5 Cc Ad C6 A7 Ce Af) */
    mmH = _mm_unpacklo_pi16(mmC, mmD);        /* (B1 C1 B9 C9 B3 C3 Bb Cb) */
    mmG = _mm_unpackhi_pi16(mmC, mmD);        /* (B5 C5 Bd Cd B7 C7 Bf Cf) */

    mmC = _mm_unpacklo_pi16(mmB, mmF);        /* (A0 B0 C0 A1 A8 B8 C8 A9) */
    mmB = _mm_srli_si64(mmB, 4 * BYTE_BIT);
    mmB = _mm_unpacklo_pi16(mmH, mmB);        /* (B1 C1 A2 B2 B9 C9 Aa Ba) */
    mmD = _mm_unpackhi_pi16(mmF, mmH);        /* (C2 A3 B3 C3 Ca Ab Bb Cb) */
    mmF = _mm_unpacklo_pi16(mmA, mmE);        /* (A4 B4 C4 A5 Ac Bc Cc Ad) */
    mmA = _mm_srli_si64(mmA, 4 * BYTE_BIT);
    mmH = _mm_unpacklo_pi16(mmG, mmA);        /* (B5 C5 A6 B6 Bd Cd Ae Be) */
    mmG = _mm_unpackhi_pi16(mmE, mmG);        /* (C6 A7 B7 C7 Ce Af Bf Cf) */

    mmA = _mm_unpacklo_pi32(mmC, mmB);        /* (A0 B0 C0 A1 B1 C1 A2 B2) */
    mmE = _mm_unpackhi_pi32(mmC, mmB);        /* (A8 B8 C8 A9 B9 C9 Aa Ba) */
    mmB = _mm_unpacklo_pi32(mmD, mmF);        /* (C2 A3 B3 C3 A4 B4 C4 A5) */
    mmF = _mm_unpackhi_pi32(mmD, mmF);        /* (Ca Ab Bb Cb Ac Bc Cc Ad) */
    mmC = _mm_unpacklo_pi32(mmH, mmG);        /* (B5 C5 A6 B6 C6 A7 B7 C7) */
    mmG = _mm_unpackhi_pi32(mmH, mmG);        /* (Bd Cd Ae Be Ce Af Bf Cf) */

    if (num_cols >= 8) {
      if (!(((long)outptr) & 7)) {
        _mm_store_si64((__m64 *)outptr, mmA);
        _mm_store_si64((__m64 *)(outptr + 8), mmB);
        _mm_store_si64((__m64 *)(outptr + 16), mmC);
        _mm_store_si64((__m64 *)(outptr + 24), mmE);
        _mm_store_si64((__m64 *)(outptr + 32), mmF);
        _mm_store_si64((__m64 *)(outptr + 40), mmG);
      } else {
        _mm_storeu_si64((__m64 *)outptr, mmA);
        _mm_storeu_si64((__m64 *)(outptr + 8), mmB);
        _mm_storeu_si64((__m64 *)(outptr + 16), mmC);
        _mm_storeu_si64((__m64 *)(outptr + 24), mmE);
        _mm_storeu_si64((__m64 *)(outptr + 32), mmF);
        _mm_storeu_si64((__m64 *)(outptr + 40), mmG);
      }
      outptr += RGB_PIXELSIZE * 16;
    } else {
      if (output_width & 1)
        col = num_cols * 6 + 3;
      else
        col = num_cols * 6;

      asm(".set noreorder\r\n"                /* st24 */

          "li       $8, 24\r\n"
          "move     $9, %7\r\n"
          "mov.s    $f4, %1\r\n"
          "mov.s    $f6, %2\r\n"
          "mov.s    $f8, %3\r\n"
          "move     $10, %8\r\n"
          "bltu     $9, $8, 1f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "gssdlc1  $f8, 7+16($10)\r\n"
          "gssdrc1  $f8, 16($10)\r\n"
          "mov.s    $f4, %4\r\n"
          "mov.s    $f6, %5\r\n"
          "mov.s    $f8, %6\r\n"
          "subu     $9, $9, 24\r\n"
          PTR_ADDU  "$10, $10, 24\r\n"

          "1:       \r\n"
          "li       $8, 16\r\n"               /* st16 */
          "bltu     $9, $8, 2f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "mov.s    $f4, $f8\r\n"
          "subu     $9, $9, 16\r\n"
          PTR_ADDU  "$10, $10, 16\r\n"

          "2:       \r\n"
          "li       $8,  8\r\n"               /* st8 */
          "bltu     $9, $8, 3f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "mov.s    $f4, $f6\r\n"
          "subu     $9, $9, 8\r\n"
          PTR_ADDU  "$10, $10, 8\r\n"

          "3:       \r\n"
          "li       $8,  4\r\n"               /* st4 */
          "mfc1     $11, $f4\r\n"
          "bltu     $9, $8, 4f\r\n"
          "nop      \r\n"
          "swl      $11, 3($10)\r\n"
          "swr      $11, 0($10)\r\n"
          "li       $8, 32\r\n"
          "mtc1     $8, $f6\r\n"
          "dsrl     $f4, $f4, $f6\r\n"
          "mfc1     $11, $f4\r\n"
          "subu     $9, $9, 4\r\n"
          PTR_ADDU  "$10, $10, 4\r\n"

          "4:       \r\n"
          "li       $8, 2\r\n"                /* st2 */
          "bltu     $9, $8, 5f\r\n"
          "nop      \r\n"
          "ush      $11, 0($10)\r\n"
          "srl      $11, 16\r\n"
          "subu     $9, $9, 2\r\n"
          PTR_ADDU  "$10, $10, 2\r\n"

          "5:       \r\n"
          "li       $8, 1\r\n"                /* st1 */
          "bltu     $9, $8, 6f\r\n"
          "nop      \r\n"
          "sb       $11, 0($10)\r\n"

          "6:       \r\n"
          "nop      \r\n"                     /* end */
          : "=m" (*outptr)
          : "f" (mmA), "f" (mmB), "f" (mmC), "f" (mmE), "f" (mmF),
            "f" (mmG), "r" (col), "r" (outptr)
          : "$f4", "$f6", "$f8", "$8", "$9", "$10", "$11", "memory"
         );
    }

#else  /* RGB_PIXELSIZE == 4 */

#ifdef RGBX_FILLER_0XFF
    xe = _mm_cmpeq_pi8(xe, xe);
    xo = _mm_cmpeq_pi8(xo, xo);
#else
    xe = _mm_xor_si64(xe, xe);
    xo = _mm_xor_si64(xo, xo);
#endif

    /* NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
     * mapping of components A, B, C, and D to red, green, and blue.
     *
     * mmA = (A0 A2 A4 A6 A8 Aa Ac Ae) = AE
     * mmB = (A1 A3 A5 A7 A9 Ab Ad Af) = AO
     * mmC = (B0 B2 B4 B6 B8 Ba Bc Be) = BE
     * mmD = (B1 B3 B5 B7 B9 Bb Bd Bf) = BO
     * mmE = (C0 C2 C4 C6 C8 Ca Cc Ce) = CE
     * mmF = (C1 C3 C5 C7 C9 Cb Cd Cf) = CO
     * mmG = (D0 D2 D4 D6 D8 Da Dc De) = DE
     * mmH = (D1 D3 D5 D7 D9 Db Dd Df) = DO
     */
    mm8 = _mm_unpacklo_pi8(mmA, mmC);         /* (A0 B0 A2 B2 A4 B4 A6 B6) */
    mm9 = _mm_unpackhi_pi8(mmA, mmC);         /* (A8 B8 Aa Ba Ac Bc Ae Be) */
    mmA = _mm_unpacklo_pi8(mmE, mmG);         /* (C0 D0 C2 D2 C4 D4 C6 D6) */
    mmE = _mm_unpackhi_pi8(mmE, mmG);         /* (C8 D8 Ca Da Cc Dc Ce De) */

    mmG = _mm_unpacklo_pi8(mmB, mmD);         /* (A1 B1 A3 B3 A5 B5 A7 B7) */
    mmB = _mm_unpackhi_pi8(mmB, mmD);         /* (A9 B9 Ab Bb Ad Bd Af Bf) */
    mmD = _mm_unpacklo_pi8(mmF, mmH);         /* (C1 D1 C3 D3 C5 D5 C7 D7) */
    mmF = _mm_unpackhi_pi8(mmF, mmH);         /* (C9 D9 Cb Db Cd Dd Cf Df) */

    mmH = _mm_unpacklo_pi16(mm8, mmA);        /* (A0 B0 C0 D0 A2 B2 C2 D2) */
    mm8 = _mm_unpackhi_pi16(mm8, mmA);        /* (A4 B4 C4 D4 A6 B6 C6 D6) */
    mmA = _mm_unpacklo_pi16(mmG, mmD);        /* (A1 B1 C1 D1 A3 B3 C3 D3) */
    mmD = _mm_unpackhi_pi16(mmG, mmD);        /* (A5 B5 C5 D5 A7 B7 C7 D7) */

    mmG = _mm_unpackhi_pi16(mm9, mmE);        /* (Ac Bc Cc Dc Ae Be Ce De) */
    mm9 = _mm_unpacklo_pi16(mm9, mmE);        /* (A8 B8 C8 D8 Aa Ba Ca Da) */
    mmE = _mm_unpacklo_pi16(mmB, mmF);        /* (A9 B9 C9 D9 Ab Bb Cb Db) */
    mmF = _mm_unpackhi_pi16(mmB, mmF);        /* (Ad Bd Cd Dd Af Bf Cf Df) */

    mmB = _mm_unpackhi_pi32(mmH, mmA);        /* (A2 B2 C2 D2 A3 B3 C3 D3) */
    mmA = _mm_unpacklo_pi32(mmH, mmA);        /* (A0 B0 C0 D0 A1 B1 C1 D1) */
    mmC = _mm_unpacklo_pi32(mm8, mmD);        /* (A4 B4 C4 D4 A5 B5 C5 D5) */
    mmD = _mm_unpackhi_pi32(mm8, mmD);        /* (A6 B6 C6 D6 A7 B7 C7 D7) */

    mmH = _mm_unpackhi_pi32(mmG, mmF);        /* (Ae Be Ce De Af Bf Cf Df) */
    mmG = _mm_unpacklo_pi32(mmG, mmF);        /* (Ac Bc Cc Dc Ad Bd Cd Dd) */
    mmF = _mm_unpackhi_pi32(mm9, mmE);        /* (Aa Ba Ca Da Ab Bb Cb Db) */
    mmE = _mm_unpacklo_pi32(mm9, mmE);        /* (A8 B8 C8 D8 A9 B9 C9 D9) */

    if (num_cols >= 8) {
      if (!(((long)outptr) & 7)) {
        _mm_store_si64((__m64 *)outptr, mmA);
        _mm_store_si64((__m64 *)(outptr + 8), mmB);
        _mm_store_si64((__m64 *)(outptr + 16), mmC);
        _mm_store_si64((__m64 *)(outptr + 24), mmD);
        _mm_store_si64((__m64 *)(outptr + 32), mmE);
        _mm_store_si64((__m64 *)(outptr + 40), mmF);
        _mm_store_si64((__m64 *)(outptr + 48), mmG);
        _mm_store_si64((__m64 *)(outptr + 56), mmH);
      } else {
        _mm_storeu_si64((__m64 *)outptr, mmA);
        _mm_storeu_si64((__m64 *)(outptr + 8), mmB);
        _mm_storeu_si64((__m64 *)(outptr + 16), mmC);
        _mm_storeu_si64((__m64 *)(outptr + 24), mmD);
        _mm_storeu_si64((__m64 *)(outptr + 32), mmE);
        _mm_storeu_si64((__m64 *)(outptr + 40), mmF);
        _mm_storeu_si64((__m64 *)(outptr + 48), mmG);
        _mm_storeu_si64((__m64 *)(outptr + 56), mmH);
      }
      outptr += RGB_PIXELSIZE * 16;
    } else {
      if (output_width & 1)
        col = num_cols * 2 + 1;
      else
        col = num_cols * 2;
      asm(".set noreorder\r\n"                /* st32 */

          "li       $8, 8\r\n"
          "move     $9, %10\r\n"
          "move     $10, %11\r\n"
          "mov.s    $f4, %2\r\n"
          "mov.s    $f6, %3\r\n"
          "mov.s    $f8, %4\r\n"
          "mov.s    $f10, %5\r\n"
          "bltu     $9, $8, 1f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "gssdlc1  $f8, 7+16($10)\r\n"
          "gssdrc1  $f8, 16($10)\r\n"
          "gssdlc1  $f10, 7+24($10)\r\n"
          "gssdrc1  $f10, 24($10)\r\n"
          "mov.s    $f4, %6\r\n"
          "mov.s    $f6, %7\r\n"
          "mov.s    $f8, %8\r\n"
          "mov.s    $f10, %9\r\n"
          "subu     $9, $9, 8\r\n"
          PTR_ADDU  "$10, $10, 32\r\n"

          "1:       \r\n"
          "li       $8, 4\r\n"                /* st16 */
          "bltu     $9, $8, 2f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "gssdlc1  $f6, 7+8($10)\r\n"
          "gssdrc1  $f6, 8($10)\r\n"
          "mov.s    $f4, $f8\r\n"
          "mov.s    $f6, $f10\r\n"
          "subu     $9, $9, 4\r\n"
          PTR_ADDU  "$10, $10, 16\r\n"

          "2:       \r\n"
          "li       $8, 2\r\n"                /* st8 */
          "bltu     $9, $8, 3f\r\n"
          "nop      \r\n"
          "gssdlc1  $f4, 7($10)\r\n"
          "gssdrc1  $f4, 0($10)\r\n"
          "mov.s    $f4, $f6\r\n"
          "subu     $9, $9, 2\r\n"
          PTR_ADDU  "$10, $10, 8\r\n"

          "3:       \r\n"
          "li       $8, 1\r\n"                /* st4 */
          "bltu     $9, $8, 4f\r\n"
          "nop      \r\n"
          "gsswlc1  $f4, 3($10)\r\n"
          "gsswrc1  $f4, 0($10)\r\n"

          "4:       \r\n"
          "li       %1, 0\r\n"                /* end */
          : "=m" (*outptr), "=r" (col)
          : "f" (mmA), "f" (mmB), "f" (mmC), "f" (mmD), "f" (mmE), "f" (mmF),
            "f" (mmG), "f" (mmH), "r" (col), "r" (outptr)
          : "$f4", "$f6", "$f8", "$f10", "$8", "$9", "$10", "memory"
         );
    }

#endif

  }

  if (!((output_width >> 1) & 7)) {
    if (output_width & 1) {
      cb = _mm_load_si64((__m64 *)inptr1);
      cr = _mm_load_si64((__m64 *)inptr2);
      y = _mm_load_si64((__m64 *)inptr0);

      decenter = 0.0;
      decenter = _mm_cmpeq_pi16(decenter, decenter);
      decenter = _mm_slli_pi16(decenter, 7);  /* { 0xFF80 0xFF80 0xFF80 0xFF80 } */

      cbl = _mm_unpacklo_pi8(cb, zero);       /* Cb(0123) */
      crl = _mm_unpacklo_pi8(cr, zero);       /* Cr(0123) */
      cbl = _mm_add_pi16(cbl, decenter);
      crl = _mm_add_pi16(crl, decenter);

      cbl2 = _mm_add_pi16(cbl, cbl);          /* 2 * CbL */
      crl2 = _mm_add_pi16(crl, crl);          /* 2 * CrL */
      bl = _mm_mulhi_pi16(cbl2, PW_MF0228);   /* (2 * CbL * -FIX(0.22800) */
      rl = _mm_mulhi_pi16(crl2, PW_F0402);    /* (2 * CrL * FIX(0.40200)) */

      bl = _mm_add_pi16(bl, PW_ONE);
      bl = _mm_srai_pi16(bl, 1);              /* (CbL * -FIX(0.22800)) */
      rl = _mm_add_pi16(rl, PW_ONE);
      rl = _mm_srai_pi16(rl, 1);              /* (CrL * FIX(0.40200)) */

      bl = _mm_add_pi16(bl, cbl);
      bl = _mm_add_pi16(bl, cbl);         /* (CbL * FIX(1.77200)) = (B - Y)L */
      rl = _mm_add_pi16(rl, crl);         /* (CrL * FIX(1.40200)) = (R - Y)L */

      gl = _mm_unpacklo_pi16(cbl, crl);
      gl = _mm_madd_pi16(gl, PW_MF0344_F0285);
      gl = _mm_add_pi32(gl, PD_ONEHALF);
      gl = _mm_srai_pi32(gl, SCALEBITS);
      gl = _mm_packs_pi32(gl, zero);  /* CbL * -FIX(0.344) + CrL * FIX(0.285) */
      gl = _mm_sub_pi16(gl, crl);  /* CbL * -FIX(0.344) + CrL * -FIX(0.714) = (G - Y)L */

      yl = _mm_unpacklo_pi8(y, zero);         /* Y(0123) */
      rl = _mm_add_pi16(rl, yl);              /* (R0 R1 R2 R3) */
      gl = _mm_add_pi16(gl, yl);              /* (G0 G1 G2 G3) */
      bl = _mm_add_pi16(bl, yl);              /* (B0 B1 B2 B3) */
      re = _mm_packs_pu16(rl, rl);
      ge = _mm_packs_pu16(gl, gl);
      be = _mm_packs_pu16(bl, bl);
#if RGB_PIXELSIZE == 3
      mmA = _mm_unpacklo_pi8(mmA, mmC);
      mmA = _mm_unpacklo_pi16(mmA, mmE);
      asm(".set noreorder\r\n"

          "move    $8, %2\r\n"
          "mov.s   $f4, %1\r\n"
          "mfc1    $9, $f4\r\n"
          "ush     $9, 0($8)\r\n"
          "srl     $9, 16\r\n"
          "sb      $9, 2($8)\r\n"
          : "=m" (*outptr)
          : "f" (mmA), "r" (outptr)
          : "$f4", "$8", "$9", "memory"
         );
#else  /* RGB_PIXELSIZE == 4 */

#ifdef RGBX_FILLER_0XFF
      xe = _mm_cmpeq_pi8(xe, xe);
#else
      xe = _mm_xor_si64(xe, xe);
#endif
      mmA = _mm_unpacklo_pi8(mmA, mmC);
      mmE = _mm_unpacklo_pi8(mmE, mmG);
      mmA = _mm_unpacklo_pi16(mmA, mmE);
      asm(".set noreorder\r\n"

          "move    $8, %2\r\n"
          "mov.s   $f4, %1\r\n"
          "gsswlc1 $f4, 3($8)\r\n"
          "gsswrc1 $f4, 0($8)\r\n"
          : "=m" (*outptr)
          : "f" (mmA), "r" (outptr)
          : "$f4", "$8", "memory"
         );
#endif
    }
  }
}


HIDDEN void
jsimd_h2v2_merged_upsample_mmi(JDIMENSION output_width, JSAMPIMAGE input_buf,
                               JDIMENSION in_row_group_ctr,
                               JSAMPARRAY output_buf)
{
  JSAMPROW inptr, outptr;

  inptr = input_buf[0][in_row_group_ctr];
  outptr = output_buf[0];

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2];
  jsimd_h2v1_merged_upsample_mmi(output_width, input_buf, in_row_group_ctr,
                                 output_buf);

  input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2 + 1];
  output_buf[0] = output_buf[1];
  jsimd_h2v1_merged_upsample_mmi(output_width, input_buf, in_row_group_ctr,
                                 output_buf);

  input_buf[0][in_row_group_ctr] = inptr;
  output_buf[0] = outptr;
}


#undef mmA
#undef mmB
#undef mmC
#undef mmD
#undef mmE
#undef mmF
#undef mmG
#undef mmH
