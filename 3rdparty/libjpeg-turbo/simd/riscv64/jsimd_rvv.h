/*
 * Copyright (C) 2014, 2026, D. R. Commander.
 * Copyright (C) 2026, Olaf Bernstein.
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

#define TRANSPOSE_8x8(row, col) { \
  vint8m1_t veven = __riscv_vmv_v_x_i8m1(-86, __riscv_vsetvlmax_e8m1()); \
  vint8m1_t vodd = __riscv_vnot(veven, __riscv_vsetvlmax_e8m1()); \
  size_t VL; \
  vint64m1_t r04l, r04h, r15l, r15h, r26l, r26h, r37l, r37h; \
  vint32m1_t c01e, c01o, c23e, c23o, c45e, c45o, c67e, c67o; \
  \
  /* row##0 = 00 01 02 03 04 05 06 07 \
   * row##1 = 10 11 12 13 14 15 16 17 \
   * row##2 = 20 21 22 23 24 25 26 27 \
   * row##3 = 30 31 32 33 34 35 36 37 \
   * row##4 = 40 41 42 43 44 45 46 47 \
   * row##5 = 50 51 52 53 54 55 56 57 \
   * row##6 = 60 61 62 63 64 65 66 67 \
   * row##7 = 70 71 72 73 74 75 76 77 \
   */ \
  \
  VL = __riscv_vsetvlmax_e64m1(); \
  r04l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1(row##0), \
                              __riscv_vreinterpret_i64m1(row##4), 0, VL); \
                                              /* 00 01 02 03 | 40 41 42 43 */ \
  r15l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1(row##1), \
                              __riscv_vreinterpret_i64m1(row##5), 0, VL); \
                                              /* 10 11 12 13 | 50 51 52 53 */ \
  r26l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1(row##2), \
                              __riscv_vreinterpret_i64m1(row##6), 0, VL); \
                                              /* 20 21 22 23 | 60 61 62 63 */ \
  r37l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1(row##3), \
                              __riscv_vreinterpret_i64m1(row##7), 0, VL); \
                                              /* 30 31 32 33 | 70 71 72 73 */ \
  r04h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1(row##4), \
                                __riscv_vreinterpret_i64m1(row##0), 0, VL); \
                                              /* 04 05 06 07 | 44 45 46 47 */ \
  r15h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1(row##5), \
                                __riscv_vreinterpret_i64m1(row##1), 0, VL); \
                                              /* 14 15 16 17 | 54 55 56 57 */ \
  r26h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1(row##6), \
                                __riscv_vreinterpret_i64m1(row##2), 0, VL); \
                                              /* 24 25 26 27 | 64 65 66 67 */ \
  r37h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1(row##7), \
                                __riscv_vreinterpret_i64m1(row##3), 0, VL); \
                                              /* 34 35 36 37 | 74 75 76 77 */ \
  \
  VL = __riscv_vsetvlmax_e32m1(); \
  c23e = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                __riscv_vreinterpret_i32m1(r26l), \
                                __riscv_vreinterpret_i32m1(r04l), 0, VL); \
                                          /* 02 03 | 22 23 | 42 43 | 62 63 */ \
  c23o = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                __riscv_vreinterpret_i32m1(r37l), \
                                __riscv_vreinterpret_i32m1(r15l), 0, VL); \
                                          /* 12 13 | 32 33 | 52 53 | 72 73 */ \
  c67e = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                __riscv_vreinterpret_i32m1(r26h), \
                                __riscv_vreinterpret_i32m1(r04h), 0, VL); \
                                          /* 06 07 | 26 27 | 46 47 | 66 67 */ \
  c67o = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                __riscv_vreinterpret_i32m1(r37h), \
                                __riscv_vreinterpret_i32m1(r15h), 0, VL); \
                                          /* 16 17 | 36 37 | 56 57 | 76 77 */ \
  c01e = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                              __riscv_vreinterpret_i32m1(r04l), \
                              __riscv_vreinterpret_i32m1(r26l), 0, VL); \
                                          /* 00 01 | 20 21 | 40 41 | 60 61 */ \
  c01o = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                              __riscv_vreinterpret_i32m1(r15l), \
                              __riscv_vreinterpret_i32m1(r37l), 0, VL); \
                                          /* 10 11 | 30 31 | 50 51 | 70 71 */ \
  c45e = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                              __riscv_vreinterpret_i32m1(r04h), \
                              __riscv_vreinterpret_i32m1(r26h), 0, VL); \
                                          /* 04 05 | 24 25 | 44 45 | 64 65 */ \
  c45o = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                              __riscv_vreinterpret_i32m1(r15h), \
                              __riscv_vreinterpret_i32m1(r37h), 0, VL); \
                                          /* 14 15 | 34 35 | 54 55 | 74 75 */ \
  \
  VL = __riscv_vsetvlmax_e16m1(); \
  col##0 = __riscv_vslide1up_mu(__riscv_vreinterpret_b16(veven), \
                                __riscv_vreinterpret_i16m1(c01e), \
                                __riscv_vreinterpret_i16m1(c01o), 0, VL); \
                                                /* 00 10 20 30 40 50 60 70 */ \
  col##2 = __riscv_vslide1up_mu(__riscv_vreinterpret_b16(veven), \
                                __riscv_vreinterpret_i16m1(c23e), \
                                __riscv_vreinterpret_i16m1(c23o), 0, VL); \
                                                /* 02 12 22 32 42 52 62 72 */ \
  col##4 = __riscv_vslide1up_mu(__riscv_vreinterpret_b16(veven), \
                                __riscv_vreinterpret_i16m1(c45e), \
                                __riscv_vreinterpret_i16m1(c45o), 0, VL); \
                                                /* 04 14 24 34 44 54 64 74 */ \
  col##6 = __riscv_vslide1up_mu(__riscv_vreinterpret_b16(veven), \
                                __riscv_vreinterpret_i16m1(c67e), \
                                __riscv_vreinterpret_i16m1(c67o), 0, VL); \
                                                /* 06 16 26 36 46 56 66 76 */ \
  col##1 = __riscv_vslide1down_mu(__riscv_vreinterpret_b16(vodd), \
                                  __riscv_vreinterpret_i16m1(c01o), \
                                  __riscv_vreinterpret_i16m1(c01e), 0, VL); \
                                                /* 01 11 21 31 41 51 61 71 */ \
  col##3 = __riscv_vslide1down_mu(__riscv_vreinterpret_b16(vodd), \
                                  __riscv_vreinterpret_i16m1(c23o), \
                                  __riscv_vreinterpret_i16m1(c23e), 0, VL); \
                                                /* 03 13 23 33 43 53 63 73 */ \
  col##5 = __riscv_vslide1down_mu(__riscv_vreinterpret_b16(vodd), \
                                  __riscv_vreinterpret_i16m1(c45o), \
                                  __riscv_vreinterpret_i16m1(c45e), 0, VL); \
                                                /* 05 15 25 35 45 55 65 75 */ \
  col##7 = __riscv_vslide1down_mu(__riscv_vreinterpret_b16(vodd), \
                                  __riscv_vreinterpret_i16m1(c67o), \
                                  __riscv_vreinterpret_i16m1(c67e), 0, VL); \
                                                /* 07 17 27 37 47 57 67 77 */ \
}


/* This version of the algorithm operates on half registers (LMUL=1/2), to
 * improve performance with implementations that have 256-bit or wider
 * registers (VLEN>=256).
 */

#define TRANSPOSE_8x8_VLEN256(row, col) { \
  vint8m1_t veven = __riscv_vmv_v_x_i8m1(-86, 16); \
  vint8m1_t vodd = __riscv_vnot(veven, 16); \
  vint64m1_t r04l, r04h, r15l, r15h, r26l, r26h, r37l, r37h; \
  vint32mf2_t c01e, c01o, c23e, c23o, c45e, c45o, c67e, c67o; \
  \
  /* row##0 = 00 01 02 03 04 05 06 07 \
   * row##1 = 10 11 12 13 14 15 16 17 \
   * row##2 = 20 21 22 23 24 25 26 27 \
   * row##3 = 30 31 32 33 34 35 36 37 \
   * row##4 = 40 41 42 43 44 45 46 47 \
   * row##5 = 50 51 52 53 54 55 56 57 \
   * row##6 = 60 61 62 63 64 65 66 67 \
   * row##7 = 70 71 72 73 74 75 76 77 \
   */ \
  \
  /* LMUL=1/2 isn't supported with SEW=64, so we need to temporarily use the \
   * full register. \
   */ \
  r04l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##0)), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##4)), \
                              0, 2); \
                                              /* 00 01 02 03 | 40 41 42 43 */ \
  r15l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##1)), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##5)), \
                              0, 2); \
                                              /* 10 11 12 13 | 50 51 52 53 */ \
  r26l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##2)), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##6)), \
                              0, 2); \
                                              /* 20 21 22 23 | 60 61 62 63 */ \
  r37l = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##3)), \
                              __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##7)), \
                              0, 2); \
                                              /* 30 31 32 33 | 70 71 72 73 */ \
  r04h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##4)), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##0)), \
                                0, 2); \
                                              /* 04 05 06 07 | 44 45 46 47 */ \
  r15h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##5)), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##1)), \
                                0, 2); \
                                              /* 14 15 16 17 | 54 55 56 57 */ \
  r26h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##6)), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##2)), \
                                0, 2); \
                                              /* 24 25 26 27 | 64 65 66 67 */ \
  r37h = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##7)), \
                                __riscv_vreinterpret_i64m1( \
                                   __riscv_vlmul_ext_v_i16mf2_i16m1(row##3)), \
                                0, 2); \
                                              /* 34 35 36 37 | 74 75 76 77 */ \
  \
  c23e = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r26l)), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r04l)), \
                                0, 4); \
                                          /* 02 03 | 22 23 | 42 43 | 62 63 */ \
  c23o = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r37l)), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r15l)), \
                                0, 4); \
                                          /* 12 13 | 32 33 | 52 53 | 72 73 */ \
  c67e = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r26h)), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r04h)), \
                                0, 4); \
                                          /* 06 07 | 26 27 | 46 47 | 66 67 */ \
  c67o = __riscv_vslide1down_mu(__riscv_vreinterpret_b64(vodd), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r37h)), \
                                __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r15h)), \
                                0, 4); \
                                          /* 16 17 | 36 37 | 56 57 | 76 77 */ \
  c01e = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r04l)), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r26l)), \
                              0, 4); \
                                          /* 00 01 | 20 21 | 40 41 | 60 61 */ \
  c01o = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r15l)), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r37l)), \
                              0, 4); \
                                          /* 10 11 | 30 31 | 50 51 | 70 71 */ \
  c45e = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r04h)), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r26h)), \
                              0, 4); \
                                          /* 04 05 | 24 25 | 44 45 | 64 65 */ \
  c45o = __riscv_vslide1up_mu(__riscv_vreinterpret_b64(veven), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r15h)), \
                              __riscv_vlmul_trunc_v_i32m1_i32mf2( \
                                   __riscv_vreinterpret_v_i64m1_i32m1(r37h)), \
                              0, 4); \
                                          /* 14 15 | 34 35 | 54 55 | 74 75 */ \
  \
  col##0 = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                                __riscv_vreinterpret_i16mf2(c01e), \
                                __riscv_vreinterpret_i16mf2(c01o), 0, 8); \
                                                /* 00 10 20 30 40 50 60 70 */ \
  col##2 = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                                __riscv_vreinterpret_i16mf2(c23e), \
                                __riscv_vreinterpret_i16mf2(c23o), 0, 8); \
                                                /* 02 12 22 32 42 52 62 72 */ \
  col##4 = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                                __riscv_vreinterpret_i16mf2(c45e), \
                                __riscv_vreinterpret_i16mf2(c45o), 0, 8); \
                                                /* 04 14 24 34 44 54 64 74 */ \
  col##6 = __riscv_vslide1up_mu(__riscv_vreinterpret_b32(veven), \
                                __riscv_vreinterpret_i16mf2(c67e), \
                                __riscv_vreinterpret_i16mf2(c67o), 0, 8); \
                                                /* 06 16 26 36 46 56 66 76 */ \
  col##1 = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                  __riscv_vreinterpret_i16mf2(c01o), \
                                  __riscv_vreinterpret_i16mf2(c01e), 0, 8); \
                                                /* 01 11 21 31 41 51 61 71 */ \
  col##3 = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                  __riscv_vreinterpret_i16mf2(c23o), \
                                  __riscv_vreinterpret_i16mf2(c23e), 0, 8); \
                                                /* 03 13 23 33 43 53 63 73 */ \
  col##5 = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                  __riscv_vreinterpret_i16mf2(c45o), \
                                  __riscv_vreinterpret_i16mf2(c45e), 0, 8); \
                                                /* 05 15 25 35 45 55 65 75 */ \
  col##7 = __riscv_vslide1down_mu(__riscv_vreinterpret_b32(vodd), \
                                  __riscv_vreinterpret_i16mf2(c67o), \
                                  __riscv_vreinterpret_i16mf2(c67e), 0, 8); \
                                                /* 07 17 27 37 47 57 67 77 */ \
}
