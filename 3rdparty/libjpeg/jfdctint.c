/*
 * jfdctint.c
 *
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * Modification developed 2003-2018 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains a slow-but-accurate integer implementation of the
 * forward DCT (Discrete Cosine Transform).
 *
 * A 2-D DCT can be done by 1-D DCT on each row followed by 1-D DCT
 * on each column.  Direct algorithms are also available, but they are
 * much more complex and seem not to be any faster when reduced to code.
 *
 * This implementation is based on an algorithm described in
 *   C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 * The primary algorithm described there uses 11 multiplies and 29 adds.
 * We use their alternate method with 12 multiplies and 32 adds.
 * The advantage of this method is that no data path contains more than one
 * multiplication; this allows a very simple and accurate implementation in
 * scaled fixed-point arithmetic, with a minimal number of shifts.
 *
 * We also provide FDCT routines with various input sample block sizes for
 * direct resolution reduction or enlargement and for direct resolving the
 * common 2x1 and 1x2 subsampling cases without additional resampling: NxN
 * (N=1...16), 2NxN, and Nx2N (N=1...8) pixels for one 8x8 output DCT block.
 *
 * For N<8 we fill the remaining block coefficients with zero.
 * For N>8 we apply a partial N-point FDCT on the input samples, computing
 * just the lower 8 frequency coefficients and discarding the rest.
 *
 * We must scale the output coefficients of the N-point FDCT appropriately
 * to the standard 8-point FDCT level by 8/N per 1-D pass.  This scaling
 * is folded into the constant multipliers (pass 2) and/or final/initial
 * shifting.
 *
 * CAUTION: We rely on the FIX() macro except for the N=1,2,4,8 cases
 * since there would be too many additional constants to pre-calculate.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"		/* Private declarations for DCT subsystem */

#ifdef DCT_ISLOW_SUPPORTED


/*
 * This module is specialized to the case DCTSIZE = 8.
 */

#if DCTSIZE != 8
  Sorry, this code only copes with 8x8 DCT blocks. /* deliberate syntax err */
#endif


/*
 * The poop on this scaling stuff is as follows:
 *
 * Each 1-D DCT step produces outputs which are a factor of sqrt(N)
 * larger than the true DCT outputs.  The final outputs are therefore
 * a factor of N larger than desired; since N=8 this can be cured by
 * a simple right shift at the end of the algorithm.  The advantage of
 * this arrangement is that we save two multiplications per 1-D DCT,
 * because the y0 and y4 outputs need not be divided by sqrt(N).
 * In the IJG code, this factor of 8 is removed by the quantization step
 * (in jcdctmgr.c), NOT in this module.
 *
 * We have to do addition and subtraction of the integer inputs, which
 * is no problem, and multiplication by fractional constants, which is
 * a problem to do in integer arithmetic.  We multiply all the constants
 * by CONST_SCALE and convert them to integer constants (thus retaining
 * CONST_BITS bits of precision in the constants).  After doing a
 * multiplication we have to divide the product by CONST_SCALE, with proper
 * rounding, to produce the correct output.  This division can be done
 * cheaply as a right shift of CONST_BITS bits.  We postpone shifting
 * as long as possible so that partial sums can be added together with
 * full fractional precision.
 *
 * The outputs of the first pass are scaled up by PASS1_BITS bits so that
 * they are represented to better-than-integral precision.  These outputs
 * require BITS_IN_JSAMPLE + PASS1_BITS + 3 bits; this fits in a 16-bit word
 * with the recommended scaling.  (For 12-bit sample data, the intermediate
 * array is INT32 anyway.)
 *
 * To avoid overflow of the 32-bit intermediate results in pass 2, we must
 * have BITS_IN_JSAMPLE + CONST_BITS + PASS1_BITS <= 26.  Error analysis
 * shows that the values given below are the most effective.
 */

#if BITS_IN_JSAMPLE == 8
#define CONST_BITS  13
#define PASS1_BITS  2
#else
#define CONST_BITS  13
#define PASS1_BITS  1		/* lose a little precision to avoid overflow */
#endif

/* Some C compilers fail to reduce "FIX(constant)" at compile time, thus
 * causing a lot of useless floating-point operations at run time.
 * To get around this we use the following pre-calculated constants.
 * If you change CONST_BITS you may want to add appropriate values.
 * (With a reasonable C compiler, you can just rely on the FIX() macro...)
 */

#if CONST_BITS == 13
#define FIX_0_298631336  ((INT32)  2446)	/* FIX(0.298631336) */
#define FIX_0_390180644  ((INT32)  3196)	/* FIX(0.390180644) */
#define FIX_0_541196100  ((INT32)  4433)	/* FIX(0.541196100) */
#define FIX_0_765366865  ((INT32)  6270)	/* FIX(0.765366865) */
#define FIX_0_899976223  ((INT32)  7373)	/* FIX(0.899976223) */
#define FIX_1_175875602  ((INT32)  9633)	/* FIX(1.175875602) */
#define FIX_1_501321110  ((INT32)  12299)	/* FIX(1.501321110) */
#define FIX_1_847759065  ((INT32)  15137)	/* FIX(1.847759065) */
#define FIX_1_961570560  ((INT32)  16069)	/* FIX(1.961570560) */
#define FIX_2_053119869  ((INT32)  16819)	/* FIX(2.053119869) */
#define FIX_2_562915447  ((INT32)  20995)	/* FIX(2.562915447) */
#define FIX_3_072711026  ((INT32)  25172)	/* FIX(3.072711026) */
#else
#define FIX_0_298631336  FIX(0.298631336)
#define FIX_0_390180644  FIX(0.390180644)
#define FIX_0_541196100  FIX(0.541196100)
#define FIX_0_765366865  FIX(0.765366865)
#define FIX_0_899976223  FIX(0.899976223)
#define FIX_1_175875602  FIX(1.175875602)
#define FIX_1_501321110  FIX(1.501321110)
#define FIX_1_847759065  FIX(1.847759065)
#define FIX_1_961570560  FIX(1.961570560)
#define FIX_2_053119869  FIX(2.053119869)
#define FIX_2_562915447  FIX(2.562915447)
#define FIX_3_072711026  FIX(3.072711026)
#endif


/* Multiply an INT32 variable by an INT32 constant to yield an INT32 result.
 * For 8-bit samples with the recommended scaling, all the variable
 * and constant values involved are no more than 16 bits wide, so a
 * 16x16->32 bit multiply can be used instead of a full 32x32 multiply.
 * For 12-bit samples, a full 32-bit multiplication will be needed.
 */

#if BITS_IN_JSAMPLE == 8
#define MULTIPLY(var,const)  MULTIPLY16C16(var,const)
#else
#define MULTIPLY(var,const)  ((var) * (const))
#endif


/*
 * Perform the forward DCT on one block of samples.
 */

GLOBAL(void)
jpeg_fdct_islow (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3;
  INT32 tmp10, tmp11, tmp12, tmp13;
  INT32 z1;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * cK represents sqrt(2) * cos(K*pi/16).
   */

  dataptr = data;
  for (ctr = 0; ctr < DCTSIZE; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part per LL&M figure 1 --- note that published figure is faulty;
     * rotator "c1" should be "c6".
     */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[7]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[6]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[5]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[4]);

    tmp10 = tmp0 + tmp3;
    tmp12 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp13 = tmp1 - tmp2;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[7]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[6]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[5]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[4]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM) ((tmp10 + tmp11 - 8 * CENTERJSAMPLE) << PASS1_BITS);
    dataptr[4] = (DCTELEM) ((tmp10 - tmp11) << PASS1_BITS);

    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS-PASS1_BITS-1);

    dataptr[2] = (DCTELEM)
      RIGHT_SHIFT(z1 + MULTIPLY(tmp12, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM)
      RIGHT_SHIFT(z1 - MULTIPLY(tmp13, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS-PASS1_BITS);

    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
     * i0..i3 in the paper are tmp0..tmp3 here.
     */

    tmp12 = tmp0 + tmp2;
    tmp13 = tmp1 + tmp3;

    z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602);       /*  c3 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS-PASS1_BITS-1);

    tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);          /* -c3+c5 */
    tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);          /* -c3-c5 */
    tmp12 += z1;
    tmp13 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223);       /* -c3+c7 */
    tmp0 = MULTIPLY(tmp0, FIX_1_501321110);              /*  c1+c3-c5-c7 */
    tmp3 = MULTIPLY(tmp3, FIX_0_298631336);              /* -c1+c3+c5-c7 */
    tmp0 += z1 + tmp12;
    tmp3 += z1 + tmp13;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447);       /* -c1-c3 */
    tmp1 = MULTIPLY(tmp1, FIX_3_072711026);              /*  c1+c3+c5-c7 */
    tmp2 = MULTIPLY(tmp2, FIX_2_053119869);              /*  c1+c3-c5+c7 */
    tmp1 += z1 + tmp13;
    tmp2 += z1 + tmp12;

    dataptr[1] = (DCTELEM) RIGHT_SHIFT(tmp0, CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM) RIGHT_SHIFT(tmp1, CONST_BITS-PASS1_BITS);
    dataptr[5] = (DCTELEM) RIGHT_SHIFT(tmp2, CONST_BITS-PASS1_BITS);
    dataptr[7] = (DCTELEM) RIGHT_SHIFT(tmp3, CONST_BITS-PASS1_BITS);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * cK represents sqrt(2) * cos(K*pi/16).
   */

  dataptr = data;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part per LL&M figure 1 --- note that published figure is faulty;
     * rotator "c1" should be "c6".
     */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*4];

    /* Add fudge factor here for final descale. */
    tmp10 = tmp0 + tmp3 + (ONE << (PASS1_BITS-1));
    tmp12 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp13 = tmp1 - tmp2;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*4];

    dataptr[DCTSIZE*0] = (DCTELEM) RIGHT_SHIFT(tmp10 + tmp11, PASS1_BITS);
    dataptr[DCTSIZE*4] = (DCTELEM) RIGHT_SHIFT(tmp10 - tmp11, PASS1_BITS);

    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS+PASS1_BITS-1);

    dataptr[DCTSIZE*2] = (DCTELEM)
      RIGHT_SHIFT(z1 + MULTIPLY(tmp12, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*6] = (DCTELEM)
      RIGHT_SHIFT(z1 - MULTIPLY(tmp13, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS+PASS1_BITS);

    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
     * i0..i3 in the paper are tmp0..tmp3 here.
     */

    tmp12 = tmp0 + tmp2;
    tmp13 = tmp1 + tmp3;

    z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602);       /*  c3 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS+PASS1_BITS-1);

    tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);          /* -c3+c5 */
    tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);          /* -c3-c5 */
    tmp12 += z1;
    tmp13 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223);       /* -c3+c7 */
    tmp0 = MULTIPLY(tmp0, FIX_1_501321110);              /*  c1+c3-c5-c7 */
    tmp3 = MULTIPLY(tmp3, FIX_0_298631336);              /* -c1+c3+c5-c7 */
    tmp0 += z1 + tmp12;
    tmp3 += z1 + tmp13;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447);       /* -c1-c3 */
    tmp1 = MULTIPLY(tmp1, FIX_3_072711026);              /*  c1+c3+c5-c7 */
    tmp2 = MULTIPLY(tmp2, FIX_2_053119869);              /*  c1+c3-c5+c7 */
    tmp1 += z1 + tmp13;
    tmp2 += z1 + tmp12;

    dataptr[DCTSIZE*1] = (DCTELEM) RIGHT_SHIFT(tmp0, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM) RIGHT_SHIFT(tmp1, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*5] = (DCTELEM) RIGHT_SHIFT(tmp2, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*7] = (DCTELEM) RIGHT_SHIFT(tmp3, CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}

#ifdef DCT_SCALING_SUPPORTED


/*
 * Perform the forward DCT on a 7x7 sample block.
 */

GLOBAL(void)
jpeg_fdct_7x7 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3;
  INT32 tmp10, tmp11, tmp12;
  INT32 z1, z2, z3;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * cK represents sqrt(2) * cos(K*pi/14).
   */

  dataptr = data;
  for (ctr = 0; ctr < 7; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[6]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[5]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[4]);
    tmp3 = GETJSAMPLE(elemptr[3]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[6]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[5]);
    tmp12 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[4]);

    z1 = tmp0 + tmp2;
    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((z1 + tmp1 + tmp3 - 7 * CENTERJSAMPLE) << PASS1_BITS);
    tmp3 += tmp3;
    z1 -= tmp3;
    z1 -= tmp3;
    z1 = MULTIPLY(z1, FIX(0.353553391));                /* (c2+c6-c4)/2 */
    z2 = MULTIPLY(tmp0 - tmp2, FIX(0.920609002));       /* (c2+c4-c6)/2 */
    z3 = MULTIPLY(tmp1 - tmp2, FIX(0.314692123));       /* c6 */
    dataptr[2] = (DCTELEM) DESCALE(z1 + z2 + z3, CONST_BITS-PASS1_BITS);
    z1 -= z2;
    z2 = MULTIPLY(tmp0 - tmp1, FIX(0.881747734));       /* c4 */
    dataptr[4] = (DCTELEM)
      DESCALE(z2 + z3 - MULTIPLY(tmp1 - tmp3, FIX(0.707106781)), /* c2+c6-c4 */
	      CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM) DESCALE(z1 + z2, CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(0.935414347));   /* (c3+c1-c5)/2 */
    tmp2 = MULTIPLY(tmp10 - tmp11, FIX(0.170262339));   /* (c3+c5-c1)/2 */
    tmp0 = tmp1 - tmp2;
    tmp1 += tmp2;
    tmp2 = MULTIPLY(tmp11 + tmp12, - FIX(1.378756276)); /* -c1 */
    tmp1 += tmp2;
    tmp3 = MULTIPLY(tmp10 + tmp12, FIX(0.613604268));   /* c5 */
    tmp0 += tmp3;
    tmp2 += tmp3 + MULTIPLY(tmp12, FIX(1.870828693));   /* c3+c1-c5 */

    dataptr[1] = (DCTELEM) DESCALE(tmp0, CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp1, CONST_BITS-PASS1_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp2, CONST_BITS-PASS1_BITS);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/7)**2 = 64/49, which we fold
   * into the constant multipliers:
   * cK now represents sqrt(2) * cos(K*pi/14) * 64/49.
   */

  dataptr = data;
  for (ctr = 0; ctr < 7; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*6];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*5];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*4];
    tmp3 = dataptr[DCTSIZE*3];

    tmp10 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*6];
    tmp11 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*5];
    tmp12 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*4];

    z1 = tmp0 + tmp2;
    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(z1 + tmp1 + tmp3, FIX(1.306122449)), /* 64/49 */
	      CONST_BITS+PASS1_BITS);
    tmp3 += tmp3;
    z1 -= tmp3;
    z1 -= tmp3;
    z1 = MULTIPLY(z1, FIX(0.461784020));                /* (c2+c6-c4)/2 */
    z2 = MULTIPLY(tmp0 - tmp2, FIX(1.202428084));       /* (c2+c4-c6)/2 */
    z3 = MULTIPLY(tmp1 - tmp2, FIX(0.411026446));       /* c6 */
    dataptr[DCTSIZE*2] = (DCTELEM) DESCALE(z1 + z2 + z3, CONST_BITS+PASS1_BITS);
    z1 -= z2;
    z2 = MULTIPLY(tmp0 - tmp1, FIX(1.151670509));       /* c4 */
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(z2 + z3 - MULTIPLY(tmp1 - tmp3, FIX(0.923568041)), /* c2+c6-c4 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*6] = (DCTELEM) DESCALE(z1 + z2, CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(1.221765677));   /* (c3+c1-c5)/2 */
    tmp2 = MULTIPLY(tmp10 - tmp11, FIX(0.222383464));   /* (c3+c5-c1)/2 */
    tmp0 = tmp1 - tmp2;
    tmp1 += tmp2;
    tmp2 = MULTIPLY(tmp11 + tmp12, - FIX(1.800824523)); /* -c1 */
    tmp1 += tmp2;
    tmp3 = MULTIPLY(tmp10 + tmp12, FIX(0.801442310));   /* c5 */
    tmp0 += tmp3;
    tmp2 += tmp3 + MULTIPLY(tmp12, FIX(2.443531355));   /* c3+c1-c5 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp0, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp1, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp2, CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 6x6 sample block.
 */

GLOBAL(void)
jpeg_fdct_6x6 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2;
  INT32 tmp10, tmp11, tmp12;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * cK represents sqrt(2) * cos(K*pi/12).
   */

  dataptr = data;
  for (ctr = 0; ctr < 6; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[5]);
    tmp11 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[4]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[3]);

    tmp10 = tmp0 + tmp2;
    tmp12 = tmp0 - tmp2;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[5]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[4]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[3]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 - 6 * CENTERJSAMPLE) << PASS1_BITS);
    dataptr[2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp12, FIX(1.224744871)),                 /* c2 */
	      CONST_BITS-PASS1_BITS);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp11, FIX(0.707106781)), /* c4 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp10 = DESCALE(MULTIPLY(tmp0 + tmp2, FIX(0.366025404)),     /* c5 */
		    CONST_BITS-PASS1_BITS);

    dataptr[1] = (DCTELEM) (tmp10 + ((tmp0 + tmp1) << PASS1_BITS));
    dataptr[3] = (DCTELEM) ((tmp0 - tmp1 - tmp2) << PASS1_BITS);
    dataptr[5] = (DCTELEM) (tmp10 + ((tmp2 - tmp1) << PASS1_BITS));

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/6)**2 = 16/9, which we fold
   * into the constant multipliers:
   * cK now represents sqrt(2) * cos(K*pi/12) * 16/9.
   */

  dataptr = data;
  for (ctr = 0; ctr < 6; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*5];
    tmp11 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*3];

    tmp10 = tmp0 + tmp2;
    tmp12 = tmp0 - tmp2;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*5];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*3];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11, FIX(1.777777778)),         /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp12, FIX(2.177324216)),                 /* c2 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp11, FIX(1.257078722)), /* c4 */
	      CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp0 + tmp2, FIX(0.650711829));             /* c5 */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp0 + tmp1, FIX(1.777777778)),   /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp1 - tmp2, FIX(1.777777778)),    /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp2 - tmp1, FIX(1.777777778)),   /* 16/9 */
	      CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 5x5 sample block.
 */

GLOBAL(void)
jpeg_fdct_5x5 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2;
  INT32 tmp10, tmp11;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * We scale the results further by 2 as part of output adaption
   * scaling for different DCT size.
   * cK represents sqrt(2) * cos(K*pi/10).
   */

  dataptr = data;
  for (ctr = 0; ctr < 5; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[4]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[3]);
    tmp2 = GETJSAMPLE(elemptr[2]);

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[4]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[3]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp2 - 5 * CENTERJSAMPLE) << (PASS1_BITS+1));
    tmp11 = MULTIPLY(tmp11, FIX(0.790569415));          /* (c2+c4)/2 */
    tmp10 -= tmp2 << 2;
    tmp10 = MULTIPLY(tmp10, FIX(0.353553391));          /* (c2-c4)/2 */
    dataptr[2] = (DCTELEM) DESCALE(tmp11 + tmp10, CONST_BITS-PASS1_BITS-1);
    dataptr[4] = (DCTELEM) DESCALE(tmp11 - tmp10, CONST_BITS-PASS1_BITS-1);

    /* Odd part */

    tmp10 = MULTIPLY(tmp0 + tmp1, FIX(0.831253876));    /* c3 */

    dataptr[1] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp0, FIX(0.513743148)), /* c1-c3 */
	      CONST_BITS-PASS1_BITS-1);
    dataptr[3] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp1, FIX(2.176250899)), /* c1+c3 */
	      CONST_BITS-PASS1_BITS-1);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/5)**2 = 64/25, which we partially
   * fold into the constant multipliers (other part was done in pass 1):
   * cK now represents sqrt(2) * cos(K*pi/10) * 32/25.
   */

  dataptr = data;
  for (ctr = 0; ctr < 5; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*4];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*3];
    tmp2 = dataptr[DCTSIZE*2];

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*4];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*3];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp2, FIX(1.28)),        /* 32/25 */
	      CONST_BITS+PASS1_BITS);
    tmp11 = MULTIPLY(tmp11, FIX(1.011928851));          /* (c2+c4)/2 */
    tmp10 -= tmp2 << 2;
    tmp10 = MULTIPLY(tmp10, FIX(0.452548340));          /* (c2-c4)/2 */
    dataptr[DCTSIZE*2] = (DCTELEM) DESCALE(tmp11 + tmp10, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*4] = (DCTELEM) DESCALE(tmp11 - tmp10, CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp0 + tmp1, FIX(1.064004961));    /* c3 */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp0, FIX(0.657591230)), /* c1-c3 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp1, FIX(2.785601151)), /* c1+c3 */
	      CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 4x4 sample block.
 */

GLOBAL(void)
jpeg_fdct_4x4 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1;
  INT32 tmp10, tmp11;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * We must also scale the output by (8/4)**2 = 2**2, which we add here.
   * cK represents sqrt(2) * cos(K*pi/16) [refers to 8-point FDCT].
   */

  dataptr = data;
  for (ctr = 0; ctr < 4; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[3]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[2]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[3]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[2]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp0 + tmp1 - 4 * CENTERJSAMPLE) << (PASS1_BITS+2));
    dataptr[2] = (DCTELEM) ((tmp0 - tmp1) << (PASS1_BITS+2));

    /* Odd part */

    tmp0 = MULTIPLY(tmp10 + tmp11, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    tmp0 += ONE << (CONST_BITS-PASS1_BITS-3);

    dataptr[1] = (DCTELEM)
      RIGHT_SHIFT(tmp0 + MULTIPLY(tmp10, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS-PASS1_BITS-2);
    dataptr[3] = (DCTELEM)
      RIGHT_SHIFT(tmp0 - MULTIPLY(tmp11, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS-PASS1_BITS-2);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * cK represents sqrt(2) * cos(K*pi/16) [refers to 8-point FDCT].
   */

  dataptr = data;
  for (ctr = 0; ctr < 4; ctr++) {
    /* Even part */

    /* Add fudge factor here for final descale. */
    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*3] + (ONE << (PASS1_BITS-1));
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*2];

    tmp10 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*3];
    tmp11 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*2];

    dataptr[DCTSIZE*0] = (DCTELEM) RIGHT_SHIFT(tmp0 + tmp1, PASS1_BITS);
    dataptr[DCTSIZE*2] = (DCTELEM) RIGHT_SHIFT(tmp0 - tmp1, PASS1_BITS);

    /* Odd part */

    tmp0 = MULTIPLY(tmp10 + tmp11, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    tmp0 += ONE << (CONST_BITS+PASS1_BITS-1);

    dataptr[DCTSIZE*1] = (DCTELEM)
      RIGHT_SHIFT(tmp0 + MULTIPLY(tmp10, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM)
      RIGHT_SHIFT(tmp0 - MULTIPLY(tmp11, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 3x3 sample block.
 */

GLOBAL(void)
jpeg_fdct_3x3 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * We scale the results further by 2**2 as part of output adaption
   * scaling for different DCT size.
   * cK represents sqrt(2) * cos(K*pi/6).
   */

  dataptr = data;
  for (ctr = 0; ctr < 3; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[2]);
    tmp1 = GETJSAMPLE(elemptr[1]);

    tmp2 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[2]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp0 + tmp1 - 3 * CENTERJSAMPLE) << (PASS1_BITS+2));
    dataptr[2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp1 - tmp1, FIX(0.707106781)), /* c2 */
	      CONST_BITS-PASS1_BITS-2);

    /* Odd part */

    dataptr[1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp2, FIX(1.224744871)),               /* c1 */
	      CONST_BITS-PASS1_BITS-2);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/3)**2 = 64/9, which we partially
   * fold into the constant multipliers (other part was done in pass 1):
   * cK now represents sqrt(2) * cos(K*pi/6) * 16/9.
   */

  dataptr = data;
  for (ctr = 0; ctr < 3; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*2];
    tmp1 = dataptr[DCTSIZE*1];

    tmp2 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*2];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 + tmp1, FIX(1.777777778)),        /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp1 - tmp1, FIX(1.257078722)), /* c2 */
	      CONST_BITS+PASS1_BITS);

    /* Odd part */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp2, FIX(2.177324216)),               /* c1 */
	      CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 2x2 sample block.
 */

GLOBAL(void)
jpeg_fdct_2x2 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  DCTELEM tmp0, tmp1, tmp2, tmp3;
  JSAMPROW elemptr;

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT.
   */

  /* Row 0 */
  elemptr = sample_data[0] + start_col;

  tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[1]);
  tmp1 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[1]);

  /* Row 1 */
  elemptr = sample_data[1] + start_col;

  tmp2 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[1]);
  tmp3 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[1]);

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/2)**2 = 2**4.
   */

  /* Column 0 */
  /* Apply unsigned->signed conversion. */
  data[DCTSIZE*0] = (tmp0 + tmp2 - 4 * CENTERJSAMPLE) << 4;
  data[DCTSIZE*1] = (tmp0 - tmp2) << 4;

  /* Column 1 */
  data[DCTSIZE*0+1] = (tmp1 + tmp3) << 4;
  data[DCTSIZE*1+1] = (tmp1 - tmp3) << 4;
}


/*
 * Perform the forward DCT on a 1x1 sample block.
 */

GLOBAL(void)
jpeg_fdct_1x1 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  DCTELEM dcval;

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  dcval = GETJSAMPLE(sample_data[0][start_col]);

  /* We leave the result scaled up by an overall factor of 8. */
  /* We must also scale the output by (8/1)**2 = 2**6. */
  /* Apply unsigned->signed conversion. */
  data[0] = (dcval - CENTERJSAMPLE) << 6;
}


/*
 * Perform the forward DCT on a 9x9 sample block.
 */

GLOBAL(void)
jpeg_fdct_9x9 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4;
  INT32 tmp10, tmp11, tmp12, tmp13;
  INT32 z1, z2;
  DCTELEM workspace[8];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * we scale the results further by 2 as part of output adaption
   * scaling for different DCT size.
   * cK represents sqrt(2) * cos(K*pi/18).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[8]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[7]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[6]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[5]);
    tmp4 = GETJSAMPLE(elemptr[4]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[8]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[7]);
    tmp12 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[6]);
    tmp13 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[5]);

    z1 = tmp0 + tmp2 + tmp3;
    z2 = tmp1 + tmp4;
    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM) ((z1 + z2 - 9 * CENTERJSAMPLE) << 1);
    dataptr[6] = (DCTELEM)
      DESCALE(MULTIPLY(z1 - z2 - z2, FIX(0.707106781)),  /* c6 */
	      CONST_BITS-1);
    z1 = MULTIPLY(tmp0 - tmp2, FIX(1.328926049));        /* c2 */
    z2 = MULTIPLY(tmp1 - tmp4 - tmp4, FIX(0.707106781)); /* c6 */
    dataptr[2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp2 - tmp3, FIX(1.083350441))    /* c4 */
	      + z1 + z2, CONST_BITS-1);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp3 - tmp0, FIX(0.245575608))    /* c8 */
	      + z1 - z2, CONST_BITS-1);

    /* Odd part */

    dataptr[3] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12 - tmp13, FIX(1.224744871)), /* c3 */
	      CONST_BITS-1);

    tmp11 = MULTIPLY(tmp11, FIX(1.224744871));        /* c3 */
    tmp0 = MULTIPLY(tmp10 + tmp12, FIX(0.909038955)); /* c5 */
    tmp1 = MULTIPLY(tmp10 + tmp13, FIX(0.483689525)); /* c7 */

    dataptr[1] = (DCTELEM) DESCALE(tmp11 + tmp0 + tmp1, CONST_BITS-1);

    tmp2 = MULTIPLY(tmp12 - tmp13, FIX(1.392728481)); /* c1 */

    dataptr[5] = (DCTELEM) DESCALE(tmp0 - tmp11 - tmp2, CONST_BITS-1);
    dataptr[7] = (DCTELEM) DESCALE(tmp1 - tmp11 + tmp2, CONST_BITS-1);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 9)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/9)**2 = 64/81, which we partially
   * fold into the constant multipliers and final/initial shifting:
   * cK now represents sqrt(2) * cos(K*pi/18) * 128/81.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*0];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*7];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*6];
    tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*5];
    tmp4 = dataptr[DCTSIZE*4];

    tmp10 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*0];
    tmp11 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*7];
    tmp12 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*6];
    tmp13 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*5];

    z1 = tmp0 + tmp2 + tmp3;
    z2 = tmp1 + tmp4;
    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(z1 + z2, FIX(1.580246914)),       /* 128/81 */
	      CONST_BITS+2);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(MULTIPLY(z1 - z2 - z2, FIX(1.117403309)),  /* c6 */
	      CONST_BITS+2);
    z1 = MULTIPLY(tmp0 - tmp2, FIX(2.100031287));        /* c2 */
    z2 = MULTIPLY(tmp1 - tmp4 - tmp4, FIX(1.117403309)); /* c6 */
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp2 - tmp3, FIX(1.711961190))    /* c4 */
	      + z1 + z2, CONST_BITS+2);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp3 - tmp0, FIX(0.388070096))    /* c8 */
	      + z1 - z2, CONST_BITS+2);

    /* Odd part */

    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12 - tmp13, FIX(1.935399303)), /* c3 */
	      CONST_BITS+2);

    tmp11 = MULTIPLY(tmp11, FIX(1.935399303));        /* c3 */
    tmp0 = MULTIPLY(tmp10 + tmp12, FIX(1.436506004)); /* c5 */
    tmp1 = MULTIPLY(tmp10 + tmp13, FIX(0.764348879)); /* c7 */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp11 + tmp0 + tmp1, CONST_BITS+2);

    tmp2 = MULTIPLY(tmp12 - tmp13, FIX(2.200854883)); /* c1 */

    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(tmp0 - tmp11 - tmp2, CONST_BITS+2);
    dataptr[DCTSIZE*7] = (DCTELEM)
      DESCALE(tmp1 - tmp11 + tmp2, CONST_BITS+2);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 10x10 sample block.
 */

GLOBAL(void)
jpeg_fdct_10x10 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14;
  DCTELEM workspace[8*2];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * we scale the results further by 2 as part of output adaption
   * scaling for different DCT size.
   * cK represents sqrt(2) * cos(K*pi/20).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[9]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[8]);
    tmp12 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[7]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[6]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[5]);

    tmp10 = tmp0 + tmp4;
    tmp13 = tmp0 - tmp4;
    tmp11 = tmp1 + tmp3;
    tmp14 = tmp1 - tmp3;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[9]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[8]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[7]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[6]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[5]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 + tmp12 - 10 * CENTERJSAMPLE) << 1);
    tmp12 += tmp12;
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.144122806)) - /* c4 */
	      MULTIPLY(tmp11 - tmp12, FIX(0.437016024)),  /* c8 */
	      CONST_BITS-1);
    tmp10 = MULTIPLY(tmp13 + tmp14, FIX(0.831253876));    /* c6 */
    dataptr[2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp13, FIX(0.513743148)),  /* c2-c6 */
	      CONST_BITS-1);
    dataptr[6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(2.176250899)),  /* c2+c6 */
	      CONST_BITS-1);

    /* Odd part */

    tmp10 = tmp0 + tmp4;
    tmp11 = tmp1 - tmp3;
    dataptr[5] = (DCTELEM) ((tmp10 - tmp11 - tmp2) << 1);
    tmp2 <<= CONST_BITS;
    dataptr[1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0, FIX(1.396802247)) +          /* c1 */
	      MULTIPLY(tmp1, FIX(1.260073511)) + tmp2 +   /* c3 */
	      MULTIPLY(tmp3, FIX(0.642039522)) +          /* c7 */
	      MULTIPLY(tmp4, FIX(0.221231742)),           /* c9 */
	      CONST_BITS-1);
    tmp12 = MULTIPLY(tmp0 - tmp4, FIX(0.951056516)) -     /* (c3+c7)/2 */
	    MULTIPLY(tmp1 + tmp3, FIX(0.587785252));      /* (c1-c9)/2 */
    tmp13 = MULTIPLY(tmp10 + tmp11, FIX(0.309016994)) +   /* (c3-c7)/2 */
	    (tmp11 << (CONST_BITS - 1)) - tmp2;
    dataptr[3] = (DCTELEM) DESCALE(tmp12 + tmp13, CONST_BITS-1);
    dataptr[7] = (DCTELEM) DESCALE(tmp12 - tmp13, CONST_BITS-1);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 10)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/10)**2 = 16/25, which we partially
   * fold into the constant multipliers and final/initial shifting:
   * cK now represents sqrt(2) * cos(K*pi/20) * 32/25.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*1];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*0];
    tmp12 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*7];
    tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*6];
    tmp4 = dataptr[DCTSIZE*4] + dataptr[DCTSIZE*5];

    tmp10 = tmp0 + tmp4;
    tmp13 = tmp0 - tmp4;
    tmp11 = tmp1 + tmp3;
    tmp14 = tmp1 - tmp3;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*1];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*0];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*7];
    tmp3 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*6];
    tmp4 = dataptr[DCTSIZE*4] - dataptr[DCTSIZE*5];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11 + tmp12, FIX(1.28)), /* 32/25 */
	      CONST_BITS+2);
    tmp12 += tmp12;
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.464477191)) - /* c4 */
	      MULTIPLY(tmp11 - tmp12, FIX(0.559380511)),  /* c8 */
	      CONST_BITS+2);
    tmp10 = MULTIPLY(tmp13 + tmp14, FIX(1.064004961));    /* c6 */
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp13, FIX(0.657591230)),  /* c2-c6 */
	      CONST_BITS+2);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(2.785601151)),  /* c2+c6 */
	      CONST_BITS+2);

    /* Odd part */

    tmp10 = tmp0 + tmp4;
    tmp11 = tmp1 - tmp3;
    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp2, FIX(1.28)),  /* 32/25 */
	      CONST_BITS+2);
    tmp2 = MULTIPLY(tmp2, FIX(1.28));                     /* 32/25 */
    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0, FIX(1.787906876)) +          /* c1 */
	      MULTIPLY(tmp1, FIX(1.612894094)) + tmp2 +   /* c3 */
	      MULTIPLY(tmp3, FIX(0.821810588)) +          /* c7 */
	      MULTIPLY(tmp4, FIX(0.283176630)),           /* c9 */
	      CONST_BITS+2);
    tmp12 = MULTIPLY(tmp0 - tmp4, FIX(1.217352341)) -     /* (c3+c7)/2 */
	    MULTIPLY(tmp1 + tmp3, FIX(0.752365123));      /* (c1-c9)/2 */
    tmp13 = MULTIPLY(tmp10 + tmp11, FIX(0.395541753)) +   /* (c3-c7)/2 */
	    MULTIPLY(tmp11, FIX(0.64)) - tmp2;            /* 16/25 */
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp12 + tmp13, CONST_BITS+2);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp12 - tmp13, CONST_BITS+2);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on an 11x11 sample block.
 */

GLOBAL(void)
jpeg_fdct_11x11 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14;
  INT32 z1, z2, z3;
  DCTELEM workspace[8*3];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * we scale the results further by 2 as part of output adaption
   * scaling for different DCT size.
   * cK represents sqrt(2) * cos(K*pi/22).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[10]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[9]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[8]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[7]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[6]);
    tmp5 = GETJSAMPLE(elemptr[5]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[10]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[9]);
    tmp12 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[8]);
    tmp13 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[7]);
    tmp14 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[6]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 - 11 * CENTERJSAMPLE) << 1);
    tmp5 += tmp5;
    tmp0 -= tmp5;
    tmp1 -= tmp5;
    tmp2 -= tmp5;
    tmp3 -= tmp5;
    tmp4 -= tmp5;
    z1 = MULTIPLY(tmp0 + tmp3, FIX(1.356927976)) +       /* c2 */
	 MULTIPLY(tmp2 + tmp4, FIX(0.201263574));        /* c10 */
    z2 = MULTIPLY(tmp1 - tmp3, FIX(0.926112931));        /* c6 */
    z3 = MULTIPLY(tmp0 - tmp1, FIX(1.189712156));        /* c4 */
    dataptr[2] = (DCTELEM)
      DESCALE(z1 + z2 - MULTIPLY(tmp3, FIX(1.018300590)) /* c2+c8-c6 */
	      - MULTIPLY(tmp4, FIX(1.390975730)),        /* c4+c10 */
	      CONST_BITS-1);
    dataptr[4] = (DCTELEM)
      DESCALE(z2 + z3 + MULTIPLY(tmp1, FIX(0.062335650)) /* c4-c6-c10 */
	      - MULTIPLY(tmp2, FIX(1.356927976))         /* c2 */
	      + MULTIPLY(tmp4, FIX(0.587485545)),        /* c8 */
	      CONST_BITS-1);
    dataptr[6] = (DCTELEM)
      DESCALE(z1 + z3 - MULTIPLY(tmp0, FIX(1.620527200)) /* c2+c4-c6 */
	      - MULTIPLY(tmp2, FIX(0.788749120)),        /* c8+c10 */
	      CONST_BITS-1);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(1.286413905));    /* c3 */
    tmp2 = MULTIPLY(tmp10 + tmp12, FIX(1.068791298));    /* c5 */
    tmp3 = MULTIPLY(tmp10 + tmp13, FIX(0.764581576));    /* c7 */
    tmp0 = tmp1 + tmp2 + tmp3 - MULTIPLY(tmp10, FIX(1.719967871)) /* c7+c5+c3-c1 */
	   + MULTIPLY(tmp14, FIX(0.398430003));          /* c9 */
    tmp4 = MULTIPLY(tmp11 + tmp12, - FIX(0.764581576));  /* -c7 */
    tmp5 = MULTIPLY(tmp11 + tmp13, - FIX(1.399818907));  /* -c1 */
    tmp1 += tmp4 + tmp5 + MULTIPLY(tmp11, FIX(1.276416582)) /* c9+c7+c1-c3 */
	    - MULTIPLY(tmp14, FIX(1.068791298));         /* c5 */
    tmp10 = MULTIPLY(tmp12 + tmp13, FIX(0.398430003));   /* c9 */
    tmp2 += tmp4 + tmp10 - MULTIPLY(tmp12, FIX(1.989053629)) /* c9+c5+c3-c7 */
	    + MULTIPLY(tmp14, FIX(1.399818907));         /* c1 */
    tmp3 += tmp5 + tmp10 + MULTIPLY(tmp13, FIX(1.305598626)) /* c1+c5-c9-c7 */
	    - MULTIPLY(tmp14, FIX(1.286413905));         /* c3 */

    dataptr[1] = (DCTELEM) DESCALE(tmp0, CONST_BITS-1);
    dataptr[3] = (DCTELEM) DESCALE(tmp1, CONST_BITS-1);
    dataptr[5] = (DCTELEM) DESCALE(tmp2, CONST_BITS-1);
    dataptr[7] = (DCTELEM) DESCALE(tmp3, CONST_BITS-1);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 11)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/11)**2 = 64/121, which we partially
   * fold into the constant multipliers and final/initial shifting:
   * cK now represents sqrt(2) * cos(K*pi/22) * 128/121.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*2];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*1];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*0];
    tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*7];
    tmp4 = dataptr[DCTSIZE*4] + dataptr[DCTSIZE*6];
    tmp5 = dataptr[DCTSIZE*5];

    tmp10 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*2];
    tmp11 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*1];
    tmp12 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*0];
    tmp13 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*7];
    tmp14 = dataptr[DCTSIZE*4] - dataptr[DCTSIZE*6];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5,
		       FIX(1.057851240)),                /* 128/121 */
	      CONST_BITS+2);
    tmp5 += tmp5;
    tmp0 -= tmp5;
    tmp1 -= tmp5;
    tmp2 -= tmp5;
    tmp3 -= tmp5;
    tmp4 -= tmp5;
    z1 = MULTIPLY(tmp0 + tmp3, FIX(1.435427942)) +       /* c2 */
	 MULTIPLY(tmp2 + tmp4, FIX(0.212906922));        /* c10 */
    z2 = MULTIPLY(tmp1 - tmp3, FIX(0.979689713));        /* c6 */
    z3 = MULTIPLY(tmp0 - tmp1, FIX(1.258538479));        /* c4 */
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(z1 + z2 - MULTIPLY(tmp3, FIX(1.077210542)) /* c2+c8-c6 */
	      - MULTIPLY(tmp4, FIX(1.471445400)),        /* c4+c10 */
	      CONST_BITS+2);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(z2 + z3 + MULTIPLY(tmp1, FIX(0.065941844)) /* c4-c6-c10 */
	      - MULTIPLY(tmp2, FIX(1.435427942))         /* c2 */
	      + MULTIPLY(tmp4, FIX(0.621472312)),        /* c8 */
	      CONST_BITS+2);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(z1 + z3 - MULTIPLY(tmp0, FIX(1.714276708)) /* c2+c4-c6 */
	      - MULTIPLY(tmp2, FIX(0.834379234)),        /* c8+c10 */
	      CONST_BITS+2);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(1.360834544));    /* c3 */
    tmp2 = MULTIPLY(tmp10 + tmp12, FIX(1.130622199));    /* c5 */
    tmp3 = MULTIPLY(tmp10 + tmp13, FIX(0.808813568));    /* c7 */
    tmp0 = tmp1 + tmp2 + tmp3 - MULTIPLY(tmp10, FIX(1.819470145)) /* c7+c5+c3-c1 */
	   + MULTIPLY(tmp14, FIX(0.421479672));          /* c9 */
    tmp4 = MULTIPLY(tmp11 + tmp12, - FIX(0.808813568));  /* -c7 */
    tmp5 = MULTIPLY(tmp11 + tmp13, - FIX(1.480800167));  /* -c1 */
    tmp1 += tmp4 + tmp5 + MULTIPLY(tmp11, FIX(1.350258864)) /* c9+c7+c1-c3 */
	    - MULTIPLY(tmp14, FIX(1.130622199));         /* c5 */
    tmp10 = MULTIPLY(tmp12 + tmp13, FIX(0.421479672));   /* c9 */
    tmp2 += tmp4 + tmp10 - MULTIPLY(tmp12, FIX(2.104122847)) /* c9+c5+c3-c7 */
	    + MULTIPLY(tmp14, FIX(1.480800167));         /* c1 */
    tmp3 += tmp5 + tmp10 + MULTIPLY(tmp13, FIX(1.381129125)) /* c1+c5-c9-c7 */
	    - MULTIPLY(tmp14, FIX(1.360834544));         /* c3 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp0, CONST_BITS+2);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp1, CONST_BITS+2);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp2, CONST_BITS+2);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp3, CONST_BITS+2);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 12x12 sample block.
 */

GLOBAL(void)
jpeg_fdct_12x12 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  DCTELEM workspace[8*4];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT.
   * cK represents sqrt(2) * cos(K*pi/24).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[11]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[10]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[9]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[8]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[7]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[6]);

    tmp10 = tmp0 + tmp5;
    tmp13 = tmp0 - tmp5;
    tmp11 = tmp1 + tmp4;
    tmp14 = tmp1 - tmp4;
    tmp12 = tmp2 + tmp3;
    tmp15 = tmp2 - tmp3;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[11]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[10]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[9]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[8]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[7]);
    tmp5 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[6]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM) (tmp10 + tmp11 + tmp12 - 12 * CENTERJSAMPLE);
    dataptr[6] = (DCTELEM) (tmp13 - tmp14 - tmp15);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.224744871)), /* c4 */
	      CONST_BITS);
    dataptr[2] = (DCTELEM)
      DESCALE(tmp14 - tmp15 + MULTIPLY(tmp13 + tmp15, FIX(1.366025404)), /* c2 */
	      CONST_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp1 + tmp4, FIX_0_541196100);    /* c9 */
    tmp14 = tmp10 + MULTIPLY(tmp1, FIX_0_765366865);   /* c3-c9 */
    tmp15 = tmp10 - MULTIPLY(tmp4, FIX_1_847759065);   /* c3+c9 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.121971054));   /* c5 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(0.860918669));   /* c7 */
    tmp10 = tmp12 + tmp13 + tmp14 - MULTIPLY(tmp0, FIX(0.580774953)) /* c5+c7-c1 */
	    + MULTIPLY(tmp5, FIX(0.184591911));        /* c11 */
    tmp11 = MULTIPLY(tmp2 + tmp3, - FIX(0.184591911)); /* -c11 */
    tmp12 += tmp11 - tmp15 - MULTIPLY(tmp2, FIX(2.339493912)) /* c1+c5-c11 */
	    + MULTIPLY(tmp5, FIX(0.860918669));        /* c7 */
    tmp13 += tmp11 - tmp14 + MULTIPLY(tmp3, FIX(0.725788011)) /* c1+c11-c7 */
	    - MULTIPLY(tmp5, FIX(1.121971054));        /* c5 */
    tmp11 = tmp15 + MULTIPLY(tmp0 - tmp3, FIX(1.306562965)) /* c3 */
	    - MULTIPLY(tmp2 + tmp5, FIX_0_541196100);  /* c9 */

    dataptr[1] = (DCTELEM) DESCALE(tmp10, CONST_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp11, CONST_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp12, CONST_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp13, CONST_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 12)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/12)**2 = 4/9, which we partially
   * fold into the constant multipliers and final shifting:
   * cK now represents sqrt(2) * cos(K*pi/24) * 8/9.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*3];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*2];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*1];
    tmp3 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*0];
    tmp4 = dataptr[DCTSIZE*4] + dataptr[DCTSIZE*7];
    tmp5 = dataptr[DCTSIZE*5] + dataptr[DCTSIZE*6];

    tmp10 = tmp0 + tmp5;
    tmp13 = tmp0 - tmp5;
    tmp11 = tmp1 + tmp4;
    tmp14 = tmp1 - tmp4;
    tmp12 = tmp2 + tmp3;
    tmp15 = tmp2 - tmp3;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*3];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*2];
    tmp2 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*1];
    tmp3 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*0];
    tmp4 = dataptr[DCTSIZE*4] - dataptr[DCTSIZE*7];
    tmp5 = dataptr[DCTSIZE*5] - dataptr[DCTSIZE*6];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11 + tmp12, FIX(0.888888889)), /* 8/9 */
	      CONST_BITS+1);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(MULTIPLY(tmp13 - tmp14 - tmp15, FIX(0.888888889)), /* 8/9 */
	      CONST_BITS+1);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.088662108)),         /* c4 */
	      CONST_BITS+1);
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp14 - tmp15, FIX(0.888888889)) +        /* 8/9 */
	      MULTIPLY(tmp13 + tmp15, FIX(1.214244803)),         /* c2 */
	      CONST_BITS+1);

    /* Odd part */

    tmp10 = MULTIPLY(tmp1 + tmp4, FIX(0.481063200));   /* c9 */
    tmp14 = tmp10 + MULTIPLY(tmp1, FIX(0.680326102));  /* c3-c9 */
    tmp15 = tmp10 - MULTIPLY(tmp4, FIX(1.642452502));  /* c3+c9 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(0.997307603));   /* c5 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(0.765261039));   /* c7 */
    tmp10 = tmp12 + tmp13 + tmp14 - MULTIPLY(tmp0, FIX(0.516244403)) /* c5+c7-c1 */
	    + MULTIPLY(tmp5, FIX(0.164081699));        /* c11 */
    tmp11 = MULTIPLY(tmp2 + tmp3, - FIX(0.164081699)); /* -c11 */
    tmp12 += tmp11 - tmp15 - MULTIPLY(tmp2, FIX(2.079550144)) /* c1+c5-c11 */
	    + MULTIPLY(tmp5, FIX(0.765261039));        /* c7 */
    tmp13 += tmp11 - tmp14 + MULTIPLY(tmp3, FIX(0.645144899)) /* c1+c11-c7 */
	    - MULTIPLY(tmp5, FIX(0.997307603));        /* c5 */
    tmp11 = tmp15 + MULTIPLY(tmp0 - tmp3, FIX(1.161389302)) /* c3 */
	    - MULTIPLY(tmp2 + tmp5, FIX(0.481063200)); /* c9 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp10, CONST_BITS+1);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp11, CONST_BITS+1);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp12, CONST_BITS+1);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp13, CONST_BITS+1);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 13x13 sample block.
 */

GLOBAL(void)
jpeg_fdct_13x13 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  INT32 z1, z2;
  DCTELEM workspace[8*5];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT.
   * cK represents sqrt(2) * cos(K*pi/26).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[12]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[11]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[10]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[9]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[8]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[7]);
    tmp6 = GETJSAMPLE(elemptr[6]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[12]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[11]);
    tmp12 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[10]);
    tmp13 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[9]);
    tmp14 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[8]);
    tmp15 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[7]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      (tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 - 13 * CENTERJSAMPLE);
    tmp6 += tmp6;
    tmp0 -= tmp6;
    tmp1 -= tmp6;
    tmp2 -= tmp6;
    tmp3 -= tmp6;
    tmp4 -= tmp6;
    tmp5 -= tmp6;
    dataptr[2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0, FIX(1.373119086)) +   /* c2 */
	      MULTIPLY(tmp1, FIX(1.058554052)) +   /* c6 */
	      MULTIPLY(tmp2, FIX(0.501487041)) -   /* c10 */
	      MULTIPLY(tmp3, FIX(0.170464608)) -   /* c12 */
	      MULTIPLY(tmp4, FIX(0.803364869)) -   /* c8 */
	      MULTIPLY(tmp5, FIX(1.252223920)),    /* c4 */
	      CONST_BITS);
    z1 = MULTIPLY(tmp0 - tmp2, FIX(1.155388986)) - /* (c4+c6)/2 */
	 MULTIPLY(tmp3 - tmp4, FIX(0.435816023)) - /* (c2-c10)/2 */
	 MULTIPLY(tmp1 - tmp5, FIX(0.316450131));  /* (c8-c12)/2 */
    z2 = MULTIPLY(tmp0 + tmp2, FIX(0.096834934)) - /* (c4-c6)/2 */
	 MULTIPLY(tmp3 + tmp4, FIX(0.937303064)) + /* (c2+c10)/2 */
	 MULTIPLY(tmp1 + tmp5, FIX(0.486914739));  /* (c8+c12)/2 */

    dataptr[4] = (DCTELEM) DESCALE(z1 + z2, CONST_BITS);
    dataptr[6] = (DCTELEM) DESCALE(z1 - z2, CONST_BITS);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(1.322312651));   /* c3 */
    tmp2 = MULTIPLY(tmp10 + tmp12, FIX(1.163874945));   /* c5 */
    tmp3 = MULTIPLY(tmp10 + tmp13, FIX(0.937797057)) +  /* c7 */
	   MULTIPLY(tmp14 + tmp15, FIX(0.338443458));   /* c11 */
    tmp0 = tmp1 + tmp2 + tmp3 -
	   MULTIPLY(tmp10, FIX(2.020082300)) +          /* c3+c5+c7-c1 */
	   MULTIPLY(tmp14, FIX(0.318774355));           /* c9-c11 */
    tmp4 = MULTIPLY(tmp14 - tmp15, FIX(0.937797057)) -  /* c7 */
	   MULTIPLY(tmp11 + tmp12, FIX(0.338443458));   /* c11 */
    tmp5 = MULTIPLY(tmp11 + tmp13, - FIX(1.163874945)); /* -c5 */
    tmp1 += tmp4 + tmp5 +
	    MULTIPLY(tmp11, FIX(0.837223564)) -         /* c5+c9+c11-c3 */
	    MULTIPLY(tmp14, FIX(2.341699410));          /* c1+c7 */
    tmp6 = MULTIPLY(tmp12 + tmp13, - FIX(0.657217813)); /* -c9 */
    tmp2 += tmp4 + tmp6 -
	    MULTIPLY(tmp12, FIX(1.572116027)) +         /* c1+c5-c9-c11 */
	    MULTIPLY(tmp15, FIX(2.260109708));          /* c3+c7 */
    tmp3 += tmp5 + tmp6 +
	    MULTIPLY(tmp13, FIX(2.205608352)) -         /* c3+c5+c9-c7 */
	    MULTIPLY(tmp15, FIX(1.742345811));          /* c1+c11 */

    dataptr[1] = (DCTELEM) DESCALE(tmp0, CONST_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp1, CONST_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp2, CONST_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp3, CONST_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 13)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/13)**2 = 64/169, which we partially
   * fold into the constant multipliers and final shifting:
   * cK now represents sqrt(2) * cos(K*pi/26) * 128/169.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*4];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*3];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*2];
    tmp3 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*1];
    tmp4 = dataptr[DCTSIZE*4] + wsptr[DCTSIZE*0];
    tmp5 = dataptr[DCTSIZE*5] + dataptr[DCTSIZE*7];
    tmp6 = dataptr[DCTSIZE*6];

    tmp10 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*4];
    tmp11 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*3];
    tmp12 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*2];
    tmp13 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*1];
    tmp14 = dataptr[DCTSIZE*4] - wsptr[DCTSIZE*0];
    tmp15 = dataptr[DCTSIZE*5] - dataptr[DCTSIZE*7];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6,
		       FIX(0.757396450)),          /* 128/169 */
	      CONST_BITS+1);
    tmp6 += tmp6;
    tmp0 -= tmp6;
    tmp1 -= tmp6;
    tmp2 -= tmp6;
    tmp3 -= tmp6;
    tmp4 -= tmp6;
    tmp5 -= tmp6;
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0, FIX(1.039995521)) +   /* c2 */
	      MULTIPLY(tmp1, FIX(0.801745081)) +   /* c6 */
	      MULTIPLY(tmp2, FIX(0.379824504)) -   /* c10 */
	      MULTIPLY(tmp3, FIX(0.129109289)) -   /* c12 */
	      MULTIPLY(tmp4, FIX(0.608465700)) -   /* c8 */
	      MULTIPLY(tmp5, FIX(0.948429952)),    /* c4 */
	      CONST_BITS+1);
    z1 = MULTIPLY(tmp0 - tmp2, FIX(0.875087516)) - /* (c4+c6)/2 */
	 MULTIPLY(tmp3 - tmp4, FIX(0.330085509)) - /* (c2-c10)/2 */
	 MULTIPLY(tmp1 - tmp5, FIX(0.239678205));  /* (c8-c12)/2 */
    z2 = MULTIPLY(tmp0 + tmp2, FIX(0.073342435)) - /* (c4-c6)/2 */
	 MULTIPLY(tmp3 + tmp4, FIX(0.709910013)) + /* (c2+c10)/2 */
	 MULTIPLY(tmp1 + tmp5, FIX(0.368787494));  /* (c8+c12)/2 */

    dataptr[DCTSIZE*4] = (DCTELEM) DESCALE(z1 + z2, CONST_BITS+1);
    dataptr[DCTSIZE*6] = (DCTELEM) DESCALE(z1 - z2, CONST_BITS+1);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(1.001514908));   /* c3 */
    tmp2 = MULTIPLY(tmp10 + tmp12, FIX(0.881514751));   /* c5 */
    tmp3 = MULTIPLY(tmp10 + tmp13, FIX(0.710284161)) +  /* c7 */
	   MULTIPLY(tmp14 + tmp15, FIX(0.256335874));   /* c11 */
    tmp0 = tmp1 + tmp2 + tmp3 -
	   MULTIPLY(tmp10, FIX(1.530003162)) +          /* c3+c5+c7-c1 */
	   MULTIPLY(tmp14, FIX(0.241438564));           /* c9-c11 */
    tmp4 = MULTIPLY(tmp14 - tmp15, FIX(0.710284161)) -  /* c7 */
	   MULTIPLY(tmp11 + tmp12, FIX(0.256335874));   /* c11 */
    tmp5 = MULTIPLY(tmp11 + tmp13, - FIX(0.881514751)); /* -c5 */
    tmp1 += tmp4 + tmp5 +
	    MULTIPLY(tmp11, FIX(0.634110155)) -         /* c5+c9+c11-c3 */
	    MULTIPLY(tmp14, FIX(1.773594819));          /* c1+c7 */
    tmp6 = MULTIPLY(tmp12 + tmp13, - FIX(0.497774438)); /* -c9 */
    tmp2 += tmp4 + tmp6 -
	    MULTIPLY(tmp12, FIX(1.190715098)) +         /* c1+c5-c9-c11 */
	    MULTIPLY(tmp15, FIX(1.711799069));          /* c3+c7 */
    tmp3 += tmp5 + tmp6 +
	    MULTIPLY(tmp13, FIX(1.670519935)) -         /* c3+c5+c9-c7 */
	    MULTIPLY(tmp15, FIX(1.319646532));          /* c1+c11 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp0, CONST_BITS+1);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp1, CONST_BITS+1);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp2, CONST_BITS+1);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp3, CONST_BITS+1);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 14x14 sample block.
 */

GLOBAL(void)
jpeg_fdct_14x14 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16;
  DCTELEM workspace[8*6];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT.
   * cK represents sqrt(2) * cos(K*pi/28).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[13]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[12]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[11]);
    tmp13 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[10]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[9]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[8]);
    tmp6 = GETJSAMPLE(elemptr[6]) + GETJSAMPLE(elemptr[7]);

    tmp10 = tmp0 + tmp6;
    tmp14 = tmp0 - tmp6;
    tmp11 = tmp1 + tmp5;
    tmp15 = tmp1 - tmp5;
    tmp12 = tmp2 + tmp4;
    tmp16 = tmp2 - tmp4;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[13]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[12]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[11]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[10]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[9]);
    tmp5 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[8]);
    tmp6 = GETJSAMPLE(elemptr[6]) - GETJSAMPLE(elemptr[7]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      (tmp10 + tmp11 + tmp12 + tmp13 - 14 * CENTERJSAMPLE);
    tmp13 += tmp13;
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.274162392)) + /* c4 */
	      MULTIPLY(tmp11 - tmp13, FIX(0.314692123)) - /* c12 */
	      MULTIPLY(tmp12 - tmp13, FIX(0.881747734)),  /* c8 */
	      CONST_BITS);

    tmp10 = MULTIPLY(tmp14 + tmp15, FIX(1.105676686));    /* c6 */

    dataptr[2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp14, FIX(0.273079590))   /* c2-c6 */
	      + MULTIPLY(tmp16, FIX(0.613604268)),        /* c10 */
	      CONST_BITS);
    dataptr[6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp15, FIX(1.719280954))   /* c6+c10 */
	      - MULTIPLY(tmp16, FIX(1.378756276)),        /* c2 */
	      CONST_BITS);

    /* Odd part */

    tmp10 = tmp1 + tmp2;
    tmp11 = tmp5 - tmp4;
    dataptr[7] = (DCTELEM) (tmp0 - tmp10 + tmp3 - tmp11 - tmp6);
    tmp3 <<= CONST_BITS;
    tmp10 = MULTIPLY(tmp10, - FIX(0.158341681));          /* -c13 */
    tmp11 = MULTIPLY(tmp11, FIX(1.405321284));            /* c1 */
    tmp10 += tmp11 - tmp3;
    tmp11 = MULTIPLY(tmp0 + tmp2, FIX(1.197448846)) +     /* c5 */
	    MULTIPLY(tmp4 + tmp6, FIX(0.752406978));      /* c9 */
    dataptr[5] = (DCTELEM)
      DESCALE(tmp10 + tmp11 - MULTIPLY(tmp2, FIX(2.373959773)) /* c3+c5-c13 */
	      + MULTIPLY(tmp4, FIX(1.119999435)),         /* c1+c11-c9 */
	      CONST_BITS);
    tmp12 = MULTIPLY(tmp0 + tmp1, FIX(1.334852607)) +     /* c3 */
	    MULTIPLY(tmp5 - tmp6, FIX(0.467085129));      /* c11 */
    dataptr[3] = (DCTELEM)
      DESCALE(tmp10 + tmp12 - MULTIPLY(tmp1, FIX(0.424103948)) /* c3-c9-c13 */
	      - MULTIPLY(tmp5, FIX(3.069855259)),         /* c1+c5+c11 */
	      CONST_BITS);
    dataptr[1] = (DCTELEM)
      DESCALE(tmp11 + tmp12 + tmp3 + tmp6 -
	      MULTIPLY(tmp0 + tmp6, FIX(1.126980169)),    /* c3+c5-c1 */
	      CONST_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 14)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/14)**2 = 16/49, which we partially
   * fold into the constant multipliers and final shifting:
   * cK now represents sqrt(2) * cos(K*pi/28) * 32/49.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*5];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*3];
    tmp13 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*2];
    tmp4 = dataptr[DCTSIZE*4] + wsptr[DCTSIZE*1];
    tmp5 = dataptr[DCTSIZE*5] + wsptr[DCTSIZE*0];
    tmp6 = dataptr[DCTSIZE*6] + dataptr[DCTSIZE*7];

    tmp10 = tmp0 + tmp6;
    tmp14 = tmp0 - tmp6;
    tmp11 = tmp1 + tmp5;
    tmp15 = tmp1 - tmp5;
    tmp12 = tmp2 + tmp4;
    tmp16 = tmp2 - tmp4;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*5];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*3];
    tmp3 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*2];
    tmp4 = dataptr[DCTSIZE*4] - wsptr[DCTSIZE*1];
    tmp5 = dataptr[DCTSIZE*5] - wsptr[DCTSIZE*0];
    tmp6 = dataptr[DCTSIZE*6] - dataptr[DCTSIZE*7];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11 + tmp12 + tmp13,
		       FIX(0.653061224)),                 /* 32/49 */
	      CONST_BITS+1);
    tmp13 += tmp13;
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(0.832106052)) + /* c4 */
	      MULTIPLY(tmp11 - tmp13, FIX(0.205513223)) - /* c12 */
	      MULTIPLY(tmp12 - tmp13, FIX(0.575835255)),  /* c8 */
	      CONST_BITS+1);

    tmp10 = MULTIPLY(tmp14 + tmp15, FIX(0.722074570));    /* c6 */

    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp14, FIX(0.178337691))   /* c2-c6 */
	      + MULTIPLY(tmp16, FIX(0.400721155)),        /* c10 */
	      CONST_BITS+1);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp15, FIX(1.122795725))   /* c6+c10 */
	      - MULTIPLY(tmp16, FIX(0.900412262)),        /* c2 */
	      CONST_BITS+1);

    /* Odd part */

    tmp10 = tmp1 + tmp2;
    tmp11 = tmp5 - tmp4;
    dataptr[DCTSIZE*7] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp10 + tmp3 - tmp11 - tmp6,
		       FIX(0.653061224)),                 /* 32/49 */
	      CONST_BITS+1);
    tmp3  = MULTIPLY(tmp3 , FIX(0.653061224));            /* 32/49 */
    tmp10 = MULTIPLY(tmp10, - FIX(0.103406812));          /* -c13 */
    tmp11 = MULTIPLY(tmp11, FIX(0.917760839));            /* c1 */
    tmp10 += tmp11 - tmp3;
    tmp11 = MULTIPLY(tmp0 + tmp2, FIX(0.782007410)) +     /* c5 */
	    MULTIPLY(tmp4 + tmp6, FIX(0.491367823));      /* c9 */
    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(tmp10 + tmp11 - MULTIPLY(tmp2, FIX(1.550341076)) /* c3+c5-c13 */
	      + MULTIPLY(tmp4, FIX(0.731428202)),         /* c1+c11-c9 */
	      CONST_BITS+1);
    tmp12 = MULTIPLY(tmp0 + tmp1, FIX(0.871740478)) +     /* c3 */
	    MULTIPLY(tmp5 - tmp6, FIX(0.305035186));      /* c11 */
    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(tmp10 + tmp12 - MULTIPLY(tmp1, FIX(0.276965844)) /* c3-c9-c13 */
	      - MULTIPLY(tmp5, FIX(2.004803435)),         /* c1+c5+c11 */
	      CONST_BITS+1);
    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp11 + tmp12 + tmp3
	      - MULTIPLY(tmp0, FIX(0.735987049))          /* c3+c5-c1 */
	      - MULTIPLY(tmp6, FIX(0.082925825)),         /* c9-c11-c13 */
	      CONST_BITS+1);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 15x15 sample block.
 */

GLOBAL(void)
jpeg_fdct_15x15 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16;
  INT32 z1, z2, z3;
  DCTELEM workspace[8*7];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT.
   * cK represents sqrt(2) * cos(K*pi/30).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[14]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[13]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[12]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[11]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[10]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[9]);
    tmp6 = GETJSAMPLE(elemptr[6]) + GETJSAMPLE(elemptr[8]);
    tmp7 = GETJSAMPLE(elemptr[7]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[14]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[13]);
    tmp12 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[12]);
    tmp13 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[11]);
    tmp14 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[10]);
    tmp15 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[9]);
    tmp16 = GETJSAMPLE(elemptr[6]) - GETJSAMPLE(elemptr[8]);

    z1 = tmp0 + tmp4 + tmp5;
    z2 = tmp1 + tmp3 + tmp6;
    z3 = tmp2 + tmp7;
    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM) (z1 + z2 + z3 - 15 * CENTERJSAMPLE);
    z3 += z3;
    dataptr[6] = (DCTELEM)
      DESCALE(MULTIPLY(z1 - z3, FIX(1.144122806)) - /* c6 */
	      MULTIPLY(z2 - z3, FIX(0.437016024)),  /* c12 */
	      CONST_BITS);
    tmp2 += ((tmp1 + tmp4) >> 1) - tmp7 - tmp7;
    z1 = MULTIPLY(tmp3 - tmp2, FIX(1.531135173)) -  /* c2+c14 */
         MULTIPLY(tmp6 - tmp2, FIX(2.238241955));   /* c4+c8 */
    z2 = MULTIPLY(tmp5 - tmp2, FIX(0.798468008)) -  /* c8-c14 */
	 MULTIPLY(tmp0 - tmp2, FIX(0.091361227));   /* c2-c4 */
    z3 = MULTIPLY(tmp0 - tmp3, FIX(1.383309603)) +  /* c2 */
	 MULTIPLY(tmp6 - tmp5, FIX(0.946293579)) +  /* c8 */
	 MULTIPLY(tmp1 - tmp4, FIX(0.790569415));   /* (c6+c12)/2 */

    dataptr[2] = (DCTELEM) DESCALE(z1 + z3, CONST_BITS);
    dataptr[4] = (DCTELEM) DESCALE(z2 + z3, CONST_BITS);

    /* Odd part */

    tmp2 = MULTIPLY(tmp10 - tmp12 - tmp13 + tmp15 + tmp16,
		    FIX(1.224744871));                         /* c5 */
    tmp1 = MULTIPLY(tmp10 - tmp14 - tmp15, FIX(1.344997024)) + /* c3 */
	   MULTIPLY(tmp11 - tmp13 - tmp16, FIX(0.831253876));  /* c9 */
    tmp12 = MULTIPLY(tmp12, FIX(1.224744871));                 /* c5 */
    tmp4 = MULTIPLY(tmp10 - tmp16, FIX(1.406466353)) +         /* c1 */
	   MULTIPLY(tmp11 + tmp14, FIX(1.344997024)) +         /* c3 */
	   MULTIPLY(tmp13 + tmp15, FIX(0.575212477));          /* c11 */
    tmp0 = MULTIPLY(tmp13, FIX(0.475753014)) -                 /* c7-c11 */
	   MULTIPLY(tmp14, FIX(0.513743148)) +                 /* c3-c9 */
	   MULTIPLY(tmp16, FIX(1.700497885)) + tmp4 + tmp12;   /* c1+c13 */
    tmp3 = MULTIPLY(tmp10, - FIX(0.355500862)) -               /* -(c1-c7) */
	   MULTIPLY(tmp11, FIX(2.176250899)) -                 /* c3+c9 */
	   MULTIPLY(tmp15, FIX(0.869244010)) + tmp4 - tmp12;   /* c11+c13 */

    dataptr[1] = (DCTELEM) DESCALE(tmp0, CONST_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp1, CONST_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp2, CONST_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp3, CONST_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 15)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/15)**2 = 64/225, which we partially
   * fold into the constant multipliers and final shifting:
   * cK now represents sqrt(2) * cos(K*pi/30) * 256/225.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*6];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*5];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*4];
    tmp3 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*3];
    tmp4 = dataptr[DCTSIZE*4] + wsptr[DCTSIZE*2];
    tmp5 = dataptr[DCTSIZE*5] + wsptr[DCTSIZE*1];
    tmp6 = dataptr[DCTSIZE*6] + wsptr[DCTSIZE*0];
    tmp7 = dataptr[DCTSIZE*7];

    tmp10 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*6];
    tmp11 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*5];
    tmp12 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*4];
    tmp13 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*3];
    tmp14 = dataptr[DCTSIZE*4] - wsptr[DCTSIZE*2];
    tmp15 = dataptr[DCTSIZE*5] - wsptr[DCTSIZE*1];
    tmp16 = dataptr[DCTSIZE*6] - wsptr[DCTSIZE*0];

    z1 = tmp0 + tmp4 + tmp5;
    z2 = tmp1 + tmp3 + tmp6;
    z3 = tmp2 + tmp7;
    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(z1 + z2 + z3, FIX(1.137777778)), /* 256/225 */
	      CONST_BITS+2);
    z3 += z3;
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(MULTIPLY(z1 - z3, FIX(1.301757503)) - /* c6 */
	      MULTIPLY(z2 - z3, FIX(0.497227121)),  /* c12 */
	      CONST_BITS+2);
    tmp2 += ((tmp1 + tmp4) >> 1) - tmp7 - tmp7;
    z1 = MULTIPLY(tmp3 - tmp2, FIX(1.742091575)) -  /* c2+c14 */
         MULTIPLY(tmp6 - tmp2, FIX(2.546621957));   /* c4+c8 */
    z2 = MULTIPLY(tmp5 - tmp2, FIX(0.908479156)) -  /* c8-c14 */
	 MULTIPLY(tmp0 - tmp2, FIX(0.103948774));   /* c2-c4 */
    z3 = MULTIPLY(tmp0 - tmp3, FIX(1.573898926)) +  /* c2 */
	 MULTIPLY(tmp6 - tmp5, FIX(1.076671805)) +  /* c8 */
	 MULTIPLY(tmp1 - tmp4, FIX(0.899492312));   /* (c6+c12)/2 */

    dataptr[DCTSIZE*2] = (DCTELEM) DESCALE(z1 + z3, CONST_BITS+2);
    dataptr[DCTSIZE*4] = (DCTELEM) DESCALE(z2 + z3, CONST_BITS+2);

    /* Odd part */

    tmp2 = MULTIPLY(tmp10 - tmp12 - tmp13 + tmp15 + tmp16,
		    FIX(1.393487498));                         /* c5 */
    tmp1 = MULTIPLY(tmp10 - tmp14 - tmp15, FIX(1.530307725)) + /* c3 */
	   MULTIPLY(tmp11 - tmp13 - tmp16, FIX(0.945782187));  /* c9 */
    tmp12 = MULTIPLY(tmp12, FIX(1.393487498));                 /* c5 */
    tmp4 = MULTIPLY(tmp10 - tmp16, FIX(1.600246161)) +         /* c1 */
	   MULTIPLY(tmp11 + tmp14, FIX(1.530307725)) +         /* c3 */
	   MULTIPLY(tmp13 + tmp15, FIX(0.654463974));          /* c11 */
    tmp0 = MULTIPLY(tmp13, FIX(0.541301207)) -                 /* c7-c11 */
	   MULTIPLY(tmp14, FIX(0.584525538)) +                 /* c3-c9 */
	   MULTIPLY(tmp16, FIX(1.934788705)) + tmp4 + tmp12;   /* c1+c13 */
    tmp3 = MULTIPLY(tmp10, - FIX(0.404480980)) -               /* -(c1-c7) */
	   MULTIPLY(tmp11, FIX(2.476089912)) -                 /* c3+c9 */
	   MULTIPLY(tmp15, FIX(0.989006518)) + tmp4 - tmp12;   /* c11+c13 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp0, CONST_BITS+2);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp1, CONST_BITS+2);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp2, CONST_BITS+2);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp3, CONST_BITS+2);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 16x16 sample block.
 */

GLOBAL(void)
jpeg_fdct_16x16 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
  DCTELEM workspace[DCTSIZE2];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * cK represents sqrt(2) * cos(K*pi/32).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[15]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[14]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[13]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[12]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[11]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[10]);
    tmp6 = GETJSAMPLE(elemptr[6]) + GETJSAMPLE(elemptr[9]);
    tmp7 = GETJSAMPLE(elemptr[7]) + GETJSAMPLE(elemptr[8]);

    tmp10 = tmp0 + tmp7;
    tmp14 = tmp0 - tmp7;
    tmp11 = tmp1 + tmp6;
    tmp15 = tmp1 - tmp6;
    tmp12 = tmp2 + tmp5;
    tmp16 = tmp2 - tmp5;
    tmp13 = tmp3 + tmp4;
    tmp17 = tmp3 - tmp4;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[15]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[14]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[13]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[12]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[11]);
    tmp5 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[10]);
    tmp6 = GETJSAMPLE(elemptr[6]) - GETJSAMPLE(elemptr[9]);
    tmp7 = GETJSAMPLE(elemptr[7]) - GETJSAMPLE(elemptr[8]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 + tmp12 + tmp13 - 16 * CENTERJSAMPLE) << PASS1_BITS);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
	      MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
	      CONST_BITS-PASS1_BITS);

    tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
	    MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

    dataptr[2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
	      + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+c10 */
	      CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
	      - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
	    MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
	    MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
	    MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
    tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
	    MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
    tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
	    MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
    tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
	    MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
    tmp10 = tmp11 + tmp12 + tmp13 -
	    MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
	    MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
    tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
	     - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
    tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
	     + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
    tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
	     + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

    dataptr[1] = (DCTELEM) DESCALE(tmp10, CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp11, CONST_BITS-PASS1_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp12, CONST_BITS-PASS1_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp13, CONST_BITS-PASS1_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == DCTSIZE * 2)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/16)**2 = 1/2**2.
   * cK represents sqrt(2) * cos(K*pi/32).
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*4];
    tmp4 = dataptr[DCTSIZE*4] + wsptr[DCTSIZE*3];
    tmp5 = dataptr[DCTSIZE*5] + wsptr[DCTSIZE*2];
    tmp6 = dataptr[DCTSIZE*6] + wsptr[DCTSIZE*1];
    tmp7 = dataptr[DCTSIZE*7] + wsptr[DCTSIZE*0];

    tmp10 = tmp0 + tmp7;
    tmp14 = tmp0 - tmp7;
    tmp11 = tmp1 + tmp6;
    tmp15 = tmp1 - tmp6;
    tmp12 = tmp2 + tmp5;
    tmp16 = tmp2 - tmp5;
    tmp13 = tmp3 + tmp4;
    tmp17 = tmp3 - tmp4;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*4];
    tmp4 = dataptr[DCTSIZE*4] - wsptr[DCTSIZE*3];
    tmp5 = dataptr[DCTSIZE*5] - wsptr[DCTSIZE*2];
    tmp6 = dataptr[DCTSIZE*6] - wsptr[DCTSIZE*1];
    tmp7 = dataptr[DCTSIZE*7] - wsptr[DCTSIZE*0];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(tmp10 + tmp11 + tmp12 + tmp13, PASS1_BITS+2);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
	      MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
	      CONST_BITS+PASS1_BITS+2);

    tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
	    MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
	      + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+10 */
	      CONST_BITS+PASS1_BITS+2);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
	      - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
	      CONST_BITS+PASS1_BITS+2);

    /* Odd part */

    tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
	    MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
	    MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
	    MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
    tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
	    MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
    tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
	    MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
    tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
	    MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
    tmp10 = tmp11 + tmp12 + tmp13 -
	    MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
	    MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
    tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
	     - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
    tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
	     + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
    tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
	     + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp10, CONST_BITS+PASS1_BITS+2);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp11, CONST_BITS+PASS1_BITS+2);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp12, CONST_BITS+PASS1_BITS+2);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp13, CONST_BITS+PASS1_BITS+2);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 16x8 sample block.
 *
 * 16-point FDCT in pass 1 (rows), 8-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_16x8 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
  INT32 z1;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 16-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/32).
   */

  dataptr = data;
  ctr = 0;
  for (ctr = 0; ctr < DCTSIZE; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[15]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[14]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[13]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[12]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[11]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[10]);
    tmp6 = GETJSAMPLE(elemptr[6]) + GETJSAMPLE(elemptr[9]);
    tmp7 = GETJSAMPLE(elemptr[7]) + GETJSAMPLE(elemptr[8]);

    tmp10 = tmp0 + tmp7;
    tmp14 = tmp0 - tmp7;
    tmp11 = tmp1 + tmp6;
    tmp15 = tmp1 - tmp6;
    tmp12 = tmp2 + tmp5;
    tmp16 = tmp2 - tmp5;
    tmp13 = tmp3 + tmp4;
    tmp17 = tmp3 - tmp4;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[15]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[14]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[13]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[12]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[11]);
    tmp5 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[10]);
    tmp6 = GETJSAMPLE(elemptr[6]) - GETJSAMPLE(elemptr[9]);
    tmp7 = GETJSAMPLE(elemptr[7]) - GETJSAMPLE(elemptr[8]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 + tmp12 + tmp13 - 16 * CENTERJSAMPLE) << PASS1_BITS);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
	      MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
	      CONST_BITS-PASS1_BITS);

    tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
	    MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

    dataptr[2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
	      + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+c10 */
	      CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
	      - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
	    MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
	    MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
	    MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
    tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
	    MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
    tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
	    MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
    tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
	    MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
    tmp10 = tmp11 + tmp12 + tmp13 -
	    MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
	    MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
    tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
	     - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
    tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
	     + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
    tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
	     + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

    dataptr[1] = (DCTELEM) DESCALE(tmp10, CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp11, CONST_BITS-PASS1_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp12, CONST_BITS-PASS1_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp13, CONST_BITS-PASS1_BITS);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by 8/16 = 1/2.
   * 8-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/16).
   */

  dataptr = data;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part per LL&M figure 1 --- note that published figure is faulty;
     * rotator "c1" should be "c6".
     */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*4];

    tmp10 = tmp0 + tmp3;
    tmp12 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp13 = tmp1 - tmp2;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*4];

    dataptr[DCTSIZE*0] = (DCTELEM) DESCALE(tmp10 + tmp11, PASS1_BITS+1);
    dataptr[DCTSIZE*4] = (DCTELEM) DESCALE(tmp10 - tmp11, PASS1_BITS+1);

    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);   /* c6 */
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(z1 + MULTIPLY(tmp12, FIX_0_765366865), /* c2-c6 */
	      CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(z1 - MULTIPLY(tmp13, FIX_1_847759065), /* c2+c6 */
	      CONST_BITS+PASS1_BITS+1);

    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
     * i0..i3 in the paper are tmp0..tmp3 here.
     */

    tmp12 = tmp0 + tmp2;
    tmp13 = tmp1 + tmp3;

    z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602);   /*  c3 */
    tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);      /* -c3+c5 */
    tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);      /* -c3-c5 */
    tmp12 += z1;
    tmp13 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223);   /* -c3+c7 */
    tmp0 = MULTIPLY(tmp0, FIX_1_501321110);          /*  c1+c3-c5-c7 */
    tmp3 = MULTIPLY(tmp3, FIX_0_298631336);          /* -c1+c3+c5-c7 */
    tmp0 += z1 + tmp12;
    tmp3 += z1 + tmp13;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447);   /* -c1-c3 */
    tmp1 = MULTIPLY(tmp1, FIX_3_072711026);          /*  c1+c3+c5-c7 */
    tmp2 = MULTIPLY(tmp2, FIX_2_053119869);          /*  c1+c3-c5+c7 */
    tmp1 += z1 + tmp13;
    tmp2 += z1 + tmp12;

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp0, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp1, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp2, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp3, CONST_BITS+PASS1_BITS+1);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 14x7 sample block.
 *
 * 14-point FDCT in pass 1 (rows), 7-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_14x7 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16;
  INT32 z1, z2, z3;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Zero bottom row of output coefficient block. */
  MEMZERO(&data[DCTSIZE*7], SIZEOF(DCTELEM) * DCTSIZE);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 14-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/28).
   */

  dataptr = data;
  for (ctr = 0; ctr < 7; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[13]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[12]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[11]);
    tmp13 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[10]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[9]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[8]);
    tmp6 = GETJSAMPLE(elemptr[6]) + GETJSAMPLE(elemptr[7]);

    tmp10 = tmp0 + tmp6;
    tmp14 = tmp0 - tmp6;
    tmp11 = tmp1 + tmp5;
    tmp15 = tmp1 - tmp5;
    tmp12 = tmp2 + tmp4;
    tmp16 = tmp2 - tmp4;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[13]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[12]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[11]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[10]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[9]);
    tmp5 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[8]);
    tmp6 = GETJSAMPLE(elemptr[6]) - GETJSAMPLE(elemptr[7]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 + tmp12 + tmp13 - 14 * CENTERJSAMPLE) << PASS1_BITS);
    tmp13 += tmp13;
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.274162392)) + /* c4 */
	      MULTIPLY(tmp11 - tmp13, FIX(0.314692123)) - /* c12 */
	      MULTIPLY(tmp12 - tmp13, FIX(0.881747734)),  /* c8 */
	      CONST_BITS-PASS1_BITS);

    tmp10 = MULTIPLY(tmp14 + tmp15, FIX(1.105676686));    /* c6 */

    dataptr[2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp14, FIX(0.273079590))   /* c2-c6 */
	      + MULTIPLY(tmp16, FIX(0.613604268)),        /* c10 */
	      CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp15, FIX(1.719280954))   /* c6+c10 */
	      - MULTIPLY(tmp16, FIX(1.378756276)),        /* c2 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp10 = tmp1 + tmp2;
    tmp11 = tmp5 - tmp4;
    dataptr[7] = (DCTELEM) ((tmp0 - tmp10 + tmp3 - tmp11 - tmp6) << PASS1_BITS);
    tmp3 <<= CONST_BITS;
    tmp10 = MULTIPLY(tmp10, - FIX(0.158341681));          /* -c13 */
    tmp11 = MULTIPLY(tmp11, FIX(1.405321284));            /* c1 */
    tmp10 += tmp11 - tmp3;
    tmp11 = MULTIPLY(tmp0 + tmp2, FIX(1.197448846)) +     /* c5 */
	    MULTIPLY(tmp4 + tmp6, FIX(0.752406978));      /* c9 */
    dataptr[5] = (DCTELEM)
      DESCALE(tmp10 + tmp11 - MULTIPLY(tmp2, FIX(2.373959773)) /* c3+c5-c13 */
	      + MULTIPLY(tmp4, FIX(1.119999435)),         /* c1+c11-c9 */
	      CONST_BITS-PASS1_BITS);
    tmp12 = MULTIPLY(tmp0 + tmp1, FIX(1.334852607)) +     /* c3 */
	    MULTIPLY(tmp5 - tmp6, FIX(0.467085129));      /* c11 */
    dataptr[3] = (DCTELEM)
      DESCALE(tmp10 + tmp12 - MULTIPLY(tmp1, FIX(0.424103948)) /* c3-c9-c13 */
	      - MULTIPLY(tmp5, FIX(3.069855259)),         /* c1+c5+c11 */
	      CONST_BITS-PASS1_BITS);
    dataptr[1] = (DCTELEM)
      DESCALE(tmp11 + tmp12 + tmp3 + tmp6 -
	      MULTIPLY(tmp0 + tmp6, FIX(1.126980169)),    /* c3+c5-c1 */
	      CONST_BITS-PASS1_BITS);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/14)*(8/7) = 32/49, which we
   * partially fold into the constant multipliers and final shifting:
   * 7-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/14) * 64/49.
   */

  dataptr = data;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*6];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*5];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*4];
    tmp3 = dataptr[DCTSIZE*3];

    tmp10 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*6];
    tmp11 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*5];
    tmp12 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*4];

    z1 = tmp0 + tmp2;
    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(z1 + tmp1 + tmp3, FIX(1.306122449)), /* 64/49 */
	      CONST_BITS+PASS1_BITS+1);
    tmp3 += tmp3;
    z1 -= tmp3;
    z1 -= tmp3;
    z1 = MULTIPLY(z1, FIX(0.461784020));                /* (c2+c6-c4)/2 */
    z2 = MULTIPLY(tmp0 - tmp2, FIX(1.202428084));       /* (c2+c4-c6)/2 */
    z3 = MULTIPLY(tmp1 - tmp2, FIX(0.411026446));       /* c6 */
    dataptr[DCTSIZE*2] = (DCTELEM) DESCALE(z1 + z2 + z3, CONST_BITS+PASS1_BITS+1);
    z1 -= z2;
    z2 = MULTIPLY(tmp0 - tmp1, FIX(1.151670509));       /* c4 */
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(z2 + z3 - MULTIPLY(tmp1 - tmp3, FIX(0.923568041)), /* c2+c6-c4 */
	      CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*6] = (DCTELEM) DESCALE(z1 + z2, CONST_BITS+PASS1_BITS+1);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(1.221765677));   /* (c3+c1-c5)/2 */
    tmp2 = MULTIPLY(tmp10 - tmp11, FIX(0.222383464));   /* (c3+c5-c1)/2 */
    tmp0 = tmp1 - tmp2;
    tmp1 += tmp2;
    tmp2 = MULTIPLY(tmp11 + tmp12, - FIX(1.800824523)); /* -c1 */
    tmp1 += tmp2;
    tmp3 = MULTIPLY(tmp10 + tmp12, FIX(0.801442310));   /* c5 */
    tmp0 += tmp3;
    tmp2 += tmp3 + MULTIPLY(tmp12, FIX(2.443531355));   /* c3+c1-c5 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp0, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp1, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp2, CONST_BITS+PASS1_BITS+1);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 12x6 sample block.
 *
 * 12-point FDCT in pass 1 (rows), 6-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_12x6 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Zero 2 bottom rows of output coefficient block. */
  MEMZERO(&data[DCTSIZE*6], SIZEOF(DCTELEM) * DCTSIZE * 2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 12-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/24).
   */

  dataptr = data;
  for (ctr = 0; ctr < 6; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[11]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[10]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[9]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[8]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[7]);
    tmp5 = GETJSAMPLE(elemptr[5]) + GETJSAMPLE(elemptr[6]);

    tmp10 = tmp0 + tmp5;
    tmp13 = tmp0 - tmp5;
    tmp11 = tmp1 + tmp4;
    tmp14 = tmp1 - tmp4;
    tmp12 = tmp2 + tmp3;
    tmp15 = tmp2 - tmp3;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[11]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[10]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[9]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[8]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[7]);
    tmp5 = GETJSAMPLE(elemptr[5]) - GETJSAMPLE(elemptr[6]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 + tmp12 - 12 * CENTERJSAMPLE) << PASS1_BITS);
    dataptr[6] = (DCTELEM) ((tmp13 - tmp14 - tmp15) << PASS1_BITS);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.224744871)), /* c4 */
	      CONST_BITS-PASS1_BITS);
    dataptr[2] = (DCTELEM)
      DESCALE(tmp14 - tmp15 + MULTIPLY(tmp13 + tmp15, FIX(1.366025404)), /* c2 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp1 + tmp4, FIX_0_541196100);    /* c9 */
    tmp14 = tmp10 + MULTIPLY(tmp1, FIX_0_765366865);   /* c3-c9 */
    tmp15 = tmp10 - MULTIPLY(tmp4, FIX_1_847759065);   /* c3+c9 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.121971054));   /* c5 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(0.860918669));   /* c7 */
    tmp10 = tmp12 + tmp13 + tmp14 - MULTIPLY(tmp0, FIX(0.580774953)) /* c5+c7-c1 */
	    + MULTIPLY(tmp5, FIX(0.184591911));        /* c11 */
    tmp11 = MULTIPLY(tmp2 + tmp3, - FIX(0.184591911)); /* -c11 */
    tmp12 += tmp11 - tmp15 - MULTIPLY(tmp2, FIX(2.339493912)) /* c1+c5-c11 */
	    + MULTIPLY(tmp5, FIX(0.860918669));        /* c7 */
    tmp13 += tmp11 - tmp14 + MULTIPLY(tmp3, FIX(0.725788011)) /* c1+c11-c7 */
	    - MULTIPLY(tmp5, FIX(1.121971054));        /* c5 */
    tmp11 = tmp15 + MULTIPLY(tmp0 - tmp3, FIX(1.306562965)) /* c3 */
	    - MULTIPLY(tmp2 + tmp5, FIX_0_541196100);  /* c9 */

    dataptr[1] = (DCTELEM) DESCALE(tmp10, CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp11, CONST_BITS-PASS1_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp12, CONST_BITS-PASS1_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp13, CONST_BITS-PASS1_BITS);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/12)*(8/6) = 8/9, which we
   * partially fold into the constant multipliers and final shifting:
   * 6-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/12) * 16/9.
   */

  dataptr = data;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*5];
    tmp11 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*3];

    tmp10 = tmp0 + tmp2;
    tmp12 = tmp0 - tmp2;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*5];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*3];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11, FIX(1.777777778)),         /* 16/9 */
	      CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp12, FIX(2.177324216)),                 /* c2 */
	      CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp11, FIX(1.257078722)), /* c4 */
	      CONST_BITS+PASS1_BITS+1);

    /* Odd part */

    tmp10 = MULTIPLY(tmp0 + tmp2, FIX(0.650711829));             /* c5 */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp0 + tmp1, FIX(1.777777778)),   /* 16/9 */
	      CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp1 - tmp2, FIX(1.777777778)),    /* 16/9 */
	      CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp2 - tmp1, FIX(1.777777778)),   /* 16/9 */
	      CONST_BITS+PASS1_BITS+1);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 10x5 sample block.
 *
 * 10-point FDCT in pass 1 (rows), 5-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_10x5 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Zero 3 bottom rows of output coefficient block. */
  MEMZERO(&data[DCTSIZE*5], SIZEOF(DCTELEM) * DCTSIZE * 3);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 10-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/20).
   */

  dataptr = data;
  for (ctr = 0; ctr < 5; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[9]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[8]);
    tmp12 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[7]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[6]);
    tmp4 = GETJSAMPLE(elemptr[4]) + GETJSAMPLE(elemptr[5]);

    tmp10 = tmp0 + tmp4;
    tmp13 = tmp0 - tmp4;
    tmp11 = tmp1 + tmp3;
    tmp14 = tmp1 - tmp3;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[9]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[8]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[7]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[6]);
    tmp4 = GETJSAMPLE(elemptr[4]) - GETJSAMPLE(elemptr[5]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 + tmp12 - 10 * CENTERJSAMPLE) << PASS1_BITS);
    tmp12 += tmp12;
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.144122806)) - /* c4 */
	      MULTIPLY(tmp11 - tmp12, FIX(0.437016024)),  /* c8 */
	      CONST_BITS-PASS1_BITS);
    tmp10 = MULTIPLY(tmp13 + tmp14, FIX(0.831253876));    /* c6 */
    dataptr[2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp13, FIX(0.513743148)),  /* c2-c6 */
	      CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(2.176250899)),  /* c2+c6 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp10 = tmp0 + tmp4;
    tmp11 = tmp1 - tmp3;
    dataptr[5] = (DCTELEM) ((tmp10 - tmp11 - tmp2) << PASS1_BITS);
    tmp2 <<= CONST_BITS;
    dataptr[1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0, FIX(1.396802247)) +          /* c1 */
	      MULTIPLY(tmp1, FIX(1.260073511)) + tmp2 +   /* c3 */
	      MULTIPLY(tmp3, FIX(0.642039522)) +          /* c7 */
	      MULTIPLY(tmp4, FIX(0.221231742)),           /* c9 */
	      CONST_BITS-PASS1_BITS);
    tmp12 = MULTIPLY(tmp0 - tmp4, FIX(0.951056516)) -     /* (c3+c7)/2 */
	    MULTIPLY(tmp1 + tmp3, FIX(0.587785252));      /* (c1-c9)/2 */
    tmp13 = MULTIPLY(tmp10 + tmp11, FIX(0.309016994)) +   /* (c3-c7)/2 */
	    (tmp11 << (CONST_BITS - 1)) - tmp2;
    dataptr[3] = (DCTELEM) DESCALE(tmp12 + tmp13, CONST_BITS-PASS1_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp12 - tmp13, CONST_BITS-PASS1_BITS);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/10)*(8/5) = 32/25, which we
   * fold into the constant multipliers:
   * 5-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/10) * 32/25.
   */

  dataptr = data;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*4];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*3];
    tmp2 = dataptr[DCTSIZE*2];

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*4];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*3];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp2, FIX(1.28)),        /* 32/25 */
	      CONST_BITS+PASS1_BITS);
    tmp11 = MULTIPLY(tmp11, FIX(1.011928851));          /* (c2+c4)/2 */
    tmp10 -= tmp2 << 2;
    tmp10 = MULTIPLY(tmp10, FIX(0.452548340));          /* (c2-c4)/2 */
    dataptr[DCTSIZE*2] = (DCTELEM) DESCALE(tmp11 + tmp10, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*4] = (DCTELEM) DESCALE(tmp11 - tmp10, CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp0 + tmp1, FIX(1.064004961));    /* c3 */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp0, FIX(0.657591230)), /* c1-c3 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp1, FIX(2.785601151)), /* c1+c3 */
	      CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on an 8x4 sample block.
 *
 * 8-point FDCT in pass 1 (rows), 4-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_8x4 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3;
  INT32 tmp10, tmp11, tmp12, tmp13;
  INT32 z1;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Zero 4 bottom rows of output coefficient block. */
  MEMZERO(&data[DCTSIZE*4], SIZEOF(DCTELEM) * DCTSIZE * 4);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * We must also scale the output by 8/4 = 2, which we add here.
   * 8-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/16).
   */

  dataptr = data;
  for (ctr = 0; ctr < 4; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part per LL&M figure 1 --- note that published figure is faulty;
     * rotator "c1" should be "c6".
     */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[7]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[6]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[5]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[4]);

    tmp10 = tmp0 + tmp3;
    tmp12 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp13 = tmp1 - tmp2;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[7]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[6]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[5]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[4]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 - 8 * CENTERJSAMPLE) << (PASS1_BITS+1));
    dataptr[4] = (DCTELEM) ((tmp10 - tmp11) << (PASS1_BITS+1));

    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS-PASS1_BITS-2);

    dataptr[2] = (DCTELEM)
      RIGHT_SHIFT(z1 + MULTIPLY(tmp12, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS-PASS1_BITS-1);
    dataptr[6] = (DCTELEM)
      RIGHT_SHIFT(z1 - MULTIPLY(tmp13, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS-PASS1_BITS-1);

    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
     * i0..i3 in the paper are tmp0..tmp3 here.
     */

    tmp12 = tmp0 + tmp2;
    tmp13 = tmp1 + tmp3;

    z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602);       /*  c3 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS-PASS1_BITS-2);

    tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);          /* -c3+c5 */
    tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);          /* -c3-c5 */
    tmp12 += z1;
    tmp13 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223);       /* -c3+c7 */
    tmp0 = MULTIPLY(tmp0, FIX_1_501321110);              /*  c1+c3-c5-c7 */
    tmp3 = MULTIPLY(tmp3, FIX_0_298631336);              /* -c1+c3+c5-c7 */
    tmp0 += z1 + tmp12;
    tmp3 += z1 + tmp13;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447);       /* -c1-c3 */
    tmp1 = MULTIPLY(tmp1, FIX_3_072711026);              /*  c1+c3+c5-c7 */
    tmp2 = MULTIPLY(tmp2, FIX_2_053119869);              /*  c1+c3-c5+c7 */
    tmp1 += z1 + tmp13;
    tmp2 += z1 + tmp12;

    dataptr[1] = (DCTELEM) RIGHT_SHIFT(tmp0, CONST_BITS-PASS1_BITS-1);
    dataptr[3] = (DCTELEM) RIGHT_SHIFT(tmp1, CONST_BITS-PASS1_BITS-1);
    dataptr[5] = (DCTELEM) RIGHT_SHIFT(tmp2, CONST_BITS-PASS1_BITS-1);
    dataptr[7] = (DCTELEM) RIGHT_SHIFT(tmp3, CONST_BITS-PASS1_BITS-1);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * 4-point FDCT kernel,
   * cK represents sqrt(2) * cos(K*pi/16) [refers to 8-point FDCT].
   */

  dataptr = data;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    /* Add fudge factor here for final descale. */
    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*3] + (ONE << (PASS1_BITS-1));
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*2];

    tmp10 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*3];
    tmp11 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*2];

    dataptr[DCTSIZE*0] = (DCTELEM) RIGHT_SHIFT(tmp0 + tmp1, PASS1_BITS);
    dataptr[DCTSIZE*2] = (DCTELEM) RIGHT_SHIFT(tmp0 - tmp1, PASS1_BITS);

    /* Odd part */

    tmp0 = MULTIPLY(tmp10 + tmp11, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    tmp0 += ONE << (CONST_BITS+PASS1_BITS-1);

    dataptr[DCTSIZE*1] = (DCTELEM)
      RIGHT_SHIFT(tmp0 + MULTIPLY(tmp10, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM)
      RIGHT_SHIFT(tmp0 - MULTIPLY(tmp11, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 6x3 sample block.
 *
 * 6-point FDCT in pass 1 (rows), 3-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_6x3 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2;
  INT32 tmp10, tmp11, tmp12;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * We scale the results further by 2 as part of output adaption
   * scaling for different DCT size.
   * 6-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/12).
   */

  dataptr = data;
  for (ctr = 0; ctr < 3; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[5]);
    tmp11 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[4]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[3]);

    tmp10 = tmp0 + tmp2;
    tmp12 = tmp0 - tmp2;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[5]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[4]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[3]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 - 6 * CENTERJSAMPLE) << (PASS1_BITS+1));
    dataptr[2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp12, FIX(1.224744871)),                 /* c2 */
	      CONST_BITS-PASS1_BITS-1);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp11, FIX(0.707106781)), /* c4 */
	      CONST_BITS-PASS1_BITS-1);

    /* Odd part */

    tmp10 = DESCALE(MULTIPLY(tmp0 + tmp2, FIX(0.366025404)),     /* c5 */
		    CONST_BITS-PASS1_BITS-1);

    dataptr[1] = (DCTELEM) (tmp10 + ((tmp0 + tmp1) << (PASS1_BITS+1)));
    dataptr[3] = (DCTELEM) ((tmp0 - tmp1 - tmp2) << (PASS1_BITS+1));
    dataptr[5] = (DCTELEM) (tmp10 + ((tmp2 - tmp1) << (PASS1_BITS+1)));

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/6)*(8/3) = 32/9, which we partially
   * fold into the constant multipliers (other part was done in pass 1):
   * 3-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/6) * 16/9.
   */

  dataptr = data;
  for (ctr = 0; ctr < 6; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*2];
    tmp1 = dataptr[DCTSIZE*1];

    tmp2 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*2];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 + tmp1, FIX(1.777777778)),        /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp1 - tmp1, FIX(1.257078722)), /* c2 */
	      CONST_BITS+PASS1_BITS);

    /* Odd part */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp2, FIX(2.177324216)),               /* c1 */
	      CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 4x2 sample block.
 *
 * 4-point FDCT in pass 1 (rows), 2-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_4x2 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  DCTELEM tmp0, tmp2, tmp10, tmp12, tmp4, tmp5;
  INT32 tmp1, tmp3, tmp11, tmp13;
  INT32 z1, z2, z3;
  JSAMPROW elemptr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT.
   * 4-point FDCT kernel,
   * cK represents sqrt(2) * cos(K*pi/16) [refers to 8-point FDCT].
   */

  /* Row 0 */
  elemptr = sample_data[0] + start_col;

  /* Even part */

  tmp4 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[3]);
  tmp5 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[2]);

  tmp0 = tmp4 + tmp5;
  tmp2 = tmp4 - tmp5;

  /* Odd part */

  z2 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[3]);
  z3 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[2]);

  z1 = MULTIPLY(z2 + z3, FIX_0_541196100);    /* c6 */
  /* Add fudge factor here for final descale. */
  z1 += ONE << (CONST_BITS-3-1);
  tmp1 = z1 + MULTIPLY(z2, FIX_0_765366865); /* c2-c6 */
  tmp3 = z1 - MULTIPLY(z3, FIX_1_847759065); /* c2+c6 */

  /* Row 1 */
  elemptr = sample_data[1] + start_col;

  /* Even part */

  tmp4 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[3]);
  tmp5 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[2]);

  tmp10 = tmp4 + tmp5;
  tmp12 = tmp4 - tmp5;

  /* Odd part */

  z2 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[3]);
  z3 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[2]);

  z1 = MULTIPLY(z2 + z3, FIX_0_541196100);    /* c6 */
  tmp11 = z1 + MULTIPLY(z2, FIX_0_765366865); /* c2-c6 */
  tmp13 = z1 - MULTIPLY(z3, FIX_1_847759065); /* c2+c6 */

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/4)*(8/2) = 2**3.
   */

  /* Column 0 */
  /* Apply unsigned->signed conversion. */
  data[DCTSIZE*0] = (tmp0 + tmp10 - 8 * CENTERJSAMPLE) << 3;
  data[DCTSIZE*1] = (tmp0 - tmp10) << 3;

  /* Column 1 */
  data[DCTSIZE*0+1] = (DCTELEM) RIGHT_SHIFT(tmp1 + tmp11, CONST_BITS-3);
  data[DCTSIZE*1+1] = (DCTELEM) RIGHT_SHIFT(tmp1 - tmp11, CONST_BITS-3);

  /* Column 2 */
  data[DCTSIZE*0+2] = (tmp2 + tmp12) << 3;
  data[DCTSIZE*1+2] = (tmp2 - tmp12) << 3;

  /* Column 3 */
  data[DCTSIZE*0+3] = (DCTELEM) RIGHT_SHIFT(tmp3 + tmp13, CONST_BITS-3);
  data[DCTSIZE*1+3] = (DCTELEM) RIGHT_SHIFT(tmp3 - tmp13, CONST_BITS-3);
}


/*
 * Perform the forward DCT on a 2x1 sample block.
 *
 * 2-point FDCT in pass 1 (rows), 1-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_2x1 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  DCTELEM tmp0, tmp1;
  JSAMPROW elemptr;

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  elemptr = sample_data[0] + start_col;

  tmp0 = GETJSAMPLE(elemptr[0]);
  tmp1 = GETJSAMPLE(elemptr[1]);

  /* We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/2)*(8/1) = 2**5.
   */

  /* Even part */

  /* Apply unsigned->signed conversion. */
  data[0] = (tmp0 + tmp1 - 2 * CENTERJSAMPLE) << 5;

  /* Odd part */

  data[1] = (tmp0 - tmp1) << 5;
}


/*
 * Perform the forward DCT on an 8x16 sample block.
 *
 * 8-point FDCT in pass 1 (rows), 16-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_8x16 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
  INT32 z1;
  DCTELEM workspace[DCTSIZE2];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 8-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/16).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part per LL&M figure 1 --- note that published figure is faulty;
     * rotator "c1" should be "c6".
     */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[7]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[6]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[5]);
    tmp3 = GETJSAMPLE(elemptr[3]) + GETJSAMPLE(elemptr[4]);

    tmp10 = tmp0 + tmp3;
    tmp12 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp13 = tmp1 - tmp2;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[7]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[6]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[5]);
    tmp3 = GETJSAMPLE(elemptr[3]) - GETJSAMPLE(elemptr[4]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM) ((tmp10 + tmp11 - 8 * CENTERJSAMPLE) << PASS1_BITS);
    dataptr[4] = (DCTELEM) ((tmp10 - tmp11) << PASS1_BITS);

    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);   /* c6 */
    dataptr[2] = (DCTELEM)
      DESCALE(z1 + MULTIPLY(tmp12, FIX_0_765366865), /* c2-c6 */
	      CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM)
      DESCALE(z1 - MULTIPLY(tmp13, FIX_1_847759065), /* c2+c6 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
     * i0..i3 in the paper are tmp0..tmp3 here.
     */

    tmp12 = tmp0 + tmp2;
    tmp13 = tmp1 + tmp3;

    z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602);   /*  c3 */
    tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);      /* -c3+c5 */
    tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);      /* -c3-c5 */
    tmp12 += z1;
    tmp13 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223);   /* -c3+c7 */
    tmp0 = MULTIPLY(tmp0, FIX_1_501321110);          /*  c1+c3-c5-c7 */
    tmp3 = MULTIPLY(tmp3, FIX_0_298631336);          /* -c1+c3+c5-c7 */
    tmp0 += z1 + tmp12;
    tmp3 += z1 + tmp13;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447);   /* -c1-c3 */
    tmp1 = MULTIPLY(tmp1, FIX_3_072711026);          /*  c1+c3+c5-c7 */
    tmp2 = MULTIPLY(tmp2, FIX_2_053119869);          /*  c1+c3-c5+c7 */
    tmp1 += z1 + tmp13;
    tmp2 += z1 + tmp12;

    dataptr[1] = (DCTELEM) DESCALE(tmp0, CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp1, CONST_BITS-PASS1_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp2, CONST_BITS-PASS1_BITS);
    dataptr[7] = (DCTELEM) DESCALE(tmp3, CONST_BITS-PASS1_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == DCTSIZE * 2)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by 8/16 = 1/2.
   * 16-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/32).
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*4];
    tmp4 = dataptr[DCTSIZE*4] + wsptr[DCTSIZE*3];
    tmp5 = dataptr[DCTSIZE*5] + wsptr[DCTSIZE*2];
    tmp6 = dataptr[DCTSIZE*6] + wsptr[DCTSIZE*1];
    tmp7 = dataptr[DCTSIZE*7] + wsptr[DCTSIZE*0];

    tmp10 = tmp0 + tmp7;
    tmp14 = tmp0 - tmp7;
    tmp11 = tmp1 + tmp6;
    tmp15 = tmp1 - tmp6;
    tmp12 = tmp2 + tmp5;
    tmp16 = tmp2 - tmp5;
    tmp13 = tmp3 + tmp4;
    tmp17 = tmp3 - tmp4;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*4];
    tmp4 = dataptr[DCTSIZE*4] - wsptr[DCTSIZE*3];
    tmp5 = dataptr[DCTSIZE*5] - wsptr[DCTSIZE*2];
    tmp6 = dataptr[DCTSIZE*6] - wsptr[DCTSIZE*1];
    tmp7 = dataptr[DCTSIZE*7] - wsptr[DCTSIZE*0];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(tmp10 + tmp11 + tmp12 + tmp13, PASS1_BITS+1);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(1.306562965)) + /* c4[16] = c2[8] */
	      MULTIPLY(tmp11 - tmp12, FIX_0_541196100),   /* c12[16] = c6[8] */
	      CONST_BITS+PASS1_BITS+1);

    tmp10 = MULTIPLY(tmp17 - tmp15, FIX(0.275899379)) +   /* c14[16] = c7[8] */
	    MULTIPLY(tmp14 - tmp16, FIX(1.387039845));    /* c2[16] = c1[8] */

    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp15, FIX(1.451774982))   /* c6+c14 */
	      + MULTIPLY(tmp16, FIX(2.172734804)),        /* c2+c10 */
	      CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(0.211164243))   /* c2-c6 */
	      - MULTIPLY(tmp17, FIX(1.061594338)),        /* c10+c14 */
	      CONST_BITS+PASS1_BITS+1);

    /* Odd part */

    tmp11 = MULTIPLY(tmp0 + tmp1, FIX(1.353318001)) +         /* c3 */
	    MULTIPLY(tmp6 - tmp7, FIX(0.410524528));          /* c13 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(1.247225013)) +         /* c5 */
	    MULTIPLY(tmp5 + tmp7, FIX(0.666655658));          /* c11 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(1.093201867)) +         /* c7 */
	    MULTIPLY(tmp4 - tmp7, FIX(0.897167586));          /* c9 */
    tmp14 = MULTIPLY(tmp1 + tmp2, FIX(0.138617169)) +         /* c15 */
	    MULTIPLY(tmp6 - tmp5, FIX(1.407403738));          /* c1 */
    tmp15 = MULTIPLY(tmp1 + tmp3, - FIX(0.666655658)) +       /* -c11 */
	    MULTIPLY(tmp4 + tmp6, - FIX(1.247225013));        /* -c5 */
    tmp16 = MULTIPLY(tmp2 + tmp3, - FIX(1.353318001)) +       /* -c3 */
	    MULTIPLY(tmp5 - tmp4, FIX(0.410524528));          /* c13 */
    tmp10 = tmp11 + tmp12 + tmp13 -
	    MULTIPLY(tmp0, FIX(2.286341144)) +                /* c7+c5+c3-c1 */
	    MULTIPLY(tmp7, FIX(0.779653625));                 /* c15+c13-c11+c9 */
    tmp11 += tmp14 + tmp15 + MULTIPLY(tmp1, FIX(0.071888074)) /* c9-c3-c15+c11 */
	     - MULTIPLY(tmp6, FIX(1.663905119));              /* c7+c13+c1-c5 */
    tmp12 += tmp14 + tmp16 - MULTIPLY(tmp2, FIX(1.125726048)) /* c7+c5+c15-c3 */
	     + MULTIPLY(tmp5, FIX(1.227391138));              /* c9-c11+c1-c13 */
    tmp13 += tmp15 + tmp16 + MULTIPLY(tmp3, FIX(1.065388962)) /* c15+c3+c11-c7 */
	     + MULTIPLY(tmp4, FIX(2.167985692));              /* c1+c13+c5-c9 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp10, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp11, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp12, CONST_BITS+PASS1_BITS+1);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp13, CONST_BITS+PASS1_BITS+1);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 7x14 sample block.
 *
 * 7-point FDCT in pass 1 (rows), 14-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_7x14 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16;
  INT32 z1, z2, z3;
  DCTELEM workspace[8*6];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 7-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/14).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[6]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[5]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[4]);
    tmp3 = GETJSAMPLE(elemptr[3]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[6]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[5]);
    tmp12 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[4]);

    z1 = tmp0 + tmp2;
    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((z1 + tmp1 + tmp3 - 7 * CENTERJSAMPLE) << PASS1_BITS);
    tmp3 += tmp3;
    z1 -= tmp3;
    z1 -= tmp3;
    z1 = MULTIPLY(z1, FIX(0.353553391));                /* (c2+c6-c4)/2 */
    z2 = MULTIPLY(tmp0 - tmp2, FIX(0.920609002));       /* (c2+c4-c6)/2 */
    z3 = MULTIPLY(tmp1 - tmp2, FIX(0.314692123));       /* c6 */
    dataptr[2] = (DCTELEM) DESCALE(z1 + z2 + z3, CONST_BITS-PASS1_BITS);
    z1 -= z2;
    z2 = MULTIPLY(tmp0 - tmp1, FIX(0.881747734));       /* c4 */
    dataptr[4] = (DCTELEM)
      DESCALE(z2 + z3 - MULTIPLY(tmp1 - tmp3, FIX(0.707106781)), /* c2+c6-c4 */
	      CONST_BITS-PASS1_BITS);
    dataptr[6] = (DCTELEM) DESCALE(z1 + z2, CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp1 = MULTIPLY(tmp10 + tmp11, FIX(0.935414347));   /* (c3+c1-c5)/2 */
    tmp2 = MULTIPLY(tmp10 - tmp11, FIX(0.170262339));   /* (c3+c5-c1)/2 */
    tmp0 = tmp1 - tmp2;
    tmp1 += tmp2;
    tmp2 = MULTIPLY(tmp11 + tmp12, - FIX(1.378756276)); /* -c1 */
    tmp1 += tmp2;
    tmp3 = MULTIPLY(tmp10 + tmp12, FIX(0.613604268));   /* c5 */
    tmp0 += tmp3;
    tmp2 += tmp3 + MULTIPLY(tmp12, FIX(1.870828693));   /* c3+c1-c5 */

    dataptr[1] = (DCTELEM) DESCALE(tmp0, CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM) DESCALE(tmp1, CONST_BITS-PASS1_BITS);
    dataptr[5] = (DCTELEM) DESCALE(tmp2, CONST_BITS-PASS1_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 14)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/7)*(8/14) = 32/49, which we
   * fold into the constant multipliers:
   * 14-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/28) * 32/49.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = 0; ctr < 7; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*5];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*3];
    tmp13 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*2];
    tmp4 = dataptr[DCTSIZE*4] + wsptr[DCTSIZE*1];
    tmp5 = dataptr[DCTSIZE*5] + wsptr[DCTSIZE*0];
    tmp6 = dataptr[DCTSIZE*6] + dataptr[DCTSIZE*7];

    tmp10 = tmp0 + tmp6;
    tmp14 = tmp0 - tmp6;
    tmp11 = tmp1 + tmp5;
    tmp15 = tmp1 - tmp5;
    tmp12 = tmp2 + tmp4;
    tmp16 = tmp2 - tmp4;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*5];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*3];
    tmp3 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*2];
    tmp4 = dataptr[DCTSIZE*4] - wsptr[DCTSIZE*1];
    tmp5 = dataptr[DCTSIZE*5] - wsptr[DCTSIZE*0];
    tmp6 = dataptr[DCTSIZE*6] - dataptr[DCTSIZE*7];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11 + tmp12 + tmp13,
		       FIX(0.653061224)),                 /* 32/49 */
	      CONST_BITS+PASS1_BITS);
    tmp13 += tmp13;
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp13, FIX(0.832106052)) + /* c4 */
	      MULTIPLY(tmp11 - tmp13, FIX(0.205513223)) - /* c12 */
	      MULTIPLY(tmp12 - tmp13, FIX(0.575835255)),  /* c8 */
	      CONST_BITS+PASS1_BITS);

    tmp10 = MULTIPLY(tmp14 + tmp15, FIX(0.722074570));    /* c6 */

    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp14, FIX(0.178337691))   /* c2-c6 */
	      + MULTIPLY(tmp16, FIX(0.400721155)),        /* c10 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp15, FIX(1.122795725))   /* c6+c10 */
	      - MULTIPLY(tmp16, FIX(0.900412262)),        /* c2 */
	      CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp10 = tmp1 + tmp2;
    tmp11 = tmp5 - tmp4;
    dataptr[DCTSIZE*7] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp10 + tmp3 - tmp11 - tmp6,
		       FIX(0.653061224)),                 /* 32/49 */
	      CONST_BITS+PASS1_BITS);
    tmp3  = MULTIPLY(tmp3 , FIX(0.653061224));            /* 32/49 */
    tmp10 = MULTIPLY(tmp10, - FIX(0.103406812));          /* -c13 */
    tmp11 = MULTIPLY(tmp11, FIX(0.917760839));            /* c1 */
    tmp10 += tmp11 - tmp3;
    tmp11 = MULTIPLY(tmp0 + tmp2, FIX(0.782007410)) +     /* c5 */
	    MULTIPLY(tmp4 + tmp6, FIX(0.491367823));      /* c9 */
    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(tmp10 + tmp11 - MULTIPLY(tmp2, FIX(1.550341076)) /* c3+c5-c13 */
	      + MULTIPLY(tmp4, FIX(0.731428202)),         /* c1+c11-c9 */
	      CONST_BITS+PASS1_BITS);
    tmp12 = MULTIPLY(tmp0 + tmp1, FIX(0.871740478)) +     /* c3 */
	    MULTIPLY(tmp5 - tmp6, FIX(0.305035186));      /* c11 */
    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(tmp10 + tmp12 - MULTIPLY(tmp1, FIX(0.276965844)) /* c3-c9-c13 */
	      - MULTIPLY(tmp5, FIX(2.004803435)),         /* c1+c5+c11 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp11 + tmp12 + tmp3
	      - MULTIPLY(tmp0, FIX(0.735987049))          /* c3+c5-c1 */
	      - MULTIPLY(tmp6, FIX(0.082925825)),         /* c9-c11-c13 */
	      CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 6x12 sample block.
 *
 * 6-point FDCT in pass 1 (rows), 12-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_6x12 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  DCTELEM workspace[8*4];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 6-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/12).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[5]);
    tmp11 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[4]);
    tmp2 = GETJSAMPLE(elemptr[2]) + GETJSAMPLE(elemptr[3]);

    tmp10 = tmp0 + tmp2;
    tmp12 = tmp0 - tmp2;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[5]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[4]);
    tmp2 = GETJSAMPLE(elemptr[2]) - GETJSAMPLE(elemptr[3]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp11 - 6 * CENTERJSAMPLE) << PASS1_BITS);
    dataptr[2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp12, FIX(1.224744871)),                 /* c2 */
	      CONST_BITS-PASS1_BITS);
    dataptr[4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp11, FIX(0.707106781)), /* c4 */
	      CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp10 = DESCALE(MULTIPLY(tmp0 + tmp2, FIX(0.366025404)),     /* c5 */
		    CONST_BITS-PASS1_BITS);

    dataptr[1] = (DCTELEM) (tmp10 + ((tmp0 + tmp1) << PASS1_BITS));
    dataptr[3] = (DCTELEM) ((tmp0 - tmp1 - tmp2) << PASS1_BITS);
    dataptr[5] = (DCTELEM) (tmp10 + ((tmp2 - tmp1) << PASS1_BITS));

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 12)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/6)*(8/12) = 8/9, which we
   * fold into the constant multipliers:
   * 12-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/24) * 8/9.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = 0; ctr < 6; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*3];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*2];
    tmp2 = dataptr[DCTSIZE*2] + wsptr[DCTSIZE*1];
    tmp3 = dataptr[DCTSIZE*3] + wsptr[DCTSIZE*0];
    tmp4 = dataptr[DCTSIZE*4] + dataptr[DCTSIZE*7];
    tmp5 = dataptr[DCTSIZE*5] + dataptr[DCTSIZE*6];

    tmp10 = tmp0 + tmp5;
    tmp13 = tmp0 - tmp5;
    tmp11 = tmp1 + tmp4;
    tmp14 = tmp1 - tmp4;
    tmp12 = tmp2 + tmp3;
    tmp15 = tmp2 - tmp3;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*3];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*2];
    tmp2 = dataptr[DCTSIZE*2] - wsptr[DCTSIZE*1];
    tmp3 = dataptr[DCTSIZE*3] - wsptr[DCTSIZE*0];
    tmp4 = dataptr[DCTSIZE*4] - dataptr[DCTSIZE*7];
    tmp5 = dataptr[DCTSIZE*5] - dataptr[DCTSIZE*6];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11 + tmp12, FIX(0.888888889)), /* 8/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(MULTIPLY(tmp13 - tmp14 - tmp15, FIX(0.888888889)), /* 8/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.088662108)),         /* c4 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp14 - tmp15, FIX(0.888888889)) +        /* 8/9 */
	      MULTIPLY(tmp13 + tmp15, FIX(1.214244803)),         /* c2 */
	      CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp1 + tmp4, FIX(0.481063200));   /* c9 */
    tmp14 = tmp10 + MULTIPLY(tmp1, FIX(0.680326102));  /* c3-c9 */
    tmp15 = tmp10 - MULTIPLY(tmp4, FIX(1.642452502));  /* c3+c9 */
    tmp12 = MULTIPLY(tmp0 + tmp2, FIX(0.997307603));   /* c5 */
    tmp13 = MULTIPLY(tmp0 + tmp3, FIX(0.765261039));   /* c7 */
    tmp10 = tmp12 + tmp13 + tmp14 - MULTIPLY(tmp0, FIX(0.516244403)) /* c5+c7-c1 */
	    + MULTIPLY(tmp5, FIX(0.164081699));        /* c11 */
    tmp11 = MULTIPLY(tmp2 + tmp3, - FIX(0.164081699)); /* -c11 */
    tmp12 += tmp11 - tmp15 - MULTIPLY(tmp2, FIX(2.079550144)) /* c1+c5-c11 */
	    + MULTIPLY(tmp5, FIX(0.765261039));        /* c7 */
    tmp13 += tmp11 - tmp14 + MULTIPLY(tmp3, FIX(0.645144899)) /* c1+c11-c7 */
	    - MULTIPLY(tmp5, FIX(0.997307603));        /* c5 */
    tmp11 = tmp15 + MULTIPLY(tmp0 - tmp3, FIX(1.161389302)) /* c3 */
	    - MULTIPLY(tmp2 + tmp5, FIX(0.481063200)); /* c9 */

    dataptr[DCTSIZE*1] = (DCTELEM) DESCALE(tmp10, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp11, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp12, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp13, CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 5x10 sample block.
 *
 * 5-point FDCT in pass 1 (rows), 10-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_5x10 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3, tmp4;
  INT32 tmp10, tmp11, tmp12, tmp13, tmp14;
  DCTELEM workspace[8*2];
  DCTELEM *dataptr;
  DCTELEM *wsptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * 5-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/10).
   */

  dataptr = data;
  ctr = 0;
  for (;;) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[4]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[3]);
    tmp2 = GETJSAMPLE(elemptr[2]);

    tmp10 = tmp0 + tmp1;
    tmp11 = tmp0 - tmp1;

    tmp0 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[4]);
    tmp1 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[3]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp10 + tmp2 - 5 * CENTERJSAMPLE) << PASS1_BITS);
    tmp11 = MULTIPLY(tmp11, FIX(0.790569415));          /* (c2+c4)/2 */
    tmp10 -= tmp2 << 2;
    tmp10 = MULTIPLY(tmp10, FIX(0.353553391));          /* (c2-c4)/2 */
    dataptr[2] = (DCTELEM) DESCALE(tmp11 + tmp10, CONST_BITS-PASS1_BITS);
    dataptr[4] = (DCTELEM) DESCALE(tmp11 - tmp10, CONST_BITS-PASS1_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp0 + tmp1, FIX(0.831253876));    /* c3 */

    dataptr[1] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp0, FIX(0.513743148)), /* c1-c3 */
	      CONST_BITS-PASS1_BITS);
    dataptr[3] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp1, FIX(2.176250899)), /* c1+c3 */
	      CONST_BITS-PASS1_BITS);

    ctr++;

    if (ctr != DCTSIZE) {
      if (ctr == 10)
	break;			/* Done. */
      dataptr += DCTSIZE;	/* advance pointer to next row */
    } else
      dataptr = workspace;	/* switch pointer to extended workspace */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/5)*(8/10) = 32/25, which we
   * fold into the constant multipliers:
   * 10-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/20) * 32/25.
   */

  dataptr = data;
  wsptr = workspace;
  for (ctr = 0; ctr < 5; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + wsptr[DCTSIZE*1];
    tmp1 = dataptr[DCTSIZE*1] + wsptr[DCTSIZE*0];
    tmp12 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*7];
    tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*6];
    tmp4 = dataptr[DCTSIZE*4] + dataptr[DCTSIZE*5];

    tmp10 = tmp0 + tmp4;
    tmp13 = tmp0 - tmp4;
    tmp11 = tmp1 + tmp3;
    tmp14 = tmp1 - tmp3;

    tmp0 = dataptr[DCTSIZE*0] - wsptr[DCTSIZE*1];
    tmp1 = dataptr[DCTSIZE*1] - wsptr[DCTSIZE*0];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*7];
    tmp3 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*6];
    tmp4 = dataptr[DCTSIZE*4] - dataptr[DCTSIZE*5];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11 + tmp12, FIX(1.28)), /* 32/25 */
	      CONST_BITS+PASS1_BITS);
    tmp12 += tmp12;
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp12, FIX(1.464477191)) - /* c4 */
	      MULTIPLY(tmp11 - tmp12, FIX(0.559380511)),  /* c8 */
	      CONST_BITS+PASS1_BITS);
    tmp10 = MULTIPLY(tmp13 + tmp14, FIX(1.064004961));    /* c6 */
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp13, FIX(0.657591230)),  /* c2-c6 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*6] = (DCTELEM)
      DESCALE(tmp10 - MULTIPLY(tmp14, FIX(2.785601151)),  /* c2+c6 */
	      CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp10 = tmp0 + tmp4;
    tmp11 = tmp1 - tmp3;
    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp2, FIX(1.28)),  /* 32/25 */
	      CONST_BITS+PASS1_BITS);
    tmp2 = MULTIPLY(tmp2, FIX(1.28));                     /* 32/25 */
    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0, FIX(1.787906876)) +          /* c1 */
	      MULTIPLY(tmp1, FIX(1.612894094)) + tmp2 +   /* c3 */
	      MULTIPLY(tmp3, FIX(0.821810588)) +          /* c7 */
	      MULTIPLY(tmp4, FIX(0.283176630)),           /* c9 */
	      CONST_BITS+PASS1_BITS);
    tmp12 = MULTIPLY(tmp0 - tmp4, FIX(1.217352341)) -     /* (c3+c7)/2 */
	    MULTIPLY(tmp1 + tmp3, FIX(0.752365123));      /* (c1-c9)/2 */
    tmp13 = MULTIPLY(tmp10 + tmp11, FIX(0.395541753)) +   /* (c3-c7)/2 */
	    MULTIPLY(tmp11, FIX(0.64)) - tmp2;            /* 16/25 */
    dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp12 + tmp13, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp12 - tmp13, CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
    wsptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 4x8 sample block.
 *
 * 4-point FDCT in pass 1 (rows), 8-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_4x8 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2, tmp3;
  INT32 tmp10, tmp11, tmp12, tmp13;
  INT32 z1;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * We must also scale the output by 8/4 = 2, which we add here.
   * 4-point FDCT kernel,
   * cK represents sqrt(2) * cos(K*pi/16) [refers to 8-point FDCT].
   */

  dataptr = data;
  for (ctr = 0; ctr < DCTSIZE; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[3]);
    tmp1 = GETJSAMPLE(elemptr[1]) + GETJSAMPLE(elemptr[2]);

    tmp10 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[3]);
    tmp11 = GETJSAMPLE(elemptr[1]) - GETJSAMPLE(elemptr[2]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp0 + tmp1 - 4 * CENTERJSAMPLE) << (PASS1_BITS+1));
    dataptr[2] = (DCTELEM) ((tmp0 - tmp1) << (PASS1_BITS+1));

    /* Odd part */

    tmp0 = MULTIPLY(tmp10 + tmp11, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    tmp0 += ONE << (CONST_BITS-PASS1_BITS-2);

    dataptr[1] = (DCTELEM)
      RIGHT_SHIFT(tmp0 + MULTIPLY(tmp10, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS-PASS1_BITS-1);
    dataptr[3] = (DCTELEM)
      RIGHT_SHIFT(tmp0 - MULTIPLY(tmp11, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS-PASS1_BITS-1);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * 8-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/16).
   */

  dataptr = data;
  for (ctr = 0; ctr < 4; ctr++) {
    /* Even part per LL&M figure 1 --- note that published figure is faulty;
     * rotator "c1" should be "c6".
     */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*4];

    /* Add fudge factor here for final descale. */
    tmp10 = tmp0 + tmp3 + (ONE << (PASS1_BITS-1));
    tmp12 = tmp0 - tmp3;
    tmp11 = tmp1 + tmp2;
    tmp13 = tmp1 - tmp2;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*7];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*6];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*5];
    tmp3 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*4];

    dataptr[DCTSIZE*0] = (DCTELEM) RIGHT_SHIFT(tmp10 + tmp11, PASS1_BITS);
    dataptr[DCTSIZE*4] = (DCTELEM) RIGHT_SHIFT(tmp10 - tmp11, PASS1_BITS);

    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS+PASS1_BITS-1);

    dataptr[DCTSIZE*2] = (DCTELEM)
      RIGHT_SHIFT(z1 + MULTIPLY(tmp12, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*6] = (DCTELEM)
      RIGHT_SHIFT(z1 - MULTIPLY(tmp13, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS+PASS1_BITS);

    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
     * i0..i3 in the paper are tmp0..tmp3 here.
     */

    tmp12 = tmp0 + tmp2;
    tmp13 = tmp1 + tmp3;

    z1 = MULTIPLY(tmp12 + tmp13, FIX_1_175875602);       /*  c3 */
    /* Add fudge factor here for final descale. */
    z1 += ONE << (CONST_BITS+PASS1_BITS-1);

    tmp12 = MULTIPLY(tmp12, - FIX_0_390180644);          /* -c3+c5 */
    tmp13 = MULTIPLY(tmp13, - FIX_1_961570560);          /* -c3-c5 */
    tmp12 += z1;
    tmp13 += z1;

    z1 = MULTIPLY(tmp0 + tmp3, - FIX_0_899976223);       /* -c3+c7 */
    tmp0 = MULTIPLY(tmp0, FIX_1_501321110);              /*  c1+c3-c5-c7 */
    tmp3 = MULTIPLY(tmp3, FIX_0_298631336);              /* -c1+c3+c5-c7 */
    tmp0 += z1 + tmp12;
    tmp3 += z1 + tmp13;

    z1 = MULTIPLY(tmp1 + tmp2, - FIX_2_562915447);       /* -c1-c3 */
    tmp1 = MULTIPLY(tmp1, FIX_3_072711026);              /*  c1+c3+c5-c7 */
    tmp2 = MULTIPLY(tmp2, FIX_2_053119869);              /*  c1+c3-c5+c7 */
    tmp1 += z1 + tmp13;
    tmp2 += z1 + tmp12;

    dataptr[DCTSIZE*1] = (DCTELEM) RIGHT_SHIFT(tmp0, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM) RIGHT_SHIFT(tmp1, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*5] = (DCTELEM) RIGHT_SHIFT(tmp2, CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*7] = (DCTELEM) RIGHT_SHIFT(tmp3, CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 3x6 sample block.
 *
 * 3-point FDCT in pass 1 (rows), 6-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_3x6 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1, tmp2;
  INT32 tmp10, tmp11, tmp12;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT;
   * furthermore, we scale the results by 2**PASS1_BITS.
   * We scale the results further by 2 as part of output adaption
   * scaling for different DCT size.
   * 3-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/6).
   */

  dataptr = data;
  for (ctr = 0; ctr < 6; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]) + GETJSAMPLE(elemptr[2]);
    tmp1 = GETJSAMPLE(elemptr[1]);

    tmp2 = GETJSAMPLE(elemptr[0]) - GETJSAMPLE(elemptr[2]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM)
      ((tmp0 + tmp1 - 3 * CENTERJSAMPLE) << (PASS1_BITS+1));
    dataptr[2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp1 - tmp1, FIX(0.707106781)), /* c2 */
	      CONST_BITS-PASS1_BITS-1);

    /* Odd part */

    dataptr[1] = (DCTELEM)
      DESCALE(MULTIPLY(tmp2, FIX(1.224744871)),               /* c1 */
	      CONST_BITS-PASS1_BITS-1);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We remove the PASS1_BITS scaling, but leave the results scaled up
   * by an overall factor of 8.
   * We must also scale the output by (8/6)*(8/3) = 32/9, which we partially
   * fold into the constant multipliers (other part was done in pass 1):
   * 6-point FDCT kernel, cK represents sqrt(2) * cos(K*pi/12) * 16/9.
   */

  dataptr = data;
  for (ctr = 0; ctr < 3; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*5];
    tmp11 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*3];

    tmp10 = tmp0 + tmp2;
    tmp12 = tmp0 - tmp2;

    tmp0 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*5];
    tmp1 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*4];
    tmp2 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*3];

    dataptr[DCTSIZE*0] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 + tmp11, FIX(1.777777778)),         /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*2] = (DCTELEM)
      DESCALE(MULTIPLY(tmp12, FIX(2.177324216)),                 /* c2 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*4] = (DCTELEM)
      DESCALE(MULTIPLY(tmp10 - tmp11 - tmp11, FIX(1.257078722)), /* c4 */
	      CONST_BITS+PASS1_BITS);

    /* Odd part */

    tmp10 = MULTIPLY(tmp0 + tmp2, FIX(0.650711829));             /* c5 */

    dataptr[DCTSIZE*1] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp0 + tmp1, FIX(1.777777778)),   /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*3] = (DCTELEM)
      DESCALE(MULTIPLY(tmp0 - tmp1 - tmp2, FIX(1.777777778)),    /* 16/9 */
	      CONST_BITS+PASS1_BITS);
    dataptr[DCTSIZE*5] = (DCTELEM)
      DESCALE(tmp10 + MULTIPLY(tmp2 - tmp1, FIX(1.777777778)),   /* 16/9 */
	      CONST_BITS+PASS1_BITS);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 2x4 sample block.
 *
 * 2-point FDCT in pass 1 (rows), 4-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_2x4 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  INT32 tmp0, tmp1;
  INT32 tmp10, tmp11;
  DCTELEM *dataptr;
  JSAMPROW elemptr;
  int ctr;
  SHIFT_TEMPS

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: process rows.
   * Note results are scaled up by sqrt(8) compared to a true DCT.
   */

  dataptr = data;
  for (ctr = 0; ctr < 4; ctr++) {
    elemptr = sample_data[ctr] + start_col;

    /* Even part */

    tmp0 = GETJSAMPLE(elemptr[0]);
    tmp1 = GETJSAMPLE(elemptr[1]);

    /* Apply unsigned->signed conversion. */
    dataptr[0] = (DCTELEM) (tmp0 + tmp1 - 2 * CENTERJSAMPLE);

    /* Odd part */

    dataptr[1] = (DCTELEM) (tmp0 - tmp1);

    dataptr += DCTSIZE;		/* advance pointer to next row */
  }

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/2)*(8/4) = 2**3.
   * 4-point FDCT kernel,
   * cK represents sqrt(2) * cos(K*pi/16) [refers to 8-point FDCT].
   */

  dataptr = data;
  for (ctr = 0; ctr < 2; ctr++) {
    /* Even part */

    tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*3];
    tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*2];

    tmp10 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*3];
    tmp11 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*2];

    dataptr[DCTSIZE*0] = (DCTELEM) ((tmp0 + tmp1) << 3);
    dataptr[DCTSIZE*2] = (DCTELEM) ((tmp0 - tmp1) << 3);

    /* Odd part */

    tmp0 = MULTIPLY(tmp10 + tmp11, FIX_0_541196100);       /* c6 */
    /* Add fudge factor here for final descale. */
    tmp0 += ONE << (CONST_BITS-3-1);

    dataptr[DCTSIZE*1] = (DCTELEM)
      RIGHT_SHIFT(tmp0 + MULTIPLY(tmp10, FIX_0_765366865), /* c2-c6 */
		  CONST_BITS-3);
    dataptr[DCTSIZE*3] = (DCTELEM)
      RIGHT_SHIFT(tmp0 - MULTIPLY(tmp11, FIX_1_847759065), /* c2+c6 */
		  CONST_BITS-3);

    dataptr++;			/* advance pointer to next column */
  }
}


/*
 * Perform the forward DCT on a 1x2 sample block.
 *
 * 1-point FDCT in pass 1 (rows), 2-point in pass 2 (columns).
 */

GLOBAL(void)
jpeg_fdct_1x2 (DCTELEM * data, JSAMPARRAY sample_data, JDIMENSION start_col)
{
  DCTELEM tmp0, tmp1;

  /* Pre-zero output coefficient block. */
  MEMZERO(data, SIZEOF(DCTELEM) * DCTSIZE2);

  /* Pass 1: empty. */

  /* Pass 2: process columns.
   * We leave the results scaled up by an overall factor of 8.
   * We must also scale the output by (8/1)*(8/2) = 2**5.
   */

  /* Even part */

  tmp0 = GETJSAMPLE(sample_data[0][start_col]);
  tmp1 = GETJSAMPLE(sample_data[1][start_col]);

  /* Apply unsigned->signed conversion. */
  data[DCTSIZE*0] = (tmp0 + tmp1 - 2 * CENTERJSAMPLE) << 5;

  /* Odd part */

  data[DCTSIZE*1] = (tmp0 - tmp1) << 5;
}

#endif /* DCT_SCALING_SUPPORTED */
#endif /* DCT_ISLOW_SUPPORTED */
