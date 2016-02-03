 /*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2005-2006, Dept. of Electronic and Information Engineering, Universita' degli Studi di Perugia, Italy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef USE_JPWL

/**
@file rs.c
@brief Functions used to compute the Reed-Solomon parity and check of byte arrays

*/

/**
 * Reed-Solomon coding and decoding
 * Phil Karn (karn@ka9q.ampr.org) September 1996
 *
 * This file is derived from the program "new_rs_erasures.c" by Robert
 * Morelos-Zaragoza (robert@spectra.eng.hawaii.edu) and Hari Thirumoorthy
 * (harit@spectra.eng.hawaii.edu), Aug 1995
 *
 * I've made changes to improve performance, clean up the code and make it
 * easier to follow. Data is now passed to the encoding and decoding functions
 * through arguments rather than in global arrays. The decode function returns
 * the number of corrected symbols, or -1 if the word is uncorrectable.
 *
 * This code supports a symbol size from 2 bits up to 16 bits,
 * implying a block size of 3 2-bit symbols (6 bits) up to 65535
 * 16-bit symbols (1,048,560 bits). The code parameters are set in rs.h.
 *
 * Note that if symbols larger than 8 bits are used, the type of each
 * data array element switches from unsigned char to unsigned int. The
 * caller must ensure that elements larger than the symbol range are
 * not passed to the encoder or decoder.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "rs.h"

/* This defines the type used to store an element of the Galois Field
 * used by the code. Make sure this is something larger than a char if
 * if anything larger than GF(256) is used.
 *
 * Note: unsigned char will work up to GF(256) but int seems to run
 * faster on the Pentium.
 */
typedef int gf;

/* Primitive polynomials - see Lin & Costello, Appendix A,
 * and  Lee & Messerschmitt, p. 453.
 */
#if(MM == 2)/* Admittedly silly */
int Pp[MM+1] = { 1, 1, 1 };

#elif(MM == 3)
/* 1 + x + x^3 */
int Pp[MM+1] = { 1, 1, 0, 1 };

#elif(MM == 4)
/* 1 + x + x^4 */
int Pp[MM+1] = { 1, 1, 0, 0, 1 };

#elif(MM == 5)
/* 1 + x^2 + x^5 */
int Pp[MM+1] = { 1, 0, 1, 0, 0, 1 };

#elif(MM == 6)
/* 1 + x + x^6 */
int Pp[MM+1] = { 1, 1, 0, 0, 0, 0, 1 };

#elif(MM == 7)
/* 1 + x^3 + x^7 */
int Pp[MM+1] = { 1, 0, 0, 1, 0, 0, 0, 1 };

#elif(MM == 8)
/* 1+x^2+x^3+x^4+x^8 */
int Pp[MM+1] = { 1, 0, 1, 1, 1, 0, 0, 0, 1 };

#elif(MM == 9)
/* 1+x^4+x^9 */
int Pp[MM+1] = { 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

#elif(MM == 10)
/* 1+x^3+x^10 */
int Pp[MM+1] = { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1 };

#elif(MM == 11)
/* 1+x^2+x^11 */
int Pp[MM+1] = { 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

#elif(MM == 12)
/* 1+x+x^4+x^6+x^12 */
int Pp[MM+1] = { 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1 };

#elif(MM == 13)
/* 1+x+x^3+x^4+x^13 */
int Pp[MM+1] = { 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

#elif(MM == 14)
/* 1+x+x^6+x^10+x^14 */
int Pp[MM+1] = { 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 };

#elif(MM == 15)
/* 1+x+x^15 */
int Pp[MM+1] = { 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

#elif(MM == 16)
/* 1+x+x^3+x^12+x^16 */
int Pp[MM+1] = { 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1 };

#else
#error "MM must be in range 2-16"
#endif

/* Alpha exponent for the first root of the generator polynomial */
#define B0  0  /* Different from the default 1 */

/* index->polynomial form conversion table */
gf Alpha_to[NN + 1];

/* Polynomial->index form conversion table */
gf Index_of[NN + 1];

/* No legal value in index form represents zero, so
 * we need a special value for this purpose
 */
#define A0  (NN)

/* Generator polynomial g(x)
 * Degree of g(x) = 2*TT
 * has roots @**B0, @**(B0+1), ... ,@^(B0+2*TT-1)
 */
/*gf Gg[NN - KK + 1];*/
gf    Gg[NN - 1];

/* Compute x % NN, where NN is 2**MM - 1,
 * without a slow divide
 */
static /*inline*/ gf
modnn(int x)
{
  while (x >= NN) {
    x -= NN;
    x = (x >> MM) + (x & NN);
  }
  return x;
}

/*#define  min(a,b)  ((a) < (b) ? (a) : (b))*/

#define  CLEAR(a,n) {\
  int ci;\
  for(ci=(n)-1;ci >=0;ci--)\
    (a)[ci] = 0;\
  }

#define  COPY(a,b,n) {\
  int ci;\
  for(ci=(n)-1;ci >=0;ci--)\
    (a)[ci] = (b)[ci];\
  }
#define  COPYDOWN(a,b,n) {\
  int ci;\
  for(ci=(n)-1;ci >=0;ci--)\
    (a)[ci] = (b)[ci];\
  }

void init_rs(int k)
{
  KK = k;
  if (KK >= NN) {
    printf("KK must be less than 2**MM - 1\n");
    exit(1);
  }

  generate_gf();
  gen_poly();
}

/* generate GF(2**m) from the irreducible polynomial p(X) in p[0]..p[m]
   lookup tables:  index->polynomial form   alpha_to[] contains j=alpha**i;
                   polynomial form -> index form  index_of[j=alpha**i] = i
   alpha=2 is the primitive element of GF(2**m)
   HARI's COMMENT: (4/13/94) alpha_to[] can be used as follows:
        Let @ represent the primitive element commonly called "alpha" that
   is the root of the primitive polynomial p(x). Then in GF(2^m), for any
   0 <= i <= 2^m-2,
        @^i = a(0) + a(1) @ + a(2) @^2 + ... + a(m-1) @^(m-1)
   where the binary vector (a(0),a(1),a(2),...,a(m-1)) is the representation
   of the integer "alpha_to[i]" with a(0) being the LSB and a(m-1) the MSB. Thus for
   example the polynomial representation of @^5 would be given by the binary
   representation of the integer "alpha_to[5]".
                   Similarily, index_of[] can be used as follows:
        As above, let @ represent the primitive element of GF(2^m) that is
   the root of the primitive polynomial p(x). In order to find the power
   of @ (alpha) that has the polynomial representation
        a(0) + a(1) @ + a(2) @^2 + ... + a(m-1) @^(m-1)
   we consider the integer "i" whose binary representation with a(0) being LSB
   and a(m-1) MSB is (a(0),a(1),...,a(m-1)) and locate the entry
   "index_of[i]". Now, @^index_of[i] is that element whose polynomial
    representation is (a(0),a(1),a(2),...,a(m-1)).
   NOTE:
        The element alpha_to[2^m-1] = 0 always signifying that the
   representation of "@^infinity" = 0 is (0,0,0,...,0).
        Similarily, the element index_of[0] = A0 always signifying
   that the power of alpha which has the polynomial representation
   (0,0,...,0) is "infinity".

*/

void
generate_gf(void)
{
  register int i, mask;

  mask = 1;
  Alpha_to[MM] = 0;
  for (i = 0; i < MM; i++) {
    Alpha_to[i] = mask;
    Index_of[Alpha_to[i]] = i;
    /* If Pp[i] == 1 then, term @^i occurs in poly-repr of @^MM */
    if (Pp[i] != 0)
      Alpha_to[MM] ^= mask;  /* Bit-wise EXOR operation */
    mask <<= 1;  /* single left-shift */
  }
  Index_of[Alpha_to[MM]] = MM;
  /*
   * Have obtained poly-repr of @^MM. Poly-repr of @^(i+1) is given by
   * poly-repr of @^i shifted left one-bit and accounting for any @^MM
   * term that may occur when poly-repr of @^i is shifted.
   */
  mask >>= 1;
  for (i = MM + 1; i < NN; i++) {
    if (Alpha_to[i - 1] >= mask)
      Alpha_to[i] = Alpha_to[MM] ^ ((Alpha_to[i - 1] ^ mask) << 1);
    else
      Alpha_to[i] = Alpha_to[i - 1] << 1;
    Index_of[Alpha_to[i]] = i;
  }
  Index_of[0] = A0;
  Alpha_to[NN] = 0;
}


/*
 * Obtain the generator polynomial of the TT-error correcting, length
 * NN=(2**MM -1) Reed Solomon code from the product of (X+@**(B0+i)), i = 0,
 * ... ,(2*TT-1)
 *
 * Examples:
 *
 * If B0 = 1, TT = 1. deg(g(x)) = 2*TT = 2.
 * g(x) = (x+@) (x+@**2)
 *
 * If B0 = 0, TT = 2. deg(g(x)) = 2*TT = 4.
 * g(x) = (x+1) (x+@) (x+@**2) (x+@**3)
 */
void
gen_poly(void)
{
  register int i, j;

  Gg[0] = Alpha_to[B0];
  Gg[1] = 1;    /* g(x) = (X+@**B0) initially */
  for (i = 2; i <= NN - KK; i++) {
    Gg[i] = 1;
    /*
     * Below multiply (Gg[0]+Gg[1]*x + ... +Gg[i]x^i) by
     * (@**(B0+i-1) + x)
     */
    for (j = i - 1; j > 0; j--)
      if (Gg[j] != 0)
        Gg[j] = Gg[j - 1] ^ Alpha_to[modnn((Index_of[Gg[j]]) + B0 + i - 1)];
      else
        Gg[j] = Gg[j - 1];
    /* Gg[0] can never be zero */
    Gg[0] = Alpha_to[modnn((Index_of[Gg[0]]) + B0 + i - 1)];
  }
  /* convert Gg[] to index form for quicker encoding */
  for (i = 0; i <= NN - KK; i++)
    Gg[i] = Index_of[Gg[i]];
}


/*
 * take the string of symbols in data[i], i=0..(k-1) and encode
 * systematically to produce NN-KK parity symbols in bb[0]..bb[NN-KK-1] data[]
 * is input and bb[] is output in polynomial form. Encoding is done by using
 * a feedback shift register with appropriate connections specified by the
 * elements of Gg[], which was generated above. Codeword is   c(X) =
 * data(X)*X**(NN-KK)+ b(X)
 */
int
encode_rs(dtype *data, dtype *bb)
{
  register int i, j;
  gf feedback;

  CLEAR(bb,NN-KK);
  for (i = KK - 1; i >= 0; i--) {
#if (MM != 8)
    if(data[i] > NN)
      return -1;  /* Illegal symbol */
#endif
    feedback = Index_of[data[i] ^ bb[NN - KK - 1]];
    if (feedback != A0) {  /* feedback term is non-zero */
      for (j = NN - KK - 1; j > 0; j--)
        if (Gg[j] != A0)
          bb[j] = bb[j - 1] ^ Alpha_to[modnn(Gg[j] + feedback)];
        else
          bb[j] = bb[j - 1];
      bb[0] = Alpha_to[modnn(Gg[0] + feedback)];
    } else {  /* feedback term is zero. encoder becomes a
         * single-byte shifter */
      for (j = NN - KK - 1; j > 0; j--)
        bb[j] = bb[j - 1];
      bb[0] = 0;
    }
  }
  return 0;
}

/*
 * Performs ERRORS+ERASURES decoding of RS codes. If decoding is successful,
 * writes the codeword into data[] itself. Otherwise data[] is unaltered.
 *
 * Return number of symbols corrected, or -1 if codeword is illegal
 * or uncorrectable.
 *
 * First "no_eras" erasures are declared by the calling program. Then, the
 * maximum # of errors correctable is t_after_eras = floor((NN-KK-no_eras)/2).
 * If the number of channel errors is not greater than "t_after_eras" the
 * transmitted codeword will be recovered. Details of algorithm can be found
 * in R. Blahut's "Theory ... of Error-Correcting Codes".
 */
int
eras_dec_rs(dtype *data, int *eras_pos, int no_eras)
{
  int deg_lambda, el, deg_omega;
  int i, j, r;
  gf u,q,tmp,num1,num2,den,discr_r;
  gf recd[NN];
  /* Err+Eras Locator poly and syndrome poly */
  /*gf lambda[NN-KK + 1], s[NN-KK + 1];
  gf b[NN-KK + 1], t[NN-KK + 1], omega[NN-KK + 1];
  gf root[NN-KK], reg[NN-KK + 1], loc[NN-KK];*/
  gf lambda[NN + 1], s[NN + 1];
  gf b[NN + 1], t[NN + 1], omega[NN + 1];
  gf root[NN], reg[NN + 1], loc[NN];
  int syn_error, count;

  /* data[] is in polynomial form, copy and convert to index form */
  for (i = NN-1; i >= 0; i--){
#if (MM != 8)
    if(data[i] > NN)
      return -1;  /* Illegal symbol */
#endif
    recd[i] = Index_of[data[i]];
  }
  /* first form the syndromes; i.e., evaluate recd(x) at roots of g(x)
   * namely @**(B0+i), i = 0, ... ,(NN-KK-1)
   */
  syn_error = 0;
  for (i = 1; i <= NN-KK; i++) {
    tmp = 0;
    for (j = 0; j < NN; j++)
      if (recd[j] != A0)  /* recd[j] in index form */
        tmp ^= Alpha_to[modnn(recd[j] + (B0+i-1)*j)];
    syn_error |= tmp;  /* set flag if non-zero syndrome =>
           * error */
    /* store syndrome in index form  */
    s[i] = Index_of[tmp];
  }
  if (!syn_error) {
    /*
     * if syndrome is zero, data[] is a codeword and there are no
     * errors to correct. So return data[] unmodified
     */
    return 0;
  }
  CLEAR(&lambda[1],NN-KK);
  lambda[0] = 1;
  if (no_eras > 0) {
    /* Init lambda to be the erasure locator polynomial */
    lambda[1] = Alpha_to[eras_pos[0]];
    for (i = 1; i < no_eras; i++) {
      u = eras_pos[i];
      for (j = i+1; j > 0; j--) {
        tmp = Index_of[lambda[j - 1]];
        if(tmp != A0)
          lambda[j] ^= Alpha_to[modnn(u + tmp)];
      }
    }
#ifdef ERASURE_DEBUG
    /* find roots of the erasure location polynomial */
    for(i=1;i<=no_eras;i++)
      reg[i] = Index_of[lambda[i]];
    count = 0;
    for (i = 1; i <= NN; i++) {
      q = 1;
      for (j = 1; j <= no_eras; j++)
        if (reg[j] != A0) {
          reg[j] = modnn(reg[j] + j);
          q ^= Alpha_to[reg[j]];
        }
      if (!q) {
        /* store root and error location
         * number indices
         */
        root[count] = i;
        loc[count] = NN - i;
        count++;
      }
    }
    if (count != no_eras) {
      printf("\n lambda(x) is WRONG\n");
      return -1;
    }
#ifndef NO_PRINT
    printf("\n Erasure positions as determined by roots of Eras Loc Poly:\n");
    for (i = 0; i < count; i++)
      printf("%d ", loc[i]);
    printf("\n");
#endif
#endif
  }
  for(i=0;i<NN-KK+1;i++)
    b[i] = Index_of[lambda[i]];

  /*
   * Begin Berlekamp-Massey algorithm to determine error+erasure
   * locator polynomial
   */
  r = no_eras;
  el = no_eras;
  while (++r <= NN-KK) {  /* r is the step number */
    /* Compute discrepancy at the r-th step in poly-form */
    discr_r = 0;
    for (i = 0; i < r; i++){
      if ((lambda[i] != 0) && (s[r - i] != A0)) {
        discr_r ^= Alpha_to[modnn(Index_of[lambda[i]] + s[r - i])];
      }
    }
    discr_r = Index_of[discr_r];  /* Index form */
    if (discr_r == A0) {
      /* 2 lines below: B(x) <-- x*B(x) */
      COPYDOWN(&b[1],b,NN-KK);
      b[0] = A0;
    } else {
      /* 7 lines below: T(x) <-- lambda(x) - discr_r*x*b(x) */
      t[0] = lambda[0];
      for (i = 0 ; i < NN-KK; i++) {
        if(b[i] != A0)
          t[i+1] = lambda[i+1] ^ Alpha_to[modnn(discr_r + b[i])];
        else
          t[i+1] = lambda[i+1];
      }
      if (2 * el <= r + no_eras - 1) {
        el = r + no_eras - el;
        /*
         * 2 lines below: B(x) <-- inv(discr_r) *
         * lambda(x)
         */
        for (i = 0; i <= NN-KK; i++)
          b[i] = (lambda[i] == 0) ? A0 : modnn(Index_of[lambda[i]] - discr_r + NN);
      } else {
        /* 2 lines below: B(x) <-- x*B(x) */
        COPYDOWN(&b[1],b,NN-KK);
        b[0] = A0;
      }
      COPY(lambda,t,NN-KK+1);
    }
  }

  /* Convert lambda to index form and compute deg(lambda(x)) */
  deg_lambda = 0;
  for(i=0;i<NN-KK+1;i++){
    lambda[i] = Index_of[lambda[i]];
    if(lambda[i] != A0)
      deg_lambda = i;
  }
  /*
   * Find roots of the error+erasure locator polynomial. By Chien
   * Search
   */
  COPY(&reg[1],&lambda[1],NN-KK);
  count = 0;    /* Number of roots of lambda(x) */
  for (i = 1; i <= NN; i++) {
    q = 1;
    for (j = deg_lambda; j > 0; j--)
      if (reg[j] != A0) {
        reg[j] = modnn(reg[j] + j);
        q ^= Alpha_to[reg[j]];
      }
    if (!q) {
      /* store root (index-form) and error location number */
      root[count] = i;
      loc[count] = NN - i;
      count++;
    }
  }

#ifdef DEBUG
  printf("\n Final error positions:\t");
  for (i = 0; i < count; i++)
    printf("%d ", loc[i]);
  printf("\n");
#endif
  if (deg_lambda != count) {
    /*
     * deg(lambda) unequal to number of roots => uncorrectable
     * error detected
     */
    return -1;
  }
  /*
   * Compute err+eras evaluator poly omega(x) = s(x)*lambda(x) (modulo
   * x**(NN-KK)). in index form. Also find deg(omega).
   */
  deg_omega = 0;
  for (i = 0; i < NN-KK;i++){
    tmp = 0;
    j = (deg_lambda < i) ? deg_lambda : i;
    for(;j >= 0; j--){
      if ((s[i + 1 - j] != A0) && (lambda[j] != A0))
        tmp ^= Alpha_to[modnn(s[i + 1 - j] + lambda[j])];
    }
    if(tmp != 0)
      deg_omega = i;
    omega[i] = Index_of[tmp];
  }
  omega[NN-KK] = A0;

  /*
   * Compute error values in poly-form. num1 = omega(inv(X(l))), num2 =
   * inv(X(l))**(B0-1) and den = lambda_pr(inv(X(l))) all in poly-form
   */
  for (j = count-1; j >=0; j--) {
    num1 = 0;
    for (i = deg_omega; i >= 0; i--) {
      if (omega[i] != A0)
        num1  ^= Alpha_to[modnn(omega[i] + i * root[j])];
    }
    num2 = Alpha_to[modnn(root[j] * (B0 - 1) + NN)];
    den = 0;

    /* lambda[i+1] for i even is the formal derivative lambda_pr of lambda[i] */
    for (i = min(deg_lambda,NN-KK-1) & ~1; i >= 0; i -=2) {
      if(lambda[i+1] != A0)
        den ^= Alpha_to[modnn(lambda[i+1] + i * root[j])];
    }
    if (den == 0) {
#ifdef DEBUG
      printf("\n ERROR: denominator = 0\n");
#endif
      return -1;
    }
    /* Apply error to data */
    if (num1 != 0) {
      data[loc[j]] ^= Alpha_to[modnn(Index_of[num1] + Index_of[num2] + NN - Index_of[den])];
    }
  }
  return count;
}


#endif /* USE_JPWL */
