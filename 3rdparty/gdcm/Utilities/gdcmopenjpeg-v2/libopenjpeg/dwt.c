/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2007, Jonathan Ballard <dzonatas@dzonux.net>
 * Copyright (c) 2007, Callum Lerwick <seg@haxxed.com>
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include "dwt.h"
#include "j2k.h"
#include "tcd.h"
#include "fix.h"
#include "opj_malloc.h"
#include "int.h"

/** @defgroup DWT DWT - Implementation of a discrete wavelet transform */
/*@{*/

#define WS(i) v->mem[(i)*2]
#define WD(i) v->mem[(1+(i)*2)]

/** @name Local data structures */
/*@{*/

typedef struct dwt_local {
  OPJ_INT32* mem;
  OPJ_INT32 dn;
  OPJ_INT32 sn;
  OPJ_INT32 cas;
} dwt_t;

typedef union {
  OPJ_FLOAT32  f[4];
} v4;

typedef struct v4dwt_local {
  v4*  wavelet ;
  OPJ_INT32    dn ;
  OPJ_INT32    sn ;
  OPJ_INT32    cas ;
} v4dwt_t ;

static const OPJ_FLOAT32 dwt_alpha =  1.586134342f; //  12994
static const OPJ_FLOAT32 dwt_beta  =  0.052980118f; //    434
static const OPJ_FLOAT32 dwt_gamma = -0.882911075f; //  -7233
static const OPJ_FLOAT32 delta = -0.443506852f; //  -3633

static const OPJ_FLOAT32 K      = 1.230174105f; //  10078
/* FIXME: What is this constant? */
static const OPJ_FLOAT32 c13318 = 1.625732422f;

/*@}*/

/**
Virtual function type for wavelet transform in 1-D
*/
typedef void (*DWT1DFN)(dwt_t* v);

/** @name Local static functions */
/*@{*/

/**
Forward lazy transform (horizontal)
*/
static void dwt_deinterleave_h(OPJ_INT32 *a, OPJ_INT32 *b, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 cas);
/**
Forward lazy transform (vertical)
*/
static void dwt_deinterleave_v(OPJ_INT32 *a, OPJ_INT32 *b, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 x, OPJ_INT32 cas);
/**
Inverse lazy transform (horizontal)
*/
static void dwt_interleave_h(dwt_t* h, OPJ_INT32 *a);
/**
Inverse lazy transform (vertical)
*/
static void dwt_interleave_v(dwt_t* v, OPJ_INT32 *a, OPJ_INT32 x);
/**
Forward 5-3 wavelet transform in 1-D
*/
static void dwt_encode_1(OPJ_INT32 *a, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 cas);
/**
Inverse 5-3 wavelet transform in 1-D
*/
static void dwt_decode_1(dwt_t *v);
/**
Forward 9-7 wavelet transform in 1-D
*/
static void dwt_encode_1_real(OPJ_INT32 *a, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 cas);
/**
Explicit calculation of the Quantization Stepsizes
*/
static void dwt_encode_stepsize(OPJ_INT32 stepsize, OPJ_INT32 numbps, opj_stepsize_t *bandno_stepsize);
/**
Inverse wavelet transform in 2-D.
*/
static bool dwt_decode_tile(opj_tcd_tilecomp_t* tilec, OPJ_UINT32 i, DWT1DFN fn);

static OPJ_UINT32 dwt_max_resolution(opj_tcd_resolution_t* restrict r, OPJ_UINT32 i);

static bool dwt_encode_procedure(opj_tcd_tilecomp_t * tilec,void (*p_function)(OPJ_INT32 *, OPJ_INT32,OPJ_INT32,OPJ_INT32) );
/*@}*/

/*@}*/

#define S(i) a[(i)*2]
#define D(i) a[(1+(i)*2)]
#define S_(i) ((i)<0?S(0):((i)>=sn?S(sn-1):S(i)))
#define D_(i) ((i)<0?D(0):((i)>=dn?D(dn-1):D(i)))
/* new */
#define SS_(i) ((i)<0?S(0):((i)>=dn?S(dn-1):S(i)))
#define DD_(i) ((i)<0?D(0):((i)>=sn?D(sn-1):D(i)))

/* <summary>                                                              */
/* This table contains the norms of the 5-3 wavelets for different bands. */
/* </summary>                                                             */
static const OPJ_FLOAT64 dwt_norms[4][10] = {
  {1.000, 1.500, 2.750, 5.375, 10.68, 21.34, 42.67, 85.33, 170.7, 341.3},
  {1.038, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
  {1.038, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9},
  {.7186, .9218, 1.586, 3.043, 6.019, 12.01, 24.00, 47.97, 95.93}
};

/* <summary>                                                              */
/* This table contains the norms of the 9-7 wavelets for different bands. */
/* </summary>                                                             */
static const OPJ_FLOAT64 dwt_norms_real[4][10] = {
  {1.000, 1.965, 4.177, 8.403, 16.90, 33.84, 67.69, 135.3, 270.6, 540.9},
  {2.022, 3.989, 8.355, 17.04, 34.27, 68.63, 137.3, 274.6, 549.0},
  {2.022, 3.989, 8.355, 17.04, 34.27, 68.63, 137.3, 274.6, 549.0},
  {2.080, 3.865, 8.307, 17.18, 34.71, 69.59, 139.3, 278.6, 557.2}
};

/*
==========================================================
   local functions
==========================================================
*/

/* <summary>                       */
/* Forward lazy transform (horizontal).  */
/* </summary>                            */
static void dwt_deinterleave_h(OPJ_INT32 *a, OPJ_INT32 *b, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 cas) {
  OPJ_INT32 i;

  OPJ_INT32 * l_dest = b;
  OPJ_INT32 * l_src = a+cas;
    for
    (i=0; i<sn; ++i)
  {
    *l_dest++ = *l_src;
    l_src += 2;
  }
  l_dest = b + sn;
  l_src = a + 1 - cas;
    for
    (i=0; i<dn; ++i)
  {
    *l_dest++=*l_src;
    l_src += 2;
  }
}

/* <summary>                             */
/* Forward lazy transform (vertical).    */
/* </summary>                            */
static void dwt_deinterleave_v(OPJ_INT32 *a, OPJ_INT32 *b, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 x, OPJ_INT32 cas) {
    OPJ_INT32 i = sn;
  OPJ_INT32 * l_dest = b;
  OPJ_INT32 * l_src = a+cas;

    while
    (i--)
  {
    *l_dest = *l_src;
    l_dest += x;
    l_src += 2;
    /* b[i*x]=a[2*i+cas]; */
  }
  l_dest = b + sn * x;
  l_src = a + 1 - cas;

  i = dn;
    while
    (i--)
  {
    *l_dest = *l_src;
    l_dest += x;
    l_src += 2;
    /*b[(sn+i)*x]=a[(2*i+1-cas)];*/
  }
}

/* <summary>                             */
/* Inverse lazy transform (horizontal).  */
/* </summary>                            */
static void dwt_interleave_h(dwt_t* h, OPJ_INT32 *a) {
    OPJ_INT32 *ai = a;
    OPJ_INT32 *bi = h->mem + h->cas;
    OPJ_INT32  i  = h->sn;
    while
    ( i-- )
  {
    *bi = *(ai++);
    bi += 2;
    }
    ai  = a + h->sn;
    bi  = h->mem + 1 - h->cas;
    i  = h->dn ;
    while
    ( i-- )
  {
    *bi = *(ai++);
    bi += 2;
    }
}

/* <summary>                             */
/* Inverse lazy transform (vertical).    */
/* </summary>                            */
static void dwt_interleave_v(dwt_t* v, OPJ_INT32 *a, OPJ_INT32 x) {
    OPJ_INT32 *ai = a;
    OPJ_INT32 *bi = v->mem + v->cas;
    OPJ_INT32  i = v->sn;
    while( i-- ) {
      *bi = *ai;
    bi += 2;
    ai += x;
    }
    ai = a + (v->sn * x);
    bi = v->mem + 1 - v->cas;
    i = v->dn ;
    while( i-- ) {
      *bi = *ai;
    bi += 2;
    ai += x;
    }
}


/* <summary>                            */
/* Forward 5-3 wavelet transform in 1-D. */
/* </summary>                           */
static void dwt_encode_1(OPJ_INT32 *a, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 cas) {
  OPJ_INT32 i;

  if (!cas) {
    if ((dn > 0) || (sn > 1)) {  /* NEW :  CASE ONE ELEMENT */
      for (i = 0; i < dn; i++) D(i) -= (S_(i) + S_(i + 1)) >> 1;
      for (i = 0; i < sn; i++) S(i) += (D_(i - 1) + D_(i) + 2) >> 2;
    }
  } else {
    if (!sn && dn == 1)        /* NEW :  CASE ONE ELEMENT */
      S(0) *= 2;
    else {
      for (i = 0; i < dn; i++) S(i) -= (DD_(i) + DD_(i - 1)) >> 1;
      for (i = 0; i < sn; i++) D(i) += (SS_(i) + SS_(i + 1) + 2) >> 2;
    }
  }
}

/* <summary>                            */
/* Inverse 5-3 wavelet transform in 1-D. */
/* </summary>                           */
static void dwt_decode_1_(OPJ_INT32 *a, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 cas) {
  OPJ_INT32 i;

  if (!cas) {
    if ((dn > 0) || (sn > 1)) { /* NEW :  CASE ONE ELEMENT */
      for (i = 0; i < sn; i++) S(i) -= (D_(i - 1) + D_(i) + 2) >> 2;
      for (i = 0; i < dn; i++) D(i) += (S_(i) + S_(i + 1)) >> 1;
    }
  } else {
    if (!sn  && dn == 1)          /* NEW :  CASE ONE ELEMENT */
      S(0) /= 2;
    else {
      for (i = 0; i < sn; i++) D(i) -= (SS_(i) + SS_(i + 1) + 2) >> 2;
      for (i = 0; i < dn; i++) S(i) += (DD_(i) + DD_(i - 1)) >> 1;
    }
  }
}

/* <summary>                            */
/* Inverse 5-3 wavelet transform in 1-D. */
/* </summary>                           */
static void dwt_decode_1(dwt_t *v) {
  dwt_decode_1_(v->mem, v->dn, v->sn, v->cas);
}

/* <summary>                             */
/* Forward 9-7 wavelet transform in 1-D. */
/* </summary>                            */
static void dwt_encode_1_real(OPJ_INT32 *a, OPJ_INT32 dn, OPJ_INT32 sn, OPJ_INT32 cas) {
  OPJ_INT32 i;
  if (!cas) {
    if ((dn > 0) || (sn > 1)) {  /* NEW :  CASE ONE ELEMENT */
      for (i = 0; i < dn; i++)
        D(i) -= fix_mul(S_(i) + S_(i + 1), 12993);
      for (i = 0; i < sn; i++)
        S(i) -= fix_mul(D_(i - 1) + D_(i), 434);
      for (i = 0; i < dn; i++)
        D(i) += fix_mul(S_(i) + S_(i + 1), 7233);
      for (i = 0; i < sn; i++)
        S(i) += fix_mul(D_(i - 1) + D_(i), 3633);
      for (i = 0; i < dn; i++)
        D(i) = fix_mul(D(i), 5038);  /*5038 */
      for (i = 0; i < sn; i++)
        S(i) = fix_mul(S(i), 6659);  /*6660 */
    }
  } else {
    if ((sn > 0) || (dn > 1)) {  /* NEW :  CASE ONE ELEMENT */
      for (i = 0; i < dn; i++)
        S(i) -= fix_mul(DD_(i) + DD_(i - 1), 12993);
      for (i = 0; i < sn; i++)
        D(i) -= fix_mul(SS_(i) + SS_(i + 1), 434);
      for (i = 0; i < dn; i++)
        S(i) += fix_mul(DD_(i) + DD_(i - 1), 7233);
      for (i = 0; i < sn; i++)
        D(i) += fix_mul(SS_(i) + SS_(i + 1), 3633);
      for (i = 0; i < dn; i++)
        S(i) = fix_mul(S(i), 5038);  /*5038 */
      for (i = 0; i < sn; i++)
        D(i) = fix_mul(D(i), 6659);  /*6660 */
    }
  }
}

static void dwt_encode_stepsize(OPJ_INT32 stepsize, OPJ_INT32 numbps, opj_stepsize_t *bandno_stepsize) {
  OPJ_INT32 p, n;
  p = int_floorlog2(stepsize) - 13;
  n = 11 - int_floorlog2(stepsize);
  bandno_stepsize->mant = (n < 0 ? stepsize >> -n : stepsize << n) & 0x7ff;
  bandno_stepsize->expn = numbps - p;
}

/*
==========================================================
   DWT interface
==========================================================
*/

/* <summary>                            */
/* Forward 5-3 wavelet transform in 2-D. */
/* </summary>                           */
INLINE bool dwt_encode_procedure(opj_tcd_tilecomp_t * tilec,void (*p_function)(OPJ_INT32 *, OPJ_INT32,OPJ_INT32,OPJ_INT32) )
{
  OPJ_INT32 i, j, k;
  OPJ_INT32 *a = 00;
  OPJ_INT32 *aj = 00;
  OPJ_INT32 *bj = 00;
  OPJ_INT32 w, l;

  OPJ_INT32 rw;      /* width of the resolution level computed   */
  OPJ_INT32 rh;      /* height of the resolution level computed  */
  OPJ_INT32 l_data_size;

  opj_tcd_resolution_t * l_cur_res = 0;
  opj_tcd_resolution_t * l_last_res = 0;

  w = tilec->x1-tilec->x0;
  l = tilec->numresolutions-1;
  a = tilec->data;

  l_cur_res = tilec->resolutions + l;
  l_last_res = l_cur_res - 1;

  rw = l_cur_res->x1 - l_cur_res->x0;
  rh = l_cur_res->y1 - l_cur_res->y0;

  l_data_size = dwt_max_resolution( tilec->resolutions,tilec->numresolutions) * sizeof(OPJ_INT32);
  bj = (OPJ_INT32*)opj_malloc(l_data_size);
  if
    (! bj)
  {
    return false;
  }
  i = l;

  while
    (i--)
  {
    OPJ_INT32 rw1;    /* width of the resolution level once lower than computed one                                       */
    OPJ_INT32 rh1;    /* height of the resolution level once lower than computed one                                      */
    OPJ_INT32 cas_col;  /* 0 = non inversion on horizontal filtering 1 = inversion between low-pass and high-pass filtering */
    OPJ_INT32 cas_row;  /* 0 = non inversion on vertical filtering 1 = inversion between low-pass and high-pass filtering   */
    OPJ_INT32 dn, sn;

    rw  = l_cur_res->x1 - l_cur_res->x0;
    rh  = l_cur_res->y1 - l_cur_res->y0;
    rw1 = l_last_res->x1 - l_last_res->x0;
    rh1 = l_last_res->y1 - l_last_res->y0;

    cas_row = l_cur_res->x0 & 1;
    cas_col = l_cur_res->y0 & 1;

    sn = rh1;
    dn = rh - rh1;
    for
      (j = 0; j < rw; ++j)
    {
      aj = a + j;
      for
        (k = 0; k < rh; ++k)
      {
        bj[k] = aj[k*w];
      }
      (*p_function) (bj, dn, sn, cas_col);
      dwt_deinterleave_v(bj, aj, dn, sn, w, cas_col);
    }
    sn = rw1;
    dn = rw - rw1;
    for (j = 0; j < rh; j++)
    {
      aj = a + j * w;
      for (k = 0; k < rw; k++)  bj[k] = aj[k];
      (*p_function) (bj, dn, sn, cas_row);
      dwt_deinterleave_h(bj, aj, dn, sn, cas_row);
    }
    l_cur_res = l_last_res;
    --l_last_res;
  }
  opj_free(bj);
  return true;
}
/* Forward 5-3 wavelet transform in 2-D. */
/* </summary>                           */
bool dwt_encode(opj_tcd_tilecomp_t * tilec)
{
  return dwt_encode_procedure(tilec,dwt_encode_1);
}

/* <summary>                            */
/* Inverse 5-3 wavelet transform in 2-D. */
/* </summary>                           */
bool dwt_decode(opj_tcd_tilecomp_t* tilec, OPJ_UINT32 numres) {
  return dwt_decode_tile(tilec, numres, &dwt_decode_1);
}


/* <summary>                          */
/* Get gain of 5-3 wavelet transform. */
/* </summary>                         */
OPJ_UINT32 dwt_getgain(OPJ_UINT32 orient) {
  if (orient == 0)
    return 0;
  if (orient == 1 || orient == 2)
    return 1;
  return 2;
}

/* <summary>                */
/* Get norm of 5-3 wavelet. */
/* </summary>               */
OPJ_FLOAT64 dwt_getnorm(OPJ_UINT32 level, OPJ_UINT32 orient) {
  return dwt_norms[orient][level];
}

/* <summary>                             */
/* Forward 9-7 wavelet transform in 2-D. */
/* </summary>                            */
bool dwt_encode_real(opj_tcd_tilecomp_t * tilec)
{
  return dwt_encode_procedure(tilec,dwt_encode_1_real);
}



/* <summary>                          */
/* Get gain of 9-7 wavelet transform. */
/* </summary>                         */
OPJ_UINT32 dwt_getgain_real(OPJ_UINT32 orient) {
  (void)orient;
  return 0;
}

/* <summary>                */
/* Get norm of 9-7 wavelet. */
/* </summary>               */
OPJ_FLOAT64 dwt_getnorm_real(OPJ_UINT32 level, OPJ_UINT32 orient) {
  return dwt_norms_real[orient][level];
}

void dwt_calc_explicit_stepsizes(opj_tccp_t * tccp, OPJ_UINT32 prec) {
  OPJ_UINT32 numbands, bandno;
  numbands = 3 * tccp->numresolutions - 2;
  for (bandno = 0; bandno < numbands; bandno++) {
    OPJ_FLOAT64 stepsize;
    OPJ_UINT32 resno, level, orient, gain;

    resno = (bandno == 0) ? 0 : ((bandno - 1) / 3 + 1);
    orient = (bandno == 0) ? 0 : ((bandno - 1) % 3 + 1);
    level = tccp->numresolutions - 1 - resno;
    gain = (tccp->qmfbid == 0) ? 0 : ((orient == 0) ? 0 : (((orient == 1) || (orient == 2)) ? 1 : 2));
    if (tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) {
      stepsize = 1.0;
    } else {
      OPJ_FLOAT64 norm = dwt_norms_real[orient][level];
      stepsize = (1 << (gain)) / norm;
    }
    dwt_encode_stepsize((OPJ_INT32) floor(stepsize * 8192.0), prec + gain, &tccp->stepsizes[bandno]);
  }
}


/* <summary>                             */
/* Determine maximum computed resolution level for inverse wavelet transform */
/* </summary>                            */
static OPJ_UINT32 dwt_max_resolution(opj_tcd_resolution_t* restrict r, OPJ_UINT32 i) {
  OPJ_UINT32 mr  = 0;
  OPJ_UINT32 w;
  while( --i ) {
    ++r;
    if( mr < ( w = r->x1 - r->x0 ) )
      mr = w ;
    if( mr < ( w = r->y1 - r->y0 ) )
      mr = w ;
  }
  return mr ;
}


/* <summary>                            */
/* Inverse wavelet transform in 2-D.     */
/* </summary>                           */
static bool dwt_decode_tile(opj_tcd_tilecomp_t* tilec, OPJ_UINT32 numres, DWT1DFN dwt_1D) {
  dwt_t h;
  dwt_t v;

  opj_tcd_resolution_t* tr = tilec->resolutions;

  OPJ_UINT32 rw = tr->x1 - tr->x0;  /* width of the resolution level computed */
  OPJ_UINT32 rh = tr->y1 - tr->y0;  /* height of the resolution level computed */

  OPJ_UINT32 w = tilec->x1 - tilec->x0;

  h.mem = (OPJ_INT32*)
  opj_aligned_malloc(dwt_max_resolution(tr, numres) * sizeof(OPJ_INT32));
  if
    (! h.mem)
  {
    return false;
  }

  v.mem = h.mem;

  while( --numres) {
    OPJ_INT32 * restrict tiledp = tilec->data;
    OPJ_UINT32 j;

    ++tr;
    h.sn = rw;
    v.sn = rh;

    rw = tr->x1 - tr->x0;
    rh = tr->y1 - tr->y0;

    h.dn = rw - h.sn;
    h.cas = tr->x0 % 2;

    for(j = 0; j < rh; ++j) {
      dwt_interleave_h(&h, &tiledp[j*w]);
      (dwt_1D)(&h);
      memcpy(&tiledp[j*w], h.mem, rw * sizeof(OPJ_INT32));
    }

    v.dn = rh - v.sn;
    v.cas = tr->y0 % 2;

    for(j = 0; j < rw; ++j){
      OPJ_UINT32 k;
      dwt_interleave_v(&v, &tiledp[j], w);
      (dwt_1D)(&v);
      for(k = 0; k < rh; ++k) {
        tiledp[k * w + j] = v.mem[k];
      }
    }
  }
  opj_aligned_free(h.mem);
  return true;
}

static void v4dwt_interleave_h(v4dwt_t* restrict w, OPJ_FLOAT32* restrict a, OPJ_INT32 x, OPJ_INT32 size){
  OPJ_FLOAT32* restrict bi = (OPJ_FLOAT32*) (w->wavelet + w->cas);
  OPJ_INT32 count = w->sn;
  OPJ_INT32 i, k;
  for(k = 0; k < 2; ++k){
    for(i = 0; i < count; ++i){
      OPJ_INT32 j = i;
      bi[i*8    ] = a[j];
      j += x;
      if(j >= size) continue;
      bi[i*8 + 1] = a[j];
      j += x;
      if(j >= size) continue;
      bi[i*8 + 2] = a[j];
      j += x;
      if(j >= size) continue;
      bi[i*8 + 3] = a[j];
    }
    bi = (OPJ_FLOAT32*) (w->wavelet + 1 - w->cas);
    a += w->sn;
    size -= w->sn;
    count = w->dn;
  }
}

static void v4dwt_interleave_v(v4dwt_t* restrict v , OPJ_FLOAT32* restrict a , OPJ_INT32 x){
  v4* restrict bi = v->wavelet + v->cas;
  OPJ_INT32 i;
  for(i = 0; i < v->sn; ++i){
    memcpy(&bi[i*2], &a[i*x], 4 * sizeof(OPJ_FLOAT32));
  }
  a += v->sn * x;
  bi = v->wavelet + 1 - v->cas;
  for(i = 0; i < v->dn; ++i){
    memcpy(&bi[i*2], &a[i*x], 4 * sizeof(OPJ_FLOAT32));
  }
}

#ifdef __SSE__

static void v4dwt_decode_step1_sse(v4* w, OPJ_INT32 count, const __m128 c){
  __m128* restrict vw = (__m128*) w;
  OPJ_INT32 i;
  for(i = 0; i < count; ++i){
    __m128 tmp = vw[i*2];
    vw[i*2] = tmp * c;
  }
}

static void v4dwt_decode_step2_sse(v4* l, v4* w, OPJ_INT32 k, OPJ_INT32 m, __m128 c){
  __m128* restrict vl = (__m128*) l;
  __m128* restrict vw = (__m128*) w;
  OPJ_INT32 i;
  for(i = 0; i < m; ++i){
    __m128 tmp1 = vl[ 0];
    __m128 tmp2 = vw[-1];
    __m128 tmp3 = vw[ 0];
    vw[-1] = tmp2 + ((tmp1 + tmp3) * c);
    vl = vw;
    vw += 2;
  }
  if(m >= k){
    return;
  }
  c += c;
  c *= vl[0];
  for(; m < k; ++m){
    __m128 tmp = vw[-1];
    vw[-1] = tmp + c;
    vw += 2;
  }
}

#else

static void v4dwt_decode_step1(v4* w, OPJ_INT32 count, const OPJ_FLOAT32 c){
  OPJ_FLOAT32* restrict fw = (OPJ_FLOAT32*) w;
  OPJ_INT32 i;
  for(i = 0; i < count; ++i){
    OPJ_FLOAT32 tmp1 = fw[i*8    ];
    OPJ_FLOAT32 tmp2 = fw[i*8 + 1];
    OPJ_FLOAT32 tmp3 = fw[i*8 + 2];
    OPJ_FLOAT32 tmp4 = fw[i*8 + 3];
    fw[i*8    ] = tmp1 * c;
    fw[i*8 + 1] = tmp2 * c;
    fw[i*8 + 2] = tmp3 * c;
    fw[i*8 + 3] = tmp4 * c;
  }
}

static void v4dwt_decode_step2(v4* l, v4* w, OPJ_INT32 k, OPJ_INT32 m, OPJ_FLOAT32 c){
  OPJ_FLOAT32* restrict fl = (OPJ_FLOAT32*) l;
  OPJ_FLOAT32* restrict fw = (OPJ_FLOAT32*) w;
  OPJ_INT32 i;
  for(i = 0; i < m; ++i){
    OPJ_FLOAT32 tmp1_1 = fl[0];
    OPJ_FLOAT32 tmp1_2 = fl[1];
    OPJ_FLOAT32 tmp1_3 = fl[2];
    OPJ_FLOAT32 tmp1_4 = fl[3];
    OPJ_FLOAT32 tmp2_1 = fw[-4];
    OPJ_FLOAT32 tmp2_2 = fw[-3];
    OPJ_FLOAT32 tmp2_3 = fw[-2];
    OPJ_FLOAT32 tmp2_4 = fw[-1];
    OPJ_FLOAT32 tmp3_1 = fw[0];
    OPJ_FLOAT32 tmp3_2 = fw[1];
    OPJ_FLOAT32 tmp3_3 = fw[2];
    OPJ_FLOAT32 tmp3_4 = fw[3];
    fw[-4] = tmp2_1 + ((tmp1_1 + tmp3_1) * c);
    fw[-3] = tmp2_2 + ((tmp1_2 + tmp3_2) * c);
    fw[-2] = tmp2_3 + ((tmp1_3 + tmp3_3) * c);
    fw[-1] = tmp2_4 + ((tmp1_4 + tmp3_4) * c);
    fl = fw;
    fw += 8;
  }
  if(m < k){
    OPJ_FLOAT32 c1;
    OPJ_FLOAT32 c2;
    OPJ_FLOAT32 c3;
    OPJ_FLOAT32 c4;
    c += c;
    c1 = fl[0] * c;
    c2 = fl[1] * c;
    c3 = fl[2] * c;
    c4 = fl[3] * c;
    for(; m < k; ++m){
      OPJ_FLOAT32 tmp1 = fw[-4];
      OPJ_FLOAT32 tmp2 = fw[-3];
      OPJ_FLOAT32 tmp3 = fw[-2];
      OPJ_FLOAT32 tmp4 = fw[-1];
      fw[-4] = tmp1 + c1;
      fw[-3] = tmp2 + c2;
      fw[-2] = tmp3 + c3;
      fw[-1] = tmp4 + c4;
      fw += 8;
    }
  }
}

#endif

/* <summary>                             */
/* Inverse 9-7 wavelet transform in 1-D. */
/* </summary>                            */
static void v4dwt_decode(v4dwt_t* restrict dwt){
  OPJ_INT32 a, b;
  if(dwt->cas == 0) {
    if(!((dwt->dn > 0) || (dwt->sn > 1))){
      return;
    }
    a = 0;
    b = 1;
  }else{
    if(!((dwt->sn > 0) || (dwt->dn > 1))) {
      return;
    }
    a = 1;
    b = 0;
  }
#ifdef __SSE__
  v4dwt_decode_step1_sse(dwt->wavelet+a, dwt->sn, _mm_set1_ps(K));
  v4dwt_decode_step1_sse(dwt->wavelet+b, dwt->dn, _mm_set1_ps(c13318));
  v4dwt_decode_step2_sse(dwt->wavelet+b, dwt->wavelet+a+1, dwt->sn, int_min(dwt->sn, dwt->dn-a), _mm_set1_ps(delta));
  v4dwt_decode_step2_sse(dwt->wavelet+a, dwt->wavelet+b+1, dwt->dn, int_min(dwt->dn, dwt->sn-b), _mm_set1_ps(dwt_gamma));
  v4dwt_decode_step2_sse(dwt->wavelet+b, dwt->wavelet+a+1, dwt->sn, int_min(dwt->sn, dwt->dn-a), _mm_set1_ps(dwt_beta));
  v4dwt_decode_step2_sse(dwt->wavelet+a, dwt->wavelet+b+1, dwt->dn, int_min(dwt->dn, dwt->sn-b), _mm_set1_ps(dwt_alpha));
#else
  v4dwt_decode_step1(dwt->wavelet+a, dwt->sn, K);
  v4dwt_decode_step1(dwt->wavelet+b, dwt->dn, c13318);
  v4dwt_decode_step2(dwt->wavelet+b, dwt->wavelet+a+1, dwt->sn, int_min(dwt->sn, dwt->dn-a), delta);
  v4dwt_decode_step2(dwt->wavelet+a, dwt->wavelet+b+1, dwt->dn, int_min(dwt->dn, dwt->sn-b), dwt_gamma);
  v4dwt_decode_step2(dwt->wavelet+b, dwt->wavelet+a+1, dwt->sn, int_min(dwt->sn, dwt->dn-a), dwt_beta);
  v4dwt_decode_step2(dwt->wavelet+a, dwt->wavelet+b+1, dwt->dn, int_min(dwt->dn, dwt->sn-b), dwt_alpha);
#endif
}

/* <summary>                             */
/* Inverse 9-7 wavelet transform in 2-D. */
/* </summary>                            */
bool dwt_decode_real(opj_tcd_tilecomp_t* restrict tilec, OPJ_UINT32 numres){
  v4dwt_t h;
  v4dwt_t v;

  opj_tcd_resolution_t* res = tilec->resolutions;

  OPJ_UINT32 rw = res->x1 - res->x0;  /* width of the resolution level computed */
  OPJ_UINT32 rh = res->y1 - res->y0;  /* height of the resolution level computed */

  OPJ_UINT32 w = tilec->x1 - tilec->x0;

  h.wavelet = (v4*) opj_aligned_malloc((dwt_max_resolution(res, numres)+5) * sizeof(v4));
  v.wavelet = h.wavelet;

  while( --numres) {
    OPJ_FLOAT32 * restrict aj = (OPJ_FLOAT32*) tilec->data;
    OPJ_UINT32 bufsize = (tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0);
    OPJ_INT32 j;

    h.sn = rw;
    v.sn = rh;

    ++res;

    rw = res->x1 - res->x0;  /* width of the resolution level computed */
    rh = res->y1 - res->y0;  /* height of the resolution level computed */

    h.dn = rw - h.sn;
    h.cas = res->x0 & 1;

    for(j = rh; j > 0; j -= 4){
      v4dwt_interleave_h(&h, aj, w, bufsize);
      v4dwt_decode(&h);
      if(j >= 4){
        OPJ_INT32 k = rw;
        while
          (--k >= 0)
        {
          aj[k    ] = h.wavelet[k].f[0];
          aj[k+w  ] = h.wavelet[k].f[1];
          aj[k+w*2] = h.wavelet[k].f[2];
          aj[k+w*3] = h.wavelet[k].f[3];
        }
      }else{
        OPJ_INT32 k = rw;
        while
          (--k >= 0)
        {
          switch(j) {
            case 3: aj[k+w*2] = h.wavelet[k].f[2];
            case 2: aj[k+w  ] = h.wavelet[k].f[1];
            case 1: aj[k    ] = h.wavelet[k].f[0];
          }
        }
      }
      aj += w*4;
      bufsize -= w*4;
    }

    v.dn = rh - v.sn;
    v.cas = res->y0 % 2;

    aj = (OPJ_FLOAT32*) tilec->data;
    for(j = rw; j > 0; j -= 4){
      v4dwt_interleave_v(&v, aj, w);
      v4dwt_decode(&v);
      if(j >= 4){
        OPJ_UINT32 k;
        for(k = 0; k < rh; ++k){
          memcpy(&aj[k*w], &v.wavelet[k], 4 * sizeof(OPJ_FLOAT32));
        }
      }else{
        OPJ_UINT32 k;
        for(k = 0; k < rh; ++k){
          memcpy(&aj[k*w], &v.wavelet[k], j * sizeof(OPJ_FLOAT32));
        }
      }
      aj += 4;
    }
  }

  opj_aligned_free(h.wavelet);
  return true;
}
