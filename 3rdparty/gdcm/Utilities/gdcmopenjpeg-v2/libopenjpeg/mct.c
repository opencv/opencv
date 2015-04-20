/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
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

#include "mct.h"
#include "fix.h"
#include "opj_malloc.h"

/* <summary> */
/* This table contains the norms of the basis function of the reversible MCT. */
/* </summary> */
static const OPJ_FLOAT64 mct_norms[3] = { 1.732, .8292, .8292 };

/* <summary> */
/* This table contains the norms of the basis function of the irreversible MCT. */
/* </summary> */
static const OPJ_FLOAT64 mct_norms_real[3] = { 1.732, 1.805, 1.573 };



const OPJ_FLOAT64 * get_mct_norms ()
{
  return mct_norms;
}

const OPJ_FLOAT64 * get_mct_norms_real ()
{
  return mct_norms_real;
}



/* <summary> */
/* Foward reversible MCT. */
/* </summary> */
void mct_encode(
    OPJ_INT32* restrict c0,
    OPJ_INT32* restrict c1,
    OPJ_INT32* restrict c2,
    OPJ_UINT32 n)
{
  OPJ_UINT32 i;
  for(i = 0; i < n; ++i) {
    OPJ_INT32 r = c0[i];
    OPJ_INT32 g = c1[i];
    OPJ_INT32 b = c2[i];
    OPJ_INT32 y = (r + (g * 2) + b) >> 2;
    OPJ_INT32 u = b - g;
    OPJ_INT32 v = r - g;
    c0[i] = y;
    c1[i] = u;
    c2[i] = v;
  }
}

/* <summary> */
/* Inverse reversible MCT. */
/* </summary> */
void mct_decode(
    OPJ_INT32* restrict c0,
    OPJ_INT32* restrict c1,
    OPJ_INT32* restrict c2,
    OPJ_UINT32 n)
{
  OPJ_UINT32 i;
  for (i = 0; i < n; ++i) {
    OPJ_INT32 y = c0[i];
    OPJ_INT32 u = c1[i];
    OPJ_INT32 v = c2[i];
    OPJ_INT32 g = y - ((u + v) >> 2);
    OPJ_INT32 r = v + g;
    OPJ_INT32 b = u + g;
    c0[i] = r;
    c1[i] = g;
    c2[i] = b;
  }
}

/* <summary> */
/* Get norm of basis function of reversible MCT. */
/* </summary> */
OPJ_FLOAT64 mct_getnorm(OPJ_UINT32 compno) {
  return mct_norms[compno];
}

/* <summary> */
/* Foward irreversible MCT. */
/* </summary> */
void mct_encode_real(
    OPJ_INT32* restrict c0,
    OPJ_INT32* restrict c1,
    OPJ_INT32* restrict c2,
    OPJ_UINT32 n)
{
  OPJ_UINT32 i;
  for(i = 0; i < n; ++i) {
    OPJ_INT32 r = c0[i];
    OPJ_INT32 g = c1[i];
    OPJ_INT32 b = c2[i];
    OPJ_INT32 y =  fix_mul(r, 2449) + fix_mul(g, 4809) + fix_mul(b, 934);
    OPJ_INT32 u = -fix_mul(r, 1382) - fix_mul(g, 2714) + fix_mul(b, 4096);
    OPJ_INT32 v =  fix_mul(r, 4096) - fix_mul(g, 3430) - fix_mul(b, 666);
    c0[i] = y;
    c1[i] = u;
    c2[i] = v;
  }
}

/* <summary> */
/* Inverse irreversible MCT. */
/* </summary> */
void mct_decode_real(
    OPJ_FLOAT32* restrict c0,
    OPJ_FLOAT32* restrict c1,
    OPJ_FLOAT32* restrict c2,
    OPJ_UINT32 n)
{
  OPJ_UINT32 i;
  for(i = 0; i < n; ++i) {
    OPJ_FLOAT32 y = c0[i];
    OPJ_FLOAT32 u = c1[i];
    OPJ_FLOAT32 v = c2[i];
    OPJ_FLOAT32 r = y + (v * 1.402f);
    OPJ_FLOAT32 g = y - (u * 0.34413f) - (v * (0.71414f));
    OPJ_FLOAT32 b = y + (u * 1.772f);
    c0[i] = r;
    c1[i] = g;
    c2[i] = b;
  }
}

/* <summary> */
/* Get norm of basis function of irreversible MCT. */
/* </summary> */
OPJ_FLOAT64 mct_getnorm_real(OPJ_UINT32 compno) {
  return mct_norms_real[compno];
}

bool mct_encode_custom(
             // MCT data
             OPJ_BYTE * pCodingdata,
             // size of components
             OPJ_UINT32 n,
             // components
             OPJ_BYTE ** pData,
             // nb of components (i.e. size of pData)
             OPJ_UINT32 pNbComp,
             // tells if the data is signed
             OPJ_UINT32 isSigned)
{
  OPJ_FLOAT32 * lMct = (OPJ_FLOAT32 *) pCodingdata;
  OPJ_UINT32 i;
  OPJ_UINT32 j;
  OPJ_UINT32 k;
  OPJ_UINT32 lNbMatCoeff = pNbComp * pNbComp;
  OPJ_INT32 * lCurrentData = 00;
  OPJ_INT32 * lCurrentMatrix = 00;
  OPJ_INT32 ** lData = (OPJ_INT32 **) pData;
  OPJ_UINT32 lMultiplicator = 1 << 13;
  OPJ_INT32 * lMctPtr;

  lCurrentData = (OPJ_INT32 *) opj_malloc((pNbComp + lNbMatCoeff) * sizeof(OPJ_INT32));
  if
    (! lCurrentData)
  {
    return false;
  }
  lCurrentMatrix = lCurrentData + pNbComp;
  for
    (i =0;i<lNbMatCoeff;++i)
  {
    lCurrentMatrix[i] = (OPJ_INT32) (*(lMct++) * lMultiplicator);
  }
  for
    (i = 0; i < n; ++i)
  {
    lMctPtr = lCurrentMatrix;
    for
      (j=0;j<pNbComp;++j)
    {
      lCurrentData[j] = (*(lData[j]));
    }
    for
      (j=0;j<pNbComp;++j)
    {
      *(lData[j]) = 0;
      for
        (k=0;k<pNbComp;++k)
      {
        *(lData[j]) += fix_mul(*lMctPtr, lCurrentData[k]);
        ++lMctPtr;
      }
      ++lData[j];
    }
  }
  opj_free(lCurrentData);
  return true;
}

bool mct_decode_custom(
             // MCT data
             OPJ_BYTE * pDecodingData,
             // size of components
             OPJ_UINT32 n,
             // components
             OPJ_BYTE ** pData,
             // nb of components (i.e. size of pData)
             OPJ_UINT32 pNbComp,
             // tells if the data is signed
             OPJ_UINT32 isSigned)
{
  OPJ_FLOAT32 * lMct;
  OPJ_UINT32 i;
  OPJ_UINT32 j;
  OPJ_UINT32 k;

  OPJ_FLOAT32 * lCurrentData = 00;
  OPJ_FLOAT32 * lCurrentResult = 00;
  OPJ_FLOAT32 ** lData = (OPJ_FLOAT32 **) pData;

  lCurrentData = (OPJ_FLOAT32 *) opj_malloc (2 * pNbComp * sizeof(OPJ_FLOAT32));
  if
    (! lCurrentData)
  {
    return false;
  }
  lCurrentResult = lCurrentData + pNbComp;

  for
    (i = 0; i < n; ++i)
  {
    lMct = (OPJ_FLOAT32 *) pDecodingData;
    for
      (j=0;j<pNbComp;++j)
    {
      lCurrentData[j] = (OPJ_FLOAT32) (*(lData[j]));
    }
    for
      (j=0;j<pNbComp;++j)
    {
      lCurrentResult[j] = 0;
      for
        (k=0;k<pNbComp;++k)
      {
        lCurrentResult[j] += *(lMct++) * lCurrentData[k];
      }
      *(lData[j]++) = (OPJ_FLOAT32) (lCurrentResult[j]);
    }
  }
  opj_free(lCurrentData);
  return true;
}

void opj_calculate_norms(OPJ_FLOAT64 * pNorms,OPJ_UINT32 pNbComps,OPJ_FLOAT32 * pMatrix)
{
  OPJ_UINT32 i,j,lIndex;
  OPJ_FLOAT32 lCurrentValue;
  OPJ_FLOAT64 * lNorms = (OPJ_FLOAT64 *) pNorms;
  OPJ_FLOAT32 * lMatrix = (OPJ_FLOAT32 *) pMatrix;

  for
    (i=0;i<pNbComps;++i)
  {
    lNorms[i] = 0;
    lIndex = i;
    for
      (j=0;j<pNbComps;++j)
    {
      lCurrentValue = lMatrix[lIndex];
      lIndex += pNbComps;
      lNorms[i] += lCurrentValue * lCurrentValue;
    }
    lNorms[i] = sqrt(lNorms[i]);
  }
}
