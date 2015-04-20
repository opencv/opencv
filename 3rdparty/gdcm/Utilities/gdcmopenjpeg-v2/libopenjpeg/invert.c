/*
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

#include "invert.h"
#include "opj_malloc.h"


bool opj_lupDecompose(OPJ_FLOAT32 * matrix,OPJ_UINT32 * permutations, OPJ_FLOAT32 * p_swap_area,OPJ_UINT32 n);
void opj_lupSolve(OPJ_FLOAT32 * pResult, OPJ_FLOAT32* pMatrix, OPJ_FLOAT32* pVector, OPJ_UINT32* pPermutations, OPJ_UINT32 n,OPJ_FLOAT32 * p_intermediate_data);
void opj_lupInvert (OPJ_FLOAT32 * pSrcMatrix,
           OPJ_FLOAT32 * pDestMatrix,
           OPJ_UINT32 n,
           OPJ_UINT32 * pPermutations,
           OPJ_FLOAT32 * p_src_temp,
           OPJ_FLOAT32 * p_dest_temp,
           OPJ_FLOAT32 * p_swap_area);

/**
 * Matrix inversion.
 */
bool opj_matrix_inversion_f(OPJ_FLOAT32 * pSrcMatrix,OPJ_FLOAT32 * pDestMatrix, OPJ_UINT32 n)
{
  OPJ_BYTE * l_data = 00;
  OPJ_UINT32 l_permutation_size = n * sizeof(OPJ_UINT32);
  OPJ_UINT32 l_swap_size = n * sizeof(OPJ_FLOAT32);
  OPJ_UINT32 l_total_size = l_permutation_size + 3 * l_swap_size;
  OPJ_UINT32 * lPermutations = 00;
  OPJ_FLOAT32 * l_double_data = 00;

  l_data = (OPJ_BYTE *) opj_malloc(l_total_size);
  if
    (l_data == 0)
  {
    return false;
  }
  lPermutations = (OPJ_UINT32 *) l_data;
  l_double_data = (OPJ_FLOAT32 *) (l_data + l_permutation_size);
  memset(lPermutations,0,l_permutation_size);

  if
    (! opj_lupDecompose(pSrcMatrix,lPermutations,l_double_data,n))
  {
    opj_free(l_data);
    return false;
  }
  opj_lupInvert(pSrcMatrix,pDestMatrix,n,lPermutations,l_double_data,l_double_data + n,l_double_data + 2*n);
  opj_free(l_data);
  return true;
}


/**
 * LUP decomposition
 */
bool opj_lupDecompose(OPJ_FLOAT32 * matrix,OPJ_UINT32 * permutations, OPJ_FLOAT32 * p_swap_area,OPJ_UINT32 n)
{
  OPJ_UINT32 * tmpPermutations = permutations;
  OPJ_UINT32 * dstPermutations;
  OPJ_UINT32 k2=0,t;
  OPJ_FLOAT32 temp;
  OPJ_UINT32 i,j,k;
  OPJ_FLOAT32 p;
  OPJ_UINT32 lLastColum = n - 1;
  OPJ_UINT32 lSwapSize = n * sizeof(OPJ_FLOAT32);
  OPJ_FLOAT32 * lTmpMatrix = matrix;
  OPJ_FLOAT32 * lColumnMatrix,* lDestMatrix;
  OPJ_UINT32 offset = 1;
  OPJ_UINT32 lStride = n-1;

  //initialize permutations
  for
    (i = 0; i < n; ++i)
  {
      *tmpPermutations++ = i;
  }



  // now make a pivot with colum switch
  tmpPermutations = permutations;
  for
    (k = 0; k < lLastColum; ++k)
  {
    p = 0.0;

    // take the middle element
    lColumnMatrix = lTmpMatrix + k;

    // make permutation with the biggest value in the column
    for
      (i = k; i < n; ++i)
    {
      temp = ((*lColumnMatrix > 0) ? *lColumnMatrix : -(*lColumnMatrix));
         if
        (temp > p)
      {
           p = temp;
           k2 = i;
         }
      // next line
      lColumnMatrix += n;
       }

       // a whole rest of 0 -> non singular
       if
      (p == 0.0)
    {
        return false;
    }

    // should we permute ?
    if
      (k2 != k)
    {
      //exchange of line
         // k2 > k
      dstPermutations = tmpPermutations + k2 - k;
      // swap indices
      t = *tmpPermutations;
         *tmpPermutations = *dstPermutations;
         *dstPermutations = t;

      // and swap entire line.
      lColumnMatrix = lTmpMatrix + (k2 - k) * n;
      memcpy(p_swap_area,lColumnMatrix,lSwapSize);
      memcpy(lColumnMatrix,lTmpMatrix,lSwapSize);
      memcpy(lTmpMatrix,p_swap_area,lSwapSize);
    }

    // now update data in the rest of the line and line after
    lDestMatrix = lTmpMatrix + k;
    lColumnMatrix = lDestMatrix + n;
    // take the middle element
    temp = *(lDestMatrix++);

    // now compute up data (i.e. coeff up of the diagonal).
       for (i = offset; i < n; ++i)
    {
      //lColumnMatrix;
      // divide the lower column elements by the diagonal value

      // matrix[i][k] /= matrix[k][k];
         // p = matrix[i][k]
      p = *lColumnMatrix / temp;
      *(lColumnMatrix++) = p;
         for
        (j = /* k + 1 */ offset; j < n; ++j)
      {
        // matrix[i][j] -= matrix[i][k] * matrix[k][j];
           *(lColumnMatrix++) -= p * (*(lDestMatrix++));
      }
      // come back to the k+1th element
      lDestMatrix -= lStride;
      // go to kth element of the next line
      lColumnMatrix += k;
       }
    // offset is now k+2
    ++offset;
    // 1 element less for stride
    --lStride;
    // next line
    lTmpMatrix+=n;
    // next permutation element
    ++tmpPermutations;
  }
    return true;
}



/**
 * LUP solving
 */
void opj_lupSolve (OPJ_FLOAT32 * pResult, OPJ_FLOAT32 * pMatrix, OPJ_FLOAT32 * pVector, OPJ_UINT32* pPermutations, OPJ_UINT32 n,OPJ_FLOAT32 * p_intermediate_data)
{
  OPJ_UINT32 i,j;
  OPJ_FLOAT32 sum;
  OPJ_FLOAT32 u;
    OPJ_UINT32 lStride = n+1;
  OPJ_FLOAT32 * lCurrentPtr;
  OPJ_FLOAT32 * lIntermediatePtr;
  OPJ_FLOAT32 * lDestPtr;
  OPJ_FLOAT32 * lTmpMatrix;
  OPJ_FLOAT32 * lLineMatrix = pMatrix;
  OPJ_FLOAT32 * lBeginPtr = pResult + n - 1;
  OPJ_FLOAT32 * lGeneratedData;
  OPJ_UINT32 * lCurrentPermutationPtr = pPermutations;


  lIntermediatePtr = p_intermediate_data;
  lGeneratedData = p_intermediate_data + n - 1;

    for
    (i = 0; i < n; ++i)
  {
         sum = 0.0;
    lCurrentPtr = p_intermediate_data;
    lTmpMatrix = lLineMatrix;
        for
      (j = 1; j <= i; ++j)
    {
      // sum += matrix[i][j-1] * y[j-1];
          sum += (*(lTmpMatrix++)) * (*(lCurrentPtr++));
        }
    //y[i] = pVector[pPermutations[i]] - sum;
        *(lIntermediatePtr++) = pVector[*(lCurrentPermutationPtr++)] - sum;
    lLineMatrix += n;
  }

  // we take the last point of the matrix
  lLineMatrix = pMatrix + n*n - 1;

  // and we take after the last point of the destination vector
  lDestPtr = pResult + n;

  for
    (i = n - 1; i != -1 ; --i)
  {
    sum = 0.0;
    lTmpMatrix = lLineMatrix;
        u = *(lTmpMatrix++);
    lCurrentPtr = lDestPtr--;
        for
      (j = i + 1; j < n; ++j)
    {
      // sum += matrix[i][j] * x[j]
          sum += (*(lTmpMatrix++)) * (*(lCurrentPtr++));
    }
    //x[i] = (y[i] - sum) / u;
        *(lBeginPtr--) = (*(lGeneratedData--) - sum) / u;
    lLineMatrix -= lStride;
  }
}

/** LUP inversion (call with the result of lupDecompose)
 */
void opj_lupInvert (
           OPJ_FLOAT32 * pSrcMatrix,
           OPJ_FLOAT32 * pDestMatrix,
           OPJ_UINT32 n,
           OPJ_UINT32 * pPermutations,
           OPJ_FLOAT32 * p_src_temp,
           OPJ_FLOAT32 * p_dest_temp,
           OPJ_FLOAT32 * p_swap_area
           )
{
  OPJ_UINT32 j,i;
  OPJ_FLOAT32 * lCurrentPtr;
  OPJ_FLOAT32 * lLineMatrix = pDestMatrix;
  OPJ_UINT32 lSwapSize = n * sizeof(OPJ_FLOAT32);

  for
    (j = 0; j < n; ++j)
  {
    lCurrentPtr = lLineMatrix++;
        memset(p_src_temp,0,lSwapSize);
      p_src_temp[j] = 1.0;
    opj_lupSolve(p_dest_temp,pSrcMatrix,p_src_temp, pPermutations, n , p_swap_area);

    for
      (i = 0; i < n; ++i)
    {
        *(lCurrentPtr) = p_dest_temp[i];
      lCurrentPtr+=n;
      }
    }
}
