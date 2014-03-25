////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability, or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef OPENCV_TRANSPOSE_HPP_
#define OPENCV_TRANSPOSE_HPP_

template <typename InputScalar, typename OutputScalar>
void transposeBlock(const size_t M, const size_t N, const InputScalar* src, size_t lda, OutputScalar* dst, size_t ldb) {
  InputScalar cache[16];
  // copy the source into the cache contiguously
  for (size_t n = 0; n < N; ++n)
    for (size_t m = 0; m < M; ++m)
      cache[m+n*4] = src[m+n*lda];
  // copy the destination out of the cache contiguously
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n)
      dst[n+m*ldb] = cache[m+n*4];
}

template <typename InputScalar, typename OutputScalar>
void transpose4x4(const InputScalar* src, size_t lda, OutputScalar* dst, size_t ldb) {
  InputScalar cache[16];
  // copy the source into the cache contiguously
  cache[0] = src[0];  cache[1] = src[1];  cache[2] = src[2];  cache[3] = src[3];  src+=lda;
  cache[4] = src[0];  cache[5] = src[1];  cache[6] = src[2];  cache[7] = src[3];  src+=lda;
  cache[8] = src[0];  cache[9] = src[1];  cache[10] = src[2]; cache[11] = src[3]; src+=lda;
  cache[12] = src[0]; cache[13] = src[1]; cache[14] = src[2]; cache[15] = src[3]; src+=lda;
  // copy the destination out of the contiguously
  dst[0] = cache[0];  dst[1] = cache[4];  dst[2] = cache[8];   dst[3] = cache[12]; dst+=ldb;
  dst[0] = cache[1];  dst[1] = cache[5];  dst[2] = cache[9];   dst[3] = cache[13]; dst+=ldb;
  dst[0] = cache[2];  dst[1] = cache[6];  dst[2] = cache[10];  dst[3] = cache[14]; dst+=ldb;
  dst[0] = cache[3];  dst[1] = cache[7];  dst[2] = cache[11];  dst[3] = cache[15]; dst+=ldb;
}


/*
 * Vanilla copy, transpose and cast
 */
template <typename InputScalar, typename OutputScalar>
void gemt(const char major, const size_t M, const size_t N, const InputScalar* a, size_t lda, OutputScalar* b, size_t ldb) {

  // 1x1 transpose is just copy
  if (M == 1 && N == 1) { *b = *a; return; }

  // get the interior 4x4 blocks, and the extra skirting
  const size_t Fblock = (major == 'R') ? N/4 : M/4;
  const size_t Frem   = (major == 'R') ? N%4 : M%4;
  const size_t Sblock = (major == 'R') ? M/4 : N/4;
  const size_t Srem   = (major == 'R') ? M%4 : N%4;

  // if less than 4x4, invoke the block transpose immediately
  if (M < 4 && N < 4) { transposeBlock(Frem, Srem, a, lda, b, ldb); return; }

  // transpose 4x4 blocks
  const InputScalar* aptr = a;
  OutputScalar* bptr = b;
  for (size_t second = 0; second < Sblock; ++second) {
    aptr = a + second*lda;
    bptr = b + second;
    for (size_t first = 0; first < Fblock; ++first) {
      transposeBlock(4, 4, aptr, lda, bptr, ldb);
      //transpose4x4(aptr, lda, bptr, ldb);
      aptr+=4;
      bptr+=4*ldb;
    }
    // transpose trailing blocks on primary dimension
    transposeBlock(Frem, 4, aptr, lda, bptr, ldb);
  }
  // transpose trailing blocks on secondary dimension
  aptr = a + 4*Sblock*lda;
  bptr = b + 4*Sblock;
  for (size_t first = 0; first < Fblock; ++first) {
    transposeBlock(4, Srem, aptr, lda, bptr, ldb);
    aptr+=4;
    bptr+=4*ldb;
  }
  // transpose bottom right-hand corner
  transposeBlock(Frem, Srem, aptr, lda, bptr, ldb);
}

#ifdef __SSE2__
/*
 * SSE2 supported fast copy, transpose and cast
 */
#include <emmintrin.h>

template <>
void transpose4x4<float, float>(const float* src, size_t lda, float* dst, size_t ldb) {
  __m128 row0, row1, row2, row3;
  row0 = _mm_loadu_ps(src);
  row1 = _mm_loadu_ps(src+lda);
  row2 = _mm_loadu_ps(src+2*lda);
  row3 = _mm_loadu_ps(src+3*lda);
  _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
  _mm_storeu_ps(dst, row0);
  _mm_storeu_ps(dst+ldb, row1);
  _mm_storeu_ps(dst+2*ldb, row2);
  _mm_storeu_ps(dst+3*ldb, row3);
}

#endif
#endif
