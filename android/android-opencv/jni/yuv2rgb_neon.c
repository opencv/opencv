/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include "yuv420sp2rgb.h"
#include <arm_neon.h>
#include <stdlib.h>

/* this source file should only be compiled by Android.mk when targeting
 * the armeabi-v7a ABI, and should be built in NEON mode
 */
void fir_filter_neon_intrinsics(short *output, const short* input, const short* kernel, int width, int kernelSize)
{
#if 1
  int nn, offset = -kernelSize / 2;

  for (nn = 0; nn < width; nn++)
  {
    int mm, sum = 0;
    int32x4_t sum_vec = vdupq_n_s32(0);
    for (mm = 0; mm < kernelSize / 4; mm++)
    {
      int16x4_t kernel_vec = vld1_s16(kernel + mm * 4);
      int16x4_t input_vec = vld1_s16(input + (nn + offset + mm * 4));
      sum_vec = vmlal_s16(sum_vec, kernel_vec, input_vec);
    }

    sum += vgetq_lane_s32(sum_vec, 0);
    sum += vgetq_lane_s32(sum_vec, 1);
    sum += vgetq_lane_s32(sum_vec, 2);
    sum += vgetq_lane_s32(sum_vec, 3);

    if (kernelSize & 3)
    {
      for (mm = kernelSize - (kernelSize & 3); mm < kernelSize; mm++)
        sum += kernel[mm] * input[nn + offset + mm];
    }

    output[nn] = (short)((sum + 0x8000) >> 16);
  }
#else /* for comparison purposes only */
  int nn, offset = -kernelSize/2;
  for (nn = 0; nn < width; nn++)
  {
    int sum = 0;
    int mm;
    for (mm = 0; mm < kernelSize; mm++)
    {
      sum += kernel[mm]*input[nn+offset+mm];
    }
    output[n] = (short)((sum + 0x8000) >> 16);
  }
#endif
}

/*
 YUV 4:2:0 image with a plane of 8 bit Y samples followed by an interleaved
 U/V plane containing 8 bit 2x2 subsampled chroma samples.
 except the interleave order of U and V is reversed.

 H V
 Y Sample Period      1 1
 U (Cb) Sample Period 2 2
 V (Cr) Sample Period 2 2
 */

/*
 size of a char:
 find . -name limits.h -exec grep CHAR_BIT {} \;
 */

#ifndef max
#define max(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a < _b ? _a : _b; })
#endif

#define bytes_per_pixel 2
#define LOAD_Y(i,j) (pY + i * width + j)
#define LOAD_V(i,j) (pUV + (i / 2) * width + bytes_per_pixel * (j / 2))
#define LOAD_U(i,j) (pUV + (i / 2) * width + bytes_per_pixel * (j / 2)+1)

const uint8_t ZEROS[8] = {220,220, 220, 220, 220, 220, 220, 220};
const uint8_t Y_SUBS[8] = {16, 16, 16, 16, 16, 16, 16, 16};
const uint8_t UV_SUBS[8] = {128, 128, 128, 128, 128, 128, 128, 128};

const uint32_t UV_MULS[] = {833, 400, 833, 400};

void color_convert_common(unsigned char *pY, unsigned char *pUV, int width, int height, unsigned char *buffer, int grey)
{

  int i, j;
  int nR, nG, nB;
  int nY, nU, nV;
  unsigned char *out = buffer;
  int offset = 0;

  uint8x8_t Y_SUBvec = vld1_u8(Y_SUBS);
  uint8x8_t UV_SUBvec = vld1_u8(UV_SUBS); // v,u,v,u v,u,v,u
  uint32x4_t UV_MULSvec = vld1q_u32(UV_MULS);
  uint8x8_t ZEROSvec =vld1_u8(ZEROS);

  uint32_t UVvec_int[8];
  if (grey)
  {
    memcpy(out, pY, width * height * sizeof(unsigned char));
  }
  else
    // YUV 4:2:0
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j += 8)
      {
        //        nY = *(pY + i * width + j);
        //        nV = *(pUV + (i / 2) * width + bytes_per_pixel * (j / 2));
        //        nU = *(pUV + (i / 2) * width + bytes_per_pixel * (j / 2) + 1);

        uint8x8_t nYvec = vld1_u8(LOAD_Y(i,j));
        uint8x8_t nUVvec = vld1_u8(LOAD_V(i,j)); // v,u,v,u v,u,v,u

        nYvec = vmul_u8(nYvec, vcle_u8(nYvec,ZEROSvec));

        // Yuv Convert
        //        nY -= 16;
        //        nU -= 128;
        //        nV -= 128;

        //        nYvec = vsub_u8(nYvec, Y_SUBvec);
        //        nUVvec = vsub_u8(nYvec, UV_SUBvec);

        uint16x8_t nYvec16 = vmovl_u8(vsub_u8(nYvec, Y_SUBvec));
        uint16x8_t nUVvec16 = vmovl_u8(vsub_u8(nYvec, UV_SUBvec));

        uint16x4_t Y_low4 = vget_low_u16(nYvec16);
        uint16x4_t Y_high4 = vget_high_u16(nYvec16);
        uint16x4_t UV_low4 = vget_low_u16(nUVvec16);
        uint16x4_t UV_high4 = vget_high_u16(nUVvec16);

        uint32x4_t UV_low4_int = vmovl_u16(UV_low4);
        uint32x4_t UV_high4_int = vmovl_u16(UV_high4);

        uint32x4_t Y_low4_int = vmull_n_u16(Y_low4, 1192);
        uint32x4_t Y_high4_int = vmull_n_u16(Y_high4, 1192);

        uint32x4x2_t UV_uzp = vuzpq_u32(UV_low4_int, UV_high4_int);

        uint32x2_t Vl = vget_low_u32(UV_uzp.val[0]);// vld1_u32(UVvec_int);
        uint32x2_t Vh = vget_high_u32(UV_uzp.val[0]);//vld1_u32(UVvec_int + 2);

        uint32x2x2_t Vll_ = vzip_u32(Vl, Vl);
        uint32x4_t* Vll = (uint32x4_t*)(&Vll_);

        uint32x2x2_t Vhh_ = vzip_u32(Vh, Vh);
        uint32x4_t* Vhh = (uint32x4_t*)(&Vhh_);

        uint32x2_t Ul =  vget_low_u32(UV_uzp.val[1]);
        uint32x2_t Uh =  vget_high_u32(UV_uzp.val[1]);

        uint32x2x2_t Ull_ = vzip_u32(Ul, Ul);
        uint32x4_t* Ull = (uint32x4_t*)(&Ull_);

        uint32x2x2_t Uhh_ = vzip_u32(Uh, Uh);
        uint32x4_t* Uhh = (uint32x4_t*)(&Uhh_);

        uint32x4_t B_int_low = vmlaq_n_u32(Y_low4_int, *Ull, 2066); //multiply by scalar accum
        uint32x4_t B_int_high = vmlaq_n_u32(Y_high4_int, *Uhh, 2066); //multiply by scalar accum
        uint32x4_t G_int_low = vsubq_u32(Y_low4_int, vmlaq_n_u32(vmulq_n_u32(*Vll, 833), *Ull, 400));
        uint32x4_t G_int_high = vsubq_u32(Y_high4_int, vmlaq_n_u32(vmulq_n_u32(*Vhh, 833), *Uhh, 400));
        uint32x4_t R_int_low = vmlaq_n_u32(Y_low4_int, *Vll, 1634); //multiply by scalar accum
        uint32x4_t R_int_high = vmlaq_n_u32(Y_high4_int, *Vhh, 1634); //multiply by scalar accum

        B_int_low = vshrq_n_u32 (B_int_low, 10);
        B_int_high = vshrq_n_u32 (B_int_high, 10);
        G_int_low = vshrq_n_u32 (G_int_low, 10);
        G_int_high = vshrq_n_u32 (G_int_high, 10);
        R_int_low = vshrq_n_u32 (R_int_low, 10);
        R_int_high = vshrq_n_u32 (R_int_high, 10);


        uint8x8x3_t RGB;
        RGB.val[0] = vmovn_u16(vcombine_u16(vqmovn_u32 (R_int_low),vqmovn_u32 (R_int_high)));
        RGB.val[1] = vmovn_u16(vcombine_u16(vqmovn_u32 (G_int_low),vqmovn_u32 (G_int_high)));
        RGB.val[2] = vmovn_u16(vcombine_u16(vqmovn_u32 (B_int_low),vqmovn_u32 (B_int_high)));

        vst3_u8 (out+i*width*3 + j*3, RGB);
      }
    }

}

