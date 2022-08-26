// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
Winograd-based convolution F(6x6, 3x3).
The code has been borrowed from ncnn inference engine (https://github.com/Tencent/ncnn)
and adapted for OpenCV by Zihao Mu.

Below is the original copyright
*/

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "../../precomp.hpp"
#include "fast_convolution.hpp"

namespace cv { namespace dnn {
enum
{
    WINO_STEP=6,
    WINO_KSIZE=3,
    WINO_SIZE= WINO_STEP + WINO_KSIZE - 1,
    WINO_AREA= WINO_SIZE * WINO_SIZE
};

#if CV_NEON

#undef _FAST_CONV_T4x4
#define _FAST_CONV_T4x4(a, b, c, d, tr0, tr1) \
    tr0 = vtrnq_f32(a, b); \
    tr1 = vtrnq_f32(c, d); \
    a = vcombine_f32(vget_low_f32(tr0.val[0]), vget_low_f32(tr1.val[0])); \
    b = vcombine_f32(vget_low_f32(tr0.val[1]), vget_low_f32(tr1.val[1])); \
    c = vcombine_f32(vget_high_f32(tr0.val[0]), vget_high_f32(tr1.val[0])); \
    d = vcombine_f32(vget_high_f32(tr0.val[1]), vget_high_f32(tr1.val[1]))

// The input is the pack4 data, and the output is unpack4 data.
static void transpose12x4(float* src, float* dst, const int cn)
{
    float32x4_t r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11;
    float32x4x2_t tr0, tr1;
    for (int i = 0; i < cn; i++, src += 48, dst += 48)
    {
        r00 = vld1q_f32(src);
        r01 = vld1q_f32(src + 4);
        r02 = vld1q_f32(src + 8);
        r03 = vld1q_f32(src + 12);
        r04 = vld1q_f32(src + 16);
        r05 = vld1q_f32(src + 20);
        r06 = vld1q_f32(src + 24);
        r07 = vld1q_f32(src + 28);
        r08 = vld1q_f32(src + 32);
        r09 = vld1q_f32(src + 36);
        r10 = vld1q_f32(src + 40);
        r11 = vld1q_f32(src + 44);

        _FAST_CONV_T4x4(r00, r01, r02, r03, tr0, tr1);
        _FAST_CONV_T4x4(r04, r05, r06, r07, tr0, tr1);
        _FAST_CONV_T4x4(r08, r09, r10, r11, tr0, tr1);

        vst1q_f32(dst, r00), vst1q_f32(dst + 4, r04), vst1q_f32(dst + 8, r08);
        vst1q_f32(dst + 12, r01), vst1q_f32(dst + 16, r05), vst1q_f32(dst + 20, r09);
        vst1q_f32(dst + 24, r02), vst1q_f32(dst + 28, r06), vst1q_f32(dst + 32, r10);
        vst1q_f32(dst + 36, r03), vst1q_f32(dst + 40, r07), vst1q_f32(dst + 44, r11);
    }
}

static void winograd_trans_input_F63(float* src, float* dst, int Channle_div4, const int tiles, const int big_step, const int line_step, const int* ofstab0)
{
    // const float itm[8][8] = {
    //     {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
    //
    //     {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
    //     {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
    //     {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
    //     {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
    //
    //     {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
    // };

    // 0 = r00 - r06 + (r04 - r02) * 5.25
    // 7 = r07 - r01 + (r03 - r05) * 5.25

    // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
    // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

    // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
    // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

    // reuse r04 * 1.25
    // reuse r03 * 2.5
    // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
    // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

    float tmp[8][8][FAST_VEC_NLANES];
    AutoBuffer<float> input_buf0_;
    input_buf0_.allocate(64 * tiles * FAST_VEC_NLANES);

    float* input_buf0 = input_buf0_.data();
    memset(input_buf0, 0, 64 * tiles * FAST_VEC_NLANES * sizeof(float ));

    for (int ti = 0; ti < tiles; ti++)
    {
        float* input0 = src + ti *  64 * 4;
        float* input = input0;
        for (int m = 0; m < 8; m++)
        {
            float32x4_t _r00 = vld1q_f32(input);
            float32x4_t _r01 = vld1q_f32(input + 4);
            float32x4_t _r02 = vld1q_f32(input + 8);
            float32x4_t _r03 = vld1q_f32(input + 12);
            float32x4_t _r04 = vld1q_f32(input + 16);
            float32x4_t _r05 = vld1q_f32(input + 20);
            float32x4_t _r06 = vld1q_f32(input + 24);
            float32x4_t _r07 = vld1q_f32(input + 28);

            float32x4_t _tmp0m = vmlaq_n_f32(vsubq_f32(_r00, _r06), vsubq_f32(_r04, _r02), 5.25f);
            float32x4_t _tmp7m = vmlaq_n_f32(vsubq_f32(_r07, _r01), vsubq_f32(_r03, _r05), 5.25f);
            vst1q_f32(tmp[0][m], _tmp0m);
            vst1q_f32(tmp[7][m], _tmp7m);

            float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_r02, _r06), _r04, 4.25f);
            float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_r01, _r05), _r03, 4.25f);

            float32x4_t _tmp1m = vaddq_f32(_tmp12a, _tmp12b);
            float32x4_t _tmp2m = vsubq_f32(_tmp12a, _tmp12b);
            vst1q_f32(tmp[1][m], _tmp1m);
            vst1q_f32(tmp[2][m], _tmp2m);

            float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_r06, _r02, 0.25f), _r04, 1.25f);
            float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 0.5f), _r03, 2.5f), _r05, 2.f);

            float32x4_t _tmp3m = vaddq_f32(_tmp34a, _tmp34b);
            float32x4_t _tmp4m = vsubq_f32(_tmp34a, _tmp34b);
            vst1q_f32(tmp[3][m], _tmp3m);
            vst1q_f32(tmp[4][m], _tmp4m);

            float32x4_t _tmp56a = vmlaq_n_f32(_r06, vmlsq_n_f32(_r02, _r04, 1.25f), 4.f);
            float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 2.f), _r03, 2.5f), _r05, 0.5f);

            float32x4_t _tmp5m = vaddq_f32(_tmp56a, _tmp56b);
            float32x4_t _tmp6m = vsubq_f32(_tmp56a, _tmp56b);
            vst1q_f32(tmp[5][m], _tmp5m);
            vst1q_f32(tmp[6][m], _tmp6m);

            input += 8 * FAST_VEC_NLANES;
        }

        float* input_buf00 = input_buf0 + ti * 4;
        float* input_buf01 = input_buf00 + tiles * 4;
        float* input_buf02 = input_buf00 + tiles * 8;
        float* input_buf03 = input_buf00 + tiles * 12;
        float* input_buf04 = input_buf00 + tiles * 16;
        float* input_buf05 = input_buf00 + tiles * 20;
        float* input_buf06 = input_buf00 + tiles * 24;
        float* input_buf07 = input_buf00 + tiles * 28;

        for (int m = 0; m < 8; m++)
        {
            float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
            float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
            float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
            float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
            float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
            float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);
            float32x4_t _tmp06 = vld1q_f32(tmp[m][6]);
            float32x4_t _tmp07 = vld1q_f32(tmp[m][7]);

            float32x4_t _r0tm0 = vmlaq_n_f32(vsubq_f32(_tmp00, _tmp06), vsubq_f32(_tmp04, _tmp02), 5.25f);
            float32x4_t _r0tm7 = vmlaq_n_f32(vsubq_f32(_tmp07, _tmp01), vsubq_f32(_tmp03, _tmp05), 5.25f);

            float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_tmp02, _tmp06), _tmp04, 4.25f);
            float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_tmp01, _tmp05), _tmp03, 4.25f);

            float32x4_t _r0tm1 = vaddq_f32(_tmp12a, _tmp12b);
            float32x4_t _r0tm2 = vsubq_f32(_tmp12a, _tmp12b);

            float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
            float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

            float32x4_t _r0tm3 = vaddq_f32(_tmp34a, _tmp34b);
            float32x4_t _r0tm4 = vsubq_f32(_tmp34a, _tmp34b);

            float32x4_t _tmp56a = vmlaq_n_f32(_tmp06, vmlsq_n_f32(_tmp02, _tmp04, 1.25f), 4.f);
            float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

            float32x4_t _r0tm5 = vaddq_f32(_tmp56a, _tmp56b);
            float32x4_t _r0tm6 = vsubq_f32(_tmp56a, _tmp56b);

            vst1q_f32(input_buf00,  _r0tm0);
            vst1q_f32(input_buf01,  _r0tm1);
            vst1q_f32(input_buf02,  _r0tm2);
            vst1q_f32(input_buf03, _r0tm3);
            vst1q_f32(input_buf04, _r0tm4);
            vst1q_f32(input_buf05, _r0tm5);
            vst1q_f32(input_buf06, _r0tm6);
            vst1q_f32(input_buf07, _r0tm7);

            input_buf00 += tiles * 32;
            input_buf01 += tiles * 32;
            input_buf02 += tiles * 32;
            input_buf03 += tiles * 32;
            input_buf04 += tiles * 32;
            input_buf05 += tiles * 32;
            input_buf06 += tiles * 32;
            input_buf07 += tiles * 32;
        }
    }

    // [line Number, input pack]
    // if InpPack == 8;
    for (int r = 0; r < 64; r++)
    {
        int ti = 0;
        float* out0 = dst + r * big_step;
        float* input0 = input_buf0 + 4 * tiles * r;

        // TODO! support tiles > 12
#if CV_NEON_AARCH64
        for (; ti + 11 < tiles; ti += 12)
        {
            float* out1 = out0 + line_step * ofstab0[ti * 2] + Channle_div4 * ofstab0[ti * 2 + 1] * 4;
            float* input1 = input0 + ti * 4;
            memcpy(out1, input1, 12 * 4 * sizeof(float ));
        }
#endif
        for (; ti + 7 < tiles; ti += 8)
        {
            float* out1 = out0 + line_step * ofstab0[ti * 2] + Channle_div4 * ofstab0[ti * 2 + 1] * 4;
            float* input1 = input0 + ti * 4;
            memcpy(out1, input1, 8 * 4 * sizeof(float ));
        }

        for (; ti + 3 < tiles; ti += 4)
        {
            float* out1 = out0 + line_step * ofstab0[ti * 2] + Channle_div4 * ofstab0[ti * 2 + 1] * 4;
            float* input1 = input0 + ti * 4;
            memcpy(out1, input1, 4 * 4 * sizeof(float ));
        }

        for (; ti + 1 < tiles; ti += 2)
        {
            float* out1 = out0 + line_step * ofstab0[ti * 2] + Channle_div4 * ofstab0[ti * 2 + 1] * 4;
            float* input1 = input0 + ti * 4;
            memcpy(out1, input1, 2 * 4 * sizeof(float ));
        }

        for (; ti < tiles; ti++)
        {
            float* out1 = out0 + line_step * ofstab0[ti * 2] + Channle_div4 * ofstab0[ti * 2 + 1] * 4;
            float* input1 = input0 + ti * 4;
            memcpy(out1, input1, 1 * 4 * sizeof(float ));
        }
    }
}

static void winograd_trans_output_F63(float* src_, float* bias_, float* fAbuf0, float minval, float maxval, bool ifMinMaxAct)
{
    // const float otm[6][8] = {
    //     {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
    // };

    // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
    // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
    // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
    // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
    // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
    // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

    float32x4_t bias0 = bias_ ? vld1q_f32(bias_) : vdupq_n_f32(0.f);
    float tmp[6][8][4];

    for (int m = 0; m < 8; m++)
    {
        float* output0 = src_ + 8 * m * FAST_VEC_NLANES;

        float32x4_t _out0tm0 = vld1q_f32(output0);
        float32x4_t _out0tm1 = vld1q_f32(output0 + FAST_VEC_NLANES * 1);
        float32x4_t _out0tm2 = vld1q_f32(output0 + FAST_VEC_NLANES * 2);
        float32x4_t _out0tm3 = vld1q_f32(output0 + FAST_VEC_NLANES * 3);
        float32x4_t _out0tm4 = vld1q_f32(output0 + FAST_VEC_NLANES * 4);
        float32x4_t _out0tm5 = vld1q_f32(output0 + FAST_VEC_NLANES * 5);
        float32x4_t _out0tm6 = vld1q_f32(output0 + FAST_VEC_NLANES * 6);
        float32x4_t _out0tm7 = vld1q_f32(output0 + FAST_VEC_NLANES * 7);

        float32x4_t _tmp024a = vaddq_f32(_out0tm1, _out0tm2);
        float32x4_t _tmp135a = vsubq_f32(_out0tm1, _out0tm2);

        float32x4_t _tmp024b = vaddq_f32(_out0tm3, _out0tm4);
        float32x4_t _tmp135b = vsubq_f32(_out0tm3, _out0tm4);

        float32x4_t _tmp024c = vaddq_f32(_out0tm5, _out0tm6);
        float32x4_t _tmp135c = vsubq_f32(_out0tm5, _out0tm6);

        float32x4_t _tmp0m = vaddq_f32(vaddq_f32(_out0tm0, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f));
        float32x4_t _tmp2m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f);
        float32x4_t _tmp4m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f);
        vst1q_f32(tmp[0][m], _tmp0m);
        vst1q_f32(tmp[2][m], _tmp2m);
        vst1q_f32(tmp[4][m], _tmp4m);

        float32x4_t _tmp1m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f);
        float32x4_t _tmp3m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f);
        float32x4_t _tmp5m = vaddq_f32(vaddq_f32(_out0tm7, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f));
        vst1q_f32(tmp[1][m], _tmp1m);
        vst1q_f32(tmp[3][m], _tmp3m);
        vst1q_f32(tmp[5][m], _tmp5m);
    }

    for (int m = 0; m < 6; m++)
    {
        float* output0 = src_ + 6 * m * FAST_VEC_NLANES;
        float* fAbuf = fAbuf0 ? fAbuf0 + 6 * m * FAST_VEC_NLANES : 0;

        float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
        float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
        float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
        float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
        float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
        float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);
        float32x4_t _tmp06 = vld1q_f32(tmp[m][6]);
        float32x4_t _tmp07 = vld1q_f32(tmp[m][7]);

        float32x4_t _tmp024a = vaddq_f32(_tmp01, _tmp02);
        float32x4_t _tmp135a = vsubq_f32(_tmp01, _tmp02);

        float32x4_t _tmp024b = vaddq_f32(_tmp03, _tmp04);
        float32x4_t _tmp135b = vsubq_f32(_tmp03, _tmp04);

        float32x4_t _tmp024c = vaddq_f32(_tmp05, _tmp06);
        float32x4_t _tmp135c = vsubq_f32(_tmp05, _tmp06);

        float32x4_t _out00 = vaddq_f32(bias0, vaddq_f32(vaddq_f32(_tmp00, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f)));
        float32x4_t _out02 = vaddq_f32(bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f));
        float32x4_t _out04 = vaddq_f32(bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f));

        float32x4_t _out01 = vaddq_f32(bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f));
        float32x4_t _out03 = vaddq_f32(bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f));
        float32x4_t _out05 = vaddq_f32(bias0, vaddq_f32(vaddq_f32(_tmp07, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f)));

        if (fAbuf)
        {
            _out00 = vaddq_f32(_out00, vld1q_f32(fAbuf));
            _out01 = vaddq_f32(_out01, vld1q_f32(fAbuf + 4));
            _out02 = vaddq_f32(_out02, vld1q_f32(fAbuf + 8));
            _out03 = vaddq_f32(_out03, vld1q_f32(fAbuf + 12));
            _out04 = vaddq_f32(_out04, vld1q_f32(fAbuf + 16));
            _out05 = vaddq_f32(_out05, vld1q_f32(fAbuf + 20));
        }

        if (ifMinMaxAct)
        {
            float32x4_t vmin = vdupq_n_f32(minval), vmax = vdupq_n_f32(maxval);
            _out00 = vminq_f32(vmaxq_f32(_out00, vmin), vmax);
            _out01 = vminq_f32(vmaxq_f32(_out01, vmin), vmax);
            _out02 = vminq_f32(vmaxq_f32(_out02, vmin), vmax);
            _out03 = vminq_f32(vmaxq_f32(_out03, vmin), vmax);
            _out04 = vminq_f32(vmaxq_f32(_out04, vmin), vmax);
            _out05 = vminq_f32(vmaxq_f32(_out05, vmin), vmax);
        }

        vst1q_f32(output0,                     _out00);
        vst1q_f32(output0 +     FAST_VEC_NLANES, _out01);
        vst1q_f32(output0 + 2 * FAST_VEC_NLANES, _out02);
        vst1q_f32(output0 + 3 * FAST_VEC_NLANES, _out03);
        vst1q_f32(output0 + 4 * FAST_VEC_NLANES, _out04);
        vst1q_f32(output0 + 5 * FAST_VEC_NLANES, _out05);
    }
}

void initWinograd63(Ptr<FastConv2d>& conv, InputArray _weightsMat, int K, int C)
{
    static const float ktm[8][3] = {
            {1.0f,      0.0f,      0.0f},
            {-2.0f / 9, -2.0f / 9, -2.0f / 9},
            {-2.0f / 9, 2.0f / 9, -2.0f / 9},
            {1.0f / 90, 1.0f / 45, 2.0f / 45},
            {1.0f / 90, -1.0f / 45, 2.0f / 45},
            {1.0f / 45, 1.0f / 90, 1.0f / 180},
            {1.0f / 45, -1.0f / 90, 1.0f / 180},
            {0.0f, 0.0f, 1.0f}
    };

    Mat weightsMat = _weightsMat.getMat();
    float* srcWeight = weightsMat.ptr<float>();
    size_t wstep = weightsMat.step1();

    int K_aligned = ((K + FAST_VEC_NLANES - 1)/FAST_VEC_NLANES) * FAST_VEC_NLANES;
    int C_aligned = ((C + FAST_VEC_NLANES - 1)/FAST_VEC_NLANES) * FAST_VEC_NLANES;
    const int winoSize = C * WINO_AREA;
    const int kArea = WINO_KSIZE * WINO_KSIZE;

    // Allocate memory for winograd.
    int nweights = K_aligned * C_aligned * WINO_AREA;

    conv->weightsWino63Buf.reserve(nweights);
    float* weightsWino63Ptr = conv->weightsWino63Buf.data();
    memset(weightsWino63Ptr, 0, nweights*sizeof(weightsWino63Ptr[0]));
    float* wptrWino = weightsWino63Ptr;

    AutoBuffer<float> kernelTm0_;
    kernelTm0_.allocate(WINO_AREA * K * C);
    float *kernelTm = kernelTm0_.data();
    memset(kernelTm, 0, WINO_AREA * K * C*sizeof(kernelTm[0]));

    // Step1 Transform : size [K, C, 8, 8]
    parallel_for_(Range(0, K), [&](const Range& r0)
    {
        for (int outc = r0.start; outc < r0.end; outc++)
        {
            for (int inc = 0; inc < C; inc++)
            {
                float *kernel_tm0 = kernelTm + outc * winoSize + inc * WINO_AREA;
                const float *kernel0 = srcWeight + outc * wstep + inc * kArea;

                // transform kernel, transposed
                const float *k0 = kernel0;
                const float *k1 = kernel0 + 3;
                const float *k2 = kernel0 + 6;

                // h
                float tmp[8][3];
                for (int i = 0; i < 8; i++)
                {
                    tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                    tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                    tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                }

                // v
                for (int j = 0; j < 8; j++)
                {
                    float *tmpp = &tmp[j][0];

                    for (int i = 0; i < 8; i++)
                    {
                        kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                    }
                }
            }
        }
    });

    // Step2 Pack 4:
    // If the number of vector registers >= 32 and outch >= 8,
    // the size = [8*8, K/4/2, C * 2, 4], otherwise [8*8, K/4, C, 4]
    for (int r = 0; r < 64; r++)
    {
        int outc = 0;
        float* out0 = wptrWino + r * K_aligned * C_aligned;
        float* tmp0 = kernelTm + r;

#if CV_NEON_AARCH64
        // Pack 8
        for (;outc + 7 < K_aligned; outc += 8)
        {
            for (int i = 0; i < 8; i++)
            {
                int outc_i = outc + i;
                int offset8 = outc_i % 8;
                int outc8 = outc_i / 8;
                float* out1 = out0 + outc8 * 8 * C_aligned + offset8;

                if (outc_i >= K)
                {
                    continue;
                }
                else
                {
                    float* tmp1 = tmp0 + outc_i * 64 * C;

                    for (int inc = 0; inc < C_aligned; inc++)
                    {
                        if (inc >= C)
                            continue;

                        out1[inc * 8] = tmp1[inc * 64];
                    }
                }
            }
        }
#endif

        // Pack 4
        for (;outc < K_aligned; outc++)
        {
            int offset4 = outc % FAST_VEC_NLANES;
            int outc4 = outc / FAST_VEC_NLANES;
            float* out1 = out0 + outc4 * 4 * C_aligned + offset4;

            if (outc >= K)
            {
                continue;
            }
            else
            {
                float* tmp1 = tmp0 + outc * 64 * C;

                for (int inc = 0; inc < C_aligned; inc++)
                {
                    if (inc >= C)
                        continue;

                    out1[inc * 4] = tmp1[inc * 64];
                }
            }
        }
    }
}

int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv2d>& conv, int ntasks, float minval,
        float maxval, ActivationLayer* activ, bool ifMinMaxAct)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    Mat fusedAddMat = _fusedAddMat.getMat();

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);
    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->K;
    int H0 = outputShape[2], W0 = outputShape[3];

    // Allocate the right memory size for output.
    // H and W is integer of 6. the output HxW is integer of 6x6
    int H_tiles = ((H0 + 5) / 6);
    int W_tiles = ((W0 + 5) / 6);
    int tiles = H_tiles * W_tiles;

    int H0_align = H_tiles * 6;
    int W0_align = W_tiles * 6;

    int Hi_align = H0_align + 2;
    int Wi_align = W0_align + 2;

    int pad_top = conv->pad_top, pad_bottom = Hi_align - pad_top - Hi;
    int pad_left = conv->pad_left, pad_right = Wi_align - pad_left - Wi;

    int in_top = pad_top, in_bottom = Hi_align - pad_bottom;
    int in_left = pad_left, in_right = Wi_align - pad_right;

    CV_Assert(in_bottom >= in_top && in_right >= in_left);

    int C_aligned = ((C + FAST_VEC_NLANES - 1)/FAST_VEC_NLANES) * FAST_VEC_NLANES;
    int K_aligned = ((K + FAST_VEC_NLANES - 1)/FAST_VEC_NLANES) * FAST_VEC_NLANES;

    int inpPack = 0;
    int lineNum =0;

#if CV_NEON_AARCH64
    if (tiles >= 12)
    {
        inpPack = 12;
        lineNum = tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2;
    }
    else
#endif
    if (tiles >= 8)
    {
        inpPack = 8;
        lineNum = tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2;
    }
    else
    if (tiles >= 4)
    {
        inpPack = 4;
        lineNum = tiles / 4 + (tiles % 4) / 2 + tiles % 2;
    }
    else if (tiles >= 2)
    {
        inpPack = 2;
        lineNum = tiles / 2 + tiles % 2;
    }
    else // tiles >= 1
    {
        inpPack = 1;
        lineNum = tiles;
    }
    CV_Assert(lineNum > 0 && inpPack > 0);
    std::vector<int> ofstab0_(tiles * 2, 0);
    int* ofstab0 = ofstab0_.data(); // [line Number, input pack]

    int tiles_tmp = tiles;
    int line_0 = 0;

    int* ofstab_tmp = ofstab0;
    int big_step = inpPack * C_aligned * lineNum;
    int line_step = inpPack * C_aligned;

    std::vector<int> linePackList = {12, 8, 4, 2, 1};
    auto iter = std::find(linePackList.begin(), linePackList.end(), inpPack);
    CV_Assert(iter != linePackList.end());
    int ptr = iter - linePackList.begin();

    while (ptr < linePackList.size() && tiles_tmp != 0)
    {
        if (tiles_tmp >= linePackList[ptr])
        {
            int num = tiles_tmp / linePackList[ptr];
            for (int i = 0; i < num; i ++)
            {
                for (int j = 0; j < linePackList[ptr]; j++)
                {
                    ofstab_tmp[0] = line_0;
                    ofstab_tmp[1] = linePackList[ptr];
                    ofstab_tmp += 2;
                }
                line_0++;
            }
            tiles_tmp -= num * linePackList[ptr];
        }
        else
        {
            ptr++;
        }
    }

    const size_t inp_planesize = (size_t)Hi*Wi;
    const size_t out_planesize = (size_t)H0*W0;

    size_t inputbuf_size = inpPack * C_aligned * lineNum * 64;
    size_t inputbufCn_size = ntasks * tiles * 4 * 8 * 8;

    size_t outputbuf_size = tiles * K_aligned * 8 * 8;
    size_t outputCnbuf_size = ntasks * 8 * 8 * 4;

    size_t part0_size = std::max(inputbuf_size, outputCnbuf_size);
    size_t allbuf_size = part0_size + std::max(inputbufCn_size, outputbuf_size);

    AutoBuffer<float> allbuf_;
    allbuf_.allocate(allbuf_size);
    float* inputbuf0 = alignPtr(allbuf_.data(), (int)(sizeof(float)));
    float* inputCnbuf0 = inputbuf0 + inputbuf_size;
    float* outputbuf0 = inputCnbuf0;
    float* outputCnbuf0 = inputbuf0;

    // Input Parallel For
    float* weight_ptr0 = conv->weightsWino63Buf.data();

    for (int bn = 0; bn < N; bn++)
    {
        float* input_ptr0 = input.ptr<float>() + bn * inp_planesize * C;
        float* output_ptr0 = output.ptr<float>() + bn * out_planesize * K;
        float* fusedAddPtr0 = fusedAddMat.empty() ? 0 : fusedAddMat.ptr<float>() + bn * out_planesize * K;

        // Transform Input
        int C_aligned_div4 = C_aligned/4;
        const int tiStep = 8 * 8 * FAST_VEC_NLANES;

        parallel_for_(Range(0, ntasks), [&](const Range& range){
        for (int task_i = range.start; task_i < range.end; task_i++)
            {
                float *inpCnbuf = inputCnbuf0 + tiles * tiStep * task_i;
                for (int inc4 = task_i; inc4 < C_aligned_div4; inc4 += ntasks)
                {
                    for (int cn = 0; cn < 4; cn++)
                    {
                        if (cn + inc4 * 4 >= C)
                        {
                            // set value to zero
                            for (int ti = 0; ti < tiles; ti++)
                            {
                                float *inpCnbuf_i = inpCnbuf + ti * 4 * 64 + cn;

                                for (int i = 0; i < 8; i++)
                                {
                                    inpCnbuf_i[0] = 0.0f;
                                    inpCnbuf_i[4] = 0.0f;
                                    inpCnbuf_i[8] = 0.0f;
                                    inpCnbuf_i[12] = 0.0f;

                                    inpCnbuf_i[16] = 0.0f;
                                    inpCnbuf_i[20] = 0.0f;
                                    inpCnbuf_i[24] = 0.0f;
                                    inpCnbuf_i[28] = 0.0f;

                                    inpCnbuf_i += 4 * 8;
                                }
                            }
                        }
                        else
                        {
                            float *input_ptr = input_ptr0 + (inc4 * 4 + cn) * Hi * Wi;

                            for (int ti = 0; ti < tiles; ti++)
                            {
                                float *input_buf0_i = inpCnbuf + ti * 256 + cn;

                                int hi = ti / W_tiles;
                                int wi = ti % W_tiles;

                                int h_top = hi * 6, h_bottom = hi * 6 + 8;
                                int w_left = wi * 6, w_right = wi * 6 + 8;

                                for (int h = h_top; h < h_bottom; h++)
                                {
                                    if (h >= in_bottom || h < in_top)
                                    {
                                        input_buf0_i[0] = 0.0f;
                                        input_buf0_i[4] = 0.0f;
                                        input_buf0_i[8] = 0.0f;
                                        input_buf0_i[12] = 0.0f;

                                        input_buf0_i[16] = 0.0f;
                                        input_buf0_i[20] = 0.0f;
                                        input_buf0_i[24] = 0.0f;
                                        input_buf0_i[28] = 0.0f;

                                        input_buf0_i += 32;
                                        continue;
                                    }

                                    for (int w = w_left; w < w_right; w++)
                                    {
                                        if (w >= in_right || w < in_left)
                                        {
                                            input_buf0_i[0] = 0.0f;
                                            input_buf0_i += 4;
                                            continue;
                                        }
                                        input_buf0_i[0] = input_ptr[(h - pad_top) * Wi + w - pad_left];
                                        input_buf0_i += 4;
                                    }
                                }
                            }
                        }
                    }

                    // Transform Compute BdB^T
                    winograd_trans_input_F63(inpCnbuf, inputbuf0, inc4, tiles, big_step, line_step, ofstab0);
                }
            }
        });
        // Matrix multiplication 8 channel
        int K_div8 = 0;
#if CV_NEON_AARCH64
        K_div8 = K_aligned/8;
        // Transpose 12
        if (inpPack == 12)
        {
            int C_div4 = C_aligned/4;
            parallel_for_(Range(0, 64), [&](const Range &range){
            for (int r = range.start; r < range.end; r++)
            {
                float* input_tm = inputbuf0 + r * big_step;

                for (int ti = 0; ti + 11 < tiles; ti += 12)
                {
                    float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                    transpose12x4(r0, r0, C_div4);
                }
            }
            });
        }

        parallel_for_(Range(0, 64), [&](const Range &range){
        for (int r = range.start; r < range.end; r++)
        {
            float* input_tm = inputbuf0 + r * big_step;
            float* output_tmp = outputbuf0 + tiles * K_aligned * r;
            float* kernel_tmp = weight_ptr0 + r * C_aligned * K_aligned;

            for (int out_div8 = 0; out_div8 < K_div8; out_div8 ++)
            {
                float* output0_tm = output_tmp + tiles * out_div8 * 8;
                float* output1_tm = output0_tm + tiles * 4;
                float* kernel_tm_i = kernel_tmp + out_div8 * 8 * C_aligned;

                int ti = 0;
                for (; ti + 11 < tiles; ti += 12)
                {
                    float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                    const float* k01 = kernel_tm_i;

                    int nn = C_aligned/4;
                    r0 = input_tm + ofstab0[ti * 2] * line_step;

                    // init 32 registers. FMA/load ratio = 96/20
                    float32x4_t r00 = vdupq_n_f32(0.0f), r01 = r00, r02 = r00, r03 = r00;
                    float32x4_t r04 = r00, r05 = r00, r06 = r00, r07 = r00;
                    float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                    float32x4_t r12 = r00, r13 = r00, r14 = r00, r15 = r00;
                    float32x4_t r16 = r00, r17 = r00, r18 = r00, r19 = r00;
                    float32x4_t r20 = r00, r21 = r00, r22 = r00, r23 = r00;
                    float32x4_t r24 = r00, r25 = r00, r26 = r00, r27 = r00;
                    float32x4_t r28 = r00, r29 = r00, r30 = r00, r31 = r00;

                    for(;nn > 0; nn--)
                    {
                        r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4), r02 = vld1q_f32(r0+8), r03 = vld1q_f32(r0+12);
                        r04 = vld1q_f32(k01), r05 = vld1q_f32(k01+4), r06 = vld1q_f32(k01+8), r07 = vld1q_f32(k01+12);
                        r0 += 16, k01 += 16;

                        // Cn0
                        // 8 ~ 19
                        r08 = vfmaq_laneq_f32(r08, r04, r00, 0);
                        r09 = vfmaq_laneq_f32(r09, r04, r00, 1);
                        r10 = vfmaq_laneq_f32(r10, r04, r00, 2);
                        r11 = vfmaq_laneq_f32(r11, r04, r00, 3);

                        r12 = vfmaq_laneq_f32(r12, r04, r01, 0);
                        r13 = vfmaq_laneq_f32(r13, r04, r01, 1);
                        r14 = vfmaq_laneq_f32(r14, r04, r01, 2);
                        r15 = vfmaq_laneq_f32(r15, r04, r01, 3);

                        r16 = vfmaq_laneq_f32(r16, r04, r02, 0);
                        r17 = vfmaq_laneq_f32(r17, r04, r02, 1);
                        r18 = vfmaq_laneq_f32(r18, r04, r02, 2);
                        r19 = vfmaq_laneq_f32(r19, r04, r02, 3);

                        // 20 ~ 31
                        r20 = vfmaq_laneq_f32(r20, r05, r00, 0);
                        r21 = vfmaq_laneq_f32(r21, r05, r00, 1);
                        r22 = vfmaq_laneq_f32(r22, r05, r00, 2);
                        r23 = vfmaq_laneq_f32(r23, r05, r00, 3);

                        r24 = vfmaq_laneq_f32(r24, r05, r01, 0);
                        r25 = vfmaq_laneq_f32(r25, r05, r01, 1);
                        r26 = vfmaq_laneq_f32(r26, r05, r01, 2);
                        r27 = vfmaq_laneq_f32(r27, r05, r01, 3);

                        r28 = vfmaq_laneq_f32(r28, r05, r02, 0);
                        r29 = vfmaq_laneq_f32(r29, r05, r02, 1);
                        r30 = vfmaq_laneq_f32(r30, r05, r02, 2);
                        r31 = vfmaq_laneq_f32(r31, r05, r02, 3);

                        // Cn1
                        r08 = vfmaq_laneq_f32(r08, r06, r03, 0);
                        r09 = vfmaq_laneq_f32(r09, r06, r03, 1);
                        r10 = vfmaq_laneq_f32(r10, r06, r03, 2);
                        r11 = vfmaq_laneq_f32(r11, r06, r03, 3);

                        r20 = vfmaq_laneq_f32(r20, r07, r03, 0);
                        r21 = vfmaq_laneq_f32(r21, r07, r03, 1);
                        r22 = vfmaq_laneq_f32(r22, r07, r03, 2);
                        r23 = vfmaq_laneq_f32(r23, r07, r03, 3);

                        r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4), r02 = vld1q_f32(r0+8), r03 = vld1q_f32(r0+12);
                        r0 += 16;

                        r12 = vfmaq_laneq_f32(r12, r06, r00, 0);
                        r13 = vfmaq_laneq_f32(r13, r06, r00, 1);
                        r14 = vfmaq_laneq_f32(r14, r06, r00, 2);
                        r15 = vfmaq_laneq_f32(r15, r06, r00, 3);

                        r16 = vfmaq_laneq_f32(r16, r06, r01, 0);
                        r17 = vfmaq_laneq_f32(r17, r06, r01, 1);
                        r18 = vfmaq_laneq_f32(r18, r06, r01, 2);
                        r19 = vfmaq_laneq_f32(r19, r06, r01, 3);

                        r24 = vfmaq_laneq_f32(r24, r07, r00, 0);
                        r25 = vfmaq_laneq_f32(r25, r07, r00, 1);
                        r26 = vfmaq_laneq_f32(r26, r07, r00, 2);
                        r27 = vfmaq_laneq_f32(r27, r07, r00, 3);

                        r28 = vfmaq_laneq_f32(r28, r07, r01, 0);
                        r29 = vfmaq_laneq_f32(r29, r07, r01, 1);
                        r30 = vfmaq_laneq_f32(r30, r07, r01, 2);
                        r31 = vfmaq_laneq_f32(r31, r07, r01, 3);

                        r04 = vld1q_f32(k01), r05 = vld1q_f32(k01+4), r06 = vld1q_f32(k01+8), r07 = vld1q_f32(k01+12);
                        k01 += 16;

                        // Cn2
                        r08 = vfmaq_laneq_f32(r08, r04, r02, 0);
                        r09 = vfmaq_laneq_f32(r09, r04, r02, 1);
                        r10 = vfmaq_laneq_f32(r10, r04, r02, 2);
                        r11 = vfmaq_laneq_f32(r11, r04, r02, 3);

                        r12 = vfmaq_laneq_f32(r12, r04, r03, 0);
                        r13 = vfmaq_laneq_f32(r13, r04, r03, 1);
                        r14 = vfmaq_laneq_f32(r14, r04, r03, 2);
                        r15 = vfmaq_laneq_f32(r15, r04, r03, 3);

                        r20 = vfmaq_laneq_f32(r20, r05, r02, 0);
                        r21 = vfmaq_laneq_f32(r21, r05, r02, 1);
                        r22 = vfmaq_laneq_f32(r22, r05, r02, 2);
                        r23 = vfmaq_laneq_f32(r23, r05, r02, 3);

                        r24 = vfmaq_laneq_f32(r24, r05, r03, 0);
                        r25 = vfmaq_laneq_f32(r25, r05, r03, 1);
                        r26 = vfmaq_laneq_f32(r26, r05, r03, 2);
                        r27 = vfmaq_laneq_f32(r27, r05, r03, 3);

                        r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4), r02 = vld1q_f32(r0+8), r03 = vld1q_f32(r0+12);
                        r0 += 16;

                        r16 = vfmaq_laneq_f32(r16, r04, r00, 0);
                        r17 = vfmaq_laneq_f32(r17, r04, r00, 1);
                        r18 = vfmaq_laneq_f32(r18, r04, r00, 2);
                        r19 = vfmaq_laneq_f32(r19, r04, r00, 3);

                        r28 = vfmaq_laneq_f32(r28, r05, r00, 0);
                        r29 = vfmaq_laneq_f32(r29, r05, r00, 1);
                        r30 = vfmaq_laneq_f32(r30, r05, r00, 2);
                        r31 = vfmaq_laneq_f32(r31, r05, r00, 3);

                        // Cn3
                        // 8 ~ 19
                        r08 = vfmaq_laneq_f32(r08, r06, r01, 0);
                        r09 = vfmaq_laneq_f32(r09, r06, r01, 1);
                        r10 = vfmaq_laneq_f32(r10, r06, r01, 2);
                        r11 = vfmaq_laneq_f32(r11, r06, r01, 3);

                        r12 = vfmaq_laneq_f32(r12, r06, r02, 0);
                        r13 = vfmaq_laneq_f32(r13, r06, r02, 1);
                        r14 = vfmaq_laneq_f32(r14, r06, r02, 2);
                        r15 = vfmaq_laneq_f32(r15, r06, r02, 3);

                        r16 = vfmaq_laneq_f32(r16, r06, r03, 0);
                        r17 = vfmaq_laneq_f32(r17, r06, r03, 1);
                        r18 = vfmaq_laneq_f32(r18, r06, r03, 2);
                        r19 = vfmaq_laneq_f32(r19, r06, r03, 3);

                        // 20 ~ 31
                        r20 = vfmaq_laneq_f32(r20, r07, r01, 0);
                        r21 = vfmaq_laneq_f32(r21, r07, r01, 1);
                        r22 = vfmaq_laneq_f32(r22, r07, r01, 2);
                        r23 = vfmaq_laneq_f32(r23, r07, r01, 3);

                        r24 = vfmaq_laneq_f32(r24, r07, r02, 0);
                        r25 = vfmaq_laneq_f32(r25, r07, r02, 1);
                        r26 = vfmaq_laneq_f32(r26, r07, r02, 2);
                        r27 = vfmaq_laneq_f32(r27, r07, r02, 3);

                        r28 = vfmaq_laneq_f32(r28, r07, r03, 0);
                        r29 = vfmaq_laneq_f32(r29, r07, r03, 1);
                        r30 = vfmaq_laneq_f32(r30, r07, r03, 2);
                        r31 = vfmaq_laneq_f32(r31, r07, r03, 3);
                    }

                    vst1q_f32(output0_tm, r08), vst1q_f32(output0_tm + 4, r09), vst1q_f32(output0_tm + 8, r10), vst1q_f32(output0_tm + 12, r11);
                    output0_tm += 16;
                    vst1q_f32(output1_tm, r20), vst1q_f32(output1_tm + 4, r21), vst1q_f32(output1_tm + 8, r22), vst1q_f32(output1_tm + 12, r23);
                    output1_tm += 16;

                    vst1q_f32(output0_tm, r12), vst1q_f32(output0_tm + 4, r13), vst1q_f32(output0_tm + 8, r14), vst1q_f32(output0_tm + 12, r15);
                    output0_tm += 16;
                    vst1q_f32(output1_tm, r24), vst1q_f32(output1_tm + 4, r25), vst1q_f32(output1_tm + 8, r26), vst1q_f32(output1_tm + 12, r27);
                    output1_tm += 16;

                    vst1q_f32(output0_tm, r16), vst1q_f32(output0_tm + 4, r17), vst1q_f32(output0_tm + 8, r18), vst1q_f32(output0_tm + 12, r19);
                    output0_tm += 16;
                    vst1q_f32(output1_tm, r28), vst1q_f32(output1_tm + 4, r29), vst1q_f32(output1_tm + 8, r30), vst1q_f32(output1_tm + 12, r31);
                    output1_tm += 16;
                }

                for (; ti + 7 < tiles; ti += 8)
                {
                    const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                    const float* k01 = kernel_tm_i;

                    int nn = C_aligned/4;

                    // init 32 registers. FMA/load ratio = 64/16
                    float32x4_t r00 = vdupq_n_f32(0.0f), r01 = r00, r02 = r00, r03 = r00;
                    float32x4_t r04 = r00, r05 = r00, r06 = r00, r07 = r00;
                    float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                    float32x4_t r12 = r00, r13 = r00, r14 = r00, r15 = r00;
                    float32x4_t r16 = r00, r17 = r00, r18 = r00, r19 = r00;
                    float32x4_t r20 = r00, r21 = r00, r22 = r00, r23 = r00;
                    float32x4_t r24 = r00, r25 = r00, r26 = r00, r27 = r00;
                    float32x4_t r28 = r00, r29 = r00, r30 = r00, r31 = r00;

                    for(;nn > 0; nn--)
                    {
                        r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4), r02 = vld1q_f32(r0+8), r03 = vld1q_f32(r0+12);
                        r08 = vld1q_f32(k01), r09 = vld1q_f32(k01+4), r10 = vld1q_f32(k01+8), r11 = vld1q_f32(k01+12);
                        r0 += 16, k01 += 16;

                        r16 = vfmaq_laneq_f32(r16, r08, r00, 0);
                        r17 = vfmaq_laneq_f32(r17, r08, r01, 0);
                        r18 = vfmaq_laneq_f32(r18, r08, r02, 0);
                        r19 = vfmaq_laneq_f32(r19, r08, r03, 0);

                        r04 = vld1q_f32(r0), r05 = vld1q_f32(r0+4), r06 = vld1q_f32(r0+8), r07 = vld1q_f32(r0+12);
                        r0 += 16;

                        r20 = vfmaq_laneq_f32(r20, r08, r04, 0);
                        r21 = vfmaq_laneq_f32(r21, r08, r05, 0);
                        r22 = vfmaq_laneq_f32(r22, r08, r06, 0);
                        r23 = vfmaq_laneq_f32(r23, r08, r07, 0);

                        r24 = vfmaq_laneq_f32(r24, r09, r00, 0);
                        r25 = vfmaq_laneq_f32(r25, r09, r01, 0);
                        r26 = vfmaq_laneq_f32(r26, r09, r02, 0);
                        r27 = vfmaq_laneq_f32(r27, r09, r03, 0);
                        r28 = vfmaq_laneq_f32(r28, r09, r04, 0);
                        r29 = vfmaq_laneq_f32(r29, r09, r05, 0);
                        r30 = vfmaq_laneq_f32(r30, r09, r06, 0);
                        r31 = vfmaq_laneq_f32(r31, r09, r07, 0);

                        r12 = vld1q_f32(k01), r13 = vld1q_f32(k01+4), r14 = vld1q_f32(k01+8), r15 = vld1q_f32(k01+12);
                        k01 += 16;

                        r16 = vfmaq_laneq_f32(r16, r10, r00, 1);
                        r17 = vfmaq_laneq_f32(r17, r10, r01, 1);
                        r18 = vfmaq_laneq_f32(r18, r10, r02, 1);
                        r19 = vfmaq_laneq_f32(r19, r10, r03, 1);
                        r20 = vfmaq_laneq_f32(r20, r10, r04, 1);
                        r21 = vfmaq_laneq_f32(r21, r10, r05, 1);
                        r22 = vfmaq_laneq_f32(r22, r10, r06, 1);
                        r23 = vfmaq_laneq_f32(r23, r10, r07, 1);

                        r24 = vfmaq_laneq_f32(r24, r11, r00, 1);
                        r25 = vfmaq_laneq_f32(r25, r11, r01, 1);
                        r26 = vfmaq_laneq_f32(r26, r11, r02, 1);
                        r27 = vfmaq_laneq_f32(r27, r11, r03, 1);
                        r28 = vfmaq_laneq_f32(r28, r11, r04, 1);
                        r29 = vfmaq_laneq_f32(r29, r11, r05, 1);
                        r30 = vfmaq_laneq_f32(r30, r11, r06, 1);
                        r31 = vfmaq_laneq_f32(r31, r11, r07, 1);

                        r16 = vfmaq_laneq_f32(r16, r12, r00, 2);
                        r17 = vfmaq_laneq_f32(r17, r12, r01, 2);
                        r18 = vfmaq_laneq_f32(r18, r12, r02, 2);
                        r19 = vfmaq_laneq_f32(r19, r12, r03, 2);
                        r20 = vfmaq_laneq_f32(r20, r12, r04, 2);
                        r21 = vfmaq_laneq_f32(r21, r12, r05, 2);
                        r22 = vfmaq_laneq_f32(r22, r12, r06, 2);
                        r23 = vfmaq_laneq_f32(r23, r12, r07, 2);

                        r24 = vfmaq_laneq_f32(r24, r13, r00, 2);
                        r25 = vfmaq_laneq_f32(r25, r13, r01, 2);
                        r26 = vfmaq_laneq_f32(r26, r13, r02, 2);
                        r27 = vfmaq_laneq_f32(r27, r13, r03, 2);
                        r28 = vfmaq_laneq_f32(r28, r13, r04, 2);
                        r29 = vfmaq_laneq_f32(r29, r13, r05, 2);
                        r30 = vfmaq_laneq_f32(r30, r13, r06, 2);
                        r31 = vfmaq_laneq_f32(r31, r13, r07, 2);

                        r16 = vfmaq_laneq_f32(r16, r14, r00, 3);
                        r17 = vfmaq_laneq_f32(r17, r14, r01, 3);
                        r18 = vfmaq_laneq_f32(r18, r14, r02, 3);
                        r19 = vfmaq_laneq_f32(r19, r14, r03, 3);
                        r20 = vfmaq_laneq_f32(r20, r14, r04, 3);
                        r21 = vfmaq_laneq_f32(r21, r14, r05, 3);
                        r22 = vfmaq_laneq_f32(r22, r14, r06, 3);
                        r23 = vfmaq_laneq_f32(r23, r14, r07, 3);

                        r24 = vfmaq_laneq_f32(r24, r15, r00, 3);
                        r25 = vfmaq_laneq_f32(r25, r15, r01, 3);
                        r26 = vfmaq_laneq_f32(r26, r15, r02, 3);
                        r27 = vfmaq_laneq_f32(r27, r15, r03, 3);
                        r28 = vfmaq_laneq_f32(r28, r15, r04, 3);
                        r29 = vfmaq_laneq_f32(r29, r15, r05, 3);
                        r30 = vfmaq_laneq_f32(r30, r15, r06, 3);
                        r31 = vfmaq_laneq_f32(r31, r15, r07, 3);
                    }

                    vst1q_f32(output0_tm, r16), vst1q_f32(output0_tm + 4, r17), vst1q_f32(output0_tm + 8, r18), vst1q_f32(output0_tm + 12, r19);
                    output0_tm += 16;
                    vst1q_f32(output1_tm, r24), vst1q_f32(output1_tm + 4, r25), vst1q_f32(output1_tm + 8, r26), vst1q_f32(output1_tm + 12, r27);
                    output1_tm += 16;

                    vst1q_f32(output0_tm, r20), vst1q_f32(output0_tm + 4, r21), vst1q_f32(output0_tm + 8, r22), vst1q_f32(output0_tm + 12, r23);
                    output0_tm += 16;
                    vst1q_f32(output1_tm, r28), vst1q_f32(output1_tm + 4, r29), vst1q_f32(output1_tm + 8, r30), vst1q_f32(output1_tm + 12, r31);
                    output1_tm += 16;
                }

                for (; ti + 3 < tiles; ti += 4)
                {
                    const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                    const float* k01 = kernel_tm_i;

                    int nn = C_aligned/4;

                    // init 20 registers. FMA/load ratio = 32/12
                    float32x4_t r00 = vdupq_n_f32(0.0f), r01 = r00, r02 = r00, r03 = r00;
                    float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                    float32x4_t r12 = r00, r13 = r00, r14 = r00, r15 = r00;
                    float32x4_t r24 = r00, r25 = r00, r26 = r00, r27 = r00;
                    float32x4_t r28 = r00, r29 = r00, r30 = r00, r31 = r00;

                    for(; nn > 0; nn--)
                    {
                        r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4), r02 = vld1q_f32(r0+8), r03 = vld1q_f32(r0+12);
                        r08 = vld1q_f32(k01), r09 = vld1q_f32(k01+4), r10 = vld1q_f32(k01+8), r11 = vld1q_f32(k01+12);
                        r0 += 16, k01 += 16;

                        r24 = vfmaq_laneq_f32(r24, r08, r00, 0);
                        r25 = vfmaq_laneq_f32(r25, r08, r01, 0);
                        r26 = vfmaq_laneq_f32(r26, r08, r02, 0);
                        r27 = vfmaq_laneq_f32(r27, r08, r03, 0);

                        r28 = vfmaq_laneq_f32(r28, r09, r00, 0);
                        r29 = vfmaq_laneq_f32(r29, r09, r01, 0);
                        r30 = vfmaq_laneq_f32(r30, r09, r02, 0);
                        r31 = vfmaq_laneq_f32(r31, r09, r03, 0);

                        r12 = vld1q_f32(k01), r13 = vld1q_f32(k01+4), r14 = vld1q_f32(k01+8), r15 = vld1q_f32(k01+12);
                        k01 += 16;

                        r24 = vfmaq_laneq_f32(r24, r10, r00, 1);
                        r25 = vfmaq_laneq_f32(r25, r10, r01, 1);
                        r26 = vfmaq_laneq_f32(r26, r10, r02, 1);
                        r27 = vfmaq_laneq_f32(r27, r10, r03, 1);

                        r28 = vfmaq_laneq_f32(r28, r11, r00, 1);
                        r29 = vfmaq_laneq_f32(r29, r11, r01, 1);
                        r30 = vfmaq_laneq_f32(r30, r11, r02, 1);
                        r31 = vfmaq_laneq_f32(r31, r11, r03, 1);

                        r24 = vfmaq_laneq_f32(r24, r12, r00, 2);
                        r25 = vfmaq_laneq_f32(r25, r12, r01, 2);
                        r26 = vfmaq_laneq_f32(r26, r12, r02, 2);
                        r27 = vfmaq_laneq_f32(r27, r12, r03, 2);

                        r28 = vfmaq_laneq_f32(r28, r13, r00, 2);
                        r29 = vfmaq_laneq_f32(r29, r13, r01, 2);
                        r30 = vfmaq_laneq_f32(r30, r13, r02, 2);
                        r31 = vfmaq_laneq_f32(r31, r13, r03, 2);

                        r24 = vfmaq_laneq_f32(r24, r14, r00, 3);
                        r25 = vfmaq_laneq_f32(r25, r14, r01, 3);
                        r26 = vfmaq_laneq_f32(r26, r14, r02, 3);
                        r27 = vfmaq_laneq_f32(r27, r14, r03, 3);

                        r28 = vfmaq_laneq_f32(r28, r15, r00, 3);
                        r29 = vfmaq_laneq_f32(r29, r15, r01, 3);
                        r30 = vfmaq_laneq_f32(r30, r15, r02, 3);
                        r31 = vfmaq_laneq_f32(r31, r15, r03, 3);
                    }

                    vst1q_f32(output0_tm, r24), vst1q_f32(output0_tm + 4, r25), vst1q_f32(output0_tm + 8, r26), vst1q_f32(output0_tm + 12, r27);
                    output0_tm += 16;
                    vst1q_f32(output1_tm, r28), vst1q_f32(output1_tm + 4, r29), vst1q_f32(output1_tm + 8, r30), vst1q_f32(output1_tm + 12, r31);
                    output1_tm += 16;
                }

                for (; ti + 1 < tiles; ti += 2)
                {
                    const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                    const float* k01 = kernel_tm_i;

                    int nn = C_aligned/4;

                    // init 14 registers. FMA/load ratio = 15/10
                    float32x4_t r00 = vdupq_n_f32(0.0f), r01 = r00;
                    float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                    float32x4_t r12 = r00, r13 = r00, r14 = r00, r15 = r00;
                    float32x4_t r24 = r00, r25 = r00;
                    float32x4_t r28 = r00, r29 = r00;

                    for (; nn > 0; nn--)
                    {
                        r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4);
                        r08 = vld1q_f32(k01), r09 = vld1q_f32(k01+4), r10 = vld1q_f32(k01+8), r11 = vld1q_f32(k01+12);
                        r0 += 8, k01 += 16;

                        r24 = vfmaq_laneq_f32(r24, r08, r00, 0);
                        r25 = vfmaq_laneq_f32(r25, r08, r01, 0);

                        r28 = vfmaq_laneq_f32(r28, r09, r00, 0);
                        r29 = vfmaq_laneq_f32(r29, r09, r01, 0);

                        r12 = vld1q_f32(k01), r13 = vld1q_f32(k01+4), r14 = vld1q_f32(k01+8), r15 = vld1q_f32(k01+12);
                        k01 += 16;

                        r24 = vfmaq_laneq_f32(r24, r10, r00, 1);
                        r25 = vfmaq_laneq_f32(r25, r10, r01, 1);

                        r28 = vfmaq_laneq_f32(r28, r11, r00, 1);
                        r29 = vfmaq_laneq_f32(r29, r11, r01, 1);

                        r24 = vfmaq_laneq_f32(r24, r12, r00, 2);
                        r25 = vfmaq_laneq_f32(r25, r12, r01, 2);

                        r28 = vfmaq_laneq_f32(r28, r13, r00, 2);
                        r29 = vfmaq_laneq_f32(r29, r13, r01, 2);

                        r24 = vfmaq_laneq_f32(r24, r14, r00, 3);
                        r25 = vfmaq_laneq_f32(r25, r14, r01, 3);

                        r28 = vfmaq_laneq_f32(r28, r15, r00, 3);
                        r29 = vfmaq_laneq_f32(r29, r15, r01, 3);
                    }

                    vst1q_f32(output0_tm, r24), vst1q_f32(output0_tm + 4, r25);
                    output0_tm += 8;
                    vst1q_f32(output1_tm, r28), vst1q_f32(output1_tm + 4, r29);
                    output1_tm += 8;
                }

                for (; ti < tiles; ti ++)
                {
                    const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                    const float* k01 = kernel_tm_i;

                    int nn = C_aligned/4;

                    float32x4_t r00 = vdupq_n_f32(0.0f);
                    float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                    float32x4_t r12 = r00, r13 = r00, r14 = r00, r15 = r00;
                    float32x4_t r24 = r00;
                    float32x4_t r28 = r00;

                    for(;nn > 0; nn--)
                    {
                        r00 = vld1q_f32(r0);
                        r08 = vld1q_f32(k01), r09 = vld1q_f32(k01+4), r10 = vld1q_f32(k01+8), r11 = vld1q_f32(k01+12);
                        r0 += 4, k01 += 16;

                        r24 = vfmaq_laneq_f32(r24, r08, r00, 0);
                        r28 = vfmaq_laneq_f32(r28, r09, r00, 0);

                        r12 = vld1q_f32(k01), r13 = vld1q_f32(k01+4), r14 = vld1q_f32(k01+8), r15 = vld1q_f32(k01+12);
                        k01 += 16;

                        r24 = vfmaq_laneq_f32(r24, r10, r00, 1);
                        r28 = vfmaq_laneq_f32(r28, r11, r00, 1);

                        r24 = vfmaq_laneq_f32(r24, r12, r00, 2);
                        r28 = vfmaq_laneq_f32(r28, r13, r00, 2);

                        r24 = vfmaq_laneq_f32(r24, r14, r00, 3);
                        r28 = vfmaq_laneq_f32(r28, r15, r00, 3);
                    }

                    vst1q_f32(output0_tm, r24);
                    output0_tm += 4;
                    vst1q_f32(output1_tm, r28);
                    output1_tm += 4;
                }
            }
        }
        });
#endif

        // Matrix multiplication, 4 output channel.
        int Ock_div4 = (K_aligned - K_div8 * 8) / 4;
        parallel_for_(Range(0, 64), [&](const Range &range){
            for (int r = range.start; r < range.end; r++)
            {
                float* input_tm = inputbuf0 + r * big_step;
                float* output_tmp = outputbuf0 + tiles * K_aligned * r;
                float* kernel_tmp = weight_ptr0 + r * C_aligned * K_aligned;

                for (int out_div4 = 0; out_div4 < Ock_div4; out_div4 ++)
                {
                    float* output0_tm = output_tmp + tiles * (out_div4 + K_div8 * 2) * 4 ;
                    float* kernel_tm_i = kernel_tmp + (out_div4 + K_div8 * 2) * 4 * C_aligned;

                    int ti = 0;
                    for (; ti + 7 < tiles; ti += 8)
                    {
                        int nn = C_aligned/4;
                        const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                        const float* k0 = kernel_tm_i;

#if CV_NEON_AARCH64
                        // init 24 registers. FMA/load ratio = 32/12
                        float32x4_t r00 = vdupq_n_f32(0.0f), r01 = r00, r02 = r00, r03 = r00;
                        float32x4_t r04 = r00, r05 = r00, r06 = r00, r07 = r00;
                        float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                        float32x4_t r16 = r00, r17 = r00, r18 = r00, r19 = r00;
                        float32x4_t r20 = r00, r21 = r00, r22 = r00, r23 = r00;

                        for(; nn > 0; nn--)
                        {
                            r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4), r02 = vld1q_f32(r0+8), r03 = vld1q_f32(r0+12);
                            r08 = vld1q_f32(k0), r09 = vld1q_f32(k0+4), r10 = vld1q_f32(k0+8), r11 = vld1q_f32(k0+12);
                            r0 += 16, k0 += 16;

                            r16 = vfmaq_laneq_f32(r16, r08, r00, 0);
                            r17 = vfmaq_laneq_f32(r17, r08, r01, 0);
                            r18 = vfmaq_laneq_f32(r18, r08, r02, 0);
                            r19 = vfmaq_laneq_f32(r19, r08, r03, 0);

                            r04 = vld1q_f32(r0), r05 = vld1q_f32(r0+4), r06 = vld1q_f32(r0+8), r07 = vld1q_f32(r0+12);
                            r0 += 16;

                            r20 = vfmaq_laneq_f32(r20, r08, r04, 0);
                            r21 = vfmaq_laneq_f32(r21, r08, r05, 0);
                            r22 = vfmaq_laneq_f32(r22, r08, r06, 0);
                            r23 = vfmaq_laneq_f32(r23, r08, r07, 0);

                            r16 = vfmaq_laneq_f32(r16, r09, r00, 1);
                            r17 = vfmaq_laneq_f32(r17, r09, r01, 1);
                            r18 = vfmaq_laneq_f32(r18, r09, r02, 1);
                            r19 = vfmaq_laneq_f32(r19, r09, r03, 1);
                            r20 = vfmaq_laneq_f32(r20, r09, r04, 1);
                            r21 = vfmaq_laneq_f32(r21, r09, r05, 1);
                            r22 = vfmaq_laneq_f32(r22, r09, r06, 1);
                            r23 = vfmaq_laneq_f32(r23, r09, r07, 1);

                            r16 = vfmaq_laneq_f32(r16, r10, r00, 2);
                            r17 = vfmaq_laneq_f32(r17, r10, r01, 2);
                            r18 = vfmaq_laneq_f32(r18, r10, r02, 2);
                            r19 = vfmaq_laneq_f32(r19, r10, r03, 2);
                            r20 = vfmaq_laneq_f32(r20, r10, r04, 2);
                            r21 = vfmaq_laneq_f32(r21, r10, r05, 2);
                            r22 = vfmaq_laneq_f32(r22, r10, r06, 2);
                            r23 = vfmaq_laneq_f32(r23, r10, r07, 2);

                            r16 = vfmaq_laneq_f32(r16, r11, r00, 3);
                            r17 = vfmaq_laneq_f32(r17, r11, r01, 3);
                            r18 = vfmaq_laneq_f32(r18, r11, r02, 3);
                            r19 = vfmaq_laneq_f32(r19, r11, r03, 3);
                            r20 = vfmaq_laneq_f32(r20, r11, r04, 3);
                            r21 = vfmaq_laneq_f32(r21, r11, r05, 3);
                            r22 = vfmaq_laneq_f32(r22, r11, r06, 3);
                            r23 = vfmaq_laneq_f32(r23, r11, r07, 3);
                        }

                        vst1q_f32(output0_tm, r16), vst1q_f32(output0_tm + 4, r17), vst1q_f32(output0_tm + 8, r18), vst1q_f32(output0_tm + 12, r19);
                        output0_tm += 16;

                        vst1q_f32(output0_tm, r20), vst1q_f32(output0_tm + 4, r21), vst1q_f32(output0_tm + 8, r22), vst1q_f32(output0_tm + 12, r23);
                        output0_tm += 16;

#else // ARMv7 16 registers.

                        // init 16 registers. FMA/load ratio = 32/12
                        float32x2_t q00 = vdup_n_f32(0.0f), q01 = q00, q02 = q00, q03 = q00,
                                    q04 = q00, q05 = q00, q06 = q00, q07 = q00;

                        float32x4_t r04 = vdupq_n_f32(0.0f), r05 = r04, r06 = r04, r07 = r04;
                        float32x4_t r08 = r04, r09 = r04, r10 = r04, r11 = r04;
                        float32x4_t r12 = r04, r13 = r04, r14 = r04, r15 = r04;

                        for (; nn > 0; nn--)
                        {
                            q00 = vld1_f32(r0), q01 = vld1_f32(r0+2), q02 = vld1_f32(r0+4), q03 = vld1_f32(r0+6);
                            q04 = vld1_f32(r0+8), q05 = vld1_f32(r0+10), q06 = vld1_f32(r0+12), q07 = vld1_f32(r0+14);
                            r04 = vld1q_f32(k0), r05 = vld1q_f32(k0+4), r06 = vld1q_f32(k0+8), r07 = vld1q_f32(k0+12);
                            r0 += 16, k0 += 16;

                            r08 = vmlaq_lane_f32(r08, r04, q00, 0);
                            r09 = vmlaq_lane_f32(r09, r04, q02, 0);
                            r10 = vmlaq_lane_f32(r10, r04, q04, 0);
                            r11 = vmlaq_lane_f32(r11, r04, q06, 0);

                            r08 = vmlaq_lane_f32(r08, r05, q00, 1);
                            r09 = vmlaq_lane_f32(r09, r05, q02, 1);
                            r10 = vmlaq_lane_f32(r10, r05, q04, 1);
                            r11 = vmlaq_lane_f32(r11, r05, q06, 1);

                            r08 = vmlaq_lane_f32(r08, r06, q01, 0);
                            r09 = vmlaq_lane_f32(r09, r06, q03, 0);
                            r10 = vmlaq_lane_f32(r10, r06, q05, 0);
                            r11 = vmlaq_lane_f32(r11, r06, q07, 0);

                            r08 = vmlaq_lane_f32(r08, r07, q01, 1);
                            r09 = vmlaq_lane_f32(r09, r07, q03, 1);
                            r10 = vmlaq_lane_f32(r10, r07, q05, 1);
                            r11 = vmlaq_lane_f32(r11, r07, q07, 1);

                            q00 = vld1_f32(r0), q01 = vld1_f32(r0+2), q02 = vld1_f32(r0+4), q03 = vld1_f32(r0+6);
                            q04 = vld1_f32(r0+8), q05 = vld1_f32(r0+10), q06 = vld1_f32(r0+12), q07 = vld1_f32(r0+14);
                            r0 += 16;

                            r12 = vmlaq_lane_f32(r12, r04, q00, 0);
                            r13 = vmlaq_lane_f32(r13, r04, q02, 0);
                            r14 = vmlaq_lane_f32(r14, r04, q04, 0);
                            r15 = vmlaq_lane_f32(r15, r04, q06, 0);

                            r12 = vmlaq_lane_f32(r12, r05, q00, 1);
                            r13 = vmlaq_lane_f32(r13, r05, q02, 1);
                            r14 = vmlaq_lane_f32(r14, r05, q04, 1);
                            r15 = vmlaq_lane_f32(r15, r05, q06, 1);

                            r12 = vmlaq_lane_f32(r12, r06, q01, 0);
                            r13 = vmlaq_lane_f32(r13, r06, q03, 0);
                            r14 = vmlaq_lane_f32(r14, r06, q05, 0);
                            r15 = vmlaq_lane_f32(r15, r06, q07, 0);

                            r12 = vmlaq_lane_f32(r12, r07, q01, 1);
                            r13 = vmlaq_lane_f32(r13, r07, q03, 1);
                            r14 = vmlaq_lane_f32(r14, r07, q05, 1);
                            r15 = vmlaq_lane_f32(r15, r07, q07, 1);
                        }

                        vst1q_f32(output0_tm, r08), vst1q_f32(output0_tm + 4, r09), vst1q_f32(output0_tm + 8, r10), vst1q_f32(output0_tm + 12, r11);
                        output0_tm += 16;

                        vst1q_f32(output0_tm, r12), vst1q_f32(output0_tm + 4, r13), vst1q_f32(output0_tm + 8, r14), vst1q_f32(output0_tm + 12, r15);
                        output0_tm += 16;
#endif
                    }

                    for (; ti + 3 < tiles; ti += 4)
                    {
                        int nn = C_aligned/4;
                        const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                        const float* k0 = kernel_tm_i;

#if CV_NEON_AARCH64
                        // init 12 registers. FMA/load ratio = 12/8
                        float32x4_t r00 = vdupq_n_f32(0.0f), r01 = r00, r02 = r00, r03 = r00;
                        float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                        float32x4_t r16 = r00, r17 = r00, r18 = r00, r19 = r00;

                        for(; nn > 0; nn--)
                        {
                            r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4), r02 = vld1q_f32(r0+8), r03 = vld1q_f32(r0+12);
                            r08 = vld1q_f32(k0), r09 = vld1q_f32(k0+4), r10 = vld1q_f32(k0+8), r11 = vld1q_f32(k0+12);
                            r0 += 16, k0 += 16;

                            r16 = vfmaq_laneq_f32(r16, r08, r00, 0);
                            r17 = vfmaq_laneq_f32(r17, r08, r01, 0);
                            r18 = vfmaq_laneq_f32(r18, r08, r02, 0);
                            r19 = vfmaq_laneq_f32(r19, r08, r03, 0);

                            r16 = vfmaq_laneq_f32(r16, r09, r00, 1);
                            r17 = vfmaq_laneq_f32(r17, r09, r01, 1);
                            r18 = vfmaq_laneq_f32(r18, r09, r02, 1);
                            r19 = vfmaq_laneq_f32(r19, r09, r03, 1);

                            r16 = vfmaq_laneq_f32(r16, r10, r00, 2);
                            r17 = vfmaq_laneq_f32(r17, r10, r01, 2);
                            r18 = vfmaq_laneq_f32(r18, r10, r02, 2);
                            r19 = vfmaq_laneq_f32(r19, r10, r03, 2);

                            r16 = vfmaq_laneq_f32(r16, r11, r00, 3);
                            r17 = vfmaq_laneq_f32(r17, r11, r01, 3);
                            r18 = vfmaq_laneq_f32(r18, r11, r02, 3);
                            r19 = vfmaq_laneq_f32(r19, r11, r03, 3);
                        }
#else
                        // init 12 registers. FMA/load ratio = 12/8
                        float32x2_t q00 = vdup_n_f32(0.0f), q01 = q00, q02 = q00, q03 = q00,
                                q04 = q00, q05 = q00, q06 = q00, q07 = q00;
                        float32x4_t r08 = vdupq_n_f32(0.0f), r09 = r08, r10 = r08, r11 = r08;
                        float32x4_t r16 = r08, r17 = r08, r18 = r08, r19 = r08;

                        for(; nn > 0; nn--)
                        {
                            q00 = vld1_f32(r0), q01 = vld1_f32(r0+2), q02 = vld1_f32(r0+4), q03 = vld1_f32(r0+6);
                            q04 = vld1_f32(r0+8), q05 = vld1_f32(r0+10), q06 = vld1_f32(r0+12), q07 = vld1_f32(r0+14);
                            r08 = vld1q_f32(k0), r09 = vld1q_f32(k0+4), r10 = vld1q_f32(k0+8), r11 = vld1q_f32(k0+12);
                            r0 += 16, k0 += 16;

                            r16 = vmlaq_lane_f32(r16, r08, q00, 0);
                            r17 = vmlaq_lane_f32(r17, r08, q02, 0);
                            r18 = vmlaq_lane_f32(r18, r08, q04, 0);
                            r19 = vmlaq_lane_f32(r19, r08, q06, 0);

                            r16 = vmlaq_lane_f32(r16, r09, q00, 1);
                            r17 = vmlaq_lane_f32(r17, r09, q02, 1);
                            r18 = vmlaq_lane_f32(r18, r09, q04, 1);
                            r19 = vmlaq_lane_f32(r19, r09, q06, 1);

                            r16 = vmlaq_lane_f32(r16, r10, q01, 0);
                            r17 = vmlaq_lane_f32(r17, r10, q03, 0);
                            r18 = vmlaq_lane_f32(r18, r10, q05, 0);
                            r19 = vmlaq_lane_f32(r19, r10, q07, 0);

                            r16 = vmlaq_lane_f32(r16, r11, q01, 1);
                            r17 = vmlaq_lane_f32(r17, r11, q03, 1);
                            r18 = vmlaq_lane_f32(r18, r11, q05, 1);
                            r19 = vmlaq_lane_f32(r19, r11, q07, 1);

                        }
#endif
                        vst1q_f32(output0_tm, r16), vst1q_f32(output0_tm + 4, r17), vst1q_f32(output0_tm + 8, r18), vst1q_f32(output0_tm + 12, r19);
                        output0_tm += 16;
                    }

                    for (; ti + 1 < tiles; ti += 2)
                    {
                        int nn = C_aligned/4;
                        const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                        const float* k0 = kernel_tm_i;

#if CV_NEON_AARCH64
                        // init 8 registers. FMA/load ratio = 8/6
                        float32x4_t r00 = vdupq_n_f32(0.0f), r01 = r00;
                        float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                        float32x4_t r16 = r00, r17 = r00;

                        for(; nn > 0; nn--)
                        {
                            r00 = vld1q_f32(r0), r01 = vld1q_f32(r0+4);
                            r08 = vld1q_f32(k0), r09 = vld1q_f32(k0+4), r10 = vld1q_f32(k0+8), r11 = vld1q_f32(k0+12);
                            r0 += 8, k0 += 16;

                            r16 = vfmaq_laneq_f32(r16, r08, r00, 0);
                            r17 = vfmaq_laneq_f32(r17, r08, r01, 0);

                            r16 = vfmaq_laneq_f32(r16, r09, r00, 1);
                            r17 = vfmaq_laneq_f32(r17, r09, r01, 1);

                            r16 = vfmaq_laneq_f32(r16, r10, r00, 2);
                            r17 = vfmaq_laneq_f32(r17, r10, r01, 2);

                            r16 = vfmaq_laneq_f32(r16, r11, r00, 3);
                            r17 = vfmaq_laneq_f32(r17, r11, r01, 3);
                        }
#else
                        // init 8 registers. FMA/load ratio = 8/6
                        float32x2_t q00 = vdup_n_f32(0.0f), q01 = q00, q02 = q00, q03 = q00;
                        float32x4_t r08 = vdupq_n_f32(0.0f), r09 = r08, r10 = r08, r11 = r08;
                        float32x4_t r16 = r08, r17 = r08;

                        for(; nn > 0; nn--)
                        {
                            q00 = vld1_f32(r0), q01 = vld1_f32(r0+2), q02 = vld1_f32(r0+4), q03 = vld1_f32(r0+6);
                            r08 = vld1q_f32(k0), r09 = vld1q_f32(k0+4), r10 = vld1q_f32(k0+8), r11 = vld1q_f32(k0+12);
                            r0 += 8, k0 += 16;

                            r16 = vmlaq_lane_f32(r16, r08, q00, 0);
                            r17 = vmlaq_lane_f32(r17, r08, q02, 0);

                            r16 = vmlaq_lane_f32(r16, r09, q00, 1);
                            r17 = vmlaq_lane_f32(r17, r09, q02, 1);

                            r16 = vmlaq_lane_f32(r16, r10, q01, 0);
                            r17 = vmlaq_lane_f32(r17, r10, q03, 0);

                            r16 = vmlaq_lane_f32(r16, r11, q01, 1);
                            r17 = vmlaq_lane_f32(r17, r11, q03, 1);
                        }
#endif
                        vst1q_f32(output0_tm, r16), vst1q_f32(output0_tm + 4, r17);
                        output0_tm += 8;
                    }

                    for (; ti < tiles; ti ++)
                    {
                        int nn = C_aligned/4;
                        const float* r0 = input_tm + ofstab0[ti * 2] * line_step;
                        const float* k0 = kernel_tm_i;

#if CV_NEON_AARCH64
                        // init 6 registers. FMA/load ratio = 6/5
                        float32x4_t r00 = vdupq_n_f32(0.0f);
                        float32x4_t r08 = r00, r09 = r00, r10 = r00, r11 = r00;
                        float32x4_t r16 = r00;

                        for(; nn > 0; nn--)
                        {
                            r00 = vld1q_f32(r0);
                            r08 = vld1q_f32(k0), r09 = vld1q_f32(k0+4), r10 = vld1q_f32(k0+8), r11 = vld1q_f32(k0+12);
                            r0 += 4, k0 += 16;

                            r16 = vfmaq_laneq_f32(r16, r08, r00, 0);
                            r16 = vfmaq_laneq_f32(r16, r09, r00, 1);
                            r16 = vfmaq_laneq_f32(r16, r10, r00, 2);
                            r16 = vfmaq_laneq_f32(r16, r11, r00, 3);
                        }
#else
                        // init 6 registers. FMA/load ratio = 6/5
                        float32x2_t q00 = vdup_n_f32(0.0f), q01 = q00;
                        float32x4_t r08 = vdupq_n_f32(0.0f), r09 = r08, r10 = r08, r11 = r08;
                        float32x4_t r16 = r08;

                        for(; nn > 0; nn--)
                        {
                            q00 = vld1_f32(r0), q01 = vld1_f32(r0+2);
                            r08 = vld1q_f32(k0), r09 = vld1q_f32(k0+4), r10 = vld1q_f32(k0+8), r11 = vld1q_f32(k0+12);
                            r0 += 4, k0 += 16;

                            r16 = vmlaq_lane_f32(r16, r08, q00, 0);
                            r16 = vmlaq_lane_f32(r16, r09, q00, 1);
                            r16 = vmlaq_lane_f32(r16, r10, q01, 0);
                            r16 = vmlaq_lane_f32(r16, r11, q01, 1);
                        }
#endif
                        vst1q_f32(output0_tm, r16);
                        output0_tm += 4;
                    }
                }
            }
        });

        int bigStepOut = tiles * K_aligned;
        AutoBuffer<float> _fAbuf;
        float* fAbuf0 = 0;
        if (fusedAddPtr0)
        {
            _fAbuf.allocate(6 * 6 * 4 * ntasks);
            fAbuf0 = _fAbuf.data();
        }

        // Transfor Ouput
        parallel_for_(Range(0, ntasks), [&](const Range& range)
        {
            for (int task_i = range.start; task_i < range.end; task_i++)
            {
                float* fAbuf = fAbuf0 ? fAbuf0 + task_i * 6 * 6 * 4 : 0;
                float* outputCnbuf = outputCnbuf0 + task_i * 8 * 8 * 4;
                for (int outCn4 = task_i; outCn4 < K_aligned / 4; outCn4 += ntasks)
                {

                    int outCn = outCn4 * 4;
                    float* output_buf = outputbuf0 + outCn * tiles;
                    float* output_ptr = output_ptr0 + outCn * W0 * H0;
                    float* fusedAddPtr = fusedAddPtr0 + outCn * W0 * H0;

                    for (int ti = 0; ti < tiles; ti++)
                    {
                        float* output_buf_i = output_buf + ti * 4;
                        float* outputCnbuf_i = outputCnbuf;
                        int hi = ti / W_tiles;
                        int wi = ti % W_tiles;

                        int wEnd = (wi + 1) * 6 > W0 ? W0 - (wi * 6) : 6;
                        int hEnd = (hi + 1) * 6 > H0 ? H0 - (hi * 6) : 6;

                        // construct the output tile.
                        for (int r = 0; r < 64; r++)
                        {
                            memcpy(outputCnbuf_i, output_buf_i, FAST_VEC_NLANES * sizeof(float ));
                            output_buf_i += bigStepOut;
                            outputCnbuf_i += FAST_VEC_NLANES;
                        }

                        // construct the fusedAdd buffer.
                        if (fAbuf && fusedAddPtr0)
                        {
                            memset(fAbuf, 0, sizeof(fAbuf[0]) * 6 * 6 * 4);
                            float* fAPtr = fusedAddPtr + (hi * W0 + wi) * 6;
                            for (int outCni = 0; outCni < FAST_VEC_NLANES; outCni++)
                            {
                                float* fAbufCnPtr = fAPtr + outCni * out_planesize; // skip channel
                                for (int i = 0; i < hEnd; i++)
                                {
                                    for (int j = 0; j < wEnd; j++)
                                    {
                                        fAbuf[(i * 6 + j) * FAST_VEC_NLANES + outCni] = fAbufCnPtr[i * W0 + j];
                                    }
                                }
                            }
                        }

                        winograd_trans_output_F63(outputCnbuf, conv->biasBuf.data() + outCn, fAbuf,
                                                  minval, maxval, ifMinMaxAct);

                        float* output_ptr_i = output_ptr + (hi * W0 + wi) * 6;

                        // write back the output data.
                        for (int outCni = 0; outCni < FAST_VEC_NLANES; outCni++)
                        {
                            float* output_ptr_i_cn = output_ptr_i + outCni * out_planesize;
                            outputCnbuf_i = outputCnbuf + outCni;

                            if (outCni + outCn < K)
                            {
                                for (int i = 0; i < hEnd; i++)
                                {
                                    for (int j = 0; j < wEnd; j++)
                                    {
                                        output_ptr_i_cn[i * W0 + j] = outputCnbuf_i[(i * 6 + j) * FAST_VEC_NLANES ];
                                    }
                                }
                            }
                        }
                    }

                    if (activ)
                    {
                        int outCnEnd = std::min(outCn + FAST_VEC_NLANES, K);
                        activ->forwardSlice(output_ptr, output_ptr, out_planesize,
                                                  out_planesize, outCn, outCnEnd);
                    }
                }
            }
        });
    }
    return 1;
}
#else

void initWinograd63(Ptr<FastConv2d>& conv, InputArray _weightsMat, int K, int C)
{
    conv->ifWinograd63 = false;
}

int runWinograd63(InputArray _input, OutputArray _output, const Ptr<FastConv2d>& conv, int ntasks, float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct)
{
    return 0;
}

#endif

}} // namespace cv::dnn
