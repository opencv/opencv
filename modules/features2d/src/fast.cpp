/* This is FAST corner detector, contributed to OpenCV by the author, Edward Rosten.
   Below is the original copyright and the references */

/*
Copyright (c) 2006, 2008 Edward Rosten
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

    *Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

    *Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

    *Neither the name of the University of Cambridge nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
The references are:
 * Machine learning for high-speed corner detection,
   E. Rosten and T. Drummond, ECCV 2006
 * Faster and better: A machine learning approach to corner detection
   E. Rosten, R. Porter and T. Drummond, PAMI, 2009
*/

#include "precomp.hpp"
#include "fast.hpp"
#include "fast_score.hpp"
#include "opencl_kernels_features2d.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"
namespace cv
{

#ifdef __RVV // 16.31 ms
#include <riscv_vector.h>

/* 16 点 FAST 主函数，替换 FAST_t<16> */
void FAST16_RVV(InputArray _img,
                 std::vector<KeyPoint>& keypoints,
                 int threshold,
                 bool nonmax_suppression)
{
    Mat img = _img.getMat();
    int pixel[25];
    makeOffsets(pixel, (int)img.step, 16);   // OpenCV 内部函数

    keypoints.clear();
    threshold = std::min(std::max(threshold, 0), 255);

    /* 三行缓冲 */
    int16_t *buf[3] = {nullptr};
    int   *cpbuf[3] = {nullptr};
    cv::utils::BufferArea area;
    for (int k = 0; k < 3; ++k) {
        area.allocate(buf[k],   img.cols);
        area.allocate(cpbuf[k], img.cols + 1);
    }
    area.commit();
    for (int k = 0; k < 3; ++k)
        memset(buf[k], 0, img.cols * sizeof(int16_t));

    /* 主循环 */
    for (int i = 3; i < img.rows - 2; ++i) {
        const uchar *ptr      = img.ptr<uchar>(i) + 3;
        int16_t       *curr     = buf[(i - 3) % 3];
        int         *cornerpos = cpbuf[(i - 3) % 3] + 1;
        int          ncorners  = 0;
        memset(curr, 0, img.cols * sizeof(int16_t));

        if (i < img.rows - 3) {
            int j = 3;
            size_t vl;
            /* 64 像素大粒度 */
            for (; j < img.cols - 3; j += vl, ptr += vl) {
                vl = __riscv_vsetvl_e16m1(img.cols - 3 - j);
                /* 快速 4 方向预筛选 */
                vint16m1_t vcen = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr, vl), vl));
                /* 载入上下阈值 */
                vint16m1_t vlo  = __riscv_vsub_vx_i16m1(vcen, threshold, vl);
                vint16m1_t vhi  = __riscv_vadd_vx_i16m1(vcen, threshold, vl);
                /* 载入中心值、上下阈值 */
                vint16m1_t vk0 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[0], vl), vl));
                vint16m1_t vk4 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[4], vl), vl));
                vint16m1_t vk8 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[8], vl), vl));
                vint16m1_t vk12 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[12], vl), vl));

                vbool16_t bright =__riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk0, vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk4, vhi, vl), vl);
                vbool16_t dark = __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk0, vl), __riscv_vmsgt_vv_i16m1_b16(vlo, vk4, vl), vl);
                bright = __riscv_vmor_mm_b16(bright, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk4, vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk8, vhi, vl), vl), vl);
                dark = __riscv_vmor_mm_b16(dark, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk4, vl), __riscv_vmsgt_vv_i16m1_b16(vlo, vk8, vl), vl), vl);
                bright = __riscv_vmor_mm_b16(bright, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk8, vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk12, vhi, vl), vl), vl);
                dark = __riscv_vmor_mm_b16(dark, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk8, vl), __riscv_vmsgt_vv_i16m1_b16(vlo, vk12, vl), vl), vl);
                bright = __riscv_vmor_mm_b16(bright, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk12, vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk0, vhi, vl), vl), vl);
                dark = __riscv_vmor_mm_b16(dark, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk12, vl), __riscv_vmsgt_vv_i16m1_b16(vlo, vk0, vl), vl), vl);

                /* 快速reject */
                if (__riscv_vfirst_m_b16(__riscv_vmor_mm_b16(bright, dark, vl), vl) < 0) continue;

                vint16m1_t vk1 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[1], vl), vl));
                vint16m1_t vk2 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[2], vl), vl));
                vint16m1_t vk3 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[3], vl), vl));

                vint16m1_t vk5 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[5], vl), vl));
                vint16m1_t vk6 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[6], vl), vl));
                vint16m1_t vk7 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[7], vl), vl));

                vint16m1_t vk9 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[9], vl), vl));
                vint16m1_t vk10 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[10], vl), vl));
                vint16m1_t vk11 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[11], vl), vl));

                vint16m1_t vk13 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[13], vl), vl));
                vint16m1_t vk14 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[14], vl), vl));
                vint16m1_t vk15 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[15], vl), vl));

                /** 中心像素 - 周边像素 d[i] = vcen - v[i]*/
                vint16m1_t d0    = __riscv_vsub_vv_i16m1(vcen, vk0, vl);
                vint16m1_t d1    = __riscv_vsub_vv_i16m1(vcen, vk1, vl);
                vint16m1_t d2    = __riscv_vsub_vv_i16m1(vcen, vk2, vl);
                vint16m1_t d3    = __riscv_vsub_vv_i16m1(vcen, vk3, vl);
                vint16m1_t d4    = __riscv_vsub_vv_i16m1(vcen, vk4, vl);
                vint16m1_t d5    = __riscv_vsub_vv_i16m1(vcen, vk5, vl);
                vint16m1_t d6    = __riscv_vsub_vv_i16m1(vcen, vk6, vl);
                vint16m1_t d7    = __riscv_vsub_vv_i16m1(vcen, vk7, vl);
                vint16m1_t d8    = __riscv_vsub_vv_i16m1(vcen, vk8, vl);
                vint16m1_t d9    = __riscv_vsub_vv_i16m1(vcen, vk9, vl);
                vint16m1_t d10    = __riscv_vsub_vv_i16m1(vcen, vk10, vl);
                vint16m1_t d11    = __riscv_vsub_vv_i16m1(vcen, vk11, vl);
                vint16m1_t d12    = __riscv_vsub_vv_i16m1(vcen, vk12, vl);
                vint16m1_t d13    = __riscv_vsub_vv_i16m1(vcen, vk13, vl);
                vint16m1_t d14    = __riscv_vsub_vv_i16m1(vcen, vk14, vl);
                vint16m1_t d15    = __riscv_vsub_vv_i16m1(vcen, vk15, vl);

                vint16m1_t va = __riscv_vmin_vv_i16m1(d7, d8, vl);
                vint16m1_t vb = __riscv_vmax_vv_i16m1(d7, d8, vl);
                vint16m1_t va0 = __riscv_vmin_vv_i16m1(va, d6, vl);
                vint16m1_t vb0 = __riscv_vmax_vv_i16m1(vb, d6, vl);
                vint16m1_t va1 = __riscv_vmin_vv_i16m1(va, d9, vl);
                vint16m1_t vb1 = __riscv_vmax_vv_i16m1(vb, d9, vl);
                va0 = __riscv_vmin_vv_i16m1(va, d5, vl);
                vb0 = __riscv_vmax_vv_i16m1(vb, d5, vl);
                va1 = __riscv_vmin_vv_i16m1(va, d10, vl);
                vb1 = __riscv_vmax_vv_i16m1(vb, d10, vl);
                va0 = __riscv_vmin_vv_i16m1(va, d4, vl);
                vb0 = __riscv_vmax_vv_i16m1(vb, d4, vl);
                va1 = __riscv_vmin_vv_i16m1(va, d11, vl);
                vb1 = __riscv_vmax_vv_i16m1(vb, d11, vl);
                va0 = __riscv_vmin_vv_i16m1(va, d3, vl);
                vb0 = __riscv_vmax_vv_i16m1(vb, d3, vl);
                va1 = __riscv_vmin_vv_i16m1(va, d12, vl);
                vb1 = __riscv_vmax_vv_i16m1(vb, d12, vl);

                vint16m1_t va00 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d2, vl), d1, vl);
                vint16m1_t vb00 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d2, vl), d1, vl);
                vint16m1_t va10 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va1, d6, vl), d5, vl);
                vint16m1_t vb10 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb1, d6, vl), d5, vl);

                vint16m1_t va01 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d9, vl), d10, vl);
                vint16m1_t vb01 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d9, vl), d10, vl);
                vint16m1_t va11 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d13, vl), d14, vl);
                vint16m1_t vb11 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d13, vl), d14, vl);

                vint16m1_t min_max = __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va00, d0, vl), __riscv_vmin_vv_i16m1(va00, d9, vl), vl);
                vint16m1_t max_min = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb00, d0, vl), __riscv_vmax_vv_i16m1(vb00, d9, vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va01, d2, vl), __riscv_vmin_vv_i16m1(va01, d11, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb01, d2, vl), __riscv_vmax_vv_i16m1(vb01, d11, vl), vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va10, d4, vl), __riscv_vmin_vv_i16m1(va10, d13, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb10, d4, vl), __riscv_vmax_vv_i16m1(vb10, d13, vl), vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va11, d6, vl), __riscv_vmin_vv_i16m1(va01, d15, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb11, d6, vl), __riscv_vmax_vv_i16m1(vb01, d15, vl), vl), vl);

                va = __riscv_vmin_vv_i16m1(d15, d0, vl);
                vb = __riscv_vmax_vv_i16m1(d15, d0, vl);
                va0 = __riscv_vmin_vv_i16m1(va, d14, vl);
                vb0 = __riscv_vmax_vv_i16m1(vb, d14, vl);
                va1 = __riscv_vmin_vv_i16m1(va, d1, vl);
                vb1 = __riscv_vmax_vv_i16m1(vb, d1, vl);
                va0 = __riscv_vmin_vv_i16m1(va, d13, vl);
                vb0 = __riscv_vmax_vv_i16m1(vb, d13, vl);
                va1 = __riscv_vmin_vv_i16m1(va, d2, vl);
                vb1 = __riscv_vmax_vv_i16m1(vb, d2, vl);
                va0 = __riscv_vmin_vv_i16m1(va, d12, vl);
                vb0 = __riscv_vmax_vv_i16m1(vb, d12, vl);
                va1 = __riscv_vmin_vv_i16m1(va, d3, vl);
                vb1 = __riscv_vmax_vv_i16m1(vb, d3, vl);
                va0 = __riscv_vmin_vv_i16m1(va, d11, vl);
                vb0 = __riscv_vmax_vv_i16m1(vb, d11, vl);
                va1 = __riscv_vmin_vv_i16m1(va, d4, vl);
                vb1 = __riscv_vmax_vv_i16m1(vb, d4, vl);

                va00 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d10, vl), d9, vl);
                vb00 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d10, vl), d9, vl);
                va10 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va1, d14, vl), d13, vl);
                vb10 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb1, d14, vl), d13, vl);

                va01 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d1, vl), d10, vl);
                vb01 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d1, vl), d10, vl);
                va11 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d5, vl), d6, vl);
                vb11 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d5, vl), d6, vl);

                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va00, d8, vl), __riscv_vmin_vv_i16m1(va00, d1, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb00, d8, vl), __riscv_vmax_vv_i16m1(vb00, d1, vl), vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va01, d10, vl), __riscv_vmin_vv_i16m1(va01, d3, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb01, d10, vl), __riscv_vmax_vv_i16m1(vb01, d3, vl), vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va10, d12, vl), __riscv_vmin_vv_i16m1(va10, d5, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb10, d12, vl), __riscv_vmax_vv_i16m1(vb10, d5, vl), vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va11, d14, vl), __riscv_vmin_vv_i16m1(va01, d7, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb11, d14, vl), __riscv_vmax_vv_i16m1(vb01, d7, vl), vl), vl);


                vint16m1_t score_v = __riscv_vmax_vv_i16m1(min_max,
                        __riscv_vneg_v_i16m1(max_min, vl), vl);
                /* 写回 score */
                if (nonmax_suppression) __riscv_vse16_v_i16m1(curr + j, score_v, vl);

                /* score > threshold ? */
                vbool16_t mask = __riscv_vmsgt_vx_i16m1_b16(score_v, threshold, vl);
                /* 获取角点下标 */
                vuint16m1_t vresult = __riscv_vcompress_vm_u16m1(__riscv_vid_v_u16m1(vl), mask, vl);
                /* 获取角点数量 */
                size_t count = __riscv_vcpop_m_b16(mask, vl);
                /* 写回角点下标 */
                __riscv_vse32_v_i32m2(&cornerpos[ncorners], __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vwaddu_vx_u32m2(vresult, j, vl)), count);
                ncorners += count;
            }
        }

        cornerpos[-1] = ncorners;

        if (i == 3) continue;

        /* 非极大值抑制 */
        const int16_t *prev  = buf[(i - 4 + 3) % 3];
        const int16_t *pprev = buf[(i - 5 + 3) % 3];

        cornerpos = cpbuf[(i - 4 + 3) % 3] + 1;
        ncorners  = cornerpos[-1];

        for (int k = 0; k < ncorners; ++k) {
            int j = cornerpos[k];
            int score = prev[j];
            if (!nonmax_suppression ||
                (score > prev[j+1] && score > prev[j-1] &&
                 score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                 score > curr[j-1] && score > curr[j] && score > curr[j+1]))
            {
                keypoints.emplace_back((float)j, (float)(i - 1), 7.f, -1, (float)score);
            }
        }
    }
}
#endif
// 16.2
template<int patternSize>
void FAST_t(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    Mat img = _img.getMat();
    const int K = patternSize/2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize);

#if CV_SIMD128
    const int quarterPatternSize = patternSize/4;
    v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char)threshold), K16 = v_setall_u8((char)K);
#if CV_TRY_AVX2
    Ptr<opt_AVX2::FAST_t_patternSize16_AVX2> fast_t_impl_avx2;
    if(CV_CPU_HAS_SUPPORT_AVX2)
        fast_t_impl_avx2 = opt_AVX2::FAST_t_patternSize16_AVX2::getImpl(img.cols, threshold, nonmax_suppression, pixel);
#endif

#endif

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    uchar* buf[3] = { 0 };
    int* cpbuf[3] = { 0 };
    utils::BufferArea area;
    for (unsigned idx = 0; idx < 3; ++idx)
    {
        area.allocate(buf[idx], img.cols);
        area.allocate(cpbuf[idx], img.cols + 1);
    }
    area.commit();

    for (unsigned idx = 0; idx < 3; ++idx)
    {
        memset(buf[idx], 0, img.cols);
    }

    for(i = 3; i < img.rows-2; i++)
    {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3] + 1; // cornerpos[-1] is used to store a value
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
        {

            j = 3;
#if CV_SIMD128
            {
                if( patternSize == 16 )
                {
#if CV_TRY_AVX2
                    if (fast_t_impl_avx2)
                        fast_t_impl_avx2->process(j, ptr, curr, cornerpos, ncorners);
#endif
                    //vz if (j <= (img.cols - 27)) //it doesn't make sense using vectors for less than 8 elements
                    {
                        for (; j < img.cols - 16 - 3; j += 16, ptr += 16)
                        {
                            v_uint8x16 v = v_load(ptr);
                            v_int8x16 v0 = v_reinterpret_as_s8(v_xor(v_add(v, t), delta));
                            v_int8x16 v1 = v_reinterpret_as_s8(v_xor(v_sub(v, t), delta));

                            v_int8x16 x0 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[0]), delta));
                            v_int8x16 x1 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[quarterPatternSize]), delta));
                            v_int8x16 x2 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[2*quarterPatternSize]), delta));
                            v_int8x16 x3 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[3*quarterPatternSize]), delta));

                            v_int8x16 m0, m1;
                            m0 = v_and(v_lt(v0, x0), v_lt(v0, x1));
                            m1 = v_and(v_lt(x0, v1), v_lt(x1, v1));
                            m0 = v_or(m0, v_and(v_lt(v0, x1), v_lt(v0, x2)));
                            m1 = v_or(m1, v_and(v_lt(x1, v1), v_lt(x2, v1)));
                            m0 = v_or(m0, v_and(v_lt(v0, x2), v_lt(v0, x3)));
                            m1 = v_or(m1, v_and(v_lt(x2, v1), v_lt(x3, v1)));
                            m0 = v_or(m0, v_and(v_lt(v0, x3), v_lt(v0, x0)));
                            m1 = v_or(m1, v_and(v_lt(x3, v1), v_lt(x0, v1)));
                            m0 = v_or(m0, m1);

                            if( !v_check_any(m0) )
                                continue;
                            if( !v_check_any(v_combine_low(m0, m0)) )
                            {
                                j -= 8;
                                ptr -= 8;
                                continue;
                            }

                            v_int8x16 c0 = v_setzero_s8();
                            v_int8x16 c1 = v_setzero_s8();
                            v_uint8x16 max0 = v_setzero_u8();
                            v_uint8x16 max1 = v_setzero_u8();
                            for( k = 0; k < N; k++ )
                            {
                                v_int8x16 x = v_reinterpret_as_s8(v_xor(v_load((ptr + pixel[k])), delta));
                                m0 = v_lt(v0, x);
                                m1 = v_lt(x, v1);

                                c0 = v_and(v_sub_wrap(c0, m0), m0);
                                c1 = v_and(v_sub_wrap(c1, m1), m1);

                                max0 = v_max(max0, v_reinterpret_as_u8(c0));
                                max1 = v_max(max1, v_reinterpret_as_u8(c1));
                            }

                            max0 = v_lt(K16, v_max(max0, max1));
                            unsigned int m = v_signmask(v_reinterpret_as_s8(max0));

                            for( k = 0; m > 0 && k < 16; k++, m >>= 1 )
                            {
                                if( m & 1 )
                                {
                                    cornerpos[ncorners++] = j+k;
                                    if(nonmax_suppression)
                                    {
                                        short d[25];
                                        for (int _k = 0; _k < 25; _k++)
                                            d[_k] = (short)(ptr[k] - ptr[k + pixel[_k]]);

                                        v_int16x8 a0, b0, a1, b1;
                                        a0 = b0 = a1 = b1 = v_load(d + 8);
                                        for(int shift = 0; shift < 8; ++shift)
                                        {
                                            v_int16x8 v_nms = v_load(d + shift);
                                            a0 = v_min(a0, v_nms);
                                            b0 = v_max(b0, v_nms);
                                            v_nms = v_load(d + 9 + shift);
                                            a1 = v_min(a1, v_nms);
                                            b1 = v_max(b1, v_nms);
                                        }
                                        curr[j + k] = (uchar)(v_reduce_max(v_max(v_max(a0, a1), v_sub(v_setzero_s16(), v_min(b0, b1)))) - 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
#endif
            for( ; j < img.cols - 3; j++, ptr++ )
            {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }
            }
        }


        cornerpos[-1] = ncorners;
        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3] + 1; // cornerpos[-1] is used to store a value
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
               (score > prev[j+1] && score > prev[j-1] &&
                score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.push_back(KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
}

#ifdef HAVE_OPENCL
template<typename pt>
struct cmp_pt
{
    bool operator ()(const pt& a, const pt& b) const { return a.y < b.y || (a.y == b.y && a.x < b.x); }
};

static bool ocl_FAST( InputArray _img, std::vector<KeyPoint>& keypoints,
                     int threshold, bool nonmax_suppression, int maxKeypoints )
{
    UMat img = _img.getUMat();
    if( img.cols < 7 || img.rows < 7 )
        return false;
    size_t globalsize[] = { (size_t)img.cols-6, (size_t)img.rows-6 };

    ocl::Kernel fastKptKernel("FAST_findKeypoints", ocl::features2d::fast_oclsrc);
    if (fastKptKernel.empty())
        return false;

    UMat kp1(1, maxKeypoints*2+1, CV_32S);

    UMat ucounter1(kp1, Rect(0,0,1,1));
    ucounter1.setTo(Scalar::all(0));

    if( !fastKptKernel.args(ocl::KernelArg::ReadOnly(img),
                            ocl::KernelArg::PtrReadWrite(kp1),
                            maxKeypoints, threshold).run(2, globalsize, 0, true))
        return false;

    Mat mcounter;
    ucounter1.copyTo(mcounter);
    int i, counter = mcounter.at<int>(0);
    counter = std::min(counter, maxKeypoints);

    keypoints.clear();

    if( counter == 0 )
        return true;

    if( !nonmax_suppression )
    {
        Mat m;
        kp1(Rect(0, 0, counter*2+1, 1)).copyTo(m);
        const Point* pt = (const Point*)(m.ptr<int>() + 1);
        for( i = 0; i < counter; i++ )
            keypoints.push_back(KeyPoint((float)pt[i].x, (float)pt[i].y, 7.f, -1, 1.f));
    }
    else
    {
        UMat kp2(1, maxKeypoints*3+1, CV_32S);
        UMat ucounter2 = kp2(Rect(0,0,1,1));
        ucounter2.setTo(Scalar::all(0));

        ocl::Kernel fastNMSKernel("FAST_nonmaxSupression", ocl::features2d::fast_oclsrc);
        if (fastNMSKernel.empty())
            return false;

        size_t globalsize_nms[] = { (size_t)counter };
        if( !fastNMSKernel.args(ocl::KernelArg::PtrReadOnly(kp1),
                                ocl::KernelArg::PtrReadWrite(kp2),
                                ocl::KernelArg::ReadOnly(img),
                                counter, counter).run(1, globalsize_nms, 0, true))
            return false;

        Mat m2;
        kp2(Rect(0, 0, counter*3+1, 1)).copyTo(m2);
        Point3i* pt2 = (Point3i*)(m2.ptr<int>() + 1);
        int newcounter = std::min(m2.at<int>(0), counter);

        std::sort(pt2, pt2 + newcounter, cmp_pt<Point3i>());

        for( i = 0; i < newcounter; i++ )
            keypoints.push_back(KeyPoint((float)pt2[i].x, (float)pt2[i].y, 7.f, -1, (float)pt2[i].z));
    }

    return true;
}
#endif

static inline int hal_FAST(cv::Mat& src, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, FastFeatureDetector::DetectorType type)
{
    if (threshold > 20)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    cv::Mat scores(src.size(), src.type());

    int error = cv_hal_FAST_dense(src.data, src.step, scores.data, scores.step, src.cols, src.rows, type);

    if (error != CV_HAL_ERROR_OK)
        return error;

    cv::Mat suppressedScores(src.size(), src.type());

    if (nonmax_suppression)
    {
        error = cv_hal_FAST_NMS(scores.data, scores.step, suppressedScores.data, suppressedScores.step, scores.cols, scores.rows);

        if (error != CV_HAL_ERROR_OK)
            return error;
    }
    else
    {
        suppressedScores = scores;
    }

    if (!threshold && nonmax_suppression) threshold = 1;

    cv::KeyPoint kpt(0, 0, 7.f, -1, 0);

    unsigned uthreshold = (unsigned) threshold;

    int ofs = 3;

    int stride = (int)suppressedScores.step;
    const unsigned char* pscore = suppressedScores.data;

    keypoints.clear();

    for (int y = ofs; y + ofs < suppressedScores.rows; ++y)
    {
        kpt.pt.y = (float)(y);
        for (int x = ofs; x + ofs < suppressedScores.cols; ++x)
        {
            unsigned score = pscore[y * stride + x];
            if (score > uthreshold)
            {
                kpt.pt.x = (float)(x);
                kpt.response = (nonmax_suppression != 0) ? (float)((int)score - 1) : 0.f;
                keypoints.push_back(kpt);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

void FAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, FastFeatureDetector::DetectorType type)
{

    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_img.isUMat() && type == FastFeatureDetector::TYPE_9_16,
               ocl_FAST(_img, keypoints, threshold, nonmax_suppression, 10000));

    cv::Mat img = _img.getMat();
    CALL_HAL(fast_dense, hal_FAST, img, keypoints, threshold, nonmax_suppression, type);
    size_t keypoints_count = 10000;
    keypoints.clear();
    keypoints.resize(keypoints_count);
    // std::vector<cv::KeyPoint> keypoints_cp = keypoints;
    CALL_HAL(fast, cv_hal_FAST, img.data, img.step, img.cols, img.rows,
             (uchar*)(keypoints.data()), &keypoints_count, threshold, nonmax_suppression, type);
    switch(type) {
    case FastFeatureDetector::TYPE_5_8:
        FAST_t<8>(_img, keypoints, threshold, nonmax_suppression);
        break;
    case FastFeatureDetector::TYPE_7_12:
        FAST_t<12>(_img, keypoints, threshold, nonmax_suppression);
        break;
    case FastFeatureDetector::TYPE_9_16:
#ifdef __CX1C
        FAST16_cx1c(_img, keypoints, threshold, nonmax_suppression);
#elif defined(__RVV)
        FAST16_RVV(_img, keypoints, threshold, nonmax_suppression);
#else
        FAST_t<16>(_img, keypoints, threshold, nonmax_suppression);
        // if(keypoints_cp.size()!=keypoints.size()){
        //     printf("rvv:%d scalar:%d\n",keypoints_cp.size(),keypoints.size());
        //     exit(0);
        // }
#endif
        break;
    }
}


void FAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    CV_INSTRUMENT_REGION();

    FAST(_img, keypoints, threshold, nonmax_suppression, FastFeatureDetector::TYPE_9_16);
}


class FastFeatureDetector_Impl CV_FINAL : public FastFeatureDetector
{
public:
    FastFeatureDetector_Impl( int _threshold, bool _nonmaxSuppression, FastFeatureDetector::DetectorType _type )
    : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression), type(_type)
    {}

    void read( const FileNode& fn) CV_OVERRIDE
    {
      // if node is empty, keep previous value
      if (!fn["threshold"].empty())
        fn["threshold"] >> threshold;
      if (!fn["nonmaxSuppression"].empty())
        fn["nonmaxSuppression"] >> nonmaxSuppression;
      if (!fn["type"].empty())
        fn["type"] >> type;
    }
    void write( FileStorage& fs) const CV_OVERRIDE
    {
      if(fs.isOpened())
      {
        fs << "name" << getDefaultName();
        fs << "threshold" << threshold;
        fs << "nonmaxSuppression" << nonmaxSuppression;
        fs << "type" << type;
      }
    }

    void detect( InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask ) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        if(_image.empty())
        {
            keypoints.clear();
            return;
        }

        Mat mask = _mask.getMat(), grayImage;
        UMat ugrayImage;
        _InputArray gray = _image;
        if( _image.type() != CV_8U )
        {
            _OutputArray ogray = _image.isUMat() ? _OutputArray(ugrayImage) : _OutputArray(grayImage);
            cvtColor( _image, ogray, COLOR_BGR2GRAY );
            gray = ogray;
        }
        FAST( gray, keypoints, threshold, nonmaxSuppression, type );
        KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    void set(int prop, double value)
    {
        if(prop == THRESHOLD)
            threshold = cvRound(value);
        else if(prop == NONMAX_SUPPRESSION)
            nonmaxSuppression = value != 0;
        else if(prop == FAST_N)
            type = static_cast<FastFeatureDetector::DetectorType>(cvRound(value));
        else
            CV_Error(Error::StsBadArg, "");
    }

    double get(int prop) const
    {
        if(prop == THRESHOLD)
            return threshold;
        if(prop == NONMAX_SUPPRESSION)
            return nonmaxSuppression;
        if(prop == FAST_N)
            return static_cast<int>(type);
        CV_Error(Error::StsBadArg, "");
        return 0;
    }

    void setThreshold(int threshold_) CV_OVERRIDE { threshold = threshold_; }
    int getThreshold() const CV_OVERRIDE { return threshold; }

    void setNonmaxSuppression(bool f) CV_OVERRIDE { nonmaxSuppression = f; }
    bool getNonmaxSuppression() const CV_OVERRIDE { return nonmaxSuppression; }

    void setType(FastFeatureDetector::DetectorType type_) CV_OVERRIDE{ type = type_; }
    FastFeatureDetector::DetectorType getType() const CV_OVERRIDE{ return type; }

    int threshold;
    bool nonmaxSuppression;
    FastFeatureDetector::DetectorType type;
};

Ptr<FastFeatureDetector> FastFeatureDetector::create( int threshold, bool nonmaxSuppression, FastFeatureDetector::DetectorType type )
{
    return makePtr<FastFeatureDetector_Impl>(threshold, nonmaxSuppression, type);
}

String FastFeatureDetector::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".FastFeatureDetector");
}

}
