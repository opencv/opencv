#include "rvv_hal.hpp"
#include "common.hpp"
#include <cstring>
#include <vector>
#include <algorithm>

namespace cv { namespace rvv_hal { namespace features2d {

inline int fast_16(const uchar* src_data, size_t src_step,
                   int width, int height,
                   std::vector<cvhalKeyPoint> &keypoints,
                   int threshold, bool nonmax_suppression)
{
    constexpr int patternSize = 16;
    constexpr int K = patternSize / 2, N = patternSize + K + 1;

    int pixel[N] = {0};
    pixel[0]  =  0 + (int)src_step * 3;
    pixel[1]  =  1 + (int)src_step * 3;
    pixel[2]  =  2 + (int)src_step * 2;
    pixel[3]  =  3 + (int)src_step * 1;
    pixel[4]  =  3 + (int)src_step * 0;
    pixel[5]  =  3 + (int)src_step * -1;
    pixel[6]  =  2 + (int)src_step * -2;
    pixel[7]  =  1 + (int)src_step * -3;
    pixel[8]  =  0 + (int)src_step * -3;
    pixel[9]  = -1 + (int)src_step * -3;
    pixel[10] = -2 + (int)src_step * -2;
    pixel[11] = -3 + (int)src_step * -1;
    pixel[12] = -3 + (int)src_step * 0;
    pixel[13] = -3 + (int)src_step * 1;
    pixel[14] = -2 + (int)src_step * 2;
    pixel[15] = -1 + (int)src_step * 3;
    for (int k = 16; k < N; k++)
        pixel[k] = pixel[k - 16];

    /* Three-row score buffer (int16) + corner position buffer */
    std::vector<int16_t> _sbuf((width + 16) * 3, 0);
    int16_t *sbuf[3] = { _sbuf.data(), _sbuf.data() + width, _sbuf.data() + 2 * width };

    std::vector<int> _cpbuf((width + 2) * 3, 0);
    int *cpbuf[3] = { _cpbuf.data() + 1, _cpbuf.data() + (width + 2) + 1,
                      _cpbuf.data() + 2 * (width + 2) + 1 };

    for (int i = 3; i < height - 2; i++)
    {
        const uchar *ptr = src_data + i * src_step + 3;
        int16_t *curr = sbuf[(i - 3) % 3];
        int *cornerpos = cpbuf[(i - 3) % 3];
        int ncorners = 0;
        std::fill(curr, curr + width, (int16_t)0);

        if (i < height - 3)
        {
            int j = 3;
            size_t vl;
            for (; j < width - 3; j += vl, ptr += vl)
            {
                vl = __riscv_vsetvl_e16m1(width - 3 - j);

                /* Load center pixel and widen to int16 */
                vint16m1_t vcen = __riscv_vreinterpret_v_u16m1_i16m1(
                    __riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr, vl), vl));
                vint16m1_t vlo = __riscv_vsub_vx_i16m1(vcen, threshold, vl);
                vint16m1_t vhi = __riscv_vadd_vx_i16m1(vcen, threshold, vl);

                /* 4-direction quick reject */
                vint16m1_t vk0  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[0],  vl), vl));
                vint16m1_t vk4  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[4],  vl), vl));
                vint16m1_t vk8  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[8],  vl), vl));
                vint16m1_t vk12 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[12], vl), vl));

                vbool16_t bright = __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk0, vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk4, vhi, vl), vl);
                vbool16_t dark   = __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk0, vl),  __riscv_vmsgt_vv_i16m1_b16(vlo, vk4, vl),  vl);
                bright = __riscv_vmor_mm_b16(bright, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk4,  vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk8,  vhi, vl), vl), vl);
                dark   = __riscv_vmor_mm_b16(dark,   __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk4, vl),  __riscv_vmsgt_vv_i16m1_b16(vlo, vk8, vl),  vl), vl);
                bright = __riscv_vmor_mm_b16(bright, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk8,  vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk12, vhi, vl), vl), vl);
                dark   = __riscv_vmor_mm_b16(dark,   __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk8, vl),  __riscv_vmsgt_vv_i16m1_b16(vlo, vk12, vl), vl), vl);
                bright = __riscv_vmor_mm_b16(bright, __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vk12, vhi, vl), __riscv_vmsgt_vv_i16m1_b16(vk0,  vhi, vl), vl), vl);
                dark   = __riscv_vmor_mm_b16(dark,   __riscv_vmand_mm_b16(__riscv_vmsgt_vv_i16m1_b16(vlo, vk12, vl), __riscv_vmsgt_vv_i16m1_b16(vlo, vk0,  vl), vl), vl);

                if (__riscv_vfirst_m_b16(__riscv_vmor_mm_b16(bright, dark, vl), vl) < 0)
                    continue;

                /* Load remaining 12 neighbors */
                vint16m1_t vk1  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[1],  vl), vl));
                vint16m1_t vk2  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[2],  vl), vl));
                vint16m1_t vk3  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[3],  vl), vl));
                vint16m1_t vk5  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[5],  vl), vl));
                vint16m1_t vk6  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[6],  vl), vl));
                vint16m1_t vk7  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[7],  vl), vl));
                vint16m1_t vk9  = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[9],  vl), vl));
                vint16m1_t vk10 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[10], vl), vl));
                vint16m1_t vk11 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[11], vl), vl));
                vint16m1_t vk13 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[13], vl), vl));
                vint16m1_t vk14 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[14], vl), vl));
                vint16m1_t vk15 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2(__riscv_vle8_v_u8mf2(ptr + pixel[15], vl), vl));

                /* Compute differences: d[i] = center - neighbor[i] */
                vint16m1_t d0  = __riscv_vsub_vv_i16m1(vcen, vk0,  vl);
                vint16m1_t d1  = __riscv_vsub_vv_i16m1(vcen, vk1,  vl);
                vint16m1_t d2  = __riscv_vsub_vv_i16m1(vcen, vk2,  vl);
                vint16m1_t d3  = __riscv_vsub_vv_i16m1(vcen, vk3,  vl);
                vint16m1_t d4  = __riscv_vsub_vv_i16m1(vcen, vk4,  vl);
                vint16m1_t d5  = __riscv_vsub_vv_i16m1(vcen, vk5,  vl);
                vint16m1_t d6  = __riscv_vsub_vv_i16m1(vcen, vk6,  vl);
                vint16m1_t d7  = __riscv_vsub_vv_i16m1(vcen, vk7,  vl);
                vint16m1_t d8  = __riscv_vsub_vv_i16m1(vcen, vk8,  vl);
                vint16m1_t d9  = __riscv_vsub_vv_i16m1(vcen, vk9,  vl);
                vint16m1_t d10 = __riscv_vsub_vv_i16m1(vcen, vk10, vl);
                vint16m1_t d11 = __riscv_vsub_vv_i16m1(vcen, vk11, vl);
                vint16m1_t d12 = __riscv_vsub_vv_i16m1(vcen, vk12, vl);
                vint16m1_t d13 = __riscv_vsub_vv_i16m1(vcen, vk13, vl);
                vint16m1_t d14 = __riscv_vsub_vv_i16m1(vcen, vk14, vl);
                vint16m1_t d15 = __riscv_vsub_vv_i16m1(vcen, vk15, vl);

                /* Score computation: 16-arc min/max */
                vint16m1_t va  = __riscv_vmin_vv_i16m1(d7, d8, vl);
                vint16m1_t vb  = __riscv_vmax_vv_i16m1(d7, d8, vl);
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

                vint16m1_t va01 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d9, vl),  d10, vl);
                vint16m1_t vb01 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d9, vl),  d10, vl);
                vint16m1_t va11 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d13, vl), d14, vl);
                vint16m1_t vb11 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d13, vl), d14, vl);

                vint16m1_t min_max = __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va00, d0, vl), __riscv_vmin_vv_i16m1(va00, d9, vl), vl);
                vint16m1_t max_min = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb00, d0, vl), __riscv_vmax_vv_i16m1(vb00, d9, vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va01, d2, vl),  __riscv_vmin_vv_i16m1(va01, d11, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb01, d2, vl),  __riscv_vmax_vv_i16m1(vb01, d11, vl), vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va10, d4, vl),  __riscv_vmin_vv_i16m1(va10, d13, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb10, d4, vl),  __riscv_vmax_vv_i16m1(vb10, d13, vl), vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va11, d6, vl),  __riscv_vmin_vv_i16m1(va01, d15, vl), vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb11, d6, vl),  __riscv_vmax_vv_i16m1(vb01, d15, vl), vl), vl);

                va  = __riscv_vmin_vv_i16m1(d15, d0, vl);
                vb  = __riscv_vmax_vv_i16m1(d15, d0, vl);
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

                va01 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d1, vl),  d10, vl);
                vb01 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d1, vl),  d10, vl);
                va11 = __riscv_vmin_vv_i16m1(__riscv_vmin_vv_i16m1(va0, d5, vl),  d6, vl);
                vb11 = __riscv_vmax_vv_i16m1(__riscv_vmax_vv_i16m1(vb0, d5, vl),  d6, vl);

                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va00, d8, vl),  __riscv_vmin_vv_i16m1(va00, d1, vl),  vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb00, d8, vl),  __riscv_vmax_vv_i16m1(vb00, d1, vl),  vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va01, d10, vl), __riscv_vmin_vv_i16m1(va01, d3, vl),  vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb01, d10, vl), __riscv_vmax_vv_i16m1(vb01, d3, vl),  vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va10, d12, vl), __riscv_vmin_vv_i16m1(va10, d5, vl),  vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb10, d12, vl), __riscv_vmax_vv_i16m1(vb10, d5, vl),  vl), vl);
                min_max = __riscv_vmax_vv_i16m1(min_max, __riscv_vmax_vv_i16m1(__riscv_vmin_vv_i16m1(va11, d14, vl), __riscv_vmin_vv_i16m1(va01, d7, vl),  vl), vl);
                max_min = __riscv_vmin_vv_i16m1(max_min, __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(vb11, d14, vl), __riscv_vmax_vv_i16m1(vb01, d7, vl),  vl), vl);

                vint16m1_t score_v = __riscv_vmax_vv_i16m1(min_max,
                        __riscv_vneg_v_i16m1(max_min, vl), vl);

                if (nonmax_suppression)
                    __riscv_vse16_v_i16m1(curr + j, score_v, vl);

                vbool16_t mask = __riscv_vmsgt_vx_i16m1_b16(score_v, threshold, vl);
                vuint16m1_t vresult = __riscv_vcompress_vm_u16m1(__riscv_vid_v_u16m1(vl), mask, vl);
                size_t count = __riscv_vcpop_m_b16(mask, vl);
                __riscv_vse32_v_i32m2(&cornerpos[ncorners],
                    __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vwaddu_vx_u32m2(vresult, (uint16_t)j, vl)), count);
                ncorners += count;
            }
        }

        cornerpos[-1] = ncorners;

        if (i == 3) continue;

        const int16_t *prev  = sbuf[(i - 4 + 3) % 3];
        const int16_t *pprev = sbuf[(i - 5 + 3) % 3];
        cornerpos = cpbuf[(i - 4 + 3) % 3];
        ncorners  = cornerpos[-1];

        for (int k = 0; k < ncorners; k++)
        {
            int j = cornerpos[k];
            int score = prev[j];
            if (!nonmax_suppression ||
                (score > prev[j+1] && score > prev[j-1] &&
                 score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                 score > curr[j-1] && score > curr[j] && score > curr[j+1]))
            {
                cvhalKeyPoint kp;
                kp.x = (float)j;
                kp.y = (float)(i - 1);
                kp.size = 7.f;
                kp.angle = -1.f;
                kp.response = (float)score;
                kp.octave = 0;
                kp.class_id = -1;
                keypoints.push_back(kp);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

int FAST(const uchar* src_data, size_t src_step,
         int width, int height, void** keypoints_data,
         size_t* keypoints_count, int threshold,
         bool nonmax_suppression, int detector_type, void* (*realloc_func)(void*, size_t))
{
    int res = CV_HAL_ERROR_UNKNOWN;
    switch(detector_type) {
        case CV_HAL_TYPE_5_8:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_7_12:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_9_16: {
            std::vector<cvhalKeyPoint> keypoints;
            res = fast_16(src_data, src_step, width, height, keypoints, threshold, nonmax_suppression);
            if (res == CV_HAL_ERROR_OK) {
                if (keypoints.size() > *keypoints_count) {
                    *keypoints_count = keypoints.size();
                    uchar *tmp = (uchar*)realloc_func(*keypoints_data, sizeof(cvhalKeyPoint)*(*keypoints_count));
                    memcpy(tmp, (uchar*)keypoints.data(), sizeof(cvhalKeyPoint)*(*keypoints_count));
                    *keypoints_data = tmp;
                } else {
                    *keypoints_count = keypoints.size();
                    memcpy(*keypoints_data, (uchar*)keypoints.data(), sizeof(cvhalKeyPoint)*(*keypoints_count));
                }
            }
            return res;
        }
        default:
            return res;
    }
}

}}} // namespace cv::rvv_hal::features2d
