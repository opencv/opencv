// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "rvv_hal.hpp"
#include <cstring>
#include <vector>

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

class HistogramInvoker : public ParallelLoopBody
{
public:
    template<typename... Args>
    HistogramInvoker(std::function<void(int, int, Args...)> _func, Args&&... args)
    {
        func = std::bind(_func, std::placeholders::_1, std::placeholders::_2, std::forward<Args>(args)...);
    }

    virtual void operator()(const Range& range) const override
    {
        func(range.start, range.end);
    }

private:
    std::function<void(int, int)> func;
};

constexpr int HIST_SZ = std::numeric_limits<uchar>::max() + 1;

static inline void hist_invoke(int start, int end, const uchar* src_data, size_t src_step, int width, int* hist, std::mutex* m)
{
    int h[HIST_SZ] = {0};
    for (int i = start; i < end; i++)
    {
        const uchar* src = src_data + i * src_step;
        int j;
        for (j = 0; j + 3 < width; j += 4)
        {
            int t0 = src[j], t1 = src[j+1];
            h[t0]++; h[t1]++;
            t0 = src[j+2]; t1 = src[j+3];
            h[t0]++; h[t1]++;
        }
        for (; j < width; j++)
        {
            h[src[j]]++;
        }
    }

    std::lock_guard<std::mutex> lk(*m);
    for (int i = 0; i < HIST_SZ; i++)
    {
        hist[i] += h[i];
    }
}

static inline void lut_invoke(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, const uchar* lut)
{
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m8(width - j);
            auto src = __riscv_vle8_v_u8m8(src_data + i * src_step + j, vl);
            auto dst = __riscv_vloxei8_v_u8m8(lut, src, vl);
            __riscv_vse8(dst_data + i * dst_step + j, dst, vl);
        }
    }
}

} // equalize_hist

// the algorithm is copied from imgproc/src/histogram.cpp,
// in the function void cv::equalizeHist
int equalize_hist(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height)
{
    int hist[HIST_SZ] = {0};
    uchar lut[HIST_SZ];

    std::mutex m;
    cv::parallel_for_(Range(0, height), HistogramInvoker({hist_invoke}, src_data, src_step, width, reinterpret_cast<int *>(hist), &m), static_cast<double>(width * height) / (1 << 15));

    int i = 0;
    while (!hist[i]) ++i;

    float scale = (HIST_SZ - 1.f)/(width * height - hist[i]);
    int sum = 0;
    for (lut[i++] = 0; i < HIST_SZ; i++)
    {
        sum += hist[i];
        lut[i] = std::min(std::max(static_cast<int>(std::round(sum * scale)), 0), HIST_SZ - 1);
    }
    cv::parallel_for_(Range(0, height), HistogramInvoker({lut_invoke}, src_data, src_step, dst_data, dst_step, width, reinterpret_cast<const uchar*>(lut)), static_cast<double>(width * height) / (1 << 15));

    return CV_HAL_ERROR_OK;
}

// ############ calc_hist ############

namespace {

constexpr int MAX_VLEN = 1024;
constexpr int MAX_E8M1 = MAX_VLEN / 8;

inline void cvt_32s32f(const int* ihist, float* fhist, int hist_size) {
    int vl;
    for (int i = 0; i < hist_size; i += vl) {
        vl = __riscv_vsetvl_e32m8(hist_size - i);
        auto iv = __riscv_vle32_v_i32m8(ihist + i, vl);
        __riscv_vse32(fhist + i, __riscv_vfcvt_f(iv, vl), vl);
    }
}

inline void cvt32s32f_add32f(const int* ihist, float* fhist, int hist_size) {
    int vl;
    for (int i = 0; i < hist_size; i += vl) {
        vl = __riscv_vsetvl_e32m8(hist_size - i);
        auto iv = __riscv_vle32_v_i32m8(ihist + i, vl);
        auto fv = __riscv_vle32_v_f32m8(fhist + i, vl);
        auto s = __riscv_vfadd(__riscv_vfcvt_f(iv, vl), fv, vl);
        __riscv_vse32(fhist + i, s, vl);
    }
}

}

int calc_hist(const uchar* src_data, size_t src_step, int src_type, int src_width, int src_height,
              float* hist_data, int hist_size, const float** ranges, bool uniform, bool accumulate) {
    int depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);

    // [TODO] support non-uniform
    // In case of CV_8U, it is already fast enough with lut
    if (depth == CV_8U || !uniform) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    std::vector<int> buf_ihist(hist_size+1, 0);
    int* ihist = buf_ihist.data();

    double low = ranges[0][0], high = ranges[0][1];
    double t = hist_size / (high - low);
    double a = t, b = -t * low;
    double v0_lo = low, v0_hi = high;

    int sz = hist_size, d0 = cn, step0 = (int)(src_step / CV_ELEM_SIZE1(src_type));
    int buf_idx[MAX_E8M1];

    if (depth == CV_16U) {
        const ushort* p0 = (const ushort*)src_data;
        if (d0 == 1) {
            while (src_height--) {
                int vl;
                for (int x = 0; x < src_width; x += vl) {
                    vl = __riscv_vsetvl_e16m2(src_width - x);

                    auto v = __riscv_vfcvt_f(__riscv_vwcvtu_x(__riscv_vwcvtu_x(__riscv_vle16_v_u16m2(p0 + x, vl), vl), vl), vl);

                    auto m0 = __riscv_vmflt(v, v0_lo, vl);
                    auto m1 = __riscv_vmfge(v, v0_hi, vl);
                    auto m = __riscv_vmor(m0, m1, vl);

                    auto fidx = __riscv_vfadd(__riscv_vfmul(v, a, vl), b, vl);
                    auto idx = __riscv_vfncvt_x(__riscv_vfsub(fidx, 0.5f - 1e-6, vl), vl);
                    idx = __riscv_vmerge(idx, 0, __riscv_vmslt(idx, 0, vl), vl);
                    idx = __riscv_vmerge(idx, sz-1, __riscv_vmsgt(idx, sz-1, vl), vl);
                    idx = __riscv_vmerge(idx, -1, m, vl);
                    __riscv_vse32(buf_idx, idx, vl);

                    for (int i = 0; i < vl; i++) {
                        int _idx = buf_idx[i] + 1;
                        ihist[_idx]++;
                    }
                }
                p0 += step0;
            }
        } else {
            while (src_height--) {
                int vl;
                for (int x = 0; x < src_width; x += vl) {
                    vl = __riscv_vsetvl_e16m2(src_width - x);

                    auto v = __riscv_vfcvt_f(__riscv_vwcvtu_x(__riscv_vwcvtu_x(__riscv_vlse16_v_u16m2(p0 + x*d0, sizeof(ushort)*d0, vl), vl), vl), vl);

                    auto m0 = __riscv_vmflt(v, v0_lo, vl);
                    auto m1 = __riscv_vmfge(v, v0_hi, vl);
                    auto m = __riscv_vmor(m0, m1, vl);

                    auto fidx = __riscv_vfadd(__riscv_vfmul(v, a, vl), b, vl);
                    auto idx = __riscv_vfncvt_x(__riscv_vfsub(fidx, 0.5f - 1e-6, vl), vl);
                    idx = __riscv_vmerge(idx, 0, __riscv_vmslt(idx, 0, vl), vl);
                    idx = __riscv_vmerge(idx, sz-1, __riscv_vmsgt(idx, sz-1, vl), vl);
                    idx = __riscv_vmerge(idx, -1, m, vl);
                    __riscv_vse32(buf_idx, idx, vl);

                    for (int i = 0; i < vl; i++) {
                        int _idx = buf_idx[i] + 1;
                        ihist[_idx]++;
                    }
                }
                p0 += step0;
            }
        }
    } else if (depth == CV_32F) {
        const float* p0 = (const float*)src_data;
        if (d0 == 1) {
            while (src_height--) {
                int vl;
                for (int x = 0; x < src_width; x += vl) {
                    vl = __riscv_vsetvl_e32m4(src_width - x);

                    auto v = __riscv_vfwcvt_f(__riscv_vle32_v_f32m4(p0 + x, vl), vl);

                    auto m0 = __riscv_vmflt(v, v0_lo, vl);
                    auto m1 = __riscv_vmfge(v, v0_hi, vl);
                    auto m = __riscv_vmor(m0, m1, vl);

                    auto fidx = __riscv_vfadd(__riscv_vfmul(v, a, vl), b, vl);
                    auto idx = __riscv_vfncvt_x(__riscv_vfsub(fidx, 0.5f - 1e-6, vl), vl);
                    idx = __riscv_vmerge(idx, 0, __riscv_vmslt(idx, 0, vl), vl);
                    idx = __riscv_vmerge(idx, sz-1, __riscv_vmsgt(idx, sz-1, vl), vl);
                    idx = __riscv_vmerge(idx, -1, m, vl);
                    __riscv_vse32(buf_idx, idx, vl);

                    for (int i = 0; i < vl; i++) {
                        int _idx = buf_idx[i] + 1;
                        ihist[_idx]++;
                    }
                }
                p0 += step0;
            }
        } else {
            while (src_height--) {
                int vl;
                for (int x = 0; x < src_width; x += vl) {
                    vl = __riscv_vsetvl_e32m4(src_width - x);

                    auto v = __riscv_vfwcvt_f(__riscv_vlse32_v_f32m4(p0 + x*d0, sizeof(float)*d0, vl), vl);

                    auto m0 = __riscv_vmflt(v, v0_lo, vl);
                    auto m1 = __riscv_vmfge(v, v0_hi, vl);
                    auto m = __riscv_vmor(m0, m1, vl);

                    auto fidx = __riscv_vfadd(__riscv_vfmul(v, a, vl), b, vl);
                    auto idx = __riscv_vfncvt_x(__riscv_vfsub(fidx, 0.5f - 1e-6, vl), vl);
                    idx = __riscv_vmerge(idx, 0, __riscv_vmslt(idx, 0, vl), vl);
                    idx = __riscv_vmerge(idx, sz-1, __riscv_vmsgt(idx, sz-1, vl), vl);
                    idx = __riscv_vmerge(idx, -1, m, vl);
                    __riscv_vse32(buf_idx, idx, vl);

                    for (int i = 0; i < vl; i++) {
                        int _idx = buf_idx[i] + 1;
                        ihist[_idx]++;
                    }
                }
                p0 += step0;
            }
        }
    }

    if (accumulate) {
        cvt32s32f_add32f(ihist+1, hist_data, hist_size);
    } else {
        std::memset(hist_data, 0, sizeof(float)*hist_size);
        cvt_32s32f(ihist+1, hist_data, hist_size);
    }

    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
