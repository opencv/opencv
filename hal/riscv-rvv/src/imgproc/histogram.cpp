// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"

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

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
