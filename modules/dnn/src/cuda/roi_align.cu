// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "grid_stride_range.hpp"
#include "math.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/roi_align.hpp"

#include <opencv2/core.hpp>

#include <cfloat>
#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        __device__ inline std::int64_t read_batch_index(const void* data, std::size_t element_size, int offset) {
            if (element_size == sizeof(std::int32_t))
                return *(reinterpret_cast<const std::int32_t*>(data) + offset);
            return *(reinterpret_cast<const std::int64_t*>(data) + offset);
        }

        /* One thread owns one output element. The ROI geometry is recomputed per thread
         * rather than cached per ROI (as the CPU path does); it is a handful of flops
         * against a memory-bound gather, and it keeps the kernel free of shared state. */
        template <class T>
        __global__ void roi_align(Span<T> output, View<T> input, View<float> rois,
                                  const void* batch_indices, std::size_t batch_index_element_size,
                                  RoiAlignParams p)
        {
            for (auto id : grid_stride_range(output.size())) {
                int rem = id;
                const int pw = rem % p.output_width;  rem /= p.output_width;
                const int ph = rem % p.output_height; rem /= p.output_height;
                const int c  = rem % p.channels;      rem /= p.channels;
                const int r  = rem;

                const float* roi = rois.data().get() + static_cast<std::int64_t>(r) * 4;
                const float x1 = roi[0] * p.spatial_scale - p.offset;
                const float y1 = roi[1] * p.spatial_scale - p.offset;
                const float x2 = roi[2] * p.spatial_scale - p.offset;
                const float y2 = roi[3] * p.spatial_scale - p.offset;

                float roi_width  = x2 - x1;
                float roi_height = y2 - y1;
                if (p.clamp_malformed_roi) {
                    roi_width  = fmaxf(roi_width, 1.f);
                    roi_height = fmaxf(roi_height, 1.f);
                }

                const float bin_size_w = roi_width / p.output_width;
                const float bin_size_h = roi_height / p.output_height;

                const int gw = device::max<int>((p.sampling_ratio > 0) ? p.sampling_ratio
                                                                       : static_cast<int>(ceilf(roi_width / p.output_width)), 1);
                const int gh = device::max<int>((p.sampling_ratio > 0) ? p.sampling_ratio
                                                                       : static_cast<int>(ceilf(roi_height / p.output_height)), 1);
                const int sample_count = gw * gh;
                const float step_y = bin_size_h / gh;
                const float step_x = bin_size_w / gw;

                const std::int64_t b = read_batch_index(batch_indices, batch_index_element_size, r);
                const T* img = input.data().get() +
                               ((b * p.channels) + c) * static_cast<std::int64_t>(p.height) * p.width;

                float outv = p.max_mode ? -FLT_MAX : 0.f;
                for (int s = 0; s < sample_count; ++s) {
                    const int iy = s / gw;
                    const int ix = s - iy * gw;
                    const float yy = y1 + ph * bin_size_h + (iy + 0.5f) * step_y;
                    const float xx = x1 + pw * bin_size_w + (ix + 0.5f) * step_x;

                    /* precomputeBilinearSample() returns an all-zero sample when the point
                     * is fully outside [-1, extent], so it contributes 0 in both modes. */
                    float v = 0.f;
                    if (!(yy < -1.f || yy > p.height || xx < -1.f || xx > p.width)) {
                        const float cy = fminf(fmaxf(yy, 0.f), static_cast<float>(p.height - 1));
                        const float cx = fminf(fmaxf(xx, 0.f), static_cast<float>(p.width - 1));

                        const int y_low = static_cast<int>(cy);
                        const int x_low = static_cast<int>(cx);
                        const int y_high = device::min<int>(y_low + 1, p.height - 1);
                        const int x_high = device::min<int>(x_low + 1, p.width - 1);

                        const float ly = cy - y_low, lx = cx - x_low;
                        const float hy = 1.f - ly,   hx = 1.f - lx;

                        const std::int64_t row_low  = static_cast<std::int64_t>(y_low) * p.width;
                        const std::int64_t row_high = static_cast<std::int64_t>(y_high) * p.width;

                        const float v00 = static_cast<float>(img[row_low  + x_low]);
                        const float v01 = static_cast<float>(img[row_low  + x_high]);
                        const float v10 = static_cast<float>(img[row_high + x_low]);
                        const float v11 = static_cast<float>(img[row_high + x_high]);

                        v = p.max_mode
                                ? fmaxf(fmaxf(hy * hx * v00, hy * lx * v01),
                                        fmaxf(ly * hx * v10, ly * lx * v11))
                                : (hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11);
                    }

                    outv = p.max_mode ? fmaxf(outv, v) : (outv + v);
                }

                outv = p.max_mode ? ((outv == -FLT_MAX) ? 0.f : outv)
                                  : (outv / static_cast<float>(sample_count));
                output[id] = static_cast<T>(outv);
            }
        }
    }

    template <class T>
    void roi_align(const Stream& stream,
        Span<T> output, View<T> input, View<float> rois,
        const void* batch_indices, std::size_t batch_index_element_size,
        const RoiAlignParams& params)
    {
        CV_Assert(batch_index_element_size == sizeof(std::int32_t) ||
                  batch_index_element_size == sizeof(std::int64_t));
        CV_Assert(params.output_height > 0 && params.output_width > 0);
        CV_Assert(params.height > 0 && params.width > 0 && params.channels > 0);

        if (output.size() == 0)
            return;

        auto kernel = raw::roi_align<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, rois, batch_indices, batch_index_element_size, params);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void roi_align(const Stream&, Span<__half>, View<__half>, View<float>, const void*, std::size_t, const RoiAlignParams&);
#endif
    template void roi_align(const Stream&, Span<float>, View<float>, View<float>, const void*, std::size_t, const RoiAlignParams&);
    template void roi_align(const Stream&, Span<int8_t>, View<int8_t>, View<float>, const void*, std::size_t, const RoiAlignParams&);
    template void roi_align(const Stream&, Span<uint8_t>, View<uint8_t>, View<float>, const void*, std::size_t, const RoiAlignParams&);
    template void roi_align(const Stream&, Span<int32_t>, View<int32_t>, View<float>, const void*, std::size_t, const RoiAlignParams&);
    template void roi_align(const Stream&, Span<int64_t>, View<int64_t>, View<float>, const void*, std::size_t, const RoiAlignParams&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
