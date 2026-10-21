// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/nms.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        __global__ void nms_iota(int* out, size_type n)
        {
            for (auto i : grid_stride_range(n))
                out[i] = static_cast<int>(i);
        }

        __global__ void nms_prepare(const float* sorted_scores, float score_threshold, const int* sorted_idx,
                                    const float* boxes, bool center_point_box, double* rects, int* above,
                                    size_type n)
        {
            for (auto i : grid_stride_range(n)) {
                if (sorted_scores[i] > score_threshold)
                    atomicMax(above, static_cast<int>(i) + 1);
                const float* bx = boxes + 4 * static_cast<std::size_t>(sorted_idx[i]);
                float x, y, w, h;
                if (!center_point_box) {
                    const float y1 = bx[0], x1 = bx[1], y2 = bx[2], x2 = bx[3];
                    x = fminf(x1, x2);
                    y = fminf(y1, y2);
                    w = fmaxf(0.f, fabsf(x2 - x1));
                    h = fmaxf(0.f, fabsf(y2 - y1));
                } else {
                    const float yc = bx[0], xc = bx[1];
                    h = fmaxf(0.f, bx[2]);
                    w = fmaxf(0.f, bx[3]);
                    x = xc - 0.5f * w;
                    y = yc - 0.5f * h;
                }
                rects[4 * i + 0] = x;
                rects[4 * i + 1] = y;
                rects[4 * i + 2] = w;
                rects[4 * i + 3] = h;
            }
        }

        __device__ inline float nms_overlap(const double* a, const double* b)
        {
            const double area_a = a[2] * a[3], area_b = b[2] * b[3];
            double distance = 0.0;
            if (area_a + area_b > DBL_EPSILON) {
                const double x1 = fmax(a[0], b[0]), y1 = fmax(a[1], b[1]);
                const double w = fmin(a[0] + a[2], b[0] + b[2]) - x1;
                const double h = fmin(a[1] + a[3], b[1] + b[3]) - y1;
                const double inter = (w > 0 && h > 0) ? w * h : 0.0;
                distance = 1.0 - inter / (area_a + area_b - inter);
            }
            return 1.f - static_cast<float>(distance);
        }

        __global__ void nms_mask(const double* rects, const int* above, int words, float iou_threshold,
                                 unsigned int* mask, size_type n)
        {
            const int count = *above;
            for (auto id : grid_stride_range(n)) {
                const int i = static_cast<int>(id / words), w = static_cast<int>(id % words);
                unsigned int bits = 0;
                if (i < count) {
                    const double* a = rects + 4 * static_cast<std::size_t>(i);
                    for (int k = 0; k < 32; k++) {
                        const int j = w * 32 + k;
                        if (j <= i || j >= count)
                            continue;
                        if (nms_overlap(a, rects + 4 * static_cast<std::size_t>(j)) > iou_threshold)
                            bits |= 1u << k;
                    }
                }
                mask[id] = bits;
            }
        }

        __global__ void nms_reduce(const unsigned int* mask, const int* above, int words, int max_output,
                                   const int* sorted_idx, int* kept, int* kept_count)
        {
            extern __shared__ unsigned int removed[];
            const int count = *above;
            for (int w = threadIdx.x; w < words; w += blockDim.x)
                removed[w] = 0;
            __syncthreads();
            int selected = 0;
            for (int i = 0; i < count; i++) {
                if (removed[i / 32] & (1u << (i % 32)))
                    continue;
                if (threadIdx.x == 0)
                    kept[selected] = sorted_idx[i];
                selected++;
                if (max_output > 0 && selected >= max_output)
                    break;
                const unsigned int* row = mask + static_cast<std::size_t>(i) * words;
                for (int w = threadIdx.x; w < words; w += blockDim.x)
                    removed[w] |= row[w];
                __syncthreads();
            }
            if (threadIdx.x == 0)
                *kept_count = selected;
        }

        __global__ void nms_compact(const int* kept_all, const int* counts, int tasks, int num_boxes, int classes,
                                    std::int64_t* output, int* total)
        {
            int offset = 0;
            for (int t = 0; t < tasks; t++) {
                const int c = counts[t];
                const std::int64_t b = t / classes, cl = t % classes;
                for (int k = threadIdx.x; k < c; k += blockDim.x) {
                    std::int64_t* dst = output + 3 * static_cast<std::size_t>(offset + k);
                    dst[0] = b;
                    dst[1] = cl;
                    dst[2] = kept_all[static_cast<std::size_t>(t) * num_boxes + k];
                }
                offset += c;
            }
            if (threadIdx.x == 0)
                *total = offset;
        }
    }

    namespace {
        struct NmsLayout {
            std::size_t sorted_scores, iota, sorted_idx, rects, mask, kept, counts, scalars, temp, temp_bytes, total;
        };

        std::size_t align256(std::size_t bytes)
        {
            return (bytes + 255) & ~static_cast<std::size_t>(255);
        }

        NmsLayout nmsLayout(int batch, int classes, int num_boxes, cudaStream_t stream)
        {
            const std::size_t n = static_cast<std::size_t>(num_boxes);
            const std::size_t words = (n + 31) / 32;
            const std::size_t tasks = static_cast<std::size_t>(batch) * classes;
            NmsLayout L;
            L.temp_bytes = 0;
            CUDA4DNN_CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(
                nullptr, L.temp_bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
                static_cast<const int*>(nullptr), static_cast<int*>(nullptr), num_boxes, 0,
                static_cast<int>(sizeof(float) * 8), stream));
            L.sorted_scores = 0;
            L.iota = L.sorted_scores + align256(n * sizeof(float));
            L.sorted_idx = L.iota + align256(n * sizeof(int));
            L.rects = L.sorted_idx + align256(n * sizeof(int));
            L.mask = L.rects + align256(n * 4 * sizeof(double));
            L.kept = L.mask + align256(n * words * sizeof(unsigned int));
            L.counts = L.kept + align256(tasks * n * sizeof(int));
            L.scalars = L.counts + align256(tasks * sizeof(int));
            L.temp = L.scalars + align256(2 * sizeof(int));
            L.total = L.temp + align256(L.temp_bytes);
            return L;
        }
    }

    std::size_t nms_workspace(const Stream& stream, int batch, int classes, int num_boxes)
    {
        return nmsLayout(batch, classes, num_boxes, stream.get()).total;
    }

    int nms(const Stream& stream, Span<std::int64_t> output, View<float> boxes, View<float> scores,
            int batch, int classes, int num_boxes, bool center_point_box, int max_output,
            float iou_threshold, float score_threshold, unsigned char* workspace, std::size_t workspace_bytes)
    {
        CV_Assert(batch > 0 && classes > 0 && num_boxes > 0 && num_boxes <= kNmsMaxBoxes);
        CV_Assert(boxes.size() == static_cast<std::size_t>(batch) * num_boxes * 4);
        CV_Assert(scores.size() == static_cast<std::size_t>(batch) * classes * num_boxes);

        cudaStream_t s = stream.get();
        const NmsLayout L = nmsLayout(batch, classes, num_boxes, s);
        CV_Assert(workspace && workspace_bytes >= L.total);

        const int tasks = batch * classes;
        const int words = (num_boxes + 31) / 32;
        const std::size_t cap = max_output > 0 ? static_cast<std::size_t>(std::min(max_output, num_boxes))
                                               : static_cast<std::size_t>(num_boxes);
        CV_Assert(output.size() >= static_cast<std::size_t>(tasks) * cap * 3);

        float* sorted_scores = reinterpret_cast<float*>(workspace + L.sorted_scores);
        int* iota = reinterpret_cast<int*>(workspace + L.iota);
        int* sorted_idx = reinterpret_cast<int*>(workspace + L.sorted_idx);
        double* rects = reinterpret_cast<double*>(workspace + L.rects);
        unsigned int* mask = reinterpret_cast<unsigned int*>(workspace + L.mask);
        int* kept = reinterpret_cast<int*>(workspace + L.kept);
        int* counts = reinterpret_cast<int*>(workspace + L.counts);
        int* above = reinterpret_cast<int*>(workspace + L.scalars);
        int* total = above + 1;
        void* temp = workspace + L.temp;
        std::size_t temp_bytes = L.temp_bytes;

        const size_type n = static_cast<size_type>(num_boxes);
        const size_type mask_elems = static_cast<size_type>(num_boxes) * words;

        auto iota_kernel = raw::nms_iota;
        launch_kernel(iota_kernel, make_policy(iota_kernel, n, 0, stream), iota, n);

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < classes; c++) {
                const int t = b * classes + c;
                const float* task_scores = scores.data().get() + static_cast<std::size_t>(t) * num_boxes;
                CUDA4DNN_CHECK_CUDA(cudaMemsetAsync(above, 0, sizeof(int), s));
                CUDA4DNN_CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(
                    temp, temp_bytes, task_scores, sorted_scores, iota, sorted_idx, num_boxes, 0,
                    static_cast<int>(sizeof(float) * 8), s));

                auto prepare = raw::nms_prepare;
                launch_kernel(prepare, make_policy(prepare, n, 0, stream), sorted_scores, score_threshold, sorted_idx,
                              boxes.data().get() + static_cast<std::size_t>(b) * num_boxes * 4, center_point_box,
                              rects, above, n);

                auto mask_kernel = raw::nms_mask;
                launch_kernel(mask_kernel, make_policy(mask_kernel, mask_elems, 0, stream), rects, above, words,
                              iou_threshold, mask, mask_elems);

                auto reduce = raw::nms_reduce;
                launch_kernel(reduce, execution_policy(1, 256, static_cast<std::size_t>(words) * sizeof(unsigned int), stream),
                              mask, above, words, max_output, sorted_idx,
                              kept + static_cast<std::size_t>(t) * num_boxes, counts + t);
            }
        }

        auto compact = raw::nms_compact;
        launch_kernel(compact, execution_policy(1, 256, stream), kept, counts, tasks, num_boxes, classes,
                      output.data().get(), total);

        int selected = 0;
        CUDA4DNN_CHECK_CUDA(cudaMemcpyAsync(&selected, total, sizeof(int), cudaMemcpyDeviceToHost, s));
        CUDA4DNN_CHECK_CUDA(cudaStreamSynchronize(s));
        return selected;
    }

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
