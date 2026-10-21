// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>

#include "types.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/topk.hpp"

#include <opencv2/core.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        struct Candidate {
            float value;
            int index;
        };

        /* Strict weak ordering matching topk2_layer.cpp's comparators: better value
         * first, and on an exact tie the smaller index. */
        __device__ __forceinline__ bool better_than(const Candidate& a, const Candidate& b, bool largest) {
            if (a.value == b.value)
                return a.index < b.index;
            return largest ? (a.value > b.value) : (a.value < b.value);
        }

        struct MergeCandidate {
            bool largest;
            __device__ __forceinline__ Candidate operator()(const Candidate& a, const Candidate& b) const {
                return better_than(a, b, largest) ? a : b;
            }
        };

        template <class T, int BlockSize>
        __global__ void topk(Span<T> values, Span<std::int64_t> indices, View<T> input,
                             int outer, int dim_axis, int inner, int k, bool largest)
        {
            using BlockReduce = cub::BlockReduce<Candidate, BlockSize>;
            __shared__ typename BlockReduce::TempStorage reduce_storage;
            __shared__ Candidate selected;

            const int nslices = outer * inner;
            const float worst = largest ? -::cuda::std::numeric_limits<float>::infinity()
                                        :  ::cuda::std::numeric_limits<float>::infinity();

            for (int slice = blockIdx.x; slice < nslices; slice += gridDim.x) {
                const int b = slice / inner;
                const int j = slice - b * inner;

                const std::int64_t in_base  = static_cast<std::int64_t>(b) * dim_axis * inner + j;
                const std::int64_t out_base = static_cast<std::int64_t>(b) * k * inner + j;

                for (int rank = 0; rank < k; ++rank) {
                    Candidate best{worst, ::cuda::std::numeric_limits<int>::max()};

                    for (int u = threadIdx.x; u < dim_axis; u += BlockSize) {
                        const float v = static_cast<float>(input.data().get()[in_base + static_cast<std::int64_t>(u) * inner]);
                        const Candidate cand{v, u};

                        /* Everything at or before the previous pick is spent. Ordering
                         * strictly means no element can be chosen twice, and duplicates
                         * are still all reachable because index breaks the tie. */
                        if (rank > 0 && !better_than(selected, cand, largest))
                            continue;

                        if (better_than(cand, best, largest))
                            best = cand;
                    }

                    const Candidate winner = BlockReduce(reduce_storage).Reduce(best, MergeCandidate{largest});

                    if (threadIdx.x == 0) {
                        selected = winner;
                        const std::int64_t out = out_base + static_cast<std::int64_t>(rank) * inner;
                        values[out] = static_cast<T>(winner.value);
                        indices[out] = static_cast<std::int64_t>(winner.index);
                    }
                    __syncthreads();  // publishes `selected` and frees reduce_storage for the next rank
                }
            }
        }

        __global__ void topk_sort_init(int* ids, int* offsets, int n, int nsegments, int dim_axis)
        {
            const int stride = gridDim.x * blockDim.x;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
                ids[i] = i % dim_axis;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i <= nsegments; i += stride)
                offsets[i] = i * dim_axis;
        }

        template <class T>
        __global__ void topk_sort_gather(Span<T> values, Span<std::int64_t> indices,
                                         const T* keys, const int* ids, int dim_axis, int k)
        {
            const int m = static_cast<int>(values.size());
            const int stride = gridDim.x * blockDim.x;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
                const int b = i / k;
                const int src = b * dim_axis + (i - b * k);
                values[i] = keys[src];
                indices[i] = static_cast<std::int64_t>(ids[src]);
            }
        }
    }

    namespace detail {
        struct TopKSortLayout {
            std::size_t keys, ids_in, ids_out, offsets, temp, total;
        };

        inline std::size_t alignTo256(std::size_t bytes)
        {
            return (bytes + 255) & ~static_cast<std::size_t>(255);
        }

        template <class T>
        TopKSortLayout topkSortLayout(int outer, int dim_axis, bool largest, cudaStream_t stream)
        {
            const std::int64_t n = static_cast<std::int64_t>(outer) * dim_axis;
            std::size_t temp_bytes = 0;
            if (outer == 1 && largest)
                CUDA4DNN_CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(
                    nullptr, temp_bytes, static_cast<const T*>(nullptr), static_cast<T*>(nullptr),
                    static_cast<const int*>(nullptr), static_cast<int*>(nullptr), n, 0,
                    static_cast<int>(sizeof(T) * 8), stream));
            else if (outer == 1)
                CUDA4DNN_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                    nullptr, temp_bytes, static_cast<const T*>(nullptr), static_cast<T*>(nullptr),
                    static_cast<const int*>(nullptr), static_cast<int*>(nullptr), n, 0,
                    static_cast<int>(sizeof(T) * 8), stream));
            else if (largest)
                CUDA4DNN_CHECK_CUDA(cub::DeviceSegmentedSort::StableSortPairsDescending(
                    nullptr, temp_bytes, static_cast<const T*>(nullptr), static_cast<T*>(nullptr),
                    static_cast<const int*>(nullptr), static_cast<int*>(nullptr), n, outer,
                    static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), stream));
            else
                CUDA4DNN_CHECK_CUDA(cub::DeviceSegmentedSort::StableSortPairs(
                    nullptr, temp_bytes, static_cast<const T*>(nullptr), static_cast<T*>(nullptr),
                    static_cast<const int*>(nullptr), static_cast<int*>(nullptr), n, outer,
                    static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), stream));

            TopKSortLayout layout;
            layout.keys = 0;
            layout.ids_in = layout.keys + alignTo256(static_cast<std::size_t>(n) * sizeof(T));
            layout.ids_out = layout.ids_in + alignTo256(static_cast<std::size_t>(n) * sizeof(int));
            layout.offsets = layout.ids_out + alignTo256(static_cast<std::size_t>(n) * sizeof(int));
            layout.temp = layout.offsets + alignTo256(static_cast<std::size_t>(outer + 1) * sizeof(int));
            layout.total = layout.temp + alignTo256(temp_bytes);
            return layout;
        }
    }

    template <class T>
    void topk(const Stream& stream,
        Span<T> values, Span<std::int64_t> indices, View<T> input,
        int outer, int dim_axis, int inner, int k, bool largest)
    {
        CV_Assert(outer > 0 && dim_axis > 0 && inner > 0);
        CV_Assert(k > 0 && k <= dim_axis);

        constexpr int block_size = 256;
        constexpr int max_blocks = 65535;
        const int nslices = outer * inner;
        const int grid_size = std::min(max_blocks, nslices);

        raw::topk<T, block_size><<<grid_size, block_size, 0, stream.get()>>>(
            values, indices, input, outer, dim_axis, inner, k, largest);
    }

    template <class T>
    std::size_t topk_sort_workspace(const Stream& stream, int outer, int dim_axis, bool largest)
    {
        return detail::topkSortLayout<T>(outer, dim_axis, largest, stream.get()).total;
    }

    template <class T>
    void topk_sort(const Stream& stream,
        Span<T> values, Span<std::int64_t> indices, View<T> input,
        int outer, int dim_axis, int k, bool largest,
        void* workspace, std::size_t workspace_bytes)
    {
        CV_Assert(outer > 0 && dim_axis > 0);
        CV_Assert(k > 0 && k <= dim_axis);

        const detail::TopKSortLayout layout = detail::topkSortLayout<T>(outer, dim_axis, largest, stream.get());
        CV_Assert(workspace != nullptr && workspace_bytes >= layout.total);

        unsigned char* base = static_cast<unsigned char*>(workspace);
        T* keys = reinterpret_cast<T*>(base + layout.keys);
        int* ids_in = reinterpret_cast<int*>(base + layout.ids_in);
        int* ids_out = reinterpret_cast<int*>(base + layout.ids_out);
        int* offsets = reinterpret_cast<int*>(base + layout.offsets);
        void* temp = base + layout.temp;
        std::size_t temp_bytes = layout.total - layout.temp;

        const int n = outer * dim_axis;
        constexpr int block_size = 256;
        constexpr int max_blocks = 4096;

        const int init_grid = std::min(max_blocks, (std::max(n, outer + 1) + block_size - 1) / block_size);
        raw::topk_sort_init<<<init_grid, block_size, 0, stream.get()>>>(ids_in, offsets, n, outer, dim_axis);

        const T* keys_in = input.data().get();
        if (outer == 1 && largest)
            CUDA4DNN_CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(
                temp, temp_bytes, keys_in, keys, ids_in, ids_out, n, 0, static_cast<int>(sizeof(T) * 8), stream.get()));
        else if (outer == 1)
            CUDA4DNN_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                temp, temp_bytes, keys_in, keys, ids_in, ids_out, n, 0, static_cast<int>(sizeof(T) * 8), stream.get()));
        else if (largest)
            CUDA4DNN_CHECK_CUDA(cub::DeviceSegmentedSort::StableSortPairsDescending(
                temp, temp_bytes, keys_in, keys, ids_in, ids_out, n, outer, offsets, offsets + 1, stream.get()));
        else
            CUDA4DNN_CHECK_CUDA(cub::DeviceSegmentedSort::StableSortPairs(
                temp, temp_bytes, keys_in, keys, ids_in, ids_out, n, outer, offsets, offsets + 1, stream.get()));

        const int m = outer * k;
        const int gather_grid = std::min(max_blocks, (m + block_size - 1) / block_size);
        raw::topk_sort_gather<T><<<gather_grid, block_size, 0, stream.get()>>>(values, indices, keys, ids_out, dim_axis, k);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void topk(const Stream&, Span<__half>, Span<std::int64_t>, View<__half>, int, int, int, int, bool);
#endif
    template void topk(const Stream&, Span<float>, Span<std::int64_t>, View<float>, int, int, int, int, bool);
    template void topk(const Stream&, Span<int8_t>, Span<std::int64_t>, View<int8_t>, int, int, int, int, bool);
    template void topk(const Stream&, Span<uint8_t>, Span<std::int64_t>, View<uint8_t>, int, int, int, int, bool);
    template void topk(const Stream&, Span<int32_t>, Span<std::int64_t>, View<int32_t>, int, int, int, int, bool);
    template void topk(const Stream&, Span<int64_t>, Span<std::int64_t>, View<int64_t>, int, int, int, int, bool);

#define CUDA4DNN_INSTANTIATE_TOPK_SORT(T) \
    template std::size_t topk_sort_workspace<T>(const Stream&, int, int, bool); \
    template void topk_sort<T>(const Stream&, Span<T>, Span<std::int64_t>, View<T>, int, int, int, bool, void*, std::size_t);

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    CUDA4DNN_INSTANTIATE_TOPK_SORT(__half)
#endif
    CUDA4DNN_INSTANTIATE_TOPK_SORT(float)
    CUDA4DNN_INSTANTIATE_TOPK_SORT(int8_t)
    CUDA4DNN_INSTANTIATE_TOPK_SORT(uint8_t)
    CUDA4DNN_INSTANTIATE_TOPK_SORT(int32_t)
    CUDA4DNN_INSTANTIATE_TOPK_SORT(int64_t)
#undef CUDA4DNN_INSTANTIATE_TOPK_SORT

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
