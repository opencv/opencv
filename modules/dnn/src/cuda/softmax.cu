// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "limits.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/softmax.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace detail {

        template <class T>
        struct softmax_accum { using type = T; };
        template <>
        struct softmax_accum<__half> { using type = float; };

        template <typename T>
        struct Add {
            __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
        };

        template <typename T>
        struct Max {
            __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? b : a; }
        };

        template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
        __device__ __forceinline__ void warp_reduce(acc_t* sum) {
            ReduceOp<acc_t> r;
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
                for (int i = 0; i < WARP_BATCH; ++i) {
                    acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
                    sum[i] = r(sum[i], b);
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
        __global__ void softmax_warp_forward(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
            constexpr int next_power_of_two = 1 << log2_elements;
            constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
            constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
            constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

            int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

            int local_batches = batch_size - first_batch;
            if (local_batches > WARP_BATCH)
                local_batches = WARP_BATCH;

            int local_idx = threadIdx.x;

            src += static_cast<std::int64_t>(first_batch) * stride + local_idx;
            dst += static_cast<std::int64_t>(first_batch) * stride + local_idx;

            acc_t elements[WARP_BATCH][WARP_ITERATIONS];
            for (int i = 0; i < WARP_BATCH; ++i) {
                int batch_element_count = (i >= local_batches) ? 0 : element_count;
                for (int it = 0; it < WARP_ITERATIONS; ++it) {
                    int element_index = local_idx + it * WARP_SIZE;
                    if (element_index < batch_element_count) {
                        elements[i][it] = src[i * element_count + it * WARP_SIZE];
                    } else {
                        elements[i][it] = static_cast<acc_t>(-INFINITY);
                    }
                }
            }

            acc_t max_value[WARP_BATCH];
#pragma unroll
            for (int i = 0; i < WARP_BATCH; ++i) {
                max_value[i] = elements[i][0];
#pragma unroll
                for (int it = 1; it < WARP_ITERATIONS; ++it) {
                    max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                }
            }
            warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

            acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
            for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
                for (int it = 0; it < WARP_ITERATIONS; ++it) {
                    if (is_log_softmax) {
                        sum[i] += std::exp((float)(elements[i][it] - max_value[i]));
                    } else {
                        elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
                        sum[i] += elements[i][it];
                    }
                }
            }
            warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

#pragma unroll
            for (int i = 0; i < WARP_BATCH; ++i) {
                if (i >= local_batches)
                    break;
                // subtract the max before the log-sum: max + log(sum) rounds away small terms at large inputs
                if (is_log_softmax) sum[i] = std::log((float)(sum[i]));
#pragma unroll
                for (int it = 0; it < WARP_ITERATIONS; ++it) {
                    int element_index = local_idx + it * WARP_SIZE;
                    if (element_index < element_count) {
                        if (is_log_softmax) {
                            dst[i * element_count + it * WARP_SIZE] = (elements[i][it] - max_value[i]) - sum[i];
                        } else {
                            dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
        __global__ void softmax_warp_forward_resource_efficient(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
            constexpr int next_power_of_two = 1 << log2_elements;
            constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
            constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;

            int local_idx = threadIdx.x;
            src += static_cast<std::int64_t>(blockIdx.x) * stride + local_idx;
            dst += static_cast<std::int64_t>(blockIdx.x) * stride + local_idx;
            extern __shared__ unsigned char smem[];
            input_t(&elements)[WARP_ITERATIONS][WARP_SIZE] = *reinterpret_cast<input_t(*)[WARP_ITERATIONS][WARP_SIZE]>(smem);
#pragma unroll
            for (int it = 0; it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < element_count) {
                    elements[it][local_idx] = src[it * WARP_SIZE];
                } else {
                    elements[it][local_idx] = static_cast<input_t>(static_cast<acc_t>(-INFINITY));
                }
            }
            input_t max_value = elements[0][local_idx];
#pragma unroll
            for (int it = 1; it < WARP_ITERATIONS; ++it) {
                max_value = (max_value > elements[it][local_idx]) ? max_value : elements[it][local_idx];
            }
            warp_reduce<input_t, 1, WARP_SIZE, Max>(&max_value);
            acc_t sum{0.0f};
            for (int it = 0; it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index >= element_count)
                    break;
                if (is_log_softmax) {
                    sum += std::exp((float)(elements[it][local_idx] - max_value));
                } else {
                    acc_t tmp = std::exp((float)(elements[it][local_idx] - max_value));
                    elements[it][local_idx] = tmp;
                    sum += tmp;
                }
            }
            warp_reduce<acc_t, 1, WARP_SIZE, Add>(&sum);
            if (is_log_softmax) sum = std::log((float)(sum));
            acc_t invsum = static_cast<acc_t>(1.0f / sum);
#pragma unroll
            for (int it = 0; it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < element_count) {
                    if (is_log_softmax) {
                        dst[it * WARP_SIZE] = ((float)elements[it][local_idx] - (float)max_value) - sum;
                    } else {
                        dst[it * WARP_SIZE] = (float)elements[it][local_idx] * invsum;
                    }
                } else {
                    break;
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        void dispatch_warpwise_softmax_forward(cudaStream_t stream, output_t* dst, const input_t* src, int softmax_elements,
                                               int softmax_elements_stride, int batch_count) {
            if (softmax_elements == 0) {
                return;
            } else {
                int log2_elements = log2_ceil(softmax_elements);
                const int next_power_of_two = 1 << log2_elements;

                int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
                int threads_per_block, shared_memory_size;
                int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
                if (log2_elements <= 10) {
                    threads_per_block = 128;
                    shared_memory_size = 0;
                } else {
                    threads_per_block = 32;
                    shared_memory_size = next_power_of_two * sizeof(input_t);
                }
                int warps_per_block = (threads_per_block / warp_size);
                int batches_per_block = warps_per_block * batches_per_warp;
                int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
                dim3 threads(warp_size, warps_per_block, 1);
                switch (log2_elements) {
#define LAUNCH_KERNEL(kernel_name, log2_elements_value)                      \
    kernel_name<input_t, output_t, acc_t, log2_elements_value, is_log_softmax> \
        <<<blocks, threads, shared_memory_size, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);

#define CASE_LOG2_ELEMENTS(kernel_name, log2_elements_value)                      \
    case log2_elements_value: {                                                   \
        LAUNCH_KERNEL(kernel_name, log2_elements_value)                           \
    } break

                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 0);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 1);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 2);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 3);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 4);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 5);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 6);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 7);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 8);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 9);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward, 10);
                    CASE_LOG2_ELEMENTS(softmax_warp_forward_resource_efficient, 11);
#undef LAUNCH_KERNEL
#undef CASE_LOG2_ELEMENTS
                }
            }
        }

        const int max_threads = 1024;

        dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
            uint64_t block_size = 1;
            uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));

            if (ILP > 1) {
                max_block_size /= 2;
            }

            while (block_size < (max_block_size)) block_size *= 2;
            block_size = std::max(block_size, static_cast<uint64_t>(GPU_WARP_SIZE));
            return dim3(static_cast<unsigned int>(block_size));
        }

        template <typename T, typename AccumT>
        struct MaxFloat {
            __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
                return ::max(max, (AccumT)v);
            }
        };

        template <typename T, typename AccumT>
        struct AddFloat {
            __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
                return sum + (AccumT)v;
            }
        };

        template <typename T, typename AccumT>
        struct SumExpFloat {
            __device__ __forceinline__ SumExpFloat(AccumT v) : max_k(v) {}

            __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
                return sum + std::exp((AccumT)v - max_k);
            }

            const AccumT max_k;
        };

        template <template <typename> class Reduction, typename AccumT>
        __device__ __forceinline__ AccumT blockReduce(AccumT* smem, AccumT val,
                                                      const Reduction<AccumT>& r,
                                                      AccumT defaultVal) {
            __syncthreads();

            smem[threadIdx.x] = val;

            __syncthreads();

            AccumT warpVal = defaultVal;

            if (threadIdx.x < GPU_WARP_SIZE) {
                int warps_per_block = blockDim.x / GPU_WARP_SIZE;
                for (int i = 0; i < warps_per_block; ++i) {
                    warpVal = r(warpVal, smem[i * GPU_WARP_SIZE + threadIdx.x]);
                }
                smem[threadIdx.x] = warpVal;
            }

            __syncthreads();

            AccumT blockVal = defaultVal;

            if (threadIdx.x == 0) {
#pragma unroll
                for (int i = 0; i < GPU_WARP_SIZE; ++i) {
                    blockVal = r(blockVal, smem[i]);
                }
                smem[0] = blockVal;
            }

            __syncthreads();
            return smem[0];
        }

        template <template <typename, typename> class Reduction, int ILP, typename T, typename AccumT>
        __device__ __forceinline__ AccumT ilpReduce(int shift,
                                                    T* data,
                                                    int size,
                                                    const Reduction<T, AccumT>& r,
                                                    AccumT defaultVal) {
            using LoadT = aligned_vector<T, ILP>;
            AccumT threadVal = defaultVal;
            int offset = threadIdx.x;

            if (shift > 0) {
                data -= shift;
                size += shift;
                if (threadIdx.x >= shift && threadIdx.x < size) {
                    threadVal = r(threadVal, data[offset]);
                }
                size -= blockDim.x;
                data += blockDim.x;
            }

            if (size <= 0) return threadVal;

            int last = size % (ILP * blockDim.x);

            T v[ILP];
            LoadT* value = reinterpret_cast<LoadT*>(&v);

            for (; offset * ILP < (size - last); offset += blockDim.x) {
                *value = reinterpret_cast<LoadT*>(data)[offset];

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    threadVal = r(threadVal, v[j]);
                }
            }

            offset = size - last + threadIdx.x;
            for (; offset < size; offset += blockDim.x)
                threadVal = r(threadVal, data[offset]);

            return threadVal;
        }

        template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
        __device__ __forceinline__ void WriteFpropResultsVectorized(int size,
                                                                    const int shift,
                                                                    scalar_t* input,
                                                                    outscalar_t* output,
                                                                    Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
            using LoadT = aligned_vector<scalar_t, ILP>;
            using StoreT = aligned_vector<outscalar_t, ILP>;

            int offset = threadIdx.x;

            if (shift > 0) {
                input -= shift;
                output -= shift;
                size += shift;

                if (threadIdx.x >= shift && threadIdx.x < size) {
                    output[offset] = epilogue(input[offset]);
                }
                size -= blockDim.x;
                input += blockDim.x;
                output += blockDim.x;
            }

            if (size <= 0) return;

            const int last = size % (ILP * blockDim.x);

            scalar_t in_v[ILP];
            LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);

            outscalar_t out_v[ILP];
            StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);

            for (; offset * ILP < (size - last); offset += blockDim.x) {
                *in_value = reinterpret_cast<LoadT*>(input)[offset];

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    out_v[j] = epilogue(in_v[j]);
                }

                reinterpret_cast<StoreT*>(output)[offset] = *out_value;
            }

            offset = size - last + threadIdx.x;
            for (; offset < size; offset += blockDim.x) {
                output[offset] = epilogue(input[offset]);
            }
        }

        template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
        __device__ __forceinline__ void WriteFpropResults(int classes,
                                                          scalar_t* input,
                                                          outscalar_t* output,
                                                          Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
            int offset = threadIdx.x;

            int last = classes % (ILP * blockDim.x);

            for (; offset < classes - last; offset += blockDim.x * ILP) {
                scalar_t tmp[ILP];

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    tmp[j] = input[offset + j * blockDim.x];
                }
#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    output[offset + j * blockDim.x] = epilogue(tmp[j]);
                }
            }

            for (; offset < classes; offset += blockDim.x) {
                output[offset] = epilogue(input[offset]);
            }
        }

        template <typename T, typename AccumT, typename OutT>
        struct LogSoftMaxForwardEpilogue {
            __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
                : max_input(max_input), logsum(std::log(sum)) {}

            __device__ __forceinline__ OutT operator()(T input) const {
                return static_cast<OutT>((AccumT)input - max_input - logsum);
            }

            const AccumT max_input;
            const AccumT logsum;
        };

        template <typename T, typename AccumT, typename OutT>
        struct SoftMaxForwardEpilogue {
            __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
                : max_input(max_input), sum(sum) {}

            __device__ __forceinline__ OutT operator()(T input) const {
                return static_cast<OutT>(std::exp((AccumT)input - max_input) / sum);
            }

            const AccumT max_input;
            const AccumT sum;
        };

        template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t,
                  template <typename, typename, typename> class Epilogue>
        __global__ void softmax_block_forward(outscalar_t* output, scalar_t* input, int classes,
                                              int input_stride, int output_stride) {
            extern __shared__ unsigned char smem[];
            auto sdata = reinterpret_cast<accscalar_t*>(smem);

            input += static_cast<std::int64_t>(blockIdx.x) * input_stride;
            output += static_cast<std::int64_t>(blockIdx.x) * output_stride;

            const int input_align_bytes = ILP * sizeof(scalar_t);
            const int output_align_bytes = ILP * sizeof(outscalar_t);

            const int shift = ((uint64_t)input) % input_align_bytes / sizeof(scalar_t);
            const int output_shift = ((uint64_t)output) % output_align_bytes / sizeof(outscalar_t);

            accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
                shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), device::numeric_limits<accscalar_t>::lowest());
            accscalar_t max_k = blockReduce<Max, accscalar_t>(
                sdata, threadMax, Max<accscalar_t>(), device::numeric_limits<accscalar_t>::lowest());

            accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
                shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
            accscalar_t sumAll = blockReduce<Add, accscalar_t>(
                sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

            Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);

            if (shift == output_shift) {
                WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
            } else {
                WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        void dispatch_blockwise_softmax_forward(cudaStream_t stream, output_t* output, const input_t* input, int softmax_elements,
                                                int input_stride, int output_stride, int batch_count) {
            dim3 grid(batch_count);
            constexpr int ILP = sizeof(float4) / sizeof(input_t);
            dim3 block = SoftMax_getBlockSize(ILP, softmax_elements);
            if (is_log_softmax) {
                softmax_block_forward<ILP, input_t, acc_t, output_t, LogSoftMaxForwardEpilogue>
                    <<<grid, block, block.x * sizeof(acc_t), stream>>>(output, const_cast<input_t*>(input),
                                                                       softmax_elements, input_stride, output_stride);
            } else {
                softmax_block_forward<ILP, input_t, acc_t, output_t, SoftMaxForwardEpilogue>
                    <<<grid, block, block.x * sizeof(acc_t), stream>>>(output, const_cast<input_t*>(input),
                                                                       softmax_elements, input_stride, output_stride);
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        __global__ void softmax_strided_forward(output_t* dst, const input_t* src,
                                                int axis_size, int inner_size, std::int64_t num_lanes) {
            for (std::int64_t lane = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 lane < num_lanes;
                 lane += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
            {
                const std::int64_t outer = lane / inner_size;
                const std::int64_t stride = inner_size;
                const std::int64_t base = outer * static_cast<std::int64_t>(axis_size) * stride + (lane - outer * stride);

                acc_t max_value = static_cast<acc_t>(-INFINITY);
                for (int k = 0; k < axis_size; k++) {
                    const acc_t v = static_cast<acc_t>(src[base + k * stride]);
                    max_value = v > max_value ? v : max_value;
                }

                acc_t sum = static_cast<acc_t>(0);
                for (int k = 0; k < axis_size; k++)
                    sum += std::exp((float)(static_cast<acc_t>(src[base + k * stride]) - max_value));

                if (is_log_softmax) {
                    const acc_t log_sum = static_cast<acc_t>(std::log((float)sum));
                    for (int k = 0; k < axis_size; k++) {
                        const std::int64_t o = base + k * stride;
                        dst[o] = static_cast<output_t>(static_cast<acc_t>(src[o]) - max_value - log_sum);
                    }
                } else {
                    for (int k = 0; k < axis_size; k++) {
                        const std::int64_t o = base + k * stride;
                        dst[o] = static_cast<output_t>(std::exp((float)(static_cast<acc_t>(src[o]) - max_value)) / sum);
                    }
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        void dispatch_strided_softmax_forward(cudaStream_t stream, output_t* output, const input_t* input,
                                              int axis_size, int outer_size, int inner_size) {
            const std::int64_t num_lanes = static_cast<std::int64_t>(outer_size) * static_cast<std::int64_t>(inner_size);
            constexpr int BLOCK_SIZE = 256;
            std::int64_t num_blocks = (num_lanes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (num_blocks > 65535)
                num_blocks = 65535;
            softmax_strided_forward<input_t, output_t, acc_t, is_log_softmax>
                <<<(unsigned int)num_blocks, BLOCK_SIZE, 0, stream>>>(output, input, axis_size, inner_size, num_lanes);
        }
    }

    template <class T>
    void softmax(const Stream& stream, Span<T> output, View<T> input, int axis_size, int outer_size, int inner_size, bool log_softmax) {
        using acc_t = typename detail::softmax_accum<T>::type;

        const int D = axis_size;
        const int N = outer_size;
        if (D == 0 || N == 0 || inner_size == 0)
            return;

        T* dst = output.data().get();
        const T* src = input.data().get();

        if (inner_size != 1) {
            if (log_softmax)
                detail::dispatch_strided_softmax_forward<T, T, acc_t, true>(stream.get(), dst, src, D, N, inner_size);
            else
                detail::dispatch_strided_softmax_forward<T, T, acc_t, false>(stream.get(), dst, src, D, N, inner_size);
            return;
        }

        const bool use_softmax_warp_forward_resource_efficient =
            1024 < D && D <= 2048 && D * static_cast<int>(sizeof(T)) <= 4096 && N >= 8192;
        if ((D <= 1024 && D * static_cast<int>(sizeof(T)) <= 4096) || use_softmax_warp_forward_resource_efficient) {
            if (log_softmax)
                detail::dispatch_warpwise_softmax_forward<T, T, acc_t, true>(stream.get(), dst, src, D, D, N);
            else
                detail::dispatch_warpwise_softmax_forward<T, T, acc_t, false>(stream.get(), dst, src, D, D, N);
        } else {
            if (log_softmax)
                detail::dispatch_blockwise_softmax_forward<T, T, acc_t, true>(stream.get(), dst, src, D, D, D, N);
            else
                detail::dispatch_blockwise_softmax_forward<T, T, acc_t, false>(stream.get(), dst, src, D, D, D, N);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void softmax(const Stream&, Span<__half>, View<__half>, int, int, int, bool);
#endif
    template void softmax(const Stream&, Span<float>, View<float>, int, int, int, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
