// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "limits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/reduce.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace detail {
        constexpr int kMaxReduceRank = 16;

        template <class T>
        struct accumulation_type { using type = T; };
        template <>
        struct accumulation_type<__half> { using type = float; };

        struct ReduceSumNdMetadata {
            int output_segment_count{};
            int reduction_segment_count{};
            std::int64_t output_segment_sizes[kMaxReduceRank]{};
            std::int64_t output_segment_strides[kMaxReduceRank]{};
            std::int64_t reduction_segment_sizes[kMaxReduceRank]{};
            std::int64_t reduction_segment_strides[kMaxReduceRank]{};
            std::int64_t output_count{};
            std::int64_t reduction_count{};
        };

        // Host-side: collapses `dims` into runs of reduced / non-reduced axes to shorten the per-element index decode.
        static inline void build_reduce_metadata(const std::vector<std::int64_t>& dims,
                                                 const std::vector<std::int64_t>& axes,
                                                 ReduceSumNdMetadata& metadata) {
            const int rank = static_cast<int>(dims.size());
            std::array<std::int64_t, kMaxReduceRank> strides{};
            std::int64_t stride = 1;
            for (int axis = rank - 1; axis >= 0; --axis) {
                CV_Assert(dims[axis] > 0);
                strides[axis] = stride;
                stride *= dims[axis];
            }

            std::array<bool, kMaxReduceRank> reduced{};
            if (axes.empty()) {
                for (int axis = 0; axis < rank; ++axis) reduced[axis] = true;
            } else {
                for (std::int64_t axis : axes) {
                    if (axis < 0) axis += rank;
                    CV_Assert(axis >= 0 && axis < rank);
                    reduced[axis] = true;
                }
            }

            std::int64_t output_count = 1;
            std::int64_t reduction_count = 1;
            for (int axis = 0; axis < rank;) {
                const bool is_reduced = reduced[axis];
                std::int64_t segment_size = 1;
                int last_axis = axis;
                do {
                    segment_size *= dims[axis];
                    last_axis = axis++;
                } while (axis < rank && reduced[axis] == is_reduced);

                if (is_reduced) {
                    const int segment = metadata.reduction_segment_count++;
                    metadata.reduction_segment_sizes[segment] = segment_size;
                    metadata.reduction_segment_strides[segment] = strides[last_axis];
                    reduction_count *= segment_size;
                } else {
                    const int segment = metadata.output_segment_count++;
                    metadata.output_segment_sizes[segment] = segment_size;
                    metadata.output_segment_strides[segment] = strides[last_axis];
                    output_count *= segment_size;
                }
            }
            metadata.output_count = output_count;
            metadata.reduction_count = reduction_count;
        }

        template <typename T>
        struct SumState {
            T sum{};

            __device__ __forceinline__ void Add(T value) { sum += value; }
            __device__ __forceinline__ T Result() const { return sum; }
        };

        template <typename T>
        struct MergeSumState {
            __device__ __forceinline__ SumState<T> operator()(SumState<T> lhs, const SumState<T>& rhs) const {
                lhs.Add(rhs.sum);
                return lhs;
            }
        };

        __device__ __forceinline__ std::int64_t reduce_output_base(const ReduceSumNdMetadata& metadata, std::int64_t output_index) {
            std::int64_t remaining = output_index;
            std::int64_t base = 0;
            for (int segment = metadata.output_segment_count - 1; segment >= 0; --segment) {
                const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.output_segment_sizes[segment];
                if (segment != 0) remaining /= metadata.output_segment_sizes[segment];
                base += coordinate * metadata.output_segment_strides[segment];
            }
            return base;
        }

        __device__ __forceinline__ std::int64_t reduce_input_offset(const ReduceSumNdMetadata& metadata, std::int64_t input_base, std::int64_t reduction_index) {
            std::int64_t remaining = reduction_index;
            std::int64_t input_index = input_base;
            for (int segment = metadata.reduction_segment_count - 1; segment >= 0; --segment) {
                const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.reduction_segment_sizes[segment];
                if (segment != 0) remaining /= metadata.reduction_segment_sizes[segment];
                input_index += coordinate * metadata.reduction_segment_strides[segment];
            }
            return input_index;
        }

        // raw bytes: a __shared__ variable cannot have a type with default member initializers
        template <typename T, int BlockSize>
        struct BlockReduceStorage {
            alignas(T) unsigned char bytes[BlockSize * sizeof(T)];
        };

        // shared-memory tree reduction over a block of BlockSize threads; every thread gets the result
        template <int BlockSize, typename T, typename Op>
        __device__ __forceinline__ T block_reduce(BlockReduceStorage<T, BlockSize>& storage, T value, Op op) {
            static_assert((BlockSize & (BlockSize - 1)) == 0, "BlockSize must be a power of two");
            T* shared = reinterpret_cast<T*>(storage.bytes);
            shared[threadIdx.x] = value;
            __syncthreads();
            for (int stride = BlockSize / 2; stride > 0; stride /= 2) {
                if (threadIdx.x < stride)
                    shared[threadIdx.x] = op(shared[threadIdx.x], shared[threadIdx.x + stride]);
                __syncthreads();
            }
            const T result = shared[0];
            __syncthreads();  // the storage can be reused right after
            return result;
        }

        // integers round half to even and saturate once, as cv::reduce's convertTo() does
        template <typename T, typename TAccum>
        __device__ __forceinline__ T CastReduceSumResult(TAccum value, std::true_type) {
            const double value_as_double = static_cast<double>(value);
            const double max_value = static_cast<double>(device::numeric_limits<T>::max());
            const double min_value = static_cast<double>(device::numeric_limits<T>::lowest());
            if (value_as_double >= max_value) return device::numeric_limits<T>::max();
            if (value_as_double <= min_value) return device::numeric_limits<T>::lowest();
            return static_cast<T>(rint(value_as_double));
        }

        template <typename T, typename TAccum>
        __device__ __forceinline__ T CastReduceSumResult(TAccum value, std::false_type) {
            return static_cast<T>(value);
        }

        template <typename T, typename TAccum>
        __device__ __forceinline__ T CastReduceSumResult(TAccum value) {
            return CastReduceSumResult<T>(value, std::is_integral<T>{});
        }

        template <typename T, typename TAccum>
        __device__ __forceinline__ T FinalizeReduceSum(TAccum sum, std::int64_t count, bool mean, std::true_type) {
            return CastReduceSumResult<T>(mean ? sum / static_cast<TAccum>(count) : sum);
        }

        template <typename T, typename TAccum>
        __device__ __forceinline__ T FinalizeReduceSum(TAccum sum, std::int64_t count, bool mean, std::false_type) {
            const TAccum inv_norm = mean ? static_cast<TAccum>(1.0 / static_cast<double>(count)) : TAccum(1);
            return static_cast<T>(sum * inv_norm);
        }

        template <typename T, int BlockSize>
        __global__ void reduce_sum_nd_kernel(const T* input, T* output, ReduceSumNdMetadata metadata, bool mean) {
            using TAccum = std::conditional_t<std::is_integral<T>::value, double, typename accumulation_type<T>::type>;
            __shared__ BlockReduceStorage<SumState<TAccum>, BlockSize> reduce_storage;
            __shared__ std::int64_t input_base;

            for (std::int64_t output_index = blockIdx.x;
                 output_index < metadata.output_count;
                 output_index += gridDim.x) {
                if (threadIdx.x == 0)
                    input_base = reduce_output_base(metadata, output_index);
                __syncthreads();

                SumState<TAccum> thread_sum{};
                for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                     reduction_index += BlockSize)
                    thread_sum.Add(static_cast<TAccum>(input[reduce_input_offset(metadata, input_base, reduction_index)]));

                const SumState<TAccum> block_sum = block_reduce<BlockSize>(reduce_storage, thread_sum, MergeSumState<TAccum>{});
                if (threadIdx.x == 0)
                    output[output_index] = FinalizeReduceSum<T>(block_sum.Result(), metadata.reduction_count, mean, std::is_integral<T>{});
                __syncthreads();
            }
        }

        template <class T>
        static void reduce_nd(const Stream& stream, Span<T> output, View<T> input,
                              const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes, bool mean) {
            CV_Assert(dims.size() <= kMaxReduceRank);

            ReduceSumNdMetadata metadata;
            build_reduce_metadata(dims, axes, metadata);
            const std::int64_t output_count = metadata.output_count;
            const std::int64_t reduction_count = metadata.reduction_count;

            if (output_count == 0 || reduction_count == 0)
                return;

            constexpr int block_size = 256;
            constexpr int max_blocks = 65535;
            const int grid_size = static_cast<int>(std::min<std::int64_t>(max_blocks, output_count));
            reduce_sum_nd_kernel<T, block_size><<<grid_size, block_size, 0, stream.get()>>>(
                input.data().get(), output.data().get(), metadata, mean);
        }
    }

    template <class T>
    void reduce_sum(const Stream& stream, Span<T> output, View<T> input,
                    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_nd<T>(stream, output, input, dims, axes, false);
    }

    template <class T>
    void reduce_mean(const Stream& stream, Span<T> output, View<T> input,
                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_nd<T>(stream, output, input, dims, axes, true);
    }

    namespace detail {
        // max/min accumulate no rounding error, so they keep T and track a running best value per thread.
        template <typename T, bool IsMax>
        struct MinMaxState {
            T value{};
            bool has_value{false};

            __device__ __forceinline__ void Add(T v) {
                if (!has_value || (IsMax ? (v > value) : (v < value))) {
                    value = v;
                    has_value = true;
                }
            }
            __device__ __forceinline__ T Result() const { return value; }
        };

        template <typename T, bool IsMax>
        struct MergeMinMaxState {
            __device__ __forceinline__ MinMaxState<T, IsMax> operator()(MinMaxState<T, IsMax> lhs, const MinMaxState<T, IsMax>& rhs) const {
                if (rhs.has_value) lhs.Add(rhs.value);
                return lhs;
            }
        };

        template <typename T, bool IsMax, int BlockSize>
        __global__ void reduce_minmax_nd_kernel(const T* input, T* output, ReduceSumNdMetadata metadata) {
            using State = MinMaxState<T, IsMax>;
            __shared__ BlockReduceStorage<State, BlockSize> reduce_storage;
            __shared__ std::int64_t input_base;

            for (std::int64_t output_index = blockIdx.x;
                 output_index < metadata.output_count;
                 output_index += gridDim.x) {
                if (threadIdx.x == 0)
                    input_base = reduce_output_base(metadata, output_index);
                __syncthreads();

                State thread_state{};
                for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                     reduction_index += BlockSize)
                    thread_state.Add(input[reduce_input_offset(metadata, input_base, reduction_index)]);

                const State block_state = block_reduce<BlockSize>(reduce_storage, thread_state, MergeMinMaxState<T, IsMax>{});
                if (threadIdx.x == 0) {
                    output[output_index] = block_state.Result();
                }
                __syncthreads();
            }
        }

        template <class T, bool IsMax>
        static void reduce_minmax_nd(const Stream& stream, Span<T> output, View<T> input,
                                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
            CV_Assert(dims.size() <= kMaxReduceRank);

            ReduceSumNdMetadata metadata;
            build_reduce_metadata(dims, axes, metadata);
            const std::int64_t output_count = metadata.output_count;
            const std::int64_t reduction_count = metadata.reduction_count;

            if (output_count == 0 || reduction_count == 0)
                return;

            constexpr int block_size = 256;
            constexpr int max_blocks = 65535;
            const int grid_size = static_cast<int>(std::min<std::int64_t>(max_blocks, output_count));
            reduce_minmax_nd_kernel<T, IsMax, block_size><<<grid_size, block_size, 0, stream.get()>>>(
                input.data().get(), output.data().get(), metadata);
        }
    }

    template <class T>
    void reduce_max(const Stream& stream, Span<T> output, View<T> input,
                    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_minmax_nd<T, true>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_min(const Stream& stream, Span<T> output, View<T> input,
                    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_minmax_nd<T, false>(stream, output, input, dims, axes);
    }

    namespace detail {
        // PROD/L1/L2/SUM_SQUARE/LOG_SUM/LOG_SUM_EXP differ only in transform, combine and finalisation,
        // so they share one kernel templated on the op.
        enum class GenericReduceOp { PROD, L1, L2, SUM_SQUARE, LOG_SUM, LOG_SUM_EXP };

        template <typename TAccum>
        struct ProdCombine {
            __device__ __forceinline__ TAccum operator()(const TAccum& lhs, const TAccum& rhs) const { return lhs * rhs; }
        };

        template <typename TAccum>
        struct SumCombine {
            __device__ __forceinline__ TAccum operator()(const TAccum& lhs, const TAccum& rhs) const { return lhs + rhs; }
        };

        template <typename TAccum>
        struct MaxCombine {
            __device__ __forceinline__ TAccum operator()(const TAccum& lhs, const TAccum& rhs) const { return lhs > rhs ? lhs : rhs; }
        };

        template <GenericReduceOp Op>
        struct GenericReduceStep;

        template <>
        struct GenericReduceStep<GenericReduceOp::PROD> {
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, T, TAccum value, const TAccum&) { acc *= value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&) { return acc; }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::L1> {
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, T raw, TAccum value, const TAccum&) {
                AccumulateAbs(acc, raw, value, std::is_integral<TAccum>{});
            }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&) { return acc; }

        private:
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void AccumulateAbs(TAccum& acc, T raw, TAccum, std::true_type) { acc += static_cast<TAccum>(raw > 0 ? raw : -raw); }
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void AccumulateAbs(TAccum& acc, T, TAccum value, std::false_type) { acc += fabs(value); }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::L2> {
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, T, TAccum value, const TAccum&) { acc += value * value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&) { return sqrt(acc); }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::SUM_SQUARE> {
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, T, TAccum value, const TAccum&) { acc += value * value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&) { return acc; }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::LOG_SUM> {
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, T, TAccum value, const TAccum&) { acc += value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&) { return log(acc); }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::LOG_SUM_EXP> {
            template <typename T, typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, T, TAccum value, const TAccum& shared_max) { acc += exp(value - shared_max); }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum& shared_max) { return log(acc) + shared_max; }
        };

        template <typename T, GenericReduceOp Op, int BlockSize>
        __global__ void reduce_generic_nd_kernel(const T* input, T* output, ReduceSumNdMetadata metadata) {
            // integers accumulate in double, as cv::reduce does; a float product is accumulated in double
            // and rounded once, so the tree order adds no rounding error
            constexpr bool kDoubleAccum = std::is_integral<T>::value ||
                (Op == GenericReduceOp::PROD && std::is_same<T, float>::value);
            using TAccum = std::conditional_t<kDoubleAccum, double, typename accumulation_type<T>::type>;
            using Step = GenericReduceStep<Op>;
            __shared__ BlockReduceStorage<TAccum, BlockSize> reduce_storage;
            __shared__ std::int64_t input_base;
            __shared__ TAccum shared_max;

            for (std::int64_t output_index = blockIdx.x;
                 output_index < metadata.output_count;
                 output_index += gridDim.x) {
                if (threadIdx.x == 0)
                    input_base = reduce_output_base(metadata, output_index);
                __syncthreads();

                // exp(x) overflows well before the reduction is done, so LOG_SUM_EXP
                // walks the reduction twice: once for max(x), then for sum(exp(x - max)).
                if (Op == GenericReduceOp::LOG_SUM_EXP) {
                    TAccum thread_max = static_cast<TAccum>(-INFINITY);
                    for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                         reduction_index += BlockSize) {
                        const TAccum value = static_cast<TAccum>(input[reduce_input_offset(metadata, input_base, reduction_index)]);
                        thread_max = value > thread_max ? value : thread_max;
                    }
                    const TAccum block_max = block_reduce<BlockSize>(reduce_storage, thread_max, MaxCombine<TAccum>{});
                    if (threadIdx.x == 0) shared_max = block_max;
                    __syncthreads();  // publishes shared_max and frees reduce_storage for the sum below
                }

                TAccum thread_acc = (Op == GenericReduceOp::PROD) ? TAccum(1) : TAccum(0);
                for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                     reduction_index += BlockSize) {
                    const T raw = input[reduce_input_offset(metadata, input_base, reduction_index)];
                    const TAccum value = static_cast<TAccum>(raw);
                    Step::Accumulate(thread_acc, raw, value, shared_max);
                }

                TAccum block_acc;
                if (Op == GenericReduceOp::PROD)
                    block_acc = block_reduce<BlockSize>(reduce_storage, thread_acc, ProdCombine<TAccum>{});
                else
                    block_acc = block_reduce<BlockSize>(reduce_storage, thread_acc, SumCombine<TAccum>{});

                if (threadIdx.x == 0)
                    output[output_index] = CastReduceSumResult<T>(Step::Finalize(block_acc, shared_max));
                __syncthreads();
            }
        }

        template <class T, GenericReduceOp Op>
        static void reduce_generic_nd(const Stream& stream, Span<T> output, View<T> input,
                                      const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
            CV_Assert(dims.size() <= kMaxReduceRank);

            ReduceSumNdMetadata metadata;
            build_reduce_metadata(dims, axes, metadata);

            if (metadata.output_count == 0 || metadata.reduction_count == 0)
                return;

            constexpr int block_size = 256;
            constexpr int max_blocks = 65535;
            const int grid_size = static_cast<int>(std::min<std::int64_t>(max_blocks, metadata.output_count));
            reduce_generic_nd_kernel<T, Op, block_size><<<grid_size, block_size, 0, stream.get()>>>(
                input.data().get(), output.data().get(), metadata);
        }
    }

    template <class T>
    void reduce_prod(const Stream& stream, Span<T> output, View<T> input,
                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::PROD>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_l1(const Stream& stream, Span<T> output, View<T> input,
                   const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::L1>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_l2(const Stream& stream, Span<T> output, View<T> input,
                   const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::L2>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_sum_square(const Stream& stream, Span<T> output, View<T> input,
                           const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::SUM_SQUARE>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_log_sum(const Stream& stream, Span<T> output, View<T> input,
                        const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::LOG_SUM>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_log_sum_exp(const Stream& stream, Span<T> output, View<T> input,
                            const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::LOG_SUM_EXP>(stream, output, input, dims, axes);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void reduce_sum(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
#endif
    template void reduce_sum(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
