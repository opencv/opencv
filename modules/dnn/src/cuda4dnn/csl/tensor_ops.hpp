// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_TENSOR_OPS_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_TENSOR_OPS_HPP

#include "stream.hpp"
#include "tensor.hpp"
#include "kernels.hpp"
#include "pointer.hpp"
#include "cublas.hpp"
#include "cudnn.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <array>
#include <vector>
#include <algorithm>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    namespace tensor_ops {

        /** @brief copies data between tensors
         *
         * Pre-conditions:
         * - \p dest and \p src must have the same shape
         *
         * Exception Gaurantee: Basic
         */
        template <class T> inline
            void copy(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            if (dest.get() != src.get())
                memcpy(dest.get(), src.get(), dest.size());
        }

        /** @brief performs generalized matrix-multiplication
         *
         * Pre-conditions:
         * - \p A and \p B must meet the mathematical requirements for matrix multiplication
         * - \p result must be large enough to hold the result
         *
         * Exception Gaurantee: Basic
         */
        template <class T> inline
        void gemm(const cublas::Handle& handle, T beta, TensorSpan<T> result, T alpha, bool transa, TensorView<T> A, bool transb, TensorView<T> B) {
            /* matrix operations can be performed only on rank two or less tensors */
            CV_Assert(get_effective_rank(A) <= 2 &&
                get_effective_rank(B) <= 2 &&
                get_effective_rank(result) <= 2);

            /* check dimension requirements for matrix multiplication */
            if (!transa && !transb) {
                CV_Assert(A.get_axis_size(-2) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-1) == B.get_axis_size(-2));
                CV_Assert(B.get_axis_size(-1) == result.get_axis_size(-1));
            } else if (!transa && transb) {
                CV_Assert(A.get_axis_size(-2) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-1) == B.get_axis_size(-1));
                CV_Assert(B.get_axis_size(-2) == result.get_axis_size(-1));
            } else if (transa && !transb) {
                CV_Assert(A.get_axis_size(-1) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-2) == B.get_axis_size(-2));
                CV_Assert(B.get_axis_size(-1) == result.get_axis_size(-1));
            } else {
                CV_Assert(A.get_axis_size(-1) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-2) == B.get_axis_size(-1));
                CV_Assert(B.get_axis_size(-2) == result.get_axis_size(-1));
            }

            const auto result_nr = result.get_axis_size(-2);
            const auto result_nc = result.get_axis_size(-1);
            const auto common_dim = A.get_axis_size(transa ? -2 : -1);
            const auto A_nc = A.get_axis_size(-1);
            const auto B_nc = B.get_axis_size(-1);

            /* tensors are stored in row-major but cublas::gemm operates on column-major matrices
             * a row-major matrix when read as column-major matrix gives the transpose of the intended matrix
             *
             * Required: C = AB
             * what cuBLAS sees: C^T = A^TB^T = (BA)^T
             *
             * By reversing operands, we effectively perform:
             * C^T = B^TA^T = (AB)^T
             *
             * which gives C = AB
             */
            cublas::gemm<T>(handle,
                transb, transa,
                result_nc, result_nr, common_dim,
                alpha, B.get(), B_nc,
                A.get(), A_nc,
                beta, result.get(), result_nc);
        }

        /** @brief performs element-wise addition with broadcasting
        *
        * Pre-conditions:
        * - \p A and \p C must be compatible tensors
        *
        * Exception Gaurantee: Basic
        */
        template <class T> inline
        void add(const cudnn::Handle& handle, T beta, TensorSpan<T> C, T alpha, TensorView<T> A) {
            CV_Assert(is_shape_compatible(A, C));

            using cudnn::TensorDescriptor;
            auto aDesc = TensorDescriptor<T>(A.shape());
            auto cDesc = TensorDescriptor<T>(C.shape());
            cudnn::add(handle, alpha, aDesc, A.get(), beta, cDesc, C.get());
        }

        /** @brief performs element-wise addition with broadcasting
        *
        * Pre-conditions:
        * - \p A and \p result must be compatible tensors
        *
        * Exception Gaurantee: Basic
        */
        template <class T> inline
        void softmax(const cudnn::Handle& handle, TensorSpan<T> output, TensorView<T> input, int channel_axis, bool log) {
            CV_Assert(is_shape_same(output, input));

            channel_axis = clamp_axis(channel_axis, input.rank);

            std::size_t outer_size = 1;
            for (int j = 0; j < channel_axis; j++)
                outer_size *= input.get_axis_size(j);

            auto channel_size = input.get_axis_size(channel_axis);

            std::size_t inner_size = 1;
            for (int j = channel_axis + 1; j < input.rank; j++)
                inner_size *= input.get_axis_size(j);

            std::array<std::size_t, 4> shape = { outer_size, channel_size, 1 , inner_size };

            using cudnn::TensorDescriptor;
            auto inputDesc = TensorDescriptor<T>(shape);
            auto outputDesc = TensorDescriptor<T>(shape);
            cudnn::softmax(handle, outputDesc, output.get(), inputDesc, input.get(), log);
        }

        template <class T> inline
        void abs(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::abs<T>(stream, dest, src);
        }

        template <class T> inline
        void bnll(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::bnll<T>(stream, dest, src);
        }

        template <class T> inline
        void relu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, T slope = 0) {
            CV_Assert(is_shape_same(dest, src));
            kernels::relu<T>(stream, dest, src, slope);
        }

        template <class T> inline
        void clipped_relu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, T min, T max) {
            CV_Assert(is_shape_same(dest, src));
            kernels::clipped_relu<T>(stream, dest, src, min, max);
        }

        template <class T> inline
        void channelwise_relu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, TensorView<T> slope) {
            CV_Assert(is_shape_same(dest, src));
            CV_Assert(src.get_axis_size(1) == slope.size());
            std::size_t inner_size = src.size() / src.get_axis_size(0);
            std::size_t channel_size = inner_size / src.get_axis_size(1);
            kernels::axiswise_relu<T>(stream, dest, src, slope, inner_size, channel_size);
        }

        template <class T> inline
        void elu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::elu<T>(stream, dest, src);
        }

        template <class T> inline
        void power(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, T exp = 1, T scale = 1, T shift = 0) {
            CV_Assert(is_shape_same(dest, src));
            kernels::power<T>(stream, dest, src, exp, scale, shift);
        }

        template <class T> inline
        void sigmoid(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::sigmoid<T>(stream, dest, src);
        }

        template <class T> inline
        void tanh(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::tanh<T>(stream, dest, src);
        }
    }

    template <class T>
    class Convolution {
        using TensorDescriptor = cudnn::TensorDescriptor<T>;
        using FilterDescriptor = cudnn::FilterDescriptor<T>;
        using ConvolutionDescriptor = cudnn::ConvolutionDescriptor<T>;
        using ConvolutionAlgorithm = cudnn::ConvolutionAlgorithm<T>;

    public:
        struct params_type {
            std::vector<std::size_t> input_shape;
            std::vector<std::size_t> filter_shape;

            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;
            std::vector<std::size_t> dialation;

            std::size_t groups;
        };

        Convolution() = default;
        Convolution(const Convolution&) = delete;
        Convolution(Convolution&&) = default;
        Convolution(cudnn::Handle handle, const params_type& params) {
            cudnnHandle = std::move(handle);

            inputTensorDesc = TensorDescriptor(params.input_shape);
            filterDesc = FilterDescriptor(params.filter_shape);
            convDesc = ConvolutionDescriptor(params.padding, params.stride, params.dialation, params.groups);

            std::vector<int> output_dims;
            getConvolutionForwardOutputDim(convDesc, filterDesc, inputTensorDesc, output_dims);
            outputTensorDesc = TensorDescriptor(output_dims);

            algo = ConvolutionAlgorithm(cudnnHandle, convDesc, filterDesc, inputTensorDesc, outputTensorDesc);
        }

        Convolution& operator=(const Convolution&) = delete;
        Convolution& operator=(Convolution&&) = default;

        std::size_t get_workspace_size() const noexcept {
            return algo.get_workspace_size();
        }

        void convolve(TensorSpan<T> output, TensorView<T> input, TensorView<T> filters, const Workspace& scratchpad) {
            cudnn::convolve<T>(
                cudnnHandle,
                convDesc, algo, scratchpad,
                filterDesc, filters.get(),
                inputTensorDesc, input.get(),
                1.0, 0.0, outputTensorDesc, output.get()
            );
        }

    private:
        cudnn::Handle cudnnHandle;
        TensorDescriptor inputTensorDesc, outputTensorDesc;
        FilterDescriptor filterDesc;
        ConvolutionDescriptor convDesc;
        ConvolutionAlgorithm algo;
    };

    template <class T>
    class Pooling {
        using TensorDescriptor = cudnn::TensorDescriptor<T>;
        using PoolingDescriptor = cudnn::PoolingDescriptor;

    public:
        enum class rounding_type {
            FLOOR,
            CEILING
        };

        using pooling_type = PoolingDescriptor::pooling_type;

        struct params_type {
            std::vector<std::size_t> input_shape;

            std::vector<std::size_t> window_size;
            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;

            rounding_type rounding_mode;
            pooling_type type;
        };

        Pooling() = default;
        Pooling(const Pooling&) = delete;
        Pooling(Pooling&&) = default;
        Pooling(cudnn::Handle handle, const params_type& params) {
            cudnnHandle = std::move(handle);

            inputTensorDesc = TensorDescriptor(params.input_shape);
            poolingDesc = PoolingDescriptor(params.window_size, params.padding, params.stride, params.type);

            const auto& input_shape = params.input_shape;
            std::vector<std::size_t> output_shape;
            output_shape.assign(std::begin(input_shape), std::end(input_shape));

            const auto& window_size = params.window_size;
            const auto& padding = params.padding;
            const auto& stride = params.stride;

            bool ceil = params.rounding_mode == rounding_type::CEILING;
            for (int i = 0; i < window_size.size(); i++) {
                double axis_sz = (input_shape[i + 2] + 2 * padding[i] - window_size[i]) / double(stride[i]) + 1;
                output_shape[i + 2] = ceil ? std::ceil(axis_sz) : std::floor(axis_sz);

                /* check if the last pooling window starts in the valid region */
                if (padding[i]) {
                    if ((output_shape[i + 2] - 1) * stride[i] >= input_shape[i + 2] + padding[i])
                        output_shape[i + 2]--;
                }
            }

            if (!ceil)
            {
                /* we must agree with cuDNN if we used floor */
                std::vector<int> output_dim;
                getPoolingForwardOutputDim(poolingDesc, inputTensorDesc, output_dim);
                CV_Assert(std::equal(std::begin(output_dim), std::end(output_dim), std::begin(output_shape)));
                CV_UNUSED(output_dim);
            }

            outputTensorDesc = TensorDescriptor(output_shape);
        }

        Pooling& operator=(const Pooling&) = delete;
        Pooling& operator=(Pooling&&) = default;

        void pool(TensorView<T> input, TensorSpan<T> output) {
            cudnn::pool<T>(
                cudnnHandle,
                poolingDesc,
                inputTensorDesc, input.get(),
                1.0, 0.0, outputTensorDesc, output.get()
            );
        }

    private:
        cudnn::Handle cudnnHandle;
        TensorDescriptor inputTensorDesc, outputTensorDesc;
        PoolingDescriptor poolingDesc;
    };

    template <class T>
    class LRN {
        using LRNDescriptor = cudnn::LRNDescriptor;
        using TensorDescriptor = cudnn::TensorDescriptor<T>;

    public:
        using lrn_type = LRNDescriptor::lrn_type;

        LRN() = default;
        LRN(const LRN&) = delete;
        LRN(LRN&&) = default;
        LRN(csl::cudnn::Handle handle, std::size_t local_size, double alpha, double beta, double k, lrn_type type) {
            cudnnHandle = std::move(handle);
            lrnDesc = LRNDescriptor(local_size, alpha, beta, k, type);
        }

        LRN& operator=(const LRN&) = delete;
        LRN& operator=(LRN&&) = default;

        void normalize(TensorView<T> input, TensorSpan<T> output) {
            cudnn::LRNForward<T>(
                cudnnHandle,
                lrnDesc,
                TensorDescriptor(input.shape()), input.get(),
                1.0, 0.0, TensorDescriptor(output.shape()), output.get()
            );
        }

    private:
        cudnn::Handle cudnnHandle;
        LRNDescriptor lrnDesc;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_TENSOR_OPS_HPP */
