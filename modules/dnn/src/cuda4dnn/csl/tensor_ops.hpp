// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_TENSOR_OPS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_TENSOR_OPS_HPP

#include "stream.hpp"
#include "tensor.hpp"
#include "pointer.hpp"
#include "cublas.hpp"
#include "cudnn.hpp"
#include "workspace.hpp"

#include "cudnn/convolution.hpp"
#include "cudnn/pooling.hpp"
#include "cudnn/lrn.hpp"
#include "cudnn/softmax.hpp"
#include "cudnn/transform.hpp"
#include "cudnn/transpose_convolution.hpp"
#include "cudnn/recurrent.hpp"

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
         * Exception Guarantee: Basic
         */
        template <class T> inline
        void copy(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            if (dest.get() != src.get())
                memcpy(dest.get(), src.get(), dest.size(), stream);
        }

        namespace detail {
            template <class T>
            void assertGEMMCompatiblity(const TensorSpan<T>& result, bool transa, const TensorView<T>& A, bool transb, const TensorView<T>& B) {
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
            }
        }

        /** @brief performs generalized matrix-multiplication
         *
         * Pre-conditions:
         * - \p A and \p B must meet the mathematical requirements for matrix multiplication
         * - \p result must be large enough to hold the result
         *
         * Exception Guarantee: Basic
         */
        template <class T> inline
        void gemm(const cublas::Handle& handle, T beta, TensorSpan<T> result, T alpha, bool transa, TensorView<T> A, bool transb, TensorView<T> B) {
            /* matrix operations can be performed only on tensors with rank two or below */
            CV_Assert(get_effective_rank(A) <= 2);
            CV_Assert(get_effective_rank(B) <= 2);
            CV_Assert(get_effective_rank(result) <= 2);

            const auto result_nr = result.get_axis_size(-2);
            const auto result_nc = result.get_axis_size(-1);
            const auto common_dim = A.get_axis_size(transa ? -2 : -1);
            const auto A_nc = A.get_axis_size(-1);
            const auto B_nc = B.get_axis_size(-1);

            detail::assertGEMMCompatiblity(result, transa, A, transb, B);

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

        /** @brief performs generalized matrix-multiplication for a strided batch of matrices
         *
         * Pre-conditions:
         * - A, B and C must be rank three tensors with dimensions (batch, rows, cols)
         * - the last two axes of \p A and \p B must meet the mathematical requirements for matrix multiplication
         * - \p result must be large enough to hold the result and the matrices must not overlap in memory
         * - batch dimension should be same in \p A, \p B and \p result
         *
         * Exception Guarantee: Basic
         */
        template <class T> inline
        void gemmStridedBatched(const cublas::Handle& handle, T beta, TensorSpan<T> result, T alpha, bool transa, TensorView<T> A, bool transb, TensorView<T> B) {
            CV_Assert(A.rank() == 3);
            CV_Assert(B.rank() == 3);
            CV_Assert(result.rank() == 3);

            const auto batch_size = result.get_axis_size(0);
            CV_Assert(batch_size == A.get_axis_size(0));
            CV_Assert(batch_size == B.get_axis_size(0));

            detail::assertGEMMCompatiblity(result, transa, A, transb, B);

            const auto result_nr = result.get_axis_size(-2);
            const auto result_nc = result.get_axis_size(-1);
            const auto common_dim = A.get_axis_size(transa ? -2 : -1);
            const auto A_nc = A.get_axis_size(-1);
            const auto B_nc = B.get_axis_size(-1);

            std::size_t strideA = (A.size() / batch_size),
                        strideB = (B.size() / batch_size),
                        strideC = (result.size() / batch_size);

            cublas::gemmStridedBatched<T>(handle,
                transb, transa,
                result_nc, result_nr, common_dim,
                alpha, B.get(), B_nc, strideB,
                A.get(), A_nc, strideA,
                beta, result.get(), result_nc, strideC,
                batch_size);
        }

        /** @brief performs element-wise addition with broadcasting
         *
         * Pre-conditions:
         * - \p A and \p result must be compatible tensors
         *
         * Exception Guarantee: Basic
         */
        template <class T> inline
        void softmax(const cudnn::Handle& handle, TensorSpan<T> output, TensorView<T> input, int channel_axis, bool log) {
            CV_Assert(is_shape_same(output, input));

            channel_axis = clamp_axis(channel_axis, input.rank());

            std::size_t outer_size = input.size_range(0, channel_axis);
            auto channel_size = input.get_axis_size(channel_axis);
            std::size_t inner_size = input.size_range(channel_axis + 1, input.rank());

            std::array<std::size_t, 4> shape = { outer_size, channel_size, 1, inner_size };

            using cudnn::TensorDescriptor;
            auto inputDesc = TensorDescriptor<T>(shape);
            auto outputDesc = TensorDescriptor<T>(shape);
            cudnn::softmax(handle, outputDesc, output.get(), inputDesc, input.get(), log);
        }
    }

    template <class T>
    class Convolution {
        using TensorDescriptor = cudnn::TensorDescriptor<T>;
        using FilterDescriptor = cudnn::FilterDescriptor<T>;
        using ConvolutionDescriptor = cudnn::ConvolutionDescriptor<T>;
        using ConvolutionAlgorithm = cudnn::ConvolutionAlgorithm<T>;
        using ActivationDescriptor = cudnn::ActivationDescriptor;

    public:
        using ActivationType = ActivationDescriptor::ActivationType;

        struct params_type {
            /* convolution */
            std::vector<std::size_t> input_shape;
            std::vector<std::size_t> filter_shape;
            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;
            std::vector<std::size_t> dilation;
            std::size_t groups;

            /* bias and activation (only RELU supported) */
            std::vector<std::size_t> bias_shape;
            ActivationType activation_type; /* MUST BE identity if there is no bias and ReLU if there is bias */
            bool eltwise;
        };

        Convolution() = default;
        Convolution(const Convolution&) = delete;
        Convolution(Convolution&&) = default;
        Convolution(cudnn::Handle handle, const params_type& params) {
            cudnnHandle = std::move(handle);

            inputTensorDesc = TensorDescriptor(params.input_shape);
            filterDesc = FilterDescriptor(params.filter_shape);
            convDesc = ConvolutionDescriptor(params.padding, params.stride, params.dilation, params.groups);

            std::vector<int> output_dims;
            getConvolutionForwardOutputDim(convDesc, filterDesc, inputTensorDesc, output_dims);
            outputTensorDesc = TensorDescriptor(output_dims);

            algo = ConvolutionAlgorithm(cudnnHandle, convDesc, filterDesc, inputTensorDesc, outputTensorDesc);

            if (!params.bias_shape.empty()) {
                CV_Assert(params.activation_type == ActivationType::RELU);
                biasTensorDesc = TensorDescriptor(params.bias_shape);
                if (params.eltwise)
                    eltwiseTensorDesc = TensorDescriptor(output_dims);
                activationDesc = ActivationDescriptor(params.activation_type, 0.0);
            } else {
                CV_Assert(params.activation_type == ActivationType::IDENTITY);
            }
        }

        Convolution& operator=(const Convolution&) = delete;
        Convolution& operator=(Convolution&&) = default;

        std::size_t get_workspace_size() const noexcept {
            return algo.get_workspace_size();
        }

        void convolve(TensorSpan<T> output, TensorView<T> input, TensorView<T> filters, WorkspaceInstance scratchpad) {
            cudnn::convolve<T>(
                cudnnHandle,
                convDesc, algo, scratchpad,
                filterDesc, filters.get(),
                inputTensorDesc, input.get(),
                1.0, 0.0, outputTensorDesc, output.get()
            );
        }

        void convolve_with_bias_activation(TensorSpan<T> output, TensorView<T> input, TensorView<T> filters, TensorView<T> bias, WorkspaceInstance scratchpad) {
            cudnn::convolve_with_bias_activation<T>(
                cudnnHandle,
                1.0, convDesc, algo, scratchpad,
                filterDesc, filters.get(),
                inputTensorDesc, input.get(),
                biasTensorDesc, bias.get(),
                activationDesc,
                outputTensorDesc, output.get()
            );
        }

        void convolve_with_bias_eltwise_activation(TensorSpan<T> output, TensorView<T> input, TensorView<T> filters, TensorView<T> bias, TensorView<T> eltwise, WorkspaceInstance scratchpad) {
            cudnn::convolve_with_bias_eltwise_activation<T>(
                cudnnHandle,
                1.0, convDesc, algo, scratchpad,
                filterDesc, filters.get(),
                inputTensorDesc, input.get(),
                biasTensorDesc, bias.get(),
                1.0, eltwiseTensorDesc, eltwise.get(),
                activationDesc,
                outputTensorDesc, output.get()
            );
        }

    private:
        cudnn::Handle cudnnHandle;
        TensorDescriptor inputTensorDesc, outputTensorDesc;
        FilterDescriptor filterDesc;
        ConvolutionDescriptor convDesc;
        ConvolutionAlgorithm algo;
        TensorDescriptor biasTensorDesc;
        TensorDescriptor eltwiseTensorDesc;
        ActivationDescriptor activationDesc;
    };

    template <class T>
    class TransposeConvolution {
        using TensorDescriptor = cudnn::TensorDescriptor<T>;
        using FilterDescriptor = cudnn::FilterDescriptor<T>;
        using ConvolutionDescriptor = cudnn::ConvolutionDescriptor<T>;
        using TransposeConvolutionAlgorithm = cudnn::TransposeConvolutionAlgorithm<T>;

    public:
        struct params_type {
            std::vector<std::size_t> input_shape;
            std::vector<std::size_t> output_shape;

            std::vector<std::size_t> filter_shape;

            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;
            std::vector<std::size_t> dilation;

            std::size_t groups;
        };

        TransposeConvolution() = default;
        TransposeConvolution(const TransposeConvolution&) = delete;
        TransposeConvolution(TransposeConvolution&&) = default;
        TransposeConvolution(cudnn::Handle handle, const params_type& params) {
            cudnnHandle = std::move(handle);

            filterDesc = FilterDescriptor(params.filter_shape);
            convDesc = ConvolutionDescriptor(params.padding, params.stride, params.dilation, params.groups);

            /* input_shape is the output shape for convolution
             * output_shape is the input shape for convolution
             */
            convInputTensorDesc = TensorDescriptor(params.output_shape);

            std::vector<int> conv_output_dims;
            getConvolutionForwardOutputDim(convDesc, filterDesc, convInputTensorDesc, conv_output_dims);

            /* the convolution output must be identical to what cuDNN expects */
            CV_Assert(std::equal(std::begin(conv_output_dims), std::end(conv_output_dims), std::begin(params.input_shape)));

            convOutputTensorDesc = TensorDescriptor(params.input_shape);

            algo = TransposeConvolutionAlgorithm(cudnnHandle, convDesc, filterDesc, convOutputTensorDesc, convInputTensorDesc);
        }

        TransposeConvolution& operator=(const TransposeConvolution&) = delete;
        TransposeConvolution& operator=(TransposeConvolution&&) = default;

        std::size_t get_workspace_size() const noexcept {
            return algo.get_workspace_size();
        }

        void transpose_convolve(TensorSpan<T> output, TensorView<T> input, TensorView<T> filters, WorkspaceInstance scratchpad) {
            cudnn::transpose_convolve<T>(
                cudnnHandle,
                convDesc, algo, scratchpad,
                filterDesc, filters.get(),
                convOutputTensorDesc, input.get(),
                1.0, 0.0, convInputTensorDesc, output.get()
            );
        }

    private:
        cudnn::Handle cudnnHandle;
        TensorDescriptor convInputTensorDesc, convOutputTensorDesc;
        FilterDescriptor filterDesc;
        ConvolutionDescriptor convDesc;
        TransposeConvolutionAlgorithm algo;
    };

    template <class T>
    class Pooling {
        using TensorDescriptor = cudnn::TensorDescriptor<T>;
        using PoolingDescriptor = cudnn::PoolingDescriptor;

    public:
        using PoolingType = PoolingDescriptor::PoolingType;

        struct params_type {
            std::vector<std::size_t> input_shape;
            std::vector<std::size_t> output_shape;

            std::vector<std::size_t> window_size;
            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;

            PoolingType type;
        };

        Pooling() = default;
        Pooling(const Pooling&) = delete;
        Pooling(Pooling&&) = default;
        Pooling(cudnn::Handle handle, const params_type& params) {
            cudnnHandle = std::move(handle);

            inputTensorDesc = TensorDescriptor(params.input_shape);
            poolingDesc = PoolingDescriptor(params.window_size, params.padding, params.stride, params.type);

            //std::vector<int> output_dim;
            //getPoolingForwardOutputDim(poolingDesc, inputTensorDesc, output_dim);
            outputTensorDesc = TensorDescriptor(params.output_shape);
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
        using LRNType = LRNDescriptor::LRNType;

        LRN() = default;
        LRN(const LRN&) = delete;
        LRN(LRN&&) = default;
        LRN(cudnn::Handle handle, std::size_t local_size, T alpha, T beta, T k, LRNType type) {
            cudnnHandle = std::move(handle);
            lrnDesc = LRNDescriptor(local_size, alpha, beta, k, type);
        }

        LRN& operator=(const LRN&) = delete;
        LRN& operator=(LRN&&) = default;

        void normalize(TensorView<T> input, TensorSpan<T> output, WorkspaceInstance workspace) {
            cudnn::LRNForward<T>(
                cudnnHandle,
                lrnDesc,
                TensorDescriptor(input.shape_as_vector()), input.get(),
                1.0, 0.0, TensorDescriptor(output.shape_as_vector()), output.get(),
                workspace
            );
        }

    private:
        cudnn::Handle cudnnHandle;
        LRNDescriptor lrnDesc;
    };

    template <class T>
    class TensorTransform {
        using TensorTransformDescriptor = cudnn::TensorTransformDescriptor;
        using TensorDescriptor = cudnn::TensorDescriptor<T>;

    public:
        TensorTransform() = default;
        TensorTransform(const TensorTransform&) = delete;
        TensorTransform(TensorTransform&&) = default;

        template <class SequenceContainer>
        TensorTransform(cudnn::Handle handle, const SequenceContainer& paddingLeft, const SequenceContainer& paddingRight) {
            cudnnHandle = std::move(handle);
            transDesc = TensorTransformDescriptor(paddingLeft, paddingRight);
        }

        TensorTransform& operator=(const TensorTransform&) = delete;
        TensorTransform& operator=(TensorTransform&&) = default;

        void transform(TensorView<T> input, TensorSpan<T> output) {
            cudnn::transform<T>(
                cudnnHandle,
                transDesc,
                TensorDescriptor(input.shape_as_vector()), input.get(),
                TensorDescriptor(output.shape_as_vector()), output.get()
            );
        }

    private:
        cudnn::Handle cudnnHandle;
        TensorTransformDescriptor transDesc;
    };

    template<class T>
    class LSTM
    {
        using TensorDescriptor = cudnn::TensorDescriptor<T>;
        using DropoutDescriptor = cudnn::DropoutDescriptor;
        using RNNDescriptor = cudnn::RNNDescriptor<T>;
        using FilterDescriptor = cudnn::FilterDescriptor<T>;
        using TensorDescriptorsArray = cudnn::TensorDescriptorsArray<T>;

    public:
        using RNNMode = typename RNNDescriptor::RNNMode;

        struct params_type
        {
            std::vector<std::size_t> weights_shape;

            int seqLength;
            int numLayers;
            int hiddenSize;
            int inputSize;
            int miniBatch;
            bool bidirectional;

            float dropout;
            RNNMode type;
        };

        LSTM() = default;
        LSTM(const LSTM&) = delete;
        LSTM(LSTM&&) = default;
        LSTM(cudnn::Handle handle, const params_type& params)
            : cudnnHandle(std::move(handle)), seqLength{params.seqLength},
              inputDesc(seqLength, {params.miniBatch, params.inputSize, 1}),
              outputDesc(seqLength,
                         {params.miniBatch,
                          params.bidirectional ? params.hiddenSize * 2 : params.hiddenSize,
                          1})
        {
            dropoutDesc = DropoutDescriptor(cudnnHandle, params.dropout);
            filterDesc = FilterDescriptor(params.weights_shape);
            rnnDesc = RNNDescriptor(cudnnHandle, params.type, params.hiddenSize,
                                    params.numLayers, params.bidirectional, dropoutDesc);

            int num_direction = params.bidirectional ? 2 : 1;
            h0TensorDesc = TensorDescriptor(
                    {num_direction, params.miniBatch, params.hiddenSize});
            c0TensorDesc = TensorDescriptor(
                    {num_direction, params.miniBatch, params.hiddenSize});

            // Get amount of work space required to execute the RNN described by rnnDesc
            // with input dimensions defined by inputDesc
            csl::WorkspaceBuilder builder;
            builder.require(cudnn::getRNNWorkspaceSize<T>(cudnnHandle, rnnDesc, seqLength, inputDesc));
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        LSTM& operator=(const LSTM&) = delete;
        LSTM& operator=(LSTM&&) = default;

        void inference(TensorView<T> input, TensorSpan<T> y_output, TensorSpan<T> yc_output, TensorView<T> filters,
                       TensorView<T> h0, TensorView<T> c0, WorkspaceInstance workspace)
        {
            cudnn::LSTMForward<T>(cudnnHandle, rnnDesc, filterDesc, filters.get(), inputDesc,
                                  input.get(), h0TensorDesc, h0.get(), c0TensorDesc, c0.get(),
                                  seqLength, outputDesc, y_output.get(), yc_output.get(), workspace);
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept { return scratch_mem_in_bytes; }

    private:
        cudnn::Handle cudnnHandle;
        std::size_t scratch_mem_in_bytes{0};
        int seqLength;

        RNNDescriptor rnnDesc;
        DropoutDescriptor dropoutDesc;

        FilterDescriptor filterDesc;
        TensorDescriptor h0TensorDesc, c0TensorDesc;

        TensorDescriptorsArray inputDesc;
        TensorDescriptorsArray outputDesc;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_TENSOR_OPS_HPP */
