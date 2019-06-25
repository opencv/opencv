// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_CONVOLUTION_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_CONVOLUTION_HPP

#include "cudnn.h"
#include "../pointer.hpp"
#include "../workspace.hpp"

#include <cudnn.h>

#include <cstddef>
#include <array>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <iterator>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    /** creates a cuDNN filter descriptor for the given filter shape
     *
     * Dimension Ordering:
     * 0: number of output feature maps
     * 1: number of input feature maps
     * 2..n: kernel dimensions
     */
    template <class T>
    class FilterDescriptor {
    public:
        FilterDescriptor() noexcept : descriptor{ nullptr } { }
        FilterDescriptor(const FilterDescriptor&) = delete;
        FilterDescriptor(FilterDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /** constructs a filter descriptor from the filter dimensions provided in \p shape */
        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        FilterDescriptor(const SequenceContainer& shape) {
            constructor(shape.begin(), shape.end());
        }

        /** constructs a filter descriptor from the filter dimensions provided in [begin, end) */
        template <class ForwardItr, typename = typename std::enable_if<!std::is_integral<ForwardItr>::value, void>::type> // TODO is_iterator
        FilterDescriptor(ForwardItr begin, ForwardItr end) {
            constructor(begin, end);
        }

        /** constructs a filter descriptor from the filter dimensions provided as arguments */
        template <class ...Sizes>
        FilterDescriptor(Sizes ...sizes) {
            static_assert(sizeof...(Sizes) >= 3, "filter descriptors must have at least three dimensions");
            static_assert(sizeof...(Sizes) <= CUDNN_DIM_MAX, "required rank exceeds maximum supported rank");
            std::array<int, sizeof...(Sizes)> dims = { static_cast<int>(sizes)... };
            constructor(std::begin(dims), std::end(dims));
        }

        ~FilterDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyFilterDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyFilterDescriptor(descriptor));
            }
        }

        FilterDescriptor& operator=(const FilterDescriptor&) = delete;
        FilterDescriptor& operator=(FilterDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnFilterDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class ForwardItr>
        void constructor(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) >= 3);
            CV_Assert(std::distance(start, end) <= CUDNN_DIM_MAX);

            CUDA4DNN_CHECK_CUDNN(cudnnCreateFilterDescriptor(&descriptor));
            try {
                const auto rank = std::distance(start, end);
                if (rank == 4) {
                    std::array<int, 4> dims;
                    std::copy(start, end, std::begin(dims));
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetFilter4dDescriptor(
                            descriptor,
                            detail::get_data_type<T>(), CUDNN_TENSOR_NCHW,
                            dims[0], dims[1], dims[2], dims[3]
                        )
                    );
                } else {
                    std::vector<int> dims(start, end);
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetFilterNdDescriptor(
                            descriptor,
                            detail::get_data_type<T>(), CUDNN_TENSOR_NCHW,
                            dims.size(), dims.data()
                        )
                    );
                }
            } catch (...) {
                /* cudnnDestroyFilterDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyFilterDescriptor(descriptor));
                throw;
            }
        }

        cudnnFilterDescriptor_t descriptor;
    };

    /** creates a cuDNN convolution descriptor */
    template <class T>
    class ConvolutionDescriptor {
    public:
        ConvolutionDescriptor() noexcept : descriptor{ nullptr } { }
        ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
        ConvolutionDescriptor(ConvolutionDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        ConvolutionDescriptor(
            const SequenceContainer& zero_padding,
            const SequenceContainer& stride,
            const SequenceContainer& dialation,
            std::size_t group_count)
        {
            constructor(zero_padding, stride, dialation, group_count);
        }

        ~ConvolutionDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyConvolutionDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(descriptor));
            }
        }

        ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;
        ConvolutionDescriptor& operator=(ConvolutionDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnConvolutionDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class SequenceContainer>
        void constructor(
            const SequenceContainer& zero_padding,
            const SequenceContainer& stride,
            const SequenceContainer& dialation,
            std::size_t group_count)
        {
            CV_Assert(std::size(zero_padding) == std::size(stride));
            CV_Assert(std::size(zero_padding) == std::size(dialation));

            CUDA4DNN_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&descriptor));
            try {
                const auto rank = std::size(zero_padding);
                if (rank == 2) {
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetConvolution2dDescriptor(
                            descriptor,
                            zero_padding[0], zero_padding[1],
                            stride[0], stride[1],
                            dialation[0], dialation[1],
                            CUDNN_CROSS_CORRELATION,
                            detail::get_data_type<T>()
                        )
                    );
                } else {
                    std::vector<int> ipadding(std::begin(zero_padding), std::end(zero_padding));
                    std::vector<int> istride(std::begin(stride), std::end(stride));
                    std::vector<int> idialation(std::begin(dialation), std::end(dialation));
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetConvolutionNdDescriptor(
                            descriptor,
                            rank, ipadding.data(), istride.data(), idialation.data(),
                            CUDNN_CROSS_CORRELATION,
                            detail::get_data_type<T>()
                        )
                    );
                }
                CUDA4DNN_CHECK_CUDNN(cudnnSetConvolutionGroupCount(descriptor, group_count));
            } catch (...) {
                /* cudnnDestroyConvolutionDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(descriptor));
                throw;
            }
        }

        cudnnConvolutionDescriptor_t descriptor;
    };

    template <class T>
    class ConvolutionAlgorithm {
    public:
        ConvolutionAlgorithm() noexcept : workspace_size{ 0 } { }
        ConvolutionAlgorithm(ConvolutionAlgorithm&) = default;
        ConvolutionAlgorithm(ConvolutionAlgorithm&&) = default;

        ConvolutionAlgorithm(
            const Handle& handle,
            const ConvolutionDescriptor<T>& conv,
            const FilterDescriptor<T>& filter,
            const TensorDescriptor<T>& input,
            const TensorDescriptor<T>& output)
        {
            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionForwardAlgorithm(
                    HandleAccessor::get(handle),
                    input.get(), filter.get(), conv.get(), output.get(),
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0, /* no memory limit */
                    &algo
                )
            );

            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionForwardWorkspaceSize(
                    HandleAccessor::get(handle),
                    input.get(), filter.get(), conv.get(), output.get(),
                    algo, &workspace_size
                )
            );
        }

        ConvolutionAlgorithm& operator=(const ConvolutionAlgorithm&) = default;
        ConvolutionAlgorithm& operator=(ConvolutionAlgorithm&& other) = default;

        auto get() const noexcept { return algo; }
        auto get_workspace_size() const noexcept { return workspace_size; }

    private:
        cudnnConvolutionFwdAlgo_t algo;
        std::size_t workspace_size;
    };

    template <class T> inline
    void getConvolutionForwardOutputDim(
        const ConvolutionDescriptor<T>& convDesc,
        const FilterDescriptor<T>& filterDesc,
        const TensorDescriptor<T>& inputDesc,
        std::vector<int>& output)
    {
        output.clear();
        output.resize(CUDNN_DIM_MAX); /* we use `output` to hold temporaries */

        std::vector<int> temp(CUDNN_DIM_MAX);
        cudnnDataType_t tempDataType;
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetTensorNdDescriptor(
                inputDesc.get(),
                CUDNN_DIM_MAX + 1, /* according to docs, this is what we do to get the rank */
                &tempDataType,
                output.data(),
                temp.data(),
                temp.data()
            )
        );

        const auto rank = output[0];
        output.resize(rank);
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetConvolutionNdForwardOutputDim(
                convDesc.get(), inputDesc.get(), filterDesc.get(), rank, output.data()
            )
        );
    }

    template <class T> inline
    void convolve(
        const Handle& handle,
        const ConvolutionDescriptor<T>& convDesc,
        const ConvolutionAlgorithm<T>& convAlgo,
        const Workspace& workspace,
        const FilterDescriptor<T>& filterDesc,
        DevicePtr<const T> filterPtr,
        const TensorDescriptor<T>& inputDesc,
        DevicePtr<const T> inputPtr,
        T alpha, T beta,
        const TensorDescriptor<T>& outputDesc,
        DevicePtr<T> outputPtr)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnConvolutionForward(
                HandleAccessor::get(handle),
                &alpha, inputDesc.get(), inputPtr.get(),
                filterDesc.get(), filterPtr.get(),
                convDesc.get(), convAlgo.get(),
                WorkspaceAccessor::get(workspace).get(), workspace.size(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_CONVOLUTION_HPP */
